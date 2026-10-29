"""Reviewed RoboCasa runner boundaries; fake sensors, no simulator/model calls."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.agent_speedup_v2_user_check import ReviewError, ReviewStore, canonical_sha256
from gpt_field.agent_speedup_v2_user_check_sim import CalibratedObservation
from gpt_field.observation import image_to_data_url
from test_robocasa_user_check_geometry import STATE, metadata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "robocasa_user_check_runner_test", ROOT / "scripts/evaluate_robocasa_user_check.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
sys.path.pop(0)

PLAN = {
    "scene": "A coffee machine with a visible button.",
    "parts": [{"id": "A", "text": "coffee machine button", "move": "sliding", "w": 20, "x": 30}],
    "hold_true": [],
    "steps": [{"do": "slide", "obj": "A", "grip": [], "ref": None,
               "sense": "close", "keep": [], "until": []}],
}


def observation():
    pixels = image_to_data_url(np.zeros((512, 512, 3), dtype=np.uint8), 512, rotate=False)
    return CalibratedObservation(pixels, pixels, tuple(STATE), runner.TASK_PROMPT,
        joint_position=(0.,) * 7, joint_velocity=(0.,) * 7, geometry=metadata())


class FakeSim:
    def __init__(self, *, obs=None, success_at=1, start_error=None, step_error=None):
        self.obs = obs or observation()
        self.success_at, self.start_error, self.step_error = success_at, start_error, step_error
        self.status = {"success": False, "robocasa_success": False, "terminated": False,
                       "contacts": {"secret_evaluator_contact": True}}
        self.metadata = {"fixture": True}
        self.actions, self.ops = [], []
        self.closed = False
        self.physical_hash, self.xml_hash = "a" * 64, "b" * 64

    def start(self):
        self.ops.append("start")
        if self.start_error:
            raise self.start_error

    def reset(self):
        self.ops.append("reset")
        return self.obs

    def snapshot(self, include_xml=False):
        self.ops.append("snapshot")
        return {"physical_sha256": self.physical_hash, "model_xml_sha256": self.xml_hash,
                "hidden_task_state": {"coffee_button": False}}

    def step(self, action):
        self.actions.append(list(action))
        if self.step_error:
            raise self.step_error
        success = self.success_at is not None and len(self.actions) >= self.success_at
        self.status.update(success=success, robocasa_success=success, terminated=success)
        return self.obs, success

    def close(self):
        self.ops.append("close")
        self.closed = True


class FakeClient:
    instances = []
    billing = "chatgpt_subscription"
    api_key_calls = 0

    def __init__(self, config, **kwargs):
        self.kwargs, self.requests = kwargs, []
        self.checked = []
        self.instances.append(self)

    def check_observation(self, obs):
        self.checked.append(obs)

    def request_plan(self, external, wrist, **kwargs):
        self.requests.append((external.shape, wrist.shape, kwargs))
        return copy.deepcopy(PLAN)

    def budget_snapshot(self):
        return {"billing": self.billing, "api_key_calls": self.api_key_calls,
                "api_calls": len(self.requests)}

    def history_stats(self):
        return {"accepted_requests": len(self.requests)}


class FakePolicy:
    instances = []
    feedback_error = None
    on_act = None

    def __init__(self, client, *, lock):
        self.client, self.lock = client, lock
        self.seen, self.feedback = [], []
        self.instances.append(self)

    def reset(self):
        pass

    def act(self, obs):
        self.seen.append(obs)
        if self.on_act:
            self.on_act()
        return SimpleNamespace(action=[0., 0., .01, 0., 0., 0., 0.],
                               diagnostics={"model_believes_finished": True})

    def observe_transition(self, before, decision, after):
        if self.feedback_error:
            raise self.feedback_error
        self.feedback.append((before, decision, after))
        return {"observed": True}

    def agent_stats(self):
        return {"observations": len(self.seen)}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # Bind real configuration and this runner, while avoiding a multi-checkout
    # source archive for every fake failure-path example.
    source = ROOT / runner.RUNNER
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    inventory = {str(source): source_digest}
    monkeypatch.setattr(runner, "runtime_inventory", lambda config, sim_config: {
        **inventory, "@configuration": canonical_sha256({"api": asdict(config), "sim": asdict(sim_config)})})
    FakeClient.instances = []
    FakePolicy.instances = []
    config, sim_config = runner.reviewed.base.configuration(), runner.RoboCasaConfig()
    args = {"config": config, "sim_config": sim_config, "token_provider": lambda: "fake-no-network"}
    return SimpleNamespace(path=tmp_path, args=args, inventory=inventory)


def prepare(setup, *, sim=None, client=FakeClient):
    sim = sim or FakeSim()
    data = runner.prepare(setup.path / "review", **setup.args,
                          sim_factory=lambda _: sim, client_factory=client)
    return ReviewStore(setup.path / "review"), sim, data


def approved(setup):
    store, sim, data = prepare(setup)
    store.approve(expected_revision=data["revision"], reviewed_steps=[0])
    return store


def execute(setup, store, *, sim=None, client=FakeClient, policy=FakePolicy):
    sim = sim or FakeSim()
    summary = runner.execute(setup.path / "run", review_dir=store.path, **setup.args,
                             sim_factory=lambda _: sim, client_factory=client,
                             policy_factory=policy, video=False)
    return summary, sim


def test_prepare_is_one_draft_zero_actions_and_unapproved(setup):
    store, sim, data = prepare(setup)
    assert data["status"] == "draft"
    assert data["approved"] is None
    assert data["plan"] == PLAN
    assert data["preparation"]["policy_actions"] == 0
    assert data["preparation"]["automatic_episode_caps"] is None
    assert data["preparation"]["initial_physics_sha256"] == sim.physical_hash
    assert sim.actions == [] and sim.closed
    assert len(FakeClient.instances[0].requests) == 1
    assert FakeClient.instances[0].requests[0][:2] == ((512, 512, 3), (512, 512, 3))
    assert len(data["preparation"]["review_images"]) == 4
    assert (store.path / "initial_physics.json").is_file()


def test_unapproved_review_blocks_before_simulator_or_client(setup):
    store, _, _ = prepare(setup)
    sim = FakeSim()
    with pytest.raises(ReviewError):
        execute(setup, store, sim=sim)
    assert sim.ops == []
    assert len(FakeClient.instances) == 1
    assert not (setup.path / "run").exists()


def test_success_needs_official_evaluator_and_feedback_and_consumes_review(setup):
    store = approved(setup)
    summary, sim = execute(setup, store, sim=FakeSim(success_at=2))
    assert summary["status"] == "success"
    assert summary["task_success"] and summary["completion_time_eligible"]
    assert summary["steps"] == summary["feedback_count"] == 2
    assert summary["pending_action"] is None
    assert sim.closed
    policy = FakePolicy.instances[-1]
    assert len(policy.seen) == len(policy.feedback) == 2
    assert all(not hasattr(obs, "status") for obs in policy.seen)
    assert all("hidden_task_state" not in obs.to_dict() for obs in policy.seen)
    assert not any("evaluator" in key for key in FakeClient.instances[-1].kwargs)
    manifest = json.loads((setup.path / "run/manifest.json").read_text())
    assert manifest["automatic_episode_caps"] is None
    assert manifest["automatic_episode_retries"] == 0
    assert manifest["reasoning_effort"] == "high"
    assert manifest["native_macros"] is False
    with pytest.raises(ReviewError):
        runner.execute(setup.path / "second", review_dir=store.path, **setup.args,
                       sim_factory=lambda _: FakeSim(), client_factory=FakeClient,
                       policy_factory=FakePolicy, video=False)
    assert not (setup.path / "second").exists()


@pytest.mark.parametrize("change", ["rgb", "calibration", "joints", "physics", "xml", "already_successful"])
def test_changed_reset_refuses_before_execution_client(setup, change):
    store = approved(setup)
    sim = FakeSim()
    if change == "rgb":
        sim.obs = replace(sim.obs, image=image_to_data_url(np.ones((512, 512, 3), dtype=np.uint8), 512))
    elif change == "calibration":
        geometry = copy.deepcopy(sim.obs.geometry)
        geometry["cameras"]["external"]["position_world_m"][0] += .01
        sim.obs = replace(sim.obs, geometry=geometry)
    elif change == "joints":
        sim.obs = replace(sim.obs, joint_position=(.01,) * 7)
    elif change == "physics":
        sim.physical_hash = "c" * 64
    elif change == "xml":
        sim.xml_hash = "c" * 64
    else:
        sim.status["success"] = True
    summary, _ = execute(setup, store, sim=sim)
    assert summary["status"] == "infrastructure_error"
    assert summary["error"]["type"] == "ReviewError"
    assert not summary["task_success"]
    assert summary["steps"] == 0
    assert len(FakeClient.instances) == 1
    assert sim.actions == [] and sim.closed


def test_config_change_requires_a_new_review(setup):
    store = approved(setup)
    setup.args["sim_config"] = replace(setup.args["sim_config"], seed=8)
    with pytest.raises(ReviewError, match="Runtime code changed"):
        execute(setup, store)
    assert store.load()["status"] == "approved"


def test_mid_step_interrupt_preserves_unknown_pending_action(setup):
    store = approved(setup)
    summary, sim = execute(setup, store, sim=FakeSim(step_error=KeyboardInterrupt()))
    assert summary["status"] == "user_stop"
    assert summary["steps"] == summary["feedback_count"] == 0
    assert summary["pending_action"] == 0
    assert not summary["task_success"] and not summary["completion_time_eligible"]
    assert sim.closed


def test_feedback_failure_cannot_score_raw_success(setup):
    class BrokenFeedback(FakePolicy):
        feedback_error = RuntimeError("feedback unavailable")
    summary, sim = execute(setup, approved(setup), policy=BrokenFeedback)
    assert summary["status"] == "infrastructure_error"
    assert summary["robocasa_success"]
    assert summary["steps"] == 1 and summary["feedback_count"] == 0
    assert not summary["task_success"] and sim.closed


def test_invalid_subscription_budget_has_terminal_evidence_and_no_action(setup):
    class WrongBilling(FakeClient):
        billing = "api_key"
        api_key_calls = 1
    summary, sim = execute(setup, approved(setup), client=WrongBilling)
    assert summary["status"] == "infrastructure_error"
    assert not summary["task_success"]
    assert summary["execution_budget"]["validation_error"]
    assert summary["api_key_calls"] == 1
    assert sim.actions == [] and sim.closed
    trace = [json.loads(line) for line in (setup.path / "run/trace.jsonl").read_text().splitlines()]
    assert trace[-1]["event"] == "terminal"


def test_simulator_start_failure_closes_partially_started_worker(setup):
    sim = FakeSim(start_error=RuntimeError("worker startup failed"))
    with pytest.raises(RuntimeError, match="worker startup failed"):
        prepare(setup, sim=sim)
    assert sim.closed and sim.ops == ["start", "close"]
    assert FakeClient.instances == []


def test_runtime_change_prevents_completed_result(setup):
    class RuntimeChanges(FakePolicy):
        def observe_transition(self, *args):
            setup.inventory["@changed"] = "e" * 64
            return super().observe_transition(*args)
    summary, sim = execute(setup, approved(setup), policy=RuntimeChanges)
    assert summary["robocasa_success"]
    assert not summary["runtime_unchanged"]
    assert not summary["task_success"] and not summary["completion_time_eligible"]
    assert sim.closed


def test_default_clients_retain_high_effort_and_native_macros_are_unavailable():
    assert runner.UserCheckV2Client.reasoning_effort == "high"
    assert runner.AgentFirstV2Client.reasoning_effort == "high"
    policy = object.__new__(runner.RoboCasaUserCheckPolicy)
    feedback = []
    policy._record_feedback = feedback.append
    policy._apply_agent({"type": "macro"}, None)
    assert policy._need_agent
    assert feedback[0]["status"] == "unavailable"
    assert feedback[0]["executed_steps"] == 0


def test_terminal_image_failure_keeps_worker_closed_and_summary(setup, monkeypatch):
    from PIL import Image
    store = approved(setup)
    original = Image.Image.save

    def fail_terminal(self, fp, *args, **kwargs):
        if isinstance(fp, (str, Path)) and Path(fp).name == "final_external.png":
            raise OSError("image export unavailable")
        return original(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_terminal)
    summary, sim = execute(setup, store)
    assert sim.closed
    assert {"component": "final_sensors", "type": "OSError"} in summary["cleanup_errors"]
    assert json.loads((setup.path / "run/run_state.json").read_text())["stage"] == "terminal"


def test_terminal_inventory_error_preserves_incomplete_summary(setup, monkeypatch):
    store = approved(setup)

    class InventoryDisappears(FakePolicy):
        def observe_transition(self, *args):
            def fail_inventory(*args):
                raise FileNotFoundError("frozen runtime source missing")
            monkeypatch.setattr(runner, "runtime_inventory", fail_inventory)
            return super().observe_transition(*args)

    summary, sim = execute(setup, store, policy=InventoryDisappears)
    assert sim.closed
    assert not summary["runtime_unchanged"]
    assert not summary["task_success"] and not summary["completion_time_eligible"]
    assert json.loads((setup.path / "run/run_state.json").read_text())["stage"] == "terminal"


def test_actual_robocasa_payloads_keep_high_effort_and_robot_configuration():
    from gpt_field.agent_speedup_v2_client import FixedAnswers, ScriptedV2Transport
    transport = ScriptedV2Transport(FixedAnswers(), effort="high")
    plan = runner.RoboCasaPlanClient(task_prompt=runner.TASK_PROMPT, transport=transport)
    lock = runner.PlanLock(PLAN, task_prompt=runner.TASK_PROMPT)
    agent = runner.RoboCasaAgentClient(task_prompt=runner.TASK_PROMPT, transport=transport,
                                      approved_plan=PLAN, lock=lock)
    request = {"fields": {}, "images": []}
    for client, mode in ((plan, "plan"), (agent, "agent")):
        payload = client.build_payload(request, mode=mode)
        assert payload["reasoning"]["effort"] == "high"
        assert "gripper0_right_grip_site" in payload["instructions"]
        assert "No base or torso commands" in payload["instructions"]
        assert "Native macros are unavailable" in payload["instructions"]
    assert transport.calls == []
