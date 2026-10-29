"""uc7.1 approval-to-policy wiring; fake simulators and transport only."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent_speedup_v2_fakes import FakeV2Sim
from gpt_field.agent_speedup_v2_client import FixedAnswers, ScriptedV2Transport
from gpt_field.agent_speedup_v2_user_check import ReviewError, VERSION
from gpt_field.agent_speedup_v2_user_check_agent_client import AgentFirstV2Client
from gpt_field.agent_speedup_v2_user_check_sim import CalibratedObservation
from gpt_field.astranomy_agentic_geometry import extract_geometry_metadata
from test_agent_speedup_v2_user_check_runner import PLAN, TASK, runner, settings
import test_robocasa_user_check_runner as coffee


def geometry():
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    model = SimpleNamespace(camera_name2id=lambda name: 0 if name == "agentview" else 1,
                            cam_fovy=[45., 45.])
    data = SimpleNamespace(cam_xpos=np.array([[1., 0., 2.], [.1, 0., 1.2]]),
                           cam_xmat=np.array([rotation.ravel(), np.eye(3).ravel()]))
    value = extract_geometry_metadata(SimpleNamespace(sim=SimpleNamespace(model=model, data=data)), 512)
    value["cameras"]["wrist"]["pose_source"] = "observed_robot_pose_fixed_mount"
    return value


class CalibratedFakeSim(FakeV2Sim):
    def calibrated(self, obs):
        return CalibratedObservation(obs.image, obs.wrist_image, obs.state, obs.prompt,
            joint_position=(0.,) * 7, joint_velocity=(0.,) * 7, geometry=geometry())

    def reset(self, *args):
        return self.calibrated(super().reset(*args))

    def step(self, action):
        obs, success = super().step(action)
        return self.calibrated(obs), success


def directional_plan():
    plan = copy.deepcopy(PLAN)
    plan["parts"].append({"id": "E", "text": "table: world +x side of cup",
                          "move": "fixed", "w": 32, "x": 32})
    plan["regions"] = [{"id": "E", "support": "observed_plane", "anchor": "A",
                        "relation": "right_of", "frame": "world", "bounds_xy_m": None}]
    plan["placement_goals"] = []
    plan["steps"][0].update(do="place_on", ref="E")
    return plan


def prepare(tmp_path):
    config, sim_config, monitor = settings()
    sim = CalibratedFakeSim(prompt=TASK.prompt)
    transport = ScriptedV2Transport(FixedAnswers(plans=[directional_plan()]), effort="high")
    store = runner.prepare_review(
        tmp_path / "review", config=config, sim_config=sim_config, task_key=TASK.key,
        episode_index=0, monitor_options=monitor, sim_factory=lambda _: sim,
        token_provider=lambda: "offline", transport=transport, video=False)
    return store, config, sim_config, monitor


class ExecutiveTransport:
    offline_mock = True

    def __init__(self):
        self.calls = []

    def __call__(self, endpoint, payload, headers, timeout):
        self.calls.append(copy.deepcopy(payload))
        decision = {"belief": "Inspect the scene.", "plan": "Observe before moving.",
                    "why": "Offline wiring test.", "type": "observe",
                    "motion": None, "macro": None, "execution": None}
        return {"id": f"offline-{len(self.calls)}", "status": "completed",
                "model": "gpt-6-astra", "reasoning": {"effort": "high"},
                "offline_mock": True, "billing": "offline_mock",
                "output": [{"type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": json.dumps(decision)}]}],
                "usage": {"input_tokens": 0, "output_tokens": 0}}


def current_fields(payload):
    return json.loads(next(message for message in reversed(payload["input"])
                           if message.get("role") == "user")["content"][0]["text"])


def test_calibrated_prepare_approval_real_executive_and_terminal_artifacts(tmp_path):
    store, config, sim_config, monitor = prepare(tmp_path)
    review = store.load()
    assert review["version"] == VERSION == "agent-speedup-v2-user-check-7.1"
    frame = review["preparation"]["view_frame"]
    assert frame["available"] and frame["right_xy"] == [0., 1.]
    store.approve(expected_revision=0, reviewed_steps=[0])
    transport, locks = ExecutiveTransport(), []

    def client_factory(*args, **kwargs):
        locks.append(kwargs["lock"])
        kwargs["transport"] = transport
        return AgentFirstV2Client(*args, **kwargs)

    sim = CalibratedFakeSim(prompt=TASK.prompt, success_at=2)
    result = runner.execute_reviewed(
        tmp_path / "trial", review_dir=store.path, config=config, sim_config=sim_config,
        task_key=TASK.key, episode_index=0, monitor_options=monitor, video=False,
        token_provider=lambda: "offline", client_factory=client_factory, sim_factory=lambda _: sim)
    assert result["status"] == "success", result.get("error")
    assert locks[0].view_frame == frame
    assert len(transport.calls) == 2
    for i, payload in enumerate(transport.calls):
        fields = current_fields(payload)
        compact = fields["context"]["region_directions"]
        assert compact["regions"]["E"]["direction_world_xy"] == [0., 1.]
        if i == 0:
            assert fields["reviewed_goal_contract"]["task_contract"]["region_directions"] == compact
        else:
            assert "reviewed_goal_contract" not in fields
    assert result["direction_semantics"] == "external_view_v1"
    manifest = json.loads((tmp_path / "trial/manifest.json").read_text())
    assert manifest["configuration"]["direction_semantics"] == "external_view_v1"
    diagnostic = json.loads((tmp_path / "trial/direction_review.json").read_text())
    assert diagnostic["display_diagnostics"][0]["part_id"] == "E"
    events = [json.loads(line) for line in (tmp_path / "trial/trace.jsonl").read_text().splitlines()]
    assert any(row["event"] == "direction_review" for row in events)


def test_tampered_frozen_basis_breaks_review_integrity(tmp_path):
    store, *_ = prepare(tmp_path)
    store.approve(expected_revision=0, reviewed_steps=[0])
    path = store.path / "review.json"
    data = json.loads(path.read_text())
    data["preparation"]["view_frame"]["right_xy"] = [1., 0.]
    path.write_text(json.dumps(data))
    with pytest.raises(ReviewError):
        store.load()


def test_old_version_cannot_load_under_uc71(tmp_path):
    store, *_ = prepare(tmp_path)
    path = store.path / "review.json"
    data = json.loads(path.read_text())
    data["version"] = "agent-speedup-v2-user-check-7"
    path.write_text(json.dumps(data))
    with pytest.raises(ReviewError, match="version"):
        store.load()


def test_robocasa_frame_survives_actual_prepare_claim_and_factory(tmp_path, monkeypatch):
    source = Path(coffee.runner.ROOT) / coffee.runner.RUNNER
    monkeypatch.setattr(coffee.runner, "runtime_inventory", lambda *_: {
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()})
    config, sim_config = coffee.runner.reviewed.base.configuration(), coffee.runner.RoboCasaConfig()
    args = dict(config=config, sim_config=sim_config, token_provider=lambda: "offline")
    review = coffee.runner.prepare(tmp_path / "review", **args,
        sim_factory=lambda _: coffee.FakeSim(), client_factory=coffee.FakeClient)
    frame = review["preparation"]["view_frame"]
    assert frame["right_xy"] == [1., 0.] and frame["away_xy"] == [0., 1.]
    coffee.runner.ReviewStore(tmp_path / "review").approve(expected_revision=0, reviewed_steps=[0])
    coffee.FakePolicy.instances = []
    result = coffee.runner.execute(tmp_path / "trial", review_dir=tmp_path / "review", **args,
        sim_factory=lambda _: coffee.FakeSim(), client_factory=coffee.FakeClient,
        policy_factory=coffee.FakePolicy, video=False)
    assert result["task_success"]
    assert coffee.FakePolicy.instances[0].lock.view_frame == frame
    assert result["direction_semantics"] == "external_view_v1"
    plan_prompt = coffee.runner.RoboCasaPlanClient(task_prompt=coffee.runner.TASK_PROMPT)._instruction_text["plan"]
    assert "No camera calibration is supplied" in plan_prompt
    assert "calibration supplied with each image" not in plan_prompt
