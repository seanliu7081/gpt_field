"""Offline final-wire, mobile, one-shot launch, source binding and stop checks."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "robocasa365"
sys.path[:0] = [str(PROJECT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "tests")]
from rc365 import foundation_vision as fv
from rc365.policy import RoboCasa365Policy
from gpt_field.agent_speedup_v2_user_check_client import PlanLock
from gpt_field.contact_skills import StepView
from gpt_field.foundation_vision.engine import EngineConfig

_spec = importlib.util.spec_from_file_location("rc_foundation_runner_fixtures", Path(__file__).with_name("test_runner.py"))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)


class Engine:
    def __init__(self):
        self.config = EngineConfig(max_age_s=5.)
        self.segmenter = SimpleNamespace(segment=lambda *_: [])

    def process(self, frame, targets, *, now_s):
        self.segmenter.segment(frame, targets)
        data = {"frame_id": frame.frame_id, "captured_at_s": frame.captured_at_s,
                "generated_at_s": now_s, "age_s": now_s-frame.captured_at_s,
                "max_age_s": 5., "status": "estimated", "geometry": {"views": {}, "surface_points_world": []},
                "objects": [{"key": target.key, "text": target.text, "status": "not_visible",
                    "identity_evidence": "unverified", "centroid_world_m": None,
                    "candidate_counts": {"external": 0, "wrist": 0}, "geometry": {"status": "unknown"}}
                    for target in targets]}
        return SimpleNamespace(to_dict=lambda: deepcopy(data))


class Transport:
    offline_mock = True

    def __init__(self):
        self.calls = []

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls.append(deepcopy(payload))
        answer = {"belief": "Grasp unverified.", "plan": "Reposition and inspect.", "why": "Improve the view.",
                  "type": "mobile", "motion": None, "macro": None, "execution": None, "obstacle_map": None,
                  "mobile": {"base": [.1, 0., 0.], "torso": .04, "steps": 2}}
        return {"id": "offline-foundation-mobile", "status": "completed", "model": payload["model"],
                "reasoning": {"effort": "high"}, "offline_mock": True, "billing": "offline_mock",
                "output": [{"type": "message", "role": "assistant", "status": "completed",
                            "content": [{"type": "output_text", "text": json.dumps(answer)}]}],
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}}


def test_final_wire_preserves_vision_mobile_schema_context_and_exact_payload_hash(tmp_path):
    transport, engine = Transport(), Engine()
    plan = deepcopy(fixtures.PLAN)
    lock = PlanLock(plan, task_prompt=fv.common.TASK_PROMPT)
    client = fv.RoboCasaFoundationVisionClient(task_prompt=fv.common.TASK_PROMPT,
        approved_plan=plan, lock=lock, transport=transport, vision_engine=engine,
        vision_output_dir=tmp_path / "vision")
    policy = fv.StopSafeFoundationVisionPolicy(client, lock=lock, stop_file=tmp_path / "STOP")
    observation = fixtures.observation()
    view = StepView(observation)
    context = policy._context(view)
    result = client.request_agent(view, context)
    assert result["type"] == "mobile" and result["mobile"]["torso"] == .04
    payload = transport.calls[0]
    fields = json.loads(payload["input"][-1]["content"][0]["text"])
    assert fields["context"]["foundation_vision"]["mode"] == "vision_conditioned_executive"
    assert fields["context"]["measured_state"]["mobile"] == observation.mobile_state
    assert "SAM 3.1" in payload["instructions"] and fv.ROBOT_CONTEXT in payload["instructions"]
    assert "gripper0_right_grip_site" in payload["instructions"]
    assert "No object poses, depth, segmentation" not in payload["instructions"]
    assert payload["instructions"] == client._instruction_text["agent"]
    assert payload["reasoning"]["effort"] == "high"
    schema = payload["text"]["format"]["schema"]
    assert "mobile" in schema["required"] and "mobile" in schema["properties"]["type"]["enum"]
    record = json.loads((tmp_path / "vision/payloads.jsonl").read_text())
    assert record["payload_sha256"] == hashlib.sha256(fv._json(payload).encode()).hexdigest()
    assert client.vision_stats()["consumed_requests"] == 1
    assert "EVALUATOR_SECRET_SENTINEL" not in fv._json(payload)


def test_stop_arriving_during_act_discards_pending_arm_or_mobile_decision(tmp_path, monkeypatch):
    policy = fv.StopSafeFoundationVisionPolicy.__new__(fv.StopSafeFoundationVisionPolicy)
    policy.foundation_stop_file = tmp_path / "STOP"
    policy.withheld_decisions_after_stop = 0
    evidence = []
    policy._record_feedback = evidence.append

    def inflight(self, observation):
        self._pending, self._primitive = (observation, "accepted", "view", "primitive"), object()
        self.foundation_stop_file.write_text("stop")
        return SimpleNamespace(action=[0.] * 7, diagnostics={"mobile_command": {"base": [.1, 0., 0.], "torso": .1}})

    monkeypatch.setattr(RoboCasa365Policy, "act", inflight)
    with pytest.raises(KeyboardInterrupt):
        policy.act(object())
    assert policy._pending is policy._primitive is None
    assert policy.withheld_decisions_after_stop == 1
    assert evidence[-1]["executed_steps"] == 0 and evidence[-1]["accepted_decision_dispatched"] is False


def args(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    sim = fixtures.FakeSim()
    (reference / "initial_sensors.json").write_text(json.dumps(fv.common.sensor_record(sim.obs)))
    (reference / "initial_physics.json").write_text(json.dumps(sim.snapshot(include_xml=True)))
    return SimpleNamespace(run_root=tmp_path / "run", sam_checkpoint=tmp_path / "sam.pt",
        da3_model=tmp_path / "da3", vision_device="cuda:0", vision_max_age_s=5.,
        reference_run=reference, seed=7, store_dir=None, profile=None, test_approved=True, no_video=True)


def fake_inventory(monkeypatch):
    path = PROJECT / "rc365/runner.py"
    monkeypatch.setattr(fv.common, "runtime_inventory", lambda config, simulator: {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest(),
        "@configuration": fv.common.canonical_sha256({"api": asdict(config), "simulator": asdict(simulator)})})


def test_fresh_launch_loads_before_auth_resets_then_plan_and_preserves_raw_success(tmp_path, monkeypatch):
    options = args(tmp_path)
    options.run_root.mkdir()
    (options.run_root / "authorization.json").write_text("{}")
    fake_inventory(monkeypatch)
    events, simulations = [], []

    def engine(_):
        events.append("preload")
        return Engine(), {"models": {"sam": {"checkpoint_sha256": "sam"}, "da3": {"checkpoint_sha256": "da3"}}}

    def auth(_):
        events.append("auth")
        return lambda: "offline-fixture-no-network"

    class Sim(fixtures.FakeSim):
        def reset(self):
            events.append("reset")
            return super().reset()

    def simulator(_):
        instance = Sim()
        simulations.append(instance)
        return instance

    class Client(fixtures.FakeClient):
        def request_plan(self, *positional, **kwargs):
            events.append("plan")
            return super().request_plan(*positional, **kwargs)

        def vision_stats(self):
            return {"fixture_only": True}

    def policy(*positional, stop_file, **kwargs):
        return fixtures.FakePolicy(*positional, **kwargs)

    result = fv.start(options, engine_factory=engine, auth_factory=auth, sim_factory=simulator,
                      plan_client_factory=Client, execution_client_factory=Client, policy_factory=policy)
    assert events == ["preload", "auth", "reset", "plan", "reset"]
    assert result["task_success"] and result["robocasa_success"] and result["steps"] == 2
    assert len(simulations) == 2 and all(sim.closed for sim in simulations)
    assert simulations[0].actions == []
    manifest = json.loads((options.run_root / "trial/manifest.json").read_text())
    assert manifest["method"] == fv.METHOD and manifest["human_reviewed_plan"] is False
    assert manifest["mobile_control"] is True and manifest["automatic_episode_caps"] is None
    assert manifest["contacts"] == "record_only" and manifest["evaluator"] == "RestockPantry._check_success"
    assert manifest["foundation_vision"]["models"]["sam"]["checkpoint_sha256"] == "sam"
    assert manifest["foundation_vision"]["extra_queries"] == []
    assert "@foundation_vision" in manifest["runtime_sha256"]
    assert str(PROJECT / "run_foundation_vision.py") in manifest["runtime_sha256"]
    assert json.loads((options.run_root / "initial_match_01.json").read_text())["matched"] is True
    with pytest.raises(FileExistsError):
        fv.start(options, engine_factory=lambda _: pytest.fail("one shot cannot reload models"))


def test_reference_mismatch_blocks_new_plan_and_execution(tmp_path, monkeypatch):
    options = args(tmp_path)
    fake_inventory(monkeypatch)
    expected = json.loads((options.reference_run / "initial_physics.json").read_text())
    expected["physical_sha256"] = "different"
    (options.reference_run / "initial_physics.json").write_text(json.dumps(expected))
    sim = fixtures.FakeSim()
    with pytest.raises(fv.common.ReviewError, match="Fresh reset"):
        fv.start(options, engine_factory=lambda _: (Engine(), {}), auth_factory=lambda _: lambda: "offline",
                 sim_factory=lambda _: sim, plan_client_factory=lambda *a, **k: pytest.fail("mismatched reset sent plan"))
    assert sim.closed and sim.actions == []
    assert not (options.run_root / "trial").exists()
    assert json.loads((options.run_root / "initial_match_01.json").read_text())["matched"] is False


def test_inventory_binds_vision_options_and_restores_original_runner(tmp_path, monkeypatch):
    options = args(tmp_path)
    fake_inventory(monkeypatch)
    original_inventory, original_method = fv.common.runtime_inventory, fv.common.METHOD
    config, simulator = fv.configuration(), fv.common.RoboCasaConfig()
    first = fv.settings(options, {})
    with fv.installed_runner(first, options.reference_run, tmp_path / "trial", {}):
        a = fv.common.runtime_inventory(config, simulator)
        assert fv.common.METHOD == fv.METHOD
    changed = {**first, "max_age_s": 8.}
    with fv.installed_runner(changed, options.reference_run, tmp_path / "trial", {}):
        b = fv.common.runtime_inventory(config, simulator)
    assert a["@foundation_vision"] != b["@foundation_vision"]
    assert fv.common.runtime_inventory is original_inventory and fv.common.METHOD == original_method


def test_preflight_and_missing_approval_do_not_load_or_authenticate(tmp_path, monkeypatch):
    options = args(tmp_path)
    fake_inventory(monkeypatch)
    monkeypatch.setattr(fv, "build_engine", lambda _: pytest.fail("preflight loaded a model"))
    monkeypatch.setattr(fv, "credential_provider", lambda _: pytest.fail("preflight read credentials"))
    result = fv.preflight(options)
    assert result["models_loaded"] is False and result["credential_reads"] == 0
    assert result["simulator_starts"] == 0 and not options.run_root.exists()
    options.test_approved = False
    with pytest.raises(ValueError, match="test-approved"):
        fv.start(options, engine_factory=lambda _: pytest.fail("missing approval loaded models"))
