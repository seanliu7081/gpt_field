"""Recorded zero-shot collision outcomes; only synthetic transport/simulator."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from gpt_field.libero import _BowlObstacleCollisionMonitor
from test_bowl_demo_runner import DIRECT_OPTIONS, HYBRID_OPTIONS
from test_demo_audit import mock_override_metadata, mock_override_reset
from test_demo_direct_subscription import prediction
from test_distance_audit import rewrite, rewrite_trace
from test_distance_policy import intent
from test_field_history import FakeTransport, envelope
from test_field_history_runner import FakeRecorder, fake_simulator, observation, rows
from test_hybrid_subscription_runner import runner, sim_config
from test_subscription_scene_runner import scene

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("zero_shot_audit_tests", ROOT / "scripts/audit_zero_shot_direct_control.py")
direct_audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(direct_audit)
hybrid_audit = direct_audit._helpers


def collision_record(actions, detected=False):
    scope = {key: list(getattr(_BowlObstacleCollisionMonitor, key))
             for key in ("monitored_scope", "allowances", "excluded_scope")}
    event = None
    if detected:
        event = {"phase": "action", "action_index": actions - 1, "physics_substep": 0,
                 "contact_distance_m": -.001,
                 "geom1": {"id": 3, "name": "PRIVATE_ROBOT_GEOM", "body": "panda", "object": "robot", "role": "robot"},
                 "geom2": {"id": 7, "name": "PRIVATE_OBSTACLE_GEOM", "body": "milk", "object": "milk_1", "role": "obstacle"},
                 "reason": "robot_scene_contact"}
    return {"enabled": True, "mode": "bowl_obstacles_v1", "detected": detected, "first_event": event,
            "counts": {"checks": 1 + actions * 4 - (3 if detected else 0),
                       "physics_substeps": actions * 3 - (2 if detected else 0),
                       "forbidden_contact_points": int(detected), "allowed_contact_points": actions,
                       "positive_distance_contacts_ignored": 0}, **scope}


def recorded_fixture(tmp_path, observation, *, direct=False, outcome="success", real_video=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    original, custom, state = scene(tmp_path)
    base, instances = fake_simulator(observation, success_at=7)

    class Simulator(base):
        def __init__(self, config):
            super().__init__(config)
            mock_override_metadata(self)
            scope = {key: list(getattr(_BowlObstacleCollisionMonitor, key))
                     for key in ("monitored_scope", "allowances", "excluded_scope")}
            self.metadata.update(collision_check_mode="bowl_obstacles_v1", collision_monitoring={
                "sampling": "every actual physics substep and final control endpoint", **scope})

        def reset(self, task_id, episode_index):
            obs = super().reset(task_id, episode_index)
            mock_override_reset(self, original)
            self.status.update(task_id=0, episode_index=0, libero_success=False, collision=collision_record(0))
            return obs

        def step(self, action):
            obs, _ = super().step(action)
            end = len(self.actions) == 7
            detected = end and outcome in ("collision", "simultaneous")
            raw = end and outcome in ("success", "simultaneous")
            self.status.update(libero_success=raw, success=raw and not detected, terminated=raw or detected,
                               collision=collision_record(len(self.actions), detected))
            return obs, self.status["success"]

    if direct:
        raw = [prediction(phase="approach", target="black_bowl"), prediction(phase="grasp", target="black_bowl")]
    else:
        raw = [intent(), intent(direct=True, phase="grasp"), intent(hold=True)]
    responses = [envelope(value, id=f"zero-audit-{i}", model="gpt-6-astra", reasoning={"effort": "medium"})
                 for i, value in enumerate(raw)]
    if outcome == "interrupted":
        responses[1] = KeyboardInterrupt("synthetic interrupted model request")
    transport = FakeTransport(*responses)
    run = tmp_path / "run"
    result = runner.execute_episode(run, config=runner.configuration(),
        sim_config=sim_config(initial_state_override=str(state), task_bddl_override=str(custom),
                              collision_check_mode="bowl_obstacles_v1"),
        zero_shot_collision_free=True, token_provider=lambda: "synthetic-oauth", transport=transport,
        sim_factory=Simulator, recorder_factory=None if real_video else FakeRecorder,
        **(DIRECT_OPTIONS if direct else HYBRID_OPTIONS))
    expected = "user_stop" if outcome == "interrupted" else "success" if outcome == "success" else "collision_failure"
    assert result["status"] == expected, result.get("error")
    assert len(instances) == 1
    return run, transport


@pytest.fixture
def skip_video(monkeypatch):
    def skip(*args):
        return {"synthetic_mock_recorder": True}, 0
    monkeypatch.setattr(direct_audit._legacy, "_video_audit", skip)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("outcome", ["success", "collision", "simultaneous", "interrupted"])
def test_complete_zero_shot_replay_separates_goal_collision_and_completion_time(
        tmp_path, observation, skip_video, direct, outcome):
    run, transport = recorded_fixture(tmp_path, observation, direct=direct, outcome=outcome)
    audit = direct_audit if direct else hybrid_audit
    report = audit.audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["zero_shot_collision_free"] and not report["demonstration_conditioned"]
    assert report["completion_time_eligible"] is (outcome == "success")
    assert report["task_success"] is (outcome == "success")
    assert report["collision_failure"] is (outcome in ("collision", "simultaneous"))
    assert report["libero_success_observed"] is (outcome in ("success", "simultaneous"))
    assert report["evaluation_completed"] is (outcome != "interrupted")
    assert report["trace_complete"]
    assert report["synthetic_test_fixture"] and not report["real_task_completion_verified"]
    assert report["replayed_recorded_requests"] == len(transport.calls)
    assert report["live_model_calls_by_audit"] == report["network_requests_by_audit"] == report["simulator_actions_by_audit"] == 0
    if direct:
        assert report["reconstructed_local_plans"] == 0
    else:
        assert report["reconstructed_local_plans"] >= 1
    for index, payload in enumerate(transport.calls):
        encoded = json.dumps(payload)
        for private in ("external_demonstration", "PRIVATE_ROBOT_GEOM", "PRIVATE_OBSTACLE_GEOM", "MUST_NOT_ENTER_POLICY"):
            assert private not in encoded
        images = sum(part.get("type") == "input_image" for item in payload["input"]
                     for part in item.get("content", []) if isinstance(part, dict))
        assert images == 2 * min(index + 1, 2)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("tamper", ["counter", "event", "score", "contact", "demo", "action", "scope", "early_collision"])
def test_collision_oracle_and_score_tampering_is_rejected(tmp_path, observation, skip_video, direct, tamper):
    run, _ = recorded_fixture(tmp_path, observation, direct=direct, outcome="simultaneous")
    trace = rows(run / "trace.jsonl")
    if tamper == "counter":
        next(row for row in trace if row["event"] == "executed")["sim_status"]["collision"]["counts"]["physics_substeps"] = 0
    elif tamper == "event":
        next(row for row in trace if row["event"] == "forbidden_collision")["step"] -= 1
    elif tamper == "score":
        value = json.loads((run / "summary.json").read_text())
        value.update(task_success=True, success_rate=1.0, completion_time_eligible=True)
        rewrite(run / "summary.json", value)
    elif tamper in ("contact", "demo"):
        path = sorted((run / "requests").glob("*.request.json"))[0]
        value = json.loads(path.read_text())
        value["payload"]["input"].insert(0, {"role": "user", "content": [{"type": "input_text", "text": json.dumps(
            {"type": "external_demonstration", "trajectory": []} if tamper == "demo" else
            {"collision": collision_record(7, True)})}]})
        rewrite(path, value)
    elif tamper == "action":
        next(row for row in trace if row["event"] == "action")["action"][0] += .01
    elif tamper == "scope":
        next(row for row in trace if row["event"] == "executed")["sim_status"]["collision"]["excluded_scope"].append("all obstacles")
    else:
        next(row for row in trace if row["event"] == "executed")["sim_status"].update(
            collision=collision_record(1, True), success=False, terminated=True)
    rewrite_trace(run, trace)
    report = (direct_audit if direct else hybrid_audit).audit_run(run, tmp_path / "audit")
    assert not report["passed"], tamper


@pytest.mark.parametrize("failed", [False, True])
def test_zero_shot_pair_matches_scene_reset_and_prompts_but_failure_is_not_timing_eligible(
        tmp_path, observation, skip_video, failed):
    hybrid, _ = recorded_fixture(tmp_path / "hybrid", observation)
    direct, _ = recorded_fixture(tmp_path / "direct", observation, direct=True,
                                 outcome="collision" if failed else "success")
    report = direct_audit.audit_run(direct, tmp_path / "audit", reference_run=hybrid)
    assert report["passed"], report["issues"]
    comparison = report["reference_comparison"]
    assert comparison["matched_checks_passed"] and comparison["source_protocols_identical"]
    assert comparison["both_recorded_episodes_completed"]
    assert comparison["both_completion_times_eligible"] is (not failed)


@pytest.mark.parametrize("direct", [False, True])
def test_collision_failure_full_video_and_event_sequence_decodes(tmp_path, observation, direct):
    run, _ = recorded_fixture(tmp_path, observation, direct=direct, outcome="collision", real_video=True)
    report = (direct_audit if direct else hybrid_audit).audit_run(run, tmp_path / "audit")
    assert report["passed"], report["issues"]
    assert report["event_frames"] > 0 and report["collision_failure"]
