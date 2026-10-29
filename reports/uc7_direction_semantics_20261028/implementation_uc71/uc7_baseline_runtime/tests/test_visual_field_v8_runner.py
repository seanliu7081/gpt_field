"""Offline V8 client -> policy -> runner checks with an independent evaluator."""
from dataclasses import replace
import json

import pytest

from gpt_field.action import ROTATION_SCALE_RAD
from gpt_field.visual_field_v8 import MetricFieldExecutorV8
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v5_runner import runner, args as run_args
from test_visual_field_v7_protocol import stream_error, current_record, current_images
from test_visual_field_v8_protocol import prediction


def fixtures(clock, responses, *, outcome="success", terminal_step=8, request_action_counts=None, joint_limit=False):
    sims = []

    class Sim:
        def __init__(self, config):
            assert config.max_steps is None and config.include_arm_joints
            assert config.collision_check_mode == "task_manipulation_v1"
            self.metadata = {}; self.actions = []; self.closed = False
            sims.append(self)

        def start(self):
            pass

        def reset(self, *args):
            self.status = {"success": False, "libero_success": False,
                "collision": {"mode":"task_manipulation_v1", "detected":False,
                              "private":"V8_EVALUATOR_PRIVATE_INITIAL"}}
            return self.obs()

        def obs(self):
            obs = observation()
            gap = .04-.002*len(self.actions)
            if joint_limit:
                obs = replace(obs, joint_position=(10., *obs.joint_position[1:]))
            return replace(obs, state=(*obs.state[:6], gap, -gap))

        def step(self, action):
            if outcome == "interrupt" and len(self.actions) == terminal_step:
                raise KeyboardInterrupt
            clock[0] += .25
            self.actions.append(tuple(action))
            if len(self.actions) == terminal_step and outcome != "interrupt":
                raw = outcome in ("success", "simultaneous")
                collision = outcome in ("collision", "simultaneous")
                # Deliberately leave success=True on simultaneous signals:
                # the runner must prioritize collision independently.
                self.status.update(success=raw, libero_success=raw,
                    collision={"mode":"task_manipulation_v1", "detected":collision,
                               "private":"V8_EVALUATOR_PRIVATE_TERMINAL"})
            return self.obs(), self.status["success"]

        def close(self):
            self.closed = True

    class Recorder:
        def __init__(self, *args): pass
        def frame(self, *args): clock[0] += .5
        def close(self): pass

    class Transport(FakeTransport):
        def __call__(self, endpoint, payload, headers, timeout_s):
            if request_action_counts is not None:
                assert len(sims[0].actions) == request_action_counts[len(self.calls)]
            assert headers["Authorization"] == "Bearer offline-oauth"
            clock[0] += 7.
            return super().__call__(endpoint, payload, headers, timeout_s)

    transport = Transport(*(value if isinstance(value, BaseException) else envelope(
        value, model="gpt-6-astra", reasoning={"effort":"medium"}) for value in responses))
    return Sim, Recorder, transport, sims


def gripper_request():
    value = prediction(controller="gripper", phase="grasp")
    value["visual_references"] = {"external":None, "wrist":None}
    value["motion_context"]["purpose"] = "contact_correction"
    value["reason"] = "Model considers the task complete; this text is not an evaluator signal."
    return value


@pytest.mark.parametrize("outcome", ["success", "collision", "simultaneous", "interrupt"])
def test_qualified_evaluator_stop_and_video_exclusion_with_accounted_subscription_retry(tmp_path, monkeypatch, outcome):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda:clock[0])
    Sim, Recorder, transport, sims = fixtures(clock, [stream_error(), gripper_request()],
        outcome=outcome, terminal_step=8, request_action_counts=[0, 0])
    config = run_args(tmp_path, method_version="v8")
    result = runner.execute(config, token_provider=lambda:"offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    assert result["method_version"] == "v8" and result["steps"] == 8
    assert result["gripper_actions"] == 8 and result["field_actions"] == result["direct_actions"] == 0
    held_grip = (0., 0., 0., MetricFieldExecutorV8.OSC_GOAL_REFRESH_RAD/ROTATION_SCALE_RAD, 0., 0., 1.)
    assert sims[0].actions == [held_grip]*8
    assert result["task_success"] is (outcome == "success")
    assert result["completion_time_eligible"] is (outcome == "success")
    assert result["libero_success"] is (outcome in ("success", "simultaneous"))
    assert result["status"] == {"success":"success", "collision":"collision_failure",
        "simultaneous":"collision_failure", "interrupt":"user_stop"}[outcome]
    assert result["budget"]["api_calls"] == result["policy_diagnostic_counters"]["model_requests"] == 2
    assert result["policy_diagnostic_counters"]["intent_refreshes"] == 1
    assert result["budget"]["api_key_calls"] == 0 and result["budget"]["unknown_usage_calls"] == 1
    assert current_images(transport.calls[0]) == current_images(transport.calls[1])
    records = [current_record(call) for call in transport.calls]
    assert records[0]["observation_index"] == records[1]["observation_index"] == 0
    assert records[0]["observation/state"] == records[1]["observation/state"]
    assert "V8_EVALUATOR_PRIVATE" not in json.dumps(transport.calls)
    assert all(call["model"] == "gpt-6-astra" and call["reasoning"] == {"effort":"medium"}
               for call in transport.calls)
    timings = result["timing_components"]
    assert timings["model_pipeline_s"] == timings["decision_s"] == 14.
    assert timings["failed_decision_s"] == 0.
    assert timings["simulator_step_s"] == 2. and timings["video_render_s"] == 4.
    assert result["control_without_video_s"] == 16. and result["control_elapsed_s"] == 20.
    manifest = json.loads((config.output/"manifest.json").read_text())
    assert manifest["method_version"] == "v8" and manifest["shots"] == 0
    assert manifest["max_policy_steps"] is None
    assert manifest["max_api_calls"] is None and manifest["max_wall_time_s"] is None
    assert "accepted Direct commands execute unchanged" in manifest["controller_handoff"]
    assert "rejected rotation blocks are discarded" in manifest["controller_handoff"]
    assert "current-pose" in manifest["rejection_hold"]
    assert sims[0].closed and result["cleanup_errors"] == []
    assert len(list((config.output/"api").glob("*.v7_retry.json"))) == 1


@pytest.mark.parametrize("cause", ["reference_only", "projected_neighbor", "unknown_geometry"])
def test_rotation_veto_executes_none_of_requested_block_then_preserves_nonrotating_direct_five(tmp_path, monkeypatch, cause):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda:clock[0])
    rotate = prediction(phase="recover", present=cause == "projected_neighbor")
    rotate["visual_references"] = {"external":None, "wrist":None}
    rotate["direct_actions"] = [[.02, 0., 0., 0., 0., .15, -1.]]*5
    context = rotate["motion_context"]
    context["purpose"] = "reference_repair" if cause == "reference_only" else "task_reorientation"
    if cause == "projected_neighbor":
        context["tool_external_roi_uv"] = [.35, .35, .5, .5]
        context["nearby_obstacles"][0]["external_roi_uv"] = [.48, .4, .62, .55]
    if cause == "unknown_geometry": context["obstacle_assessment"] = "unobservable"
    correction = prediction(phase="grasp")
    correction["visual_references"] = {"external":None, "wrist":None}
    correction["motion_context"].update(purpose="contact_correction", tool_external_roi_uv=None,
                                         obstacle_assessment="unobservable")
    commands = [[.01*(i+1), -.02, .0, .0, .0, .0, -1.] for i in range(5)]
    correction["direct_actions"] = commands
    Sim, Recorder, transport, sims = fixtures(clock, [rotate, correction], terminal_step=6,
                                                request_action_counts=[0, 1])
    config = run_args(tmp_path, method_version="v8")
    result = runner.execute(config, token_provider=lambda:"offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    hold = (0., 0., 0., MetricFieldExecutorV8.OSC_GOAL_REFRESH_RAD/ROTATION_SCALE_RAD, 0., 0., 0.)
    assert sims[0].actions == [hold]+[tuple(command) for command in commands]
    assert result["steps"] == 6 and result["direct_actions"] == 6
    assert result["policy_diagnostic_counters"]["direct_rotation_vetoes"] == 1
    assert result["policy_diagnostic_counters"]["model_requests"] == 2
    rows = [json.loads(line) for line in (config.output/"trace.jsonl").read_text().splitlines()]
    actions = [row for row in rows if row["event"] == "action"]
    rejected = actions[0]["diagnostics"]
    assert rejected["selection"]["accepted"] is False
    assert rejected["rejected_model_action"] == rotate["direct_actions"][0]
    assert rejected["prediction"]["direct_actions"] == rotate["direct_actions"]
    assert rejected["projected_rotation_risk"]["clearance_certified"] is False
    assert rejected["projected_rotation_risk"]["reason"] == {
        "reference_only":"rotation_for_reference_repair_not_allowed",
        "projected_neighbor":"rotation_projected_obstacle_risk",
        "unknown_geometry":"rotation_neighborhood_unobservable"}[cause]
    context_after_veto = current_record(transport.calls[1])["observable_context"]
    assert context_after_veto["visual_feedback"]["projected_rotation_risk"]["accepted"] is False
    assert context_after_veto["refresh_reason"].startswith("rotation_")
    assert actions[0]["action"] == list(hold)
    assert all(row["action"][3:6] == [0., 0., 0.] for row in actions[1:])
    assert result["task_success"] and result["libero_success"]
    assert result["control_without_video_s"] == 15.5
    assert result["budget"]["api_key_calls"] == 0
    assert "V8_EVALUATOR_PRIVATE" not in json.dumps(transport.calls)


@pytest.mark.parametrize("failures", [[stream_error(), stream_error()], [stream_error(code="invalid_api_key", status=401)]])
def test_terminal_transport_error_never_reaches_robot_and_retains_full_attempt_time(tmp_path, monkeypatch, failures):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda:clock[0])
    Sim, Recorder, transport, sims = fixtures(clock, failures, request_action_counts=[0]*len(failures))
    result = runner.execute(run_args(tmp_path, method_version="v8"), token_provider=lambda:"offline-oauth",
                            transport=transport, sim_factory=Sim, recorder_factory=Recorder)
    assert result["status"] == "subscription_error" and result["steps"] == 0
    assert not sims[0].actions and sims[0].closed
    assert result["budget"]["api_calls"] == len(failures)
    assert result["budget"]["api_key_calls"] == 0
    assert result["timing_components"]["failed_decision_s"] == 7.*len(failures)
    assert result["control_without_video_s"] == 7.*len(failures)
    assert not result["completion_time_eligible"]


def test_rejected_gripper_current_pose_refresh_is_acknowledged_without_executing_grip(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda:clock[0])
    next_request = prediction()
    next_request["visual_references"] = {"external":None, "wrist":None}
    Sim, Recorder, transport, sims = fixtures(clock, [gripper_request(), next_request],
        outcome="interrupt", terminal_step=1, request_action_counts=[0, 1], joint_limit=True)
    config = run_args(tmp_path, method_version="v8")
    result = runner.execute(config, token_provider=lambda:"offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    hold = (0., 0., 0., MetricFieldExecutorV8.OSC_GOAL_REFRESH_RAD/ROTATION_SCALE_RAD, 0., 0., 0.)
    assert result["status"] == "user_stop"  # No rejected-command acknowledgement exception.
    assert result["steps"] == 1 and sims[0].actions == [hold]
    assert result["policy_diagnostic_counters"]["gripper_settle_actions"] == 0
    rows = [json.loads(line) for line in (config.output/"trace.jsonl").read_text().splitlines()]
    feedback = next(row for row in rows if row["event"] == "feedback")["feedback"]
    assert feedback["gripper_execution"]["executed_actions"] == 0
    first = next(row for row in rows if row["event"] == "action")["diagnostics"]
    assert first["selection"]["accepted"] is False and first["selection"]["status"] == "measured_joint_limit"
    assert first["rejection_hold"]["requested_motion_executed"] is False
    assert result["cleanup_errors"] == [] and sims[0].closed


@pytest.mark.parametrize("changes", [{"controller":"direct"}, {"demo":"not-a-demo.json"}])
def test_v8_remains_zero_shot_hybrid_only_without_launching_other_arms(tmp_path, changes):
    with pytest.raises(ValueError, match="zero-shot Hybrid"):
        runner.execute(run_args(tmp_path, method_version="v8", **changes),
                       token_provider=lambda:pytest.fail("Must fail before authorization"))
    assert not (tmp_path/"trial").exists()
