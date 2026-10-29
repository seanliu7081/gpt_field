"""V14 fake-client -> real-policy -> runner boundaries and durable timing prefix."""
import copy
import json
import hashlib

import pytest

from test_visual_field_v7_protocol import stream_error, current_record, current_images
from test_visual_field_v8_runner import fixtures, gripper_request as request_v8, runner, run_args
from test_visual_field_v14_protocol import prediction as prediction_base


def prediction(**kwargs):
    value = prediction_base(**kwargs)
    value["target_geometry"].update(center_offset_m=[.3, .3, .3], half_extents_m=[.01, .01, .01])
    return value


def gripper_request():
    value = request_v8()
    value["subgoal_update"] = {"action": "none", "subgoal_id": None, "reason": "No active navigation subgoal."}
    value["motion_context"].update(translation_mode="transit", intended_contact_id=None)
    value.update(target_geometry={"id":value["target_id"], "center_offset_m":[.3,.3,.3],
        "half_extents_m":[.01,.01,.01], "uncertainty_m":.005, "allowed_contact":"none"}, contact_alignment=None)
    return value


def checkpoints(monkeypatch):
    original = runner.write_json
    captured = []

    def write(path, value):
        if path.name != "run_state.json.tmp":
            assert path.name != "run_state.json", "V14 must publish checkpoints by atomic replacement"
            return original(path, value)
        previous_path = path.with_name("run_state.json")
        previous = previous_path.read_bytes() if previous_path.exists() else None
        if previous is not None:
            json.loads(previous)  # The old checkpoint is always a complete JSON record.
        original(path, value)
        assert json.loads(path.read_text()) == value
        if previous is not None:
            assert previous_path.read_bytes() == previous
        captured.append(copy.deepcopy(value))

    monkeypatch.setattr(runner, "write_json", write)
    return captured


@pytest.mark.parametrize("outcome", ["success", "collision", "simultaneous", "interrupt"])
def test_v14_evaluator_priority_same_observation_retry_and_finalized_video_excluded_time(tmp_path, monkeypatch, outcome):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    snapshots = checkpoints(monkeypatch)
    Sim, Recorder, transport, sims = fixtures(clock, [stream_error(), gripper_request()],
        outcome=outcome, terminal_step=8, request_action_counts=[0, 0])
    config = run_args(tmp_path, method_version="v14")
    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    assert result["method_version"] == "v14" and result["steps"] == 8
    assert result["gripper_actions"] == 8 and result["field_actions"] == result["direct_actions"] == 0
    assert result["status"] == {"success": "success", "collision": "collision_failure",
        "simultaneous": "collision_failure", "interrupt": "user_stop"}[outcome]
    assert result["task_success"] is result["completion_time_eligible"] is (outcome == "success")
    assert result["libero_success"] is (outcome in ("success", "simultaneous"))
    assert len(transport.calls) == result["budget"]["api_calls"] == result["policy_diagnostic_counters"]["model_requests"] == 2
    assert result["budget"]["unknown_usage_calls"] == 1 and result["budget"]["api_key_calls"] == 0
    records = [current_record(call) for call in transport.calls]
    assert records[0]["observation_index"] == records[1]["observation_index"] == 0
    assert records[0]["observation/state"] == records[1]["observation/state"]
    assert current_images(transport.calls[0]) == current_images(transport.calls[1])
    assert "V8_EVALUATOR_PRIVATE" not in json.dumps(transport.calls)
    assert all(p["model"] == "gpt-6-astra" and p["reasoning"] == {"effort": "medium"} for p in transport.calls)
    final = json.loads((config.output / "run_state.json").read_text())
    assert final["finalized"] is True and final["stage"] == "finished"
    assert final["timing_components"] == result["timing_components"]
    assert final["elapsed_control_s"] == result["control_elapsed_s"] == 20.
    assert final["control_without_video_s"] == result["control_without_video_s"] == 16.
    assert final["timing_components"]["video_render_s"] == 4.
    assert final["timing_components"]["model_pipeline_s"] == 14.
    observed = [item for item in snapshots if item["stage"] == "observed"]
    assert len(observed) == 8 and all(not item["finalized"] for item in observed)
    assert observed[-1]["control_without_video_s"] == 16.
    assert all("observed prefix" in item["timing_scope"] for item in snapshots)
    assert not (config.output / "run_state.json.tmp").exists()
    manifest = json.loads((config.output / "manifest.json").read_text())
    assert manifest["method_version"] == "v14" and manifest["shots"] == 0
    assert manifest["max_policy_steps"] is manifest["max_api_calls"] is manifest["max_wall_time_s"] is None
    assert "persistent world goal" in manifest["subgoal_reuse"]
    assert "atomic run_state" in manifest["timing_checkpoint"]
    assert "no 3D clearance certificate" in manifest["projected_translation_screen"]
    assert not manifest["robot_tool_geometry"]["certifies_environment_clearance"]
    assert not manifest["robot_tool_geometry"]["environment_inputs"]
    assert manifest["feedback_representation"]["version"] == "observable-feedback-actionable-v14-v1"
    for name, digest in manifest["v14_runtime_source_sha256"].items():
        assert hashlib.sha256((config.output / "frozen_source" / name).read_bytes()).hexdigest() == digest
        assert digest == manifest["software"]["source_sha256"][name]
    assert {"robot_tool_geometry.py", "visual_translation_guard_v14.py", "visual_feedback_compact_v14.py", "contact_geometry_v13.py", "visual_contact_alignment_v13.py"} <= set(manifest["v14_runtime_source_sha256"])
    assert sims[0].closed and not result["cleanup_errors"]


@pytest.mark.parametrize("outcome", ["interrupt", "success", "collision", "simultaneous"])
def test_operator_stop_waits_for_complete_feedback_and_yields_to_evaluator_outcome(tmp_path, monkeypatch, outcome):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    Sim, Recorder, transport, sims = fixtures(clock, [gripper_request()], outcome=outcome, terminal_step=1)
    config = run_args(tmp_path, method_version="v14")

    class StopAfterStep(Sim):
        def step(self, action):
            value = super().step(action)
            (config.output / "operator_stop.request.json").write_text('{"reason":"operator-reviewed boundary stop"}')
            return value

    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=StopAfterStep, recorder_factory=Recorder)
    rows = [json.loads(line) for line in (config.output / "trace.jsonl").read_text().splitlines()]
    for event in ("action", "executed", "feedback"):
        assert sum(row["event"] == event for row in rows) == 1
    assert len(transport.calls) == result["steps"] == 1
    if outcome == "interrupt":
        assert result["status"] == "user_stop" and result["error"]["type"] == "OperatorStopFile"
        assert rows[-1]["event"] == "operator_stop"
        assert rows[-1]["completed_actions"] == 1
    else:
        assert result["status"] == ("success" if outcome == "success" else "collision_failure")
        assert not any(row["event"] == "operator_stop" for row in rows)
    final = json.loads((config.output / "run_state.json").read_text())
    assert final["finalized"] is True and final["step"] == 1
    assert final["control_without_video_s"] == result["control_without_video_s"] == 7.25
    assert final["timing_components"]["video_render_s"] == .5
    assert sims[0].closed and not result["cleanup_errors"]


def test_pending_request_preserves_exact_completed_prefix_until_final_failed_request_accounting(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    snapshots = checkpoints(monkeypatch)
    first = prediction()
    first["visual_references"] = {"external": None, "wrist": None}
    first["direct_actions"] = [[0., 0., 0., 0., 0., 0., -1.]] * 5
    failure = stream_error(code="invalid_api_key", status=401)
    Sim, Recorder, transport, sims = fixtures(clock, [first, failure], outcome="interrupt",
        terminal_step=99, request_action_counts=[0, 5])
    config = run_args(tmp_path, method_version="v14")
    pending = []

    class InspectPending:
        def __call__(self, *args):
            if len(transport.calls) != 1:
                return transport(*args)
            path = config.output / "run_state.json"
            before = path.read_bytes()
            try:
                return transport(*args)
            finally:
                assert path.read_bytes() == before
                pending.append((json.loads(before), clock[0]))

    class ClosingRecorder(Recorder):
        def close(self):
            clock[0] += 11.  # Final encoding belongs to cleanup, outside control time.

    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=InspectPending(),
                            sim_factory=Sim, recorder_factory=ClosingRecorder)
    assert len(pending) == 1
    prefix, now = pending[0]
    assert prefix["stage"] == "deciding" and prefix["step"] == 5 and not prefix["finalized"]
    assert prefix["elapsed_control_s"] == 10.75
    assert prefix["timing_components"]["video_render_s"] == 2.5
    assert prefix["control_without_video_s"] == 8.25
    assert prefix["timing_components"]["model_pipeline_s"] == 7.
    assert prefix["budget"]["api_calls"] == 1  # Second request began after the checkpoint.
    assert now == 118.25  # 100 + initial video .5 + prefix10.75 + pending model7.
    observed = [item for item in snapshots if item["stage"] == "observed"][-1]
    assert observed["step"] == 5 and observed["control_without_video_s"] == 8.25
    assert result["status"] == "subscription_error" and result["steps"] == 5
    assert len(transport.calls) == result["budget"]["api_calls"] == 2
    assert result["budget"]["api_key_calls"] == 0
    assert result["control_elapsed_s"] == 17.75 and result["control_without_video_s"] == 15.25
    assert result["wall_elapsed_s"] == 29.25
    assert result["timing_components"]["decision_s"] == 14.
    assert result["timing_components"]["failed_decision_s"] == 7.
    final = json.loads((config.output / "run_state.json").read_text())
    assert final["finalized"] is True and final["control_without_video_s"] == 15.25
    assert final["timing_components"] == result["timing_components"]
    assert final["budget"]["api_calls"] == 2
    assert not result["completion_time_eligible"] and not (config.output / "run_state.json.tmp").exists()
    assert "V8_EVALUATOR_PRIVATE" not in json.dumps(transport.calls)
    assert sims[0].closed and not result["cleanup_errors"]


@pytest.mark.parametrize("change", [{"controller": "direct"}, {"demo": "unused.json"}])
def test_v14_refuses_unrequested_baseline_or_demonstration(tmp_path, change):
    with pytest.raises(ValueError, match="zero-shot Hybrid"):
        runner.execute(run_args(tmp_path, method_version="v14", **change),
                       token_provider=lambda: pytest.fail("Must not authorize"))


def test_translation_veto_discards_block_then_executes_allowed_direct_five_unchanged(tmp_path, monkeypatch):
    from gpt_field.action import ROTATION_SCALE_RAD
    from gpt_field.visual_field_v8 import MetricFieldExecutorV8

    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    rejected = prediction()
    rejected["visual_references"] = {"external": None, "wrist": None}
    rejected["motion_context"]["tool_external_roi_uv"] = None
    rejected["direct_actions"] = [[.1, 0., 0., 0., 0., 0., -1.]] * 5
    correction = prediction(phase="grasp", mode="interaction")
    correction["visual_references"] = {"external": None, "wrist": None}
    commands = [[0., 0., 0., 0., 0., 0., float(i % 2)] for i in range(5)]
    correction["direct_actions"] = commands
    Sim, Recorder, transport, sims = fixtures(clock, [rejected, correction],
        terminal_step=6, request_action_counts=[0, 1])
    config = run_args(tmp_path, method_version="v14")
    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=Sim, recorder_factory=Recorder)
    hold = (0., 0., 0., MetricFieldExecutorV8.OSC_GOAL_REFRESH_RAD/ROTATION_SCALE_RAD, 0., 0., 0.)
    assert sims[0].actions == [hold] + [tuple(command) for command in commands]
    assert result["status"] == "success" and result["steps"] == result["direct_actions"] == 6
    assert result["budget"]["api_calls"] == 2 and result["policy_diagnostic_counters"]["projected_translation_vetoes"] == 1
    rows = [json.loads(line) for line in (config.output / "trace.jsonl").read_text().splitlines()]
    actions = [row for row in rows if row["event"] == "action"]
    first = actions[0]["diagnostics"]
    assert first["selection"]["accepted"] is False
    assert first["rejected_model_action"] == rejected["direct_actions"][0]
    assert first["prediction"]["direct_actions"] == rejected["direct_actions"]
    assert not first["rejection_hold"]["requested_motion_executed"]
    assert not first["translation_admission"]["accepted"]
    record = current_record(transport.calls[1])["observable_context"]
    assert record["visual_feedback"]["translation_guard"]["clearance_certified"] is False
    assert "translation" in record["refresh_reason"]
    assert "V8_EVALUATOR_PRIVATE" not in json.dumps(transport.calls)
    assert [row["action"] for row in actions[1:]] == commands


def test_legacy_default_components_remain_v1(tmp_path):
    config = run_args(tmp_path)
    del config.method_version
    client, policy, _, _ = runner.method_components(config)
    assert client is runner.VisualFieldSubscriptionClient
    assert policy is runner.VisualFieldPolicy


@pytest.mark.parametrize("near_neighbor", [False, True])
def test_current_observed_direct_contact_translation_can_execute_five_without_stale_field_geometry(tmp_path, monkeypatch, near_neighbor):
    from dataclasses import replace
    import numpy as np
    from gpt_field.distance_tasks import resolve_distance_task
    from gpt_field.observation import decode_image, image_to_data_url
    from test_visual_field_v6_policy import observation as observed, prediction as tracked_prediction

    clock = [100.]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])
    commands = [[.04, 0., 0., 0., 0., 0., 0.]] * 5
    value = tracked_prediction("direct", phase="grasp", actions=commands)
    value.update(target_geometry={"id": value["target_id"], "center_offset_m": [.3,.3,.3],
        "half_extents_m": [.01,.01,.01], "uncertainty_m": .005, "allowed_contact": "none"}, contact_alignment=None)
    value["subgoal_update"] = {"action": "none", "subgoal_id": None, "reason": "Current visible contact pose needs a bounded correction."}
    value["motion_context"] = {"purpose": "contact_correction", "tool_external_roi_uv": [.21,.27,.33,.40],
        "nearby_obstacles": [], "obstacle_assessment": "none_relevant",
        "translation_mode": "interaction", "intended_contact_id": value["target_id"]}
    if near_neighbor:
        value["motion_context"].update(tool_external_roi_uv=[.21,.27,.43,.40],
            obstacle_assessment="present", nearby_obstacles=[{"id":"fixture", "external_roi_uv":[.37,.30,.45,.39]}])
    Sim, Recorder, transport, sims = fixtures(clock, [value], terminal_step=5, request_action_counts=[0])

    class TrackedSim(Sim):
        def obs(self):
            offset = np.zeros(3)
            for action in self.actions:
                offset += np.asarray(action[:3])*.05*.25
            observation = replace(observed(offset), prompt=resolve_distance_task("bowl_plate").prompt)
            if near_neighbor:
                pixels = np.asarray(decode_image(observation.image)).copy()
                pixels[68:83,83:98] = np.random.default_rng(808).integers(0,256,(15,15,3),dtype=np.uint8)
                observation = replace(observation, image=image_to_data_url(pixels,rotate=False))
            return observation

        def step(self, action):
            result = super().step(action)
            if near_neighbor:
                (config.output / "operator_stop.request.json").write_text('{"reason":"offline test ends after observed refusal"}')
            return result

    config = run_args(tmp_path, method_version="v14")
    result = runner.execute(config, token_provider=lambda: "offline-oauth", transport=transport,
                            sim_factory=TrackedSim, recorder_factory=Recorder)
    if near_neighbor:
        assert result["status"] == "user_stop" and result["steps"] == 1 and not result["task_success"]
        assert sims[0].actions[0][:3] == (0.,0.,0.)
        rows = [json.loads(line) for line in (config.output / "trace.jsonl").read_text().splitlines()]
        action = next(row for row in rows if row["event"] == "action")
        assert action["diagnostics"]["translation_admission"]["accepted"] is False
        assert action["diagnostics"]["rejected_model_action"] == commands[0]
        assert result["policy_diagnostic_counters"]["projected_translation_vetoes"] == 1
        assert len(transport.calls) == 1
        return
    assert result["task_success"] and result["steps"] == result["direct_actions"] == 5
    assert result["field_actions"] == result["gripper_actions"] == 0
    assert sims[0].actions == [tuple(command) for command in commands]
    assert len(transport.calls) == result["budget"]["api_calls"] == 1
    assert result["policy_diagnostic_counters"]["projected_translation_vetoes"] == 0
    assert result["control_without_video_s"] == 8.25
    assert "V8_EVALUATOR_PRIVATE" not in json.dumps(transport.calls)


@pytest.mark.parametrize('controller', ['direct', 'gripper'])
def test_empty_neighbor_lists_and_finger_permission_cannot_execute_palm_target_closure(tmp_path, monkeypatch, controller):
    import numpy as np
    from gpt_field.robot_tool_geometry import world_aabbs
    from test_visual_field_protocol import observation
    from gpt_field.visual_field_v8 import measured_current_pose_hold_action

    clock = [100.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    value = prediction(controller=controller, phase='grasp', mode='interaction')
    value['visual_references'] = {'external': None, 'wrist': None}
    hand = next(part for part in world_aabbs(observation().state) if part['name'] == 'hand_collision')
    center = (np.array(hand['offset_min_xyz']) + hand['offset_max_xyz']) / 2
    value['target_geometry'].update(center_offset_m=center.tolist(), half_extents_m=[.002]*3, allowed_contact='fingers')
    if controller == 'direct': value['direct_actions'] = [[0.,0.,0.,0.,0.,0.,1.]]*5
    else: value['gripper_intent']['command'] = 'close'
    assert not value['motion_context']['nearby_obstacles']
    Sim, Recorder, transport, sims = fixtures(clock, [value], terminal_step=99)
    config = run_args(tmp_path, method_version='v14')

    class StopAfterHeldRefusal(Sim):
        def step(self, action):
            result = super().step(action)
            (config.output/'operator_stop.request.json').write_text('{"reason":"offline boundary after target refusal"}')
            return result

    result = runner.execute(config, token_provider=lambda:'offline-oauth', transport=transport,
        sim_factory=StopAfterHeldRefusal, recorder_factory=Recorder)
    assert result['status'] == 'user_stop' and result['steps'] == 1 and not result['completion_time_eligible']
    assert sims[0].actions == [tuple(measured_current_pose_hold_action())]
    rows = [json.loads(line) for line in (config.output/'trace.jsonl').read_text().splitlines()]
    diag = next(row['diagnostics'] for row in rows if row['event'] == 'action')
    assert diag['selection']['status'] == 'target_forbidden_part_intersection'
    assert not diag['target_contact_admission']['accepted']
    assert any(part['tool_part'] == 'hand_collision' for part in diag['target_contact_admission']['diagnostics']['intersections'])
    assert diag['rejected_model_action'][6] == 1. and not diag['rejection_hold']['requested_motion_executed']
    assert len(transport.calls) == 1 and result['budget']['api_key_calls'] == 0


def test_direct_positive_grip_cannot_clear_cached_two_view_misalignment_by_omitting_descriptor(tmp_path, monkeypatch):
    from dataclasses import replace
    from gpt_field.distance_tasks import resolve_distance_task
    from gpt_field.observation import image_to_data_url
    from gpt_field.visual_field_v8 import measured_current_pose_hold_action
    from test_visual_contact_alignment_v13 import fixture, render, HAND, EXT_TARGET, WRIST_TARGET, TOOL_ROI, box
    from test_visual_field_v6_policy import observation as observed

    clock = [100.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    _, _, _, descriptor = fixture()
    first = prediction(controller='visual_field', mode='interaction')
    first['target_id'] = first['target_geometry']['id'] = 'part'
    first['target_geometry']['allowed_contact'] = 'fingers'
    first['motion_context'].update(intended_contact_id='part', tool_external_roi_uv=TOOL_ROI.tolist())
    first['field_intent']['goal_offset_m'] = [0.,0.,0.]  # Local arrival requests a new decision, not task success.
    first['contact_alignment'] = descriptor
    first['visual_references'] = {
        'external': {'update':'seed', 'moving_id':'tool', 'moving_kind':'tool', 'target_id':'part',
            'goal_offset_uv':((HAND-EXT_TARGET)/223).tolist(), 'seed':{
                'moving_point_uv':(HAND/223).tolist(), 'moving_roi_uv':box(HAND).tolist(),
                'target_anchor_uv':(EXT_TARGET/223).tolist(), 'target_roi_uv':box(EXT_TARGET).tolist(),
                'tool_point_uv':None, 'tool_roi_uv':None}},
        'wrist': {'update':'seed', 'target_id':'part', 'desired_target_uv':(WRIST_TARGET/223).tolist(),
            'seed':{'target_anchor_uv':(WRIST_TARGET/223).tolist(), 'target_roi_uv':box(WRIST_TARGET).tolist()}}}
    closing = prediction(phase='grasp', mode='interaction', action='retain')
    closing['target_id'] = closing['target_geometry']['id'] = 'part'
    closing['target_geometry']['allowed_contact'] = 'fingers'
    closing['motion_context'].update(intended_contact_id='part', tool_external_roi_uv=TOOL_ROI.tolist())
    closing['visual_references'] = copy.deepcopy(first['visual_references'])
    for ref in closing['visual_references'].values(): ref.update(update='retain', seed=None)
    closing['contact_alignment'] = None
    closing['direct_actions'] = [[0.,0.,0.,0.,0.,0.,0.]] + [[0.,0.,0.,0.,0.,0.,1.]]*4
    Sim, Recorder, transport, sims = fixtures(clock, [first, closing], terminal_step=99, request_action_counts=[0,1])
    config = run_args(tmp_path, method_version='v14')
    rgb, wrist = render([0.,0.,0.])

    class StableTrackedSim(Sim):
        def obs(self):
            return replace(observed(), prompt=resolve_distance_task('bowl_plate').prompt,
                image=image_to_data_url(rgb,rotate=False), wrist_image=image_to_data_url(wrist,rotate=False))

        def step(self, action):
            result = super().step(action)
            if len(self.actions) == 3:
                (config.output/'operator_stop.request.json').write_text('{"reason":"offline retained-warning boundary"}')
            return result

    result = runner.execute(config, token_provider=lambda:'offline-oauth', transport=transport,
        sim_factory=StableTrackedSim, recorder_factory=Recorder)
    assert result['status'] == 'user_stop' and result['steps'] == 3
    assert len(transport.calls) == 2
    rows = [json.loads(line) for line in (config.output/'trace.jsonl').read_text().splitlines()]
    actions = [row for row in rows if row['event'] == 'action']
    assert actions[0]['diagnostics']['feedback']['contact_alignment']['active']
    assert actions[1]['diagnostics']['feedback']['contact_alignment']['active']
    assert actions[1]['action'] == closing['direct_actions'][0]
    final = actions[-1]['diagnostics']
    assert final['selection']['status'] == 'observed_contact_alignment_incompatible_with_close'
    assert final['rejected_model_action'][6] == 1.
    assert sims[0].actions[-1] == tuple(measured_current_pose_hold_action())
    assert all(command[6] == 0. for command in sims[0].actions)
    assert final['target_contact_admission']['accepted']  # Refused by image mismatch, not synthetic palm geometry.
    assert 'V8_EVALUATOR_PRIVATE' not in json.dumps(transport.calls)
