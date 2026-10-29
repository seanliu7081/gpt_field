from copy import deepcopy

import pytest

from robodojo_uc71.schema import ExecutionLimits, validate_decision, validate_observation, validate_calibration
from robodojo_uc71.planning import NativePlanLock, validate_plan, plan_validation, semantic_plan
from client_fixtures import observation, calibration, decision, plan, frame, TASK


def test_measured_sensor_boundary_and_proxy():
    obs = observation()
    checked = validate_observation(obs)
    assert checked["robot"]["left"]["finger_gap_m"] == .06
    for key in ("reward", "remaining_steps", "object_poses"):
        polluted = observation()
        polluted[key] = 1
        with pytest.raises(ValueError):
            validate_observation(polluted)
    obs["robot"]["left"]["finger_gap_m"] = .01
    with pytest.raises(ValueError, match="proxy"):
        validate_observation(obs)


def test_calibration_is_robot_only_rigid_and_native_resolution():
    value = calibration()
    value["tools"]["left"]["object_pose"] = [0, 0, 0]
    with pytest.raises(ValueError):
        validate_calibration(value)
    value = calibration()
    value["cameras"]["head"]["world_from_optical"][0][0] = 2
    with pytest.raises(ValueError, match="rigid"):
        validate_calibration(value)
    value = calibration()
    value["cameras"]["head"]["width"] = 512
    with pytest.raises(ValueError, match="640x480"):
        validate_calibration(value)


def test_one_selected_arm_all_segments_and_no_macro():
    answer = decision()
    answer.update(type="execute", arm_id="left", execution={"segments": [{"skill": "move_relative", "xyz_m": [.01, 0, 0],
        "quaternion_wxyz": None, "gripper": "keep", "profile": "transit", "max_steps": 20, "observation_after": True}]})
    assert validate_decision(answer)["arm_id"] == "left"
    answer["execution"]["segments"][0]["arm_id"] = "right"
    with pytest.raises(ValueError):
        validate_decision(answer)
    answer = decision()
    answer["type"] = "macro"
    with pytest.raises(ValueError, match="macros"):
        validate_decision(answer)


def test_future_invalid_segment_rejected_and_time_conversion_declared():
    limits = ExecutionLimits()
    assert (limits.max_segment_steps, limits.max_execution_steps, limits.min_open_steps, limits.min_close_steps) == (150, 300, 10, 15)
    answer = decision()
    answer.update(type="execute", arm_id="right", execution={"segments": [{"skill": "set_gripper", "xyz_m": None,
        "quaternion_wxyz": None, "gripper": "close", "profile": "contact", "max_steps": 14, "observation_after": False}]})
    with pytest.raises(ValueError, match="minimum"):
        validate_decision(answer)
    answer["execution"]["segments"][0]["max_steps"] = 15
    assert validate_decision(answer)


def test_plan_preserves_three_cells_and_original_semantic_failure():
    raw = plan()
    assert validate_plan(raw) == raw
    invalid = deepcopy(raw)
    invalid["steps"][0]["sense"] = "close"
    assert validate_plan(invalid, semantic=False) == invalid
    assert plan_validation(invalid)["valid"] is False
    with pytest.raises(ValueError):
        validate_plan(invalid)
    assert invalid["steps"][0]["sense"] == "close"


def test_plan_lock_allows_only_grounding_changes_and_preserves_original():
    raw = plan()
    lock = NativePlanLock(raw, task_prompt=TASK, direction_frame=frame())
    moved = deepcopy(raw)
    moved["parts"][0]["r"] = 22
    assert semantic_plan(moved) == semantic_plan(raw)
    lock.check_plan(moved, source="grounding")
    moved["parts"][0]["text"] = "different package: body"
    with pytest.raises(ValueError, match="semantics"):
        lock.check_plan(moved, source="test")
    assert lock.plan() == raw
