"""V8 controller-contract counterexamples, with no model or task replay."""
import copy
import math

import numpy as np
import pytest

from gpt_field.action import _quaternion_product, _rotvec_quaternion
from gpt_field.visual_field_v6 import MetricFieldExecutorV6
from gpt_field.visual_field_v8 import (
    GripperSettleExecutorV8, MetricFieldExecutorV8,
    measured_current_pose_hold_action, world_orientation_error_rotvec,
)
from test_visual_field_v4 import Q, box, state
from test_visual_field_v6 import axis_intent


def matrix(rotvec):
    """Independent Rodrigues reference, not the production quaternion helper."""
    vector = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(vector)
    if theta < 1e-15:
        return np.eye(3)
    x, y, z = vector/theta
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(theta)*skew + (1-np.cos(theta))*(skew@skew)


def rotvec_from_quat(quat):
    quat = np.asarray(quat)
    if quat[3] < 0:
        quat = -quat
    sine = np.linalg.norm(quat[:3])
    return np.zeros(3) if sine < 1e-15 else quat[:3]/sine*2*np.arctan2(sine, quat[3])


def configured(measured=None, **changes):
    measured = state() if measured is None else measured.copy()
    value = axis_intent(orientation_target_xyzw=_rotvec_quaternion(measured[3:6]).tolist(), **changes)
    executor = MetricFieldExecutorV8()
    executor.configure(measured, value)
    return executor, measured


def acknowledge_drift(executor, measured, drift):
    chosen = executor.select(measured, Q)
    assert chosen["accepted"]
    after = measured.copy()
    after[:3] += np.asarray(chosen["action"][:3])*.05*.25
    after[3:6] = rotvec_from_quat(_quaternion_product(
        _rotvec_quaternion(np.asarray(drift)), _rotvec_quaternion(measured[3:6])))
    executor.observe(measured, chosen["action"], after)
    return after


def test_first_zero_error_command_replaces_stale_osc_target_without_reorientation():
    executor, measured = configured()
    selected = executor.select(measured, Q)
    physical_rotation = np.asarray(selected["action"][3:6])*.5
    # Installed OSC set_goal only replaces goal_ori for a nonzero delta. The
    # epsilon must survive that exact gate, including at the first action.
    assert any(not math.isclose(v, 0.) for v in physical_rotation)
    assert np.linalg.norm(physical_rotation) == pytest.approx(1e-9, abs=1e-15)
    assert selected["diagnostics"]["orientation_hold"]["goal_refresh_epsilon_used"]
    assert selected["action"][6] == 0.


@pytest.mark.parametrize("initial", [[.8, -.4, .6], [math.pi-.001, .0, .0]])
@pytest.mark.parametrize("world_drift", [[.012, .008, -.004], [.0, -.035, .0]])
def test_world_axis_correction_reduces_so3_error_at_noncommuting_and_near_pi_poses(initial, world_drift):
    measured = state(); measured[3:6] = initial
    executor, measured = configured(measured)
    after = acknowledge_drift(executor, measured, world_drift)
    chosen = executor.select(after, Q)
    assert chosen["accepted"]
    correction = np.asarray(chosen["action"][3:6])*.5
    # Actual OSC convention is R_next_goal = exp(delta_world) @ R_current.
    target, current = matrix(initial), matrix(after[3:6])
    corrected = matrix(correction)@current
    assert np.linalg.norm(corrected-target) < np.linalg.norm(current-target)*.5
    assert np.linalg.norm(correction) <= .02+1e-12
    expected_error = world_orientation_error_rotvec(_rotvec_quaternion(np.asarray(initial)), after[3:6])
    assert matrix(expected_error)@current == pytest.approx(target, abs=1e-12)
    assert matrix(expected_error)@current != pytest.approx(current@matrix(expected_error), abs=1e-5)


def test_correction_norm_is_capped_as_a_vector_not_each_component():
    executor, measured = configured()
    after = acknowledge_drift(executor, measured, np.ones(3)*.022)
    chosen = executor.select(after, Q)
    assert chosen["accepted"]
    correction = np.asarray(chosen["action"][3:6])*.5
    assert np.linalg.norm(correction) == pytest.approx(.02)
    assert correction/np.linalg.norm(correction) == pytest.approx(-np.ones(3)/np.sqrt(3))


def test_near_pi_rotation_vector_sign_change_does_not_create_false_large_error():
    measured = state(); measured[3] = np.pi-1e-5
    executor, measured = configured(measured)
    chosen = executor.select(measured, Q)
    after = measured.copy(); after[3] = -np.pi+1e-5; after[0] += .001
    executor.observe(measured, chosen["action"], after)
    result = executor.select(after, Q)
    assert result["accepted"]
    assert np.linalg.norm(result["action"][3:6])*.5 == pytest.approx(2e-5, abs=1e-12)


def test_large_measured_drift_still_latches_stop_and_never_expands_tolerance():
    executor, measured = configured()
    after = acknowledge_drift(executor, measured, [.04001, 0., 0.])
    result = executor.select(after, Q)
    assert result["status"] == "orientation_requires_direct"
    assert not result["accepted"]
    assert result["action"] == [0.]*7
    assert result["diagnostics"]["orientation_hold"]["hard_drift_limit_rad"] == .04


def test_incoming_task_defined_orientation_is_rejected_and_measured_target_is_immutable():
    value = axis_intent(orientation_target_xyzw=_rotvec_quaternion(np.array([0., .01, 0.])).tolist())
    with pytest.raises(ValueError, match="measured refresh"):
        MetricFieldExecutorV8().configure(state(), value)
    measured = state(); measured[3:6] = [.6, -.2, .1]
    value = axis_intent(orientation_target_xyzw=(-_rotvec_quaternion(measured[3:6])).tolist())
    original = copy.deepcopy(value)
    executor = MetricFieldExecutorV8(); executor.configure(measured, value)
    assert value == original
    first_target = executor.summary()["orientation_hold"]["target_xyzw"]
    after = acknowledge_drift(executor, measured, [.003, .004, .0])
    assert executor.retarget(executor.goal+[0., .002, 0.])
    assert executor.summary()["orientation_hold"]["target_xyzw"] == first_target
    assert executor.select(after, Q)["accepted"]


def test_pending_acknowledges_exact_corrected_rotation_and_rejects_zeroed_command():
    executor, measured = configured()
    after = acknowledge_drift(executor, measured, [0., 0., .018])
    chosen = executor.select(after, Q)
    corrupted = chosen["action"].copy(); corrupted[3:6] = [0.]*3
    with pytest.raises(ValueError, match="match"):
        executor.observe(after, corrupted, after)
    with pytest.raises(RuntimeError, match="acknowledge"):
        executor.select(after, Q)
    executor.observe(after, chosen["action"], after)
    assert executor.summary()["executed_actions"] == 2


@pytest.mark.parametrize("geometry", ["obstacles", "unknown_regions"])
def test_orientation_holding_preserves_planner_path_axis_mask_and_region_inflation(geometry):
    value = axis_intent(**{geometry: [box(high=(.0251, .008, .008))]}, max_actions=120)
    old, new = MetricFieldExecutorV6(), MetricFieldExecutorV8()
    old.configure(state(), value); new.configure(state(), value)
    a, b = old.select(state(), Q), new.select(state(), Q)
    assert a["accepted"] and b["accepted"]
    assert a["action"][:3] == pytest.approx(b["action"][:3])
    assert a["diagnostics"]["plan"] == b["diagnostics"]["plan"]
    assert old.regions == new.regions
    assert new.summary()["max_actions"] == 120
    assert new.summary()["grid_spacing_m"] == .005
    assert new.summary()["grid_size"] == 33
    assert not new.checked.segment_valid([.01, 0., 1.001], [.04, 0., 1.001])
    after = state(); after[2] += .0031
    new.observe(state(), b["action"], after)
    assert new.select(after, Q)["status"] == "locked_axis_pose_drift"


def test_stale_osc_goal_plant_counterexample_reuses_field_without_repeated_model_correction():
    results = {}
    for kind, executor in [("zero_rotation", MetricFieldExecutorV6()), ("absolute_hold", MetricFieldExecutorV8())]:
        measured = state()
        executor.configure(measured, axis_intent(max_actions=120, command_limit_m=.003))
        # Previously requested Direct yaw target is unfinished. OSC preserves
        # it on exact-zero rotation; nonzero rotation updates in world axes.
        persistent_yaw_goal = .10
        yaw_velocity = .015
        for index in range(35):
            result = executor.select(measured, Q)
            if not result["accepted"]:
                break
            delta = np.asarray(result["action"])*np.array([.05]*3+[.5]*3+[1.])
            if any(not math.isclose(v, 0.) for v in delta[3:6]):
                persistent_yaw_goal = measured[5]+delta[5]
            yaw_velocity = .5*yaw_velocity+.25*(persistent_yaw_goal-measured[5])
            after = measured.copy(); after[:3] += delta[:3]*.4; after[5] += yaw_velocity
            executor.observe(measured, result["action"], after)
            measured = after
        results[kind] = executor.summary(), measured
    old, old_state = results["zero_rotation"]
    new, new_state = results["absolute_hold"]
    assert old["stop_reason"] == "orientation_requires_direct"
    assert old["executed_actions"] < 5
    assert new["executed_actions"] == 35
    assert new["stop_reason"] is None
    assert new["orientation_hold"]["max_measured_error_rad"] < .02
    assert new_state[0] > .04 and new_state[0] > old_state[0]*5
    assert not new["clearance_certified"] and not new["task_success_certified"]


@pytest.mark.parametrize("command,sign", [("open", -1.), ("close", 1.)])
def test_gripper_keeps_explicit_sign_zero_translation_and_17_measured_settle_actions(command, sign):
    executor, measured = GripperSettleExecutorV8(), state()
    executor.configure(measured, {"command": command, "max_actions": 25})
    held_orientation = executor.summary()["orientation_hold"]["target_xyzw"]
    for index in range(17):
        chosen = executor.select(measured, Q)
        assert chosen["accepted"]
        assert chosen["action"][:3] == [0.]*3
        assert chosen["action"][6] == sign
        after = measured.copy()
        if index < 14:
            after[6:] += [-.0004, .0004]
        # A bounded externally caused deviation is corrected in-place; no new
        # target or automatic closure is inferred from those finger changes.
        after[5] = .01 if index == 0 else measured[5]+chosen["action"][5]*.5*.4
        executor.observe(measured, chosen["action"], after)
        measured = after
    result = executor.select(measured, Q)
    assert not result["accepted"] and result["status"] == "gripper_settled"
    assert result["action"] == [0.]*7
    assert executor.summary()["executed_actions"] == 17
    assert executor.summary()["orientation_hold"]["target_xyzw"] == held_orientation
    assert not executor.summary()["grasp_certified"]
    assert not executor.summary()["task_success_certified"]


@pytest.mark.parametrize("index,amount,reason", [
    (0, .0051, "gripper_pose_drift"), (3, .0401, "gripper_orientation_drift"),
])
def test_gripper_hold_never_relaxes_original_pose_or_orientation_guard(index, amount, reason):
    executor, measured = GripperSettleExecutorV8(), state()
    executor.configure(measured, {"command": "close", "max_actions": 20})
    chosen = executor.select(measured, Q)
    after = measured.copy(); after[index] += amount
    executor.observe(measured, chosen["action"], after)
    assert executor.select(after, Q)["status"] == reason
    assert executor.summary()["executed_actions"] == 1


def test_gripper_hold_requires_exact_acknowledgement_and_does_not_consume_budget_on_select():
    executor, measured = GripperSettleExecutorV8(), state()
    executor.configure(measured, {"command": "close", "max_actions": 10})
    chosen = executor.select(measured, Q)
    assert executor.summary()["executed_actions"] == 0
    assert chosen["diagnostics"]["orientation_hold"]["goal_refresh_epsilon_used"]
    corrupted = chosen["action"].copy(); corrupted[3] = 0.
    # The inherited acknowledgement tolerance is 1e-9 normalized; the initial
    # refresh is 2e-9 normalized, so dropping it is still detectable.
    with pytest.raises(ValueError, match="match"):
        executor.observe(measured, corrupted, measured)
    executor.observe(measured, chosen["action"], measured)
    assert executor.summary()["executed_actions"] == 1


def test_gripper_hold_preserves_max_action_budget_when_fingers_do_not_settle():
    executor, measured = GripperSettleExecutorV8(), state()
    executor.configure(measured, {"command": "open", "max_actions": 10})
    for _ in range(10):
        chosen = executor.select(measured, Q)
        after = measured.copy(); after[6:] += [.0003, -.0003]
        executor.observe(measured, chosen["action"], after)
        measured = after
    assert executor.select(measured, Q)["status"] == "gripper_action_budget"


@pytest.mark.parametrize("controller", ["field", "gripper"])
def test_rejected_pose_hold_is_explicit_exact_acknowledged_without_budget_or_stop_reset(controller):
    executor, measured = configured() if controller == "field" else (GripperSettleExecutorV8(), state())
    if controller == "gripper":
        executor.configure(measured, {"command": "open", "max_actions": 20})
    bad_joints = Q.copy(); bad_joints[3] = 0.
    assert executor.select(measured, bad_joints)["status"] == "measured_joint_limit"
    hold = measured_current_pose_hold_action()
    wrong_before = measured.copy(); wrong_before[0] += .001
    with pytest.raises(ValueError, match="match the measured state"):
        executor.observe(wrong_before, hold, measured)
    wrong_command = hold.copy(); wrong_command[0] = .0001
    with pytest.raises(RuntimeError, match="No accepted"):
        executor.observe(measured, wrong_command, measured)
    after = measured.copy(); after[3] += 1e-9
    executor.observe(measured, hold, after)
    assert executor.summary()["executed_actions"] == 0
    assert executor.summary()["stop_reason"] == "measured_joint_limit"
    assert executor.expected_state == pytest.approx(after)
    assert executor.select(after, Q)["status"] == "measured_joint_limit"


@pytest.mark.parametrize("factory", [MetricFieldExecutorV8, GripperSettleExecutorV8])
def test_pose_hold_cannot_be_acknowledged_before_configuration(factory):
    with pytest.raises(RuntimeError, match="configure"):
        factory().observe(state(), measured_current_pose_hold_action(), state())
