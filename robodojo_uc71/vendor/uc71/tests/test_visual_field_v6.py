"""Measured-pose counterexamples for constrained planning and local settling."""
import copy

import numpy as np
import pytest

from gpt_field.visual_field_v5 import MetricFieldExecutorV5
from gpt_field.visual_field_v6 import (
    GripperSettleExecutor,
    MetricFieldExecutorV6,
    validate_gripper_settle_intent,
    validate_metric_intent_v6,
)
from test_visual_field_v4 import Q, box, intent, state


def axis_intent(axes="xy", **changes):
    return {**intent(**changes), "motion_axes": axes}


@pytest.mark.parametrize("axes,goal,locked", [
    ("xy", [.06, .035, 0.], [2]),
    ("z", [0., 0., .07], [0, 1]),
])
def test_axis_graph_path_and_executed_commands_remain_in_requested_subspace(axes, goal, locked):
    executor, measured = MetricFieldExecutorV6(), state()
    executor.configure(measured, axis_intent(axes, goal_offset_m=goal, max_actions=120))
    for _ in range(40):
        selected = executor.select(measured, Q)
        if not selected["accepted"]:
            assert selected["status"] == "measured_local_goal_arrival"
            break
        positions = np.asarray(selected["diagnostics"]["plan"]["candidates"][0]["positions"])
        assert np.max(abs(positions[:, locked]-state()[locked])) < 1e-12
        assert np.max(abs(np.asarray(selected["action"])[locked])) < 1e-12
        after = measured.copy()
        after[:3] += np.asarray(selected["action"][:3])*.05*.4
        executor.observe(measured, selected["action"], after)
        measured = after
    else:
        pytest.fail("Bounded synthetic motion should reach the local subgoal")
    assert executor.summary()["max_actions"] == 120
    assert executor.summary()["grid_size"] == 33
    assert executor.summary()["grid_spacing_m"] == .005


@pytest.mark.parametrize("axes,goal,drift_axis", [("xy", [.07, 0., 0.], 2), ("z", [0., 0., .07], 0)])
@pytest.mark.parametrize("drift", [.0001, .0029])
def test_subgrid_measured_drift_connects_back_to_axis_graph_without_false_mask_failure(axes, goal, drift_axis, drift):
    executor, measured = MetricFieldExecutorV6(), state()
    executor.configure(measured, axis_intent(axes, goal_offset_m=goal))
    first = executor.select(measured, Q)
    assert first["accepted"]
    after = measured.copy()
    after[:3] += np.asarray(first["action"][:3])*.05*.2
    after[drift_axis] += drift
    executor.observe(measured, first["action"], after)
    following = executor.select(after, Q)
    assert following["accepted"], following["status"]
    delta = np.asarray(following["action"][:3])*.05
    assert executor.checked.segment_valid(after[:3], after[:3]+delta)
    assert abs(after[drift_axis]+delta[drift_axis]-measured[drift_axis]) <= drift+1e-12


@pytest.mark.parametrize("axes,goal,drift_axis", [("xy", [.07, 0., 0.], 2), ("z", [0., 0., .07], 1)])
def test_corridor_drift_stops_and_never_recenters_origin(axes, goal, drift_axis):
    executor, measured = MetricFieldExecutorV6(), state()
    executor.configure(measured, axis_intent(axes, goal_offset_m=goal))
    first = executor.select(measured, Q)
    after = measured.copy(); after[drift_axis] += .0031
    executor.observe(measured, first["action"], after)
    result = executor.select(after, Q)
    assert result["status"] == "locked_axis_pose_drift"
    assert not result["accepted"]
    assert np.array_equal(executor.origin, measured[:3])
    assert not executor.retarget(executor.goal)


@pytest.mark.parametrize("group", ["obstacles", "unknown_regions"])
def test_xy_cannot_escape_over_a_wall_that_xyz_can_avoid(group):
    wall = box(low=(.025, -.2, -.003), high=(.026, .2, .003))
    unrestricted, planar = MetricFieldExecutorV6(), MetricFieldExecutorV6()
    unrestricted.configure(state(), axis_intent("xyz", **{group: [wall]}))
    planar.configure(state(), axis_intent("xy", **{group: [wall]}))
    free = unrestricted.select(state(), Q)
    assert free["accepted"]
    path = np.asarray(free["diagnostics"]["plan"]["candidates"][0]["positions"])
    assert np.max(abs(path[:, 2]-state()[2])) > .003
    constrained = planar.select(state(), Q)
    assert constrained["status"] == "field_goal_unreachable_in_known_grid"
    assert not constrained["accepted"]
    assert planar.regions == unrestricted.regions
    assert not planar.checked.segment_valid(state()[:3], planar.goal)


def test_z_cannot_detour_sideways_around_a_retained_obstacle():
    wall = box(low=(-.008, -.008, .025), high=(.008, .008, .026))
    vertical, unrestricted = MetricFieldExecutorV6(), MetricFieldExecutorV6()
    vertical.configure(state(), axis_intent("z", goal_offset_m=[0., 0., .07], obstacles=[wall]))
    unrestricted.configure(state(), axis_intent("xyz", goal_offset_m=[0., 0., .07], obstacles=[wall]))
    assert unrestricted.select(state(), Q)["accepted"]
    assert vertical.select(state(), Q)["status"] == "field_goal_unreachable_in_known_grid"


def test_xy_obstacle_detour_stays_planar_and_preserves_continuous_checks():
    executor = MetricFieldExecutorV6()
    executor.configure(state(), axis_intent(obstacles=[box(high=(.0251, .008, .008))]))
    result = executor.select(state(), Q)
    assert result["accepted"]
    path = np.asarray(result["diagnostics"]["plan"]["candidates"][0]["positions"])
    assert max(abs(path[:, 1])) >= .015
    assert np.max(abs(path[:, 2]-1.)) < 1e-12
    assert all(executor.checked.segment_valid(a, b) for a, b in zip(path[:-1], path[1:]))
    # A between-lattice connector crossing the original inflated obstacle is
    # still invalid even though both endpoints satisfy the axis corridor.
    assert not executor.checked.segment_valid([.01, 0., 1.001], [.04, 0., 1.001])


def test_xyz_retains_v5_path_without_weakening_masks():
    v5, v6 = MetricFieldExecutorV5(), MetricFieldExecutorV6()
    base = intent(unknown_regions=[box()], max_actions=80)
    v5.configure(state(), base)
    v6.configure(state(), {**base, "motion_axes": "xyz"})
    old, new = v5.select(state(), Q), v6.select(state(), Q)
    assert old["accepted"] and new["accepted"]
    assert old["action"] == pytest.approx(new["action"])
    assert old["diagnostics"]["plan"] == new["diagnostics"]["plan"]


def test_retarget_never_moves_locked_components_or_carves_unknown_geometry():
    executor = MetricFieldExecutorV6()
    executor.configure(state(), axis_intent())
    assert executor.retarget(executor.goal+[0., .002, 0.])
    assert not executor.retarget(executor.goal+[0., 0., .0001])
    executor.configure(state(), axis_intent("z", goal_offset_m=[0., 0., .07]))
    assert not executor.retarget(executor.goal+[.0001, 0., 0.])
    assert not executor.retarget(executor.goal+[0., .0001, 0.])
    executor.configure(state(), axis_intent(unknown_regions=[
        box(low=(.067, .009, -.002), high=(.073, .012, .002))]))
    assert not executor.retarget(executor.goal+[0., .003, 0.])


@pytest.mark.parametrize("changes", [
    {"motion_axes": "x"}, {"motion_axes": ["xy"]},
    {"motion_axes": "xy", "goal_offset_m": [.07, 0., .001]},
    {"motion_axes": "z", "goal_offset_m": [.001, 0., .07]},
    {"max_actions": 121}, {"extra": True},
])
def test_invalid_axis_intent_is_rejected_without_mutating_or_clipping(changes):
    value = {**axis_intent(), **changes}
    original = copy.deepcopy(value)
    with pytest.raises(ValueError):
        validate_metric_intent_v6(value)
    assert value == original
    with pytest.raises(ValueError, match="motion_axes"):
        validate_metric_intent_v6(intent())


@pytest.mark.parametrize("command,sign", [("close", 1.), ("open", -1.)])
def test_settle_waits_for_measured_fingers_and_minimum_actions_without_claiming_grasp(command, sign):
    executor, measured = GripperSettleExecutor(), state()
    executor.configure(measured, {"command": command, "max_actions": 20})
    for index in range(7):
        chosen = executor.select(measured, Q)
        assert chosen["accepted"]
        assert chosen["action"] == [0.]*6+[sign]
        after = measured.copy()
        if index < 4:
            after[6:] += [-.001, .001]
        executor.observe(measured, chosen["action"], after)
        measured = after
    result = executor.select(measured, Q)
    assert result["status"] == "gripper_settled"
    assert result["diagnostics"]["executed_actions"] == 7
    assert not result["accepted"]
    assert result["diagnostics"]["grasp_certified"] is False
    assert result["diagnostics"]["task_success_certified"] is False


def test_no_finger_movement_still_executes_five_actions_then_reports_only_settled():
    executor, measured = GripperSettleExecutor(), state()
    executor.configure(measured, {"command": "close", "max_actions": 10})
    for _ in range(5):
        action = executor.select(measured, Q)
        assert action["accepted"]
        executor.observe(measured, action["action"], measured)
    assert executor.select(measured, Q)["status"] == "gripper_settled"
    assert executor.summary()["grasp_certified"] is False


def test_gripper_budget_stops_even_with_unsettled_fingers():
    executor, measured = GripperSettleExecutor(), state()
    executor.configure(measured, {"command": "open", "max_actions": 10})
    for _ in range(10):
        chosen = executor.select(measured, Q)
        assert chosen["accepted"]
        after = measured.copy(); after[6:] += [.0003, -.0003]
        executor.observe(measured, chosen["action"], after)
        measured = after
    assert executor.select(measured, Q)["status"] == "gripper_action_budget"


@pytest.mark.parametrize("index,amount,reason", [
    (0, .0051, "gripper_pose_drift"), (3, .0401, "gripper_orientation_drift"),
])
def test_gripper_pose_drift_stops_before_stability_or_budget(index, amount, reason):
    executor, measured = GripperSettleExecutor(), state()
    executor.configure(measured, {"command": "close", "max_actions": 30})
    chosen = executor.select(measured, Q)
    after = measured.copy(); after[index] += amount
    report = executor.observe(measured, chosen["action"], after)
    assert report["stop_reason"] == reason
    assert executor.select(after, Q)["status"] == reason
    assert report["executed_actions"] == 1


def test_gripper_drift_is_cumulative_and_orientation_uses_so3():
    executor, measured = GripperSettleExecutor(), state()
    measured[3] = np.pi-.00001
    executor.configure(measured, {"command": "close", "max_actions": 30})
    for index in range(3):
        chosen = executor.select(measured, Q)
        assert chosen["accepted"]
        after = measured.copy(); after[0] += .002
        after[3] = -np.pi+.00001
        executor.observe(measured, chosen["action"], after)
        measured = after
    assert executor.select(measured, Q)["status"] == "gripper_pose_drift"
    assert executor.summary()["max_measured_orientation_drift_rad"] < .0001


def test_gripper_acknowledgement_and_measured_joint_limits_are_mandatory():
    executor, measured = GripperSettleExecutor(), state()
    executor.configure(measured, {"command": "close", "max_actions": 20})
    first = executor.select(measured, Q)
    with pytest.raises(RuntimeError, match="observe"):
        executor.select(measured, Q)
    with pytest.raises(ValueError, match="match"):
        executor.observe(measured, [0.]*7, measured)
    executor.observe(measured, first["action"], measured)
    after = measured.copy(); after[0] += .001
    assert executor.select(after, Q)["status"] == "unacknowledged_state_change"
    executor.configure(measured, {"command": "open", "max_actions": 20})
    bad_joints = Q.copy(); bad_joints[3] = 0.
    assert executor.select(measured, bad_joints)["status"] == "measured_joint_limit"


@pytest.mark.parametrize("bad_intent", [
    {}, {"command": "hold", "max_actions": 20},
    {"command": "close", "max_actions": 9}, {"command": "open", "max_actions": 31},
    {"command": "close", "max_actions": True}, {"command": "close", "max_actions": 20.},
    {"command": "close", "max_actions": 20, "target_pose": [0., 0., 0.]},
])
def test_gripper_intent_requires_explicit_bounded_authorization(bad_intent):
    with pytest.raises(ValueError):
        validate_gripper_settle_intent(bad_intent)
