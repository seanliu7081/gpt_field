"""Execution tests use measured synthetic motion, never model or simulator truth."""
import copy

import numpy as np
import pytest

from gpt_field.visual_field_v4 import MetricFieldExecutorV4, validate_metric_intent


Q = [0., -.5, 0., -1., 0., 1., 0.]


def state():
    return np.array([0., 0., 1., 0., 0., 0., .04, -.04])


def intent(**updates):
    result = {"goal_offset_m": [.07, 0., 0.], "orientation_target_xyzw": [0., 0., 0., 1.],
              "gripper_command": "hold", "goal_tolerance_m": .003, "max_actions": 30,
              "max_travel_m": .16, "command_limit_m": .01, "tool_radius_m": .005,
              "uncertainty_m": .001, "obstacles": [], "unknown_regions": []}
    result.update(updates)
    return result


def box(ident="obstacle", *, low=(.025, -.008, -.008), high=(.026, .008, .008)):
    return {"id": ident, "lower_offset_m": list(low), "upper_offset_m": list(high), "uncertainty_m": .001}


def step(executor, before, *, gain=.25, **kwargs):
    selected = executor.select(before, Q, **kwargs)
    assert selected["accepted"], selected["status"]
    after = before.copy()
    after[:3] += np.asarray(selected["action"][:3])*.05*gain
    executor.observe(before, selected["action"], after)
    return after, selected


def test_one_intent_drives_more_than_five_measured_moving_actions_without_reconfigure():
    executor, measured = MetricFieldExecutorV4(), state()
    executor.configure(measured, intent())
    for _ in range(20):
        measured, result = step(executor, measured)
        assert result["status"] == "metric_field_tracking"
        assert result["action"][3:] == [0.]*4
    report = executor.summary()
    assert report["effective_moving_actions"] == report["nonzero_actions"] == 20
    assert measured[0] == pytest.approx(.05)
    assert report["plan_calls"] == 1
    assert report["clearance_certified"] is report["task_success_certified"] is False


def test_tracking_chases_observed_pose_instead_of_consuming_waypoints_by_action_count():
    executor, measured = MetricFieldExecutorV4(), state()
    executor.configure(measured, intent())
    first = executor.select(measured, Q)
    # A stalled first command does not advance the path to the next waypoint.
    executor.observe(measured, first["action"], measured)
    repeated = executor.select(measured, Q)
    assert repeated["action"] == pytest.approx(first["action"])
    assert repeated["diagnostics"]["tracking"]["progress_m"] == 0.
    after = measured.copy(); after[0] += .002
    executor.observe(measured, repeated["action"], after)
    following = executor.select(after, Q)
    assert following["diagnostics"]["tracking"]["progress_m"] == pytest.approx(.002)


def test_path_horizon_replanning_reuses_the_same_cached_distance_solution():
    executor, measured = MetricFieldExecutorV4(), state()
    executor.configure(measured, intent(goal_offset_m=[.07, .07, .07], command_limit_m=.02))
    for _ in range(20):
        selected = executor.select(measured, Q)
        if not selected["accepted"]:
            assert selected["status"] == "measured_local_goal_arrival"
            break
        after = measured.copy(); after[:3] += np.asarray(selected["action"][:3])*.05
        executor.observe(measured, selected["action"], after)
        measured = after
    else:
        pytest.fail("Synthetic perfect tracking should arrive within twenty actions")
    assert executor.summary()["distance_cache_hits"] >= 1
    assert executor.summary()["plan_calls"] >= 2


@pytest.mark.parametrize("group", ["obstacles", "unknown_regions"])
def test_inflated_thin_regions_force_a_checked_detour(group):
    executor = MetricFieldExecutorV4()
    executor.configure(state(), intent(**{group: [box(high=(.0251, .008, .008))]}))
    result = executor.select(state(), Q)
    assert result["accepted"]
    path = np.asarray(result["diagnostics"]["plan"]["candidates"][0]["positions"])
    assert max(np.max(abs(path[:, 1])), np.max(abs(path[:, 2]-1.))) >= .015
    assert not executor.checked.segment_valid(state()[:3], executor.goal)
    assert all(executor.checked.segment_valid(a, b) for a, b in zip(path[:-1], path[1:]))
    assert executor.summary()["inflated_regions"][0]["inflation_m"] == pytest.approx(.007)


def test_separating_unknown_wall_has_no_path_and_never_executes_a_shortcut():
    executor = MetricFieldExecutorV4()
    executor.configure(state(), intent(unknown_regions=[box(low=(.025, -.2, -.2), high=(.0251, .2, .2))]))
    result = executor.select(state(), Q)
    assert result["status"] == "field_goal_unreachable_in_known_grid"
    assert not result["accepted"] and result["need_refresh"]
    assert result["action"] == [0.]*7


@pytest.mark.parametrize("goal_region", [False, True])
def test_start_or_goal_conflict_remains_hard_without_automatic_carving(goal_region):
    x = .07 if goal_region else 0.
    executor = MetricFieldExecutorV4()
    executor.configure(state(), intent(obstacles=[box(low=(x-.001, -.001, -.001), high=(x+.001, .001, .001))]))
    result = executor.select(state(), Q)
    assert result["status"] == ("invalid_goal_hypothesis" if goal_region else "invalid_measured_start")
    assert len(executor.boxes) == 1 and not result["accepted"]


def test_all_eight_obstacles_and_eight_unknown_regions_are_retained():
    groups = {name: [box(f"{name}_{n}", low=(.2+n*.005, .2, .2), high=(.201+n*.005, .201, .201))
                     for n in range(8)] for name in ("obstacles", "unknown_regions")}
    executor = MetricFieldExecutorV4()
    executor.configure(state(), intent(**groups))
    result = executor.select(state(), Q)
    assert result["accepted"]
    assert len(executor.boxes) == len(result["diagnostics"]["inflated_regions"]) == 16


def test_pose_orientation_requires_direct_and_joint_limit_stops_motion():
    executor = MetricFieldExecutorV4()
    executor.configure(state(), intent(orientation_target_xyzw=[0., 0., np.sin(.1), np.cos(.1)]))
    assert executor.select(state(), Q)["status"] == "orientation_requires_direct"
    executor.configure(state(), intent())
    q = Q.copy(); q[3] = 0.
    assert executor.select(state(), q)["status"] == "measured_joint_limit"


def test_tighter_command_cap_and_prospective_measured_travel_budget():
    executor, measured = MetricFieldExecutorV4(), state()
    executor.configure(measured, intent(max_travel_m=.015))
    measured, first = step(executor, measured, gain=1., command_cap_m=.004)
    assert np.linalg.norm(first["action"][:3])*.05 == pytest.approx(.004)
    measured, second = step(executor, measured, gain=1.)
    assert measured[0] == pytest.approx(.014)
    third = executor.select(measured, Q)
    assert third["accepted"]
    assert np.linalg.norm(third["action"][:3])*.05 <= .001+1e-12
    after = measured.copy(); after[:3] += np.asarray(third["action"][:3])*.05
    executor.observe(measured, third["action"], after)
    assert executor.select(after, Q)["status"] == "intent_travel_budget"


def test_observed_segment_violation_latches_refresh_and_cannot_be_cleared_by_selection():
    executor, measured = MetricFieldExecutorV4(), state()
    executor.configure(measured, intent(obstacles=[box()]))
    selected = executor.select(measured, Q)
    assert selected["accepted"]
    after = measured.copy(); after[0] = .06  # Actual motion crosses the box.
    executor.observe(measured, selected["action"], after)
    assert executor.select(after, Q)["status"] == "measured_segment_outside_hypothesis"


def test_stall_is_observed_motion_failure_not_a_timer():
    executor, measured = MetricFieldExecutorV4(), state()
    executor.configure(measured, intent())
    for _ in range(5):
        measured, _ = step(executor, measured, gain=0.)
    report = executor.summary()
    assert report["effective_moving_actions"] == 0
    assert report["nonzero_actions"] == 5
    assert executor.select(measured, Q)["status"] == "measured_motion_stall"


def test_intent_action_budget_and_measured_arrival_request_model_refresh():
    executor, measured = MetricFieldExecutorV4(), state()
    executor.configure(measured, intent(max_actions=10))
    for _ in range(10):
        measured, _ = step(executor, measured, gain=.1)
    assert executor.select(measured, Q)["status"] == "intent_action_budget"
    executor.configure(state(), intent(goal_offset_m=[.003, 0., 0.]))
    assert executor.select(state(), Q)["status"] == "measured_local_goal_arrival"


def test_action_acknowledgement_and_state_consistency_cannot_be_bypassed():
    executor, measured = MetricFieldExecutorV4(), state()
    executor.configure(measured, intent())
    selected = executor.select(measured, Q)
    with pytest.raises(RuntimeError, match="observe"):
        executor.select(measured, Q)
    with pytest.raises(ValueError, match="match"):
        executor.observe(measured, [0.]*7, measured)
    executor.observe(measured, selected["action"], measured)
    changed = measured.copy(); changed[0] += .001
    assert executor.select(changed, Q)["status"] == "unacknowledged_state_change"


def test_retarget_preserves_geometry_and_budgets_and_limits_accumulated_shift():
    executor, measured = MetricFieldExecutorV4(), state()
    executor.configure(measured, intent())
    measured, _ = step(executor, measured)
    previous = executor.summary()
    for _ in range(3):
        assert executor.retarget(executor.goal+[0., .003, 0.])
    before_rejection = executor.goal.copy()
    assert not executor.retarget(executor.goal+[0., .002, 0.])
    assert not executor.retarget(executor.goal+[0., 0., .001])
    assert np.array_equal(executor.goal, before_rejection)
    assert np.array_equal(executor.origin, state()[:3])
    assert executor.summary()["executed_actions"] == previous["executed_actions"]
    assert executor.summary()["measured_travel_m"] == previous["measured_travel_m"]
    assert executor.tracker.positions is None
    assert executor.select(measured, Q)["accepted"]


def test_retarget_cannot_enter_a_retained_unknown_region():
    executor = MetricFieldExecutorV4()
    # Inflation starts at y=.002, just beyond the original goal.
    executor.configure(state(), intent(unknown_regions=[box(low=(.067, .009, -.002), high=(.073, .012, .002))]))
    original = executor.goal.copy()
    assert not executor.retarget(original+[0., .003, 0.])
    assert np.array_equal(executor.goal, original)


@pytest.mark.parametrize("update", [
    {"goal_offset_m": [.081, 0., 0.]}, {"gripper_command": "close"}, {"max_actions": True},
    {"command_limit_m": float("nan")}, {"orientation_target_xyzw": [0., 0., 0., 0.]},
    {"obstacles": [box()] * 9}, {"obstacles": [box()], "unknown_regions": [box()]},
])
def test_invalid_intent_rejected_without_clipping(update):
    original = intent(**update)
    snapshot = copy.deepcopy(original)
    with pytest.raises(ValueError):
        validate_metric_intent(original)
    assert repr(original) == repr(snapshot)


def test_region_retains_full_center_extent_envelope():
    region = box(low=(-1., -.1, -.1), high=(0., .1, .1))
    checked = validate_metric_intent(intent(obstacles=[region]))
    assert checked["obstacles"] == [region]
    with pytest.raises(ValueError, match="ordered"):
        validate_metric_intent(intent(obstacles=[box(low=(-1.001, -.1, -.1))]))
