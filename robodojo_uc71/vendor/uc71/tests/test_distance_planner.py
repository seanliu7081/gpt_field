"""Geometry, shortest-path and cache contracts for the distance-field variant."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.compact_field import expand_compact_field
from gpt_field.distance_planner import DistanceFieldPlanner
from gpt_field.field import GridField
from gpt_field.planner import PlannerConfig
from gpt_field.schema import FieldPrediction, GridSpec


CENTER = (0., 0., .5)


def make_case(*, size=9, spacing=.01, goal=(.03, 0., .5), **changes):
    compact = {"stage": "transport", "orientation_target_xyzw": [0., 0., 0., 1.],
               "gripper_command": "close", "motion_mode": "move",
               "goal_offset_m": (np.asarray(goal) - CENTER).tolist(),
               "attraction_weights": [1., 1., 1.], "repulsors": [],
               "blocked_boxes": [], "unknown_boxes": [], **changes}
    grid = GridSpec(size=size, spacing_m=spacing)
    field = GridField(grid, CENTER, FieldPrediction.from_dict(expand_compact_field(compact, grid), grid))
    return field, compact, np.asarray(goal)


def plan(planner, field, compact, goal, position=None, tolerance=.005):
    return planner.plan(field, CENTER if position is None else position,
                        goal_xyz=goal, compact_prediction=compact, goal_tolerance_m=tolerance)


def assert_path(result, field, planner):
    assert result.status == "move"
    assert result.candidate_count == result.valid_count == len(result.candidates) == 1
    assert result.selected_index == 0
    candidate = result.candidates[0]
    points = np.asarray(candidate.positions)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert 1 <= len(lengths) <= planner.config.horizon
    assert np.all(lengths > 0)
    assert np.all(lengths <= planner.config.step_m + 1e-12)
    assert candidate.path_length_m == pytest.approx(lengths.sum())
    assert candidate.path_length_m <= planner.config.horizon * planner.config.step_m + 1e-12
    assert all(field.segment_valid(a, b) for a, b in zip(points[:-1], points[1:]))
    assert result.start_potential >= candidate.immediate_potential >= candidate.terminal_potential >= 0
    assert result.predicted_improvement > 0
    json.dumps(result.to_dict(), allow_nan=False)


def test_regular_33_grid_distance_is_meters_and_arbitrary_potential_is_ignored():
    field, compact, goal = make_case(size=33, spacing=.005, goal=(.06, 0., .5))
    planner = DistanceFieldPlanner()
    result = plan(planner, field, compact, goal)
    assert_path(result, field, planner)
    assert result.search_metadata["method"] == "dijkstra_distance_field"
    assert result.start_potential == pytest.approx(.055)
    assert result.candidates[0].positions[-1] == pytest.approx((.055, 0., .5))
    assert result.search_metadata["endpoint_in_goal_region"]
    assert planner.distance_grid_m.shape == (33, 33, 33)
    assert not planner.distance_grid_m.flags.writeable
    changed = replace(field.prediction, values=tuple(-5000. for _ in field.prediction.values))
    second = plan(planner, GridField(field.grid, field.center, changed), compact, goal)
    assert second.candidates == result.candidates
    assert not second.search_metadata["cache_hit"]


def test_diagonal_edges_measure_euclidean_length_not_manhattan_length():
    field, compact, goal = make_case(goal=(.03, .03, .53))
    planner = DistanceFieldPlanner()
    result = plan(planner, field, compact, goal, tolerance=.001)
    assert_path(result, field, planner)
    assert result.start_potential == pytest.approx(np.sqrt(3) * .03)
    # The graph contains the exact goal vertex; its zero cost need not spend
    # extra motion to exit the goal region at a continuous boundary.
    assert result.start_potential < .09
    assert result.search_metadata["endpoint_in_goal_region"]


def test_short_goal_stops_in_region_without_overshoot():
    field, compact, goal = make_case(goal=(.012, 0., .5))
    result = plan(DistanceFieldPlanner(), field, compact, goal, tolerance=.001)
    points = np.asarray(result.candidates[0].positions)
    assert 0 < points[-1, 0] <= goal[0]
    assert np.linalg.norm(points[-1] - goal) <= .001 + 1e-12
    assert result.search_metadata["path_termination"] == "goal_region"


def test_fractional_start_and_goal_region_with_no_grid_vertex():
    field, compact, goal = make_case(goal=(.027, .003, .502))
    result = plan(DistanceFieldPlanner(), field, compact, goal,
                  position=(.001, -.002, .503), tolerance=.0005)
    assert_path(result, field, DistanceFieldPlanner())
    assert np.linalg.norm(np.asarray(result.candidates[0].positions[-1]) - goal) <= .0005 + 1e-12
    assert result.candidates[0].positions[0] == (.001, -.002, .503)


def test_detours_around_blocked_wall_and_cost_increases():
    wall = {"lower_offset_m": [.01, -.02, -.04], "upper_offset_m": [.01, .02, .04]}
    field, compact, goal = make_case(blocked_boxes=[wall])
    planner = DistanceFieldPlanner()
    result = plan(planner, field, compact, goal)
    assert_path(result, field, planner)
    assert result.start_potential > .05
    assert np.max(np.abs(np.asarray(result.candidates[0].positions)[:, 1])) >= .029
    assert result.search_metadata["endpoint_in_goal_region"]


def test_unknown_wall_is_not_traversable_and_unreachable_is_explicit():
    wall = {"lower_offset_m": [.01, -.04, -.04], "upper_offset_m": [.01, .04, .04]}
    field, compact, goal = make_case(unknown_boxes=[wall])
    result = plan(DistanceFieldPlanner(), field, compact, goal)
    assert result.status == "no_valid_candidates"
    assert result.reason == "goal_unreachable_in_known_grid"
    assert result.candidate_count == 0
    json.dumps(result.to_dict(), allow_nan=False)


def test_subvoxel_continuous_box_cannot_be_missed_by_vertex_masks():
    wall = {"lower_offset_m": [.003, -.04, -.04], "upper_offset_m": [.004, .04, .04]}
    field, compact, goal = make_case(blocked_boxes=[wall])
    assert field.prediction.blocked_indices == ()
    result = plan(DistanceFieldPlanner(), field, compact, goal)
    assert result.reason == "goal_unreachable_in_known_grid"


def test_diagonal_corner_cutting_is_forbidden():
    field, compact, goal = make_case(size=3, spacing=.01, goal=(.01, .01, .5))
    grid = field.grid
    # Only start and diagonal goal vertices remain. A diagonal must not cross
    # the two masked corners between them despite valid endpoints.
    keep = {13, 25}
    blocked = tuple(i for i in range(grid.count) if i not in keep)
    field = GridField(grid, field.center, replace(field.prediction, blocked_indices=blocked))
    result = plan(DistanceFieldPlanner(), field, compact, goal, tolerance=.001)
    assert result.reason == "goal_unreachable_in_known_grid"


@pytest.mark.parametrize("goal", [(0.05, 0., .5), (0., -.05, .5), (0., 0., .55)])
def test_goal_outside_known_cube_requests_observation_instead_of_clipping(goal):
    field, compact, goal = make_case(goal=goal)
    result = plan(DistanceFieldPlanner(), field, compact, goal)
    assert result.status == "needs_observation"
    assert result.reason == "goal_outside_known_grid"
    assert result.search_metadata["needs_observation"]
    assert not result.search_metadata["boundary_clipped"]
    assert not result.search_metadata["goal_reached"]
    assert result.candidates == ()


def test_goal_in_unknown_region_requests_observation():
    box = {"lower_offset_m": [.02, -.01, -.01], "upper_offset_m": [.04, .01, .01]}
    field, compact, goal = make_case(unknown_boxes=[box])
    result = plan(DistanceFieldPlanner(), field, compact, goal)
    assert result.status == "needs_observation"
    assert result.reason == "goal_region_not_known_valid"


def test_cache_reused_from_new_robot_position_but_reset_by_goal_mask_center_and_explicit_reset():
    field, compact, goal = make_case()
    planner = DistanceFieldPlanner()
    assert not plan(planner, field, compact, goal).search_metadata["cache_hit"]
    assert plan(planner, field, compact, goal, position=(.005, 0., .5)).search_metadata["cache_hit"]
    assert not plan(planner, field, compact, (.025, 0., .5)).search_metadata["cache_hit"]
    masked = GridField(field.grid, field.center, replace(field.prediction, blocked_indices=(0,)))
    assert not plan(planner, masked, compact, (.025, 0., .5)).search_metadata["cache_hit"]
    moved = GridField(field.grid, (.001, 0., .5), masked.prediction)
    assert not plan(planner, moved, compact, (.025, 0., .5)).search_metadata["cache_hit"]
    assert not plan(planner, moved, compact, (.025, 0., .5), tolerance=.003).search_metadata["cache_hit"]
    planner.reset_cache()
    assert planner.distance_grid_m is None
    assert not plan(planner, moved, compact, (.025, 0., .5)).search_metadata["cache_hit"]


def test_repulsor_is_positive_traversal_cost_and_can_choose_detour():
    field, compact, goal = make_case(size=17, spacing=.005, goal=(.035, 0., .5))
    unweighted = plan(DistanceFieldPlanner(), field, compact, goal)
    repulsor = {"center_offset_m": [.0175, 0., 0.], "radius_m": .005, "weight": 10.}
    weighted = copy.deepcopy(compact)
    weighted["repulsors"] = [repulsor]
    result = plan(DistanceFieldPlanner(), field, weighted, goal)
    assert result.search_metadata["cost_unit"] == "weighted_m"
    assert result.start_potential > unweighted.start_potential
    assert result.candidates[0].path_length_m > unweighted.candidates[0].path_length_m
    assert np.max(np.linalg.norm(np.asarray(result.candidates[0].positions)[:, 1:] - [0., .5], axis=1)) > .005


def test_workspace_wall_cannot_be_bypassed_outside_workspace():
    wall = {"lower_offset_m": [.01, -.02, -.04], "upper_offset_m": [.01, .02, .04]}
    field, compact, goal = make_case(blocked_boxes=[wall])
    config = PlannerConfig(horizon=10, workspace_min=(-1., -.02, 0.), workspace_max=(1., .02, 1.5))
    result = plan(DistanceFieldPlanner(config), field, compact, goal)
    assert result.reason == "goal_unreachable_in_known_grid"


def test_configured_horizon_and_step_limits_and_fractional_tracking_replan():
    field, compact, goal = make_case(size=17, spacing=.005, goal=(.035, .035, .535))
    planner = DistanceFieldPlanner(PlannerConfig(horizon=2, step_m=.003))
    current = np.asarray(CENTER)
    for iteration in range(30):
        result = plan(planner, field, compact, goal, position=current)
        if result.status == "goal_reached":
            break
        assert_path(result, field, planner)
        assert result.search_metadata["cache_hit"] == (iteration > 0)
        current = np.asarray(result.candidates[0].positions[-1])
    else:
        pytest.fail("repeated bounded replanning never reached the goal region")
    assert np.linalg.norm(current - goal) <= .005 + 1e-12


def test_hold_and_measured_arrival_have_no_invented_candidate():
    field, compact, goal = make_case()
    hold = GridField(field.grid, field.center, replace(field.prediction, motion_mode="hold"))
    result = plan(DistanceFieldPlanner(), hold, compact, goal)
    assert result.status == "hold" and result.candidate_count == 0
    result = plan(DistanceFieldPlanner(), field, compact, goal, position=goal)
    assert result.status == "goal_reached" and result.candidate_count == 0
    assert result.search_metadata["goal_reached"]


@pytest.mark.parametrize("tolerance", [0., -1., True, float("inf"), float("nan")])
def test_invalid_goal_tolerance_is_rejected(tolerance):
    field, compact, goal = make_case()
    with pytest.raises(ValueError, match="goal_tolerance_m"):
        plan(DistanceFieldPlanner(), field, compact, goal, tolerance=tolerance)


def test_distance_variant_rejects_segments_longer_than_one_centimeter():
    with pytest.raises(ValueError, match="must not exceed"):
        DistanceFieldPlanner(PlannerConfig(step_m=.02, horizon=10))


def test_extreme_finite_coordinates_cannot_leak_infinity_to_diagnostics():
    field, compact, _ = make_case()
    with pytest.raises(ValueError, match="distance between"):
        plan(DistanceFieldPlanner(), field, compact, (1e308, 1e308, 1e308))


def test_compatibility_hook_keeps_truthful_algorithm_metadata():
    field, compact, goal = make_case()
    result = DistanceFieldPlanner().plan_analytic(field, CENTER, goal_xyz=goal, compact_prediction=compact)
    assert result.search_metadata["method"] == "dijkstra_distance_field"
    assert result.search_metadata["uses_attraction_potential"] is False
