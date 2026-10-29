"""Fine lattice accuracy, geometry, cost and compact-storage regressions."""
import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.compact_field import expand_compact_field
from gpt_field.distance_planner import DistanceFieldPlanner
from gpt_field.field import GridField
from gpt_field.fine_distance_planner import FINE_GRID, FineCompactField, FineDistancePlanner
from gpt_field.schema import FieldPrediction, GridSpec

CENTER = (0., 0., .5)


def case(grid=FINE_GRID, **changes):
    compact = {"stage": "transport", "orientation_target_xyzw": [0., 0., 0., 1.],
               "gripper_command": "close", "motion_mode": "move", "goal_offset_m": [.06, 0., 0.],
               "attraction_weights": [1., 2., 3.], "repulsors": [],
               "blocked_boxes": [], "unknown_boxes": [], **changes}
    return FineCompactField(grid, CENTER, compact), compact


def plan(field, compact, *, start=CENTER, tolerance=.0025, planner=None):
    return (planner or FineDistancePlanner()).plan(field, start,
        goal_xyz=np.asarray(CENTER) + compact["goal_offset_m"], compact_prediction=compact,
        goal_tolerance_m=tolerance)


def assert_valid(result, field):
    assert result.status == "move"
    assert result.candidate_count == result.valid_count == 1
    points = np.asarray(result.candidates[0].positions)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert 1 <= len(lengths) <= 10
    assert np.all(lengths > 0)
    assert np.all(lengths <= .01 + 1e-12)
    assert all(field.segment_valid(a, b) for a, b in zip(points[:-1], points[1:]))
    json.dumps(result.to_dict(), allow_nan=False)


def test_actual_1mm_lattice_and_small_serialization_without_coordinate_allocation(monkeypatch):
    monkeypatch.setattr(GridSpec, "coordinates", lambda *args: pytest.fail("dense coordinates allocated"))
    field, compact = case()
    assert field.grid == GridSpec(161, .001)
    assert field.grid.count == 4173281
    assert np.allclose(field.upper - field.lower, .16)
    assert len(json.dumps(field.to_dict())) < 2000
    assert field.prediction.to_dict()["dense_values_materialized"] is False
    result = plan(field, compact)
    assert_valid(result, field)
    assert result.search_metadata["logical_vertex_count"] == 4173281
    assert result.search_metadata["settled_vertex_count"] == 0
    assert result.start_potential == pytest.approx(.0575)


def test_lazy_trilinear_values_equal_dense_expansion_at_vertices_and_fractional_points():
    grid = GridSpec(17, .005)
    field, compact = case(grid, goal_offset_m=[.02, .01, -.01], repulsors=[{
        "center_offset_m": [-.01, -.01, 0.], "radius_m": .01, "weight": 2.}])
    dense = GridField(grid, CENTER, FieldPrediction.from_dict(expand_compact_field(compact, grid), grid))
    rng = np.random.default_rng(7)
    for p in np.r_[grid.coordinates(CENTER)[::89], rng.uniform(field.lower, field.upper, (100, 3))]:
        assert field.query(p) == pytest.approx(dense.query(p), abs=1e-12)


def test_box_stencil_validity_agrees_with_dense_and_continuous_check():
    grid = GridSpec(9, .01)
    field, compact = case(grid, goal_offset_m=[.03, 0., 0.], blocked_boxes=[{
        "lower_offset_m": [.005, -.012, -.012], "upper_offset_m": [.015, .012, .012]}])
    dense = GridField(grid, CENTER, FieldPrediction.from_dict(expand_compact_field(compact, grid), grid))
    old = DistanceFieldPlanner()
    boxes = old._boxes(dense, compact)
    rng = np.random.default_rng(2)
    for a, b in rng.uniform(field.lower, field.upper, (60, 2, 3)):
        assert field.segment_valid(a, b) == old._valid_segment(dense, a, b, boxes)


@pytest.mark.parametrize("kind", ["blocked_boxes", "unknown_boxes"])
def test_submillimeter_continuous_wall_not_lost_by_fine_vertex_sampling(kind):
    grid = GridSpec(11, .001)
    field, compact = case(grid, goal_offset_m=[.004, 0., 0.], **{kind: [{
        "lower_offset_m": [.0012, -.005, -.005], "upper_offset_m": [.0013, .005, .005]}]})
    result = plan(field, compact, tolerance=.0001)
    assert result.status == "no_valid_candidates"
    assert result.reason == "goal_unreachable_in_known_grid"


def test_1mm_wall_detour_retains_actual_lattice_and_continuous_validity():
    field, compact = case(GridSpec(21, .001), goal_offset_m=[.009, 0., 0.], blocked_boxes=[{
        "lower_offset_m": [.003, -.003, -.01], "upper_offset_m": [.004, .003, .01]}])
    result = plan(field, compact, tolerance=.0005)
    assert_valid(result, field)
    assert result.search_metadata["method"] == "lazy_fine_grid_astar"
    assert result.start_potential > .01
    points = np.asarray(result.candidates[0].positions)
    assert np.max(np.abs(points[:, 1])) >= .004 - 1e-12
    assert result.search_metadata["grid"]["spacing_m"] == .001


@pytest.mark.parametrize("kind", ["blocked_boxes", "unknown_boxes"])
def test_astar_shortest_cost_matches_reference_dijkstra_for_masks(kind):
    grid = GridSpec(9, .005)
    field, compact = case(grid, goal_offset_m=[.015, 0., 0.], **{kind: [{
        "lower_offset_m": [.005, -.005, -.02], "upper_offset_m": [.005, .005, .02]}]})
    dense = GridField(grid, CENTER, FieldPrediction.from_dict(expand_compact_field(compact, grid), grid))
    old = plan(dense, compact, tolerance=.0005, planner=DistanceFieldPlanner())
    new = plan(field, compact, tolerance=.0005)
    assert_valid(new, field)
    assert new.start_potential == pytest.approx(old.start_potential, abs=1e-12)


def test_repulsor_cost_not_discarded_and_exact_astar_matches_dijkstra():
    grid = GridSpec(13, .005)
    field, compact = case(grid, goal_offset_m=[.025, 0., 0.], repulsors=[{
        "center_offset_m": [.0125, 0., 0.], "radius_m": .005, "weight": 10.}])
    dense = GridField(grid, CENTER, FieldPrediction.from_dict(expand_compact_field(compact, grid), grid))
    new, old = plan(field, compact, tolerance=.0005), plan(dense, compact, tolerance=.0005, planner=DistanceFieldPlanner())
    assert_valid(new, field)
    assert new.search_metadata["method"] == "lazy_fine_grid_astar"
    assert new.start_potential == pytest.approx(old.start_potential, abs=1e-12)
    assert np.max(np.linalg.norm(np.asarray(new.candidates[0].positions)[:, 1:] - [0., .5], axis=1)) > .005


def test_dense_masks_forbid_diagonal_corner_cutting():
    grid = GridSpec(3, .001)
    field, compact = case(grid, goal_offset_m=[.001, .001, 0.])
    prediction = FieldPrediction.from_dict(expand_compact_field(compact, grid), grid)
    dense = GridField(grid, CENTER, replace(prediction, blocked_indices=tuple(i for i in range(27) if i not in {13, 25})))
    result = plan(dense, compact, tolerance=.0001)
    assert result.reason == "goal_unreachable_in_known_grid"


def test_fractional_start_goal_tiny_region_and_tracking_tolerance():
    field, compact = case(goal_offset_m=[.02713, .00327, .00234])
    start = (.00117, -.00219, .50311)
    result = plan(field, compact, start=start, tolerance=.00005)
    assert_valid(result, field)
    goal = np.asarray(CENTER) + compact["goal_offset_m"]
    assert np.linalg.norm(np.asarray(result.candidates[0].positions[-1]) - goal) == pytest.approx(.00005)
    tracked = FineDistancePlanner().plan_analytic(field, start, goal_xyz=goal, compact_prediction=compact, goal_tolerance_m=.005)
    assert tracked.search_metadata["goal_tolerance_m"] == .0025


def test_cache_context_changes_when_goal_or_geometry_changes_and_no_dense_distance():
    planner = FineDistancePlanner()
    field, compact = case()
    assert not plan(field, compact, planner=planner).search_metadata["cache_hit"]
    assert plan(field, compact, planner=planner, start=(.001, 0., .5)).search_metadata["cache_hit"]
    changed = copy.deepcopy(compact)
    changed["goal_offset_m"][0] = .05
    assert not plan(field, changed, planner=planner).search_metadata["cache_hit"]
    assert planner.distance_grid_m is None
    planner.reset_cache()
    assert not plan(field, changed, planner=planner).search_metadata["cache_hit"]


@pytest.mark.parametrize("changes,reason", [
    ({"motion_mode": "hold"}, "explicit_motion_hold"),
    ({"goal_offset_m": [.09, 0., 0.]}, "goal_outside_known_grid"),
    ({"goal_offset_m": [.001, 0., 0.]}, "measured_goal_within_tolerance"),
    ({"unknown_boxes": [{"lower_offset_m": [.055, -.01, -.01], "upper_offset_m": [.065, .01, .01]}]}, "goal_region_not_known_valid"),
])
def test_stop_contracts(changes, reason):
    field, compact = case(**changes)
    assert plan(field, compact).reason == reason
