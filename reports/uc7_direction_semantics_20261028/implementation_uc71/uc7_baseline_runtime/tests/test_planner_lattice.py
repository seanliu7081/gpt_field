"""Exact merged-state planning must preserve exhaustive ranking and path safety."""

from dataclasses import replace
import json

import numpy as np
import pytest

from gpt_field.field import GridField
from gpt_field.planner import LocalPlanner, PlannerConfig
from gpt_field.schema import FieldPrediction, GridSpec
from test_planner import AUX, make_field


def selected(result):
    return result.candidates[result.selected_index]


def equivalent_to_exhaustive(field, config, position=None):
    planner = LocalPlanner(config)
    position = field.center if position is None else position
    exhaustive = planner.plan(field, position)
    merged = planner._plan_lattice(field, position)
    assert merged.status == exhaustive.status
    assert merged.delta_xyz == exhaustive.delta_xyz
    assert merged.search_metadata["represented_valid_sequence_count"] == exhaustive.valid_count
    assert merged.candidate_count == merged.valid_count == len(merged.candidates) <= 6
    if exhaustive.selected_index is not None:
        assert selected(merged).primitives == selected(exhaustive).primitives
        np.testing.assert_allclose(selected(merged).positions, selected(exhaustive).positions,
                                   rtol=0, atol=1e-15)
        assert merged.predicted_improvement == pytest.approx(exhaustive.predicted_improvement)
    else:
        assert merged.selected_index is None
    return merged, exhaustive


@pytest.mark.parametrize("horizon", [3, 4])
@pytest.mark.parametrize("seed", [2, 19])
def test_random_fields_obstacles_and_workspace_match_exhaustive(horizon, seed):
    grid = GridSpec(size=13, spacing_m=.01)
    rng = np.random.default_rng(seed)
    data = {**AUX, "values": rng.uniform(-1, 1, grid.count).tolist(),
            "blocked_indices": [], "unknown_indices": []}
    center = (0., 0., .5)
    coordinates = grid.coordinates(center)
    # Keep start and first moves observable while exercising later segment masks.
    blocked = [i for i, point in enumerate(coordinates)
               if point[0] >= .019 and abs(point[1]) < .011 and point[2] < .491]
    unknown = [i for i, point in enumerate(coordinates)
               if point[1] >= .019 and abs(point[0]) < .011 and point[2] > .509]
    data.update(blocked_indices=blocked, unknown_indices=unknown)
    field = GridField(grid, center, FieldPrediction.from_dict(data, grid))
    config = PlannerConfig(horizon=horizon, workspace_min=(-.025, -.025, .47),
                           workspace_max=(.035, .025, .53))
    equivalent_to_exhaustive(field, config, (.001, -.002, .501))


@pytest.mark.parametrize("scale", [1e-20, 1., 1e20])
def test_tied_terminal_immediate_and_lexicographic_ranking_is_preserved(scale):
    field = make_field(lambda point: -scale * (point[0] + point[1]))
    merged, _ = equivalent_to_exhaustive(field, PlannerConfig(horizon=4))
    assert selected(merged).primitives == ("+x",) * 4
    constant = make_field(lambda _point: scale)
    merged, _ = equivalent_to_exhaustive(constant, PlannerConfig(horizon=3))
    assert merged.status == "plateau_move"
    assert merged.candidate_count == 6


def test_global_terminal_band_precedes_per_first_move_merging():
    # A branch-local minimum of .75*tolerance would incorrectly admit its
    # lexicographically earlier 1.5*tolerance endpoint. The global minimum is
    # zero in a different branch, so that earlier path must be excluded.
    grid = GridSpec(size=11, spacing_m=.01)
    tolerance = 64 * np.finfo(float).eps
    values = np.ones((grid.size,) * 3)
    midpoint = (grid.size - 1) // 2

    def set_value(offset, value):
        values[tuple(midpoint + coordinate for coordinate in offset)] = value

    set_value((1, 0, 0), -.5)  # First-step preference, unreachable as a depth-four endpoint.
    set_value((4, 0, 0), 1.5 * tolerance)
    set_value((3, 1, 0), .75 * tolerance)
    set_value((0, 0, 4), 0.)
    field = GridField(grid, (0., 0., .5), FieldPrediction.from_dict(
        {**AUX, "values": values.ravel().tolist(), "blocked_indices": [], "unknown_indices": []}, grid))
    merged, _ = equivalent_to_exhaustive(field, PlannerConfig(horizon=4))
    assert selected(merged).primitives == ("+x", "+x", "+x", "+y")
    assert merged.search_metadata["terminal_minimum"] == 0.
    assert all(candidate.terminal_potential <= tolerance for candidate in merged.candidates)


def test_state_merging_counts_all_valid_sequences_without_logging_them():
    field = make_field(lambda _point: 1.)
    result, exhaustive = equivalent_to_exhaustive(field, PlannerConfig(horizon=4))
    assert result.search_metadata["nominal_candidate_count"] == 6 ** 4
    assert result.search_metadata["represented_valid_sequence_count"] == 6 * 5 ** 3 == 750
    assert result.search_metadata["layer_state_counts"] == [1, 6, 30, 150, 558]
    assert result.search_metadata["candidate_semantics"] == "terminal_band_finalists"
    assert result.candidate_count == 6
    assert exhaustive.candidate_count == 1296
    assert "search" not in exhaustive.to_dict()


def test_ten_step_hybrid_grid_paths_turn_stay_valid_and_execute_only_first_step():
    grid = GridSpec(size=33, spacing_m=.005)
    field = make_field(lambda point: -point[0] - point[1], grid=grid)
    planner = LocalPlanner(PlannerConfig(horizon=10))
    result = planner.plan(field, field.center)
    assert result.status == "move"
    assert result.delta_xyz == (.01, 0., 0.)
    assert result.search_metadata["method"] == "exact_lattice_dp"
    assert result.search_metadata["nominal_candidate_count"] == 60_466_176
    assert 0 < result.search_metadata["represented_valid_sequence_count"] <= 6 * 5 ** 9
    assert max(result.search_metadata["layer_state_counts"]) < 20_000
    assert len(result.search_metadata["layer_state_counts"]) == 11
    assert result.candidate_count == result.valid_count <= 6
    opposite = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y", "+z": "-z", "-z": "+z"}
    for candidate in result.candidates:
        assert len(candidate.primitives) == 10 and len(candidate.positions) == 11
        assert candidate.path_length_m == pytest.approx(.1)
        assert all(opposite[left] != right for left, right in zip(candidate.primitives, candidate.primitives[1:]))
        for left, right in zip(candidate.positions, candidate.positions[1:]):
            assert np.linalg.norm(np.asarray(right) - left) == pytest.approx(.01)
            assert field.segment_valid(left, right)
        assert np.all(np.asarray(candidate.positions) >= field.lower - 1e-15)
        assert np.all(np.asarray(candidate.positions) <= field.upper + 1e-15)
    assert len(set(selected(result).primitives)) >= 2  # A straight 10 cm path exits the 8 cm half-width.
    assert len(json.dumps(result.to_dict(), allow_nan=False)) < 10_000


def test_no_complete_ten_step_path_never_executes_a_shorter_prefix():
    field = make_field(lambda point: -point[0])
    planner = LocalPlanner(PlannerConfig(horizon=10, workspace_min=(-.001, -.001, .499),
                                        workspace_max=(.015, .001, .501)))
    result = planner.plan(field, field.center)
    assert result.status == "no_valid_candidates" and result.reason == "all_candidates_rejected"
    assert result.delta_xyz == (0., 0., 0.)
    assert result.candidate_count == result.valid_count == 0
    assert result.search_metadata["represented_valid_sequence_count"] == 0
    assert result.search_metadata["layer_state_counts"] == [1, 1] + [0] * 9


def test_long_horizon_hold_and_invalid_start_are_visible_without_search():
    field = make_field(lambda point: -point[0], motion_mode="hold")
    planner = LocalPlanner(PlannerConfig(horizon=10))
    held = planner.plan(field, field.center)
    assert held.status == "hold" and held.search_metadata["expanded_state_count"] == 0
    moving = GridField(field.grid, field.center, replace(field.prediction, motion_mode="move"))
    invalid = planner.plan(moving, (2., 0., .5))
    assert invalid.reason == "invalid_start"
    assert invalid.search_metadata["expanded_state_count"] == 0


def test_fully_valid_grid_skips_general_segment_traversal(monkeypatch):
    field = make_field(lambda point: -point[0] - point[1],
                       grid=GridSpec(size=33, spacing_m=.005))

    def unexpected_segment_check(*_args):
        pytest.fail("a fully valid convex grid needs only validated endpoints")

    monkeypatch.setattr(GridField, "segment_valid", unexpected_segment_check)
    result = LocalPlanner(PlannerConfig(horizon=10)).plan(field, field.center)
    assert result.status == "move"
    assert selected(result).primitives == ("+x",) * 8 + ("+y",) * 2
    assert result.search_metadata["represented_valid_sequence_count"] > 0


@pytest.mark.parametrize("mask_name", ["blocked_indices", "unknown_indices"])
def test_masked_grid_checks_intermediate_vertices_between_valid_endpoints(monkeypatch, mask_name):
    grid = GridSpec(size=33, spacing_m=.005)
    field = make_field(lambda point: -point[0], grid=grid)
    # +x's one-centimeter edge crosses a masked half-centimeter vertex,
    # although both its endpoints are valid. Endpoint-only checks must not run.
    midpoint = (grid.size - 1) // 2
    masked = (midpoint + 1) * grid.size**2 + midpoint * grid.size + midpoint
    field = GridField(grid, field.center, replace(field.prediction, **{mask_name: (masked,)}))
    assert field.query(field.center) is not None
    assert field.query((.01, 0., .5)) is not None
    segment_valid = GridField.segment_valid
    calls = []

    def checked_segment(self, start, end):
        calls.append((start, end))
        return segment_valid(self, start, end)

    monkeypatch.setattr(GridField, "segment_valid", checked_segment)
    planner = LocalPlanner(PlannerConfig(horizon=3, workspace_min=(-.035, -.001, .499),
                                        workspace_max=(.035, .001, .501)))
    result = planner._plan_lattice(field, field.center)
    assert calls
    assert selected(result).primitives == ("-x",) * 3
    assert result.search_metadata["represented_valid_sequence_count"] == 1


@pytest.mark.parametrize("position,minimum,maximum", [
    ((.079, .001, .501), (-1., -1., 0.), (1., 1., 1.5)),
    ((.015, .001, .501), (-.016, -.016, .484), (.016, .016, .516)),
])
def test_fully_valid_fast_path_retains_grid_and_workspace_boundaries(
        monkeypatch, position, minimum, maximum):
    field = make_field(lambda point: -point[0] + .3 * point[1],
                       grid=GridSpec(size=33, spacing_m=.005))
    planner = LocalPlanner(PlannerConfig(horizon=3, workspace_min=minimum, workspace_max=maximum))
    exhaustive = planner.plan(field, position)

    def unexpected_segment_check(*_args):
        pytest.fail("the unmasked lattice path must use endpoint bounds")

    monkeypatch.setattr(GridField, "segment_valid", unexpected_segment_check)
    merged = planner._plan_lattice(field, position)
    assert selected(merged).primitives == selected(exhaustive).primitives
    assert merged.search_metadata["represented_valid_sequence_count"] == exhaustive.valid_count
    for candidate in merged.candidates:
        points = np.asarray(candidate.positions)
        assert np.all(points >= np.maximum(field.lower, minimum) - 1e-15)
        assert np.all(points <= np.minimum(field.upper, maximum) + 1e-15)


@pytest.mark.parametrize("horizon", [0, 11, True, 3.5])
def test_horizon_limits_remain_validated(horizon):
    with pytest.raises(ValueError):
        PlannerConfig(horizon=horizon)
