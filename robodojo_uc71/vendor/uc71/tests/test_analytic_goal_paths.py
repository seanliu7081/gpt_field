"""Opt-in analytic paths stop at measured goals and respect local validity."""

from dataclasses import replace
import json

import numpy as np
import pytest

from gpt_field.compact_field import expand_compact_field
from gpt_field.field import GridField
from gpt_field.planner import LocalPlanner, PlannerConfig
from gpt_field.schema import FieldPrediction, GridSpec
from gpt_field.tracking import PathTracker


CENTER = (0., 0., .5)
GRID = GridSpec(size=33, spacing_m=.005)


def compact(goal=(.025, 0., 0.), **changes):
    return {"stage": "approach", "orientation_target_xyzw": [0., 0., 0., 1.],
            "gripper_command": "open", "motion_mode": "move",
            "goal_offset_m": list(goal), "attraction_weights": [.2, 3., 1.],
            "repulsors": [], "blocked_boxes": [], "unknown_boxes": [], **changes}


def field_for(raw):
    return GridField(GRID, CENTER, FieldPrediction.from_dict(expand_compact_field(raw, GRID), GRID))


def plan(raw=None, *, position=CENTER, config=None, **kwargs):
    raw = compact() if raw is None else raw
    field = field_for(raw)
    result = LocalPlanner(config or PlannerConfig(horizon=10)).plan_analytic(
        field, position, goal_xyz=np.asarray(CENTER) + raw["goal_offset_m"],
        compact_prediction=raw, **kwargs)
    return result, field


def selected(result):
    return result.candidates[result.selected_index]


def test_off_grid_goal_shortens_last_segment_and_never_extends_to_fixed_horizon():
    result, field = plan(compact((.023, 0., 0.)))
    candidate = selected(result)
    assert result.status == "move"
    assert len(candidate.primitives) == 3
    assert candidate.path_length_m == pytest.approx(.023)
    assert np.linalg.norm(np.diff(candidate.positions, axis=0), axis=1) == pytest.approx([.01, .01, .003])
    assert candidate.positions[-1] == pytest.approx((.023, 0., .5))
    assert result.search_metadata["endpoint_is_goal"]
    assert not result.search_metadata["goal_reached"]  # Planned endpoint is not measured arrival.
    assert result.search_metadata["path_termination"] == "goal"
    assert all(field.segment_valid(left, right)
               for left, right in zip(candidate.positions[:-1], candidate.positions[1:]))
    json.dumps(result.to_dict(), allow_nan=False)


def test_unequal_constant_weights_have_straight_geodesic_and_diagonal_deltas():
    result, _ = plan(compact((.03, .04, 0.)))
    candidate = selected(result)
    assert candidate.path_length_m == pytest.approx(.05)
    assert len(candidate.primitives) == 5
    assert result.delta_xyz == pytest.approx((.006, .008, 0.))
    points = np.asarray(candidate.positions) - CENTER
    assert points[:, 0] * 4 == pytest.approx(points[:, 1] * 3)
    assert candidate.positions[-1] == pytest.approx((.03, .04, .5))


def test_far_diagonal_goal_has_at_most_ten_one_centimeter_segments():
    result, _ = plan(compact((.3, .3, 0.)))
    candidate = selected(result)
    assert len(candidate.primitives) == 10
    assert candidate.path_length_m == pytest.approx(.1)
    assert np.linalg.norm(np.diff(candidate.positions, axis=0), axis=1) == pytest.approx([.01] * 10)
    assert result.search_metadata["path_termination"] == "horizon"
    assert not result.search_metadata["boundary_clipped"]


def test_goal_outside_fixed_grid_clips_at_boundary_and_reports_refresh_condition():
    result, field = plan(compact((.3, 0., 0.)))
    candidate = selected(result)
    assert len(candidate.primitives) == 8
    assert candidate.path_length_m == pytest.approx(.08)
    assert candidate.positions[-1] == pytest.approx((.08, 0., .5))
    assert field.query(candidate.positions[-1]) is not None
    assert result.search_metadata["boundary_clipped"]
    assert not result.search_metadata["endpoint_is_goal"]
    assert result.reason == "local_boundary_clipped"


def test_workspace_boundary_can_shorten_first_segment():
    config = PlannerConfig(horizon=10, workspace_max=(.006, 1., 1.5))
    result, _ = plan(compact((.03, 0., 0.)), config=config)
    assert len(selected(result).primitives) == 1
    assert result.delta_xyz == pytest.approx((.006, 0., 0.))
    assert result.search_metadata["boundary_clipped"]


def test_outward_direction_at_boundary_holds_without_forced_lattice_detour():
    result, _ = plan(compact((.3, 0., 0.)), position=(.08, 0., .5))
    assert result.status == "direction_invalid"
    assert result.selected_index is None
    assert result.delta_xyz == (0., 0., 0.)
    assert result.search_metadata["boundary_clipped"]
    assert result.search_metadata["direction_invalid"]


@pytest.mark.parametrize("position", [(2., 0., .5), (.081, 0., .5)])
def test_invalid_measured_start_does_not_plan(position):
    result, _ = plan(position=position)
    assert result.reason == "invalid_start"
    assert result.delta_xyz == (0., 0., 0.)


def test_measured_goal_tolerance_returns_no_motion_or_artificial_loop():
    result, _ = plan(position=(.023, .001, .5))
    assert result.status == "goal_reached"
    assert result.search_metadata["goal_reached"]
    assert result.candidates == ()
    assert result.delta_xyz == (0., 0., 0.)


def test_explicit_model_hold_precedes_analytic_goal_motion():
    result, _ = plan(compact(motion_mode="hold"))
    assert result.status == "hold"
    assert result.reason == "explicit_motion_hold"
    assert result.delta_xyz == (0., 0., 0.)


@pytest.mark.parametrize("changes,reason", [
    ({"repulsors": [{"center_offset_m": [.02, 0., 0.], "radius_m": .01, "weight": 2.}]},
     "compact_repulsors_present"),
    ({"blocked_boxes": [{"lower_offset_m": [.3, .3, .3], "upper_offset_m": [.4, .4, .4]}]},
     "compact_boxes_present"),
    ({"unknown_boxes": [{"lower_offset_m": [.3, .3, .3], "upper_offset_m": [.4, .4, .4]}]},
     "compact_boxes_present"),
])
def test_nonpure_compact_predictions_preserve_legacy_selected_path(changes, reason):
    raw = compact(**changes)
    result, field = plan(raw, config=PlannerConfig(horizon=3))
    legacy = LocalPlanner(PlannerConfig(horizon=3)).plan(field, CENTER)
    assert result.search_metadata["analytic_fallback_reason"] == reason
    assert selected(result) == selected(legacy)
    assert result.delta_xyz == legacy.delta_xyz


@pytest.mark.parametrize("mask_name", ["blocked_indices", "unknown_indices"])
def test_dense_masks_force_fallback_and_cannot_be_bypassed_by_pure_metadata(mask_name):
    raw = compact()
    field = field_for(raw)
    # This masks an intermediate grid vertex of the +x 1 cm segment while
    # leaving both segment endpoints queryable. Restrict workspace to the x ray.
    masked = np.ravel_multi_index((17, 16, 16), (GRID.size,) * 3)
    field = GridField(GRID, CENTER, replace(field.prediction, **{mask_name: (int(masked),)}))
    assert field.query((.01, 0., .5)) is not None
    assert not field.segment_valid(CENTER, (.01, 0., .5))
    planner = LocalPlanner(PlannerConfig(horizon=3, workspace_min=(-.04, -.001, .499),
                                         workspace_max=(.04, .001, .501)))
    result = planner.plan_analytic(field, CENTER, goal_xyz=(.025, 0., .5), compact_prediction=raw)
    assert result.search_metadata["analytic_fallback_reason"] == "dense_masks_present"
    assert selected(result).primitives == ("-x",) * 3


def test_analytic_mode_checks_complete_segments_even_with_empty_predicted_masks(monkeypatch):
    calls = []

    def reject_middle_segment(self, start, end):
        calls.append((tuple(start), tuple(end)))
        return end[0] < .02

    monkeypatch.setattr(GridField, "segment_valid", reject_middle_segment)
    result, _ = plan()
    assert len(calls) == 2
    assert result.status == "direction_invalid"
    assert result.reason == "invalid_field_segment"
    assert result.delta_xyz == (0., 0., 0.)


@pytest.mark.parametrize("kwargs", [{"goal_tolerance_m": 0.}, {"goal_tolerance_m": float("nan")},
                                    {"goal_tolerance_m": True}])
def test_analytic_arrival_tolerance_is_validated(kwargs):
    with pytest.raises(ValueError):
        plan(**kwargs)


def test_nonfinite_explicit_goal_and_position_are_rejected():
    raw = compact()
    planner = LocalPlanner()
    with pytest.raises(ValueError):
        planner.plan_analytic(field_for(raw), CENTER, goal_xyz=(float("nan"), 0., .5),
                              compact_prediction=raw)
    with pytest.raises(ValueError):
        plan(position=(float("nan"), 0., .5))


def test_projection_at_end_is_not_measured_arrival():
    tracker = PathTracker(endpoint_tolerance_m=.005)
    tracker.reset((CENTER, (.04, 0., .5)))
    point = (.06, .02, .5)
    assert not tracker.endpoint_reached(point)
    delta, details = tracker.command(point, field_for(compact()), (-1., -1., 0.), (1., 1., 1.5))
    assert details["progress_m"] == pytest.approx(.04)
    assert not details["endpoint_reached"]
    assert details["status"] == "tracking"
    assert delta == pytest.approx((-.02, -.02, 0.))


def test_tracker_stops_at_measured_endpoint_tolerance_and_reset_clears_arrival():
    tracker = PathTracker(endpoint_tolerance_m=.005)
    tracker.reset((CENTER, (.023, 0., .5)))
    point = (.021, .001, .5)
    assert tracker.endpoint_reached(point)
    delta, details = tracker.command(point, field_for(compact()), (-1., -1., 0.), (1., 1., 1.5))
    assert delta == (0., 0., 0.)
    assert details["status"] == "path_complete"
    assert details["endpoint_distance_m"] == pytest.approx(np.sqrt(.002 ** 2 + .001 ** 2))
    tracker.reset()
    assert not tracker.endpoint_reached(point)
    assert tracker.endpoint_distance(point) is None


def test_endpoint_tolerance_does_not_override_invalid_measured_pose():
    tracker = PathTracker(endpoint_tolerance_m=.005)
    tracker.reset((CENTER, (.079, 0., .5)))
    delta, details = tracker.command((.081, 0., .5), field_for(compact()),
                                     (-1., -1., 0.), (1., 1., 1.5))
    assert delta == (0., 0., 0.)
    assert details["status"] == "invalid_measured_start"


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True])
def test_tracker_arrival_tolerance_is_validated(value):
    with pytest.raises(ValueError):
        PathTracker(endpoint_tolerance_m=value)
