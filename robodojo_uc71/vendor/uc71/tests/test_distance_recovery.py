"""The optional viewpoint recovery must remain its declared short +Z motion."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from test_distance_policy import controller, intent, tick
from test_field_history import observation


@pytest.mark.parametrize("drift", [(0., 0., 0.), (.0007, -.0013, .0021)])
def test_recovery_with_repulsor_and_fractional_grid_start_stays_vertical(drift):
    raw = intent(phase="verify", hold=True)
    raw["field"]["repulsors"] = [
        {"center_offset_m": [0., 0., .01], "radius_m": .008, "weight": 10.}]
    original = copy.deepcopy(raw)
    policy, _ = controller(raw)
    obs = observation()
    position = np.asarray(obs.state[:3]) + drift
    for step in range(5):
        obs, _ = tick(policy, obs, actual_position=position if step == 4 else None)
    assert policy._recovery_active
    before = np.asarray(obs.state[:3])
    obs, decision = tick(policy, obs)
    delta = np.asarray(obs.state[:3]) - before
    assert delta[:2] == pytest.approx([0., 0.], abs=1e-12)
    assert 0 < delta[2] <= .02 + 1e-12
    result = decision.diagnostics["plan"]
    assert result["search"]["method"] == "local_vertical_reobservation"
    assert result["search"]["soft_repulsors_ignored_for_reobservation"]
    points = np.asarray(result["candidates"][0]["positions"])
    differences = np.diff(points, axis=0)
    assert np.all(np.abs(differences[:, :2]) <= 1e-12)
    assert np.all((differences[:, 2] > 0) & (differences[:, 2] <= .01 + 1e-12))
    assert result["candidates"][0]["path_length_m"] <= .02 + 1e-12
    assert "distance_field_display" not in decision.diagnostics
    assert raw == original
    assert policy._field_cache.effective_compact_field["repulsors"] == original["field"]["repulsors"]
    assert policy.pending_field_event["reason"] == "placement_unverified"


@pytest.mark.parametrize("mask", ["blocked_boxes", "unknown_boxes"])
def test_reobservation_keeps_thin_hard_geometry_masks(mask):
    raw = intent(phase="verify", hold=True)
    raw["field"][mask] = [{"lower_offset_m": [-.01, -.01, .011],
                            "upper_offset_m": [.01, .01, .012]}]
    policy, _ = controller(raw)
    obs = observation()
    for _ in range(5):
        obs, decision = tick(policy, obs)
        assert decision.action[:3] == (0., 0., 0.)
    assert not policy._recovery_active
    assert policy.pending_field_event["reason"] == "placement_unverified"


def test_recovery_planner_does_not_descend_after_measured_height_overshoot():
    policy, _ = controller(intent(phase="verify", hold=True))
    obs = observation()
    for _ in range(5):
        obs, _ = tick(policy, obs)
    goal = np.asarray(policy.field.center) + policy._field_cache.effective_compact_field["goal_offset_m"]
    result = policy.planner.plan_analytic(policy.field, goal + [0., 0., .01], goal_xyz=goal,
        compact_prediction=policy._field_cache.effective_compact_field)
    assert result.delta_xyz == (0., 0., 0.)
    assert not result.candidates
    assert result.reason == "reobservation_height_reached"


def test_recovery_does_not_correct_lateral_drift_between_control_ticks():
    policy, _ = controller(intent(phase="verify", hold=True))
    obs = observation()
    for _ in range(5):
        obs, _ = tick(policy, obs)
    drifted = np.asarray(obs.state[:3]) + [.003, 0., .002]
    obs, _ = tick(policy, obs, actual_position=drifted)
    assert policy._recovery_active and policy.pending_field_event is None
    _, decision = tick(policy, obs)
    assert decision.action[:2] == (0., 0.)
    assert decision.action[2] > 0.
    assert decision.diagnostics["plan_refreshed"]


def test_fresh_model_field_cannot_inherit_previous_recovery_override():
    policy, transport = controller(intent(phase="verify", hold=True),
                                   intent(phase="transport", goal=(.03, 0., .03)))
    obs = observation()
    for _ in range(6):
        obs, _ = tick(policy, obs)
    assert policy.pending_field_event["reason"] == "placement_unverified"
    assert policy._recovery_active
    _, decision = tick(policy, obs, fraction=.1)
    assert len(transport.calls) == 2
    assert decision.diagnostics["refreshed"]
    assert decision.diagnostics["plan"]["search"]["method"] == "dijkstra_distance_field"
    assert decision.action[0] > 0.
    assert not decision.diagnostics["local_recovery_active"]
