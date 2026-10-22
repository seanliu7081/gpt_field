"""Regression cases from the first real v4 rollout and budget lifetime checks."""
import copy

import numpy as np
import pytest

from gpt_field.visual_field_policy import images
from gpt_field.visual_monitor_v4 import RoleAwareVisualMonitor, _orientation_distance
from test_visual_field_v4_policy import ORIGIN, observation, prediction


# Archived real feedback: representation changed by ~2*pi but the physical
# orientation changed by only 0.00063 rad. No simulator state is needed here.
BEFORE_PI = [-3.140048975110972, -.0026582558059387424, .08994692559434596]
AFTER_PI = [3.140143483350895, .001949066782532726, -.09021239477107826]


def oriented_state(obs, rotation):
    state = np.asarray(obs.state).copy()
    state[3:6] = rotation
    return state


@pytest.mark.parametrize("before,after,expected", [
    (BEFORE_PI, AFTER_PI, .0006308379104827055),
    ([3.140363470837365, .00197582327399729, -.0868419064494488],
     [-3.1403854610810753, -.002037647376938629, .0867944491478418],
     .00006112872610331703),
    ([np.pi, 0., 0.], [-np.pi, 0., 0.], 0.),
    ([np.pi, 0., 0.], [0., np.pi, 0.], np.pi),
    ([0., 0., 0.], [.08, 0., 0.], .08),
])
def test_orientation_distance_uses_so3_not_rotation_vector_subtraction(before, after, expected):
    assert _orientation_distance(before, after) == pytest.approx(expected, abs=1e-10)
    assert _orientation_distance(after, before) == pytest.approx(expected, abs=1e-10)


def test_near_pi_observation_preserves_response_attachment_and_path():
    monitor = RoleAwareVisualMonitor()
    refs = prediction(payload=True)["visual_references"]
    obs = observation(payload=True)
    monitor.configure(images(obs), refs, oriented_state(obs, BEFORE_PI), intent_id=1)
    for n in range(1, 7):
        obs = observation((.001*n, 0., 0.), payload=True)
        rotation = AFTER_PI if n == 6 else BEFORE_PI
        monitor.observe(images(obs), oriented_state(obs, rotation), [.1, 0., 0., 0., 0., 0., 0.])
    assert monitor.response_ready
    assert monitor.response.summary()["sample_count"] == 6
    assert monitor.attachment_evidence
    assert monitor.path_m == pytest.approx(.006)
    assert monitor.intent_path_m == pytest.approx(.006)
    assert monitor.validation_count >= 2

    # Configure applies the same invariant check to both attachment and response
    # continuity, and must retain them across another representation change.
    refs = copy.deepcopy(refs)
    refs["external"].update(moving_point_uv=[56/223, 90/223], tool_point_uv=[36/223, 70/223])
    response, epoch, uncertain_path = monitor.response, monitor.epoch, monitor.unvalidated_path_m
    monitor.configure(images(obs), refs, oriented_state(obs, BEFORE_PI), intent_id=1)
    assert monitor.response is response and monitor.epoch == epoch
    assert monitor.attachment_evidence and monitor.response_ready
    assert monitor.unvalidated_path_m == uncertain_path


def test_near_pi_branch_does_not_disable_planar_visual_correction():
    monitor = RoleAwareVisualMonitor()
    refs = prediction()["visual_references"]
    refs["external"]["target_point_uv"] = [64/223, 96/223]
    obs = observation()
    monitor.configure(images(obs), refs, oriented_state(obs, BEFORE_PI), intent_id=1)
    xy = np.zeros(2)
    for n in range(12):
        xy[n % 2] += .001
        obs = observation((*xy, 0.))
        rotation = AFTER_PI if n == 11 else BEFORE_PI
        monitor.observe(images(obs), oriented_state(obs, rotation), [.1, .1, 0., 0., 0., 0., 0.])
    assert monitor.response_ready and monitor.response.rank == 2
    goal = ORIGIN+[.02, .01, 0.]
    correction = monitor.planar_correction(goal)
    assert correction is not None
    assert 0 < np.linalg.norm(correction-goal) <= .00300001


@pytest.mark.parametrize("change", ["orientation", "fingers"])
def test_repeated_domain_resets_do_not_replenish_same_intent_uncertainty_budget(change):
    monitor = RoleAwareVisualMonitor()
    refs = prediction()["visual_references"]
    obs = observation()
    monitor.configure(images(obs), refs, obs.state, intent_id="request-1")
    for n in range(1, 9):
        obs = observation((.002*n, 0., 0.))
        state = np.asarray(obs.state).copy()
        if change == "orientation":
            state[3] = .08*(n % 2)
        else:
            state[6:8] += .003*(n % 2)
        monitor.observe(images(obs), state, [.1, 0., 0., 0., 0., 0., 0.])
        assert not monitor.response_ready
        assert monitor.path_m == 0.  # Mapping-local evidence still resets.
        assert monitor.unvalidated_path_m == pytest.approx(.002*n)
    assert monitor.control_limit() == (0., "visual_response_budget_exhausted")
    assert monitor.summary()["intent_observed_path_m"] == pytest.approx(.016)

    # A changed reference in the same intent also cannot renew that budget.
    refs = copy.deepcopy(refs)
    refs["external"].update(moving_id="reseeded_tool", moving_point_uv=[66/223, 90/223])
    monitor.configure(images(obs), refs, state, intent_id="request-1")
    assert monitor.control_limit() == (0., "visual_response_budget_exhausted")
    # Only the policy's explicit new request sequence authorizes fresh bootstrap.
    monitor.configure(images(obs), refs, state, intent_id="request-2")
    assert monitor.control_limit() == (.005, None)
    assert monitor.unvalidated_path_m == monitor.intent_path_m == 0.


def test_validated_motion_does_not_consume_additional_uncertainty_budget():
    monitor = RoleAwareVisualMonitor()
    obs = observation(payload=True)
    monitor.configure(images(obs), prediction(payload=True)["visual_references"], obs.state, intent_id=1)
    for n in range(1, 7):
        obs = observation((.001*n, 0., 0.), payload=True)
        monitor.observe(images(obs), obs.state, [.1, 0., 0., 0., 0., 0., 0.])
    assert monitor.response_ready and monitor.attachment_evidence
    uncertain_path = monitor.unvalidated_path_m
    for n in range(7, 15):
        obs = observation((.001*n, 0., 0.), payload=True)
        monitor.observe(images(obs), obs.state, [.1, 0., 0., 0., 0., 0., 0.])
    assert monitor.unvalidated_path_m == uncertain_path
    assert monitor.intent_path_m == pytest.approx(.014)
    assert monitor.control_limit() == (None, None)

    # A genuine reorientation still invalidates the learned mapping and grasp
    # evidence, and its unvalidated translation is charged to the same intent.
    obs = observation((.015, 0., 0.), payload=True)
    monitor.observe(images(obs), oriented_state(obs, [.08, 0., 0.]), [.1, 0., 0., 0., 0., 0., 0.])
    assert not monitor.response_ready and not monitor.attachment_evidence
    assert monitor.unvalidated_path_m == pytest.approx(uncertain_path+.001)


def test_omitted_intent_id_does_not_silently_reset_budget():
    monitor = RoleAwareVisualMonitor()
    obs = observation()
    refs = prediction()["visual_references"]
    monitor.configure(images(obs), refs, obs.state)
    obs = observation((.004, 0., 0.))
    monitor.observe(images(obs), obs.state, [.1, 0., 0., 0., 0., 0., 0.])
    refs = copy.deepcopy(refs)
    refs["external"].update(moving_id="new_correspondence", moving_point_uv=[54/223, 90/223])
    monitor.configure(images(obs), refs, obs.state)
    assert monitor.unvalidated_path_m == pytest.approx(.004)
