"""Risk field routing with real image response and measured pose feedback."""
import copy

import numpy as np
import pytest

from gpt_field.visual_field_policy_v9 import VisualFieldPolicyV9
from test_visual_field_v6_policy import Client, observation
from test_visual_field_v8_policy import value


def uncertain_neighbor(*, unknown=False, assessment="present"):
    pred = value(goal=(.07, 0., 0.))
    pred["motion_context"].update(obstacle_assessment=assessment,
        nearby_obstacles=[{"id": "neighbor", "external_roi_uv": [.7, .2, .9, .4]}])
    pred["field_intent"].update(tool_radius_m=.02, uncertainty_m=.02)
    pred["field_intent"]["unknown_regions" if unknown else "obstacles"] = [{
        "id": "neighbor", "lower_offset_m": [-.1, -.16, -.1],
        "upper_offset_m": [.1, -.10, .1], "uncertainty_m": .1}]
    return pred


def test_visible_uncertain_neighbor_preserved_and_field_reused_with_measured_rgb_response():
    pred = uncertain_neighbor()
    original = copy.deepcopy(pred)
    client = Client(pred)
    policy = VisualFieldPolicyV9(client)
    obs, offset = observation(), np.zeros(3)
    for _ in range(45):
        decision = policy.act(obs)
        assert decision.diagnostics["policy_version"] == "visual-field-policy-v9"
        if not decision.diagnostics["selection"]["accepted"]:
            assert decision.diagnostics["selection"]["status"] == "measured_local_goal_arrival"
            row = policy.observe_transition(obs, decision, obs)
            break
        assert decision.action[6] == 0.
        offset += np.asarray(decision.action[:3])*.05*.25
        after = observation(offset)
        row = policy.observe_transition(obs, decision, after)
        obs = after
        if policy.refresh_reason:
            break
    assert client.calls == 1 and policy.field_effective_actions > 15
    assert np.linalg.norm(offset) > .03
    assert policy.local.intent["obstacles"] == original["field_intent"]["obstacles"]
    assert client.value == original
    assert not row["task_success_certified"]
    assert "geometry_semantics" in row


@pytest.mark.parametrize("unknown,assessment", [(True, "present"), (False, "unobservable")])
def test_unknown_or_unobservable_geometry_cannot_be_implicitly_softened(unknown, assessment):
    pred = uncertain_neighbor(unknown=unknown, assessment=assessment)
    policy = VisualFieldPolicyV9(Client(pred)); obs = observation()
    decision = policy.act(obs)
    assert not decision.diagnostics["selection"]["accepted"]
    assert decision.diagnostics["selection"]["status"] == "invalid_measured_start"
    assert np.linalg.norm(decision.action[:3]) == 0.
    assert decision.diagnostics["rejection_hold"]["requested_motion_executed"] is False
    policy.observe_transition(obs, decision, obs)
    if assessment == "unobservable":
        assert policy.local.intent["max_travel_m"] <= .015
        assert policy.local.intent["command_limit_m"] <= .005
    assert policy.local.intent["unknown_regions"] == pred["field_intent"]["unknown_regions"]


def test_new_request_does_not_inherit_previous_visible_scene_permission():
    client = Client(uncertain_neighbor()); policy = VisualFieldPolicyV9(client); obs = observation()
    first = policy.act(obs)
    assert first.diagnostics["selection"]["accepted"]
    policy.observe_transition(obs, first, obs)
    policy.refresh_reason = "new_scene_review"
    client.value = uncertain_neighbor(assessment="unobservable")
    second = policy.act(obs)
    assert not second.diagnostics["selection"]["accepted"]
    assert client.calls == 2
