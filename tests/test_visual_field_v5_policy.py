"""Task handoff stays model-directed; normal budgets are not failure evidence."""
import copy

import numpy as np

from gpt_field.visual_field_policy_v5 import VisualFieldPolicyV5
from test_visual_field_v4_policy import Client, observation, prediction


def test_normal_intent_end_is_not_reported_as_recovery():
    policy = VisualFieldPolicyV5(Client(prediction()))
    policy.prediction = prediction()
    policy._failure("intent_action_budget")
    assert policy.refresh_reason == "intent_action_budget"
    assert policy.recovery_episodes == 0
    assert policy.failed_hypotheses == []
    assert policy._observable_feedback()["controller_handoff"]["normal_field_refreshes"] == 1


def test_track_loss_advises_new_interaction_decision_but_never_closes_gripper():
    p = prediction()
    # Both references unavailable: no fabricated optical evidence or contact action.
    p["visual_references"] = {"external": None, "wrist": None}
    policy = VisualFieldPolicyV5(Client(p))
    obs = observation()
    d = policy.act(obs)
    assert d.action == (0.,)*7
    assert d.diagnostics["policy_version"] == "visual-field-policy-v5"
    assert d.diagnostics["selection"]["status"] == "external_reference_unavailable"
    feedback = d.diagnostics["feedback"]["controller_handoff"]
    assert not feedback["interaction_readiness_certified"]
    assert len(feedback["recent_reference_failures"]) == 1
    assert "Direct5" in feedback["decision_rule"]


def test_direct_contact_executes_five_model_actions_despite_missing_field_tracks():
    actions = [[0., 0., -.05, 0., 0., 0., 1.] for _ in range(5)]
    p = prediction(field=False, phase="grasp", actions=actions)
    p["visual_references"] = {"external": None, "wrist": None}
    client = Client(p)
    policy = VisualFieldPolicyV5(client)
    obs = observation()
    for action in actions:
        d = policy.act(obs)
        assert list(d.action) == action
        policy.observe_transition(obs, d, obs)
    assert client.calls == 1
    assert len(policy.attempts) == 1
    assert policy.field_nonzero_actions == 0
    assert not policy.monitor.attachment_evidence


def test_twenty_field_actions_still_require_real_response_and_feedback():
    p = prediction()
    p["field_intent"].update(max_actions=80, command_limit_m=.012)
    client = Client(p)
    policy = VisualFieldPolicyV5(client)
    offset = np.zeros(3)
    obs = observation(offset)
    for _ in range(20):
        d = policy.act(obs)
        assert d.diagnostics["selection"]["accepted"]
        offset += np.asarray(d.action[:3])*.05*.25
        after = observation(offset)
        policy.observe_transition(obs, d, after)
        obs = after
    assert client.calls == 1
    assert policy.field_effective_actions == 20
    assert policy.monitor.response_ready
    assert offset[0] > .03


def test_schema_repair_counts_actual_requests_separately_from_intents():
    from test_visual_field_v5_protocol import client as subscription_client, prediction as wire_prediction
    from test_visual_field_protocol import observation as wire_observation
    invalid = wire_prediction()
    invalid["visual_references"]["external"]["moving_roi_uv"] = [.8, .8, .9, .9]
    client, transport = subscription_client(invalid, wire_prediction())
    policy = VisualFieldPolicyV5(client)
    decision = policy.act(wire_observation())
    assert len(transport.calls) == 2
    assert policy.model_requests == 2
    assert policy.intent_refreshes == 1
    assert decision.diagnostics["feedback"]["efficiency"]["model_requests"] == 2


def test_wrong_clear_route_goal_refreshes_when_measured_image_error_grows():
    p = prediction()
    # Metric hypothesis asks right; image objective asks left. Actual tool
    # response is coherent, so this must be distinguished from tracking loss.
    p["visual_references"]["external"]["target_point_uv"] = [20/223, 90/223]
    p["field_intent"].update(max_actions=80)
    client = Client(p)
    policy = VisualFieldPolicyV5(client)
    offset = np.zeros(3)
    obs = observation(offset)
    for _ in range(25):
        d = policy.act(obs)
        offset += np.asarray(d.action[:3])*.05*.25
        after = observation(offset)
        policy.observe_transition(obs, d, after)
        obs = after
        if policy.refresh_reason:
            break
    assert policy.refresh_reason == "visual_goal_divergence"
    assert policy.monitor.response_ready
    assert client.calls == 1
    assert offset[0] < .03
    assert policy.attempts == []


def test_independent_direct_anchor_keeps_interaction_identity_and_observed_attachment_into_field():
    direct = prediction(field=False, payload=True, phase="grasp")
    direct["target_id"] = "object"
    direct["visual_references"]["wrist"] = {"target_id": "object",
        "target_anchor_uv": [.5, .5], "desired_target_uv": [.5, .5], "target_roi_uv": None}
    client = Client(direct)
    policy = VisualFieldPolicyV5(client)
    obs = observation(payload=True)
    for n in range(1, 6):
        decision = policy.act(obs)
        visual = decision.diagnostics["feedback"]["visual"]
        assert decision.diagnostics["target_id"] == "object"
        assert visual["interaction_target_id"] == "object"
        assert visual["external_stationary_anchor_id"] == "destination"
        assert not visual["image_goal_targets_interaction"]
        assert visual["external_goal_error_px"] is None
        assert visual["anchor_relative_projection_error_px"] is not None
        after = observation((.001*n, 0., 0.), payload=True)
        policy.observe_transition(obs, decision, after)
        obs = after
    assert policy.monitor.attachment_evidence
    attachment_samples = policy.monitor.attachment_samples

    # Only the interaction/navigation target changes. The same physically
    # observed payload/tool pair and stationary anchor keep their identities.
    transport = prediction(payload=True, phase="transport")
    transport["visual_references"]["external"].update(
        moving_point_uv=[55/223, 90/223], tool_point_uv=[35/223, 70/223])
    client.value = transport
    decision = policy.act(obs)
    assert decision.diagnostics["selection"]["accepted"]
    assert policy.monitor.attachment_evidence
    assert policy.monitor.attachment_samples == attachment_samples
    assert client.blocks[0]["target_id"] == "object"
    assert client.blocks[0]["end_visual_feedback"]["visual"]["interaction_target_id"] == "object"
    assert decision.diagnostics["target_id"] == "destination"
    assert decision.diagnostics["feedback"]["visual"]["image_goal_targets_interaction"]
    assert not decision.diagnostics["feedback"]["visual"]["clearance_certified"]


def test_unrelated_anchor_cannot_drive_planar_correction_or_goal_divergence():
    # Protocol validation rejects this Field role mismatch. The policy must
    # also remain correct for a custom client that bypasses that validator.
    p = prediction()
    p["visual_references"]["external"].update(target_id="stationary_background",
        target_point_uv=[20/223, 90/223])
    p["field_intent"]["max_actions"] = 80
    client = Client(p)
    policy = VisualFieldPolicyV5(client)
    offset = np.zeros(3)
    obs = observation(offset)
    for _ in range(20):
        decision = policy.act(obs)
        assert decision.diagnostics["selection"]["accepted"]
        offset += np.asarray(decision.action[:3])*.05*.25
        after = observation(offset)
        row = policy.observe_transition(obs, decision, after)
        assert not row["visual_goal_check"]["straight_route_check_supported"]
        obs = after
    assert policy.refresh_reason is None
    assert client.calls == 1 and policy.visual_corrections == 0
    assert policy.monitor.response_ready
    assert not policy._allows_planar_correction()


def test_independent_anchor_role_change_cannot_invent_payload_attachment():
    direct = prediction(field=False, payload=True, phase="grasp")
    direct["target_id"] = "object"
    client = Client(direct)
    policy = VisualFieldPolicyV5(client)
    obs = observation(payload=True)
    for n in range(1, 6):
        decision = policy.act(obs)
        # The measured tool moves, but the payload stays behind in the image.
        after = observation((.001*n, 0., 0.), payload=True, slip=True)
        policy.observe_transition(obs, decision, after)
        obs = after
    assert not policy.monitor.attachment_evidence
    transport = prediction(payload=True, phase="transport")
    transport["visual_references"]["external"]["tool_point_uv"] = [35/223, 70/223]
    client.value = transport
    decision = policy.act(obs)
    assert not decision.diagnostics["selection"]["accepted"]
    assert decision.diagnostics["selection"]["status"] == "payload_co_motion_required_use_direct_test"
    assert decision.action == (0.,)*7
