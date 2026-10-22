"""Real V19 Field reuse retains contact, inventory and causal reference checks."""
import copy

from gpt_field.visual_field_policy_v19 import VisualFieldPolicyV19
from test_visual_field_v6_policy import Client
from test_visual_field_v17_policy import prediction
from test_visual_field_v13_execution import observation, transition


def test_real_field_reuse_keeps_feedback_and_near_contact_correction():
    value = prediction()
    client = Client(value)
    policy = VisualFieldPolicyV19(client)
    obs = observation()
    accepted = 0
    for _ in range(30):
        obs, decision, row = transition(policy, obs)
        accepted += bool(decision.diagnostics["selection"]["accepted"])
        assert policy.pending is None and policy.local.pending is None
        assert policy.prediction == value and client.value == value
        if policy.refresh_reason:
            break
    assert accepted > 10 and client.calls == 1
    assert policy.visual_corrections > 0
    assert decision.diagnostics["policy_version"] == "visual-field-policy-v19"
    assert row["scene_inventory"]["entries"]
    assert not row["task_success_certified"]
    assert not row["reference_replay"]["accepted"]


def test_reference_review_still_interrupts_without_changing_geometry_or_debt():
    client = Client(prediction())
    policy = VisualFieldPolicyV19(client)
    obs = observation()
    obs, decision, row = transition(policy, obs)
    geometry = copy.deepcopy(policy.inventory.summary())
    debt = policy.translation_guard.observation_travel_m
    report = {"review_required": True, "review_recommended": True,
              "review_id": "actual-reference-review", "candidates": []}
    policy.reference_candidates.observe = lambda *args: copy.deepcopy(report)
    policy.reference_candidates.summary = lambda: copy.deepcopy(report)
    obs, decision, row = transition(policy, obs)
    assert policy.refresh_reason == "causal_controlled_reference_review"
    assert row["visual"]["reference_health"]["candidate_review_required"]
    assert policy.translation_guard.observation_travel_m >= debt
    assert [x["id"] for x in policy.inventory.summary()["entries"]] == [
        x["id"] for x in geometry["entries"]]
    assert not row["task_success_certified"]


def test_unknown_neighbor_still_prevents_closure():
    from test_visual_field_v13_payload_policy import prediction as grip_prediction
    from test_visual_field_v6_policy import observation as grip_observation
    value = grip_prediction()
    geometry = copy.deepcopy(value['target_geometry'])
    permission = geometry.pop('allowed_contact')
    identity = geometry.pop('id')
    value.update(target_contact=permission, scene_inventory=[
        {'id': identity, 'external_roi_uv': [.68, .61, .8, .74],
         'wrist_roi_uv': None, 'geometry': geometry},
        {'id': 'unlocalized', 'external_roi_uv': [.24, .28, .31, .38],
         'wrist_roi_uv': None, 'geometry': None}], raw_wire_prediction={'offline_fixture': True})
    policy = VisualFieldPolicyV19(Client(value))
    decision = policy.act(grip_observation())
    assert not decision.diagnostics['selection']['accepted']
    assert decision.action[6] == 0.
    assert decision.diagnostics['selection']['status'] == 'scene_visible_region_needs_metric_localization'
    assert policy.gripper.pending is None
