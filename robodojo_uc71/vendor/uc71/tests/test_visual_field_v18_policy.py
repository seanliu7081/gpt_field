"""Candidate review is an observation handoff, not motion or success evidence."""
import copy

from gpt_field.visual_field_policy_v18 import VisualFieldPolicyV18
from test_visual_field_v6_policy import Client
from test_visual_field_v17_policy import prediction
from test_visual_field_v13_execution import observation, transition


def test_real_clear_field_reuses_one_intent_with_candidate_observer_enabled():
    value = prediction()
    client = Client(value)
    policy = VisualFieldPolicyV18(client)
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
    assert decision.diagnostics["policy_version"] == "visual-field-policy-v18"
    assert not row["reference_repair"]["review_required"]
    assert not row["reference_replay"]["accepted"]
    assert not row["task_success_certified"]
    assert row["scene_inventory"]["entries"]


def test_early_review_handoff_preserves_geometry_and_physical_accounting():
    client = Client(prediction())
    policy = VisualFieldPolicyV18(client)
    obs = observation()
    # Integration seam: the bank's actual image evidence is covered separately;
    # here its review must reach the next model input without changing controls.
    obs, decision, row = transition(policy, obs)
    geometry = copy.deepcopy(policy.inventory.summary())
    debt = policy.translation_guard.observation_travel_m
    experiments = copy.deepcopy(policy.translation_guard.experiments)
    report = {"review_required": True, "review_recommended": True,
              "review_id": "one-observed-motion-episode", "candidates": []}
    bank = policy.reference_candidates
    bank.observe = lambda *args: copy.deepcopy(report)
    bank.summary = lambda: copy.deepcopy(report)
    obs, decision, row = transition(policy, obs)
    assert policy.refresh_reason == "causal_controlled_reference_review"
    assert row["reference_repair"]["review_id"] == report["review_id"]
    assert policy.translation_guard.observation_travel_m >= debt
    assert len(policy.translation_guard.experiments) >= len(experiments)
    assert [x["id"] for x in policy.inventory.summary()["entries"]] == [
        x["id"] for x in geometry["entries"]]
    assert not row["task_success_certified"]
    assert row["visual"]["reference_health"]["candidate_review_required"]
    # The same static ticket must not create another automatic early refresh.
    policy.refresh_reason = None
    obs, decision, row = transition(policy, obs)
    assert policy.refresh_reason != "causal_controlled_reference_review"
    assert client.calls == 1
    assert policy.block[-1]["reference_repair"] == row["reference_repair"]
