"""Whole-policy continuation retains Field reuse and explicit recovery dispatch."""
import numpy as np

from gpt_field.visual_field_policy_v25 import VisualFieldPolicyV25
from gpt_field.visual_field_policy_v26 import VisualFieldPolicyV26
from gpt_field.visual_translation_guard_v26 import ProjectedTranslationGuardV26
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v23_policy import archive_initial, OfferClient
from test_visual_field_v25_policy import explicit


def test_no_representation_revision_preserves_commands_and_single_request_reuse():
    old = VisualFieldPolicyV25(Client(explicit(prediction())))
    new = VisualFieldPolicyV26(Client(explicit(prediction())))
    a, b = observation(), observation()
    for _ in range(12):
        a, da, _ = fixed_rgb_transition(old, a)
        b, db, _ = fixed_rgb_transition(new, b)
        assert db.action == da.action
        assert db.diagnostics['selection']['accepted']
        assert db.diagnostics['policy_version'] == 'visual-field-policy-v26'
    assert old.client.calls == new.client.calls == 1
    assert new.local.pending is new.pending is None
    assert isinstance(new.translation_guard, ProjectedTranslationGuardV26)
    assert new.local.translation_guard is new.translation_guard


def test_current_egress_offer_uses_same_guard_in_preview_and_actual_selection():
    obs, value = archive_initial()

    class ExplicitOfferClient(OfferClient):
        def predict(self, record):
            value = super().predict(record)
            value['recovery_route'] = ('normal' if self.calls == 1
                                       else self.offer['required_recovery_route'])
            return value

    client = ExplicitOfferClient(value)
    policy = VisualFieldPolicyV26(client)
    first = policy.act(obs)
    assert first.diagnostics['selection']['status'] == 'invalid_measured_start'
    policy.observe_transition(obs, first, obs)
    second = policy.act(obs)
    assert second.diagnostics['selection']['accepted'], second.diagnostics['selection']
    assert client.offer['required_recovery_route'] == 'initial_uncertainty_egress'
    assert policy.local.recovery_route_binding['recovery_route'] == 'initial_uncertainty_egress'
    assert np.allclose(np.asarray(second.action[:3]) * .05,
                       client.offer['preview_first_command_delta_m'])
    assert policy.translation_guard.egress_evidence.spent_m == 0
    assert client.calls == 2


def test_missing_route_is_still_rejected_without_motion_or_goal():
    policy = VisualFieldPolicyV26(Client(prediction()))
    obs = observation()
    decision = policy.act(obs)
    assert not decision.diagnostics['selection']['accepted']
    assert decision.diagnostics['selection']['status'] == 'subgoal_recovery_route_invalid'
    assert np.linalg.norm(decision.action[:3]) == 0
    assert policy.subgoals.active is None
    policy.observe_transition(obs, decision, obs)
    assert policy.local.pending is policy.pending is None
