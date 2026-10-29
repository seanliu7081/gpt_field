"""Persistent route identity and whole-policy dispatch, without model calls."""
import copy

import numpy as np

from gpt_field.visual_field_policy_v24 import VisualFieldPolicyV24
from gpt_field.visual_field_policy_v25 import VisualFieldPolicyV25, RecoverySubgoalLedgerV25
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v23_policy import archive_initial, OfferClient


def explicit(value, family='normal'):
    value = copy.deepcopy(value)
    value['recovery_route'] = family
    return value


def test_normal_policy_preserves_actual_commands_and_one_request_reuse():
    old = VisualFieldPolicyV24(Client(prediction()))
    new = VisualFieldPolicyV25(Client(explicit(prediction())))
    a, b = observation(), observation()
    for _ in range(12):
        a, da, _ = fixed_rgb_transition(old, a)
        b, db, _ = fixed_rgb_transition(new, b)
        assert db.action == da.action
        assert db.diagnostics['selection']['accepted']
        assert db.diagnostics['recovery_route_binding']['recovery_route'] == 'normal'
    assert old.client.calls == new.client.calls == 1
    assert new.local.pending is new.pending is None


def test_route_change_needs_explicit_replace_and_direct_preserves_it():
    state = observation().state
    ledger = RecoverySubgoalLedgerV25()
    value = explicit(prediction(), 'reverse_observed')
    assert ledger.apply(state, value)['accepted']
    original = copy.deepcopy(ledger.active)
    value['subgoal_update']['action'] = 'retain'
    value['recovery_route'] = 'initial_uncertainty_egress'
    result = ledger.apply(state, value)
    assert not result['accepted']
    assert result['status'] == 'recovery_route_change_requires_explicit_replace'
    assert ledger.active == original
    direct = copy.deepcopy(value)
    direct.update(controller='direct', recovery_route='normal', field_intent=None)
    assert ledger.apply(state, direct)['accepted']
    assert ledger.active['recovery_route'] == 'reverse_observed'
    value['subgoal_update'].update(action='replace', reason='Explicitly select a different recovery family')
    assert ledger.apply(state, value)['accepted']
    assert ledger.active['recovery_route'] == 'initial_uncertainty_egress'


def test_missing_explicit_route_cannot_execute_or_create_a_goal():
    policy = VisualFieldPolicyV25(Client(prediction()))
    obs = observation()
    decision = policy.act(obs)
    assert not decision.diagnostics['selection']['accepted']
    assert decision.diagnostics['selection']['status'] == 'subgoal_recovery_route_invalid'
    assert np.linalg.norm(decision.action[:3]) == 0
    assert policy.subgoals.active is None
    policy.observe_transition(obs, decision, obs)
    assert policy.local.pending is policy.pending is None


def test_current_egress_offer_binds_same_family_in_preview_and_actual_policy():
    obs, value = archive_initial()

    class ExplicitOfferClient(OfferClient):
        def predict(self, record):
            value = super().predict(record)
            value['recovery_route'] = ('normal' if self.calls == 1
                                       else self.offer['required_recovery_route'])
            return value

    client = ExplicitOfferClient(value)
    policy = VisualFieldPolicyV25(client)
    first = policy.act(obs)
    assert first.diagnostics['selection']['status'] == 'invalid_measured_start'
    policy.observe_transition(obs, first, obs)
    second = policy.act(obs)
    assert second.diagnostics['selection']['accepted'], second.diagnostics['selection']
    assert client.offer['required_recovery_route'] == 'initial_uncertainty_egress'
    assert policy.local.recovery_route_binding['recovery_route'] == 'initial_uncertainty_egress'
    assert np.allclose(np.asarray(second.action[:3])*.05,
                       client.offer['preview_first_command_delta_m'])
    assert policy.translation_guard.egress_evidence.spent_m == 0
    assert client.calls == 2
