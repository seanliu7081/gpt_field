"""Prospective executor graphs must never traverse live policy transports.

The client owns an actual non-pickleable lock. Tests use real configured Field
executors, image tracks and admission guards; no model or simulator is called.
"""
import threading

import numpy as np
import pytest

from gpt_field.visual_field_policy_v32 import VisualFieldPolicyV32
from gpt_field.visual_field_proposals_v31 import FieldProposalBankV31
from gpt_field.visual_field_v4 import _CheckedField
from gpt_field.visual_field_v23 import _DirectedEgressValidity
from gpt_field.visual_field_v24 import _CoreSeparatingValidity
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v25_policy import explicit
from test_visual_field_v31_policy import snapshot


@pytest.fixture
def configured():
    value = explicit(prediction())
    value.update(scene_revisions=[], field_proposal_id=None)
    value['visual_references']['external']['reference_role'] = 'tool_anatomy'
    client = Client(value)
    client.transport_lock = threading.Lock()
    policy = VisualFieldPolicyV32(client)
    obs, decision, _ = fixed_rgb_transition(policy, observation())
    assert decision.diagnostics['selection']['accepted']
    try:
        yield policy, obs
    finally:
        policy.finish()


@pytest.mark.parametrize('wrapper', [_CheckedField, _DirectedEgressValidity, _CoreSeparatingValidity])
def test_every_validity_owner_cycle_resolves_to_the_single_prospective_executor(configured, wrapper):
    policy, obs = configured
    policy.local.checked = wrapper(policy.local)
    before = snapshot(policy)
    local, guard = FieldProposalBankV31._clone(policy)
    assert type(local.checked) is wrapper
    assert local.checked is not policy.local.checked
    assert local.checked.owner is local
    assert policy.local.checked.owner is policy.local
    assert local.translation_guard is guard and guard is not policy.translation_guard
    assert local.visual_monitor is not policy.monitor
    assert local.visual_monitor.tracks is policy.monitor.tracks
    assert local.robot_response is policy.robot_response
    assert 'scene_assessor' not in local.__dict__
    assert snapshot(policy) == before
    assert local.intent is not policy.local.intent
    assert not np.shares_memory(local.expected_state, policy.local.expected_state)
    local.expected_state[0] += .01
    local.intent['max_actions'] = 10
    guard.observation_travel_m += .001
    assert snapshot(policy) == before


def test_complete_proposal_preview_with_live_transport_lock_keeps_pending_and_debts_isolated(configured):
    policy, obs = configured
    before = snapshot(policy)
    tracks = {name: id(track) for name, track in policy.monitor.tracks.items()}
    report = policy.field_proposals.build(policy, obs,
        {'candidates': [{'goal_offset_m': [.01, 0., 0.]}]})
    offers = [row for row in report['offers'] if row['source'] == 'visual_goal_field']
    assert offers and offers[0]['preview_status'] == 'metric_field_tracking'
    assert np.linalg.norm(offers[0]['preview_action'][:3]) > 0
    assert policy.local.pending is None and policy.pending is None
    assert not offers[0]['physical_debt_replenished']
    assert not offers[0]['clearance_certified']
    assert tracks == {name: id(track) for name, track in policy.monitor.tracks.items()}
    assert snapshot(policy) == before


def test_cloned_actual_selection_acknowledges_only_its_own_pending_state(configured, monkeypatch):
    policy, obs = configured
    before = snapshot(policy)
    original_clone = FieldProposalBankV31._clone
    copied = []
    def capture(source):
        cloned = original_clone(source)
        copied.append(cloned)
        return cloned
    monkeypatch.setattr(FieldProposalBankV31, '_clone', staticmethod(capture))
    offer, reason = policy.field_proposals._preview(policy, obs, [.01, 0., 0.])
    assert reason is None and offer['preview_status'] == 'metric_field_tracking'
    local, guard = copied[-1]
    assert local.pending is not None and local.checked.owner is local
    assert policy.local.pending is None
    assert snapshot(policy) == before
    action = np.asarray(offer['preview_action'])
    after = np.asarray(obs.state).copy()
    after[:3] += action[:3]*.05*.5  # Deterministic measured-response plant.
    prior = local.actions
    local.observe(obs.state, action, after)
    assert local.pending is None and local.actions == prior+1
    assert snapshot(policy) == before


def test_failed_deepcopy_leaves_original_graph_and_physical_ledger_intact(configured):
    policy, obs = configured
    class BrokenCopy:
        def __deepcopy__(self, memo):
            raise RuntimeError('deliberate local planner copy failure')
    policy.local._test_copy_failure = BrokenCopy()
    before = snapshot(policy)
    checked = policy.local.checked
    with pytest.raises(RuntimeError, match='deliberate local planner'):
        FieldProposalBankV31._clone(policy)
    assert policy.local.checked is checked and checked.owner is policy.local
    assert snapshot(policy) == before

