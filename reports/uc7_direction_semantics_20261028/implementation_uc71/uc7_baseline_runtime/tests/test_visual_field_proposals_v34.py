"""Explicit phase-transition offers; real local preview, no model/simulator."""
import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

from gpt_field.visual_field_policy_v33 import VisualFieldPolicyV33
from gpt_field.visual_field_proposals_v34 import FieldProposalBankV34
from gpt_field.visual_objective_v32 import FieldProposalBankV32
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction, fixed_rgb_transition
from test_visual_field_v25_policy import explicit
from test_visual_field_v31_policy import snapshot


@pytest.fixture
def configured(monkeypatch):
    value = explicit(prediction())
    value.update(scene_revisions=[], field_proposal_id=None)
    value['visual_references']['external']['reference_role'] = 'tool_anatomy'
    policy = VisualFieldPolicyV33(Client(value))
    obs, decision, _ = fixed_rgb_transition(policy, observation())
    assert decision.diagnostics['selection']['accepted']
    policy.prediction['phase'] = 'recover'
    policy.recovery_options = {'candidates': []}
    report = {'accepted': True, 'reason': 'visual_objective_ranked_fixed_goal_proposals',
              'candidates': [{'goal_offset_m': [.020, 0., 0.], 'predicted_error_reduction_px': 3.},
                             {'goal_offset_m': [.010, 0., 0.], 'predicted_error_reduction_px': 2.}]}
    calls = []
    def current(state, **kwargs):
        calls.append((tuple(state), copy.deepcopy(kwargs)))
        return copy.deepcopy(report)
    monkeypatch.setattr(policy.visual_goal_field, 'propose', current)
    try:
        yield policy, obs, report, calls
    finally:
        policy.finish()


def test_recovery_offers_twenty_mm_approach_under_actual_preview_without_live_mutation(configured):
    policy, obs, visual, calls = configured
    old = FieldProposalBankV32().build(policy, obs, visual)
    assert old['offers'] == []
    before, prediction_before = snapshot(policy), copy.deepcopy(policy.prediction)
    goal_before = policy.local.goal.copy()
    bank = FieldProposalBankV34()
    report = bank.build(policy, obs, visual)
    assert report['offers']
    first = report['offers'][0]
    assert first['source'] == 'visual_goal_field' and first['phase'] == 'approach'
    assert first['motion_context_overrides']['translation_mode'] == 'transit'
    assert first['motion_context_overrides']['intended_contact_id'] is None
    assert first['field_intent']['goal_offset_m'] == [.020, 0., 0.]
    assert first['field_intent']['uncertainty_m'] == policy.local.intent['uncertainty_m']
    assert first['subgoal_update']['action'] == 'replace'
    assert first['subgoal_update']['subgoal_id'] == policy.subgoals.active['subgoal_id']
    assert np.linalg.norm(first['preview_action'][:3]) > 0
    assert first['actual_selection_rechecks_all_constraints']
    assert not first['physical_debt_replenished'] and not first['clearance_certified']
    assert policy.prediction == prediction_before and snapshot(policy) == before
    assert np.array_equal(policy.local.goal, goal_before)
    assert policy.pending is policy.local.pending is None and policy.client.calls == 1
    assert calls[0][0] == tuple(obs.state)
    assert calls[0][1]['joint_position'] == obs.joint_position
    assert report['phase_transition']['live_phase_changed'] is False
    assert bank.resolve(first['proposal_id'], obs.to_dict()) == first
    changed = list(obs.state); changed[0] += .0001
    with pytest.raises(ValueError, match='observation changed'):
        bank.resolve(first['proposal_id'], replace(obs, state=tuple(changed)))


def test_visual_descent_precedes_short_normal_recovery_but_both_use_correct_phase(configured):
    policy, obs, visual, _ = configured
    policy.recovery_options = {'candidates': [{'direction_xyz': [1., 0., 0.],
        'recommended_goal_offset_m': [.008, 0., 0.], 'required_recovery_route': 'normal'}]}
    report = FieldProposalBankV34().build(policy, obs, visual)
    assert [row['source'] for row in report['offers']] == [
        'visual_goal_field', 'visual_goal_field', 'checked_normal_recovery']
    assert [row['phase'] for row in report['offers']] == ['approach', 'approach', 'recover']
    assert report['offers'][-1]['motion_context_overrides']['translation_mode'] == 'separate'


@pytest.mark.parametrize('case', ['proxy', 'payload', 'contact', 'gripper', 'pending', 'unqualified', 'transport', 'release'])
def test_new_transition_never_bypasses_existing_role_contact_pending_or_stage_scope(configured, case):
    policy, obs, visual, calls = configured
    if case == 'proxy': policy.prediction['visual_references']['external']['reference_role'] = 'rigid_transit_proxy'
    elif case == 'payload': policy.prediction['visual_references']['external']['moving_kind'] = 'payload'
    elif case == 'contact': policy.prediction['target_contact'] = 'fingers'
    elif case == 'gripper': policy.prediction['controller'] = 'gripper'
    elif case == 'pending': policy.local.pending = {'not_acknowledged': True}
    elif case == 'unqualified': visual['accepted'] = False
    else: policy.prediction['phase'] = case
    report = FieldProposalBankV34().build(policy, obs, visual)
    assert not report['offers']
    assert report['phase_transition'] is None and calls == []
    if case == 'proxy':
        assert report['reason'] == 'proxy_motion_support_is_not_anatomical_goal_support'


def test_stale_or_forged_visual_report_is_not_preview_authority(configured):
    policy, obs, visual, calls = configured
    stale = copy.deepcopy(visual)
    for row in stale['candidates']:
        row['goal_offset_m'][0] += .001
    before = snapshot(policy)
    report = FieldProposalBankV34().build(policy, obs, stale)
    assert report['offers'] == [] and len(calls) == 1
    assert all(row['reason'] == 'proposal_visual_goal_not_currently_qualified' for row in report['rejected'])
    assert snapshot(policy) == before


def test_qualified_visual_goal_still_fails_full_unknown_region_geometry(configured):
    policy, obs, visual, _ = configured
    policy.local.intent['unknown_regions'] = [{'id': 'unseen-space',
        'lower_offset_m': [-.04, -.04, -.04], 'upper_offset_m': [.04, .04, .04],
        'uncertainty_m': .01}]
    before = snapshot(policy)
    report = FieldProposalBankV34().build(policy, obs, visual)
    assert report['offers'] == []
    assert all(row['reason'] == 'proposal_full_straight_segment_blocked' for row in report['rejected'])
    assert snapshot(policy) == before


def test_existing_approach_behavior_and_constraints_remain_exact(configured):
    policy, obs, visual, calls = configured
    policy.prediction['phase'] = 'approach'
    old = FieldProposalBankV32().build(policy, obs, visual)
    new = FieldProposalBankV34().build(policy, obs, visual)
    assert new['offers'] == old['offers'] and new['rejected'] == old['rejected']
    assert new['phase_transition'] is None and calls == []


def test_actual_archived_step62_twenty_mm_route_passes_preview_without_execution():
    directory = Path(__file__).resolve().parents[1] / 'runs/visual_field_v34_design_001'
    checkpoint = directory / 'policy_before62_checkpoint.pkl'
    if not checkpoint.exists():
        pytest.skip('Optional locally reconstructed strict62-transition checkpoint unavailable')
    # Register only the archived fake client's class; do not run its replay or
    # the simulator. This checkpoint was produced from our own saved trace.
    source = directory.parent / 'visual_field_v30_design_001/archived_approach_inputs.py'
    spec = importlib.util.spec_from_file_location('v34_archived_inputs', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with checkpoint.open('rb') as stream:
        policy, obs, visual = pickle.load(stream)
    assert policy.prediction['phase'] == 'recover' and visual['accepted']
    assert len(visual['candidates']) == 4
    before, old_prediction = snapshot(policy), copy.deepcopy(policy.prediction)
    previous_calls = policy.client.calls
    result = FieldProposalBankV34().build(policy, obs, visual)
    assert [row['phase'] for row in result['offers']] == ['approach'] * 3
    assert [row['preview_status'] for row in result['offers']] == ['metric_field_tracking'] * 3
    lengths = [np.linalg.norm(row['field_intent']['goal_offset_m']) for row in result['offers']]
    assert lengths == pytest.approx([.020212808704250945, .020, .010])
    assert result['rejected'] == []
    assert all(row['full_route_geometry_sha256'] for row in result['offers'])
    assert all(row['field_intent']['max_travel_m'] == .015 for row in result['offers'])
    assert all(np.linalg.norm(row['preview_action'][:3]) * .05 == pytest.approx(.003)
               for row in result['offers'])
    assert policy.prediction == old_prediction and snapshot(policy) == before
    assert policy.client.calls == previous_calls
    assert policy.pending is policy.local.pending is None
