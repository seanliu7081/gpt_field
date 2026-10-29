"""Current wrist reports require the real preview, including proxy role binding."""
import copy

import numpy as np
import pytest

from gpt_field.visual_field_proposals_v35 import FieldProposalBankV35
from test_visual_field_v34_preview import reconstructed, archived


def install_wrist_report(policy, monkeypatch):
    report = {'accepted': True, 'objective_source': 'wrist_bearing', 'candidates': [
        {'goal_offset_m': [.008, 0., 0.], 'predicted_error_reduction_px': 2.,
         'objective_source': 'wrist_bearing'}]}
    monkeypatch.setattr(policy.visual_goal_field, 'propose', lambda *args, **kwargs: copy.deepcopy(report))
    policy.recovery_options = {'candidates': []}
    policy.prediction['phase'] = 'recover'
    return report


def test_actual_proxy_handoff_previews_noncontact_goal_after_configure_without_live_binding(archived, monkeypatch):
    policy, obs, _, snapshot = archived
    report = install_wrist_report(policy, monkeypatch)
    before, prediction = snapshot(policy), copy.deepcopy(policy.prediction)
    role = copy.deepcopy(policy.translation_guard.reference_role_context)
    result = FieldProposalBankV35().build(policy, obs, report)
    assert result['offers'], result['rejected']
    first = result['offers'][0]
    assert first['phase'] == 'approach'
    assert first['visual_prediction']['objective_source'] == 'wrist_bearing'
    assert first['preview_status'] == 'metric_field_tracking'
    assert np.linalg.norm(first['preview_action'][:3]) > 0
    assert first['actual_selection_rechecks_all_constraints']
    assert policy.translation_guard.reference_role_context == role
    assert role['controller'] == 'direct'
    assert snapshot(policy) == before and policy.prediction == prediction
    assert result['phase_transition']['explicit_model_choice_required']
    assert not result['wrist_bearing_is_contact_or_depth_evidence']


@pytest.mark.parametrize('fault', ['stale', 'anatomy', 'unqualified', 'contact', 'pending', 'lost_hand', 'unknown'])
def test_wrist_offer_cannot_skip_current_qualification_or_physical_checks(archived, monkeypatch, fault):
    policy, obs, _, snapshot = archived
    report = install_wrist_report(policy, monkeypatch)
    presented = copy.deepcopy(report)
    if fault == 'stale': presented['candidates'][0]['goal_offset_m'][0] += .001
    elif fault == 'anatomy': presented['objective_source'] = 'anatomical_relative_pixels'
    elif fault == 'unqualified': presented['accepted'] = False
    elif fault == 'contact': policy.prediction['target_contact'] = 'fingers'
    elif fault == 'pending': policy.local.pending = {'unacknowledged': True}
    elif fault == 'lost_hand': policy.monitor.tracks['tool'].reliable = False
    else:
        center = np.asarray(obs.state[:3]) - policy.local.origin
        policy.local.intent['unknown_regions'].append({'id': 'unobserved',
            'lower_offset_m': (center - .04).tolist(), 'upper_offset_m': (center + .04).tolist(),
            'uncertainty_m': .01})
    before = snapshot(policy)
    result = FieldProposalBankV35().build(policy, obs, presented)
    assert not result['offers'], fault
    assert snapshot(policy) == before
