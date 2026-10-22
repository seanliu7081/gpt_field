"""V20 fixed-goal control dependency tests using actual RGB tracking.

Static rendered RGB is a deliberately weak projection, not a certified robot
feature. Acknowledged robot motion must keep its existing visual debt while
only the new exact candidate geometry can permit further fixed-goal travel.
"""
import copy
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.visual_field_policy_v19 import VisualFieldPolicyV19
from gpt_field.visual_field_policy_v20 import VisualFieldPolicyV20
from test_visual_field_v6_policy import Client
from test_visual_field_v17_policy import prediction as contact_prediction
from test_visual_field_v13_execution import observation


def prediction():
    value = contact_prediction()
    value.update(phase='approach', contact_alignment=None, target_contact='none')
    value['motion_context'].update(translation_mode='transit', intended_contact_id=None)
    value['field_intent'].update(goal_offset_m=[.065, 0., 0.], command_limit_m=.006,
                                 max_actions=120, max_travel_m=.12)
    geometry = value['scene_inventory'][0]['geometry']
    geometry.update(center_offset_m=[.3, .3, .3], half_extents_m=[.01]*3, uncertainty_m=.01)
    value['target_geometry'].update(**copy.deepcopy(geometry), allowed_contact='none')
    return value


def fixed_rgb_transition(policy, obs, gain=.35):
    decision = policy.act(obs)
    delta = np.asarray(decision.action[:3])*.05*gain
    state = np.asarray(obs.state).copy(); state[:3] += delta
    after = replace(obs, state=tuple(state))  # Both actual RGB inputs are unchanged.
    row = policy.observe_transition(obs, decision, after)
    return after, decision, row


def blocked_policy():
    """A declared monitor boundary fixture, not invented success evidence."""
    policy = VisualFieldPolicyV20(Client(prediction())); obs = observation()
    policy._refresh(obs)
    policy.monitor._active_reference.update(
        blocked_reason=policy.monitor.uncertainty_budget_reason, unvalidated_travel_m=.031)
    policy.monitor.status = policy.monitor.uncertainty_budget_reason
    return policy, obs


def test_actual_static_rgb_fixed_goal_reuses_one_request_beyond_visual_budget():
    client = Client(prediction()); policy = VisualFieldPolicyV20(client); obs = observation()
    initial = np.asarray(obs.state[:3]); accepted = 0; exceptional = 0
    goal = initial + np.array([.065, 0., 0.])
    for _ in range(35):
        obs, decision, row = fixed_rgb_transition(policy, obs)
        selected = decision.diagnostics['selection']
        assert selected['accepted'], selected
        accepted += 1
        admission = selected['diagnostics'].get('proprioceptive_transit_admission')
        if admission:
            assert admission['accepted']
            assert np.linalg.norm(np.asarray(decision.action[:3])*.05) <= .005+1e-12
            exceptional += 1
        assert policy.goal == pytest.approx(goal)
        assert policy.local.goal == pytest.approx(goal)
        assert not policy.monitor.response_ready and not row['visual']['response_validated']
        assert policy.pending is None and policy.local.pending is None
        assert policy.local.proprioceptive_transit_request is None
        assert not policy.refresh_reason
        assert not row['task_success_certified']
    assert np.linalg.norm(np.asarray(obs.state[:3])-initial) > .035
    assert client.calls == 1 and accepted == 35 and exceptional > 2
    assert policy.monitor._active_reference['unvalidated_travel_m'] > .035
    assert policy.monitor.measured_control_limit() == (0., policy.monitor.uncertainty_budget_reason)
    assert policy.translation_guard.observation_travel_m == 0.
    assert policy.visual_corrections == policy.local.goal_revisions == 0
    assert not row['proprioceptive_transit']['image_response_validated_by_this_branch']
    assert not row['proprioceptive_transit']['physical_budget_reset_by_this_branch']


def test_legacy_default_hooks_still_stop_at_same_unvalidated_visual_budget():
    policy = VisualFieldPolicyV19(Client(prediction())); obs = observation(); initial = np.asarray(obs.state[:3])
    assert policy._field_control_limits(obs) == policy.monitor.measured_control_limit()
    assert policy._post_field_control_limit(obs) == policy.monitor.control_limit()
    for _ in range(35):
        obs, decision, row = fixed_rgb_transition(policy, obs)
        if policy.refresh_reason:
            break
    assert policy.refresh_reason == policy.monitor.uncertainty_budget_reason
    assert policy.client.calls == 1
    assert .029 <= np.linalg.norm(np.asarray(obs.state[:3])-initial) < .033
    assert not policy.monitor.response_ready


@pytest.mark.parametrize('case', ['raw_payload','resolved_payload','contact_permission','contact_descriptor',
    'align_phase','retarget','lost_moving','lost_anchor','missing','retain_rejected','domain_invalid',
    'anchor_latch','contradiction_status','retained_payload','unconfirmed_candidate','wrong_reason'])
def test_provisional_exception_excludes_other_control_dependencies(case):
    policy, obs = blocked_policy(); reason = policy.monitor.uncertainty_budget_reason
    assert policy._transit_eligibility(reason, obs.state)
    if case == 'raw_payload': policy.prediction['visual_references']['external']['moving_kind'] = 'payload'
    elif case == 'resolved_payload': policy.monitor.refs['external']['moving_kind'] = 'payload'
    elif case == 'contact_permission': policy.prediction['target_contact'] = 'fingers'
    elif case == 'contact_descriptor': policy.prediction['contact_alignment'] = {'update':'retain'}
    elif case == 'align_phase': policy.prediction['phase'] = 'align'
    elif case == 'retarget': policy.local.goal_revisions = 1
    elif case == 'lost_moving': policy.monitor.tracks['moving'].reliable = False
    elif case == 'lost_anchor': policy.monitor.tracks['target'].reliable = False
    elif case == 'missing': policy.monitor.missing = 2
    elif case == 'retain_rejected': policy.monitor.reference_update_status = 'external_retain_rejected'
    elif case == 'domain_invalid': policy.translation_guard.domain_invalid = 'translation_orientation_domain_changed'
    elif case == 'anchor_latch': policy._transit_anchor_shifted = True
    elif case == 'contradiction_status': policy.monitor.status = 'visual_response_contradiction'
    elif case in ('retained_payload','unconfirmed_candidate'):
        result = policy.target_contact.remember_grasp_candidate(obs.state, policy.prediction['target_id'])
        assert result['accepted']
        if case == 'retained_payload':
            assert policy.target_contact.confirm_payload_attachment(obs.state,
                policy.prediction['target_id'], observed=True)['accepted']
    elif case == 'wrong_reason': reason = 'external_reference_unavailable'
    assert not policy._transit_eligibility(reason, obs.state)


def test_new_cumulative_contradiction_rejects_even_when_weak_block_hides_label():
    policy, obs = blocked_policy(); monitor = policy.monitor
    for _ in range(6):
        monitor._prediction_window.append({'delta':np.array([.001,0,0]),
            'predicted':np.array([.3,0]), 'observed':np.zeros(2), 'noise':.08})
    assert monitor._response_evidence()['state'] == 'weakly_observable'
    assert monitor.cumulative_response_evidence()['contradicted']
    assert not policy._transit_eligibility(monitor.uncertainty_budget_reason, obs.state)
    cap, reason = policy._field_control_limits(obs)
    assert cap == 0. and reason == monitor.uncertainty_budget_reason
    assert policy.local.proprioceptive_transit_request is None


def test_provisional_request_does_not_retarget_and_is_not_cached_after_act():
    policy, obs = blocked_policy(); goal = policy.goal.copy(); debt = policy.monitor._active_reference['unvalidated_travel_m']
    assert policy._field_control_limits(obs) == (None, None)
    assert policy.local.proprioceptive_transit_request['state8'] == list(obs.state)
    assert not policy._allows_planar_correction()
    assert policy._refine_contact_goal(obs) is None
    decision = policy.act(obs)
    assert decision.diagnostics['selection']['accepted']
    assert policy.local.proprioceptive_transit_request is None
    assert policy.goal == pytest.approx(goal)
    assert policy.monitor._active_reference['unvalidated_travel_m'] == debt
    policy.observe_transition(obs, decision, obs)
    assert policy.pending is policy.local.pending is None


def test_observed_anchor_jump_is_latched_and_retain_cannot_erase_it():
    policy, obs = blocked_policy(); anchor = policy.monitor.tracks['target']
    policy._anchor_before_observe = anchor.position.copy()
    anchor.position = anchor.position + [6.1, 0.]
    policy._observe_anchor_shift()
    assert policy._transit_anchor_shifted
    assert not policy._transit_eligibility(policy.monitor.uncertainty_budget_reason, obs.state)
    # Return the point to its image location to isolate refresh identity behavior;
    # the independently observed jump must remain latched on this same track.
    anchor.position = anchor.position - [6.1, 0.]
    retained = prediction(); retained['subgoal_update']['action'] = 'retain'
    for view in ('external','wrist'):
        retained['visual_references'][view].update(update='retain', seed=None)
    policy.client.value = retained
    policy._refresh(obs)
    assert policy.monitor.tracks['target'] is anchor
    assert policy._transit_anchor_shifted
    assert not policy._transit_eligibility(policy.monitor.uncertainty_budget_reason, obs.state)


def test_explicit_new_anchor_seed_can_start_new_observation_domain_without_refunding_debt():
    policy, obs = blocked_policy(); anchor = policy.monitor.tracks['target']
    policy._transit_anchor_shifted = True
    debt = policy.monitor._active_reference['unvalidated_travel_m']
    policy.client.value = prediction()
    policy.client.value['subgoal_update']['action'] = 'retain'
    policy._refresh(obs)
    assert policy.monitor.tracks['target'] is not anchor
    assert not policy._transit_anchor_shifted
    assert policy.monitor._active_reference['unvalidated_travel_m'] == debt
