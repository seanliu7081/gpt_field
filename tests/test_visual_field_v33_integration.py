"""V33 policy/preview transactions with real RGB tracks and complete geometry.

The paid adverse-heading row is an explicit boundary fixture, not generated
image evidence or a reported task result. Subsequent acknowledgements use a
deterministic measured-state plant with unchanged RGB; no archived frame is
reused as the outcome of a counterfactual action.
"""
import copy
import pickle
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.visual_field_policy_v33 import VisualFieldPolicyV33
from test_visual_field_v6_policy import Client
from test_visual_field_v13_execution import observation
from test_visual_field_v20_policy import prediction
from test_visual_field_v25_policy import explicit


def configured_policy():
    value = explicit(prediction())
    value.update(scene_revisions=[], field_proposal_id=None)
    value['visual_references']['external']['reference_role'] = 'tool_anatomy'
    policy = VisualFieldPolicyV33(Client(value))
    obs = observation()
    policy._refresh(obs)
    guard, field = policy.translation_guard, policy.local
    state = np.asarray(obs.state)
    contract = field.nominal_regions_for_segment(
        state[:3], state[:3]+[.002, 0., 0.])[0]['directional_retreat_contract']
    guard.experiments = [{'direction': [1., 0., 0.],
        'support_generation': guard._support_generation,
        'measured_travel_m': .0025, 'acknowledged_actions': 5}]
    guard._last_charged_experiment = 0
    guard._charge_uncertain_motion = True
    guard.observation_travel_m = .0025
    guard.worsening = {guard.target_id}
    guard.worsening_motion[guard.target_id] = {
        'measured_direction_xyz': [1., 0., 0.],
        'measured_net_translation_m': .012, 'state8': state.tolist(),
        'observation_epoch': guard.sweep_observation_epoch,
        'geometry_sha256': contract['geometry_sha256'],
        'risk_evidence': {'state': 'worsening', 'source': 'explicit_boundary_fixture'}}
    return policy, obs, value


def reverse_response(value, offset=(-.008, 0., 0.)):
    result = copy.deepcopy(value)
    result.update(phase='recover', recovery_route='reverse_observed')
    result['subgoal_update'].update(action='replace',
        reason='Reverse acknowledged adverse motion before continuing acquisition')
    result['field_intent'].update(goal_offset_m=list(offset), max_travel_m=.008,
                                 command_limit_m=.002)
    result['motion_context'].update(translation_mode='separate',
        purpose='reference_repair', intended_contact_id=None)
    result['scene_inventory'][0]['geometry'] = None
    result['visual_references']['external'].update(update='retain', seed=None)
    return result


def snapshot(policy):
    field, guard = policy.local, policy.translation_guard
    return {
        'field_dict': field.__dict__, 'guard_dict': guard.__dict__,
        'field_pending': field.pending, 'guard_pending': guard._reverse_abort_pending,
        'summary': copy.deepcopy(guard.summary()),
        'field_state': pickle.dumps({key: getattr(field, key) for key in (
            'origin', 'goal', 'intent', 'pending', 'actions', 'measured_travel_m',
            'plan_calls', 'recovery_route_binding', 'heading_recovery')}),
        'tracker': field.tracker, 'tracker_state': pickle.dumps(field.tracker.__dict__),
        'planner': field.planner, 'tool': guard.tool_track,
        'visual_track_states': pickle.dumps({key: track.__dict__
            for key, track in policy.monitor.tracks.items() if track is not None}),
        'physical_debt': copy.deepcopy((guard.experiments, guard.observation_travel_m,
                                      guard.egress_evidence.summary())),
    }


def assert_unchanged(before, policy):
    after = snapshot(policy)
    for key in ('field_dict', 'guard_dict', 'field_pending', 'guard_pending',
                'tracker', 'planner', 'tool'):
        assert after[key] is before[key], key
    for key in ('summary', 'field_state', 'tracker_state', 'visual_track_states', 'physical_debt'):
        assert after[key] == before[key], key


def test_actual_executor_preview_is_pure_and_does_not_license_unreviewed_execution():
    policy, obs, value = configured_policy()
    try:
        before = snapshot(policy)
        preview = policy.local.preview_recovery(obs.state, obs.joint_position, [-.008, 0., 0.])
        assert preview['accepted'], preview['status']
        assert preview['requires_explicit_model_choice']
        details = preview['diagnostics']['translation_admission']['diagnostics']
        assert details['precheckpoint_reverse_abort']['last_admission']['requires_explicit_model_choice']
        assert_unchanged(before, policy)
        assert policy.client.calls == 1
        # Merely installing a matching executor route, without a model review
        # bound by the policy, cannot reuse a preview's temporary permission.
        field, guard = policy.local, policy.translation_guard
        intent = copy.deepcopy(field.intent)
        intent.update(goal_offset_m=[-.008, 0., 0.], max_travel_m=.008,
                      command_limit_m=.002, motion_axes='xyz')
        guard.context = {**guard.context, 'translation_mode': 'separate',
                         'purpose': 'reference_repair', 'intended_contact_id': None}
        field.configure(obs.state, intent, scene_assessment=guard.context,
                        inventory_materialization=field._transit_materialization)
        assert field.bind_recovery_route('reverse_observed', obs.state)['accepted']
        selected = field.select(obs.state, obs.joint_position)
        assert not selected['accepted']
        assert selected['status'] == 'translation_recovery_direction_change_requires_review'
        assert field.pending is None and guard._reverse_abort_pending is None
        assert guard.observation_travel_m == .0025 and len(guard.experiments) == 1
    finally:
        policy.finish()


@pytest.mark.parametrize('fault', ['overlong', 'scene_rejected', 'scene_exception'])
def test_unaccepted_preview_restores_context_and_does_not_authorize_ordinary_selection(fault):
    policy, obs, _ = configured_policy()
    try:
        before = snapshot(policy)
        kwargs = {}
        offset = [-.008, 0., 0.]
        if fault == 'overlong':
            offset = [-.00808, 0., 0.]
        elif fault == 'scene_rejected':
            kwargs['scene_assessor'] = lambda *args: {
                'accepted': False, 'reason': 'fixture_current_scene_refusal', 'diagnostics': {}}
        else:
            def raises(*args):
                raise RuntimeError('fixture_scene_exception')
            kwargs['scene_assessor'] = raises
        if fault == 'scene_exception':
            with pytest.raises(RuntimeError, match='fixture_scene_exception'):
                policy.local.preview_recovery(obs.state, obs.joint_position, offset, **kwargs)
        else:
            preview = policy.local.preview_recovery(obs.state, obs.joint_position, offset, **kwargs)
            assert not preview['accepted']
            if fault == 'overlong':
                assert preview['status'] == 'heading_recovery_goal_exceeds_bounded_route'
        assert_unchanged(before, policy)
        assert policy.translation_guard._reverse_abort_preview_context is None
        selected = policy.local.select(obs.state, obs.joint_position)
        assert not selected['accepted']
        assert policy.translation_guard._reverse_abort_pending is None
        assert not policy.translation_guard.reverse_abort_events
        assert policy.translation_guard.observation_travel_m == .0025
    finally:
        policy.finish()


def test_policy_replacement_acknowledges_once_and_pending_preview_preserves_exact_command():
    policy, obs, value = configured_policy()
    try:
        old_row = copy.deepcopy(policy.translation_guard.experiments[0])
        policy.client.value = reverse_response(value)
        policy._refresh(obs)
        decision = policy.act(obs)
        assert decision.diagnostics['selection']['accepted']
        guard, field = policy.translation_guard, policy.local
        assert guard.reverse_abort_context['model_requests'] == policy.model_requests == 2
        assert guard.reverse_abort_context['goal_xyz_m'] == field.goal.tolist()
        assert np.asarray(decision.action[:3])*.05 == pytest.approx([-.002, 0., 0.])
        assert not guard.reverse_abort_events and len(guard.experiments) == 1
        before = snapshot(policy)
        rejected = field.preview_recovery(obs.state, obs.joint_position, [-.008, 0., 0.])
        assert not rejected['accepted']
        assert rejected['status'] == 'recovery_preview_unacknowledged_action'
        assert_unchanged(before, policy)
        # One fresh synthetic measured transition; no historical future RGB.
        state = np.asarray(obs.state).copy()
        state[:3] += np.asarray(decision.action[:3])*.05*.25
        after = replace(obs, state=tuple(state))
        policy.observe_transition(obs, decision, after)
        assert policy.pending is field.pending is guard._reverse_abort_pending is None
        assert len(guard.reverse_abort_events) == 1
        assert guard.experiments[0] == old_row
        assert len(guard.experiments) == 2
        assert guard.experiments[1]['measured_travel_m'] == pytest.approx(.0005)
        assert guard.observation_travel_m == pytest.approx(.003)
        assert guard.worsening == {guard.target_id}
        assert guard.reverse_abort_events[0]['goal_xyz_m'] == field.goal.tolist()
        assert policy.client.calls == 2
    finally:
        policy.finish()
