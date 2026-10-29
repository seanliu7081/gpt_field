"""V13 cross-layer contact refinement with measured synthetic RGB transitions.

No simulator, model inference, scene truth, forced tracker/response flags or
attachment certificates are used. The rendered plant is only the test sensor.
"""
import copy
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.visual_field_policy_v13 import VisualFieldPolicyV13
from gpt_field.visual_field_v13 import ContactSubgoalLedger
from gpt_field.visual_field_v8 import measured_current_pose_hold_action
from test_visual_field_v4_policy import ORIGIN, Q
from test_visual_field_v6_policy import Client
from test_visual_field_v11_execution import prediction as base_prediction
from test_visual_contact_alignment_v13 import (HAND, EXT_TARGET, WRIST_TARGET,
    TARGET_STATE, TOOL_ROI, render, box, external_delta, wrist_delta)


def observation(position=(0.,0.,0.)):
    external,wrist=render(position)
    return PolicyObservation(image_to_data_url(external,rotate=False),
        image_to_data_url(wrist,rotate=False),
        tuple(ORIGIN+np.asarray(position))+(0.,0.,0.,.04,-.04),
        'Align the open fingers to the visible contact point',Q,(0.,)*7)


def prediction(goal=(.045,0.,0.), *, retain=False):
    result=base_prediction(goal=goal,phase='align')
    result['target_id']='part'
    result['field_intent'].update(max_actions=120,command_limit_m=.006,
        goal_tolerance_m=.003,max_travel_m=.12)
    result['motion_context'].update(tool_external_roi_uv=TOOL_ROI.tolist(),
        translation_mode='interaction',intended_contact_id='part')
    result['target_geometry']={'id':'part',
        'center_offset_m':(TARGET_STATE+[0.,0.,-.015]).tolist(),
        'half_extents_m':[.004,.004,.004], 'uncertainty_m':.003,
        'allowed_contact':'fingers'}
    result['visual_references']={'external':{'update':'retain' if retain else 'seed',
        'moving_id':'tool','moving_kind':'tool','target_id':'part',
        'goal_offset_uv':((HAND+external_delta(goal)-EXT_TARGET)/223).tolist(),
        'seed':None if retain else {'moving_point_uv':(HAND/223).tolist(),
            'moving_roi_uv':box(HAND).tolist(), 'target_anchor_uv':(EXT_TARGET/223).tolist(),
            'target_roi_uv':box(EXT_TARGET).tolist(),'tool_point_uv':None,'tool_roi_uv':None}},
        'wrist':{'update':'retain' if retain else 'seed','target_id':'part',
            'desired_target_uv':((WRIST_TARGET+wrist_delta(TARGET_STATE))/223).tolist(),
            'seed':None if retain else {'target_anchor_uv':(WRIST_TARGET/223).tolist(),
                'target_roi_uv':box(WRIST_TARGET).tolist()}}}
    result['contact_alignment']={'update':'retain' if retain else 'seed','target_id':'part',
        'external':None if retain else {'target_contact_point_uv':(EXT_TARGET/223).tolist(),
            'tool_pinch_point_uv':((EXT_TARGET-external_delta(TARGET_STATE))/223).tolist()},
        'wrist':None if retain else {'target_contact_point_uv':(WRIST_TARGET/223).tolist(),
            'tool_pinch_point_uv':((WRIST_TARGET+wrist_delta(TARGET_STATE))/223).tolist()}}
    if retain:result['subgoal_update']['action']='retain'
    return result


def transition(policy,obs,*,gain=.35):
    decision=policy.act(obs)
    delta=np.asarray(decision.action[:3])*.05
    point=np.asarray(obs.state[:3])-ORIGIN+delta*gain
    after=observation(point)
    row=policy.observe_transition(obs,decision,after)
    return after,decision,row


def test_real_rgb_contact_goal_retargets_and_executes_within_one_response():
    initial=prediction();client=Client(initial);policy=VisualFieldPolicyV13(client)
    obs=observation();model_goal=ORIGIN+initial['field_intent']['goal_offset_m']
    goals=[];changes=[];accepted=0
    for _ in range(35):
        before_goal=None if policy.goal is None else policy.goal.copy()
        obs,decision,row=transition(policy,obs)
        if decision.diagnostics['selection']['accepted']:
            accepted+=1
            assert decision.action[6]==0.  # Alignment never closes automatically.
        goals.append(policy.goal.copy())
        if before_goal is not None and np.linalg.norm(policy.goal-before_goal)>1e-10:
            changes.append(float(np.linalg.norm(policy.goal-before_goal)))
        assert policy.goal==pytest.approx(policy.subgoals.active['world_goal_xyz'])
        assert policy.goal==pytest.approx(policy.local.goal)
        assert np.linalg.norm(policy.goal-model_goal)<=.03+1e-12
        assert policy.subgoals.active['contact_correction_travel_m']<=.06+1e-12
        assert policy.pending is None and policy.local.pending is None
        if policy.refresh_reason:break
    assert client.calls==1 and accepted>10
    assert policy.visual_corrections>=3 and changes and max(changes)<=.006+1e-12
    assert policy.contact_alignment.response_ready
    assert policy.local.goal_revisions==policy.subgoals.active['contact_correction_count']
    assert np.linalg.norm(goals[-1]-model_goal)>.012
    assert policy.subgoals.active['model_world_goal_xyz']==pytest.approx(model_goal)
    assert policy.prediction==initial and client.value==initial
    assert not row['task_success_certified'] and not row['contact_alignment']['automatic_close']
    assert not row['contact_alignment']['grasp_certified']


def test_unsupported_contact_fit_preserves_useful_nominal_field_motion():
    policy=VisualFieldPolicyV13(Client(prediction()));obs=observation()
    initial_goal=ORIGIN+np.array([.045,0.,0.])
    for _ in range(5):
        obs,decision,row=transition(policy,obs)
        assert decision.diagnostics['selection']['accepted']
        assert np.linalg.norm(decision.action[:3])>0.
        assert not policy.contact_alignment.response_ready
        assert policy.goal==pytest.approx(initial_goal)
        assert policy.visual_corrections==0 and not policy.refresh_reason
    assert policy.client.calls==1 and policy.field_effective_actions==5
    assert obs.state[0]>ORIGIN[0]+.003
    assert row['contact_alignment']['response']['sample_count']>0


def test_two_view_alignment_reviews_immediately_without_additional_descent_or_close():
    initial=prediction();contact_at=np.array([.018,0.,0.])
    initial['contact_alignment']['external']['tool_pinch_point_uv']=(
        (EXT_TARGET-external_delta(contact_at))/223).tolist()
    initial['contact_alignment']['wrist']['tool_pinch_point_uv']=(
        (WRIST_TARGET+wrist_delta(contact_at))/223).tolist()
    initial['visual_references']['wrist']['desired_target_uv']=copy.deepcopy(
        initial['contact_alignment']['wrist']['tool_pinch_point_uv'])
    policy=VisualFieldPolicyV13(Client(initial));obs=observation()
    for _ in range(30):
        obs,decision,row=transition(policy,obs)
        assert decision.diagnostics['selection']['accepted']
        if policy.contact_alignment.ready_for_model_review:break
    assert policy.contact_alignment.ready_for_model_review
    assert policy.contact_alignment.response_ready
    assert policy.client.calls==1 and not policy.refresh_reason
    before=copy.deepcopy(policy.subgoals.active)
    decision=policy.act(obs)
    assert not decision.diagnostics['selection']['accepted']
    assert decision.diagnostics['selection']['status']=='contact_alignment_observed'
    assert list(decision.action)==measured_current_pose_hold_action()
    assert policy.local.pending is None
    assert policy.client.calls==1 and policy.refresh_reason=='contact_alignment_observed'
    row=policy.observe_transition(obs,decision,obs)
    assert policy.subgoals.active['world_goal_xyz']==before['world_goal_xyz']
    assert policy.subgoals.active['contact_correction_travel_m']==before['contact_correction_travel_m']
    assert not row['contact_alignment']['grasp_certified'] and policy.gripper_actions==0


def test_retain_refresh_preserves_real_fit_world_goal_and_contact_correction_debt():
    client=Client(prediction());policy=VisualFieldPolicyV13(client);obs=observation()
    for _ in range(9):obs,decision,row=transition(policy,obs)
    assert policy.visual_corrections>0 and policy.contact_alignment.response_ready
    before=copy.deepcopy(policy.subgoals.active)
    sample_count=len(policy.contact_alignment.model.samples)
    identities=dict(policy.monitor.tracks)
    retained=prediction(goal=(.065,0.,0.),retain=True)
    # The next model report retains the same synthetic fixed target hypothesis.
    retained['target_geometry']['center_offset_m']=(
        np.array(retained['target_geometry']['center_offset_m'])-(np.array(obs.state[:3])-ORIGIN)).tolist()
    client.value=retained;policy.refresh_reason='offline_requested_observation_refresh'
    decision=policy.act(obs)
    assert decision.diagnostics['selection']['accepted'],decision.diagnostics['selection']
    assert client.calls==2 and len(policy.contact_alignment.model.samples)==sample_count
    assert policy.contact_alignment.response_ready
    assert all(policy.monitor.tracks[k] is v for k,v in identities.items())
    for key in ('model_world_goal_xyz','world_goal_xyz','contact_correction_count','contact_correction_travel_m'):
        assert policy.subgoals.active[key]==before[key]
    assert policy.goal==pytest.approx(before['world_goal_xyz'])
    assert policy.local.original_goal==pytest.approx(before['world_goal_xyz'])
    policy.observe_transition(obs,decision,obs)
    assert policy.pending is None and policy.local.pending is None


def test_contact_ledger_step_radius_and_cumulative_bounds_survive_controller_refreshes():
    ledger=ContactSubgoalLedger();initial=prediction();obs=observation()
    assert ledger.apply(obs.state,initial)['accepted']
    ledger.enable_contact_refinement(True)
    original=np.asarray(ledger.active['model_world_goal_xyz']);current=original.copy()
    assert not ledger.record_contact_correction(current+[.0061,0.,0.])
    for i in range(5):
        current=current+[0.,.006,0.]
        assert ledger.record_contact_correction(current)
    assert ledger.active['contact_correction_travel_m']==pytest.approx(.03)
    assert not ledger.record_contact_correction(current+[0.,.001,0.])  # 30mm model radius.
    repair=base_prediction('direct',phase='align')
    repair['subgoal_update']={'action':'retain','subgoal_id':'local-goal','reason':'Observe current contact'}
    assert ledger.apply(obs.state,repair)['accepted']
    ledger.enable_contact_refinement(False)
    assert ledger.active['contact_correction_count']==5
    resumed=prediction(retain=True)
    assert ledger.apply(obs.state,resumed)['accepted']
    ledger.enable_contact_refinement(True)
    assert ledger.active['model_world_goal_xyz']==pytest.approx(original)
    for i in range(5):
        current=current-[0.,.006,0.]
        assert ledger.record_contact_correction(current)
    assert ledger.active['contact_correction_count']==10
    assert ledger.active['contact_correction_travel_m']==pytest.approx(.06)
    assert not ledger.record_contact_correction(current+[.001,0.,0.])
    assert ledger.active['world_goal_xyz']==pytest.approx(original)
    assert not ledger.active['task_success_certified']


def test_actual_planned_curve_supports_xyz_contact_correction_and_reduces_two_view_error():
    # The local graph route collects multiple translation directions while
    # approaching a valid approximate goal. No separate calibration sequence.
    initial=prediction(tuple(TARGET_STATE*1.4))
    policy=VisualFieldPolicyV13(Client(initial));obs=observation();errors=[]
    for _ in range(35):
        obs,decision,row=transition(policy,obs)
        if not decision.diagnostics['selection']['accepted']:
            assert decision.diagnostics['selection']['status']=='contact_alignment_observed'
            break
        errors.append(float(np.linalg.norm(policy.contact_alignment.current['error'])))
        if policy.refresh_reason:break
    assert policy.client.calls==1 and policy.field_effective_actions>12
    assert policy.contact_alignment.response_ready and policy.contact_alignment.model.rank==3
    assert policy.visual_corrections>=3
    assert errors[-1]<errors[0]*.3
    assert np.linalg.norm(np.asarray(obs.state[:3])-ORIGIN-TARGET_STATE)<.005
    assert policy.subgoals.active['contact_correction_travel_m']<=.06+1e-12
    assert not row['contact_alignment']['grasp_certified']


def test_runtime_recovery_cannot_be_pulled_back_toward_contact_by_a_descriptor():
    # Deliberately bypass the wire validator to test defense in depth. A valid
    # client now rejects this descriptor in recover, but runtime must also keep
    # a retreat goal intact if passed one by another caller.
    initial=prediction();initial['phase']='recover'
    policy=VisualFieldPolicyV13(Client(initial));obs=observation()
    original=ORIGIN+initial['field_intent']['goal_offset_m']
    for _ in range(12):
        obs,decision,row=transition(policy,obs)
        assert decision.diagnostics['selection']['accepted']
        assert policy.goal==pytest.approx(original)
        assert not policy.local.contact_refinement_enabled
        assert not policy.subgoals.active['contact_refinement_enabled']
        assert policy.last_contact_refinement is None
    assert policy.contact_alignment.response_ready  # A real fit cannot override the recovery role.
    assert policy.visual_corrections==0 and policy.client.calls==1
    assert policy.subgoals.active['contact_correction_travel_m']==0.
    assert policy.local.goal_revisions==0 and not row['task_success_certified']
