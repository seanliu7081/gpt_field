"""Actual V13 policy lifecycle integration with explicit attachment-boundary injection.

RGB and robot-state transition plumbing remain real. Two tests inject the local
monitor's *already evaluated* attachment result to exercise policy wiring; these
are not perceptual evidence or simulated task-success claims.
"""
import copy
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.visual_field_policy_v13 import VisualFieldPolicyV13
from gpt_field.visual_field_v8 import measured_current_pose_hold_action
from test_visual_field_v6_policy import Client, observation
from test_visual_field_v11_execution import prediction as base_prediction


def prediction(controller='gripper',*,gripper='close'):
    pred=base_prediction(controller,phase='grasp' if controller=='gripper' and gripper=='close' else
                         'release' if controller=='gripper' else 'approach')
    if controller=='gripper':pred['gripper_intent']['command']=gripper
    pred['target_geometry']={'id':pred['target_id'],'center_offset_m':[0.,.0435,-.0036],
        'half_extents_m':[.001]*3,'uncertainty_m':.003,'allowed_contact':'fingers'}
    pred['motion_context'].update(translation_mode='interaction',intended_contact_id=pred['target_id'])
    pred['contact_alignment']=None
    return pred


def activate_known_payload(policy,obs,*,palm_overlap=False):
    identity='previous_payload'
    region={'id':identity,'center_offset_m':[0.,0.,-.08] if palm_overlap else [0.,.0435,-.0036],
            'half_extents_m':[.001]*3,'uncertainty_m':.003,'allowed_contact':'fingers'}
    policy.target_contact.configure(obs.state,region,target_id=identity)
    policy.target_contact.remember_grasp_candidate(obs.state,identity)
    policy.target_contact.confirm_payload_attachment(obs.state,identity,observed=True)
    return identity


def test_first_accepted_close_captures_preclosure_pose_once_across_real_policy_transitions():
    pred=prediction();client=Client(pred);policy=VisualFieldPolicyV13(client);obs=observation()
    original=obs.state
    for index in range(3):
        decision=policy.act(obs)
        assert decision.diagnostics['selection']['accepted'] and decision.action[6]==1.
        after=replace(obs,state=(obs.state[0]+.0001,*obs.state[1:6],.038-.002*index,-.038+.002*index))
        policy.observe_transition(obs,decision,after);obs=after
    assert client.calls==1 and policy.gripper_actions==3
    assert policy.target_contact.carried_summary(obs.state)['candidate_ids']==[pred['target_id']]
    assert not policy.target_contact.carried_summary(obs.state)['active']
    policy.target_contact.confirm_payload_attachment(obs.state,pred['target_id'],observed=True)
    assert policy.target_contact.carried_summary(obs.state)['snapshot_state8']==list(original)
    assert policy.pending is None and policy.gripper.pending is None


def test_policy_activates_snapshot_from_local_payload_moving_id_after_feedback_not_destination_id(monkeypatch):
    pred=prediction();policy=VisualFieldPolicyV13(Client(pred));obs=observation()
    decision=policy.act(obs)  # Captures unconfirmed preclosure target geometry.
    payload_id=pred['target_id']
    assert not policy.target_contact.carried_summary(obs.state)['active']
    real_alignment_observe=policy.contact_alignment.observe
    def observed_attachment_boundary(*args,**kwargs):
        real_alignment_observe(*args,**kwargs)
        # Explicit injected local-monitor boundary, not a model or evaluator flag.
        policy.monitor.refs['external']={'moving_kind':'payload','moving_id':payload_id,'target_id':'support'}
        policy.monitor.attachment_evidence=True
    monkeypatch.setattr(policy.contact_alignment,'observe',observed_attachment_boundary)
    after=replace(obs,state=(*obs.state[:6],.037,-.037))
    row=policy.observe_transition(obs,decision,after)
    carried=policy.target_contact.carried_summary(after.state)
    assert carried['active'] and carried['payload_id']==payload_id
    assert carried['snapshot_state8']==list(obs.state)
    assert carried['confirmed_state8']==list(after.state)
    assert not row['task_success_certified']
    assert row['target_contact_geometry']['carried_payload']['payload_id']==payload_id


@pytest.mark.parametrize('controller',['visual_field','gripper'])
def test_carried_palm_veto_rolls_back_provisional_controller_and_acknowledges_logged_hold(controller):
    policy=VisualFieldPolicyV13(Client(prediction(controller)));obs=observation()
    activate_known_payload(policy,obs,palm_overlap=True)
    decision=policy.act(obs)
    assert not decision.diagnostics['selection']['accepted']
    assert decision.diagnostics['selection']['status']=='carried_payload_forbidden_palm_intersection'
    assert decision.action==tuple(measured_current_pose_hold_action())
    assert not decision.diagnostics['rejection_hold']['requested_motion_executed']
    assert policy.local.pending is None and policy.gripper.pending is None
    assert policy.local.orientation_goal_refresh_commands==0 and policy.gripper.orientation_goal_refresh_commands==0
    assert np.allclose(policy.local.last_orientation_command_rad,0.)
    row=policy.observe_transition(obs,decision,obs)
    assert policy.pending is None and policy.field_nonzero_actions==0 and policy.gripper_actions==0
    assert policy.target_contact.carried_summary(obs.state)['active']
    assert not row['task_success_certified']


def test_actual_opening_aperture_increase_cancels_only_unconfirmed_candidate():
    for active in (False,True):
        pred=prediction(gripper='open');policy=VisualFieldPolicyV13(Client(pred));obs=observation(fingers=(.02,-.02))
        candidate=pred['target_id']
        policy.target_contact.configure(obs.state,pred['target_geometry'],target_id=candidate)
        policy.target_contact.remember_grasp_candidate(obs.state,candidate)
        policy._last_grasp_candidate_id=candidate
        if active:policy.target_contact.confirm_payload_attachment(obs.state,candidate,observed=True)
        decision=policy.act(obs)
        assert decision.diagnostics['selection']['accepted'] and decision.action[6]<0
        after=replace(obs,state=(*obs.state[:6],.023,-.023))
        policy.observe_transition(obs,decision,after)
        carried=policy.target_contact.carried_summary(after.state)
        assert policy._release_commanded
        assert carried['active'] is active
        assert carried['candidate_ids']==([candidate] if active else [])
        assert policy.gripper_actions==1 and policy.pending is None
