"""Cross-controller V10 tests using RGB textures and robot feedback only."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.observation import JOINT_OBSERVATION_KEYS, PolicyObservation, decode_image
from gpt_field.visual_field_policy_v10 import VisualFieldPolicyV10
from gpt_field.visual_field_v8 import measured_current_pose_hold_action
from test_visual_field_v6_policy import Client, observation
from test_visual_field_v8_policy import value
from test_visual_field_v9_policy import uncertain_neighbor


def subgoal(pred=None, action='new', ident='local-goal', reason='Current views support this local goal'):
    result=copy.deepcopy(pred if pred is not None else value())
    result['subgoal_update']={'action':action,'subgoal_id':None if action=='none' else ident,'reason':reason}
    return result


def advance(policy,obs,offset,*,gain=.25):
    decision=policy.act(obs)
    assert decision.diagnostics['selection']['accepted'], decision.diagnostics['selection']
    offset=offset+np.asarray(decision.action[:3])*.05*gain
    after=observation(offset)
    row=policy.observe_transition(obs,decision,after)
    return after,offset,decision,row


def direct_repair(policy,client,obs,offset,*,commands=None,actual_delta=None):
    commands=commands or [[0.,0.,.08,0.,0.,0.,0.]]*5
    repair=value('direct',phase='grasp',retain=True,actions=commands)
    repair['target_id']='payload_being_repaired'
    client.value=subgoal(repair,'retain')
    policy.refresh_reason='offline_requested_reference_repair'
    for command in commands:
        decision=policy.act(obs)
        assert list(decision.action)==command
        offset=offset+(np.asarray(command[:3])*.05*.25 if actual_delta is None else actual_delta)
        after=observation(offset)
        policy.observe_transition(obs,decision,after)
        obs=after
    return obs,offset


def test_field_direct_repair_field_keeps_accepted_world_goal_and_reports_fresh_proposal_separately():
    initial=subgoal(value(goal=(0.,0.,.05),motion_axes='z',phase='transport'))
    client=Client(initial);policy=VisualFieldPolicyV10(client)
    obs,offset=observation(),np.zeros(3)
    obs,offset,first,_=advance(policy,obs,offset)
    stored=policy.goal.copy()
    obs,offset=direct_repair(policy,client,obs,offset)
    client.value=subgoal(value(goal=(0.,0.,.05),motion_axes='z',phase='transport',retain=True),'retain')
    decision=policy.act(obs)
    assert decision.diagnostics['selection']['accepted']
    assert policy.goal==pytest.approx(stored)
    assert policy.local.intent['goal_offset_m'][2]==pytest.approx(stored[2]-obs.state[2])
    assert decision.diagnostics['prediction']['field_intent']['goal_offset_m']==[0.,0.,.05]
    update=decision.diagnostics['subgoal_update']
    assert update['feedback']['last_update']['proposed_world_goal_xyz'][2]>stored[2]
    assert update['feedback']['active']['world_goal_xyz']==pytest.approx(stored)
    assert policy.monitor.reference_update_status=='external_retained'
    row=policy.observe_transition(obs,decision,obs)
    assert client.calls==3 and policy.field_nonzero_actions==2
    assert row['subgoal_ledger']['active']['phase']=='transport'
    assert row['subgoal_ledger']['active']['target_id']=='destination'
    assert not row['task_success_certified']


def test_direct_measured_motion_finishes_old_goal_before_resume_preventing_re_lift():
    client=Client(subgoal(value(goal=(0.,0.,.025),motion_axes='z',phase='transport')))
    policy=VisualFieldPolicyV10(client);obs,offset=observation(),np.zeros(3)
    obs,offset,_,_=advance(policy,obs,offset)
    obs,offset=direct_repair(policy,client,obs,offset,commands=[[0.,0.,.12,0.,0.,0.,0.]]*5,
                             actual_delta=np.array([0.,0.,.006]))
    assert policy.subgoals.summary()['local_complete']
    client.value=subgoal(value(goal=(0.,0.,.025),motion_axes='z',phase='transport',retain=True),'retain')
    resumed=policy.act(obs)
    assert resumed.diagnostics['selection']['status']=='subgoal_local_complete'
    assert tuple(resumed.action)==tuple(measured_current_pose_hold_action())
    assert not resumed.diagnostics['selection']['accepted'] and not policy.field_configured
    row=policy.observe_transition(obs,resumed,obs)
    assert row['subgoal_ledger']['local_complete'] and not row['task_success_certified']
    assert policy.field_nonzero_actions==1 and policy.pending is None
    # Finishing this local goal permits a separately named next-stage goal.
    client.value=subgoal(value('direct',phase='align',actions=[[0.]*7]*5),'finish')
    finished=policy.act(obs);policy.observe_transition(obs,finished,obs)
    assert policy.subgoals.active is None


@pytest.mark.parametrize('controller',['visual_field','direct','gripper'])
def test_missing_ledger_retain_is_explicit_hold_and_can_be_acknowledged_before_valid_recovery(controller):
    pred=value(controller,phase='grasp' if controller!='visual_field' else 'approach')
    pred['visual_references']={'external':None,'wrist':None}
    client=Client(subgoal(pred,'retain'));policy=VisualFieldPolicyV10(client);obs=observation()
    rejected=policy.act(obs)
    assert rejected.diagnostics['selection']['status']=='subgoal_missing'
    assert tuple(rejected.action)==tuple(measured_current_pose_hold_action())
    assert rejected.diagnostics['rejection_hold']['requested_motion_executed'] is False
    row=policy.observe_transition(obs,rejected,obs)
    assert policy.pending is None and policy.gripper_actions==0 and policy.field_nonzero_actions==0
    assert not row['task_success_certified']
    recovery=value('direct',phase='grasp',actions=[[.04,0.,0.,0.,0.,0.,0.]]*5)
    recovery['visual_references']={'external':None,'wrist':None}
    client.value=subgoal(recovery,'none')
    chosen=policy.act(obs);assert chosen.action[0]==.04
    policy.observe_transition(obs,chosen,obs)
    assert client.calls==2


def test_rejected_gripper_after_previous_valid_block_does_not_acknowledge_stale_executor():
    pred=value('gripper',phase='grasp');pred['visual_references']={'external':None,'wrist':None}
    client=Client(subgoal(pred,'none'));policy=VisualFieldPolicyV10(client);obs=observation()
    for index in range(17):
        chosen=policy.act(obs)
        assert chosen.action[6]==1.
        gap=max(.012,.04-.002*(index+1))
        after=replace(obs,state=(*obs.state[:6],gap,-gap))
        policy.observe_transition(obs,chosen,after);obs=after
    assert policy.refresh_reason=='gripper_settled' and policy.gripper_actions==17
    client.value=subgoal(pred,'retain',ident='nonexistent')
    rejected=policy.act(obs);assert rejected.diagnostics['selection']['status']=='subgoal_missing'
    policy.observe_transition(obs,rejected,obs)
    assert policy.gripper_actions==17 and policy.pending is None
    assert not policy.monitor.attachment_evidence


def test_accepted_direct_five_commands_are_exact_and_do_not_auto_create_field_or_gripper():
    commands=[[.01*i,-.02,.01,0.,.01*i,0.,-1.]for i in range(5)]
    pred=value('direct',phase='grasp',actions=commands);pred['visual_references']={'external':None,'wrist':None}
    original=subgoal(pred,'none');client=Client(original);policy=VisualFieldPolicyV10(client);obs=observation()
    for command in commands:
        decision=policy.act(obs)
        assert list(decision.action)==command
        policy.observe_transition(obs,decision,obs)
    assert client.calls==1 and policy.gripper_actions==0 and policy.field_nonzero_actions==0
    assert policy.subgoals.active is None and policy.prediction==original and client.value==original


def test_visible_uncertainty_cost_preserves_boxes_and_reuses_field_under_real_rgb_motion():
    pred=subgoal(uncertain_neighbor());original=copy.deepcopy(pred)
    client=Client(pred);policy=VisualFieldPolicyV10(client);obs,offset=observation(),np.zeros(3)
    for _ in range(25):
        obs,offset,chosen,row=advance(policy,obs,offset)
        assert chosen.diagnostics['selection']['diagnostics']['soft_uncertainty_enabled']
        if policy.refresh_reason:break
    assert client.calls==1 and policy.field_effective_actions>15 and np.linalg.norm(offset)>.03
    assert policy.local.intent['obstacles']==original['field_intent']['obstacles']
    assert policy.prediction==original and client.value==original
    assert not row['task_success_certified']


@pytest.mark.parametrize('unknown,assessment',[(True,'present'),(False,'unobservable')])
def test_v9_unknown_and_unobservable_halos_remain_hard(unknown,assessment):
    pred=subgoal(uncertain_neighbor(unknown=unknown,assessment=assessment))
    policy=VisualFieldPolicyV10(Client(pred));obs=observation()
    rejected=policy.act(obs)
    assert rejected.diagnostics['selection']['status']=='invalid_measured_start'
    assert tuple(rejected.action)==tuple(measured_current_pose_hold_action())
    policy.observe_transition(obs,rejected,obs)
    assert policy.subgoals.active is not None  # rejection cannot erase the goal
    assert policy.local.intent['unknown_regions']==pred['field_intent']['unknown_regions']
    if assessment=='unobservable':
        assert policy.local.intent['command_limit_m']<=.005
        assert policy.local.intent['max_travel_m']<=.015


def test_unobservable_scene_caps_cannot_be_lifted_by_valid_feature_motion():
    pred=value(goal=(.08,0.,0.));pred['motion_context']['obstacle_assessment']='unobservable'
    original=subgoal(pred);client=Client(original);policy=VisualFieldPolicyV10(client)
    obs,offset=observation(),np.zeros(3)
    for _ in range(45):
        decision=policy.act(obs)
        assert np.linalg.norm(decision.action[:3])*.05<=.005+1e-12
        offset+=np.asarray(decision.action[:3])*.05*.25
        after=observation(offset);row=policy.observe_transition(obs,decision,after);obs=after
        if policy.refresh_reason:break
    assert client.calls==1 and np.linalg.norm(offset)<=.015+1e-9
    assert policy.refresh_reason=='field_command_below_useful_motion_floor'
    assert policy.local.intent['max_travel_m']==.015 and policy.local.intent['command_limit_m']==.005
    assert policy.local.intent['goal_offset_m']==original['field_intent']['goal_offset_m']
    assert row['scene_execution_limits']['goal_and_regions_preserved']
    assert policy.prediction==original


def test_accepted_local_planar_correction_is_recorded_and_survives_direct_repair(monkeypatch):
    client=Client(subgoal(value(goal=(.06,0.,0.),motion_axes='xy',phase='align')))
    policy=VisualFieldPolicyV10(client);obs,offset=observation(),np.zeros(3)
    obs,offset,_,_=advance(policy,obs,offset)
    original=policy.goal.copy();correction=original+np.array([0.,.002,0.])
    # Isolate the handoff from the separately tested RGB estimator: this fixture
    # supplies its already-validated, bounded correction; real planner and
    # ledger must both accept and persist it before any action is selected.
    monkeypatch.setattr(policy.monitor,'planar_correction',lambda goal:correction.copy())
    policy.last_correction_step=-5
    chosen=policy.act(obs)
    assert chosen.diagnostics['selection']['accepted']
    assert policy.goal==pytest.approx(correction)
    assert policy.subgoals.active['world_goal_xyz']==pytest.approx(correction)
    assert policy.subgoals.active['visual_correction_count']==1
    policy.observe_transition(obs,chosen,obs)
    monkeypatch.setattr(policy.monitor,'planar_correction',lambda goal:None)
    obs,offset=direct_repair(policy,client,obs,offset,commands=[[.04,0.,0.,0.,0.,0.,0.]]*5)
    client.value=subgoal(value(goal=(.06,0.,0.),motion_axes='xy',phase='align',retain=True),'retain')
    resumed=policy.act(obs)
    assert resumed.diagnostics['selection']['accepted']
    assert policy.goal==pytest.approx(correction)
    policy.observe_transition(obs,resumed,obs)


def test_failed_ledger_correction_after_successful_local_retarget_executes_only_hold(monkeypatch):
    client=Client(subgoal(value(goal=(.06,0.,0.),motion_axes='xy',phase='align')))
    policy=VisualFieldPolicyV10(client);obs,offset=observation(),np.zeros(3)
    obs,offset,_,_=advance(policy,obs,offset)
    stored=policy.goal.copy();proposal=stored+np.array([0.,.002,0.])
    monkeypatch.setattr(policy.monitor,'planar_correction',lambda goal:proposal.copy())
    monkeypatch.setattr(policy.subgoals,'record_visual_correction',lambda goal:False)
    policy.last_correction_step=-5
    rejected=policy.act(obs)
    assert rejected.diagnostics['selection']['status']=='subgoal_visual_correction_requires_explicit_replan'
    assert policy.local.goal==pytest.approx(stored)  # rejected change was rolled back
    assert policy.goal==pytest.approx(stored) and policy.subgoals.active['world_goal_xyz']==pytest.approx(stored)
    assert tuple(rejected.action)==tuple(measured_current_pose_hold_action())
    policy.observe_transition(obs,rejected,obs)
    assert policy.pending is None and policy.local.pending is None
    assert policy.field_nonzero_actions==1 and policy.refresh_reason=='subgoal_visual_correction_requires_explicit_replan'


def test_policy_and_ledger_never_receive_evaluator_or_environment_truth():
    class RecordingClient(Client):
        def predict(self,obs):
            self.last_observation=copy.deepcopy(obs)
            return super().predict(obs)
    base=observation();image=np.asarray(decode_image(base.image));wrist=np.asarray(decode_image(base.wrist_image))
    class Poison:
        def __array__(self,*args,**kwargs):raise AssertionError('Hidden environmental state was accessed')
    simulator_observation={'robot0_eef_pos':base.state[:3],'robot0_eef_quat':[0.,0.,0.,1.],
        'robot0_gripper_qpos':base.state[6:],'robot0_joint_pos':base.joint_position,
        'robot0_joint_vel':base.joint_velocity,'agentview_image':image,'robot0_eye_in_hand_image':wrist,
        'object_poses':Poison(),'depth':Poison(),'camera_calibration':Poison(),'libero_success':True,'reward':1.}
    allowed=PolicyObservation.from_libero(simulator_observation,base.prompt,include_arm_joints=True)
    pred=value('direct',phase='grasp');pred['visual_references']={'external':None,'wrist':None}
    client=RecordingClient(subgoal(pred,'none'));policy=VisualFieldPolicyV10(client)
    decision=policy.act(allowed);row=policy.observe_transition(allowed,decision,allowed)
    assert set(client.last_observation)==JOINT_OBSERVATION_KEYS
    def keys(value):
        if isinstance(value,dict):
            return set(value).union(*(keys(v)for v in value.values()))
        if isinstance(value,list):return set().union(*(keys(v)for v in value))
        return set()
    assert not {'object_poses','depth','camera_calibration','libero_success','reward'}&keys(client.observable_context)
    assert not row['task_success_certified'] and not row['subgoal_ledger']['task_success_certified']


def test_measured_field_crossing_goal_schedules_local_completion_without_a_reverse_command():
    client=Client(subgoal(value(goal=(.007,0.,0.),motion_axes='xy')))
    policy=VisualFieldPolicyV10(client);obs=observation()
    decision=policy.act(obs);assert decision.diagnostics['selection']['accepted']
    # An endpoint farther than tolerance beyond the goal still crosses the
    # region during this measured transition; old executor endpoint-distance
    # logic alone would request a return motion on the next selection.
    # A weak RGB projection keeps this test within the real tracker's feature
    # footprint; the larger proprioceptive displacement is the arrival signal.
    # No depth/calibration or success flag is supplied to the policy.
    after=replace(observation((.003,0.,0.)),state=observation((.013,0.,0.)).state)
    row=policy.observe_transition(obs,decision,after)
    assert row['subgoal_ledger']['local_complete']
    assert np.linalg.norm(policy.local.goal-np.asarray(after.state[:3]))>policy.local.intent['goal_tolerance_m']
    assert policy.refresh_reason=='subgoal_local_complete'
    client.value=subgoal(value(goal=(.007,0.,0.),motion_axes='xy',retain=True),'retain')
    following=policy.act(after)
    assert following.diagnostics['selection']['status']=='subgoal_local_complete'
    assert tuple(following.action)==tuple(measured_current_pose_hold_action())
    policy.observe_transition(after,following,after)
    assert client.calls==2 and policy.field_nonzero_actions==1
