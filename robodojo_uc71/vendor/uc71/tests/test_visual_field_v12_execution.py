"""Cross-layer V12 tests: command units, measured budgets and real RGB feedback.

These are offline test-plant observations and exact command acknowledgements,
not simulator outcomes or measured task-speed results.
"""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from gpt_field.observation import JOINT_OBSERVATION_KEYS, PolicyObservation, decode_image
from gpt_field.robot_translation_response import MeasuredTranslationResponse
from gpt_field.visual_field_policy_v12 import VisualFieldPolicyV12
from gpt_field.visual_field_v8 import measured_current_pose_hold_action
from gpt_field.visual_field_v12 import MetricFieldExecutorV12
from test_visual_field_v4 import Q, state
from test_visual_field_v6 import axis_intent
from test_visual_field_v6_policy import Client, observation
from test_visual_field_v9 import assessment
from test_visual_field_v11_execution import FixedAdmission, prediction


def calibrated_response():
    response=MeasuredTranslationResponse();current=state();current[0]-=.0022
    for _ in range(5):
        after=current.copy();after[0]+=.00044
        response.observe(current,[.04,0.,0.,0.,0.,0.,0.],after,before_joints=Q,after_joints=Q)
        current=after
    return response,current


def executor(*,guard=None,intent=None,response=None,current=None):
    if response is None:response,current=calibrated_response()
    field=MetricFieldExecutorV12()
    field.bind_translation_guard(guard,None,response)
    field.configure(current,intent or axis_intent(),scene_assessment=assessment())
    return field,response,current


def test_submillimeter_measured_allowance_supports_useful_command_via_observed_robot_response():
    field,response,current=executor(guard=FixedAdmission(),intent=axis_intent(command_limit_m=.002))
    chosen=field.select(current,Q,measured_cap_m=.00065)
    assert chosen['accepted']
    command=np.asarray(chosen['action'][:3])*.05
    assert np.linalg.norm(command)==pytest.approx(.002)
    converted=chosen['diagnostics']['measured_control_allowance']['conversion']
    assert converted['supported'] and converted['prediction']['max_measured_norm_m']<=.00065
    assert converted['command_cap_m']>.001  # Meaningful COMMAND despite <1mm MEASURED allowance.
    assert field.pending[1]==pytest.approx(chosen['action'])
    after=current.copy();after[:3]+=command*.22
    field.observe(current,chosen['action'],after)
    assert field.summary()['executed_actions']==1 and field.pending is None
    assert field.summary()['measured_travel_m']==pytest.approx(.00044)
    assert not field.summary()['clearance_certified']


def test_measured_conversion_then_guard_cap_reselects_exact_final_command_without_double_counting():
    gate=FixedAdmission(cap=.0011);field,response,current=executor(guard=gate)
    chosen=field.select(current,Q,measured_cap_m=.0004)
    assert chosen['accepted']
    assert len(gate.requests)==2
    assert .0011<np.linalg.norm(gate.requests[0][0])<.002
    assert np.linalg.norm(gate.requests[1][0])==pytest.approx(.0011)
    assert np.linalg.norm(np.asarray(chosen['action'][:3])*.05)==pytest.approx(.0011)
    assert field.orientation_goal_refresh_commands==1
    wrong=chosen['action'].copy();wrong[0]+=.001
    with pytest.raises(ValueError,match='match the selected Field command'):
        field.observe(current,wrong,current)
    assert field.pending is not None
    after=current.copy();after[0]+=.000242
    field.observe(current,chosen['action'],after)
    assert field.pending is None and field.summary()['executed_actions']==1
    assert field.summary()['plan_calls']==1


def test_remaining_actual_budget_too_small_for_any_supported_useful_command_is_hold_not_round_up():
    field,response,current=executor(guard=FixedAdmission())
    rejected=field.select(current,Q,measured_cap_m=.0001)
    assert not rejected['accepted']
    assert rejected['status']=='measured_allowance_has_no_useful_supported_command'
    assert rejected['action']==[0.]*7 and field.pending is None
    assert field.orientation_goal_refresh_commands==0
    field.observe(current,measured_current_pose_hold_action(),current)
    assert field.summary()['executed_actions']==0


def test_reselection_limit_discards_every_provisional_command_and_acknowledges_only_hold():
    class ShrinkingGuard(FixedAdmission):
        def assess(self,delta_xyz,**kwargs):
            result=super().assess(delta_xyz,**kwargs)
            result['cap_m']=float(np.linalg.norm(delta_xyz))*.8
            return result
    gate=ShrinkingGuard();field,response,current=executor(guard=gate)
    rejected=field.select(current,Q)
    assert not rejected['accepted'] and rejected['status']=='translation_supported_scale_requires_replan'
    assert len(gate.requests)==3
    assert field.pending is None and field.orientation_goal_refresh_commands==0
    assert field.last_orientation_command_rad==pytest.approx(np.zeros(3))
    field.observe(current,measured_current_pose_hold_action(),current)
    assert field.summary()['executed_actions']==0


def test_pending_accepted_action_cannot_be_replaced_before_exact_observation_ack():
    field,response,current=executor(guard=FixedAdmission())
    selected=field.select(current,Q)
    with pytest.raises(RuntimeError,match='acknowledge'):
        field.select(current,Q,measured_cap_m=.00065)
    assert field.pending[1]==pytest.approx(selected['action'])
    field.observe(current,selected['action'],current)
    assert field.summary()['executed_actions']==1


def test_nominal_segment_distance_accounts_for_entire_segment_and_known_tool_parts():
    pred=prediction(neighbor=True)
    obs=observation();field=MetricFieldExecutorV12()
    field.configure(obs.state,pred['field_intent'],scene_assessment=pred['motion_context'])
    records=field.nominal_regions_for_segment(np.asarray(obs.state[:3]),field.goal)
    assert len(records)==1 and records[0]['id']=='fixture'
    geometry=field.region_geometry[0]
    part=geometry['tool_core_parts'][0]
    center=(np.asarray(part['lower_xyz'])+part['upper_xyz'])/2
    # Both endpoints are outside one core, but the segment crosses its center.
    crossing=field.nominal_regions_for_segment(center-[.3,0.,0.],center+[.3,0.,0.])
    assert crossing[0]['core_distance_m']==0.
    assert crossing[0]['geometry_is_model_hypothesis']


def test_one_real_rgb_field_response_is_reused_while_robot_response_learns_and_commands_grow():
    initial=prediction(goal=(.06,0.,0.));client=Client(initial);policy=VisualFieldPolicyV12(client)
    obs=observation();offset=np.zeros(3);norms=[];rows=[]
    for _ in range(55):
        chosen=policy.act(obs)
        if not chosen.diagnostics['selection']['accepted']:
            assert chosen.diagnostics['selection']['status'] in ('measured_local_goal_arrival','subgoal_local_complete')
            policy.observe_transition(obs,chosen,obs)
            break
        delta=np.asarray(chosen.action[:3])*.05;norms.append(float(np.linalg.norm(delta)))
        assert np.array_equal(policy.local.pending[1],chosen.action)
        offset+=delta*.22
        after=observation(offset);rows.append(policy.observe_transition(obs,chosen,after));obs=after
        if policy.refresh_reason:
            assert policy.refresh_reason in ('measured_local_goal_arrival','subgoal_local_complete')
            break
    assert client.calls==1 and policy.field_effective_actions>20
    assert offset[0]>.04 and norms[0]<=.002+1e-12
    assert max(norms)>.0044  # Calibrated growth, not fixed 2mm cold commands forever.
    assert policy.robot_response.transitions==len(norms)
    fits=[r['robot_translation_response']['last_observation'] for r in rows]
    assert policy.robot_response.nonzero_samples==sum(r['sample_added'] for r in fits)
    # Crossing the 4cm local robot-response domain is real progress, but that
    # boundary transition cannot become a sample for the new response fit.
    excluded=[r for r in fits if not r['sample_added']]
    assert len(excluded)==1 and not excluded[0]['is_hold']
    assert excluded[0]['domain_reset_reason']=='robot_response_position_domain_changed'
    assert policy.robot_response.domain_resets==1
    assert any(r['visual']['cumulative_response']['supported']for r in rows)
    assert not rows[-1]['task_success_certified']
    assert not rows[-1]['robot_translation_response']['clearance_certified']
    assert policy.prediction==initial and client.value==initial


def test_direct_no_neighbor_observation_executes_exact_five_small_model_actions_and_samples_once():
    commands=[[.02+.004*i,-.005,0.,0.,0.,0.,0.]for i in range(5)]
    initial=prediction('direct',phase='align',actions=commands)
    client=Client(initial);policy=VisualFieldPolicyV12(client);obs=observation();offset=np.zeros(3)
    for index,command in enumerate(commands):
        chosen=policy.act(obs)
        assert list(chosen.action)==command,chosen.diagnostics.get('selection')
        assert chosen.diagnostics['translation_admission']['accepted']
        offset+=np.array(command[:3])*.05*.22
        after=observation(offset);row=policy.observe_transition(obs,chosen,after);obs=after
        assert policy.robot_response.transitions==index+1
    assert client.calls==1 and policy.field_nonzero_actions==0 and policy.gripper_actions==0
    assert policy.robot_response.nonzero_samples==5 and policy.subgoals.active is None
    assert policy.prediction==initial and not row['task_success_certified']


def test_ambiguous_direct_does_not_borrow_stale_field_clearance_or_silently_clip():
    commands=[[.08,0.,0.,0.,0.,0.,1.]]*5
    initial=prediction('direct',neighbor=True,phase='grasp',actions=commands)
    client=Client(initial);policy=VisualFieldPolicyV12(client);obs=observation()
    chosen=policy.act(obs)
    assert not chosen.diagnostics['selection']['accepted']
    assert chosen.diagnostics['rejected_model_action']==commands[0]
    assert list(chosen.action)==measured_current_pose_hold_action() and policy.chunk==[]
    policy.observe_transition(obs,chosen,obs)
    assert policy.robot_response.transitions==1 and policy.robot_response.nonzero_samples==0
    assert policy.pending is None and policy.local.pending is None
    assert policy.field_nonzero_actions==0 and policy.gripper_actions==0


def test_rejected_field_hold_does_not_train_robot_as_successful_motion_or_delete_subgoal(monkeypatch):
    initial=prediction(goal=(.06,0.,0.));policy=VisualFieldPolicyV12(Client(initial));obs=observation()
    monkeypatch.setattr(policy.monitor,'measured_control_limit',lambda:(.0001,None))
    rejected=policy.act(obs)
    assert not rejected.diagnostics['selection']['accepted']
    assert list(rejected.action)==measured_current_pose_hold_action()
    goal=copy.deepcopy(policy.subgoals.active['world_goal_xyz'])
    row=policy.observe_transition(obs,rejected,obs)
    assert policy.robot_response.transitions==1 and policy.robot_response.nonzero_samples==0
    assert policy.robot_response.holds==1 and policy.field_nonzero_actions==0
    assert policy.subgoals.active['world_goal_xyz']==goal
    assert policy.pending is None and policy.local.pending is None
    assert not row['task_success_certified']


def test_explicit_gripper_hold_transitions_do_not_create_translation_fit_or_grasp_certificate():
    policy=VisualFieldPolicyV12(Client(prediction('gripper',phase='grasp')));obs=observation()
    for index in range(17):
        chosen=policy.act(obs)
        assert chosen.diagnostics['selection']['accepted'] and chosen.action[6]==1.
        gap=max(.012,.04-.002*(index+1))
        after=replace(obs,state=(*obs.state[:6],gap,-gap))
        row=policy.observe_transition(obs,chosen,after);obs=after
    assert policy.gripper_actions==17 and policy.refresh_reason=='gripper_settled'
    assert policy.robot_response.transitions==17 and policy.robot_response.nonzero_samples==0
    assert len(policy.robot_response.samples)==0
    assert not row['gripper_execution']['grasp_certified'] and not policy.monitor.attachment_evidence


def test_policy_response_and_guards_receive_robot_and_rgb_only_not_evaluator_or_environment_truth():
    class RecordingClient(Client):
        def predict(self,obs):self.last_observation=copy.deepcopy(obs);return super().predict(obs)
    class Poison:
        def __array__(self,*args,**kwargs):raise AssertionError('Privileged environment information accessed')
    base=observation()
    raw={'robot0_eef_pos':base.state[:3],'robot0_eef_quat':[0.,0.,0.,1.],
         'robot0_gripper_qpos':base.state[6:],'robot0_joint_pos':base.joint_position,
         'robot0_joint_vel':base.joint_velocity,'agentview_image':np.asarray(decode_image(base.image)),
         'robot0_eye_in_hand_image':np.asarray(decode_image(base.wrist_image)),
         'object_poses':Poison(),'depth':Poison(),'camera_calibration':Poison(),'contacts':Poison(),
         'success':True,'libero_success':True,'reward':1.}
    allowed=PolicyObservation.from_libero(raw,base.prompt,include_arm_joints=True)
    pred=prediction('direct',phase='verify',actions=[[0.]*7]*5)
    pred['visual_references']={'external':None,'wrist':None}
    client=RecordingClient(pred);policy=VisualFieldPolicyV12(client)
    chosen=policy.act(allowed);row=policy.observe_transition(allowed,chosen,allowed)
    assert set(client.last_observation)==JOINT_OBSERVATION_KEYS
    text=json.dumps(client.observable_context)
    assert not any('"'+key+'"' in text for key in ('object_poses','depth','camera_calibration','contacts','success','libero_success','reward'))
    assert row['robot_translation_response']['environment_inputs']==[]
    assert not row['task_success_certified'] and not row['translation_guard']['clearance_certified']
