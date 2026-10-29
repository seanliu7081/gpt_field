"""Independent V11 planner/robot-envelope/admission/acknowledgement regressions.

Synthetic RGB is a measured test plant. Robot feedback is the only geometric
state supplied to policy; these tests never load a simulator or infer success.
"""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from gpt_field.observation import JOINT_OBSERVATION_KEYS, PolicyObservation, decode_image
from gpt_field.visual_field_policy_v11 import VisualFieldPolicyV11
from gpt_field.visual_field_v8 import measured_current_pose_hold_action
from gpt_field.visual_field_v10 import MetricFieldExecutorV10
from gpt_field.visual_field_v11 import MetricFieldExecutorV11, ToolEnvelopeRiskPlanner
from test_visual_field_v4 import Q, box, state
from test_visual_field_v6 import axis_intent
from test_visual_field_v6_policy import Client, observation
from test_visual_field_v8_policy import value
from test_visual_field_v9 import assessment
from test_visual_field_v10_policy import subgoal


def context(*, neighbor=False, mode='transit', assessment_status=None):
    # Contains the real synthetic tracked tool point (60,75)/223.
    return {'purpose':'task_translation', 'tool_external_roi_uv':[.21,.27,.33,.40],
            'nearby_obstacles':([{'id':'fixture','external_roi_uv':[.25,.30,.34,.40]}] if neighbor else []),
            'obstacle_assessment':assessment_status or ('present' if neighbor else 'none_relevant'),
            'translation_mode':mode, 'intended_contact_id':None}


def prediction(controller='visual_field', *, neighbor=False, mode='transit', **kwargs):
    result=value(controller, **kwargs)
    result['motion_context']=context(neighbor=neighbor,mode=mode)
    if controller=='visual_field' and neighbor:
        # Metric hypotheses predict this named object far from the path. The
        # independent IMAGE conflict must still prevent translation.
        result['field_intent']['obstacles']=[box('fixture',low=(.19,.19,.19),high=(.20,.20,.20))]
    return subgoal(result,'new' if controller=='visual_field' else 'none')


class FixedAdmission:
    """Isolate executor cap/reselection accounting from separately tested RGB."""
    def __init__(self, *, cap=None, accepted=True):
        self.cap, self.accepted, self.requests = cap, accepted, []
    def summary(self):
        return {'conflicting_obstacle_ids':[]}
    def assess(self,delta_xyz,**kwargs):
        self.requests.append((np.array(delta_xyz),kwargs))
        return {'accepted':self.accepted,'cap_m':self.cap,'need_refresh':not self.accepted,
                'reason':'projected_translation_test_cap' if self.accepted else 'translation_projected_neighbor_conflict',
                'diagnostics':{'clearance_certified':False}}


def configured(intent=None, guard=None):
    executor=MetricFieldExecutorV11()
    executor.configure(state(),intent or axis_intent(),scene_assessment=assessment())
    if guard is not None:executor.bind_translation_guard(guard,None)
    return executor


def test_known_finger_geometry_blocks_origin_old_scalar_tool_radius_admitted():
    neighboring=box('finger_side',low=(-.001,.060,-.002),high=(.001,.061,.002))
    intent=axis_intent(obstacles=[neighboring])
    old,new=MetricFieldExecutorV10(),MetricFieldExecutorV11()
    old.configure(state(),intent,scene_assessment=assessment('finger_side'))
    new.configure(state(),intent,scene_assessment=assessment('finger_side'))
    assert old.select(state(),Q)['accepted']
    rejected=new.select(state(),Q)
    assert not rejected['accepted'] and rejected['status']=='invalid_measured_start'
    assert not new.checked.segment_valid(state()[:3],state()[:3])
    geom=new.summary()['region_feasibility'][0]
    assert geom['tool_core']['contains_current']
    assert len(geom['tool_core_parts'])==6  # Five robot parts plus preserved model extent.
    assert any(p['tool_part']=='finger1_collision' and
               np.all(state()[:3]>=p['lower_xyz']) and np.all(state()[:3]<=p['upper_xyz'])
               for p in geom['tool_core_parts'])
    assert new.intent==intent  # No obstacle deletion or origin carve-out.
    assert not new.summary()['robot_tool_envelope']['scene_geometry_certified']


def test_tool_parts_union_does_not_use_large_display_bounds_as_collision_box():
    obstacle=box('between_parts',low=(.019,.019,-.005),high=(.021,.021,-.004))
    executor=MetricFieldExecutorV11()
    executor.configure(state(),axis_intent(obstacles=[obstacle]),scene_assessment=assessment('between_parts'))
    geometry=executor.summary()['region_feasibility'][0]
    union_display=geometry['tool_core']
    # The merged box includes the empty gap between distinct tool components.
    assert np.all(state()[:3]>=union_display['lower_xyz']) and np.all(state()[:3]<=union_display['upper_xyz'])
    assert geometry['union_bounds_are_display_only']
    assert not union_display['contains_current']
    assert executor.checked.segment_valid(state()[:3],state()[:3])


def test_unknown_full_halo_stays_hard_where_visible_uncertainty_is_soft():
    region=box('side',low=(-.001,.092,.005),high=(.001,.094,.007))
    visible=MetricFieldExecutorV11();unknown=MetricFieldExecutorV11()
    visible.configure(state(),axis_intent(obstacles=[region],uncertainty_m=.030),scene_assessment=assessment('side'))
    unknown.configure(state(),axis_intent(unknown_regions=[region],uncertainty_m=.030),scene_assessment=assessment('side'))
    assert visible.checked.segment_valid(state()[:3],state()[:3])
    assert not unknown.checked.segment_valid(state()[:3],state()[:3])
    assert unknown.select(state(),Q)['status']=='invalid_measured_start'
    assert unknown.planner.risk_regions==[] and len(visible.planner.risk_regions)==1
    assert unknown.intent['unknown_regions']==[region]
    assert all(np.all(np.asarray(halo['lower_xyz']) < core['lower_xyz']) and
               np.all(np.asarray(halo['upper_xyz']) > core['upper_xyz'])
               for core,halo in zip(unknown.region_geometry[0]['tool_core_parts'],
                                    unknown.region_geometry[0]['full_halo_parts']))


def test_risk_is_one_contribution_per_scene_object_even_when_robot_parts_overlap():
    region=box('one_object',low=(.04,.04,-.04),high=(.042,.042,-.038))
    executor=MetricFieldExecutorV11()
    executor.configure(state(),axis_intent(obstacles=[region]),scene_assessment=assessment('one_object'))
    risk=executor.planner.risk_regions[0]
    part=risk['parts'][0]
    center=(np.array(part['lower_xyz'])+part['upper_xyz'])/2
    density=executor.planner._density(center,executor.field,executor.compact)
    assert density==pytest.approx(9.)  # Not 1+8*6 for six configuration boxes.
    repeated=copy.deepcopy(risk);repeated['parts']*=3
    same_object=ToolEnvelopeRiskPlanner([], [repeated])
    assert same_object._density(center,None,None)==pytest.approx(9.)
    another=copy.deepcopy(risk);another['id']='second_object'
    two_objects=ToolEnvelopeRiskPlanner([], [risk,another])
    assert two_objects._density(center,None,None)==pytest.approx(17.)
    positions=np.array([center, center+[1.,0.,0.]])
    assert np.all(executor.planner._density(positions,executor.field,executor.compact)>=1.)


def test_v11_cache_key_tracks_tool_parts_and_risk_sigma_independently_of_grid_mask():
    executor=configured()
    key=executor.planner._key(executor.field,executor.goal,.003,executor.compact)
    executor.planner.risk_regions=[{'id':'hypothesis','sigma_m':.01,
                                  'parts':[{'lower_xyz':[0.,0.,0.],'upper_xyz':[.001,.001,.001]}]}]
    first=executor.planner._key(executor.field,executor.goal,.003,executor.compact)
    assert first!=key
    executor.planner.risk_regions[0]['parts'][0]['upper_xyz'][0]+=.0001
    second=executor.planner._key(executor.field,executor.goal,.003,executor.compact)
    assert second!=first
    executor.planner.risk_regions[0]['sigma_m']+=.002
    assert executor.planner._key(executor.field,executor.goal,.003,executor.compact)!=second


@pytest.mark.parametrize('axes,goal,locked',[('xy',[.06,.02,0.],2),('z',[0.,0.,.06],0)])
def test_robot_envelope_rebuild_keeps_original_persistent_axis_corridor(axes,goal,locked):
    executor=MetricFieldExecutorV11();current=state();origin=current[:3].copy();current[locked]+=.002
    executor.configure(current,axis_intent(axes,goal_offset_m=goal),scene_assessment=assessment(),subgoal_origin_xyz=origin)
    outside=current[:3].copy();outside[locked]=origin[locked]+.0031
    assert not executor.checked.segment_valid(current[:3],outside)
    selected=executor.select(current,Q)
    assert selected['accepted'] and abs(selected['action'][locked])<1e-9


def test_guard_cap_reselects_checked_field_command_and_exact_pending_ack_counts_once():
    guard=FixedAdmission(cap=.002);executor=configured(guard=guard)
    selected=executor.select(state(),Q)
    assert selected['accepted']
    assert len(guard.requests)==2
    assert np.linalg.norm(guard.requests[0][0])>.002
    assert np.linalg.norm(np.array(selected['action'][:3])*.05)==pytest.approx(.002)
    assert np.array_equal(executor.pending[1],selected['action'])
    assert executor.orientation_goal_refresh_commands==1
    wrong=np.array(selected['action']);wrong[0]+=.001
    with pytest.raises(ValueError,match="match the selected Field command"):executor.observe(state(),wrong,state())
    assert executor.pending is not None
    after=state();after[:3]+=np.array(selected['action'][:3])*.05*.5
    executor.observe(state(),selected['action'],after)
    assert executor.pending is None and executor.summary()['executed_actions']==1
    assert executor.summary()['effective_moving_actions']==1
    assert executor.summary()['plan_calls']==1


@pytest.mark.parametrize('cap,accepted,status',[(.0009,True,'translation_allowance_below_useful_motion_floor'),
                                               (None,False,'translation_projected_neighbor_conflict')])
def test_guard_veto_discards_pending_unexecuted_command_and_acknowledges_explicit_hold(cap,accepted,status):
    guard=FixedAdmission(cap=cap,accepted=accepted);executor=configured(guard=guard)
    selected=executor.select(state(),Q)
    assert not selected['accepted'] and selected['status']==status
    assert selected['action']==[0.]*7
    assert executor.pending is None and executor.orientation_goal_refresh_commands==0
    assert np.array_equal(executor.last_orientation_command_rad,np.zeros(3))
    executor.observe(state(),measured_current_pose_hold_action(),state())
    assert executor.summary()['executed_actions']==0
    assert executor.select(state(),Q)['status']==status


def test_real_rgb_far_field_reuses_one_model_response_for_more_than_direct_horizon():
    client=Client(prediction(goal=(.06,0.,0.)));policy=VisualFieldPolicyV11(client)
    obs=observation();offset=np.zeros(3)
    for _ in range(18):
        chosen=policy.act(obs)
        assert chosen.diagnostics['selection']['accepted'],chosen.diagnostics['selection']
        assert chosen.diagnostics['selection']['diagnostics']['translation_admission']['accepted']
        offset+=np.asarray(chosen.action[:3])*.05*.25
        after=observation(offset);row=policy.observe_transition(obs,chosen,after);obs=after
    assert client.calls==1 and policy.field_effective_actions==18
    assert offset[0]>.025 and policy.translation_vetoes==0
    assert not row['task_success_certified'] and not row['translation_guard']['clearance_certified']
    assert policy.local.pending is None and policy.pending is None


def test_current_image_overlap_vetoes_metric_clear_field_before_any_motion_and_preserves_goal():
    initial=prediction(neighbor=True);client=Client(initial);policy=VisualFieldPolicyV11(client);obs=observation()
    chosen=policy.act(obs)
    assert policy.local.checked.segment_valid(np.array(obs.state[:3]),policy.goal)
    assert chosen.diagnostics['selection']['status']=='translation_projected_neighbor_conflict'
    assert not chosen.diagnostics['selection']['accepted']
    assert list(chosen.action)==measured_current_pose_hold_action()
    assert policy.local.pending is None and policy.subgoals.active is not None
    goal=copy.deepcopy(policy.subgoals.active['world_goal_xyz'])
    row=policy.observe_transition(obs,chosen,obs)
    assert policy.pending is None and policy.field_nonzero_actions==0
    assert policy.subgoals.active['world_goal_xyz']==goal
    assert policy.translation_vetoes==1 and not row['task_success_certified']
    assert policy.prediction==initial and client.value==initial


def test_direct_accepted_five_commands_are_byte_for_byte_model_values_with_no_field_or_auto_grasp():
    commands=[[.01+.005*i,-.01,.01,0.,.001*i,0.,-1.]for i in range(5)]
    initial=prediction('direct',phase='grasp',actions=commands)
    client=Client(initial);policy=VisualFieldPolicyV11(client);obs=observation()
    for command in commands:
        chosen=policy.act(obs)
        assert list(chosen.action)==command
        assert chosen.diagnostics['translation_admission']['accepted']
        policy.observe_transition(obs,chosen,obs)
    assert client.calls==1 and policy.field_nonzero_actions==0 and policy.gripper_actions==0
    assert policy.translation_vetoes==0 and policy.subgoals.active is None
    assert policy.prediction==initial and client.value==initial


def test_direct_overlap_veto_discards_entire_chunk_instead_of_clipping_and_has_valid_hold_ack():
    commands=[[.08,0.,0.,0.,0.,0.,1.]]*5
    initial=prediction('direct',neighbor=True,phase='grasp',actions=commands)
    client=Client(initial);policy=VisualFieldPolicyV11(client);obs=observation()
    rejected=policy.act(obs)
    assert rejected.diagnostics['selection']['status']=='translation_projected_neighbor_conflict'
    assert rejected.diagnostics['rejected_model_action']==commands[0]
    assert list(rejected.action)==measured_current_pose_hold_action()
    assert policy.chunk==[] and policy.pending[1] is rejected
    policy.observe_transition(obs,rejected,obs)
    assert policy.pending is None and policy.translation_vetoes==1 and policy.gripper_actions==0
    assert policy.refresh_reason=='translation_projected_neighbor_conflict'
    # Removing a named projected obstruction on refresh must not erase it.
    client.value=prediction('direct',phase='recover',retain=True,actions=[[.01,0.,0.,0.,0.,0.,0.]]*5)
    repeated=policy.act(obs)
    assert not repeated.diagnostics['selection']['accepted']
    assert list(repeated.action)==measured_current_pose_hold_action()
    assert len(policy.translation_guard.summary()['neighbors'])==1
    policy.observe_transition(obs,repeated,obs)


def test_direct_guard_cap_is_veto_not_silent_action_scaling(monkeypatch):
    command=[.08,0.,0.,0.,0.,0.,0.]
    policy=VisualFieldPolicyV11(Client(prediction('direct',phase='align',actions=[command]*5)))
    gate=FixedAdmission(cap=.002)
    monkeypatch.setattr(policy.translation_guard,'assess',gate.assess)
    obs=observation();chosen=policy.act(obs)
    assert chosen.diagnostics['selection']['status']=='direct_translation_exceeds_observed_allowance'
    assert chosen.diagnostics['rejected_model_action']==command
    assert list(chosen.action)==measured_current_pose_hold_action() and not policy.chunk
    policy.observe_transition(obs,chosen,obs)
    assert policy.pending is None and policy.translation_vetoes==1


def test_gripper_settling_remains_explicit_17_measured_actions_and_not_attachment_proof():
    initial=prediction('gripper',phase='grasp');client=Client(initial)
    policy=VisualFieldPolicyV11(client);obs=observation()
    for index in range(17):
        chosen=policy.act(obs)
        assert chosen.diagnostics['selection']['accepted'] and chosen.action[6]==1.
        assert np.linalg.norm(chosen.action[:3])==0.
        gap=max(.012,.04-.002*(index+1))
        after=replace(obs,state=(*obs.state[:6],gap,-gap))
        row=policy.observe_transition(obs,chosen,after);obs=after
    assert policy.refresh_reason=='gripper_settled' and policy.gripper_actions==17
    assert client.calls==1 and policy.field_nonzero_actions==0 and not policy.monitor.attachment_evidence
    assert not row['gripper_execution']['grasp_certified']


def test_unobservable_context_cannot_turn_existing_caps_into_permission_or_erase_geometry():
    initial=prediction(goal=(.06,0.,0.));initial['motion_context']['obstacle_assessment']='unobservable'
    policy=VisualFieldPolicyV11(Client(initial));obs=observation();chosen=policy.act(obs)
    assert not chosen.diagnostics['selection']['accepted']
    assert chosen.diagnostics['selection']['status']=='translation_neighborhood_unobservable_or_inconsistent'
    assert policy.local.intent['max_travel_m']==.015 and policy.local.intent['command_limit_m']==.005
    assert policy.local.goal==pytest.approx(np.asarray(obs.state[:3])+[.06,0.,0.])
    assert not policy.local.summary()['soft_uncertainty_enabled']
    assert list(chosen.action)==measured_current_pose_hold_action()
    policy.observe_transition(obs,chosen,obs)
    assert policy.pending is None and policy.field_nonzero_actions==0


def test_policy_robot_geometry_guard_and_ledger_never_access_environment_truth():
    class RecordingClient(Client):
        def predict(self,obs):
            self.last_observation=copy.deepcopy(obs)
            return super().predict(obs)
    class Poison:
        def __array__(self,*args,**kwargs):raise AssertionError('Hidden environment input accessed')
    base=observation()
    raw={'robot0_eef_pos':base.state[:3],'robot0_eef_quat':[0.,0.,0.,1.],
         'robot0_gripper_qpos':base.state[6:],'robot0_joint_pos':base.joint_position,
         'robot0_joint_vel':base.joint_velocity,'agentview_image':np.asarray(decode_image(base.image)),
         'robot0_eye_in_hand_image':np.asarray(decode_image(base.wrist_image)),
         'object_poses':Poison(),'depth':Poison(),'camera_calibration':Poison(),
         'contact_positions':Poison(),'libero_success':True,'reward':1.}
    allowed=PolicyObservation.from_libero(raw,base.prompt,include_arm_joints=True)
    # from_libero rotates images, so use Direct hold to isolate the input boundary
    # from irrelevant reference-seeding changes in this fixture.
    pred=prediction('direct',phase='verify',actions=[[0.]*7]*5)
    pred['visual_references']={'external':None,'wrist':None}
    client=RecordingClient(pred);policy=VisualFieldPolicyV11(client)
    chosen=policy.act(allowed);row=policy.observe_transition(allowed,chosen,allowed)
    assert set(client.last_observation)==JOINT_OBSERVATION_KEYS
    text=json.dumps(client.observable_context)
    assert not any('"'+key+'"' in text for key in ('object_poses','depth','camera_calibration','contact_positions','libero_success','reward'))
    assert not row['task_success_certified'] and not row['subgoal_ledger']['task_success_certified']
    assert not row['translation_guard']['clearance_certified']
