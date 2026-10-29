"""Model target hypotheses versus known robot parts, without simulator truth."""
import copy
import math

import numpy as np
import pytest

from gpt_field.contact_geometry_v13 import (FINGER_CONTACT_PARTS, HAND_PART,
    MODEL_EXTENT_PART, TargetContactGeometry)
from gpt_field.robot_tool_geometry import world_aabbs
from gpt_field.visual_field_v8 import measured_current_pose_hold_action


def state(xyz=(0.,0.,0.),rot=(0.,0.,0.),fingers=(.04,-.04)):
    return np.array([*xyz,*rot,*fingers])


def geometry(center=(0.,.0435,-.0036),half=(.001,.001,.001),contact='fingers',identity='object',uncertainty=.012):
    return dict(id=identity,center_offset_m=list(center),half_extents_m=list(half),
                uncertainty_m=uncertainty,allowed_contact=contact)


def configured(value=None,current=None,**kwargs):
    result=TargetContactGeometry()
    result.configure(state() if current is None else current,geometry() if value is None else value,target_id='object',**kwargs)
    return result


def partnames(config):return {p['tool_part']for p in config['core_parts']}


def test_finger_contact_exempts_only_four_named_parts_and_keeps_palm_without_coarse_ball():
    target=configured();c=target.configuration(state())
    assert partnames(c)=={HAND_PART} and set(c['exempt_parts'])==FINGER_CONTACT_PARTS
    assert not c['model_extent_included']
    close=[0.,0.,0.,0.,0.,0.,1.]
    admitted=target.assess(state(),close)
    assert admitted['accepted'] and admitted['action']==close
    assert not admitted['diagnostics']['hold_only']
    assert not admitted['diagnostics']['clearance_certified']
    denied=configured(geometry(contact='none')).assess(state(),close)
    assert not denied['accepted']
    assert any('finger' in p['tool_part']for p in denied['diagnostics']['intersections'])


def test_finger_permission_never_allows_palm_contact_during_closure_or_opening():
    target=configured(geometry(center=(0.,0.,-.08)))
    for grip in (-1.,1.):
        result=target.assess(state(),[0.,0.,0.,0.,0.,0.,grip])
        assert not result['accepted'] and result['reason']=='target_forbidden_part_intersection'
        assert [p['tool_part']for p in result['diagnostics']['intersections']]==[HAND_PART]


def test_top_down_descent_cannot_push_palm_into_target_despite_allowed_finger_contact():
    robot=state(rot=(math.pi,0.,0.));target=configured(geometry(center=(0.,0.,-.005)),robot)
    assert target.assess(robot,[0.,0.,-.1,0.,0.,0.,0.])['accepted']
    command=[0.,0.,-.8,0.,0.,0.,0.]
    denied=target.assess(robot,command)
    assert not denied['accepted'] and denied['action']==command
    hit=denied['diagnostics']['intersections'][0]
    assert hit['tool_part']==HAND_PART and not hit['start_inside']


def test_closed_gripper_does_not_remove_target_constraints_around_hand_or_wrist():
    robot=state(fingers=(0.,0.));target=configured(geometry(center=(0.,0.,-.10)),robot)
    result=target.assess(robot,[.05,0.,0.,0.,0.,0.,0.])
    assert not result['accepted'] and HAND_PART in partnames(target.configuration(robot))


def test_payload_support_removes_only_payload_proxy_with_local_observation_evidence():
    region=geometry(center=(.025,0.,.02),contact='payload_support')
    absent=configured(region);supported=configured(region,payload_observed=True)
    assert MODEL_EXTENT_PART in partnames(absent.configuration(state()))
    assert MODEL_EXTENT_PART not in partnames(supported.configuration(state()))
    assert partnames(supported.configuration(state()))=={p['name']for p in world_aabbs(state())}
    command=[.001,0.,0.,0.,0.,0.,0.]
    assert not absent.assess(state(),command)['accepted']
    assert supported.assess(state(),command)['accepted']
    assert absent.configuration(state())['payload_support_constraint']=='coarse_extent_retained_without_observed_payload'
    # Fresh local evidence can also be revoked without deleting the target.
    assert MODEL_EXTENT_PART in partnames(supported.configuration(state(),payload_observed=False))


def test_payload_support_does_not_exempt_robot_fingers_from_target_support():
    target=configured(geometry(contact='payload_support'),payload_observed=True)
    result=target.assess(state(),[0.,0.,0.,0.,0.,0.,1.])
    assert not result['accepted']
    assert any(p['tool_part']in FINGER_CONTACT_PARTS for p in result['diagnostics']['intersections'])


def test_empty_other_obstacle_list_cannot_omit_mandatory_target_geometry():
    obstacles=[];target=configured(geometry(center=(0.,0.,-.08)))
    assert obstacles==[] and target.configuration(state())['core_parts']
    assert not target.assess(state(),[0.,0.,0.,0.,0.,0.,1.])['accepted']


def test_configuration_uses_actual_rotated_robot_offsets_and_world_anchored_target():
    robot=state(xyz=(.1,-.2,1.),rot=(.7,-.4,1.2));region=geometry(center=(.13,.01,-.02))
    target=configured(region,robot);c=target.configuration(robot,orientation_margin_rad=0.)
    hand=next(p for p in world_aabbs(robot)if p['name']==HAND_PART)
    nominal_lo=robot[:3]+region['center_offset_m']-np.asarray(region['half_extents_m'])
    nominal_hi=robot[:3]+region['center_offset_m']+np.asarray(region['half_extents_m'])
    assert c['core_parts'][0]['lower_xyz']==pytest.approx(nominal_lo-hand['offset_max_xyz'])
    assert c['core_parts'][0]['upper_xyz']==pytest.approx(nominal_hi-hand['offset_min_xyz'])
    moved=robot.copy();moved[:3]+=[.02,-.01,.03]
    shifted=target.configuration(moved,orientation_margin_rad=0.)
    assert shifted['nominal']==c['nominal']
    assert shifted['core_parts'][0]['lower_xyz']==pytest.approx(c['core_parts'][0]['lower_xyz'])


def test_uncertainty_halos_remain_separate_and_never_shrink_under_contact_exemption():
    target=configured(geometry(uncertainty=.023));c=target.configuration(state())
    assert c['sigma_m']==.023
    for core,halo in zip(c['core_parts'],c['uncertainty_halo_parts']):
        assert np.asarray(core['lower_xyz'])-halo['lower_xyz']==pytest.approx([.023]*3)
        assert np.asarray(halo['upper_xyz'])-core['upper_xyz']==pytest.approx([.023]*3)
    assert c['tool_core_parts']==c['core_parts'] and c['full_halo_parts']==c['uncertainty_halo_parts']


def test_thin_target_crossed_between_clear_endpoints_is_rejected_continuously():
    robot=state(xyz=(-.02,0.,0.))
    target=configured(geometry(center=(.02,.0435,-.0036),half=(.0001,.0001,.0001),contact='none'),robot,tool_radius_m=0.)
    c=target.configuration(robot)
    end=robot.copy();end[0]+=.04
    # Gripper command forces stationary collision checks at each endpoint.
    assert target.assess(robot,[0.,0.,0.,0.,0.,0.,1.])['accepted']
    assert target.assess(end,[0.,0.,0.,0.,0.,0.,1.])['accepted']
    result=target.assess(robot,[.8,0.,0.,0.,0.,0.,0.])
    assert not result['accepted']
    assert all(not p['start_inside']and not p['end_inside']for p in result['diagnostics']['intersections'])


def test_rotation_sweep_bound_rejects_palm_target_that_zero_rotation_would_clear():
    target=configured(geometry(center=(.10,0.,-.08)))
    assert target.assess(state(),[0.,0.,0.,0.,0.,0.,1.])['accepted']
    rotation=[0.,0.,0.,0.,0.,1.,0.]
    result=target.assess(state(),rotation)
    assert not result['accepted'] and result['diagnostics']['rotation_bound_rad']==.5
    assert result['action']==rotation


@pytest.mark.parametrize('hold',[[0.]*7,measured_current_pose_hold_action()])
def test_rejected_command_pose_reset_hold_remains_possible_without_clearance_claim(hold):
    overlapping=configured(geometry(center=(0.,0.,-.08)))
    result=overlapping.assess(state(),hold)
    assert result['accepted'] and result['diagnostics']['hold_only']
    assert result['reason']=='target_geometry_pose_hold_only' and result['diagnostics']['intersections']
    assert not result['diagnostics']['clearance_certified']
    unavailable=TargetContactGeometry();unavailable.configure(state(),None,target_id='object')
    assert unavailable.assess(state(),hold)['accepted']
    assert not unavailable.assess(state(),[0.,0.,0.,0.,0.,0.,1.])['accepted']


def test_missing_geometry_preserves_world_box_by_identity_and_revokes_old_contact_permissions():
    target=configured();before=target.configuration(state());moved=state(xyz=(.03,0.,0.))
    target.configure(moved,None,target_id='object')
    missing=target.configuration(moved)
    assert not missing['available'] and missing['retained']
    assert missing['nominal']==before['nominal']
    assert missing['effective_allowed_contact']=='none' and len(missing['core_parts'])==6
    assert not target.assess(moved,[.02,0.,0.,0.,0.,0.,0.])['accepted']
    target.configure(moved,None,target_id='another')
    assert target.configuration(moved)['core_parts']==[]
    target.configure(moved,None,target_id='object')
    assert target.configuration(moved)['nominal']==before['nominal']
    target.reset();target.configure(moved,None,target_id='object')
    assert target.configuration(moved)['core_parts']==[]


def test_separation_is_explicit_exit_from_initial_overlap_not_a_general_collision_bypass():
    target=configured(geometry(center=(0.,0.,-.08)))
    command=[-.9,0.,0.,0.,0.,0.,0.]
    assert not target.assess(state(),command)['accepted']
    result=target.assess(state(),command,allow_separating=True)
    assert result['accepted'] and result['reason']=='target_nominal_separating_recovery'
    assert all(p['explicit_nearest_face_separation']for p in result['diagnostics']['intersections'])
    # Remaining inside, rotating, or closing cannot borrow the separation path.
    assert not target.assess(state(),[.1,0.,0.,0.,0.,0.,0.],allow_separating=True)['accepted']
    assert not target.assess(state(),[-.9,0.,0.,0.,0.,.01,0.],allow_separating=True)['accepted']
    assert not target.assess(state(),[-.9,0.,0.,0.,0.,0.,1.],allow_separating=True)['accepted']


@pytest.mark.parametrize('change',[
    {'id':'wrong'}, {'allowed_contact':'all'}, {'half_extents_m':[0.,.01,.01]},
    {'half_extents_m':[.00001,.01,.01]}, {'center_offset_m':[math.nan,0.,0.]},
    {'uncertainty_m':True}, {'object_pose':[0.,0.,0.]},
])
def test_malformed_model_region_cannot_mutate_previous_world_target(change):
    target=configured();before=target.configuration(state());bad=geometry();bad.update(change)
    with pytest.raises(ValueError):target.configure(state(),bad,target_id='object')
    assert target.configuration(state())==before


def test_output_mutation_and_input_mutation_cannot_rewrite_stored_world_geometry():
    raw=geometry();target=configured(raw);before=target.configuration(state())
    raw['center_offset_m'][0]=.4
    altered=target.configuration(state());altered['core_parts'][0]['lower_xyz'][0]=999.
    assert target.configuration(state())==before
    assert before['environment_inputs']==[] and not before['task_success_certified']


def test_summary_is_available_before_first_request_and_uses_last_measured_action_state():
    target=TargetContactGeometry()
    assert not target.summary()['available'] and target.summary()['core_parts']==[]
    target.configure(state(),geometry(),target_id='object')
    moved=state(xyz=(.01,0.,0.),rot=(0.,0.,.2))
    target.assess(moved,measured_current_pose_hold_action())
    assert {k:v for k,v in target.summary().items() if k!='carried_payload'}==target.configuration(moved)
    target.reset()
    assert target.summary()['reason']=='target_geometry_not_configured'


def test_unavailable_geometry_permits_only_nontranslating_gripper_opening_not_motion_or_close():
    target=configured(geometry(center=(0.,0.,-.08)))
    target.configure(state(),None,target_id='object')
    for rotation in (0.,2e-9):
        command=[0.,0.,0.,rotation,0.,0.,-1.]
        admitted=target.assess(state(),command)
        assert admitted['accepted'] and admitted['action']==command
        assert admitted['diagnostics']['opening_only'] and admitted['diagnostics']['intersections']
        assert not admitted['diagnostics']['clearance_certified']
    for command in ([.001,0.,0.,0.,0.,0.,-1.],[0.,0.,0.,.001,0.,0.,-1.],[0.,0.,0.,0.,0.,0.,1.]):
        assert not target.assess(state(),command)['accepted']


def test_payload_candidate_follows_nothing_until_observed_attachment_and_is_not_overwritten_by_closure():
    target=configured();initial=target.configuration(state())['nominal']
    assert target.remember_grasp_candidate(state(),'object')['accepted']
    lifted=state(xyz=(.01,0.,.02))
    assert target.remember_grasp_candidate(lifted,'object')['reason']=='existing_grasp_candidate_retained'
    assert not target.carried_summary(lifted)['active']
    assert target.configuration(lifted)['nominal']==initial
    assert not target.confirm_payload_attachment(lifted,'object',observed=False)['accepted']
    assert target.confirm_payload_attachment(lifted,'object',observed=True)['accepted']
    carried=target.carried_summary(lifted)
    assert carried['snapshot_state8']==pytest.approx(state())
    assert carried['nominal_world']['lower_xyz']==pytest.approx(np.asarray(initial['lower_xyz'])+lifted[:3])
    assert target.configuration(lifted)['nominal']==initial  # Static box never dragged.


def test_observed_payload_survives_destination_switch_and_transforms_only_by_known_robot_pose():
    target=configured();target.remember_grasp_candidate(state(),'object')
    robot=state(xyz=(.1,-.2,.3),rot=(0.,0.,math.pi/2))
    target.confirm_payload_attachment(robot,'object',observed=True)
    target.configure(robot,geometry(identity='destination',center=(.2,0.,0.),contact='payload_support'),
                     target_id='destination',payload_observed=True)
    info=target.carried_summary(robot)
    assert info['active'] and info['payload_id']=='object'
    center=(np.array(info['nominal_world']['lower_xyz'])+info['nominal_world']['upper_xyz'])/2
    assert center==pytest.approx(robot[:3]+[-.0435,0.,-.0036])
    assert info['uncertainty_m']==.012 and not info['static_world_boxes_moved']
    assert target.assess_carried_payload(robot,[.1,0.,0.,0.,0.,.1,0.])['accepted']
    assert target.summary()['carried_payload']['payload_id']=='object'


def test_carried_palm_overlap_cannot_be_erased_by_new_target_or_whole_hand_translation():
    target=configured(geometry(center=(0.,0.,-.08)))
    target.remember_grasp_candidate(state(),'object')
    robot=state(xyz=(.04,.05,.06),rot=(.2,.1,-.4))
    target.confirm_payload_attachment(robot,'object',observed=True)
    target.configure(robot,geometry(identity='destination',center=(.3,0.,0.),contact='payload_support'),
                     target_id='destination',payload_observed=True)
    assert target.assess(robot,[.1,0.,0.,0.,0.,0.,0.])['accepted']  # Destination itself is remote.
    for action in ([.1,0.,0.,0.,0.,0.,0.],[0.,0.,0.,.1,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.]):
        denied=target.assess_carried_payload(robot,action)
        assert not denied['accepted'] and denied['action']==action
        assert denied['diagnostics']['forbidden_part_intersections']==[HAND_PART]
    for action in (measured_current_pose_hold_action(),[0.,0.,0.,0.,0.,0.,-1.]):
        admitted=target.assess_carried_payload(robot,action)
        assert admitted['accepted'] and not admitted['diagnostics']['clearance_certified']


def test_attachment_uncertainty_and_opening_do_not_erase_active_carried_geometry():
    target=configured();target.remember_grasp_candidate(state(),'object')
    target.confirm_payload_attachment(state(),'object',observed=True)
    assert not target.confirm_payload_attachment(state(),'object',observed=False)['accepted']
    assert not target.cancel_unconfirmed_candidate('object')['accepted']
    assert not target.release_payload('object',observed_separation=False)['accepted']
    target.assess_carried_payload(state(),[0.,0.,0.,0.,0.,0.,-1.])
    assert target.carried_summary(state())['active']
    assert target.release_payload('object',observed_separation=True)['accepted']
    assert not target.carried_summary(state())['active']
    assert target.configuration(state())['nominal'] is not None


def test_observed_reopening_can_cancel_only_unconfirmed_candidate_and_new_grasp_uses_fresh_geometry():
    target=configured();target.remember_grasp_candidate(state(),'object')
    assert target.cancel_unconfirmed_candidate('object')['accepted']
    assert not target.confirm_payload_attachment(state(),'object',observed=True)['accepted']
    current=state(xyz=(.03,0.,0.))
    target.configure(current,geometry(center=(.01,.04,.005)),target_id='object')
    target.remember_grasp_candidate(current,'object')
    target.confirm_payload_attachment(current,'object',observed=True)
    assert target.carried_summary(current)['snapshot_state8']==pytest.approx(current)
    assert target.carried_summary(current)['nominal_world']==target.configuration(current)['nominal']


def test_unknown_or_stale_model_box_cannot_become_arbitrarily_attached_payload():
    target=configured();target.configure(state(),None,target_id='object')
    assert not target.remember_grasp_candidate(state(),'object')['accepted']
    assert not target.confirm_payload_attachment(state(),'object',observed=True)['accepted']
    assert not target.confirm_payload_attachment(state(),'unseen',observed=True)['accepted']
    assert not target.carried_summary(state())['active']
