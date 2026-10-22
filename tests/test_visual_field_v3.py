import copy
import json

import numpy as np
import pytest

from gpt_field.visual_field_v3 import VisualFieldControllerV3, preflight_reference, _FootprintField
from test_visual_field_v2 import BASE, images, state, seeds as old_seeds


def seeds(*, payload=False, obstacles=False, unknown=False):
    values = old_seeds(payload=payload, obstacles=obstacles)
    for value in values.values():
        point = np.asarray(value['moving_point_uv'])
        value.pop('moving_radius_uv')
        value['moving_footprint_uv'] = {'lower_uv': (point-[.025,.035]).tolist(), 'upper_uv': (point+[.04,.025]).tolist()}
        for box in value['blocked_boxes_uv']:
            box['role'] = 'obstacle'
        if unknown:
            value['unknown_boxes_uv'] = [{'object_id':'invisible', 'source':'occluded', 'lower_uv':[.75,.08], 'upper_uv':[.82,.14]}]
    return values


def train(*, count=4, unknown=False, obstacles=False, payload=False, commands=None, refs=None):
    c = VisualFieldControllerV3(grid_size=16, joint_lower=[-1]*7, joint_upper=[1]*7)
    q, eef = np.zeros(7), np.zeros(2)
    c.configure(images(payload=payload), refs or seeds(payload=payload,unknown=unknown,obstacles=obstacles), q, state8=state())
    for n in range(count):
        cmd = commands[n] if commands else [.1,0,0,0,0,0,1]
        eef += np.asarray(cmd[:2])*.05; q[:3] += np.asarray(cmd[:3])*.1
        c.observe(images(eef,payload=payload),q,state8=state(eef),executed_action=cmd)
    return c,q,eef


def grant(*, directions=None, codes=None):
    return {'blocker_codes':codes or ['response_unsupported'], 'directions':directions or [[.1,0,0]],
        'max_steps':4,'max_displacement_m':.02,'corridors_uv':{v:{'lower_uv':[.1,.1],'upper_uv':[.9,.9]} for v in ('external','wrist')}}


def test_asymmetric_footprint_preflight_detects_contradictions_before_learning():
    ref=seeds()['external']
    point=np.array(ref['moving_point_uv'])
    ref['unknown_boxes_uv']=[{'object_id':'hidden','source':'occluded','lower_uv':(point+[.03,0]).tolist(),'upper_uv':(point+[.06,.04]).tolist()}]
    result=preflight_reference(ref,(128,128),grid_size=40)
    assert result['semantic_repair_required'] and not result['start_valid']
    assert any(b['code']=='start_unknown_conflict' and b['region_id']=='hidden' for b in result['blockers'])
    assert result['footprint_offsets_uv']['lower_uv']==pytest.approx([-.025,-.035])
    c=VisualFieldControllerV3(grid_size=40)
    c.configure(images(),{'external':ref,'wrist':seeds()['wrist']},np.zeros(7),state8=state())
    assert c.summary()['views']['external']['distance_cost_px'] is not None
    assert c.select_action([.1,0,0,0,0,0,1],observation_intent=grant())['status']=='semantic_correction_required'


def test_goal_contact_permission_is_explicit_and_does_not_allow_field_contact():
    ref=seeds()['external'];goal=np.asarray(ref['target_point_uv'])
    region={'object_id':'anything_named_plate','role':'obstacle','lower_uv':(goal-.02).tolist(),'upper_uv':(goal+.02).tolist()}
    ref['blocked_boxes_uv']=[region]
    blocked=preflight_reference(ref,(128,128),40)
    assert any(b['code']=='goal_obstacle_conflict' for b in blocked['blockers'])
    region['role']='contact_target'
    allowed=preflight_reference(ref,(128,128),40)
    assert allowed['goal_valid'] and allowed['terminal_contact_regions']==['anything_named_plate']
    assert not allowed['semantic_repair_required']
    c,_,_=train()
    for view in c.views.values():
        p=c._point(view,'moving');region=copy.deepcopy(view['ref'])
        region['blocked_boxes_uv']=[{'object_id':'contact','role':'contact_target','lower_uv':(p+[.09,-.02]).tolist(),'upper_uv':(p+[.12,.02]).tolist()}]
        # Inject a narrow future contact envelope into the candidate check.
        view['boxes']=[{'id':'contact','kind':'blocked','role':'contact_target','source':None,'low':p+[.051,-.01],'high':p+[.06,.01],
            'patch':None,'anchor':None,'response':view['feature_model'],'age':0,'growth_px':0.,'omitted':False,'self_offsets':None}]
    checked,reason=c._candidate(np.array([.1,0,0]),validated=True)
    assert checked is None and reason=='contact_handoff_required'


def test_valid_body_self_occlusion_does_not_erase_external_unknown_extent():
    refs=seeds()
    for ref in refs.values():
        ref['unknown_boxes_uv']=[{'object_id':'body','source':'self_occlusion',**copy.deepcopy(ref['moving_footprint_uv'])}]
        assert not preflight_reference(ref,(128,128),40)['semantic_repair_required']
    c,_,_=train(refs=refs)
    assert c.select_action([.1,0,0,0,0,0,1])['accepted']
    ref=refs['external'];ref['unknown_boxes_uv'][0]['upper_uv'][0]+=.15
    result=preflight_reference(ref,(128,128),40)
    assert result['semantic_repair_required']
    assert any(b['code']=='self_occlusion_outside_footprint' for b in result['blockers'])


def test_footprint_outside_image_is_not_clipped_or_silently_shrunk():
    ref=seeds()['external'];ref['moving_footprint_uv']['lower_uv'][0]=-.1
    report=preflight_reference(ref,(128,128))
    assert not report['start_valid']
    assert report['blockers'][0]['code']=='footprint_out_of_view'
    assert ref['moving_footprint_uv']['lower_uv'][0]==-.1


def test_border_preflight_and_field_use_actual_footprint_not_extra_grid_border():
    ref=seeds()['external'];ref.update(moving_point_uv=[.03,.5],target_point_uv=[.5,.5],moving_footprint_uv={'lower_uv':[.02,.49],'upper_uv':[.04,.51]})
    pre=preflight_reference(ref,(128,128),40)
    assert pre['start_valid'] and pre['goal_valid']
    f=_FootprintField((128,128),[.5,.5],(np.array([-.01,-.01]),np.array([.01,.01])),[],40)
    assert f.cost([.03,.5]) is not None
    assert f.segment_valid([.03,.5],[.04,.5])


def test_untextured_far_unknown_retains_uncertainty_without_global_veto():
    c,_,_=train(unknown=True)
    info=c.summary()
    assert info['geometry_reliable']
    for v in info['views'].values():
        region=v['obstacle_tracks'][0]
        assert region['source']=='occluded' and not region['interior_track_required']
        assert not region['track_reliable'] and region['uncertainty_growth_px']>=4
    selected=c.select_action([.1,0,0,0,0,0,1])
    assert selected['accepted'] and selected['status']=='visual_distance_descent'
    assert selected['diagnostics']['risk_unknown']


def test_untracked_far_visible_box_does_not_require_response_to_execute_elsewhere():
    refs=seeds()
    for ref in refs.values():
        ref['blocked_boxes_uv']=[{'object_id':'flat_distant_obstacle','role':'obstacle','lower_uv':[.1,.72],'upper_uv':[.18,.8]}]
    c,_,_=train(refs=refs)
    assert all(not v['obstacle_tracks'][0]['track_reliable'] for v in c.summary()['views'].values())
    assert c.select_action([.1,0,0,0,0,0,1])['accepted']


def test_candidate_cannot_cross_unknown_even_with_probe_grant():
    c,_,_=train(count=2)
    for view in c.views.values():
        p=c._point(view,'moving')
        view['boxes']=[{'id':'hidden','kind':'unknown','role':None,'source':'occluded','low':p+[.05,-.03],'high':p+[.07,.03],
            'patch':None,'anchor':None,'response':view['feature_model'],'age':0,'growth_px':0.,'omitted':False,'self_offsets':None}]
    result=c.select_action([.1,0,0,0,0,0,1],observation_intent=grant(codes=['candidate_unknown_intersection','response_unsupported']))
    assert not result['accepted']
    assert 'candidate_unknown_intersection' in result['diagnostics']['rejection_reasons']


def test_reseed_of_other_view_retains_healthy_view_response_epoch_and_samples():
    c,q,eef=train()
    old=c.views['external']['feature_model'];epoch=c.views['external']['model_epoch'];refs=seeds()
    for ref in refs.values():
        ref['moving_point_uv']=((BASE['moving']+[8,0])/127).tolist()
        ref['moving_footprint_uv']={k:(np.array(v)+[8/127,0]).tolist() for k,v in ref['moving_footprint_uv'].items()}
    refs['wrist']['target_id']='new_anchor_identity'
    c.configure(images(eef),refs,q,state8=state(eef))
    assert c.views['external']['feature_model'] is old
    assert c.views['external']['model_epoch']==epoch
    assert c.views['external']['feature_model'].summary()['reliable']
    assert not c.views['wrist']['feature_model'].summary()['reliable']


def test_per_view_learning_can_start_when_other_view_has_no_seed():
    refs=seeds();refs['wrist']=None
    c,_,_=train(refs=refs)
    assert c.summary()['view_models']['external']['eef_to_image']['reliable']
    assert c.summary()['view_models']['wrist']['eef_to_image']['sample_count']==0
    assert c.select_action([.1,0,0,0,0,0,1])['status']=='unobserved_view_scope'


def test_missing_view_restricts_motion_and_retains_old_constraints():
    c,q,eef=train(unknown=True)
    refs=seeds(unknown=True)
    refs['external']['moving_point_uv']=((BASE['moving']+[8,0])/127).tolist()
    refs['external']['moving_footprint_uv']={k:(np.array(v)+[8/127,0]).tolist() for k,v in refs['external']['moving_footprint_uv'].items()}
    refs['wrist']=None
    c.configure(images(eef),refs,q,state8=state(eef))
    result=c.select_action([.2,0,0,0,0,0,1])
    assert result['accepted'] and np.linalg.norm(result['action'][:3])<=.05+1e-10
    assert result['diagnostics']['missing_views']==['wrist']
    assert c.summary()['views']['wrist']['obstacle_track_count']==1
    for _ in range(4):
        c.observe(images(eef),q,state8=state(eef),executed_action=[0]*7)
    assert not c.select_action([.1,0,0,0,0,0,1])['accepted']


def test_missing_view_relevant_retained_unknown_vetoes_single_view_action():
    c,q,eef=train()
    c.views['wrist']['provided']=False;c.views['wrist']['reliable']=False
    view=c.views['wrist'];p=c._point(view,'moving')
    view['boxes']=[{'id':'retained_hidden','kind':'unknown','role':None,'source':'occluded','low':p+[.045,-.03],'high':p+[.06,.03],
        'patch':None,'anchor':None,'response':view['feature_model'],'age':0,'growth_px':0.,'omitted':True,'self_offsets':None}]
    result=c.select_action([.05,0,0,0,0,0,1])
    assert not result['accepted']
    assert 'candidate_unknown_intersection' in result['diagnostics']['rejection_reasons']


def test_omitted_mask_is_retained_and_cannot_clear_semantic_conflict():
    refs=seeds();p=np.asarray(refs['external']['moving_point_uv'])
    refs['external']['unknown_boxes_uv']=[{'object_id':'hidden','source':'occluded','lower_uv':(p-.02).tolist(),'upper_uv':(p+.02).tolist()}]
    c=VisualFieldControllerV3(grid_size=16);q=np.zeros(7)
    c.configure(images(),refs,q,state8=state())
    c.configure(images(),seeds(),q,state8=state())
    assert c.summary()['feasibility']['semantic_repair_required']
    assert c.summary()['views']['external']['obstacle_tracks'][0]['retained_after_omission']


def test_local_probe_needs_grant_empirical_support_and_all_reliable_corridors():
    c,_,_=train(count=2)
    assert not c.select_action([.1,0,0,0,0,0,1])['accepted']
    result=c.select_action([.1,0,0,0,0,0,1],observation_intent=grant())
    assert result['accepted'] and result['status']=='local_information_probe'
    assert result['action'][3:]==[0,0,0,0]
    assert 0<np.linalg.norm(result['diagnostics']['predicted_eef_delta_m'])<=.0051
    assert not result['need_refresh']
    assert not result['diagnostics']['model_grant_is_clearance_certificate']
    bad=grant();bad['corridors_uv']['wrist']=None
    assert not c.select_action([.1,0,0,0,0,0,1],observation_intent=bad)['accepted']
    unseen=grant(directions=[[0,.1,0]])
    assert not c.select_action([0,.1,0,0,0,0,1],observation_intent=unseen)['accepted']
    no_data,_,_=train(count=0)
    assert not no_data.select_action([.1,0,0,0,0,0,1],observation_intent=grant())['accepted']


def test_probe_corridor_and_displacement_bounds_are_prospective():
    c,_,_=train(count=2)
    narrow=grant();narrow['corridors_uv']['external']={'lower_uv':[.1,.1],'upper_uv':[.38,.9]}
    assert not c.select_action([.1,0,0,0,0,0,1],observation_intent=narrow)['accepted']
    tiny=grant();tiny['max_displacement_m']=.001
    assert not c.select_action([.1,0,0,0,0,0,1],observation_intent=tiny)['accepted']
    wrong=grant(codes=['goal_obstacle_conflict'])
    assert not c.select_action([.1,0,0,0,0,0,1],observation_intent=wrong)['accepted']


def test_neutral_probes_do_not_reset_open_gripper_projection_models():
    c,_,_=train(count=6,commands=[[.1,0,0,0,0,0,g] for g in [-1,0,0,0,-1,0]])
    assert all(v['eef_to_image']['sample_count']==6 and v['eef_to_image']['reliable'] for v in c.summary()['views'].values())
    assert c._last_gripper==-1


def test_large_requested_action_includes_smaller_supported_candidate():
    c,_,_=train()
    result=c.select_action([.6,0,0,0,0,0,1])
    assert result['accepted'] and result['status']=='visual_distance_descent'
    assert np.linalg.norm(result['action'][:3])<=.10001


def test_obstacle_shift_outside_image_keeps_unclipped_bounds_without_crash():
    c,q,eef=train(obstacles=True)
    for view in c.views.values():
        b=view['boxes'][0]
        b['patch'].position[0]-=100
        b['anchor'][0]+=30
        c._build_view(view)
        assert c._region_bounds(view,b)[0][0]<0
    json.dumps(c.summary(),allow_nan=False)


def test_payload_attachment_epoch_and_tool_coordinates_remain_available():
    c,q,eef=train(payload=True)
    info=c.summary()
    assert info['co_motion_evidence']
    epoch=info['attachment_epoch']
    assert all(v['tool_point_uv'] is not None for v in info['views'].values())
    c.observe(images(eef,payload=True,missing=('tool',)),q,state8=state(eef),executed_action=[0]*7)
    assert not c.summary()['co_motion_evidence']
    assert c.summary()['attachment_epoch']==epoch+1


@pytest.mark.parametrize('change',[
    lambda r:r.update(object_pose=[0,0,0]),
    lambda r:r.update(moving_radius_uv=.1),
    lambda r:r['blocked_boxes_uv'].append({'object_id':'plate','lower_uv':[.1,.1],'upper_uv':[.2,.2]}),
    lambda r:r['unknown_boxes_uv'].append({'object_id':'hidden','source':'free','lower_uv':[.1,.1],'upper_uv':[.2,.2]}),
])
def test_noncanonical_or_untyped_reference_rejected(change):
    ref=seeds()['external'];change(ref)
    with pytest.raises(ValueError):preflight_reference(ref,(128,128))


def test_rejected_close_or_open_intent_emits_only_neutral_action():
    refs=seeds();p=np.asarray(refs['external']['moving_point_uv'])
    refs['external']['unknown_boxes_uv']=[{'object_id':'hidden','source':'occluded','lower_uv':(p-.02).tolist(),'upper_uv':(p+.02).tolist()}]
    c=VisualFieldControllerV3(grid_size=16)
    c.configure(images(),refs,np.zeros(7),state8=state())
    for gripper in (-1,1):
        result=c.select_action([.1,0,0,0,0,0,gripper])
        assert not result['accepted'] and result['status']=='semantic_correction_required'
        assert result['action']==[0.]*7


def test_local_observation_probe_need_not_already_have_route_to_task_goal():
    refs=seeds()
    for ref in refs.values():
        ref['unknown_boxes_uv']=[{'object_id':'wall','source':'occluded','lower_uv':[.57,0.],'upper_uv':[.59,1.]}]
    c,_,_=train(refs=refs)
    assert not c.summary()['feasibility']['semantic_repair_required']
    assert all(v['route_cost_px'] is None for v in c.summary()['views'].values())
    result=c.select_action([.1,0,0,0,0,0,1],observation_intent=grant(directions=[[-.1,0,0]],codes=['route_unavailable']))
    assert result['accepted'] and result['status']=='local_information_probe'
    assert result['action'][0]<0  # away from the unknown wall, within the granted corridor
    assert result['action'][3:]==[0,0,0,0]
