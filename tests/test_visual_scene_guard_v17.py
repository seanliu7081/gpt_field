"""Command-swept coverage, partial support, and unchanged contact contracts."""
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_scene_guard_v16 import SceneCoverageGuardV16
from gpt_field.visual_scene_guard_v17 import SceneCoverageGuardV17, segment_box_interval
from test_visual_scene_guard_v16 import fixture as old_fixture, object_record, assess


def fixture(records=None):
    _, state, material, field, pred = old_fixture(records)
    return SceneCoverageGuardV17(), state, material, field, pred


def partial(record):
    view = record['views']['external']
    view.update(current_visible_roi_uv=view['current_roi_uv'], current_roi_uv=None,
                partial_current_support=True, support_inlier_count=5)
    return record


@pytest.mark.parametrize('support', ['missing', 'partial'])
def test_uncertain_object_inside_old_cube_but_disjoint_actual_sweep_allows_reuse(support):
    record = object_record(center=(.12, 0., 0.))
    if support == 'partial': partial(record)
    else: record['views']['external']['current_roi_uv'] = None
    values = fixture([record]); _, state, material, field, pred = values
    assert not SceneCoverageGuardV16().assess(state, [-.005,0,0], materialization=material,
                                             field=field, prediction=pred)['accepted']
    for _ in range(12):
        result = assess(values, [-.005,0,0])
        assert result['accepted']
        assert result['diagnostics']['metric_only_disjoint_ids'] == ['obstacle']
        assert result['diagnostics']['checked_metric_ids'] == ['obstacle']
        assert result['diagnostics']['sweep_checks'][0]['configuration_part_count'] == 6
        state[0] -= .001  # Measured local feedback; no extra model decision.
    assert not result['diagnostics']['clearance_certified']


@pytest.mark.parametrize('support', ['missing', 'partial'])
def test_partial_support_never_licenses_entering_full_uncertainty_halo(support):
    record = object_record(center=(.08,0,0))
    if support == 'partial': partial(record)
    else: record['views']['external']['current_roi_uv'] = None
    result = assess(fixture([record]), [.015,0,0])
    assert result['reason'] == 'scene_incomplete_image_support_intersects_candidate_uncertainty_sweep'
    assert result['diagnostics']['sweep_checks'][0]['intersected_parts']


def test_long_command_with_clear_endpoints_cannot_tunnel_through_box():
    values = fixture([object_record(center=(.13,0,0), current=False)])
    guard,state,_,_,pred=values
    boxes=guard._sweep_boxes(values[2]['objects'][0],state,pred,.04)
    assert all(segment_box_interval(state[:3],state[:3],b['lower_xyz'],b['upper_xyz']) is None for b in boxes)
    assert all(segment_box_interval([.26,0,0],[.26,0,0],b['lower_xyz'],b['upper_xyz']) is None for b in boxes)
    result=assess(values,[.26,0,0])
    assert not result['accepted']
    intervals=result['diagnostics']['sweep_checks'][0]['intersected_parts']
    assert any(0 < x['segment_parameter_interval'][0] < x['segment_parameter_interval'][1] < 1 for x in intervals)
    assert segment_box_interval([0,0,0],[1,0,0],[.5,0,0],[.5,1,1])==[.5,.5]


@pytest.mark.parametrize('omission',['metric_ids','metric_actual','image_ids','image_actual'])
def test_disjointness_does_not_remove_named_objects_from_consumers(omission):
    values=fixture([partial(object_record(center=(1,1,1)))])
    _,_,mat,field,_=values
    if omission=='metric_ids':mat['applied_metric_ids']=[]
    if omission=='metric_actual':field.region_geometry=[]
    if omission=='image_ids':mat['applied_image_ids']=[]
    if omission=='image_actual':field.translation_guard.neighbors=[]
    result=assess(values)
    assert not result['accepted'] and 'coverage_missing_after_materialization' in result['reason']


def test_current_full_observation_retains_risk_semantics_not_a_hard_halo_check():
    values=fixture([object_record(center=(.08,0,0))])
    values[4]['controller']='direct'
    result=assess(values,[.015,0,0])
    assert result['accepted'] and result['diagnostics']['sweep_checks'][0]['intersected_parts']
    assert not result['diagnostics']['clearance_certified']  # Hard nominal and RGB gates remain independent.


@pytest.mark.parametrize('textured',[True,False])
def test_boundary_full_rectangle_does_not_establish_full_extent_even_on_model_refresh(textured):
    record=object_record(center=(.08,0,0),roi=(.8,.2,1.,.5))
    view=record['views']['external']
    view.update(extent_visibility='boundary_limited',extent_touches_image_boundary=True,
                partial_current_support=textured,current_visible_roi_uv=[.8,.2,1.,.5] if textured else None)
    values=fixture([record]);values[4]['controller']='direct'
    result=assess(values,[.015,0,0])
    assert not result['accepted']
    assert result['diagnostics']['checked_image_ids']==['obstacle']
    assert result['diagnostics']['sweep_checks'][0]['image_support']!='full_current'
    # Changing the same saved view to a truly current interior declaration
    # changes only visibility semantics, not geometry or image-ID coverage.
    view.update(current_roi_uv=[.79,.2,.99,.5],extent_visibility='within_image',
                extent_touches_image_boundary=False,partial_current_support=False)
    assert assess(values,[.015,0,0])['accepted']


def test_image_only_partial_near_tool_and_zero_translation_closure_require_localization():
    values=fixture([partial(object_record(geometry=False,roi=(.28,.25,.4,.4)))])
    values[4]['controller']='gripper'
    assert assess(values,[0,0,0])['reason']=='pose_hold_no_scene_clearance_claim'
    assert assess(values,[0,0,0],action7=[0,0,0,0,0,0,1])['reason']=='scene_visible_region_needs_metric_localization'
    assert not assess(values,[0,0,0],requires_scene_coverage=True)['accepted']


def test_direct_uses_current_geometry_and_actual_rotation_or_prediction_maximum():
    values=fixture([object_record(center=(.13,0,-.08),current=False)])
    _,state,mat,field,pred=values
    field.origin=np.ones(3)*100
    pred.update(controller='direct', direct_actions=[[0,0,0,0,0,0,0],[0,0,0,0,0,1,0]])
    assert assess(values,[0,0,0],requires_scene_coverage=True,action7=[0]*7)['accepted']
    result=assess(values,[0,0,0],requires_scene_coverage=True)
    assert result['diagnostics']['orientation_margin_rad']==.5
    assert not result['accepted']
    assert not assess(values,[0,0,0],action7=[0,0,0,0,0,1,0])['accepted']


def test_locally_observed_payload_retained_extent_participates_in_sweep():
    values=fixture([object_record(center=(.2,0,0),current=False)])
    assert assess(values,[-.005,0,0])['accepted']
    values[3].contact_geometry=SimpleNamespace(carried_summary=lambda state:{'active':True,
        'nominal_world':{'lower_xyz':[.14,-.01,-.01],'upper_xyz':[.18,.01,.01]},
        'nominal_hand_frame':{'lower_xyz':[.14,-.01,-.01],'upper_xyz':[.18,.01,.01]},'uncertainty_m':.005})
    result=assess(values,[-.005,0,0])
    assert not result['accepted'] and result['diagnostics']['carried_payload_hypothesis_in_sweep']
    assert any(p['tool_part']=='observed_carried_payload_hypothesis' for p in result['diagnostics']['sweep_checks'][0]['intersected_parts'])


@pytest.mark.parametrize('phase',['approach','align','recover'])
def test_target_palm_handoff_survives_sweep_change_and_away_motion_remains_possible(phase):
    values=fixture();values[4]['phase']=phase
    assert assess(values,[.005,0,0])['reason']=='near_target_approach_requires_current_two_view_contact'
    assert assess(values,[-.005,0,0])['accepted']
    descriptor={'active':True,'target_id':'part','current_points_px':{k:[10,10] for k in ('tool','external_target','wrist_target','wrist_pinch')}}
    assert assess(values,[.005,0,0],contact_alignment=descriptor)['accepted']
    descriptor['current_points_px'].pop('wrist_pinch')
    assert not assess(values,[.005,0,0],contact_alignment=descriptor)['accepted']


def test_no_command_or_budget_mutation_including_existing_pending_selection():
    values=fixture([partial(object_record(center=(.12,0,0)))])
    guard,state,mat,field,pred=values
    field.pending={'action':[.1,0,0,0,0,0,0]};field.translation_guard.observation_travel_m=.017
    before=pickle.dumps(values)
    for cmd in ([-.005,0,0],[.02,0,0],[0,0,0]):
        original=copy.deepcopy(cmd);assess(values,cmd)
        assert cmd==original and pickle.dumps(values)==before


def test_saved_v16_002_coverage_failure_has_supported_disjoint_candidate_not_new_rollout():
    path=Path(__file__).resolve().parents[1]/'runs/visual_field_v16_zero_shot_002/trials/bowl_original_zero_hybrid_v16/trace.jsonl'
    if not path.exists():pytest.skip('Optional archived V16 run absent')
    rows=[json.loads(line) for line in path.open()]
    action=next(r for r in rows if r['event']=='action' and r['step']==22)
    before=next(r['feedback'] for r in rows if r['event']=='feedback' and r['step']==21)
    pred=next(r['diagnostics']['prediction'] for r in rows if r['event']=='action' and r['step']==19)
    prior=next(r for r in rows if r['event']=='action' and r['step']==21)
    records=before['scene_inventory']['entries'];ids=[r['id'] for r in records]
    field=SimpleNamespace(origin=np.array(before['field']['origin_xyz']),grid=SimpleNamespace(half_width_m=.08),
        region_geometry=before['field']['region_feasibility'],target_configuration=before['field']['target_contact_geometry'],
        translation_guard=SimpleNamespace(neighbors=before['translation_guard']['neighbors']))
    mat={'objects':records,'applied_metric_ids':ids,'applied_image_ids':ids}
    # The rejected V16 command was not logged. Recheck continuation of the
    # immediately preceding actual +Y command at the saved next robot state;
    # this is a candidate-admission counterfactual, not an alternate rollout.
    delta=np.array(prior['action'][:3])*.05
    assert np.linalg.norm(delta)==pytest.approx(.005) and delta[1]>.00499
    old=SceneCoverageGuardV16().assess(before['after_state'],delta,materialization=mat,field=field,prediction=pred)
    assert old['reason']==action['diagnostics']['selection']['status']=='scene_local_metric_region_lacks_current_image_support'
    new=SceneCoverageGuardV17().assess(before['after_state'],delta,materialization=mat,field=field,prediction=pred)
    assert new['accepted'] and new['diagnostics']['checked_metric_ids']==ids
    cabinet=next(x for x in new['diagnostics']['sweep_checks'] if x['id']=='drawer_cabinet')
    assert cabinet['full_uncertainty_sweep_disjoint'] and cabinet['image_support']=='missing_or_stale'
    # Same remembered geometry blocks a candidate into its full halo; no
    # object or uncertainty disappeared when the cube check was replaced.
    toward=np.array([0,-.15,0])
    negative=SceneCoverageGuardV17().assess(before['after_state'],toward,materialization=mat,field=field,prediction=pred)
    assert not negative['accepted']
    assert any(x['id']=='drawer_cabinet' for x in negative['diagnostics']['coverage_issues'])
