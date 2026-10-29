"""Scene completeness and contact handoff without truth inputs or actions."""
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_scene_guard_v16 import SceneCoverageGuardV16
from test_visual_contact_alignment_v13 import fixture as contact_fixture


def object_record(identity="obstacle", center=(.05, .02, 0.), *, roi=(.5, .5, .6, .6), geometry=True, current=True):
    geom = {"center_xyz_m": list(center), "half_extents_m": [.01]*3, "uncertainty_m": .02} if geometry else None
    return {"id": identity, "world_geometry": geom,
            "views": {"external": {"current_roi_uv": list(roi) if current and roi is not None else None,
                "last_known_roi_uv": None if roi is None else list(roi), "reliable": current,
                "status": "tracked" if current else "lost"}}, "omitted_from_latest_inventory": False}


def fixture(records=None):
    state = np.r_[np.zeros(6), .04, -.04]
    records = records if records is not None else [object_record()]
    ids = [r["id"] for r in records]
    materialization = {"objects": records, "applied_metric_ids": ids.copy(), "applied_image_ids": ids.copy(),
                       "geometrically_disjoint_ids": [], "controlled_payload_excluded_id": None}
    guard = SimpleNamespace(neighbors=[{"id": i} for i in ids], scale=np.array([223,223]),
                            tool_box=np.array([.2,.2,.3,.3])*223)
    field = SimpleNamespace(origin=np.zeros(3), grid=SimpleNamespace(half_width_m=.08),
        region_geometry=[{"id": i} for i in ids], translation_guard=guard,
        target_configuration={"target_id":"part", "sigma_m":.05,
            "core_parts":[{"tool_part":"palm", "lower_xyz":[.04,-.01,-.01], "upper_xyz":[.06,.01,.01]}]})
    pred = {"controller":"visual_field", "phase":"approach", "target_id":"part",
            "field_intent":{"tool_radius_m":.045,"uncertainty_m":.005},
            "motion_context":{"tool_external_roi_uv":[.2,.2,.3,.3]}, "contact_alignment":None}
    return SceneCoverageGuardV16(),state,materialization,field,pred


def assess(fixture_value, delta=(-.005,0,0), **kwargs):
    guard,state,mat,field,pred=fixture_value
    return guard.assess(state,delta,materialization=mat,field=field,prediction=pred,**kwargs)


@pytest.mark.parametrize("omission", ["metric_ids","metric_actual","image_ids","image_actual"])
def test_retained_named_object_cannot_disappear_from_either_consumer(omission):
    values=fixture();_,_,mat,field,_=values
    mat['objects'][0]['omitted_from_latest_inventory']=True
    if omission=='metric_ids':mat['applied_metric_ids']=[]
    if omission=='metric_actual':field.region_geometry=[]
    if omission=='image_ids':mat['applied_image_ids']=[]
    if omission=='image_actual':field.translation_guard.neighbors=[]
    result=assess(values)
    assert not result['accepted'] and 'coverage_missing_after_materialization' in result['reason']
    assert result['diagnostics']['coverage_issues'][0]['id']=='obstacle'


def test_missing_image_only_blocks_local_full_inflated_cube_and_checks_claim():
    near=fixture([object_record(current=False)])
    assert assess(near)['reason']=='scene_local_metric_region_lacks_current_image_support'
    # A claimed disjoint list cannot silently remove an intersecting robot halo.
    near[2]['geometrically_disjoint_ids']=['obstacle']
    assert assess(near)['reason']=='scene_disjoint_claim_not_supported_by_full_bounds'
    far=fixture([object_record(center=(1.,1.,1.),current=False)])
    far[2].update(applied_metric_ids=[],applied_image_ids=[],geometrically_disjoint_ids=['obstacle'])
    far[3].region_geometry=[];far[3].translation_guard.neighbors=[]
    result=assess(far)
    assert result['accepted'] and result['diagnostics']['metric_only_disjoint_ids']==['obstacle']


def test_fresh_untextured_declaration_is_not_confused_with_stale_memory():
    values=fixture();view=values[2]['objects'][0]['views']['external']
    view.update(reliable=False,status='current_model_declaration')
    assert assess(values)['accepted']  # Tracking/continuous collision checks remain separate.
    view.update(current_roi_uv=None,status='stale')
    assert not assess(values)['accepted']


def test_image_only_near_tool_or_approach_target_requires_localization_not_occupancy():
    near_tool=fixture([object_record(geometry=False,roi=(.28,.25,.4,.4))])
    result=assess(near_tool)
    assert result['reason']=='scene_visible_region_needs_metric_localization'
    assert result['diagnostics']['coverage_issues'][0]['near_tool_image']
    target=object_record('part',center=(.05,0,0),roi=(.7,.7,.8,.8))
    near_target=fixture([target,object_record(geometry=False,roi=(.79,.72,.9,.82))])
    result=assess(near_target)
    assert result['reason']=='scene_visible_region_needs_metric_localization'
    assert result['diagnostics']['coverage_issues'][0]['near_approach_target_image']
    far=fixture([object_record(geometry=False,roi=(.85,.85,.95,.95))])
    result=assess(far)
    assert result['accepted'] and result['diagnostics']['unresolved_far_image_ids']==['obstacle']
    assert not result['diagnostics']['clearance_certified']


@pytest.mark.parametrize('phase',['approach','align','recover'])
def test_near_target_toward_motion_hands_off_but_away_tangent_and_far_do_not(phase):
    values=fixture();values[4]['phase']=phase
    result=assess(values,(.005,0,0))
    assert result['reason']=='near_target_approach_requires_current_two_view_contact'
    assert not result['diagnostics']['near_contact']['learned_response_required']
    assert assess(values,(-.005,0,0))['accepted']
    assert assess(values,(0,.001,0))['accepted']
    values[1][0]=-.2
    assert assess(values,(.005,0,0))['accepted']


def test_live_seeded_two_view_alignment_bootstraps_without_a_trained_response():
    values=fixture();module,monitor,state,descriptor=contact_fixture()
    values[4]['contact_alignment']=descriptor
    assert module.active and not module.response_ready
    result=assess(values,(.005,0,0),monitor=monitor,contact_alignment=module)
    assert result['accepted'] and result['diagnostics']['near_contact']['current_two_view_descriptor']
    # A non-null wire descriptor cannot stand in for lost current evidence.
    monitor.tracks['wrist_target'].reliable=False
    module.observe(monitor,state,[0]*7)
    assert not assess(values,(.005,0,0),monitor=monitor,contact_alignment=module)['accepted']
    assert not assess(values,(.005,0,0),contact_alignment=descriptor)['accepted']


def test_direct_uses_current_cube_and_does_not_borrow_field_contact_license():
    values=fixture();_,_,mat,field,pred=values
    pred.update(controller='direct',contact_alignment=None)
    field.origin=np.array([10.,10.,10.]);field.region_geometry=[]
    assert assess(values,(.005,0,0))['accepted']  # Existing Direct hard/contact/translation checks still required.
    mat['objects'][0]['views']['external']['current_roi_uv']=None
    assert assess(values,(.005,0,0))['reason']=='scene_local_metric_region_lacks_current_image_support'


@pytest.mark.parametrize('controller',['direct','gripper'])
def test_zero_translation_closure_or_rotation_still_requires_scene_coverage(controller):
    values=fixture([object_record(geometry=False,roi=(.28,.25,.4,.4))])
    values[4]['controller']=controller
    assert assess(values,(0,0,0))['reason']=='pose_hold_no_scene_clearance_claim'
    result=assess(values,(0,0,0),requires_scene_coverage=True)
    assert result['reason']=='scene_visible_region_needs_metric_localization'
    assert not result['accepted']


def test_current_wrist_box_does_not_invent_external_localization_for_near_world_region():
    values=fixture([object_record(current=False)])
    values[4]['controller']='gripper'
    values[2]['objects'][0]['views']['wrist']={'current_roi_uv':[.1,.1,.3,.3],'reliable':True}
    assert assess(values,(0,0,0),requires_scene_coverage=True)['reason']=='scene_local_metric_region_lacks_current_image_support'


def test_admission_is_pure_preserves_command_and_never_acknowledges_or_changes_budget():
    values=fixture();guard,state,mat,field,pred=values
    field.pending={'action':[.1,0,0,0,0,0,0]};field.translation_guard.observation_travel_m=.017
    before=pickle.dumps((state,mat,field,pred,guard.__dict__))
    for command in ([.005,0,0],[-.005,0,0],[0,0,0]):
        cmd=np.array(command,float);original=cmd.copy()
        result=assess(values,cmd)
        assert np.array_equal(cmd,original)
        assert not result['diagnostics']['budgets_modified']
        assert not result['diagnostics']['command_modified']
        assert pickle.dumps((state,mat,field,pred,guard.__dict__))==before


def test_saved_v15_terminal_descent_requests_contact_observation_before_execution():
    path=Path(__file__).resolve().parents[1]/'runs/visual_field_v15_zero_shot_001/trials/bowl_original_zero_hybrid_v15/trace.jsonl'
    if not path.exists():pytest.skip('Optional immutable V15 archive not present')
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    action=next(r for r in rows if r.get('event')=='action' and r['step']==235)
    feedback=next(r['feedback'] for r in rows if r.get('event')=='feedback' and r['step']==235)
    pred=next(r['diagnostics']['prediction'] for r in rows if r.get('event')=='action' and r['step']==229)
    saved=action['diagnostics']['selection']['diagnostics'];target=saved['target_contact_geometry']
    # Coverage is isolated here: its missing-distractor counterexample has a
    # separate model-inventory replay. Only saved target/core/robot data enter.
    field=SimpleNamespace(origin=np.array(saved['origin_xyz']),grid=SimpleNamespace(half_width_m=.08),
        region_geometry=[],translation_guard=None,target_configuration=target)
    result=SceneCoverageGuardV16().assess(feedback['before_state'],np.array(action['action'][:3])*.05,
        materialization={'objects':[]},field=field,prediction=pred)
    assert result['reason']=='near_target_approach_requires_current_two_view_contact'
    assert result['diagnostics']['near_contact']['end_core_distance_m']==pytest.approx(.03147807663779856)
    assert result['diagnostics']['near_contact']['handoff_band_m']==pytest.approx(.07)
    assert pred['contact_alignment'] is None
