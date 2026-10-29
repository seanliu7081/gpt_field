"""Exact whole-tool model sweep can enable progress without clearing debt."""
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, decode_image
from gpt_field.visual_field_v19 import MetricFieldExecutorV19
from gpt_field.visual_scene_guard_v17 import SceneCoverageGuardV17, segment_box_interval
from gpt_field.visual_translation_guard_v19 import ProjectedTranslationGuardV19
from gpt_field.visual_monitor_v12 import RoleAwareVisualMonitorV12
from test_visual_translation_guard_v11 import plant as old_plant, pixels, context
from test_visual_translation_guard_v14 import observe


def bind_geometry(guard, state, objects, prediction):
    field = MetricFieldExecutorV19()
    field.translation_guard = guard
    field.region_geometry = []
    for record in objects:
        bare = copy.deepcopy(record);bare['world_geometry']['uncertainty_m']=0.
        zero = copy.deepcopy(prediction);zero['field_intent']['uncertainty_m']=0.
        parts = SceneCoverageGuardV17._sweep_boxes(bare,state,zero,.04)
        sigma=record['world_geometry']['uncertainty_m']+prediction['field_intent']['uncertainty_m']
        field.region_geometry.append({'id':record['id'],'kind':'obstacles',
            'tool_core_parts':parts,'margins_m':{'combined_uncertainty':sigma}})
    material={'objects':objects,'applied_metric_ids':[r['id'] for r in objects],
        'applied_image_ids':[r['id'] for r in objects]}
    scene=SceneCoverageGuardV17()
    field.scene_assessor=lambda current,delta:scene.assess(current,delta,
        materialization=material,field=field,prediction=prediction)
    return field


def plant():
    ctx=context(tool=(43,53,125,77))
    _,monitor,state=old_plant(ctx)
    guard=ProjectedTranslationGuardV19()
    guard.configure({'external':pixels(),'wrist':pixels()},ctx,state,target_id='target',monitor=monitor)
    objects=[{'id':'neighbor','world_geometry':{'center_xyz_m':[.25,0,0],
        'half_extents_m':[.02]*3,'uncertainty_m':.06},
        'views':{'external':{'current_roi_uv':[98/223,53/223,122/223,77/223]}}}]
    pred={'controller':'visual_field','phase':'transport','target_id':'target',
        'field_intent':{'uncertainty_m':.035,'tool_radius_m':.045}}
    return guard,monitor,state,bind_geometry(guard,state,objects,pred)


def select(guard,monitor,field,command,**kwargs):
    defaults={'nominal_clear':True,'nominal_regions':field.nominal_regions_for_segment(
        guard.state[:3],guard.state[:3]+command)}
    defaults.update(kwargs)
    return guard.assess(command,monitor=monitor,**defaults)


def test_exact_sweep_allows_motion_despite_exhausted_uncertain_budget_without_reset():
    guard,monitor,state,field=plant();guard.observation_travel_m=.02
    command=np.array([.002,0,0]);records=field.nominal_regions_for_segment(state[:3],state[:3]+command)
    legacy=copy.deepcopy(records)
    for row in legacy:row.pop('model_uncertainty_sweep')
    assert guard.assess(command,monitor=monitor,nominal_clear=True,nominal_regions=legacy)['reason']=='translation_observation_budget_exhausted'
    result=select(guard,monitor,field,command)
    assert result['accepted'] and result['reason']=='projected_model_uncertainty_sweep_disjoint'
    assert result['cap_m']==pytest.approx(.002)
    assert guard.observation_travel_m==.02 and guard.experiments==[]
    # The proof exists only during admission, not ordinary image monitoring.
    assert guard._near_metric('neighbor') and guard._admission_sweep_ids is None
    assert not result['diagnostics']['clearance_certified']


@pytest.mark.parametrize('corrupt',['endpoint','state_orientation','epoch','rotation','part_count','scope','missing'])
def test_stale_or_incomplete_proof_cannot_bypass_exhausted_budget(corrupt):
    guard,monitor,state,field=plant();guard.observation_travel_m=.02
    command=np.array([.002,0,0]);records=field.nominal_regions_for_segment(state[:3],state[:3]+command)
    proof=records[0]['model_uncertainty_sweep']
    if corrupt=='endpoint':proof['end_xyz_m'][1]+=.001
    if corrupt=='state_orientation':proof['state8'][3]+=.001
    if corrupt=='epoch':proof['observation_epoch']-=1
    if corrupt=='rotation':proof['orientation_margin_rad']=.001
    if corrupt=='part_count':proof['configuration_part_count']=1
    if corrupt=='scope':proof['coverage_scope']='whole_grid'
    if corrupt=='missing':records[0].pop('model_uncertainty_sweep')
    result=guard.assess(command,monitor=monitor,nominal_clear=True,nominal_regions=records)
    assert not result['accepted'] and guard.observation_travel_m==.02


def test_full_halo_intersection_is_not_an_exemption_and_hard_collision_still_rejects():
    guard,monitor,state,field=plant();command=np.array([.002,0,0])
    record={'id':'neighbor','world_geometry':{'center_xyz_m':[.13,0,0],
        'half_extents_m':[.02]*3,'uncertainty_m':.06},
        'views':{'external':{'current_roi_uv':[98/223,53/223,122/223,77/223]}}}
    pred={'controller':'visual_field','phase':'transport','target_id':'target',
        'field_intent':{'uncertainty_m':.035,'tool_radius_m':.045}}
    field=bind_geometry(guard,state,[record],pred)
    result=select(guard,monitor,field,command)
    assert result['accepted'] and result['reason']=='projected_risk_finite_directional_experiment'
    assert result['diagnostics']['model_uncertainty_sweep_admission']['per_object']['neighbor']=='intersects_no_exemption'
    assert not select(guard,monitor,field,command,nominal_clear=False)['accepted']


def test_observed_worsening_or_lost_tool_cannot_be_overridden_by_model_clearance():
    guard,monitor,state,field=plant();command=np.array([.002,0,0])
    guard.worsening.add('neighbor')
    assert select(guard,monitor,field,command)['reason']=='translation_observed_neighbor_approach'
    guard.worsening.clear();monitor.tracks['moving'].reliable=False
    assert select(guard,monitor,field,command)['reason']=='translation_tool_correspondence_lost'


def test_exact_preview_preserves_pending_ack_and_then_real_observe_invalidates_old_proof():
    guard,monitor,state,field=plant();command=np.array([.002,0,0])
    records=field.nominal_regions_for_segment(state[:3],state[:3]+command)
    accepted=select(guard,monitor,field,command)
    original=pickle.dumps(guard);track=guard.tool_track
    preview=guard.preview_assess(command,monitor=monitor,nominal_clear=True,nominal_regions=records)
    assert preview['accepted']==accepted['accepted'] and preview['reason']==accepted['reason']
    assert pickle.dumps(guard)==original and guard.tool_track is track
    state=observe(guard,monitor,state,command,[.001,0,0],pixel_x=1)
    assert guard._admission_sweep_ids is None
    guard.observation_travel_m=.02
    old=guard.assess(command,monitor=monitor,nominal_clear=True,nominal_regions=records)
    assert not old['accepted']


def test_downstream_veto_hold_retains_previous_uncertain_debt_classification():
    guard,monitor,state,field=plant();command=np.array([.002,0,0])
    records=field.nominal_regions_for_segment(state[:3],state[:3]+command)
    for row in records:row.pop('model_uncertainty_sweep')
    assert guard.assess(command,monitor=monitor,nominal_clear=True,nominal_regions=records)['accepted']
    state=observe(guard,monitor,state,command,[.001,0,0],pixel_x=1)
    debt=guard.observation_travel_m
    assert select(guard,monitor,field,command)['accepted']
    state=observe(guard,monitor,state,[0,0,0],[.0002,0,0],pixel_x=1)
    assert guard.observation_travel_m==pytest.approx(debt+.0002)
    assert len(guard.experiments)==1


def test_repeated_actual_rgb_transitions_reuse_model_clear_field_beyond_uncertain_budget():
    guard,monitor,state,field=plant();guard.observation_travel_m=.02
    for i in range(1,18):
        choice=select(guard,monitor,field,np.array([.002,0,0]))
        assert choice['accepted'],choice['reason']
        state=observe(guard,monitor,state,[.002,0,0],[.001,0,0],pixel_x=-i)
    assert state[0]==pytest.approx(.017)
    assert guard.observation_travel_m==pytest.approx(.02)
    assert guard.experiments==[]


def test_already_remote_field_keeps_original_command_cap():
    guard,monitor,state,field=plant();command=np.array([.01,0,0])
    records=field.nominal_regions_for_segment(state[:3],state[:3]+command)
    records[0]['core_distance_m']=1.
    result=guard.assess(command,monitor=monitor,nominal_clear=True,nominal_regions=records)
    assert result['accepted'] and result['cap_m'] is None
    assert result['reason']=='projected_risk_no_near_conflict'
    assert not result['diagnostics']['model_uncertainty_sweep_admission']['branch_exercised']


@pytest.mark.parametrize('bad_end',[None,[],[0,0],['bad',0,0]])
def test_malformed_scene_callback_does_not_mint_sweep_proof(bad_end):
    guard,monitor,state,field=plant()
    field.scene_assessor=lambda *_:{'accepted':True,'diagnostics':{
        'coverage_scope':'actual_commanded_whole_tool_sweep','start_xyz_m':state[:3].tolist(),
        'end_xyz_m':bad_end}}
    records=field.nominal_regions_for_segment(state[:3],state[:3]+[.002,0,0])
    assert all('model_uncertainty_sweep' not in row for row in records)


def test_carried_payload_full_halo_intersection_never_uses_bare_tool_clearance():
    guard,monitor,state,field=plant();command=np.array([.002,0,0])
    assert select(guard,monitor,field,command)['reason']=='projected_model_uncertainty_sweep_disjoint'
    # A locally confirmed retained payload envelope extends toward the fixture;
    # this fixture supplies the contact-geometry public boundary, not env truth.
    field.contact_geometry=SimpleNamespace(carried_summary=lambda _:{'active':True,
        'nominal_world':{'lower_xyz':[.11,-.01,-.01],'upper_xyz':[.16,.01,.01]},
        'nominal_hand_frame':{'lower_xyz':[.11,-.01,-.01],'upper_xyz':[.16,.01,.01]},
        'uncertainty_m':.01})
    result=select(guard,monitor,field,command)
    assert result['reason']=='projected_risk_finite_directional_experiment'
    assert result['diagnostics']['model_uncertainty_sweep_admission']['per_object']['neighbor']=='intersects_no_exemption'


def test_current_pose_refresh_invalidates_previous_geometry_proof():
    guard,monitor,state,field=plant();command=np.array([.002,0,0])
    records=field.nominal_regions_for_segment(state[:3],state[:3]+command)
    guard.configure({'external':pixels(),'wrist':pixels()},context(tool=(43,53,125,77)),
                    state,target_id='target',monitor=monitor)
    guard.observation_travel_m=.02
    assert not guard.assess(command,monitor=monitor,nominal_clear=True,nominal_regions=records)['accepted']
    assert select(guard,monitor,field,command)['accepted']


def archive_blocked_proposal():
    root=Path(__file__).resolve().parents[1]
    path=root/'runs/visual_field_v18_zero_shot_001/trials/bowl_original_zero_hybrid_v18/trace.jsonl'
    if not path.exists():pytest.skip('Optional archived V18 trace unavailable')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    action=next(r for r in rows if r['event']=='action' and r['step']==10)
    feedback=next(r['feedback'] for r in rows if r['event']=='feedback' and r['step']==9)
    obs=PolicyObservation.from_dict(next(r['observation'] for r in rows if r['event']=='executed' and r['step']==9))
    pred=copy.deepcopy(next(r['diagnostics']['prediction'] for r in rows if r['event']=='action' and r['step']==7))
    track=feedback['visual']['tracking'];refs=copy.deepcopy(pred['visual_references'])
    refs['external']['update']='seed'
    refs['external']['seed'].update(moving_point_uv=track['moving']['tracked_point_uv'],
        moving_roi_uv=track['moving']['current_roi_uv'],target_anchor_uv=track['target']['tracked_point_uv'])
    images={'external':np.asarray(decode_image(obs.image)),'wrist':np.asarray(decode_image(obs.wrist_image))}
    monitor=RoleAwareVisualMonitorV12();monitor.configure(images,refs,obs.state,intent_id=100)
    context_value=copy.deepcopy(feedback['effective_motion_context'])
    context_value['tool_external_roi_uv']=(np.asarray(feedback['translation_guard']['tool_box_px'])/223).tolist()
    guard=ProjectedTranslationGuardV19();guard.configure(images,context_value,obs.state,
        target_id=pred['target_id'],monitor=monitor)
    guard.observation_travel_m=feedback['translation_guard']['observation_travel_m']
    guard.experiments=copy.deepcopy(feedback['translation_guard']['recovery_experiments']['trials'])
    guard._last_charged_experiment=0
    proposal=action['diagnostics']['selection']['diagnostics']['translation_admission']['diagnostics']['checked_proposal']
    objects=copy.deepcopy(feedback['scene_inventory']['entries'])
    field=bind_geometry(guard,np.asarray(obs.state),objects,pred)
    return guard,monitor,np.asarray(obs.state),field,np.asarray(proposal['delta_xyz_m']),objects,pred


def test_actual_v18_blocked_proposal_gets_bounded_current_sweep_admission():
    guard,monitor,state,field,command,objects,pred=archive_blocked_proposal()
    records=field.nominal_regions_for_segment(state[:3],state[:3]+command)
    old=copy.deepcopy(records)
    for row in old:row.pop('model_uncertainty_sweep',None)
    prior=guard.preview_assess(command,monitor=monitor,nominal_clear=True,nominal_regions=old)
    assert not prior['accepted'] and prior['reason']=='translation_measured_allowance_below_useful_command_floor'
    debt,experiments=guard.observation_travel_m,copy.deepcopy(guard.experiments)
    new=select(guard,monitor,field,command)
    assert new['accepted'] and new['reason']=='projected_model_uncertainty_sweep_disjoint'
    assert new['cap_m']==pytest.approx(.005)
    assert new['diagnostics']['model_uncertainty_sweep_admission']['materially_overridden_near_ids']==['drawer_cabinet']
    smaller=command/np.linalg.norm(command)*new['cap_m']
    assert select(guard,monitor,field,smaller)['accepted']
    assert guard.observation_travel_m==debt and guard.experiments==experiments
    # Counterfactual retained model object intersects the actual candidate.
    cabinet=next(row for row in objects if row['id']=='drawer_cabinet')
    cabinet['world_geometry']['center_xyz_m']=state[:3].tolist()
    crossed=bind_geometry(guard,state,objects,pred)
    core=next(row for row in crossed.region_geometry if row['id']=='drawer_cabinet')['tool_core_parts']
    core_clear=not any(segment_box_interval(state[:3],state[:3]+smaller,b['lower_xyz'],b['upper_xyz'])
                       is not None for b in core)
    assert not core_clear
    rejected=select(guard,monitor,crossed,smaller,nominal_clear=core_clear)
    assert not rejected['accepted']


def test_actual_executor_unknown_region_remains_hard_and_cannot_get_scene_exemption():
    from test_visual_field_v17_policy import prediction
    from test_visual_field_v13_execution import observation
    obs=observation();value=prediction();intent=copy.deepcopy(value['field_intent'])
    intent['unknown_regions']=[{'id':'unseen','lower_offset_m':[.01,-.01,-.01],
        'upper_offset_m':[.02,.01,.01],'uncertainty_m':.02}]
    field=MetricFieldExecutorV19()
    field.configure(obs.state,intent,scene_assessment=value['motion_context'],
                    inventory_materialization={'objects':[],'obstacles':[]})
    assert not field.checked.segment_valid(np.asarray(obs.state[:3]),np.asarray(obs.state[:3])+[.03,0,0])
    region=next(row for row in field.region_geometry if row['id']=='unseen')
    assert region['kind']=='unknown_regions'
