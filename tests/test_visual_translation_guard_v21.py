"""Finite reverse recovery uses actual motion and preserves every risk ledger."""
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, decode_image
from gpt_field.visual_field_v21 import MetricFieldExecutorV21
from gpt_field.visual_translation_guard_v21 import ProjectedTranslationGuardV21
from gpt_field.visual_monitor_v12 import RoleAwareVisualMonitorV12
from gpt_field.visual_field_v16 import InventoryTargetGeometryV16
from gpt_field.visual_scene_guard_v17 import SceneCoverageGuardV17
from test_visual_translation_guard_v11 import pixels, context
from test_visual_field_v6 import axis_intent
from test_visual_field_v4 import Q, state


def setup(state8, images, refs, ctx, intent, material):
    monitor=RoleAwareVisualMonitorV12();monitor.configure(images,refs,state8,intent_id=1)
    guard=ProjectedTranslationGuardV21();guard.configure(images,ctx,state8,target_id=material['target_id'],monitor=monitor)
    field=MetricFieldExecutorV21();contact=InventoryTargetGeometryV16()
    contact.configure(state8,material['target_geometry'],target_id=material['target_id'],tool_radius_m=intent['tool_radius_m'])
    field.contact_geometry=contact;field.bind_translation_guard(guard,monitor)
    field.configure(state8,intent,scene_assessment=ctx,inventory_materialization=material)
    pred={'controller':'visual_field','phase':'recover','target_id':material['target_id'],'field_intent':intent}
    scene=SceneCoverageGuardV17()
    field.scene_assessor=lambda current,delta:scene.assess(current,delta,materialization=material,
        field=field,monitor=monitor,prediction=pred)
    return guard,monitor,field


def plant():
    measured=state();image=pixels();images={'external':image,'wrist':image}
    refs={'external':{'update':'seed','moving_id':'tool','moving_kind':'tool','target_id':'neighbor',
        'goal_offset_uv':[0.,0.],'seed':{'moving_point_uv':[55/223,65/223],
        'moving_roi_uv':[43/223,53/223,67/223,77/223],'target_anchor_uv':[110/223,65/223],
        'target_roi_uv':[98/223,53/223,122/223,77/223],'tool_point_uv':None,'tool_roi_uv':None}},'wrist':None}
    ctx=context(tool=(43,53,90,77));intent=axis_intent('xyz',tool_radius_m=.045,uncertainty_m=.035,max_actions=120)
    geometry={'id':'neighbor','center_offset_m':[.23,0,0],'half_extents_m':[.02]*3,
        'uncertainty_m':.06,'allowed_contact':'none'}
    record={'id':'neighbor','world_geometry':{'center_xyz_m':(measured[:3]+[.23,0,0]).tolist(),
        'half_extents_m':[.02]*3,'uncertainty_m':.06},'views':{'external':{'current_roi_uv':[98/223,53/223,122/223,77/223]}}}
    material={'target_id':'neighbor','target_geometry':geometry,'objects':[record],'obstacles':[],
        'applied_metric_ids':['neighbor'],'applied_image_ids':['neighbor']}
    guard,monitor,field=setup(measured,images,refs,ctx,intent,material)
    return guard,monitor,field,measured


def assess(guard,monitor,field,command,**changes):
    args={'monitor':monitor,'nominal_clear':field.checked.segment_valid(guard.state[:3],guard.state[:3]+command),
        'nominal_away':field._nominal_away(guard.state[:3],command),
        'nominal_regions':field.nominal_regions_for_segment(guard.state[:3],guard.state[:3]+command)}
    args.update(changes)
    return guard.assess(command,**args)


def observe(guard,monitor,measured,command,actual,*,pixel_x):
    after=measured.copy();after[:3]+=actual;action=np.r_[np.asarray(command)/.05,[0.,0.,0.,0.]]
    images={'external':pixels(pixel_x),'wrist':pixels(pixel_x)}
    monitor.observe(images,after,action);guard.observe(images,after,action,monitor=monitor)
    return after


def triggered():
    guard,monitor,field,measured=plant()
    for i in range(1,14):
        choice=assess(guard,monitor,field,np.array([.002,0,0]));assert choice['accepted'],choice
        measured=observe(guard,monitor,measured,[.002,0,0],[.001,0,0],pixel_x=i)
        if guard.worsening:break
    assert guard.worsening=={'neighbor'} and guard.worsening_motion
    guard.context={**guard.context,'translation_mode':'separate'}
    return guard,monitor,field,measured,i


def test_actual_rgb_risk_window_offers_reverse_not_continuing_and_keeps_worsening():
    guard,monitor,field,measured,_=triggered()
    choice=guard.recovery_directions()[0]
    assert choice['direction_xyz']==pytest.approx([-1.,0,0])
    assert choice['source']=='reverse_observed_worsening_motion' and not choice['nominal_away_required']
    assert not assess(guard,monitor,field,np.array([.002,0,0]))['accepted']
    accepted=assess(guard,monitor,field,np.array([-.003,0,0]))
    assert accepted['accepted'] and accepted['reason']=='projected_worsening_bounded_observed_motion_reverse'
    assert guard.worsening=={'neighbor'} and guard.experiments==[] and guard.observation_travel_m==0.


def test_observed_reverse_charges_actual_debt_and_can_demonstrate_gap_progress():
    guard,monitor,field,measured,pixel=triggered();start=measured.copy()
    for step in range(1,7):
        selected=assess(guard,monitor,field,np.array([-.002,0,0]))
        assert selected['accepted'],selected
        actual=min(.001,selected['cap_m']*.5)
        command=[-min(.002,selected['cap_m']),0,0]
        measured=observe(guard,monitor,measured,command,[-actual,0,0],pixel_x=pixel-step)
    assert guard.observation_travel_m==pytest.approx(np.linalg.norm(measured[:3]-start[:3]))
    assert len(guard.experiments)==1 and guard.experiments[0]['measured_travel_m']>0
    assert guard.retreat_progress['all_trigger_gaps_improved']
    assert guard.retreat_progress['acknowledged_commands']==6


def test_preview_and_downstream_hold_do_not_create_experiment_or_renew_budget():
    guard,monitor,field,measured,pixel=triggered();command=np.array([-.003,0,0])
    records=field.nominal_regions_for_segment(measured[:3],measured[:3]+command)
    before=pickle.dumps(guard);track=guard.tool_track
    preview=guard.preview_assess(command,monitor=monitor,nominal_clear=True,nominal_away=False,
        nominal_regions=records,prospective_translation_mode='separate')
    assert preview['accepted'] and pickle.dumps(guard)==before and guard.tool_track is track
    choice=assess(guard,monitor,field,command,nominal_away=False);assert choice['accepted']
    observe(guard,monitor,measured,[0,0,0],[0,0,0],pixel_x=pixel)
    assert guard.experiments==[] and guard.retreat_progress is None
    assert guard.observation_travel_m==0.


@pytest.mark.parametrize('fault',['geometry_changed','unknown','payload','contact','epoch','contradiction'])
def test_reverse_never_overrides_new_geometry_role_or_reference_constraints(fault):
    guard,monitor,field,measured,_=triggered();command=np.array([-.003,0,0])
    records=field.nominal_regions_for_segment(measured[:3],measured[:3]+command)
    c=records[0]['directional_retreat_contract']
    if fault=='geometry_changed':c['geometry_sha256']='changed-model-hypothesis'
    elif fault=='unknown':records[0]['kind']='unknown_regions'
    elif fault=='payload':c['empty_tool']=False
    elif fault=='contact':c['target_contact']='fingers'
    elif fault=='epoch':c['observation_epoch']-=1
    elif fault=='contradiction':monitor.cumulative_response_evidence=lambda:{'contradicted':True}
    result=guard.assess(command,monitor=monitor,nominal_clear=True,nominal_away=False,nominal_regions=records)
    assert not result['accepted'] and guard.experiments==[] and guard.observation_travel_m==0.


def archive_trigger():
    path=Path(__file__).resolve().parents[1]/'runs/visual_field_v20_zero_shot_001/trials/bowl_original_zero_hybrid_v20/trace.jsonl'
    if not path.exists():pytest.skip('Optional V20 archive unavailable')
    rows=[json.loads(line)for line in path.read_text().splitlines()]
    actions={r['step']:r for r in rows if r['event']=='action'}
    feedback={r['step']:r['feedback']for r in rows if r['event']=='feedback'}
    obs=PolicyObservation.from_dict(next(r['observation']for r in rows if r['event']=='executed' and r['step']==156))
    pred=copy.deepcopy(actions[137]['diagnostics']['prediction']);fb=feedback[156]
    material=copy.deepcopy(actions[137]['diagnostics']['scene_materialization'])
    for row in material['objects']:
        if row['world_geometry'] is None:continue
        row['current_geometry']={**row['world_geometry'],'center_offset_m':(np.asarray(row['world_geometry']['center_xyz_m'])-obs.state[:3]).tolist()}
    target=next(r for r in material['objects']if r['id']==pred['target_id'])['world_geometry']
    material['target_geometry']={'id':pred['target_id'],'center_offset_m':(np.array(target['center_xyz_m'])-obs.state[:3]).tolist(),
        'half_extents_m':target['half_extents_m'],'uncertainty_m':target['uncertainty_m'],'allowed_contact':'none'}
    refs=copy.deepcopy(pred['visual_references']);track=fb['visual']['tracking'];refs['external']['update']='seed'
    refs['external']['seed'].update(moving_point_uv=track['moving']['tracked_point_uv'],moving_roi_uv=track['moving']['current_roi_uv'],
        target_anchor_uv=track['target']['tracked_point_uv'],target_roi_uv=track['target']['current_roi_uv'])
    ctx=copy.deepcopy(fb['effective_motion_context']);ctx['translation_mode']='separate';ctx['intended_contact_id']=None
    ctx['tool_external_roi_uv']=(np.asarray(fb['translation_guard']['tool_box_px'])/223).tolist()
    images={'external':np.asarray(decode_image(obs.image)),'wrist':np.asarray(decode_image(obs.wrist_image))}
    intent=copy.deepcopy(pred['field_intent']);intent['goal_offset_m']=[.03,.03,.03]
    from gpt_field.action import _rotvec_quaternion
    intent['orientation_target_xyzw']=_rotvec_quaternion(np.asarray(obs.state[3:6])).tolist()
    guard,monitor,field=setup(obs.state,images,refs,ctx,intent,material)
    # Reconstruct the historical trigger from acknowledged saved states and
    # recorded risk measurements; no invented live observation or success.
    delta=np.asarray(fb['after_state'][:3])-feedback[136]['after_state'][:3]
    contract=field.nominal_regions_for_segment(obs.state[:3],np.asarray(obs.state[:3])+[.001,0,0])[0]['directional_retreat_contract']
    guard.worsening=set(fb['translation_guard']['risk_progress']['worsening_ids'])
    for identity in guard.worsening:
        guard.worsening_motion[identity]={'measured_direction_xyz':(delta/np.linalg.norm(delta)).tolist(),
            'measured_net_translation_m':float(np.linalg.norm(delta)),'state8':list(obs.state),
            'geometry_sha256':contract['geometry_sha256'],'risk_evidence':fb['translation_guard']['risk_progress']['per_neighbor'][identity]}
    return guard,monitor,field,np.asarray(obs.state),delta/np.linalg.norm(delta)


def test_actual_v20_worsening_reverse_is_admitted_but_continuing_is_not():
    guard,monitor,field,measured,direction=archive_trigger()
    reverse=-direction*.003
    assert not field._nominal_away(measured[:3],reverse)  # Cabinet model distance shrinks.
    forward=assess(guard,monitor,field,direction*.003)
    assert not forward['accepted']
    chosen=assess(guard,monitor,field,reverse)
    assert chosen['accepted'],chosen
    assert chosen['reason']=='projected_worsening_bounded_observed_motion_reverse'
    assert chosen['diagnostics']['directional_retreat']['diagnostics']['nominal_away'] is False
    assert chosen['cap_m']<=.003 and guard.observation_travel_m==0.
    assert guard.worsening=={'drawer_cabinet','hotplate','ramekin'}
    assert field.checked.segment_valid(measured[:3],measured[:3]+reverse)


def test_archived_reverse_is_offered_by_policy_even_with_previous_unrelated_long_goal():
    from gpt_field.visual_field_policy_v21 import VisualFieldPolicyV21

    guard,monitor,field,measured,direction=archive_trigger()
    # Exercise the actual candidate-screen method with the archived perception
    # and geometry boundary, without issuing an inference request or action.
    policy=object.__new__(VisualFieldPolicyV21)
    policy.local,policy.translation_guard,policy.monitor=field,guard,monitor
    policy.robot_response=None
    policy.step_index=157
    policy.recovery_options={'candidates':[],'rejected_candidates':[],
        'preview_scope':'archived current observation'}
    policy._scene_assess=field.scene_assessor
    previous_goal=field.goal.copy()
    previous_debt=guard.observation_travel_m
    policy._screen_recovery_options(measured)
    offers=policy.recovery_options['candidates']
    assert len(offers)==1,policy.recovery_options
    offer=offers[0]
    assert offer['source']=='reverse_observed_worsening_motion'
    assert offer['nominal_away_required'] is False
    assert np.allclose(offer['recommended_goal_offset_m'],-direction*.008)
    assert offer['recommended_goal_tolerance_m']==.003
    assert offer['preview_reason']=='projected_worsening_bounded_observed_motion_reverse'
    assert offer['scene_coverage_preview']['accepted']
    assert np.array_equal(field.goal,previous_goal)
    assert guard.observation_travel_m==previous_debt
    assert guard.experiments==[] and guard._retreat_pending is None


def test_reversal_without_projected_progress_stops_at_finite_checkpoint():
    guard,monitor,field,measured,pixel=triggered()
    commands=0
    for _ in range(12):
        selected=assess(guard,monitor,field,np.array([-.002,0,0]))
        if not selected['accepted']:break
        cap=selected['cap_m']; actual=min(.001,cap)
        measured=observe(guard,monitor,measured,[-cap,0,0],[-actual,0,0],pixel_x=pixel)
        commands+=1
    else:pytest.fail('Unresolved reverse must not continue without a finite checkpoint')
    assert commands<=4
    assert selected['reason'] in ('translation_reverse_checkpoint_requires_observed_gap_progress',
                                  'translation_reverse_checkpoint_has_no_useful_command',
                                  'translation_reverse_reference_unavailable_or_contradicted')
    assert guard.observation_travel_m<=.004+1e-10
    assert guard.worsening=={'neighbor'} and not guard.retreat_progress['all_trigger_gaps_improved']


def test_paid_four_mm_checkpoint_cannot_be_renewed_when_gap_evidence_is_inconclusive():
    guard,monitor,field,measured,pixel=triggered()
    command=np.array([-.002,0,0]);assert assess(guard,monitor,field,command)['accepted']
    measured=observe(guard,monitor,measured,command,[-.001,0,0],pixel_x=pixel)
    # Isolate the persisted ledger boundary: neither a new direction nor a new
    # configure may grant the same paid four mm again while evidence is weak.
    guard.retreat_progress['measured_travel_m']=.004
    guard.retreat_progress['all_trigger_gaps_improved']=False
    guard.experiments[0]['measured_travel_m']=.004
    guard.observation_travel_m=.004
    selected=assess(guard,monitor,field,command)
    assert selected['reason']=='translation_reverse_checkpoint_requires_observed_gap_progress'
    assert guard.observation_travel_m==.004 and len(guard.experiments)==1


def test_directional_allowance_is_not_renewed_by_preview_or_repeated_separate_declaration():
    guard,monitor,field,measured,pixel=triggered()
    command=np.array([-.002,0,0]);selected=assess(guard,monitor,field,command)
    measured=observe(guard,monitor,measured,command,[-.001,0,0],pixel_x=pixel-1)
    debt=guard.observation_travel_m
    for _ in range(3):
        guard.context={**guard.context,'translation_mode':'separate'}
        selected=assess(guard,monitor,field,command)
        assert selected['accepted']
    assert guard.observation_travel_m==debt and len(guard.experiments)==1
    # The provisional selection was vetoed by a later gate: its residual hold
    # belongs to the previously executed uncertain reverse, not a new trial.
    measured=observe(guard,monitor,measured,[0,0,0],[-.0002,0,0],pixel_x=pixel-1)
    assert guard.observation_travel_m==pytest.approx(debt+.0002)
    assert guard.retreat_progress['measured_travel_m']==pytest.approx(debt+.0002)
    assert len(guard.experiments)==1


def test_equivalent_axis_angle_wrap_does_not_invalidate_recorded_motion_domain():
    guard,monitor,field,measured,_=triggered()
    # The record is a public measured-state boundary: 2pi around X represents
    # exactly the current identity orientation, unlike Euclidean rotvec delta.
    guard.worsening_motion['neighbor']['state8'][3]=2*np.pi
    result=assess(guard,monitor,field,np.array([-.002,0,0]))
    assert result['accepted'],result


def test_archived_reverse_flows_through_real_field_cap_reselect_and_exact_acknowledgement():
    guard,monitor,field,measured,direction=archive_trigger()
    intent=copy.deepcopy(field.intent)
    intent.update(goal_offset_m=(-direction*.008).tolist(),goal_tolerance_m=.003)
    field.configure(measured,intent,scene_assessment=guard.context,
                    inventory_materialization=field._transit_materialization)
    selected=field.select(measured,Q)
    assert selected['accepted'],selected
    delta=np.asarray(selected['action'][:3])*.05
    assert 0 < np.linalg.norm(delta)<=.003+1e-12
    assert np.dot(delta/np.linalg.norm(delta),-direction)>np.cos(np.pi/6)
    assert np.array_equal(field.pending[1],selected['action'])
    assert field.orientation_goal_refresh_commands==1
    wrong=np.asarray(selected['action']).copy();wrong[0]+=.001
    with pytest.raises(ValueError,match='match the selected Field command'):
        field.observe(measured,wrong,measured)
    assert field.pending is not None
    # Synthetic acknowledgment tests controller accounting only; it does not
    # pretend this counterfactual command was executed in the archived trial.
    after=measured.copy();after[:3]+=delta*.3
    field.observe(measured,selected['action'],after)
    assert field.pending is None and field.actions==1


def test_actual_hard_core_at_current_pose_blocks_reverse_before_any_exception():
    guard,monitor,field,measured,direction=archive_trigger()
    material=copy.deepcopy(field._transit_materialization)
    obstacle=next(row for row in material['objects']if row['id']=='drawer_cabinet')
    obstacle['world_geometry']['center_xyz_m']=measured[:3].tolist()
    field.configure(measured,field.intent,scene_assessment=guard.context,inventory_materialization=material)
    assert not field.checked.segment_valid(measured[:3],measured[:3]-direction*.003)
    rejected=field.select(measured,Q)
    assert not rejected['accepted'] and rejected['status']=='invalid_measured_start'
    assert field.pending is None and field.actions==0


@pytest.mark.parametrize('intervening',['metadata_free_selection','downstream_hold'])
def test_new_worsening_cannot_borrow_unacknowledged_or_stale_geometry(intervening):
    guard,monitor,field,measured=plant();command=np.array([.002,0,0])
    assert assess(guard,monitor,field,command)['accepted']
    measured=observe(guard,monitor,measured,command,[.001,0,0],pixel_x=1)
    assert not guard.worsening
    assert assess(guard,monitor,field,command)['accepted']
    if intervening=='metadata_free_selection':
        guard.assess(command,monitor=monitor,nominal_clear=True,nominal_regions=[])
        actual_command=command
    else:
        actual_command=[0.,0.,0.]
    measured=observe(guard,monitor,measured,actual_command,[.001,0,0],pixel_x=2)
    assert guard.worsening=={'neighbor'}
    assert guard.worsening_motion=={} and guard.recovery_directions()==[]
