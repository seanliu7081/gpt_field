"""V23 wrappers can exempt only current, explicitly bound initial halo risk."""
import copy
import pickle
import numpy as np
import pytest

from gpt_field.visual_scene_guard_v23 import SceneCoverageGuardV23
from test_visual_field_v23 import archive_field


def active_archive():
    field,guard,monitor,inventory,obs,pred,assessor=archive_field()
    goal=field.egress_candidates(obs.state)[0]['goal_offset_m']
    intent=copy.deepcopy(field.intent)
    intent.update(goal_offset_m=goal,goal_tolerance_m=.003,command_limit_m=.003,max_travel_m=.04,max_actions=120)
    guard.context.update(translation_mode='separate',purpose='reference_repair',intended_contact_id=None)
    field.configure(obs.state,intent,scene_assessment=guard.context,inventory_materialization=field._transit_materialization)
    field.egress_context={'phase':'recover','purpose':'reference_repair','translation_mode':'separate'}
    pred=copy.deepcopy(pred);pred.update(phase='recover',field_intent=copy.deepcopy(intent),target_contact='none')
    pred['motion_context']=copy.deepcopy(guard.context)
    delta=np.asarray(goal)/np.linalg.norm(goal)*.002
    return field,guard,monitor,inventory,obs,pred,delta


def assess_guard(bundle):
    field,guard,monitor,inventory,obs,pred,delta=bundle
    proof=field.egress_contract(obs.state,delta)
    return guard.assess(delta,monitor=monitor,nominal_clear=True,nominal_away=True,
        nominal_regions=field.nominal_regions_for_segment(obs.state[:3],np.asarray(obs.state[:3])+delta),
        egress_contract=proof,rotation_rad=0.,orientation_hold=True)


def test_contextual_budget_uses_old_shared_debt_and_does_not_mutate_ledger_during_admission():
    bundle=active_archive();field,guard,*_=bundle
    guard.observation_travel_m=.020
    before=pickle.dumps(guard.egress_evidence.__dict__)
    result=assess_guard(bundle)
    assert result['accepted'],result['reason']
    assert result['cap_m']<=.003
    assert pickle.dumps(guard.egress_evidence.__dict__)==before
    guard.observation_travel_m=.040
    assert not assess_guard(bundle)['accepted']


@pytest.mark.parametrize('failure',['worsening','domain','lost','role','missing_wrist'])
def test_proof_cannot_override_current_visual_contradiction_or_support_requirement(failure):
    bundle=active_archive();field,guard,monitor,inventory,obs,pred,delta=bundle
    if failure=='worsening':guard.worsening={'drawer_cabinet'}
    elif failure=='domain':guard.domain_invalid='translation_scene_motion_inconsistent'
    elif failure=='lost':monitor.tracks['moving'].reliable=False
    elif failure=='role':guard.context['purpose']='task_translation'
    else:inventory.objects['drawer_cabinet']['views']['wrist']['current_visible_roi_uv']=None
    result=assess_guard(bundle)
    assert not result['accepted'],result
    assert guard.egress_evidence.spent_m==0
    assert guard._egress_pending is None


@pytest.mark.parametrize('fault',['missing_metric','missing_image','new_part','stale_proof','missing_geometry'])
def test_scene_wrapper_keeps_all_unrelated_coverage_and_full_intersection_diagnostics(fault):
    field,guard,monitor,inventory,obs,pred,delta=active_archive()
    material=inventory.materialize(obs.state,pred['target_id'],allowed_contact='none')
    material['applied_metric_ids']=[r['id'] for r in field.region_geometry]
    material['applied_image_ids']=[r['id'] for r in guard.neighbors]
    scene=SceneCoverageGuardV23()
    positive=scene.assess(obs.state,delta,materialization=material,field=field,monitor=monitor,prediction=pred)
    assert positive['accepted'],positive
    cabinet=next(x for x in positive['diagnostics']['sweep_checks'] if x['id']=='drawer_cabinet')
    assert not cabinet['full_uncertainty_sweep_disjoint']
    if fault=='missing_metric':material['applied_metric_ids'].remove('drawer_cabinet')
    elif fault=='missing_image':material['applied_image_ids'].remove('drawer_cabinet')
    elif fault=='missing_geometry':
        next(r for r in material['objects'] if r['id']=='drawer_cabinet')['world_geometry']=None
    else:
        original=field.egress_contract
        def changed(state,delta):
            proof=original(state,delta)
            if fault=='stale_proof':proof['observation_epoch']-=1
            else:proof['initial_halo_parts']=[{'id':'drawer_cabinet','tool_part':'different_part'}]
            return proof
        field.egress_contract=changed
    result=scene.assess(obs.state,delta,materialization=material,field=field,monitor=monitor,prediction=pred)
    assert not result['accepted'],result
    assert guard.egress_evidence.spent_m==0


def test_egress_and_residual_holds_charge_global_debt_without_charging_old_heading_row():
    from dataclasses import replace
    from gpt_field.observation import decode_image
    bundle=active_archive();field,guard,monitor,inventory,obs,pred,delta=bundle
    guard.experiments=[{'direction':[1.,0.,0.],'support_generation':0,
        'measured_travel_m':.003,'acknowledged_actions':4}]
    guard._last_charged_experiment=0;guard.observation_travel_m=.003
    original=copy.deepcopy(guard.experiments)
    admission=assess_guard(bundle)
    assert admission['accepted'],admission['reason']
    action=np.r_[delta/.05,0.,0.,0.,0.]
    state=np.array(obs.state);state[:3]+=delta*.2
    after=replace(obs,state=tuple(state))
    images={'external':np.asarray(decode_image(after.image)),'wrist':np.asarray(decode_image(after.wrist_image))}
    monitor.observe(images,after.state,action)
    guard.observe(images,after.state,action,monitor=monitor)
    inventory.observe(obs,after)
    guard.observe_egress(obs.state,action,after.state)
    assert guard.experiments==original
    assert guard._last_charged_experiment==0
    assert guard.observation_travel_m==pytest.approx(.0034)
    assert guard.egress_evidence.spent_m==pytest.approx(.0034)
    state2=state.copy();state2[:3]+=delta*.1
    held=replace(after,state=tuple(state2));hold=np.zeros(7)
    monitor.observe(images,held.state,hold)
    guard.observe(images,held.state,hold,monitor=monitor)
    inventory.observe(after,held)
    guard.observe_egress(after.state,hold,held.state)
    assert guard.experiments==original
    assert guard.observation_travel_m==pytest.approx(.0036)
    assert guard.egress_evidence.spent_m==pytest.approx(.0036)
    assert guard.egress_evidence.egress_travel_m==pytest.approx(.0006)
