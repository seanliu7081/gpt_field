"""Actual RGB + robot-state paths through V24; no model or simulator calls."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from gpt_field.observation import PolicyObservation,decode_image
from gpt_field.visual_field_v24 import MetricFieldExecutorV24
from gpt_field.visual_translation_guard_v24 import ProjectedTranslationGuardV24
from gpt_field.visual_reference_candidates_v24 import CausalReferenceCandidatesV24
from gpt_field.visual_scene_guard_v23 import SceneCoverageGuardV23
from test_visual_field_v23 import synthetic_egress_field,archive_field


def upgraded_plant():
    old,_,monitor,inventory,response,plant,state,observation,obs=synthetic_egress_field()
    images={'external':np.asarray(decode_image(obs.image)),'wrist':np.asarray(decode_image(obs.wrist_image))}
    context=copy.deepcopy(old.translation_guard.context)
    guard=ProjectedTranslationGuardV24();guard.configure(images,context,state,target_id='target',monitor=monitor)
    bank=CausalReferenceCandidatesV24();bank.configure(obs,monitor,context)
    guard.bind_egress_runtime(inventory,monitor);guard.bind_reference_witness(bank)
    field=MetricFieldExecutorV24();field.bind_translation_guard(guard,monitor,response);field.contact_geometry=old.contact_geometry
    field.configure(state,old.intent,scene_assessment=context,inventory_materialization=old._transit_materialization)
    field.egress_context=copy.deepcopy(old.egress_context);scene=SceneCoverageGuardV23()
    def assess(current,delta):
        material=inventory.materialize(current,'target',allowed_contact='none')
        material['applied_metric_ids']=[r['id']for r in field.region_geometry]
        material['applied_image_ids']=[r['id']for r in guard.neighbors]
        return scene.assess(current,delta,materialization=material,field=field,monitor=monitor,
            prediction={'controller':'visual_field','phase':'recover','target_id':'target','target_contact':'none',
                'motion_context':context,'field_intent':field.intent})
    field.scene_assessor=assess
    return field,guard,monitor,inventory,response,plant,state,observation,obs,bank


def test_actual_causal_rgb_fixed_route_exposure_beyond30mm_without_favorable_gap_grants():
    field,guard,monitor,inventory,response,plant,state,observation,obs,bank=upgraded_plant()
    goal=field.goal.copy();modes=[]
    for step in range(120):
        selected=field.select(state,obs.joint_position)
        assert selected['accepted'],(step,selected['status'],bank.raw_support_evidence(state,monitor))
        action=np.array(selected['action']);delta=action[:3]*.05
        modes.append(selected['diagnostics']['translation_admission']['reason'])
        before=state.copy();state[:3]+=delta*.2;plant.travel+=float(np.linalg.norm(delta*.2))
        after=observation();images={'external':np.asarray(decode_image(after.image)),'wrist':np.asarray(decode_image(after.wrist_image))}
        field.observe(before,action,state);response.observe(before,action,state,before_joints=obs.joint_position,after_joints=after.joint_position)
        monitor.observe(images,state,action);inventory.observe(obs,after);guard.observe(images,state,action,monitor=monitor)
        bank.observe(obs,after,monitor);guard.observe_egress(before,action,state);obs=after
        assert field.goal==pytest.approx(goal) and field.pending is None
        if plant.travel>.031:break
    assert field.actions>50 and plant.travel>.031
    assert 'conditional_model_route' in modes and modes[0]=='initial_reference_probe'
    assert guard.egress_evidence.grants==0
    assert guard.egress_evidence.spent_m==pytest.approx(plant.travel)
    assert bank.raw_support_evidence(state,monitor)['causal_supported']
    assert not guard.egress_evidence.summary()['positive_image_gap_required']


def test_original_fixed_core_support_planes_prevent_tangential_halo_exit_toward_nominal_core():
    old,guard,monitor,inventory,obs,pred,assessor=archive_field()
    field=MetricFieldExecutorV24();field.bind_translation_guard(guard,monitor);field.contact_geometry=old.contact_geometry
    field.configure(obs.state,old.intent,scene_assessment=guard.context,inventory_materialization=old._transit_materialization)
    first=field.egress_candidates(obs.state)[0]
    assert field.core_separation_contract(obs.state[:3],np.array(obs.state[:3])+first['goal_offset_m'])['accepted']
    proof=field.core_separation_contract(obs.state[:3],np.array(obs.state[:3])+[0,-.001,.002])
    assert not proof['accepted'] and proof['reason']=='egress_nominal_core_separation_decreases'


def test_nominal_separation_floor_commits_only_first_exact_egress_ack_and_survives_refresh():
    field,guard,monitor,inventory,response,plant,state,observation,obs,bank=upgraded_plant()
    original_geometry=field.egress_geometry
    # A pre-egress residual hold does not rebase the original planes, but its
    # current actual position is the provisional starting separation.
    direction=np.array(field.egress_candidates(state)[0]['goal_offset_m'])
    direction/=np.linalg.norm(direction)
    drifted=state.copy();drifted[:3]-=direction*.0000162926
    proof=field.core_separation_contract(drifted[:3],drifted[:3]+direction*.002)
    assert proof['accepted'] and field._committed_core_separation_state is None
    assert all(not p['separation_baseline_committed'] for p in proof['parts'])
    assert field.egress_geometry is original_geometry

    selected=field.select(state,obs.joint_position)
    assert selected['accepted']
    action=np.asarray(selected['action']);after=state.copy();after[:3]+=action[:3]*.05*.2
    wrong=action.copy();wrong[0]+=.01
    with pytest.raises(ValueError):field.observe(state,wrong,after)
    assert field._committed_core_separation_state is None
    field.observe(state,action,after)
    assert field._committed_core_separation_state==pytest.approx(state)
    committed=field._committed_core_separation_state.copy()
    assert not field.core_separation_contract(drifted[:3],drifted[:3]+direction*.002)['accepted']
    field.configure(after,field.intent,scene_assessment=guard.context,
        inventory_materialization=field._transit_materialization)
    assert field._committed_core_separation_state==pytest.approx(committed)
    assert field.egress_geometry is original_geometry
    assert not field.core_separation_contract(drifted[:3],drifted[:3]+direction*.002)['accepted']


def test_actual003_static_raw_reference_never_qualifies_despite_legacy_tracker_flow():
    from gpt_field.visual_monitor_v21 import RoleAwareVisualMonitorV21
    p=Path(__file__).resolve().parents[1]/'runs/visual_field_v23_design_001/controller_diagnostic_003'
    if not p.exists():pytest.skip('Optional archived deterministic controller run unavailable')
    rows=list(map(json.loads,(p/'trace.jsonl').read_text().splitlines()))
    pred=next(r['diagnostics']['prediction']for r in rows if r['event']=='action' and r['step']==1)
    observations={r['step']:PolicyObservation.from_dict(r['observation'])for r in rows if r['event']=='executed'}
    before=observations[0];images={'external':np.asarray(decode_image(before.image)),'wrist':np.asarray(decode_image(before.wrist_image))}
    monitor=RoleAwareVisualMonitorV21();monitor.configure(images,pred['visual_references'],before.state,intent_id=1)
    bank=CausalReferenceCandidatesV24();bank.configure(before,monitor,pred['motion_context'])
    for step in range(1,9):
        after=observations[step];action=next(r['action']for r in rows if r['event']=='action' and r['step']==step)
        images={'external':np.asarray(decode_image(after.image)),'wrist':np.asarray(decode_image(after.wrist_image))}
        monitor.observe(images,after.state,action);bank.observe(before,after,monitor);before=after
    evidence=bank.raw_support_evidence(before.state,monitor)
    assert evidence['current_bound'] and evidence['reliable']
    assert not evidence['causal_supported'] and evidence['raw_support_changed_fraction']==0
    assert evidence['raw_support_max_rgb_change']==0
