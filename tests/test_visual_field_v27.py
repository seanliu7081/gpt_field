"""Current full-margin segment evidence; no model or simulator launches."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from gpt_field.visual_field_v27 import MetricFieldExecutorV27, full_margin_union_segment_proof
from gpt_field.visual_field_v25 import MetricFieldExecutorV25
from test_visual_translation_guard_v21 import plant
from test_visual_field_v4 import Q


def box(low,high,name='part'):
    return {'tool_part':name,'lower_xyz':low,'upper_xyz':high}


def upgraded():
    guard,monitor,field,state=plant();field.__class__=MetricFieldExecutorV27
    for k,v in MetricFieldExecutorV27().__dict__.items():
        if k not in field.__dict__:field.__dict__[k]=v
    assert field.bind_recovery_route('normal',state)['accepted']
    return field,guard,monitor,state


def test_exact_segment_distance_detects_interior_approach_with_far_endpoints():
    proof=full_margin_union_segment_proof([-2.,1.,0.],[2.,1.,0.],[box([-.1,-.1,-.1],[.1,.1,.1])])
    assert proof['start_gap_m']==pytest.approx(proof['end_gap_m'])
    assert proof['minimum_segment_gap_m']==pytest.approx(.9)
    assert not proof['gap_nondecreasing'] and proof['full_uncertainty_sweep_disjoint']
    crossing=full_margin_union_segment_proof([-2.,0.,0.],[2.,0.,0.],[box([-.0001,-.1,-.1],[.0001,.1,.1])])
    assert crossing['minimum_segment_gap_m']==0. and not crossing['full_uncertainty_sweep_disjoint']


def test_union_lower_envelope_does_not_require_irrelevant_distant_parts_to_separate():
    proof=full_margin_union_segment_proof([0.,0.,0.],[1.,0.,0.],[box([-2,-1,-1],[-1,1,1],'closest'),box([10,-1,-1],[11,1,1],'far_approaching')])
    assert proof['gap_nondecreasing']
    assert proof['start_gap_m']==pytest.approx(1.) and proof['end_gap_m']==pytest.approx(2.)
    assert proof['minimum_segment_part']=='closest'


def test_closest_part_switch_and_concave_lower_envelope_cannot_hide_late_approach():
    proof=full_margin_union_segment_proof([0.,0.,0.],[2.,0.,0.],[box([-2,-1,-1],[-1,1,1],'left'),box([3,-1,-1],[4,1,1],'right')])
    assert proof['start_gap_m']==proof['end_gap_m']==1.
    assert not proof['gap_nondecreasing']
    assert any(r['tool_part']=='right' for r in proof['decreasing_parameter_intervals'])


@pytest.mark.parametrize('fault',['nan','bad_shape','inverted_box','empty'])
def test_bad_geometry_never_produces_a_gap_proof(fault):
    start=[0.,0.,0.];boxes=[box([1,-1,-1],[2,1,1])]
    if fault=='nan':start[0]=float('nan')
    elif fault=='bad_shape':start=[0,0]
    elif fault=='inverted_box':boxes=[box([2,-1,-1],[1,1,1])]
    elif fault=='empty':boxes=[]
    with pytest.raises(ValueError):full_margin_union_segment_proof(start,[.1,0,0],boxes)


def test_actual_scene_contract_has_every_object_and_preserves_all_physical_state():
    field,guard,monitor,state=upgraded();before=pickle.dumps((guard.__dict__,field.pending,field.actions,field.measured_travel_m))
    start=state[:3];end=start+[-.003,0,0]
    records=field.nominal_regions_for_segment(start,end);proof=records[0]['transit_reconciliation_contract']
    assert proof['eligible'],proof
    assert proof['full_scene_ids']==proof['scene_ids']==['neighbor']
    assert set(proof['object_proofs'])==set(proof['scene_ids'])
    assert proof['object_proofs']['neighbor']['gap_nondecreasing']
    assert proof['state8']==pytest.approx(state) and proof['end_xyz_m']==pytest.approx(end)
    assert proof['remaining_actions']==120 and proof['remaining_measured_travel_m']>0
    expected=hashlib.sha256(json.dumps(proof['geometry_fingerprint'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert proof['geometry_sha256']==expected
    assert not proof['physical_budget_replenished'] and not proof['clearance_certified']
    assert pickle.dumps((guard.__dict__,field.pending,field.actions,field.measured_travel_m))==before


def test_toward_scene_proof_is_retained_but_cannot_be_called_nondecreasing():
    field,guard,monitor,state=upgraded();start=state[:3]
    proof=field._transit_reconciliation_contract(state,start,start+[.003,0,0])
    assert proof['eligible']  # Geometry/role eligibility is separate from warned-object direction.
    assert not proof['object_proofs']['neighbor']['gap_nondecreasing']
    assert proof['object_proofs']['neighbor']['end_minus_start_gap_m']<0


@pytest.mark.parametrize('fault',['state','guard_state','start','goal','origin','unbound','reverse','egress','payload','candidate','contact','visual_goal','refinement','action_budget','travel_budget','scene','geometry_missing','duplicate','fingerprint'])
def test_ineligible_binding_scene_role_or_debt_never_authorizes_reconciliation(fault):
    field,guard,monitor,state=upgraded();start=state[:3].copy();end=start+[-.003,0,0];base=None
    if fault=='state':state=state.copy();state[0]+=.00001
    elif fault=='guard_state':guard.state=guard.state.copy();guard.state[0]+=.00001
    elif fault=='start':start=start+[.00001,0,0]
    elif fault=='goal':field.goal=field.goal+[.001,0,0]
    elif fault=='origin':field.origin=field.origin+[.001,0,0]
    elif fault=='unbound':field.recovery_route_binding=None
    elif fault=='reverse':field.recovery_route_binding['recovery_route']='reverse_observed'
    elif fault=='egress':field.recovery_route_binding['recovery_route']='initial_uncertainty_egress'
    elif fault in ('payload','candidate'):
        field.contact_geometry.carried_summary=lambda s:{'active':fault=='payload','candidate_ids':['item']if fault=='candidate'else[]}
    elif fault=='contact':field.target_configuration['effective_allowed_contact']='fingers'
    elif fault=='visual_goal':field.goal_revisions=1
    elif fault=='refinement':field.contact_refinement_enabled=True
    elif fault=='action_budget':field.actions=field.intent['max_actions']
    elif fault=='travel_budget':field.measured_travel_m=field.intent['max_travel_m']
    elif fault=='scene':field.scene_assessor=lambda *args:{'accepted':False,'diagnostics':{}}
    elif fault=='geometry_missing':field._transit_materialization['objects'][0]['world_geometry']=None
    elif fault=='duplicate':field._transit_materialization['objects'].append(copy.deepcopy(field._transit_materialization['objects'][0]))
    elif fault=='fingerprint':
        base=field._retreat_geometry_contract(state,start,end);field._transit_materialization['objects'][0]['world_geometry']['uncertainty_m']+=.001
    proof=field._transit_reconciliation_contract(state,start,end,base_contract=base)
    assert not proof['eligible'],(fault,proof)
    assert guard.observation_travel_m==0


def test_changed_scene_current_fingerprint_and_unknown_hard_segment_are_checked():
    field,guard,monitor,state=upgraded();start=state[:3];end=start+[-.003,0,0]
    good=field._transit_reconciliation_contract(state,start,end)
    assert good['eligible']
    # Same measured pose, newly revised remembered model intersects command.
    field._transit_materialization['objects'][0]['world_geometry']['center_xyz_m']=start.tolist()
    bad=field._transit_reconciliation_contract(state,start,end)
    assert not bad['eligible'] and not bad['object_proofs']['neighbor']['full_uncertainty_sweep_disjoint']
    assert bad['geometry_sha256']!=good['geometry_sha256']
    field,guard,monitor,state=upgraded()
    # Real parent checked-segment path includes explicitly unknown continuous geometry.
    field.boxes.append((start+[-.0016,-.001,-.001],start+[-.0014,.001,.001]))
    field.intent['unknown_regions']=[{'id':'unseen','lower_offset_m':[-.0016,-.001,-.001],'upper_offset_m':[-.0014,.001,.001],'uncertainty_m':0.}]
    bad=field._transit_reconciliation_contract(state,start,end)
    assert not bad['eligible'] and not bad['hard_segment_clear']


def test_existing_normal_selection_pending_ack_and_action_are_unchanged():
    new,gn,mn,sn=upgraded();old,go,mo,so=upgraded();old.__class__=MetricFieldExecutorV25
    chosen=new.select(sn,Q);baseline=old.select(so,Q)
    assert chosen['accepted'] and chosen['action']==baseline['action']
    wrong=np.array(chosen['action']);wrong[0]+=.01
    with pytest.raises(ValueError):new.observe(sn,wrong,sn)
    after=sn.copy();after[:3]+=np.array(chosen['action'][:3])*.05*.2
    new.observe(sn,chosen['action'],after)
    assert new.pending is None and new.actions==1


def test_saved_approach_terminal_cabinet_proof_is_fresh_and_separating():
    root=Path(__file__).resolve().parents[1];directory=root/'runs/visual_field_v26_design_001'
    path=directory/'approach_diagnostic_001/trace.jsonl'
    if not path.exists():pytest.skip('Optional completed V26 diagnostic unavailable')
    spec=importlib.util.spec_from_file_location('v27_archived_approach',directory/'approach_diagnostic.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    from gpt_field.observation import PolicyObservation
    from gpt_field.visual_field_policy_v26 import VisualFieldPolicyV26
    scene,initial,decisions,_,_=helper.replay_data();client=helper.ApproachClient(decisions);policy=VisualFieldPolicyV26(client)
    policy.local.__class__=MetricFieldExecutorV27;policy.local.transit_reconciliation_geometry=None
    rows=[json.loads(x)for x in path.read_text().splitlines()]
    actions={r['step']:r for r in rows if r['event']=='action'};obs=PolicyObservation.from_dict(initial)
    for r in rows:
        if r['event']!='executed':continue
        decision=policy.act(obs);assert decision.action==pytest.approx(actions[r['step']]['action'])
        after=PolicyObservation.from_dict(r['observation']);policy.observe_transition(obs,decision,after);obs=after
    selected=policy.act(obs)
    assert not selected.diagnostics['selection']['accepted']  # Proof alone never changes legacy guard admission.
    proof=policy.local.transit_reconciliation_geometry
    assert proof['eligible'] and proof['route_family']=='normal'
    assert len(proof['object_proofs'])==len(proof['full_scene_ids'])==8
    assert proof['state8']==pytest.approx(obs.state)
    cabinet=proof['object_proofs']['cabinet']
    assert cabinet['start_gap_m']==pytest.approx(.04581595575083721)
    assert cabinet['end_minus_start_gap_m']==pytest.approx(.001767436387670629)
    assert cabinet['gap_nondecreasing'] and cabinet['full_uncertainty_sweep_disjoint']
    assert not proof['object_proofs']['black_bowl']['gap_nondecreasing']
    assert client.calls==2 and policy.translation_guard.observation_travel_m==0.
    policy.finish()


def test_current_valid_carried_payload_and_stale_epoch_remain_ineligible():
    field,guard,monitor,state=upgraded();start=state[:3];end=start+[-.003,0,0]
    field.contact_geometry.carried_summary=lambda s:{'active':True,'candidate_ids':[],
        'nominal_world':{'lower_xyz':(state[:3]-.005).tolist(),'upper_xyz':(state[:3]+.005).tolist()},
        'nominal_hand_frame':{'lower_xyz':[-.005]*3,'upper_xyz':[.005]*3},'uncertainty_m':.001}
    proof=field._transit_reconciliation_contract(state,start,end)
    assert not proof['eligible'] and not proof['empty_tool']
    assert proof['object_proofs']['neighbor']['configuration_part_count']==7
    field,guard,monitor,state=upgraded()
    old=field._retreat_geometry_contract(state,start,end);guard.sweep_observation_epoch+=1
    proof=field._transit_reconciliation_contract(state,start,end,base_contract=old)
    assert not proof['eligible']


def test_proof_exception_does_not_change_pending_or_any_physical_accounting():
    field,guard,monitor,state=upgraded();before=pickle.dumps((field.pending,field.actions,field.measured_travel_m,guard.__dict__))
    def broken(*args):raise RuntimeError('offline callback fault')
    field.scene_assessor=broken
    with pytest.raises(RuntimeError,match='offline callback fault'):
        field._transit_reconciliation_contract(state,state[:3],state[:3]+[-.003,0,0])
    assert pickle.dumps((field.pending,field.actions,field.measured_travel_m,guard.__dict__))==before


@pytest.mark.parametrize('caller_cap',[None,.002])
def test_warning_transit_constructs_only_capped_candidate_and_acknowledges_exact_action(monkeypatch,caller_cap):
    # Isolate actual executor/proof/reselection from perception qualification:
    # the shared fixture supplies a qualified response and the inherited
    # projected-warning refusal. No synthetic witness is claimed as real RGB.
    from test_visual_translation_guard_v27 import setup
    guard,field,measured,_,kwargs=setup(monkeypatch)
    intent=copy.deepcopy(field.intent);intent['goal_offset_m']=[-.06,0.,0.]
    field.configure(measured,intent,scene_assessment=guard.context,
        inventory_materialization=copy.deepcopy(field._transit_materialization))
    assert field.bind_recovery_route('normal',measured)['accepted']
    field.bind_translation_guard(guard,kwargs['monitor'],kwargs['translation_response'])
    calls=[];original=guard.assess
    def capture(delta,**options):
        proof=copy.deepcopy(options['nominal_regions'][0]['transit_reconciliation_contract'])
        calls.append((np.asarray(delta).copy(),proof))
        return original(delta,**options)
    monkeypatch.setattr(guard,'assess',capture)
    selected=field.select(measured,Q,command_cap_m=caller_cap)
    assert selected['accepted'],selected['status']
    assert len(calls)==1  # No overlong, never-executed proposal is checked first.
    expected=.003 if caller_cap is None else caller_cap
    delta=np.asarray(selected['action'][:3])*.05
    assert np.linalg.norm(delta)==pytest.approx(expected)
    assert calls[0][0]==pytest.approx(delta)
    assert calls[0][1]['end_xyz_m']==pytest.approx(measured[:3]+delta)
    assert field.actions==0 and guard.experiments==[] and guard.observation_travel_m==0.
    wrong=np.asarray(selected['action']).copy();wrong[0]=-.01/.05
    with pytest.raises(ValueError):field.observe(measured,wrong,measured)
    after=measured.copy();after[:3]+=delta*.2
    field.observe(measured,selected['action'],after)
    assert field.pending is None and field.actions==1
    assert field.measured_travel_m==pytest.approx(expected*.2)


def test_warning_candidate_cap_does_not_bypass_changed_scene_intersection(monkeypatch):
    from test_visual_translation_guard_v27 import setup
    guard,field,measured,_,kwargs=setup(monkeypatch)
    intent=copy.deepcopy(field.intent);intent['goal_offset_m']=[-.06,0.,0.]
    field.configure(measured,intent,scene_assessment=guard.context,
        inventory_materialization=copy.deepcopy(field._transit_materialization))
    assert field.bind_recovery_route('normal',measured)['accepted']
    field.bind_translation_guard(guard,kwargs['monitor'],kwargs['translation_response'])
    field._transit_materialization['objects'][0]['world_geometry']['center_xyz_m']=measured[:3].tolist()
    selected=field.select(measured,Q)
    assert not selected['accepted']
    assert not field.transit_reconciliation_geometry['eligible']
    assert field.pending is None and field.actions==0
    assert guard.observation_travel_m==0. and guard.experiments==[]


def test_recorded_twenty_action_prefix_continues_on_exact_shorter_admissible_segment():
    root=Path(__file__).resolve().parents[1];directory=root/'runs/visual_field_v27_design_001'
    path=directory/'approach_diagnostic_001/trace.jsonl'
    if not path.exists():pytest.skip('Optional completed V27 controller diagnostic unavailable')
    spec=importlib.util.spec_from_file_location('v27_recorded_twenty_step_prefix',directory/'approach_diagnostic.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    from gpt_field.observation import PolicyObservation
    from gpt_field.visual_field_policy_v27 import VisualFieldPolicyV27
    _,initial,decisions,_,_=helper.replay_data();client=helper.ApproachClient(decisions)
    policy=VisualFieldPolicyV27(client);obs=PolicyObservation.from_dict(initial)
    try:
        rows=[json.loads(x)for x in path.read_text().splitlines()]
        actions={r['step']:r for r in rows if r['event']=='action'}
        executed={r['step']:r['observation']for r in rows if r['event']=='executed'}
        for step in range(20):
            decision=policy.act(obs)
            assert np.allclose(decision.action,actions[step]['action'],atol=1e-10,rtol=0.),step
            after=PolicyObservation.from_dict(executed[step])
            policy.observe_transition(obs,decision,after);obs=after
        debt=policy.translation_guard.observation_travel_m
        experiments=copy.deepcopy(policy.translation_guard.experiments)
        selected=policy.act(obs)
        assert selected.diagnostics['selection']['accepted'],selected.diagnostics['selection']['status']
        delta=np.asarray(selected.action[:3])*.05
        assert np.linalg.norm(delta)==pytest.approx(.003)
        proof=policy.local.transit_reconciliation_geometry
        assert proof['eligible'] and proof['end_xyz_m']==pytest.approx(np.asarray(obs.state[:3])+delta)
        assert proof['object_proofs']['cabinet']['gap_nondecreasing']
        assert proof['object_proofs']['cabinet']['minimum_segment_gap_m']==pytest.approx(.046878821282318094)
        longproof=policy.local._transit_reconciliation_contract(np.asarray(obs.state),
            np.asarray(obs.state[:3]),np.asarray(obs.state[:3])+delta/.003*.005)
        assert not longproof['object_proofs']['cabinet']['gap_nondecreasing']
        assert policy.translation_guard.last_projected_warning_transit['accepted']
        assert policy.translation_guard.observation_travel_m==debt
        assert policy.translation_guard.experiments==experiments and client.calls==2
        # No fabricated observation for this newly different, unexecuted step.
        assert policy.local.pending is not None
    finally:policy.finish()
