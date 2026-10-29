"""Target attraction is distinct from obstacle approach; all evidence stays real."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest
from gpt_field.visual_translation_guard_v30 import ProjectedTranslationGuardV30
from gpt_field.visual_translation_guard_v29 import ProjectedTranslationGuardV29
from test_visual_translation_guard_v29 import setup as previous_setup, actual_observer, proof_for_current
from test_visual_translation_guard_v21 import observe


def setup(monkeypatch):
    guard,field,state,_,kwargs=previous_setup(monkeypatch)
    guard.__class__=ProjectedTranslationGuardV30
    for k,v in ProjectedTranslationGuardV30().__dict__.items():
        if k not in guard.__dict__:guard.__dict__[k]=v
    guard._measurement_available=True
    guard.context.update(purpose='task_translation',translation_mode='transit')
    guard.bind_target_approach_context(dict(controller='visual_field',phase='approach',target_id='neighbor'))
    field.goal=state[:3]+[.06,0,0]
    field.recovery_route_binding=None
    field.bind_recovery_route('normal',state)
    delta=np.array([.003,0.,0.])
    kwargs['nominal_regions']=proof_for_current(field,guard,state,delta)
    kwargs['translation_response'].predict=lambda *a:{'supported':True,'responsive':True,'mean_delta_m':[.001,0,0]}
    guard.reference_witness=SimpleNamespace(raw_support_evidence=lambda *a:{'current_bound':True,
        'reliable':True,'causal_supported':True,'contradicted':False,'confirmation_required':False,
        'evidence_kind':'prospective'})
    guard.boundary_closing_ids.add('neighbor')
    return guard,field,state,delta,kwargs


def test_expected_target_closing_uses_real_geometry_and_preserves_history(monkeypatch):
    g,f,s,d,k=setup(monkeypatch)
    p=k['nominal_regions'][0]['transit_reconciliation_contract']['object_proofs']['neighbor']
    assert p['minimum_segment_gap_m']>0 and not p['gap_nondecreasing']
    history=copy.deepcopy((g.worsening,g.boundary_closing_ids,g.observation_travel_m,g.experiments))
    chosen=g.assess(d,**k)
    assert chosen['accepted'],chosen['reason']
    assert chosen['reason']=='projected_warning_empirically_validated_fixed_intent_execution'
    assert g.last_target_approach_execution['eligible']
    assert (g.worsening,g.boundary_closing_ids,g.observation_travel_m,g.experiments)==history
    g.__class__=ProjectedTranslationGuardV29
    assert not g.assess(d,**k)['accepted']


@pytest.mark.parametrize('fault',['contact_band','contact','payload','unknown','zero_margin','other_object',
    'no_role','wrong_target','phase','purpose','goal_away','past_goal','state','epoch','raw_static',
    'raw_retrospective','robot_opposed','image_contradiction','boundary_unavailable','monitor_contradiction'])
def test_target_exception_keeps_earlier_and_later_refusals(monkeypatch,fault):
    g,f,s,d,k=setup(monkeypatch);c=k['nominal_regions'][0]['transit_reconciliation_contract']
    if fault=='contact_band':c['outside_target_contact_band']=False
    elif fault=='contact':c['target_contact']='fingers'
    elif fault=='payload':c['empty_tool']=False
    elif fault=='unknown':c['hard_segment_clear']=False
    elif fault=='zero_margin':c['object_proofs']['neighbor']['minimum_segment_gap_m']=0
    elif fault=='other_object':g.target_id='another';c['target_id']='another'
    elif fault=='no_role':g.target_approach_context=None
    elif fault=='wrong_target':g.target_approach_context['target_id']='another'
    elif fault=='phase':g.target_approach_context['phase']='recover'
    elif fault=='purpose':g.context['purpose']='reference_repair'
    elif fault=='goal_away':c['fixed_goal_xyz_m']=(s[:3]-[.03,0,0]).tolist()
    elif fault=='past_goal':c['fixed_goal_xyz_m']=(s[:3]+[.001,0,0]).tolist()
    elif fault=='state':c['state8'][0]+=.001
    elif fault=='epoch':c['observation_epoch']+=1
    elif fault=='raw_static':g.reference_witness.raw_support_evidence=lambda *a:{'causal_supported':False}
    elif fault=='raw_retrospective':g.reference_witness.raw_support_evidence=lambda *a:{'current_bound':True,'reliable':True,'causal_supported':True,'confirmation_required':False,'evidence_kind':'retrospective'}
    elif fault=='robot_opposed':
        k['translation_response'].predict=lambda *a:{'supported':True,'responsive':False,'mean_delta_m':[-.001,0,0]}
        g.boundary_evidence['neighbor']['classification']='closing'
    elif fault=='image_contradiction':g._neighbor_responses['neighbor']=SimpleNamespace(evidence=lambda *a:{'validated':False,'recent_measured_sample':True,'held_out_count':3,'held_out_zero_sse':1.,'held_out_prediction_sse':2.})
    elif fault=='boundary_unavailable':g.boundary_evidence['neighbor']['comparable']=False
    elif fault=='monitor_contradiction':k['monitor'].status='visual_response_contradiction'
    chosen=g.assess(d,**k)
    assert not chosen['accepted'],(fault,chosen)
    assert g.supported_execution_actions==0 and g._validated_pending is None


def test_inconclusive_target_allows_existing_finite_acquisition_without_validation_credit(monkeypatch):
    g,f,s,d,k=setup(monkeypatch)
    g._neighbor_responses['neighbor']=SimpleNamespace(evidence=lambda *a:{'validated':False})
    first=g.assess(d,**k)
    assert first['accepted'] and first['cap_m']<=.003
    assert 'heading_evidence' in first['reason']
    assert g._validated_pending is None and g.supported_execution_actions==0
    g.observation_travel_m=.020
    assert not g.assess(d,**k)['accepted']
    assert g.boundary_closing_ids=={'neighbor'}


def test_current_robust_closing_requires_validated_prediction(monkeypatch):
    g,f,s,d,k=setup(monkeypatch);g.boundary_evidence['neighbor']['classification']='closing'
    assert g.assess(d,**k)['accepted']
    g._neighbor_responses['neighbor']=SimpleNamespace(evidence=lambda *a:{'validated':False})
    assert not g.assess(d,**k)['accepted']
    assert g.last_target_approach_execution['reason'].endswith('closing_requires_validated_response')


def test_preview_reselection_and_downstream_veto_do_not_commit_or_refund(monkeypatch):
    g,f,s,d,k=setup(monkeypatch);g.observation_travel_m=.006;g._charge_uncertain_motion=True
    original=g.__dict__;assert g.preview_assess(d,**k)['accepted'];assert g.__dict__ is original
    assert g._validated_pending is None and g.last_target_approach_execution is None
    k['translation_response'].command_cap_for_measured_limit=lambda *a:{'command_cap_m':.001}
    capped=g.assess(d,**k);assert capped['accepted'] and capped['cap_m']==.001 and g._validated_pending is None
    small=d/3;k['nominal_regions']=proof_for_current(f,g,s,small)
    assert g.assess(small,**k)['accepted']
    actual_observer(monkeypatch,g,k)
    after=observe(g,k['monitor'],s,[0,0,0],[.0002,0,0],pixel_x=1)
    assert g.supported_execution_actions==0 and g.observation_travel_m==pytest.approx(.0062)


def test_repeated_acknowledged_approach_preserves_old_debt_and_caps_fixed_intent(monkeypatch):
    g,f,s,d,k=setup(monkeypatch);g.observation_travel_m=.020
    # Isolate semantic admission and inherited ACK accounting; actual observer
    # sees real generated RGB/robot transitions. Whole-policy archive test is separate.
    bank=g.reference_witness;actual_observer(monkeypatch,g,k)
    for step in range(20):
        monkeypatch.setattr(k['monitor'],'_external_reliable',lambda:True)
        monkeypatch.setattr(k['monitor'],'cumulative_response_evidence',lambda **kw:{'contradicted':False})
        k['monitor'].status='ready';g.domain_invalid=None;g.worsening={'neighbor'}
        g.boundary_evidence['neighbor']={'comparable':True,'classification':'closing',
            'measurement_epoch':g.neighbor_measurement_epochs['neighbor']}
        g._neighbor_responses['neighbor']=SimpleNamespace(evidence=lambda *a:{'validated':True})
        k['nominal_regions']=proof_for_current(f,g,s,d)
        assert g.assess(d,**k)['accepted']
        monkeypatch.delattr(k['monitor'],'_external_reliable')
        monkeypatch.delattr(k['monitor'],'cumulative_response_evidence')
        g._neighbor_responses={}
        s=observe(g,k['monitor'],s,d,[.001,0,0],pixel_x=step+1)
    assert g.supported_execution_actions==20 and g.supported_execution_travel_m==pytest.approx(.020)
    assert g.observation_travel_m==.020 and g.boundary_closing_ids=={'neighbor'}
    c=k['nominal_regions']=proof_for_current(f,g,s,d)
    c[0]['transit_reconciliation_contract']['remaining_actions']=0
    assert not g.assess(d,**k)['accepted']


def test_role_rebinding_before_acknowledgement_rejects_credit_but_counts_motion(monkeypatch):
    g,f,s,d,k=setup(monkeypatch);g.observation_travel_m=.006;g._charge_uncertain_motion=True
    assert g.assess(d,**k)['accepted']
    assert g._validated_pending['target_approach_role']['phase']=='approach'
    assert not g.bind_target_approach_context({'controller':'direct','phase':'recover','target_id':'neighbor'})['bound']
    actual_observer(monkeypatch,g,k)
    observe(g,k['monitor'],s,d,[.001,0,0],pixel_x=1)
    assert not g.last_validated_intent_observation['matched']
    assert g.supported_execution_actions==0 and g.observation_travel_m==pytest.approx(.007)
    assert g.domain_invalid=='translation_validated_intent_acknowledgement_mismatch'
