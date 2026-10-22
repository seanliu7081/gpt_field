"""Fixed authorization accounting; reference results are an explicit boundary.

Real raw-RGB qualification is additionally exercised in executor/archive tests.
"""
import copy
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from test_visual_egress_evidence_v23 import Plant
from gpt_field.visual_route_exposure_v24 import RouteExposureV24


def plant(*,qualified=False,prior_debt=0.):
    p=Plant(prior_debt=prior_debt);p.ledger=RouteExposureV24()
    status={'current_bound':True,'reliable':True,'causal_supported':qualified,'contradicted':False,
        'confirmation_required':not qualified,'selected_candidate_id':None,'evidence_kind':'prospective'}
    p.guard.reference_witness=SimpleNamespace(raw_support_evidence=lambda state,monitor:copy.deepcopy(status))
    p.guard.reference_selection=None
    original=p.proof
    p.proof=lambda command:dict(original(command),nominal_core_separation_nondecreasing=True)
    return p,status


def test_fixed_route_uses_no_positive_gap_grants_and_cannot_replenish_shared_total():
    p,status=plant(qualified=True,prior_debt=.007)
    while p.travel<.031:
        a=p.assess();assert a['accepted'],a
        p.move(a,actual=.0005)
    assert p.ledger.grants==0 and p.ledger.spent_m==pytest.approx(.038)
    assert p.ledger.egress_travel_m==pytest.approx(.031)
    assert p.ledger.summary()['original_full_goal_bound']
    for _ in range(10):
        a=p.assess()
        if not a['accepted']:break
        p.move(a,actual=.0005)
    assert not a['accepted'] and p.ledger.spent_m<=.04+1e-10


def test_static_reference_cannot_use_fixed_route_and_holds_cannot_commit_confirmation():
    p,status=plant()
    while p.travel<.0035:p.move(p.assess())
    before=pickle.dumps(p.ledger.__dict__)
    assert not p.assess()['accepted']
    assert pickle.dumps(p.ledger.__dict__)==before
    status.update(selected_candidate_id='explicit-candidate',confirmation_required=True)
    p.guard.reference_selection={'candidate_id':'explicit-candidate','track_identity':id(p.tool)}
    a=p.assess();assert a['accepted']
    p.move(a,actual=0.,hold=True)
    assert p.ledger.confirmation_candidate_id is None
    p.move(p.assess())
    assert p.ledger.confirmation_candidate_id=='explicit-candidate'
    start=p.ledger.confirmation_start_travel_m
    while p.ledger.egress_travel_m-start<.0035-1e-10:p.move(p.assess())
    assert not p.assess()['accepted']
    status['selected_candidate_id']='renamed-candidate'
    p.guard.reference_selection={'candidate_id':'renamed-candidate','track_identity':id(p.tool)}
    assert p.assess()['reason']=='route_exposure_confirmation_cannot_be_renewed'
    assert p.ledger.confirmation_start_travel_m==start


@pytest.mark.parametrize('revision',['goal','origin','geometry','halo'])
def test_actual_acknowledged_full_context_not_preview_binds_fixed_route(revision):
    p,status=plant(qualified=True)
    before=pickle.dumps(p.ledger.__dict__)
    for _ in range(3):assert p.assess()['accepted']
    assert pickle.dumps(p.ledger.__dict__)==before
    p.move(p.assess());proof=p.proof(np.array([.002,0,0]))
    if revision=='goal':proof['complete_route_goal_xyz'][0]+=.002
    elif revision=='origin':proof['geometry_origin_state8'][0]+=.001
    elif revision=='geometry':proof['geometry_sha256']='replacement'
    else:proof['initial_halo_parts'][0]['tool_part']='different-part'
    result=p.ledger.assess(proof,p.state,[.002,0,0],guard=p.guard,monitor=p.monitor,inventory=p.inventory)
    assert not result['accepted'] and result['reason']=='route_exposure_original_geometry_or_goal_changed'


@pytest.mark.parametrize('fault',['stale','lost','contradicted','retrospective','core_separation'])
def test_causal_witness_and_geometry_negatives_cannot_become_full_route_authority(fault):
    p,status=plant(qualified=True)
    # Spend the bootstrap so retrospective-only data cannot masquerade as a
    # prospective qualified reference.
    for _ in range(8):p.move(p.assess())
    if fault=='stale':status['current_bound']=False
    if fault=='lost':status['reliable']=False
    if fault=='contradicted':status['contradicted']=True
    if fault=='retrospective':status['evidence_kind']='retrospective_selection_only'
    proof=p.proof(np.array([.002,0,0]))
    if fault=='core_separation':proof['nominal_core_separation_nondecreasing']=False
    result=p.ledger.assess(proof,p.state,[.002,0,0],guard=p.guard,monitor=p.monitor,inventory=p.inventory)
    assert not result['accepted']
