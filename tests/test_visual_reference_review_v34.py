"""Paid-history repair may be needed even while feature tracks remain reliable."""
import copy
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpt_field.visual_reference_candidates_v30 import ReferenceReviewKind as Kind
from gpt_field.visual_reference_review_v30 import reference_review_event as legacy_event
from gpt_field.visual_reference_review_v34 import reference_review_event, TypedReferenceReviewPolicyMixinV34
from test_visual_reference_candidates_v30 import plant
from test_visual_reference_candidates_v24 import advance


def policy(bank,monitor,*,reason='new_future_handoff_v999',prediction=None):
    return SimpleNamespace(reference_candidates=bank,monitor=monitor,prediction=prediction,
        _typed_reference_control_failure=False,_requested_reference_reviews=set(),events=[],
        refresh_reason=reason,step_index=5,physical_debt={'heading':.005,'shared':.011})


def test_reliable_tracks_at_repair_handoff_extract_existing_motion_without_failure():
    bank,monitor,context,obs=plant();obs=advance(bank,monitor,obs)
    assert monitor.tracks['moving'].reliable and monitor.tracks['target'].reliable
    monitor.summary=lambda:{'reference_health':{'repair_required':True}}
    assert legacy_event(monitor,handoff_present=True) is None
    event=reference_review_event(monitor,handoff_present=True,detail_reason='unseen_budget_reason')
    assert event.kind is Kind.CONTROLLED_RESPONSE_UNRELIABLE
    subject=policy(bank,monitor)
    before=copy.deepcopy((bank.path_m,bank.sequence,monitor._active_reference,subject.physical_debt))
    TypedReferenceReviewPolicyMixinV34._review_paid_reference(subject,obs,subject.refresh_reason)
    report=bank.summary()
    assert report['extraction_status']=='available' and report['candidate_extraction_count']==1
    assert report['candidates'] and len(subject.events)==1
    assert (bank.path_m,bank.sequence,monitor._active_reference,subject.physical_debt)==before
    assert not subject._typed_reference_control_failure
    assert subject.reference_review_dispatch['failed_control_handoff'] is False
    assert not report['control_debt_modified'] and not report['automatic_reference_promotion']
    assert monitor.tracks['moving'].reliable
    for _ in range(3):
        TypedReferenceReviewPolicyMixinV34._review_paid_reference(subject,obs,'another_unused_wording')
    assert len(subject.events)==1 and bank.summary()['candidate_extraction_count']==1


@pytest.mark.parametrize('handoff,repair,failed,expected',[
    (False,True,False,None),(True,False,False,None),(False,False,True,Kind.CONTROL_RISK_HANDOFF),
    (True,False,True,Kind.CONTROL_RISK_HANDOFF),(True,True,True,Kind.CONTROLLED_RESPONSE_UNRELIABLE),
    (True,True,False,Kind.CONTROLLED_RESPONSE_UNRELIABLE)])
def test_handoff_and_health_are_typed_conditions_not_stop_text(handoff,repair,failed,expected):
    bank,monitor,context,obs=plant();monitor.summary=lambda:{'reference_health':{'repair_required':repair}}
    event=reference_review_event(monitor,handoff_present=handoff,failed_control_handoff=failed,
                                detail_reason='arbitrary future text')
    assert (None if event is None else event.kind)==expected


@pytest.mark.parametrize('lost,expected',[('moving',Kind.CONTROLLED_REFERENCE_LOST),('target',Kind.TARGET_REFERENCE_LOST)])
def test_existing_loss_classification_and_anchor_ineligibility_are_preserved(lost,expected):
    bank,monitor,context,obs=plant();obs=advance(bank,monitor,obs)
    monitor.tracks[lost].reliable=False
    event=reference_review_event(monitor,handoff_present=False)
    assert event.kind is expected and legacy_event(monitor).kind is expected
    report=bank.review(obs,monitor,event)
    if lost=='target':
        assert report['extraction_status']=='ineligible'
        assert report['eligibility_reason']=='stationary_anchor_review_required'
    else:assert report['extraction_status']=='available'


def test_explicit_null_external_is_not_reported_as_natural_tracker_loss_or_restored():
    bank,monitor,context,obs=plant();obs=advance(bank,monitor,obs)
    monitor.refs['external']=None;monitor.tracks={'moving':None,'target':None}
    subject=policy(bank,monitor,prediction={'visual_references':{'external':None,'wrist':None}})
    before=copy.deepcopy((bank.path_m,bank.sequence,bank.summary()['candidate_extraction_count']))
    TypedReferenceReviewPolicyMixinV34._review_paid_reference(subject,obs,'external_reference_unavailable')
    assert subject.reference_review_dispatch['dispatch_status']=='model_external_reference_omitted'
    assert subject.reference_review_dispatch['event'] is None
    assert not subject.reference_review_dispatch['automatic_reference_restore']
    assert monitor.refs['external'] is None and subject.events==[]
    assert (bank.path_m,bank.sequence,bank.summary()['candidate_extraction_count'])==before


def test_stale_history_cannot_be_promoted_by_new_dispatch():
    bank,monitor,context,obs=plant();stale=copy.deepcopy(obs);obs=advance(bank,monitor,obs)
    monitor.summary=lambda:{'reference_health':{'repair_required':True}}
    subject=policy(bank,monitor)
    TypedReferenceReviewPolicyMixinV34._review_paid_reference(subject,stale,subject.refresh_reason)
    assert bank.summary()['extraction_status']=='ineligible'
    assert bank.summary()['eligibility_reason']=='observation_binding_mismatch'
    assert not subject.events and bank.summary()['candidate_extraction_count']==0


@pytest.mark.parametrize('step',[119,134])
def test_actual_archived_reliable_track_budget_handoff_runs_paid_extraction(step):
    path=Path(__file__).resolve().parents[1]/f'runs/visual_field_v34_design_001/reference_step{step}_checkpoint.pkl'
    if not path.exists():pytest.skip('Optional strictly reconstructed V33 paid-history snapshot unavailable')
    bank,monitor,obs,reason=pickle.loads(path.read_bytes())
    assert monitor.tracks['moving'].reliable and monitor.tracks['target'].reliable
    assert monitor.summary()['reference_health']['repair_required']
    assert bank.summary()['extraction_status']=='not_requested'
    assert legacy_event(monitor,handoff_present=True,detail_reason=reason) is None
    subject=policy(bank,monitor,reason=reason)
    before=copy.deepcopy((bank.path_m,bank.sequence,monitor.summary(),subject.physical_debt))
    oldcount=bank.summary()['candidate_extraction_count']
    TypedReferenceReviewPolicyMixinV34._review_paid_reference(subject,obs,reason)
    report=bank.summary()
    assert report['candidate_extraction_count']==oldcount+1
    assert report['extraction_status'] in ('available','extracted_none')
    assert report['eligibility_reason'] is None
    assert (bank.path_m,bank.sequence,monitor.summary(),subject.physical_debt)==before
    assert not report['control_debt_modified'] and not report['automatic_reference_promotion']


def test_actual_step269_model_omission_does_not_become_a_natural_loss_event():
    import json
    path=Path(__file__).resolve().parents[1]/'runs/visual_field_v33_zero_shot_001/trials/bowl_original_zero_hybrid_v33/trace.jsonl'
    if not path.exists():pytest.skip('Optional V33 archive unavailable')
    prediction=None
    for line in path.open():
        row=json.loads(line)
        if row['event']=='action' and row['step']==269:
            prediction=row['diagnostics']['prediction'];break
    assert prediction['raw_wire_prediction']['visual_references']=={'external':None,'wrist':None}
    monitor=SimpleNamespace(refs=copy.deepcopy(prediction['visual_references']),tracks={},
        summary=lambda:{'reference_health':{'repair_required':True}})
    class Bank:
        def review(self,*args):raise AssertionError('Explicit omitted reference is not natural loss')
    subject=policy(Bank(),monitor,prediction=prediction)
    TypedReferenceReviewPolicyMixinV34._review_paid_reference(subject,None,'external_reference_unavailable')
    assert subject.reference_review_dispatch['dispatch_status']=='model_external_reference_omitted'
    assert subject.reference_review_dispatch['event'] is None and not subject.events
