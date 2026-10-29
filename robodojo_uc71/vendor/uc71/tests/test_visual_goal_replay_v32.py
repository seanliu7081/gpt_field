"""Training-only import with current bindings and genuinely subsequent holdouts."""
import copy
import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.observation import decode_image,PolicyObservation
from gpt_field.visual_field import _gray
from gpt_field.visual_goal_field_v31 import VisualGoalFieldV31
from gpt_field.visual_goal_replay_v32 import warm_start_visual_goal
from gpt_field.visual_reference_replay_v18 import evidence_digest
from test_visual_reference_replay_v18 import fixture as reference_fixture
from test_visual_field_v6_policy import observation, prediction


def configured_field(obs, pred, candidate, report):
    rgb=np.asarray(decode_image(obs.image));gray=_gray(rgb)
    ext=copy.deepcopy(pred['visual_references']['external']);ext['reference_role']='tool_anatomy'
    seed=ext['seed'];scale=np.array([223.,223.])
    def track(point):
        return SimpleNamespace(position=np.array(point)*scale,scale=scale,reliable=True,
            last_residual_px=0.,last_inliers=8,previous_image=gray)
    moving,anchor=track(seed['moving_point_uv']),track(seed['target_anchor_uv'])
    monitor=SimpleNamespace(refs={'external':ext},tracks={'moving':moving,'target':anchor},
        state=np.array(obs.state),_external_reliable=lambda:True,_pending_prediction=None,
        _v18_replay_applied=(candidate['candidate_id'],candidate['evidence_sha256'],report['last_sequence']))
    field=VisualGoalFieldV31();pred=copy.deepcopy(pred);pred['visual_references']['external']=ext
    field.configure(obs.state,monitor,pred,joint_position=obs.joint_position)
    return field,monitor,pred


def fixture():
    _,_,candidate,old_obs=reference_fixture()
    # Synthetic gain500px/m: same6pixels take12mm, giving four3mm aggregates.
    candidate=copy.deepcopy(candidate);rows=candidate['evidence']['chronological_samples']
    for row in rows: row['state8'][0] *= 2
    obs=replace(old_obs,state=tuple(rows[-1]['state8']))
    rgb=np.asarray(decode_image(obs.image))
    candidate['observation_binding']=hashlib.sha256(rgb.tobytes()+np.asarray(obs.state,'<f8').tobytes()).hexdigest()
    candidate['evidence_sha256']=evidence_digest(candidate['evidence'])
    report={'accepted':True,'candidate_id':candidate['candidate_id'],'evidence_sha256':candidate['evidence_sha256'],
        'kind':'retrospective_candidate_conditioned_training_only','last_sequence':6,
        'historical_samples_are_held_out':False,'fresh_live_validation_required':True,'physical_debt_modified':False}
    pred=prediction();pred['visual_references']['external']['seed']['moving_point_uv']=candidate['current_point_uv']
    field,monitor,pred=configured_field(obs,pred,candidate,report)
    return field,monitor,candidate,obs,report,pred


def frozen(field,monitor):
    # Lambdas are test-only; compare every mutable observer/field value otherwise.
    return pickle.dumps((field.__dict__,{k:v for k,v in monitor.__dict__.items() if k!='_external_reliable'}))


def test_import_only_training_preserves_current_objective_and_no_validation():
    f,m,c,o,r,p=fixture();before=copy.deepcopy(f.current);state=m.state.copy();count=f.observed_transitions
    result=warm_start_visual_goal(f,m,c,o,r)
    assert result['accepted'] and result['training_aggregates']==3
    assert not result['historical_samples_are_held_out'] and not result['physical_debt_modified']
    assert not f.held_out and f.observed_transitions==count
    assert np.array_equal(m.state,state)
    assert all(np.array_equal(f.current[k],v) if isinstance(v,np.ndarray) else f.current[k]==v for k,v in before.items())
    assert f.model.matrix is not None and len(f.sample_positions)==3
    proposed=f.propose(o.state,joint_position=o.joint_position)
    assert not proposed['accepted'] and not proposed['candidates']


def test_only_actual_later_aggregates_produce_validation_holds_do_not():
    f,m,c,o,r,p=fixture();assert warm_start_visual_goal(f,m,c,o,r)['accepted']
    for _ in range(4):f.observe(o.state,[0]*7,o.state,monitor=m,prediction=p,joint_position=o.joint_position)
    assert not f.held_out
    state=np.array(o.state)
    for n in range(1,9):
        after=state.copy();after[0]+=.001  # New acknowledged same-heading movement towards the image goal.
        m.state=after.copy();m.tracks['moving'].position[0]+=.5
        f.observe(state,[.02,0,0,0,0,0,0],after,monitor=m,prediction=p,joint_position=o.joint_position)
        state=after
    assert len(f.held_out)==2
    assert all(row['training_samples_before_prediction']>=3 for row in f.held_out)
    assert f._heading_evidence(np.array([1.,0,0]))['validated']
    assert f.propose(state,joint_position=o.joint_position)['accepted']
    # Changing direction does not borrow the two actual positive-heading tests.
    assert not f._heading_evidence(np.array([-1.,0,0]))['validated']


@pytest.mark.parametrize('change,reason',[
    ('proxy','requires_anatomical'),('unaccepted','not_accepted'),('digest','digest_mismatch'),
    ('binding','observation_binding'),('parent_key','monitor_import_binding'),
    ('state','current_state'),('joints','current_joints'),('track','current_track_binding'),
    ('pixels','track_observation'),('pending','pending_observation'),('already_trained','existing_live'),
])
def test_incompatible_current_evidence_is_atomic(change,reason):
    f,m,c,o,r,p=fixture()
    if change=='proxy':m.refs['external']['reference_role']='rigid_transit_proxy'
    elif change=='unaccepted':r['accepted']=False
    elif change=='digest':c['evidence']['chronological_samples'][0]['state8'][0]+=.00001
    elif change=='binding':c['observation_binding']='stale'
    elif change=='parent_key':m._v18_replay_applied=('wrong','wrong',0)
    elif change=='state':m.state[0]+=.001
    elif change=='joints':f._current_joints[0]+=.001
    elif change=='track':m.tracks['moving']=copy.deepcopy(m.tracks['moving'])
    elif change=='pixels':m.tracks['moving'].previous_image=m.tracks['moving'].previous_image+1
    elif change=='pending':m._pending_prediction={'not_acknowledged':True}
    elif change=='already_trained':f.model.add([.003,0,0],[1.5,0])
    before=frozen(f,m);result=warm_start_visual_goal(f,m,c,o,r)
    assert not result['accepted'] and reason in result['reason']
    assert frozen(f,m)==before


def test_duplicate_import_never_renews_or_clears_new_evidence():
    f,m,c,o,r,p=fixture();assert warm_start_visual_goal(f,m,c,o,r)['accepted']
    before=frozen(f,m);result=warm_start_visual_goal(f,m,c,o,r)
    assert result['reason']=='goal_replay_already_imported' and frozen(f,m)==before


def test_actual_selected_candidate31_imports_eight_aggregates_but_no_offer():
    path=Path(__file__).parents[1]/'runs/visual_field_v31_zero_shot_001/trials/bowl_original_zero_hybrid_v31/trace.jsonl'
    if not path.exists():pytest.skip('Recorded V31 observations unavailable')
    current=None
    for line in path.open():
        row=json.loads(line)
        if row['event']=='executed' and row['step']==30:current=PolicyObservation.from_dict(row['observation'])
        if row['event']=='action' and row['step']==31:
            pred=row['diagnostics']['prediction'];report=row['diagnostics']['reference_replay'];break
    candidate=pred['reference_candidate_resolution']
    field,monitor,pred=configured_field(current,pred,candidate,report)
    result=warm_start_visual_goal(field,monitor,candidate,current,report)
    assert result['accepted'] and result['training_aggregates']==8
    assert field.model.rank==1 and not field.held_out
    assert not field.propose(current.state,joint_position=current.joint_position)['accepted']
    assert field.observed_transitions==0
    assert result['training_measured_path_m']>.027
