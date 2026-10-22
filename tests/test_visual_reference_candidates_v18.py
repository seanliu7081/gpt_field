"""Passive repair hypotheses, bounded evidence, and actual archived RGB."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.observation import image_to_data_url, decode_image
from gpt_field.visual_reference_candidates_v18 import CausalReferenceCandidatesV18


def observation(rgb, distance=0., *, rot=0., fingers=.04):
    return {"observation/image":image_to_data_url(rgb,rotate=False),
            "observation/state":[distance,0.,0.,rot,0.,0.,fingers,-fingers],
            "observation/joint_position":[0.]*7}


def render(offset=0, *, camera=False, lighting=0, noise=0):
    rng=np.random.default_rng(431)
    img=np.full((224,224,3),90,np.uint8)
    for x,y,size in [(50,80,36),(160,160,28),(145,40,28)]:
        patch=rng.integers(20,220,(size,size,3),dtype=np.uint8)
        dx=offset if x==50 or camera and x==160 else 0
        img[y:y+size,x+dx:x+dx+size]=patch
    if lighting:
        img=np.clip(img.astype(int)+lighting,0,255).astype(np.uint8)
    if noise:
        img=np.clip(img.astype(int)+rng.integers(-noise,noise+1,img.shape),0,255).astype(np.uint8)
    return img


def setup():
    moving=SimpleNamespace(reliable=True,position=np.array([158.,54.]),
        points=np.array([[152.,48.],[160.,50.],[157.,59.],[165.,60.]]))
    anchor=SimpleNamespace(reliable=True,position=np.array([174.,174.]))
    monitor=SimpleNamespace(refs={"external":{"moving_kind":"tool","moving_id":"claimed_tool","target_id":"anchor"}},
        tracks={"moving":moving,"target":anchor},_active_reference={"unvalidated_travel_m":.017})
    context={"tool_external_roi_uv":[.60,.12,.82,.35]}
    obs=observation(render());bank=CausalReferenceCandidatesV18();bank.configure(obs,monitor,context)
    return bank,monitor,context,obs


def advance(bank,monitor,obs,count=5,**kwargs):
    for i in range(1,count+1):
        if kwargs.get("camera"):
            monitor.tracks['target'].position=np.array([174.+i,174.])
        after=observation(render(i,**kwargs),i*.001)
        bank.observe(obs,after,monitor);obs=after
    return obs


def test_positive_motion_outside_wrong_tool_roi_offers_bounded_hypotheses_early():
    bank,monitor,context,obs=setup();debt=copy.deepcopy(monitor._active_reference)
    obs=advance(bank,monitor,obs,4);report=bank.summary()
    assert report['review_required'] and report['measured_path_since_reference_m']==pytest.approx(.004)
    assert 1<=len(report['candidates'])<=6
    assert any(c['current_point_uv'][0]<.5 for c in report['candidates'])
    assert not report['two_distinct_informative_directions']
    assert monitor._active_reference==debt and not report['control_debt_modified']
    assert not report['automatic_reference_promotion'] and not report['semantic_identity_certified']
    assert all(c['inliers']>=3 and not c['distal_tool_identity_confirmed'] for c in report['candidates'])
    assert 'data:image' not in json.dumps(report) and 'chronological_samples' not in json.dumps(report)


def test_exact_candidate_resolution_digest_and_history_are_bound_and_isolated():
    bank,monitor,_,obs=setup();obs=advance(bank,monitor,obs)
    candidate=bank.summary()['candidates'][0];result=bank.resolve_candidate(candidate['candidate_id'],obs)
    assert result['current_point_uv']==candidate['current_point_uv']
    assert result['current_roi_uv']==candidate['current_roi_uv']
    evidence=result['evidence'];samples=evidence['chronological_samples']
    assert len(samples)>=4 and samples[-1]['state8']==obs['observation/state']
    assert len(samples)<=bank.maximum_frames and all(a['sequence']<b['sequence'] for a,b in zip(samples,samples[1:]))
    assert evidence['guard_anchor_source']=='explicit_stationary_anchor'
    assert not evidence['validation_is_prospective'] and evidence['fresh_held_out_confirmation_required']
    assert result['evidence_sha256']==hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    samples[-1]['state8'][0]=99
    assert bank.resolve_candidate(candidate['candidate_id'],obs)['evidence']['chronological_samples'][-1]['state8']==obs['observation/state']
    stale=copy.deepcopy(obs);stale['observation/state'][0]-=.001
    with pytest.raises(ValueError,match='observation_mismatch'):bank.resolve_candidate(candidate['candidate_id'],stale)
    with pytest.raises(ValueError,match='unavailable'):bank.resolve_candidate('invented-id',obs)


def test_stationary_holds_preserve_review_identity_and_do_not_reextract_or_renew():
    bank,monitor,context,obs=setup();obs=advance(bank,monitor,obs)
    before=bank.summary();binding=before['observation_binding'];image=bank.image(obs)
    for _ in range(25):
        bank.configure(obs,monitor,context);bank.observe(obs,obs,monitor)
    after=bank.summary()
    assert before['review_id']==after['review_id']
    assert before['candidate_extraction_count']==after['candidate_extraction_count']==1
    assert after['measured_path_m']==before['measured_path_m']
    assert after['observation_binding']==binding
    assert len(after['candidates'])==len(before['candidates'])
    assert image.startswith('data:image/png') and bank.image(obs).startswith('data:image/png')
    assert decode_image(bank.image(obs)).size==(896,556)


def test_partially_out_of_image_candidate_is_never_offered_or_resolved():
    bank,monitor,_,obs=setup();obs=advance(bank,monitor,obs)
    candidate=bank._candidates[0];cid=candidate['id']
    candidate['track'].translation[0]-=250
    assert cid not in [c['candidate_id'] for c in bank.summary()['candidates']]
    with pytest.raises(ValueError,match='unavailable'):bank.resolve_candidate(cid,obs)


def test_global_camera_like_flow_is_not_independent_candidate_motion():
    bank,monitor,_,obs=setup();obs=advance(bank,monitor,obs,7,camera=True)
    assert not bank.summary()['candidates'] and not bank.summary()['review_required']


@pytest.mark.parametrize('kind',['static','brightness','noise'])
def test_static_scene_and_nontranslating_appearance_do_not_offer_motion(kind):
    bank,monitor,_,obs=setup()
    for i in range(1,7):
        after=observation(render(0,lighting=i*3 if kind=='brightness' else 0,noise=1 if kind=='noise' else 0),i*.001)
        bank.observe(obs,after,monitor);obs=after
    assert not bank.summary()['candidates'] and not bank.summary()['review_required']


def test_unrelated_moving_texture_is_only_a_hypothesis_never_semantic_evidence():
    # This renderer could depict a moving shadow/object just as well as a tool;
    # the numerical evidence must never answer that semantic question.
    bank,monitor,_,obs=setup();obs=advance(bank,monitor,obs)
    report=bank.summary()
    assert report['candidates']
    assert all(not c['semantic_identity_certified'] and not c['distal_tool_identity_confirmed']
               for c in report['candidates'])
    assert 'shadows' in report['selection_requires']
    assert not report['automatic_reference_promotion']


def test_second_distinct_direction_strengthens_reason_without_second_review_event():
    bank,monitor,_,obs=setup();obs=advance(bank,monitor,obs,5);review=bank.summary()['review_id']
    for j in range(1,5):
        after=observation(render(5+j),.005);after['observation/state'][2]=j*.001
        bank.observe(obs,after,monitor);obs=after
    report=bank.summary()
    assert report['two_distinct_informative_directions']
    assert report['review_id']==review
    assert report['reason']=='uninformative_reference_after_distinct_motion'


def test_retained_frames_and_resolved_rows_stay_bounded_during_long_holds():
    bank,monitor,_,obs=setup();obs=advance(bank,monitor,obs,5)
    cid=bank.summary()['candidates'][0]['candidate_id']
    for _ in range(90):bank.observe(obs,obs,monitor)
    assert bank.summary()['retained_history_frames']<=bank.maximum_frames
    assert len(bank.resolve_candidate(cid,obs)['evidence']['chronological_samples'])<=bank.maximum_frames


@pytest.mark.parametrize('change',['rotation','fingers','lost_anchor'])
def test_domain_loss_invalidates_resolver_without_refunding_debt(change):
    bank,monitor,_,obs=setup();obs=advance(bank,monitor,obs);cid=bank.summary()['candidates'][0]['candidate_id'];oldpath=bank.path_m
    if change=='lost_anchor':monitor.tracks['target'].reliable=False
    after=copy.deepcopy(obs)
    if change=='rotation':after['observation/state'][3]=.08
    if change=='fingers':after['observation/state'][6]=.01
    bank.observe(obs,after,monitor)
    assert not bank.summary()['candidates'] and bank.path_m==oldpath
    with pytest.raises(ValueError):bank.resolve_candidate(cid,after)


def test_new_seed_or_renaming_does_not_modify_any_external_budget_and_requires_new_motion():
    bank,monitor,context,obs=setup();obs=advance(bank,monitor,obs);path=bank.path_m;debt=copy.deepcopy(monitor._active_reference)
    cid=bank.summary()['candidates'][0]['candidate_id']
    cached=bank.resolve_candidate(cid,obs)
    frames=bank.summary()['retained_history_frames']
    monitor.tracks['moving']=copy.deepcopy(monitor.tracks['moving']);monitor.refs['external']['moving_id']='renamed'
    report=bank.configure(obs,monitor,context)
    assert not report['review_required'] and report['review_id'] is None
    assert bank.path_m==path and monitor._active_reference==debt
    assert not report['candidates']  # No six-bundle tracking cost after selection.
    assert report['retained_history_frames']==frames
    assert cached['evidence']['chronological_samples'][-1]['state8']==obs['observation/state']
    with pytest.raises(ValueError):bank.resolve_candidate(cid,obs)


def test_bad_acknowledgment_and_stale_image_never_repair():
    bank,monitor,_,obs=setup();old=obs;obs=advance(bank,monitor,obs)
    assert bank.image(old) is None
    report=bank.observe(old,obs,monitor)
    assert report['status']=='unacknowledged_transition_gap' and not report['candidates']


def test_archived_v17_exposes_moving_alternatives_before_eight_mm():
    path=Path('runs/visual_field_v17_zero_shot_001/trials/bowl_original_zero_hybrid_v17/trace.jsonl')
    if not path.exists():pytest.skip('optional recorded RGB archive unavailable')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line]
    initial=next(r['observation'] for r in rows if r['event']=='reset')
    actions={r['step']:r for r in rows if r['event']=='action'}
    executed={r['step']:r for r in rows if r['event']=='executed'}
    feedback={r['step']:r['feedback'] for r in rows if r['event']=='feedback'}
    moving=SimpleNamespace(reliable=True,position=np.array([.634,.371])*223,
        points=np.array([[138.,77.],[147.,80.],[141.,87.],[150.,90.]]))
    anchor=SimpleNamespace(reliable=True,position=np.array([.246,.585])*223)
    monitor=SimpleNamespace(refs={'external':{'moving_kind':'tool','moving_id':'tool','target_id':'black_bowl'}},tracks={'moving':moving,'target':anchor})
    bank=CausalReferenceCandidatesV18();bank.configure(initial,monitor,actions[0]['diagnostics']['prediction']['motion_context']);obs=initial
    first=None
    for step in range(30):
        summary=feedback[step]['visual']['tracking']['target'];anchor.position=np.array(summary['tracked_point_uv'])*223;anchor.reliable=summary['reliable']
        after=executed[step]['observation'];report=bank.observe(obs,after,monitor);obs=after
        if report['review_required'] and first is None:first=copy.deepcopy(report)
    assert first is not None
    assert .004<=first['measured_path_m']<.008
    assert first['support_max_mean_rgb_change']==0
    assert any(np.linalg.norm(c['relative_displacement_px'])>.5 for c in first['candidates'])
    assert len(first['candidates'])<=6
    assert not first['semantic_identity_certified']
