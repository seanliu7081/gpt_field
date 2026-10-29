"""Immutable RGB evidence, paid handoffs and actual diagnostic003 regression."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from gpt_field.visual_reference_candidates_v24 import CausalReferenceCandidatesV24
from gpt_field.visual_reference_track_v21 import PartFeatureTrackV21
from gpt_field.visual_monitor_v5 import LocalFeatureBundle
from gpt_field.visual_field import _gray
from gpt_field.observation import decode_image,PolicyObservation
from test_visual_reference_candidates_v18 import render,observation


def setup(*,moving=True):
    rgb=render();pixels=_gray(rgb)
    seed=[68,98] if moving else [158,54]
    tool=PartFeatureTrackV21(pixels,np.array(seed)/223,'claimed_tool',roi_uv=np.r_[np.array(seed)-11,np.array(seed)+11]/223)
    anchor=LocalFeatureBundle(pixels,np.array([174,174])/223,'anchor',roi_uv=np.array([162,162,186,186])/223)
    obs=observation(rgb)
    monitor=SimpleNamespace(tracks={'moving':tool,'target':anchor},state=np.array(obs['observation/state']),
        refs={'external':{'moving_kind':'tool','moving_id':'claimed_tool','target_id':'anchor'}},
        _active_reference={'unvalidated_travel_m':.017})
    context={'tool_external_roi_uv':[.2,.30,.43,.57]}
    bank=CausalReferenceCandidatesV24();bank.configure(obs,monitor,context)
    return bank,monitor,context,obs


def advance(bank,monitor,obs,*,steps=4,camera=False,lighting=False,static=False):
    for i in range(1,steps+1):
        rgb=render(0 if static or lighting else i,camera=camera,lighting=i*3 if lighting else 0)
        after=observation(rgb,i*.0009)
        for t in monitor.tracks.values():assert t.track(_gray(rgb)),t.failure_reason
        monitor.state=np.array(after['observation/state']);bank.observe(obs,after,monitor);obs=after
    return obs


def test_prospective_actual_patch_motion_is_supported_without_semantic_or_clearance_claim():
    b,m,c,o=setup();o=advance(b,m,o);r=b.raw_support_evidence(m.state,m)
    assert r['current_bound'] and r['reliable'] and r['causal_supported'],r
    assert r['evidence_kind']=='prospective' and r['fresh_observation_count']==4
    assert not r['semantic_identity_certified'] and not r['clearance_certified']
    assert b.summary()['raw_support_evidence']==r


@pytest.mark.parametrize('kind',['static','global_flow','brightness'])
def test_static_common_flow_and_lighting_do_not_qualify_causal_motion(kind):
    b,m,c,o=setup();o=advance(b,m,o,static=kind=='static',camera=kind=='global_flow',lighting=kind=='brightness')
    r=b.raw_support_evidence(m.state,m)
    assert r['current_bound'] and r['reliable'] and not r['causal_supported'],r
    assert r['confirmation_required'] and not r['contradicted']
    if kind=='static':assert r['raw_support_max_rgb_change']==0


@pytest.mark.parametrize('fault',['stale_state','wrong_track','lost_tool','lost_anchor','rotated','finger_change'])
def test_current_endpoint_identity_and_domain_are_required(fault):
    b,m,c,o=setup();o=advance(b,m,o);assert b.raw_support_evidence(m.state,m)['causal_supported']
    state=m.state.copy()
    if fault=='stale_state':state[0]-=.001
    elif fault=='wrong_track':m.tracks['moving']=copy.deepcopy(m.tracks['moving'])
    elif fault=='lost_tool':m.tracks['moving'].reliable=False
    elif fault=='lost_anchor':m.tracks['target'].reliable=False
    else:
        state[3 if fault=='rotated' else 6]+=.08
        m.state=state.copy();b.current.state=state.copy()
    assert not b.raw_support_evidence(state,m)['causal_supported']


def test_handoff_extracts_existing_sub4mm_history_once_without_motion_or_budget_change():
    b,m,c,o=setup(moving=False);o=advance(b,m,o);oldpath=b.path_m;olddebt=copy.deepcopy(m._active_reference)
    assert oldpath<.004 and not b.summary()['candidates']
    r=b.handoff(o,m,'uncertainty_egress_no_useful_command_within_remaining_budget')
    assert r['review_required'] and r['candidates'];ticket=r['review_id'];count=r['candidate_extraction_count']
    for _ in range(10):
        again=b.handoff(o,m,'uncertainty_egress_no_useful_command_within_remaining_budget')
        assert again['review_id']==ticket and again['candidate_extraction_count']==count
    assert b.path_m==oldpath and m._active_reference==olddebt
    assert not r['control_debt_modified']


@pytest.mark.parametrize('fault',['wrong_reason','stale_observation','lost_anchor','short_history'])
def test_handoff_does_not_bypass_observation_or_domain_gates(fault):
    b,m,c,o=setup(moving=False);old=copy.deepcopy(o)
    o=advance(b,m,o,steps=2 if fault=='short_history' else 4)
    if fault=='lost_anchor':m.tracks['target'].reliable=False
    r=b.handoff(old if fault=='stale_observation' else o,m,
        'arbitrary_reason' if fault=='wrong_reason' else 'uncertainty_egress_fresh_progress_required')
    assert not r['review_required'] and not r['candidates']


def test_candidate_selection_has_retrospective_only_evidence_until_fresh_motion():
    b,m,c,o=setup(moving=False);o=advance(b,m,o)
    r=b.handoff(o,m,'uncertainty_egress_no_useful_command_within_remaining_budget')
    cand=r['candidates'][0];resolved=b.resolve_candidate(cand['candidate_id'],o)
    pixel=_gray(np.array(decode_image(o['observation/image'])))
    m.tracks['moving']=PartFeatureTrackV21(pixel,resolved['current_point_uv'],'selected_tool',roi_uv=resolved['current_roi_uv'])
    b.configure(o,m,c);w=b.raw_support_evidence(m.state,m)
    assert w['selected_candidate_id']==cand['candidate_id'] and w['confirmation_required']
    assert w['evidence_kind']=='retrospective_selection_only' and not w['causal_supported']
    assert not resolved['evidence']['validation_is_prospective']
    path=b.path_m
    for _ in range(5):b.configure(o,m,c);b.observe(o,o,m)
    assert not b.raw_support_evidence(m.state,m)['causal_supported'] and b.path_m==path


@pytest.fixture(scope='module')
def archive003():
    path=Path('runs/visual_field_v23_design_001/controller_diagnostic_003')
    if not path.exists():pytest.skip('optional actual controller RGB archive not present')
    from gpt_field.visual_monitor_v21 import RoleAwareVisualMonitorV21
    rows=[json.loads(x)for x in(path/'trace.jsonl').read_text().splitlines()]
    initial=PolicyObservation.from_dict(next(r['observation']for r in rows if r['event']=='reset'))
    after={r['step']:PolicyObservation.from_dict(r['observation'])for r in rows if r['event']=='executed'}
    actions={r['step']:r['action']for r in rows if r['event']=='action'}
    predictions=[r['prediction']for r in json.loads((path/'deterministic_decisions.json').read_text())]
    b=CausalReferenceCandidatesV24();m=RoleAwareVisualMonitorV21();o=initial
    def images(obs):return {'external':np.array(decode_image(obs.image)),'wrist':np.array(decode_image(obs.wrist_image))}
    for step in range(9):
        if step in (0,1):
            p=predictions[step];m.configure(images(o),p['visual_references'],o.state,intent_id=f'offline-{step}')
            b.configure(o,m,p['motion_context'])
        m.observe(images(after[step]),after[step].state,actions[step]);b.observe(o,after[step],m);o=after[step]
    return b,m,o,predictions,after


def test_actual003_static_raster_is_rejected_despite_sequential_tracker_false_flow(archive003):
    b,m,o,p,after=archive003
    first=np.array(decode_image(after[0].image));last=np.array(decode_image(o.image))
    assert np.array_equal(first[70:91,130:151],last[70:91,130:151])
    w=b.raw_support_evidence(o.state,m)
    assert w['current_bound'] and w['reliable'] and not w['causal_supported'],w
    assert w['reason']=='immutable_raw_support_unchanged' and w['raw_support_max_rgb_change']==0
    assert b.path_m-b._reference_origin_path==pytest.approx(.0039846245853188585)


def test_actual003_handoff_exposes_inside_candidates_without_extra_motion(archive003):
    b,m,o,p,after=archive003
    path=b.path_m;before=b.summary()['candidate_extraction_count']
    r=b.handoff(o,m,'uncertainty_egress_no_useful_command_within_remaining_budget')
    assert r['review_required'] and len(r['candidates'])<=6
    roi=np.asarray(b._declared_roi)
    inside=[c for c in r['candidates']if np.all(np.asarray(c['current_point_uv'])>=roi[:2])and np.all(np.asarray(c['current_point_uv'])<=roi[2:])]
    outside=[c for c in r['candidates']if c not in inside]
    assert len(inside)>=2 and len(outside)>=2
    assert any(abs(c['prior_point_uv'][0]*223-103)<1 and abs(c['prior_point_uv'][1]*223-55)<1 for c in inside)
    assert b.path_m==path and r['candidate_extraction_count']==before+1
    assert all(c['initial_changed_descriptor_pixels']>0 for c in r['candidates'])
    assert all(not c['distal_tool_identity_confirmed'] for c in r['candidates'])


def test_changed_tracker_image_at_same_pose_cannot_reuse_cached_raw_evidence():
    b,m,c,o=setup();o=advance(b,m,o);assert b.raw_support_evidence(m.state,m)['causal_supported']
    for t in m.tracks.values():assert t.track(_gray(render(5)))
    r=b.raw_support_evidence(m.state,m)
    assert not r['current_bound'] and not r['causal_supported']
    assert r['reason']=='reference_rgb_endpoint_mismatch'


def test_actual003_visible_inside_hand_patch_has_nonzero_immutable_motion(archive003):
    # Counterfactual seed evaluation on already recorded pixels: the hypothesis
    # was inspected after the run. This test is not an online semantic selection
    # or a claim of prospective task success.
    _,_,_,predictions,after=archive003
    old=after[0];rgb=np.asarray(decode_image(old.image));pixels=_gray(rgb)
    tool=PartFeatureTrackV21(pixels,np.array([103,51])/223,'visible_hand_hypothesis',roi_uv=np.array([97,45,109,57])/223)
    anchor=LocalFeatureBundle(pixels,[.249,.582],'anchor',roi_uv=[.179,.527,.326,.638])
    m=SimpleNamespace(tracks={'moving':tool,'target':anchor},state=np.array(old.state),
        refs={'external':{'moving_kind':'tool','moving_id':'visible_hand_hypothesis','target_id':'black_bowl'}})
    b=CausalReferenceCandidatesV24();b.configure(old,m,predictions[1]['motion_context'])
    for step in range(1,9):
        current=after[step];gray=_gray(np.array(decode_image(current.image)))
        for track in m.tracks.values():assert track.track(gray),track.failure_reason
        m.state=np.array(current.state);b.observe(old,current,m);old=current
    r=b.raw_support_evidence(old.state,m)
    assert r['causal_supported'],r
    assert np.linalg.norm(r['bias_corrected_relative_motion_px'])>.35
    assert r['raw_support_changed_fraction']>.05 and r['photometric_motion_gain']>0
    assert not r['semantic_identity_certified'] and not r['clearance_certified']


def test_immutable_template_support_survives_more_than_initial_search_radius():
    b,m,c,o=setup();o=advance(b,m,o,steps=18)
    r=b.raw_support_evidence(m.state,m)
    assert r['causal_supported'],r
    assert r['bias_corrected_relative_motion_px'][0]>16
    assert not r['semantic_identity_certified']
