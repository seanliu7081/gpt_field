"""Real RGB trackers with synthetic separated motion/hand roles; no model/sim."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_monitor_v21 import RoleAwareVisualMonitorV21
from gpt_field.visual_monitor_v32 import RoleAwareVisualMonitorV32
from gpt_field.visual_translation_guard_v32 import ProjectedTranslationGuardV32
from gpt_field.visual_reference_candidates_v32 import CausalReferenceCandidatesV32
from gpt_field.visual_field import _gray
from gpt_field.observation import decode_image

T = [np.random.default_rng(3200+i).integers(0, 256, (21, 21, 3), dtype=np.uint8) for i in range(3)]
P = np.array([120, 40]); H = np.array([65, 85]); A = np.array([175, 155])

def images(proxy=0, hand=0, *, missing_hand=False, missing_proxy=False):
    rgb = np.full((224, 224, 3), 50, np.uint8)
    for i, point in enumerate([P+[proxy, 0], H+[hand, 0], A]):
        if (i == 0 and missing_proxy) or (i == 1 and missing_hand): continue
        x,y=point;rgb[y-10:y+11,x-10:x+11]=T[i]
    return {'external':rgb, 'wrist':rgb}

def roi(p, r=14):return np.r_[p-r,p+r].astype(float)/223

def references(role='rigid_transit_proxy', retain=False):
    return {'external':{'reference_role':role,'update':'retain' if retain else 'seed',
        'moving_kind':'tool','moving_id':'eef_rigid_proxy','target_id':'target','goal_offset_uv':[0.,0.],
        'seed':None if retain else {'moving_point_uv':(P/223).tolist(),'moving_roi_uv':roi(P).tolist(),
        'target_anchor_uv':(A/223).tolist(),'target_roi_uv':roi(A).tolist(),
        'tool_point_uv':(H/223).tolist(),'tool_roi_uv':roi(H).tolist()}},'wrist':None}

def context():
    return {'purpose':'task_translation','translation_mode':'transit','intended_contact_id':None,
        'tool_external_roi_uv':roi(H).tolist(),'obstacle_assessment':'present',
        'nearby_obstacles':[{'id':'target','external_roi_uv':roi(A).tolist()}]}

def role_context():
    return {'controller':'visual_field','phase':'approach','target_id':'target','target_contact':'none',
        'contact_alignment':None,'visual_references':references(), 'motion_context':context(),
        'field_intent':{'gripper_command':'hold','orientation_target_xyzw':[0.,0.,0.,1.]}}

def plant():
    m=RoleAwareVisualMonitorV32();state=np.r_[0.,0.,1.,0.,0.,0.,.04,-.04];im=images()
    m.configure(im,references(),state,intent_id=0)
    g=ProjectedTranslationGuardV32();g.configure(im,context(),state,target_id='target',monitor=m)
    g.bind_reference_role_context(role_context())
    return m,g,state

def assess(g,m,delta=(.001,0,0)):
    return g.assess(delta,monitor=m,nominal_clear=True,
        nominal_regions=[{'id':'target','core_distance_m':1.,'uncertainty_m':.03}],orientation_hold=True)

def advance(m,g,state,index,**kwargs):
    nxt=state.copy();nxt[0]=.001*index;im=images(proxy=2*index,hand=index,**kwargs)
    action=[.02,0.,0.,0.,0.,0.,0.]
    m.observe(im,nxt,action);g.observe(im,nxt,action,monitor=m)
    return nxt

def test_proxy_outside_hand_envelope_is_independently_tracked_and_guarded():
    m,g,state=plant()
    assert m._external_reliable() and m.proxy_support_reason() is None
    assert g.context_invalid is None and g.tool_track is m.tracks['tool']
    assert g.tool_track is not m.tracks['moving']
    original=g.tool_box.copy()
    for i in range(1,13):
        result=assess(g,m);assert result['accepted'],result['reason']
        state=advance(m,g,state,i)
    assert g.tool_box[0]-original[0] == pytest.approx(12.,abs=.3)
    assert m.tracks['moving'].position[0]-P[0] == pytest.approx(24.,abs=.3)
    assert g.response.matrix is not None
    assert abs(g.response.matrix[0,0]) < 1300 # actual hand, not the 2x proxy pixel response
    assert not m.attachment_evidence
    assert m.planar_correction([.1,0,1]) is None
    assert m.summary()['external_goal_error_px'] is None

@pytest.mark.parametrize('missing', ['missing_hand','missing_proxy'])
def test_either_observer_loss_blocks_no_proxy_fallback(missing):
    m,g,state=plant();assert assess(g,m)['accepted']
    state=advance(m,g,state,1,**{missing:True})
    result=assess(g,m);assert not result['accepted']
    assert m.control_limit()[0] == 0.
    assert g.tool_track is m.tracks.get('tool')

@pytest.mark.parametrize('mutation', ['direct','gripper','contact','payload','alignment','mode','target','orientation'])
def test_proxy_role_cannot_authorize_contact_rotation_payload_or_other_controller(mutation):
    m,g,_=plant();c=role_context()
    if mutation in ('direct','gripper'):c['controller']=mutation
    elif mutation=='contact':c['target_contact']='fingers'
    elif mutation=='payload':c['visual_references']['external']['moving_kind']='payload'
    elif mutation=='alignment':c['contact_alignment']={'target_id':'target'}
    elif mutation=='mode':c['motion_context']['translation_mode']='interaction'
    elif mutation=='target':c['target_id']='different'
    else:c['field_intent']['orientation_target_xyzw']=[0.,0.,.1,.995]
    g.bind_reference_role_context(c)
    result=assess(g,m);assert not result['accepted'],mutation
    assert result['reason'].startswith('translation_proxy_')
    assert g._validated_pending is None and g._proxy_role_pending is None

@pytest.mark.parametrize('fault', ['missing_hand_point','missing_hand_roi','same_support','role_change'])
def test_monitor_enforces_independent_hand_and_explicit_role_lifecycle(fault):
    m,g,state=plant();refs=references(retain=fault=='role_change')
    if fault=='missing_hand_point':refs['external']['seed']['tool_point_uv']=None
    elif fault=='missing_hand_roi':refs['external']['seed']['tool_roi_uv']=None
    elif fault=='same_support':
        refs['external']['seed']['tool_point_uv']=(P/223).tolist();refs['external']['seed']['tool_roi_uv']=roi(P).tolist()
    else:refs['external']['reference_role']='tool_anatomy'
    m.configure(images(),refs,state,intent_id=1)
    assert m.control_limit()[0] == 0.


def test_retain_preserves_both_tracks_and_does_not_restore_physical_debt():
    m,g,state=plant();assert assess(g,m)['accepted'];state=advance(m,g,state,1)
    tracks=m.tracks.copy();debt=copy.deepcopy(m._active_reference)
    m.configure(images(2,1),references(retain=True),state,intent_id=0)
    assert m.tracks['moving'] is tracks['moving'] and m.tracks['tool'] is tracks['tool']
    encode=lambda value:json.dumps(value,sort_keys=True,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))
    assert encode(m._active_reference) == encode(debt)
    m.tracks['tool'].reliable=False
    m.configure(images(2,1),references(retain=True),state,intent_id=0)
    assert m.control_limit()[0] == 0. and not m.tracks['tool'].reliable


def test_cumulative_pose_domain_change_rejects_even_small_per_step_rotation():
    m,g,state=plant()
    for i in range(1,5):
        nxt=state.copy();nxt[3]=i*.01;m.observe(images(),nxt,[0.]*7);g.observe(images(),nxt,[0.]*7,monitor=m);state=nxt
    assert m.control_limit()[1]=='proxy_fixed_configuration_domain_changed'
    assert not assess(g,m)['accepted']


def test_role_rebind_after_admission_is_acknowledgement_mismatch_with_actual_travel_retained():
    m,g,state=plant();assert assess(g,m)['accepted']
    c=role_context();c['target_contact']='fingers';g.bind_reference_role_context(c)
    after=advance(m,g,state,1)
    assert g.domain_invalid=='translation_proxy_acknowledgement_context_mismatch'
    assert np.array_equal(g.state,after)
    assert g.proxy_unmatched_travel_m == pytest.approx(.001)
    assert m.path_m == pytest.approx(.001)


def test_anatomy_uses_original_moving_envelope_not_independent_tool():
    m=RoleAwareVisualMonitorV32();im=images();state=np.r_[0.,0.,1.,0.,0.,0.,.04,-.04]
    m.configure(im,references('tool_anatomy'),state)
    ctx=context();ctx['tool_external_roi_uv']=roi(P).tolist();g=ProjectedTranslationGuardV32()
    g.configure(im,ctx,state,target_id='target',monitor=m)
    assert g.tool_track is m.tracks['moving'] and g.context_invalid is None
    assert m.summary()['external_goal_error_px'] is not None


def test_noncontact_align_is_consistent_with_proxy_wire_role():
    m,g,_=plant();c=role_context();c['phase']='align';g.bind_reference_role_context(c)
    assert assess(g,m)['accepted']


def test_preview_restores_role_pending_and_existing_physical_ledgers():
    m,g,state=plant();assert assess(g,m)['accepted']
    original=g.__dict__;pending=g._proxy_role_pending
    result=g.preview_assess([.001,0,0],monitor=m,nominal_clear=True,
        nominal_regions=[{'id':'target','core_distance_m':1.,'uncertainty_m':.03}],orientation_hold=True)
    assert result['accepted'] and g.__dict__ is original and g._proxy_role_pending is pending
    assert g.observation_travel_m==0.
    advance(m,g,state,1)
    assert g.domain_invalid is None


def test_saved_v31_rgb_supports_independent_hand_without_using_proxy_affine():
    """Passive recorded RGB, new explicitly labelled hand hypothesis; no actions."""
    path=Path(__file__).parents[1]/'runs/visual_field_v31_zero_shot_001/trials/bowl_original_zero_hybrid_v31/trace.jsonl'
    if not path.exists():pytest.skip('Optional recorded V31 RGB archive unavailable')
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    actions={row['step']:row for row in rows if row['event']=='action'}
    observations={row['step']:row['observation'] for row in rows if row['event']=='executed'}
    def ims(obs):return {'external':decode_image(obs['observation/image']),
                        'wrist':decode_image(obs['observation/wrist_image'])}
    refs=copy.deepcopy(actions[31]['diagnostics']['prediction']['visual_references'])
    ext=refs['external'];ext['reference_role']='rigid_transit_proxy';refs['wrist']=None
    # The model-selected candidate and actual recorded transitions stay exact.
    # This independently inspected hand patch is a NEW offline hypothesis,
    # not the archived model output, semantic truth, or a task-success test.
    ext['seed']['tool_point_uv']=[.55,.28]
    ext['seed']['tool_roi_uv']=[.49,.22,.615,.345]
    m=RoleAwareVisualMonitorV32();initial=observations[30]
    m.configure(ims(initial),refs,initial['observation/state'])
    assert m._external_reliable()
    ctx=copy.deepcopy(actions[31]['diagnostics']['prediction']['motion_context'])
    g=ProjectedTranslationGuardV32()
    g.configure(ims(initial),ctx,initial['observation/state'],target_id='black_bowl',monitor=m)
    source=g.tool_source.copy()
    for i in range(31,43):
        obs=observations[i];m.observe(ims(obs),obs['observation/state'],actions[i]['action'])
        g.observe(ims(obs),obs['observation/state'],actions[i]['action'],monitor=m)
        assert m._external_reliable() and g.tool_track is m.tracks['tool']
    hand=m.tracks['tool'];proxy=m.tracks['moving']
    corners=source@hand.matrix+hand.translation
    assert np.allclose(g.tool_box,np.r_[corners.min(0),corners.max(0)])
    assert not np.allclose(hand.matrix,proxy.matrix)
    assert not g.summary()['rigid_proxy_separation']['proxy_motion_copied_to_hand_envelope']


def test_candidate_summary_allows_role_choice_without_semantic_claim_or_motion_credit():
    b=CausalReferenceCandidatesV32();s=b.summary()
    assert s['candidate_role_options']==['tool_anatomy','rigid_transit_proxy']
    assert 'separate actual-hand' in s['selection_requires']
    assert not s['proxy_defines_anatomical_goal_or_clearance']
    assert not s['automatic_reference_promotion'] and not s['control_debt_modified']
