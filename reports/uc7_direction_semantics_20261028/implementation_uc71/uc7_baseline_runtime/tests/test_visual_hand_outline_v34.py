"""Independent outline-witness evidence tests; no policy permission is installed."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from gpt_field.visual_hand_outline_v34 import DistributedHandOutlineWitnessV34
from gpt_field.visual_boundary_evidence_v27 import capture_boundary_evidence,compare_gap_evidence
from gpt_field.visual_reference_track_v32 import ExactDescriptorPartTrackV32
from gpt_field.visual_field import _gray

ROI=np.array([30,25,195,150])/223
STATE=np.array([0.,0.,1.,0.,0.,0.,.04,-.04])
ANCHOR=np.array([208.,180.])

class Plant:
    def __init__(self, *, static_majority=False):
        rng=np.random.default_rng(641)
        self.base=np.full((224,224),110,np.uint8)
        self.body=np.full_like(self.base,110)
        self.mask=np.zeros_like(self.base)
        # Independent textures over a broad actual-body region, with a separate
        # static high-contrast fixture inside the model-declared hand rectangle.
        for y in (42,75,108):
            for x in (45,78,111):
                tile=rng.integers(35,220,(18,18),dtype=np.uint8)
                self.body[y:y+18,x:x+18]=tile
                self.mask[y:y+18,x:x+18]=255
        self.base[34:145,153:191]=rng.integers(0,256,(111,38),dtype=np.uint8)
        if static_majority:
            self.body[:]=110;self.mask[:]=0
            self.body[48:70,53:75]=rng.integers(25,230,(22,22),dtype=np.uint8)
            self.mask[48:70,53:75]=255
        self.base[168:202,192:220]=rng.integers(0,255,(34,28),dtype=np.uint8)

    def image(self, dx=0., dy=0., scale=1., *, illumination=0., camera=(0.,0.), occluded=False):
        cx,cy=96.,80.
        # PIL takes the inverse output->source affine, preserving actual scale.
        transform=(1/scale,0,cx-(cx+dx)/scale,0,1/scale,cy-(cy+dy)/scale)
        body=np.asarray(Image.fromarray(self.body).transform((224,224),Image.Transform.AFFINE,
                         transform,Image.Resampling.BILINEAR,fillcolor=110))
        mask=np.asarray(Image.fromarray(self.mask).transform((224,224),Image.Transform.AFFINE,
                         transform,Image.Resampling.NEAREST,fillcolor=0))
        rgb=self.base.copy();rgb[mask>0]=body[mask>0]
        if occluded:rgb[20:155,25:197]=110
        rgb=np.clip(rgb.astype(float)+illumination,0,255).astype(np.uint8)
        if camera!=(0.,0.):
            rgb=np.asarray(Image.fromarray(rgb).transform((224,224),Image.Transform.AFFINE,
                (1,0,-camera[0],0,1,-camera[1]),Image.Resampling.BILINEAR,fillcolor=110))
        return np.repeat(rgb[...,None],3,axis=2)


def configured(plant=None):
    p=plant or Plant();w=DistributedHandOutlineWitnessV34()
    w.configure(p.image(),ROI,STATE,anchor_position_px=ANCHOR,identity='actual_hand',representation_key=1)
    return p,w

def advance(p,w,n=14,*,step=.5,scale_step=0.,camera=False):
    reports=[]
    for k in range(1,n+1):
        state=STATE.copy();state[0]=.001*k
        rgb=p.image(dx=0 if camera else k*step,scale=1+k*scale_step,
                    camera=(k*step,0.) if camera else (0.,0.))
        reports.append(w.observe(rgb,state,anchor_position_px=ANCHOR+[k*step,0.] if camera else ANCHOR))
    return reports


def test_distributed_translating_body_qualifies_preserving_full_declared_outline():
    p,w=configured();rows=advance(p,w)
    assert any(r['available']for r in rows)
    assert rows[-1]['available']
    assert abs(rows[-1]['scale']-1)<.02
    np.testing.assert_array_equal(w.source_corners,[[30,25],[195,25],[195,150],[30,150]])
    assert w.last_inliers>=6
    assert w.boundary_evidence()['available']
    assert rows[-1]['physical_budgets_modified'] is False


def test_static_majority_cannot_qualify_a_small_local_moving_patch_as_whole_outline():
    p,w=configured(Plant(static_majority=True));rows=advance(p,w)
    assert not any(r['available']for r in rows)
    assert w.selected_ids is None


def test_static_image_and_robot_motion_do_not_qualify():
    p,w=configured();rows=advance(p,w,step=0.)
    assert not any(r['available']for r in rows)
    assert rows[-1]['support']['scene_relative_motion_candidates']==0


def test_observed_common_camera_translation_is_rejected():
    p,w=configured();rows=advance(p,w,step=.5,camera=True)
    assert not any(r['available']for r in rows)


def test_lighting_only_change_is_not_body_translation():
    p,w=configured()
    for k in range(1,15):
        s=STATE.copy();s[0]=k*.001
        r=w.observe(p.image(illumination=k),s,anchor_position_px=ANCHOR)
        assert not r['available']


def test_fresh_robot_motion_with_unchanged_selected_image_does_not_confirm():
    p,w=configured()
    for k in range(1,12):
        s=STATE.copy();s[0]=k*.001
        rgb=p.image(dx=k*.5);w.observe(rgb,s,anchor_position_px=ANCHOR)
        if w.selected_ids is not None:break
    assert w.selected_ids is not None and not w.reliable
    for j in range(1,8):
        state=s.copy();state[0]+=.001*j
        r=w.observe(rgb,state,anchor_position_px=ANCHOR)
        assert not r['available']


@pytest.mark.parametrize('failure',['occlusion','anchor','fingers','rotation'])
def test_qualified_support_loss_or_domain_change_is_unavailable(failure):
    p,w=configured();advance(p,w);assert w.reliable
    state=STATE.copy();state[0]=.015
    if failure=='fingers':state[6]+=.01
    if failure=='rotation':state[3]=.1
    r=w.observe(p.image(dx=7.5,occluded=failure=='occlusion'),state,
                anchor_position_px=ANCHOR,anchor_reliable=failure!='anchor')
    assert not r['available']
    assert not w.boundary_evidence()['available']


def test_real_scale_is_preserved_and_supported_closing_is_not_hidden():
    p,w=configured();advance(p,w,n=10,step=.7,scale_step=.003)
    assert w.reliable
    before=w.boundary_evidence()
    # A separately observed stationary neighbor is represented by actual texture.
    neighbor=ExactDescriptorPartTrackV32(_gray(p.image()),[206/223,180/223],'neighbor',
        roi_uv=np.array([192,168,220,202])/223,support_radius_px=10)
    nc=np.array([[207,165],[222,165],[222,210],[207,210]],float)
    n=capture_boundary_evidence(neighbor,nc,representation_key='neighbor')
    for k in range(11,21):
        s=STATE.copy();s[0]=k*.001
        w.observe(p.image(dx=k*.7,scale=1+k*.003),s,anchor_position_px=ANCHOR)
    assert w.reliable
    assert w.summary()['scale']>1.04
    after=w.boundary_evidence();gap=compare_gap_evidence(before,after,n,n)
    assert gap['comparable'] and gap['classification']=='closing',gap


def test_queries_are_pure_and_do_not_add_features_or_motion():
    p,w=configured();advance(p,w)
    first=w.summary();ids=w.initial_points.copy();path=w.path_m
    for _ in range(5):w.summary();w.boundary_evidence()
    assert w.summary()==first;assert w.path_m==path
    np.testing.assert_array_equal(w.initial_points,ids)


def test_actual_v33_saved_sequence_qualifies_without_posthoc_support_mask():
    from gpt_field.observation import decode_image
    trace=Path('runs/visual_field_v33_zero_shot_001/trials/bowl_original_zero_hybrid_v33/trace.jsonl')
    if not trace.exists():pytest.skip('optional actual archive not distributed')
    rows=[json.loads(x)for x in trace.read_text().splitlines()]
    a={r['step']:r['diagnostics']for r in rows if r['event']=='action'}
    e={r['step']:r['observation']for r in rows if r['event']=='executed'}
    f={r['step']:r['feedback']for r in rows if r['event']=='feedback'}
    anchor=lambda s:np.asarray(f[s]['visual']['tracking']['target']['tracked_point_uv'])*223
    image=lambda s:decode_image(e[s]['observation/image'])
    w=DistributedHandOutlineWitnessV34();roi=a[217]['prediction']['motion_context']['tool_external_roi_uv']
    w.configure(image(216),roi,e[216]['observation/state'],anchor_position_px=anchor(216))
    ready=[];ids=None
    for s in range(217,245):
        r=w.observe(image(s),e[s]['observation/state'],anchor_position_px=anchor(s))
        if r['available']:ready.append(s)
        if w.selected_ids is not None:
            if ids is None:ids=w.selected_ids.copy()
            np.testing.assert_array_equal(ids,w.selected_ids)
    assert ready and ready[0]<235
    assert 243 in ready and 244 in ready
    assert w.last_inliers>=12
    assert .98<w.summary()['scale']<1.06
    assert w.boundary_evidence()['available']
