"""Acknowledged RGB/state plant, without a model, simulator or object truth."""
import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_egress_evidence_v23 import EgressEvidenceV23,valid_egress_contract
from gpt_field.visual_monitor_v5 import LocalFeatureBundle, _sample
from gpt_field.visual_reference_track_v21 import PartFeatureTrackV21


def warp(image,shift=0.,scale=1.,center=(0.,0.),background=0.):
    yy,xx=np.indices(image.shape)
    destination=np.stack((xx,yy),axis=-1)
    origin=(destination-np.asarray(center)-[shift,0.])/scale+center
    valid=(origin[...,0]>=0)&(origin[...,1]>=0)&(origin[...,0]<image.shape[1]-1)&(origin[...,1]<image.shape[0]-1)
    out=np.full_like(image,background)
    out[valid]=_sample(image,origin[valid])
    return out


class Plant:
    def __init__(self,*,static_tool=False,global_flow=False,wrist_growth=0.,prior_debt=0.):
        self.static_tool,self.global_flow,self.wrist_growth=static_tool,global_flow,wrist_growth
        self.state=np.array([0.,0.,1.,0.,0.,0.,.04,-.04])
        self.origin_state=self.state.copy()
        self.travel=0.;self.frame=0
        self.textures=[np.random.default_rng(180+i).integers(30,230,(31,31)).astype(np.float32) for i in range(3)]
        self.textures=[np.lib.stride_tricks.sliding_window_view(np.pad(t,1),(3,3)).mean(axis=(-1,-2)) for t in self.textures]
        external,wrist=self.images()
        self.tool=PartFeatureTrackV21(external,[100/223,100/223],'tool',roi_uv=[83/223,83/223,117/223,117/223])
        self.ext=LocalFeatureBundle(external,[160/223,110/223],'cab',roi_uv=[143/223,93/223,177/223,127/223])
        self.wrist=LocalFeatureBundle(wrist,[90/223,95/223],'cab-wrist',roi_uv=[73/223,78/223,107/223,112/223])
        self.target=LocalFeatureBundle(external,[60/223,175/223],'target',roi_uv=[43/223,158/223,77/223,192/223])
        self.monitor=SimpleNamespace(tracks={'moving':self.tool},refs={'external':{'moving_kind':'tool'}},state=self.state.copy(),status='pending',
            _external_reliable=lambda:True,cumulative_response_evidence=lambda:{})
        self.guard=SimpleNamespace(tool_track=self.tool,tool_box=np.array([82.,82.,118.,118.]),
            tool_source=np.array([[82.,82.],[118.,82.],[118.,118.],[82.,118.]]),scale=np.array([223.,223.]),
            state=self.state.copy(),observation_travel_m=prior_debt,sweep_observation_epoch=0,
            _tool=lambda m:m.tracks['moving'])
        self.inventory=SimpleNamespace(state=self.state.copy(),observation_index=0,objects={
            'cab':{'views':{'external':self.view(self.ext,[142,92,178,128]),'wrist':self.view(self.wrist,[72,77,108,113])}},
            'target':{'views':{'external':self.view(self.target,[42,157,78,193])}}})
        self.ledger=EgressEvidenceV23()

    def view(self,track,bounds):
        box=np.array(bounds,float)
        source=np.array([box[:2],[box[2],box[1]],box[2:],[box[0],box[3]]])
        return dict(track=track,source=source,projected_roi_uv=(box/223).tolist(),current_visible_roi_uv=(box/223).tolist(),
            support_observed_at=self.frame,support_inlier_count=track.last_inliers)

    def images(self):
        im=np.full((224,224),10.,np.float32);wrist=im.copy()
        for index,point in enumerate([(100,100),(160,110),(60,175)]):
            x,y=point;layer=np.zeros_like(im);layer[y-15:y+16,x-15:x+16]=self.textures[index]
            shift=0. if index or self.static_tool else -1500*self.travel
            if self.global_flow:shift=-1500*self.travel
            shifted=warp(layer,shift=shift)
            im=np.where(shifted>0,shifted,im)
        wrist[80:111,75:106]=self.textures[1]
        scale=1+self.wrist_growth*self.travel
        wrist=warp(wrist,scale=scale,center=(90,95),background=10.)
        return im,wrist

    def proof(self,command):
        return dict(accepted=True,geometry_sha256='fixed-scene',state8=self.state.tolist(),
            geometry_origin_state8=self.origin_state.tolist(),
            start_xyz_m=self.state[:3].tolist(),end_xyz_m=(self.state[:3]+command).tolist(),
            observation_epoch=self.guard.sweep_observation_epoch,target_id='target',initial_halo_ids=['cab'],
            initial_halo_parts=[{'id':'cab','tool_part':'hand_collision'}],scene_ids=['cab','target'],
            complete_route_goal_xyz=[.031,0,1],nominal_core_clear=True,unknown_clear=True,
            no_new_halo_entry=True,monotone_egress=True,exit_goal_full_halo_disjoint=True,
            empty_tool=True,outside_target_contact_band=True,full_finger_range=True,
            phase='recover',translation_mode='separate',purpose='reference_repair',target_contact='none',orientation_margin_rad=.04)

    def assess(self,command=(.002,0,0)):
        command=np.asarray(command,float)
        for _ in range(3):
            result=self.ledger.assess(self.proof(command),self.state,command,guard=self.guard,
                monitor=self.monitor,inventory=self.inventory)
            if not result['accepted'] or result['cap_m']>=np.linalg.norm(command)-1e-10:
                return result
            command=command/np.linalg.norm(command)*result['cap_m']
        raise AssertionError('test executor cap reselection did not converge')

    def move(self,admission,*,actual=.0005,hold=False):
        before=self.state.copy();self.travel+=actual;self.state[0]+=actual;self.frame+=1
        self.guard.observation_travel_m+=actual
        external,wrist=self.images()
        for track,image in [(self.tool,external),(self.ext,external),(self.target,external),(self.wrist,wrist)]:
            assert track.track(image),track.failure_reason
        self.guard.tool_box=np.r_[np.min(self.guard.tool_source@self.tool.matrix+self.tool.translation,axis=0),
                                   np.max(self.guard.tool_source@self.tool.matrix+self.tool.translation,axis=0)]
        self.guard.state=self.state.copy();self.monitor.state=self.state.copy();self.guard.sweep_observation_epoch+=1
        self.inventory.state=self.state.copy();self.inventory.observation_index=self.frame
        for item in self.inventory.objects.values():
            for view in item['views'].values():
                points=view['source']@view['track'].matrix+view['track'].translation
                box=np.r_[points.min(0),points.max(0)]/223
                view.update(projected_roi_uv=box.tolist(),current_visible_roi_uv=box.tolist(),
                    support_observed_at=self.frame,support_inlier_count=view['track'].last_inliers)
        action=np.zeros(7) if hold else np.r_[admission['pending']['delta']/.05,0.,0.,0.,0.]
        return self.ledger.observe(before,action,self.state,pending=admission.get('pending'),guard=self.guard,
            monitor=self.monitor,inventory=self.inventory)


def test_pure_assessment_and_same_observation_previews_do_not_grant_or_spend():
    p=Plant();before=pickle.dumps(p.ledger.__dict__)
    for _ in range(8):assert p.assess()['accepted']
    assert pickle.dumps(p.ledger.__dict__)==before


@pytest.mark.parametrize('gain',[.2,.5])
def test_tracked_synthetic_rgb_slow_motion_continues_over30mm_using_fresh_checkpoints(gain):
    p=Plant()
    for _ in range(200):
        admission=p.assess()
        assert admission['accepted'],(admission,p.ledger.summary())
        p.move(admission,actual=gain*np.linalg.norm(admission['pending']['delta']))
        if p.travel>=.031:break
    report=p.ledger.summary()
    assert .031<=report['egress_measured_m']<=.032
    assert report['spent_measured_m']==pytest.approx(p.travel)
    assert report['fresh_progress_grants']>=8
    assert report['acknowledged_egress_actions']==p.frame
    assert not report['clearance_certified']


@pytest.mark.parametrize('global_flow',[False,True])
def test_static_controlled_seed_or_common_camera_flow_cannot_earn_next_tranche(global_flow):
    p=Plant(static_tool=True,global_flow=global_flow)
    for _ in range(12):
        admission=p.assess()
        if not admission['accepted']:break
        p.move(admission)
    assert not admission['accepted']
    assert p.ledger.egress_travel_m<=.004
    assert p.ledger.grants==0
    assert not p.ledger.last_progress['controlled_feature_motion_supported']


def test_wrist_expansion_vetoes_even_when_external_gap_improves():
    p=Plant(wrist_growth=30.)
    for _ in range(16):
        admission=p.assess()
        if not admission['accepted']:break
        p.move(admission)
    assert not admission['accepted']
    assert p.ledger.last_progress['worsening']
    assert p.ledger.grants==0


def test_prior_uncertain_debt_counts_toward40mm_not_added_to_it():
    p=Plant(prior_debt=.037)
    for _ in range(10):
        admission=p.assess()
        if not admission['accepted']:break
        p.move(admission)
    assert not admission['accepted']
    assert p.ledger.spent_m<=.040+1e-12
    assert p.ledger.egress_travel_m<=.003+1e-12


def test_downstream_hold_does_not_commit_candidate_and_residual_hold_is_charged():
    p=Plant();admission=p.assess();p.move(admission,actual=0.,hold=True)
    assert p.ledger.observations==p.ledger.grants==0
    assert p.ledger.baseline is None
    p.move(p.assess(),actual=.0005)
    p.move(p.assess(),actual=.0002,hold=True)
    assert p.ledger.observations==1
    assert p.ledger.egress_travel_m==pytest.approx(.0007)
    assert p.ledger.spent_m==pytest.approx(.0007)


@pytest.mark.parametrize('fault',['epoch','endpoint','core','unknown','newhalo','target','payload','contact','role'])
def test_malformed_or_stale_geometry_cannot_open_evidence_branch(fault):
    p=Plant();delta=np.array([.002,0,0]);proof=p.proof(delta)
    if fault=='epoch':proof['observation_epoch']+=1
    elif fault=='endpoint':proof['end_xyz_m'][0]+=.001
    elif fault=='target':proof['outside_target_contact_band']=False
    elif fault=='payload':proof['empty_tool']=False
    elif fault=='contact':proof['target_contact']='fingers'
    elif fault=='role':proof['phase']='approach'
    else:proof[{'core':'nominal_core_clear','unknown':'unknown_clear','newhalo':'no_new_halo_entry'}[fault]]=False
    assert not valid_egress_contract(proof,p.state,delta,p.guard.sweep_observation_epoch)


def test_track_loss_rotation_and_visibility_loss_are_not_inconclusive_motion_permission():
    p=Plant();p.move(p.assess())
    p.tool.reliable=False
    assert not p.assess()['accepted']
    p.tool.reliable=True;p.state[3]=.03;p.guard.state=p.state.copy();p.inventory.state=p.state.copy();p.monitor.state=p.state.copy()
    assert p.assess()['reason']=='uncertainty_egress_pose_domain_changed'
    p.state[3]=0.;p.guard.state=p.state.copy();p.inventory.state=p.state.copy();p.monitor.state=p.state.copy()
    p.inventory.objects['cab']['views']['wrist']['current_visible_roi_uv']=None
    assert not p.assess()['accepted']


@pytest.mark.parametrize('revision',['geometry','origin','halo_membership'])
def test_initial_geometry_context_is_bound_only_by_actual_motion_and_cannot_borrow_granted_credit(revision):
    p=Plant()
    first=p.assess()
    assert p.ledger.committed_geometry_context is None
    for _ in range(7):p.move(p.assess())
    assert p.ledger.grants>=1
    before=pickle.dumps(p.ledger.__dict__)
    delta=np.array([.002,0,0]);proof=p.proof(delta)
    if revision=='geometry':proof['geometry_sha256']='revised-model-scene'
    elif revision=='origin':proof['geometry_origin_state8'][0]+=.002
    else:proof['initial_halo_parts'].append({'id':'cab','tool_part':'finger2_collision'})
    result=p.ledger.assess(proof,p.state,delta,guard=p.guard,monitor=p.monitor,inventory=p.inventory)
    assert not result['accepted']
    assert result['reason']=='uncertainty_egress_initial_geometry_context_changed'
    assert pickle.dumps(p.ledger.__dict__)==before


def test_static_reference_reseeding_or_repeated_same_frame_does_not_renew_probe():
    p=Plant(static_tool=True)
    for _ in range(20):
        a=p.assess()
        if not a['accepted']:break
        p.move(a)
    remaining=p.ledger.summary()['observation_tranche_remaining_m']
    external,_=p.images()
    p.tool=PartFeatureTrackV21(external,[100/223,100/223],'renamed-tool')
    p.monitor.tracks['moving']=p.tool;p.guard.tool_track=p.tool
    for _ in range(10):assert not p.assess()['accepted']
    assert p.ledger.summary()['observation_tranche_remaining_m']==remaining
    assert p.ledger.grants==0


def test_duplicate_acknowledgement_cannot_spend_or_grant_evidence_twice():
    p=Plant();a=p.assess();before=p.state.copy();p.move(a)
    spent=p.ledger.spent_m;travel=p.ledger.egress_travel_m
    p.ledger.observe(before,np.r_[a['pending']['delta']/.05,0,0,0,0],p.state,pending=a['pending'],
        guard=p.guard,monitor=p.monitor,inventory=p.inventory)
    assert p.ledger.spent_m==spent and p.ledger.egress_travel_m==travel
    assert p.ledger.last_reason=='uncertainty_egress_acknowledgement_gap'


def test_previous_favorable_motion_does_not_support_another_static_tranche():
    p=Plant()
    while p.ledger.grants<1:p.move(p.assess())
    granted=p.ledger.grants;fixed=p.images()
    p.images=lambda:fixed
    for _ in range(15):
        admission=p.assess()
        if not admission['accepted']:break
        p.move(admission)
    assert not admission['accepted']
    assert p.ledger.grants==granted
    assert not p.ledger.last_progress['controlled_feature_motion_supported']


def test_slow_wrist_looming_cannot_be_erased_by_short_external_progress_checkpoints():
    p=Plant(wrist_growth=4.)
    for _ in range(70):
        admission=p.assess()
        if not admission['accepted']:break
        p.move(admission)
    assert not admission['accepted']
    assert p.ledger.last_progress['worsening']
    assert p.ledger.last_progress['regions']['cab']['wrist_from_first_correspondence']['worsening']
    assert p.ledger.egress_travel_m<.03
