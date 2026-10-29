"""Passive, distributed RGB support for a declared rigid-hand outline.

This witness measures a *hypothesized* rigid image region. It neither segments
robot anatomy nor changes the declared outline, a controller, or any budget.
A fixed initial corner bank avoids an indefinitely self-validating template
chain. Scene-relative motion and new acknowledged correspondence continuity
separate distributed hand-region support from static pixels inside its box.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import numpy as np

from .visual_field import _gray
from .visual_monitor_v4 import _orientation_distance
from .visual_monitor_v5 import LocalFeatureBundle, _sample, _ncc, _similarity
from .visual_reference_candidates_v24 import CausalReferenceCandidatesV24
from .visual_boundary_evidence_v27 import capture_boundary_evidence


class DistributedHandOutlineWitnessV34:
    """Immutable ROI/corner bank; no simulator data or automatic control.

    `configure` establishes one explicit measurement epoch. `observe` consumes
    actual RGB, measured state and a separately observed stationary-reference
    position. Queries are pure. Reseeding this witness confers no physical
    allowance and must not clear any caller-owned warning or evidence ledger.
    """
    minimum_selection_path_m = .004
    minimum_confirmation_path_m = .001
    minimum_features = 6
    maximum_features = 64
    minimum_motion_px = .35
    maximum_rotation_rad = .035
    maximum_finger_change_m = .002
    consensus_residual_px = .75

    def __init__(self):
        self.configured = False
        self.reliable = False
        self.reason = 'outline_not_configured'
        self.selected_ids = None
        self._selection_path = None
        self._selection_observations = None
        self._current_bound = None
        self._last_report = None
        self.qualification_observation = None
        self._selection_snapshot = None

    @staticmethod
    def _state(value):
        v = np.asarray(value, float)
        if v.shape != (8,) or not np.isfinite(v).all():
            raise ValueError('state8 must be eight finite measured values')
        return v.copy()

    @staticmethod
    def _anchor(value):
        v = np.asarray(value, float)
        if v.shape != (2,) or not np.isfinite(v).all():
            raise ValueError('anchor_position_px must contain two finite coordinates')
        return v.copy()

    def configure(self, pixels, roi_uv, state8, *, anchor_position_px,
                  identity='declared_actual_hand', representation_key=None):
        gray = _gray(pixels)
        roi = np.asarray(roi_uv, float)
        if (roi.shape != (4,) or not np.isfinite(roi).all()
                or np.any(roi < 0) or np.any(roi > 1) or np.any(roi[:2] >= roi[2:])):
            raise ValueError('outline ROI must be an ordered current image rectangle')
        self.state = self._state(state8)
        self.origin_state = self.state.copy()
        self.anchor = self._anchor(anchor_position_px)
        self.origin_anchor = self.anchor.copy()
        self.scale = np.array([gray.shape[1]-1, gray.shape[0]-1], float)
        self.roi_uv = roi.copy()
        self.box = roi*np.tile(self.scale, 2)
        lo, hi = self.box[:2], self.box[2:]
        self.source_corners = np.array([lo, [hi[0],lo[1]], hi, [lo[0],hi[1]]])
        self.ident = str(identity)
        self.representation_key = copy.deepcopy(representation_key)
        self.seed_image = gray.copy()
        self.previous_image = gray.copy()
        self.offsets = np.stack(np.meshgrid(np.arange(-3,4),np.arange(-3,4)),axis=-1)
        # Fixed spatial quotas prevent one high-contrast fixture edge filling
        # the bank. These are support candidates, not automatically hand pixels.
        points = []
        for fy in (1/6, .5, 5/6):
            for fx in (.125, .375, .625, .875):
                point = lo+(hi-lo)*[fx,fy]
                try:
                    bank = LocalFeatureBundle(gray,point/self.scale,self.ident,
                        roi_uv=roi,support_radius_px=12,maximum_features=6)
                except ValueError:
                    continue
                for p in bank.initial_points:
                    if not any(np.linalg.norm(p-q)<1e-8 for q in points):
                        points.append(p.copy())
        self.initial_points = np.asarray(points[:self.maximum_features],float).reshape(-1,2)
        self.templates = [_sample(gray,p+self.offsets) for p in self.initial_points]
        self.self_matches = [CausalReferenceCandidatesV24._match_around(t,gray,p)
                             for p,t in zip(self.initial_points,self.templates)]
        self._positions = self.initial_points.copy()
        self._valid = np.array([x is not None for x in self.self_matches],bool)
        self.matrix, self.translation = np.eye(2), np.zeros(2)
        self.active = np.array([],int)
        self.points = np.empty((0,2))
        self.position = (lo+hi)/2
        self.last_inliers, self.last_residual_px = 0, 0.
        self.path_m, self.observations = 0., 0
        self.selected_ids = None
        self._selection_path = self._selection_observations = None
        self.qualification_observation = None
        self._selection_snapshot = None
        self.configured = True
        self.reliable = False
        self.reason = ('collecting_distributed_motion_support'
                       if len(self.initial_points)>=self.minimum_features
                       else 'insufficient_initial_outline_texture')
        self._invalid = None
        self._metrics = {}
        self._current_bound = self._binding(gray,self.state)
        self._last_report = None
        return self.summary()

    @staticmethod
    def _binding(gray, state):
        return hashlib.sha256(np.ascontiguousarray(gray).tobytes()
                              +np.ascontiguousarray(state).tobytes()).hexdigest()

    def _diverse(self, ids):
        if len(ids)<self.minimum_features:
            return False
        points = self.initial_points[np.asarray(ids,int)]
        uv = (points-self.box[:2])/(self.box[2:]-self.box[:2])
        cells = np.minimum(np.maximum((uv*[4,3]).astype(int),0),[3,2])
        centered=points-points.mean(0)
        return (len(set(map(tuple,cells)))>=3 and len(set(cells[:,0]))>=2
                and len(set(cells[:,1]))>=2 and np.linalg.matrix_rank(centered)==2)

    def _fit(self, ids, destination, anchor_delta):
        ids=np.asarray(ids,int); dst=np.asarray(destination,float)
        if not self._diverse(ids):return None
        src=self.initial_points[ids]
        best=None
        for a,b in itertools.combinations(range(len(ids)),2):
            if np.linalg.norm(src[a]-src[b])<7:continue
            fitted=_similarity(src[[a,b]],dst[[a,b]])
            if fitted is None:continue
            matrix,shift=fitted
            # Same total-scale range as the inherited local feature matcher.
            if not .7<=np.linalg.norm(matrix[0])<=1.4:continue
            error=np.linalg.norm(src@matrix+shift-dst,axis=1)
            keep=error<=self.consensus_residual_px
            if not self._diverse(ids[keep]):continue
            key=(int(keep.sum()),-float(error[keep].sum()))
            if best is None or key>best[0]:best=(key,keep)
        if best is None:return None
        keep=best[1]; ids=ids[keep];src=src[keep];dst=dst[keep]
        matrix,shift=_similarity(src,dst)
        residual=np.linalg.norm(src@matrix+shift-dst,axis=1)
        precision=max(.08,float(np.sqrt(np.mean(residual*residual)))/np.sqrt(len(ids)))
        relative=np.median(dst-src-anchor_delta,axis=0)
        threshold=max(self.minimum_motion_px,3*precision)
        return {'ids':ids,'source':src,'destination':dst,'matrix':matrix,
            'translation':shift,'residual':float(np.sqrt(np.mean(residual*residual))),
            'point_precision_px':precision,'relative_center_flow_px':relative,
            'motion_threshold_px':threshold,
            'relative_motion_rms_px':float(np.sqrt(np.mean(np.sum((dst-src-anchor_delta)**2,axis=1))))}

    def observe(self, pixels, state8, *, anchor_position_px, anchor_reliable=True):
        if not self.configured:raise RuntimeError('configure outline witness first')
        gray=_gray(pixels);state=self._state(state8)
        distance=float(np.linalg.norm(state[:3]-self.state[:3]))
        self.path_m+=distance;self.observations+=1
        self.state=state;self._current_bound=self._binding(gray,state)
        if gray.shape!=self.seed_image.shape:self._invalid='image_shape_changed'
        if not anchor_reliable:self._invalid='stationary_reference_unavailable'
        try:anchor=self._anchor(anchor_position_px)
        except ValueError:
            anchor=self.anchor;self._invalid='stationary_reference_unavailable'
        if (_orientation_distance(state[3:6],self.origin_state[3:6])>self.maximum_rotation_rad
                or np.max(abs(state[6:]-self.origin_state[6:]))>self.maximum_finger_change_m):
            self._invalid='rigid_hand_configuration_domain_changed'
        if self._invalid:
            self.reliable=False;self.reason=self._invalid;return self.summary()
        anchor_delta=anchor-self.origin_anchor
        self.anchor=anchor
        positions=self._positions.copy();valid=np.zeros(len(positions),bool);gains=np.zeros(len(positions))
        for i,(source,template,baseline) in enumerate(zip(self.initial_points,self.templates,self.self_matches)):
            if baseline is None:continue
            # Search centers can follow the last accepted match. Descriptors
            # remain the original raster; inactive recovery uses that same bank.
            center=positions[i]
            forward=CausalReferenceCandidatesV24._match_around(template,gray,center)
            if forward is None:continue
            backward=CausalReferenceCandidatesV24._match_around(
                _sample(gray,forward[0]+self.offsets),self.seed_image,source)
            if backward is None or np.linalg.norm(backward[0]-source)>.9:continue
            # Independent immutable baseline self-match removes numerical NCC
            # interpolation bias without treating a changed descriptor as static.
            point=source+forward[0]-baseline[0]
            translated=_sample(gray,point+self.offsets)
            scene_only=_sample(gray,source+anchor_delta+self.offsets)
            if translated is None or scene_only is None:continue
            gains[i]=_ncc(template,translated)-_ncc(template,scene_only)
            positions[i]=point;valid[i]=True
        self._positions=positions;self._valid=valid
        flow=positions-self.initial_points-anchor_delta
        # These are the established raw-motion tests, evaluated causally on
        # every current observation, not a hand-picked final-frame motion mask.
        eligible=valid&(np.linalg.norm(flow,axis=1)>self.minimum_motion_px)&(gains>1e-4)
        ids=np.flatnonzero(eligible)
        if self.selected_ids is not None:
            ids=np.array([i for i in self.selected_ids if eligible[i]],int)
        fit=self._fit(ids,positions[ids],anchor_delta)
        self._metrics={'bank_features':len(positions),'matched_features':int(valid.sum()),
            'scene_relative_motion_candidates':int(eligible.sum()),'anchor_delta_px':anchor_delta.tolist(),
            'selection_was_retrospective':self.selected_ids is not None,
            'current_motion_gains':gains[ids].tolist()}
        self.previous_image=gray.copy()
        if fit is None or fit['relative_motion_rms_px']<=fit['motion_threshold_px']:
            self.reliable=False
            self.reason=('selected_distributed_support_unavailable' if self.selected_ids is not None
                         else 'distributed_scene_relative_motion_inconclusive')
            return self.summary()
        self.active,self.points=fit['ids'],fit['destination']
        self.matrix,self.translation=fit['matrix'],fit['translation']
        self.position=((self.box[:2]+self.box[2:])/2)@self.matrix+self.translation
        self.last_inliers,self.last_residual_px=len(self.active),fit['residual']
        self._metrics.update({k:(v.tolist() if isinstance(v,np.ndarray) else v)for k,v in fit.items()
                              if k not in('ids','source','destination','matrix','translation')})
        if self.selected_ids is None:
            if self.path_m<self.minimum_selection_path_m:
                self.reason='awaiting_acknowledged_selection_motion';self.reliable=False
                return self.summary()
            self.selected_ids=self.active.copy()
            self._selection_path=self.path_m
            self._selection_observations=self.observations
            self._selection_snapshot={'image':gray.copy(),'anchor':anchor.copy(),
                'positions':{int(i):positions[i].copy() for i in self.selected_ids}}
            self.reason='awaiting_fresh_selected_support_continuity';self.reliable=False
            return self.summary()
        first=self._selection_snapshot
        fresh_flows=[];fresh_gains=[]
        for i in self.active:
            original=first['positions'][int(i)]
            prior=_sample(first['image'],original+self.offsets)
            moved=_sample(gray,positions[i]+self.offsets)
            unchanged=_sample(gray,original+anchor-first['anchor']+self.offsets)
            if any(x is None for x in (prior,moved,unchanged)):continue
            fresh_flows.append(positions[i]-original-(anchor-first['anchor']))
            fresh_gains.append(_ncc(prior,moved)-_ncc(prior,unchanged))
        fresh_rms=(float(np.sqrt(np.mean(np.sum(np.asarray(fresh_flows)**2,axis=1))))
                   if fresh_flows else 0.)
        fresh_gain=float(np.median(fresh_gains)) if fresh_gains else 0.
        fresh=(self.path_m-self._selection_path>=self.minimum_confirmation_path_m
               and self.observations-self._selection_observations>=2
               and len(fresh_flows)>=self.minimum_features
               and fresh_rms>fit['motion_threshold_px'] and fresh_gain>1e-4)
        self._metrics.update(fresh_selected_relative_motion_rms_px=fresh_rms,
                             fresh_selected_photometric_gain=fresh_gain)
        self.reliable=bool(fresh)
        self.reason=('distributed_outline_correspondence_supported' if fresh
                     else 'awaiting_fresh_selected_support_continuity')
        if fresh and self.qualification_observation is None:
            self.qualification_observation=self.observations
        return self.summary()

    def boundary_evidence(self, *, representation_key=None):
        result=capture_boundary_evidence(self,self.source_corners,
            representation_key=self.representation_key if representation_key is None else representation_key)
        result.update({'evidence_source':'independent_distributed_outline_v34',
            'observation_binding':self._current_bound,'state8':self.state.tolist(),
            'semantic_rigidity_certified':False,'declared_outline_shrunk':False,
            'physical_budgets_modified':False})
        return result

    def summary(self):
        result={'version':'distributed-immutable-hand-outline-v34','configured':self.configured,
            'available':self.reliable,'reason':self.reason,'semantic_identity_certified':False,
            'physical_budgets_modified':False,'clearance_certified':False}
        if not self.configured:return result
        corners=self.source_corners@self.matrix+self.translation
        result.update({'observation_binding':self._current_bound,'state8':self.state.tolist(),
            'identity':self.ident,'original_roi_uv':self.roi_uv.tolist(),
            'source_corners_px':self.source_corners.tolist(),
            'candidate_corners_px':corners.tolist(),'scale':float(np.linalg.norm(self.matrix[0])),
            'rotation_rad':float(np.arctan2(self.matrix[0,1],self.matrix[0,0])),
            'measured_path_m':self.path_m,'observations':self.observations,
            'selected_feature_ids':None if self.selected_ids is None else self.selected_ids.tolist(),
            'active_feature_ids':self.active.tolist(),'selection_path_m':self._selection_path,
            'qualification_observation':self.qualification_observation,'support':copy.deepcopy(self._metrics),
            'limitations':['The model-declared hand ROI can include other links or background.',
                'Distributed coherent motion is not semantic segmentation or metric clearance.',
                'Fixed orientation and finger state bound the rigid-region assumption.',
                'The full original rectangle is transported; no geometry or warning is deleted.']})
        return result
