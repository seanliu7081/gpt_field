"""Paid-history reference repair and immutable-window RGB motion evidence.

No semantic robot identity, environment calibration, clearance or motion budget
is inferred here. Candidate selection remains explicit. Prospective evidence is
computed only from observations acknowledged after the current reference seed.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_reference_candidates_v18 import CausalReferenceCandidatesV18
from .visual_field import _gray
from .visual_monitor_v4 import _orientation_distance
from .visual_monitor_v5 import _ncc, _sample


class CausalReferenceCandidatesV24(CausalReferenceCandidatesV18):
    version = 'handoff-immutable-support-reference-v24'
    handoff_reasons = frozenset({
        'uncertainty_egress_no_useful_command_within_remaining_budget',
        'uncertainty_egress_fresh_progress_required',
        'egress_observation_inconclusive',
        'translation_measured_allowance_below_useful_command_floor',
        'translation_observation_budget_exhausted',
        'field_command_below_useful_motion_floor',
        'route_exposure_no_useful_command_within_remaining_budget',
        'route_exposure_reference_confirmation_or_total_budget_exhausted',
        'route_exposure_raw_reference_contradicted',
        'route_exposure_confirmation_cannot_be_renewed'})

    def __init__(self):
        super().__init__()
        self._witness = None
        self._support_cache = None
        self._pending_selection = None
        self._selected_candidate_id = None
        self._handoff_attempt_path = -1.
        self._last_handoff = None

    @staticmethod
    def _live(monitor):
        tracks = getattr(monitor, 'tracks', {})
        return tracks.get('moving'), tracks.get('target')

    def configure(self, observation, monitor, motion_context):
        old = None if self._witness is None else self._witness['track']
        super().configure(observation, monitor, motion_context)
        track, anchor = self._live(monitor)
        if track is not old:
            selected = self._pending_selection
            self._selected_candidate_id = None
            if (selected is not None and selected['observation_binding'] == self._binding(self.current.rgb,self.current.state)
                    and track is not None and np.allclose(track.uv,selected['current_point_uv'],atol=1e-9,rtol=0)):
                self._selected_candidate_id = selected['candidate_id']
            self._pending_selection = None
            self._witness = None
            self._handoff_attempt_path = -1.
            if track is not None and anchor is not None:
                self._witness = {'track':track, 'anchor':anchor, 'rgb':self.current.rgb.copy(),
                    'gray':_gray(self.current.rgb), 'state':self.current.state.copy(),
                    'path':self.path_m, 'sequence':self.sequence,
                    'tool_points':np.asarray(track.points,float).copy(),
                    'anchor_points':np.asarray(anchor.points,float).copy(),
                    'tool_transform':(track.matrix.copy(),track.translation.copy()),
                    'anchor_transform':(anchor.matrix.copy(),anchor.translation.copy())}
        self._support_cache = None
        return self.summary()

    def observe(self, before, after, monitor):
        result = super().observe(before, after, monitor)
        self._support_cache = None
        # Compute lazily when control/feedback asks, once per observation.
        return self.summary()

    def resolve_candidate(self, candidate_id, observation):
        resolved = super().resolve_candidate(candidate_id, observation)
        self._pending_selection = copy.deepcopy(resolved)
        return resolved

    def _build_candidates(self):
        # Keep the inherited bounded extraction/FB/identity tests unchanged.
        # Reserve displayed hypotheses in both the declared tool envelope and
        # outside it, instead of discarding all distal candidates by global rank.
        maximum = self.maximum_candidates
        self.maximum_candidates = self.maximum_candidate_trials
        try:
            super()._build_candidates()
        finally:
            self.maximum_candidates = maximum
        pool = list(self._candidates)
        if len(pool) <= maximum or self._declared_roi is None:
            return
        roi = self._declared_roi*np.tile(pool[0]['track'].scale,2)
        inside = [c for c in pool if np.all(c['track'].position >= roi[:2]) and np.all(c['track'].position <= roi[2:])]
        outside = [c for c in pool if not any(c is x for x in inside)]
        chosen = inside[:2]+outside[:2]
        chosen += [c for c in pool if not any(c is x for x in chosen)][:maximum-len(chosen)]
        # Preserve the quality ordering within the balanced displayed set.
        self._candidates = [c for c in pool if any(c is x for x in chosen)]
        for index,candidate in enumerate(self._candidates,1):
            candidate['id'] = f'motion-{self._board_generation}-{index}'
        self._image_cache = None

    def handoff(self, observation, monitor, reason):
        """Expose already-paid motion at a real inconclusive control handoff."""
        if reason not in self.handoff_reasons or not self._matches(observation):
            return self.summary()
        track,anchor = self._live(monitor)
        state=np.asarray(getattr(monitor,'state',[]),float)
        if (self.status!='observing' or self._moving_kind!='tool'
                or track is not self._reference_track or anchor is None or not anchor.reliable
                or state.shape!=(8,) or not np.array_equal(state,self.current.state)
                or self._reference_origin_state is None
                or _orientation_distance(state[3:6],self._reference_origin_state[3:6])>.035
                or np.max(abs(state[6:]-self._reference_origin_state[6:]))>.002
                or len(self._history())<4 or self.path_m-self._reference_origin_path<.001):
            return self.summary()
        # A handoff changes only presentation. It cannot restore either a
        # measured exploration allowance or any estimator's held-out evidence.
        if not self._candidates and self.path_m-self._handoff_attempt_path>=.001:
            self._handoff_attempt_path=self.path_m
            self._build_candidates()
        if self._candidates and self._review_id is None:
            self._review_id=f'reference-{self.reference_generation}-handoff-{self.sequence}'
        self._last_handoff={'reason':reason,'sequence':self.sequence,'physical_debt_modified':False}
        return self.summary()

    @staticmethod
    def _match_around(template, destination, center, search=7):
        if template is None or template.std()<4:
            return None
        x,y=np.rint(center).astype(int)
        left,right=max(3,x-search),min(destination.shape[1]-5,x+search)
        top,bottom=max(3,y-search),min(destination.shape[0]-5,y+search)
        if right<=left or bottom<=top:return None
        patches=np.lib.stride_tricks.sliding_window_view(destination[top-3:bottom+4,left-3:right+4],(7,7))
        normalized=template-template.mean()
        energy=np.maximum(np.sum(patches*patches,axis=(-1,-2))-patches.mean(axis=(-1,-2))**2*49,0.)
        denominator=np.sqrt(energy)*np.linalg.norm(normalized)
        scores=np.divide(np.einsum('ijkl,kl->ij',patches,normalized),denominator,
            out=np.full(energy.shape,-1.),where=denominator>1e-8)
        row,col=np.unravel_index(np.argmax(scores),scores.shape);score=float(scores[row,col])
        second=scores.copy();second[max(0,row-2):row+3,max(0,col-2):col+3]=-1.
        if score<.75 or score-second.max()<.025:return None
        refined=np.array([left+col,top+row],float)
        for axis,coordinate in enumerate((col,row)):
            line=scores[row,:] if axis==0 else scores[:,col]
            if 0<coordinate<len(line)-1:
                a,b,c=line[coordinate-1:coordinate+2];curvature=a-2*b+c
                if curvature < -1e-8:refined[axis]+=np.clip(.5*(a-c)/curvature,-.5,.5)
        if np.max(abs(refined-center))>=search-1:return None
        return refined,score

    @classmethod
    def _immutable_motion(cls, track, source, current, points, source_transform=None):
        """Baseline-to-current registration minus baseline self-match bias.

        Parabolic NCC peak interpolation is not zero-preserving even when the
        local raster is unchanged. Its deterministic self bias is not motion.
        No sequential tracker displacement enters this independent estimate.
        """
        flows, gains = [], []
        matrix,shift=(np.eye(2),np.zeros(2)) if source_transform is None else source_transform
        predicted=(points-shift)@np.linalg.inv(matrix)@track.matrix+track.translation
        for point,search_center in zip(points,predicted):
            template=_sample(source,point+track.offsets)
            baseline=cls._match_around(template,source,point)
            # The live transform supplies only a bounded search center. Motion
            # still has to match the original raster, FB and photometric tests.
            forward=cls._match_around(template,current,search_center)
            if baseline is None or forward is None:
                continue
            backward=cls._match_around(_sample(current,forward[0]+track.offsets),source,point)
            if backward is None or np.linalg.norm(backward[0]-point)>.9:
                continue
            delta=forward[0]-baseline[0]
            original=_sample(source,point+track.offsets)
            unchanged=_sample(current,point+track.offsets)
            translated=_sample(current,point+delta+track.offsets)
            if any(x is None for x in (original,unchanged,translated)):
                continue
            flows.append(delta)
            gains.append(_ncc(original,translated)-_ncc(original,unchanged))
        if len(flows)<3:
            return None
        flows=np.asarray(flows); center=np.median(flows,axis=0)
        residual=np.linalg.norm(flows-center,axis=1)
        bound=max(.35,3*1.4826*float(np.median(residual)))
        keep=residual<=bound
        if int(keep.sum())<max(3,int(np.ceil(len(points)*.6))):
            return None
        center=np.median(flows[keep],axis=0)
        precision=max(.08,1.4826*float(np.median(np.linalg.norm(flows[keep]-center,axis=1)))/np.sqrt(keep.sum()))
        return {'delta':center,'point_precision_px':precision,'inliers':int(keep.sum()),
            'median_photometric_motion_gain':float(np.median(np.asarray(gains)[keep]))}

    def raw_support_evidence(self, state8, monitor):
        state=np.asarray(state8,float);track,anchor=self._live(monitor);w=self._witness
        result={'current_bound':False,'reliable':False,'causal_supported':False,'contradicted':False,
            'confirmation_required':True,'selected_candidate_id':self._selected_candidate_id,
            'evidence_kind':'unavailable','reason':'reference_support_unavailable',
            'semantic_identity_certified':False,'clearance_certified':False,'control_debt_modified':False}
        if (self.current is None or w is None or state.shape!=(8,) or not np.isfinite(state).all()
                or not np.array_equal(state,self.current.state)
                or np.asarray(getattr(monitor,'state',[])).shape!=(8,)
                or not np.array_equal(monitor.state,state) or track is not w['track'] or anchor is not w['anchor']):
            return result
        gray=_gray(self.current.rgb)
        if (not np.array_equal(getattr(track,'previous_image',None),gray)
                or not np.array_equal(getattr(anchor,'previous_image',None),gray)):
            result['reason']='reference_rgb_endpoint_mismatch'
            return result
        result.update(current_bound=True,state8=state.tolist(),reference_generation=self.reference_generation,
            reference_identity=id(track),observation_binding=self._binding(self.current.rgb,state),
            current_sequence=self.sequence,reference_seed_sequence=w['sequence'])
        if (self.status!='observing' or self._moving_kind!='tool'
                or not track.reliable or not anchor.reliable
                or _orientation_distance(state[3:6],w['state'][3:6])>.035
                or np.max(abs(state[6:]-w['state'][6:]))>.002):
            result['reason']='reference_support_domain_or_track_unavailable'
            return result
        if self._support_cache is not None:
            return copy.deepcopy(self._support_cache)
        used=self.path_m-w['path']; count=self.sequence-w['sequence']
        mask=np.zeros(self.current.rgb.shape[:2],bool)
        for x,y in np.rint(w['tool_points']).astype(int):
            if 3<=x<mask.shape[1]-3 and 3<=y<mask.shape[0]-3:mask[y-3:y+4,x-3:x+4]=True
        diff=abs(self.current.rgb[mask].astype(np.int16)-w['rgb'][mask].astype(np.int16))
        changed=float((diff.max(1)>6).mean()) if len(diff) else 0.
        result.update(reliable=True,measured_path_m=used,fresh_observation_count=count,
            raw_support_changed_fraction=changed,raw_support_max_rgb_change=int(diff.max()) if len(diff) else 0,
            evidence_kind='prospective' if count else 'retrospective_selection_only' if self._selected_candidate_id else 'unobserved_seed')
        if not len(diff) or not diff.any():
            result['reason']='immutable_raw_support_unchanged'
        elif used<.001 or count<3:
            result['reason']='fresh_motion_support_insufficient'
        else:
            current=_gray(self.current.rgb)
            tool=self._immutable_motion(track,w['gray'],current,w['tool_points'],w['tool_transform'])
            target=self._immutable_motion(anchor,w['gray'],current,w['anchor_points'],w['anchor_transform'])
            if tool is None or target is None:
                result['reason']='immutable_registration_unreliable'
            else:
                relative=tool['delta']-target['delta']
                threshold=max(.35,3*np.hypot(tool['point_precision_px'],target['point_precision_px']))
                supported=(changed>.05 and np.linalg.norm(relative)>threshold
                    and tool['median_photometric_motion_gain']>1e-4)
                result.update(causal_supported=bool(supported),confirmation_required=not bool(supported),
                    bias_corrected_tool_motion_px=tool['delta'].tolist(),
                    bias_corrected_anchor_motion_px=target['delta'].tolist(),
                    bias_corrected_relative_motion_px=relative.tolist(),motion_threshold_px=float(threshold),
                    tool_inliers=tool['inliers'],anchor_inliers=target['inliers'],
                    photometric_motion_gain=tool['median_photometric_motion_gain'],
                    reason='fresh_immutable_relative_motion_supported' if supported else 'immutable_relative_motion_inconclusive')
        self._support_cache=copy.deepcopy(result)
        return result

    def summary(self):
        result=super().summary()
        result['last_handoff']=copy.deepcopy(getattr(self,'_last_handoff',None))
        result['display_quota']='reserve_up_to_two_inside_and_two_outside_declared_whole_tool_region'
        result['raw_support_evidence']=copy.deepcopy(getattr(self,'_support_cache',None))
        return result
