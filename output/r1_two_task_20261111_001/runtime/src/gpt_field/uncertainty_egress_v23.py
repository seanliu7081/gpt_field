"""Pure directed feasibility through an initially overlapping model halo.

All boxes remain present. Nominal robot/tool cores and unknown regions are
always forbidden. Only the originally overlapped non-target uncertainty parts
can be exited, with nonincreasing interior depth and no subsequent re-entry.
These are model-box statements, not certified physical free space.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
import numpy as np
from .visual_monitor_v4 import _orientation_distance
from .visual_scene_guard_v17 import segment_box_interval


class UncertaintyEgressRoute:
    MAX_EXTENT_M = .04

    def __init__(self, state8, regions, *, target_id, target_band_m=.03):
        self.state = np.asarray(state8,float).copy()
        self.regions = copy.deepcopy(regions)
        self.target_id = target_id
        self.target_band_m = max(.03,float(target_band_m))
        self.parts, self.cores, self.unknown = [], [], []
        self.initial_parts = set()
        self.invalid_reason = None
        canonical=[]
        ids=[r['id'] for r in self.regions]
        if len(ids)!=len(set(ids)) or target_id not in ids:
            self.invalid_reason='egress_incomplete_or_duplicate_geometry'
        for region in self.regions:
            kind=region['kind']; identity=region['id']
            canonical.append({k:region[k] for k in ('id','kind','tool_core_parts','full_halo_parts')})
            for p in region['tool_core_parts']:
                self.cores.append((identity,p['tool_part'],np.array(p['lower_xyz']),np.array(p['upper_xyz'])))
            for p in region['full_halo_parts']:
                key=(identity,p['tool_part']);lo=np.array(p['lower_xyz']);hi=np.array(p['upper_xyz'])
                item=(identity,p['tool_part'],lo,hi)
                if kind=='unknown_regions':self.unknown.append(item)
                else:self.parts.append(item)
                if self._inside(self.state[:3],lo,hi):
                    if kind=='unknown_regions' or identity==target_id:
                        self.invalid_reason='egress_initial_target_or_unknown_overlap'
                    else:self.initial_parts.add(key)
        self.geometry_sha256=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if any(self._inside(self.state[:3],lo,hi) for _,_,lo,hi in self.cores):
            self.invalid_reason='egress_initial_nominal_core_overlap'
        if not self.initial_parts and self.invalid_reason is None:
            self.invalid_reason='egress_no_initial_uncertainty_overlap'

    @staticmethod
    def _inside(point,lo,hi):
        return bool(np.all(point>=lo-1e-12) and np.all(point<=hi+1e-12))

    def check(self,start,end,*,require_progress=False):
        start,end=np.asarray(start,float),np.asarray(end,float)
        result={'accepted':False,'reason':self.invalid_reason,'geometry_sha256':self.geometry_sha256,
            'scene_ids':[r['id'] for r in self.regions], 'target_id':self.target_id,
            'initial_halo_ids':sorted({i for i,p in self.initial_parts}),
            'initial_halo_parts':[{'id':i,'tool_part':p} for i,p in sorted(self.initial_parts)],
            'start_xyz_m':start.tolist(),'end_xyz_m':end.tolist(),
            'nominal_core_clear':True,'unknown_clear':True,'no_new_halo_entry':True,
            'monotone_egress':True,'part_progress':[], 'clearance_certified':False,
            'full_uncertainty_sweep_disjoint':False,'conditional_initial_halo_exception':True}
        if self.invalid_reason:return result
        if not np.isfinite(np.r_[start,end]).all():
            result['reason']='egress_nonfinite_segment';return result
        if max(np.linalg.norm(start-self.state[:3]),np.linalg.norm(end-self.state[:3]))>self.MAX_EXTENT_M+1e-10:
            result['reason']='egress_context_extent_exceeded';return result
        for identity,part,lo,hi in self.cores:
            if segment_box_interval(start,end,lo,hi) is not None:
                result.update(reason='egress_nominal_core_intersection',nominal_core_clear=False,blocked_part=[identity,part]);return result
        for identity,part,lo,hi in self.unknown:
            if segment_box_interval(start,end,lo,hi) is not None:
                result.update(reason='egress_unknown_intersection',unknown_clear=False,blocked_part=[identity,part]);return result
        changed=False
        for identity,part,lo,hi in self.parts:
            hit=segment_box_interval(start,end,lo,hi)
            began=self._inside(start,lo,hi)
            if (identity,part) not in self.initial_parts or not began:
                if hit is not None:
                    result.update(reason='egress_new_halo_entry_or_reentry',no_new_halo_entry=False,blocked_part=[identity,part]);return result
                continue
            distances=np.r_[start-lo,hi-start]
            end_distances=np.r_[end-lo,hi-end]
            depth=float(np.min(distances));end_depth=float(np.min(end_distances))
            slopes=np.r_[end-start,start-end]
            nearest=distances<=depth+1e-10
            # Interior depth is the minimum of six affine functions, hence
            # concave along the segment. Its right derivative at the start
            # bounds every later derivative. A nonpositive derivative proves
            # that depth never increases, without sampling the trajectory.
            derivative=float(np.min(slopes[nearest]))
            if derivative>1e-10:
                result.update(reason='egress_initial_halo_depth_increases',monotone_egress=False,blocked_part=[identity,part]);return result
            changed|=end_depth<depth-1e-9
            result['part_progress'].append({'id':identity,'tool_part':part,
                'start_interior_depth_m':depth,'end_signed_interior_depth_m':end_depth,
                'initial_depth_derivative_m':derivative,'exit_fraction':None if hit is None else hit[1]})
        target=[(lo,hi) for identity,_,lo,hi in self.cores if identity==self.target_id]
        low,high=np.minimum(start,end),np.maximum(start,end)
        distance=min(float(np.linalg.norm(np.maximum(np.maximum(lo-high,low-hi),0))) for lo,hi in target)
        result.update(target_core_distance_lower_bound_m=distance,target_band_m=self.target_band_m)
        if distance<=self.target_band_m+1e-10:
            result['reason']='egress_target_contact_band';return result
        if require_progress and not changed:
            result['reason']='egress_goal_has_no_halo_progress';return result
        result['exit_goal_full_halo_disjoint']=not any(self._inside(end,lo,hi)for _,_,lo,hi in self.parts+self.unknown)
        result.update(accepted=True,reason='conditional_monotone_initial_uncertainty_egress')
        return result

    def candidates(self,max_count=12):
        """Generic face-derived endpoints; no object/task identities are used."""
        active=[x for x in self.parts if (x[0],x[1]) in self.initial_parts]
        if self.invalid_reason or len(active)>8:return []
        options=[]
        # Enumerate at most the nearest three exit faces per starting part.
        for _,_,lo,hi in active:
            d=np.r_[self.state[:3]-lo,hi-self.state[:3]]
            options.append(sorted(range(6),key=lambda k:d[k])[:3])
        accepted=[];seen=set()
        for faces in itertools.product(*options):
            lower=np.full(3,-np.inf);upper=np.full(3,np.inf)
            for (_,_,lo,hi),face in zip(active,faces):
                # The wire's minimum local-arrival tolerance is 3mm. Leave
                # another millimetre so arriving anywhere in that ball cannot
                # silently label an inside-halo position a completed exit.
                if face<3:upper[face]=min(upper[face],lo[face]-self.state[face]-.004)
                else:lower[face-3]=max(lower[face-3],hi[face-3]-self.state[face-3]+.004)
            if np.any(lower>upper):continue
            delta=np.clip(np.zeros(3),lower,upper);norm=float(np.linalg.norm(delta))
            if norm<.004:delta*=.004/max(norm,1e-12);norm=float(np.linalg.norm(delta))
            # A partial egress is allowed, but must genuinely reduce at least
            # one initial depth and never worsen any other initial depth.
            if norm>self.MAX_EXTENT_M:delta*=self.MAX_EXTENT_M/norm
            key=tuple(np.round(delta,10))
            if key in seen:continue
            seen.add(key);proof=self.check(self.state[:3],self.state[:3]+delta,require_progress=True)
            if proof['accepted']:
                clearance=min(float(np.linalg.norm(np.maximum(np.maximum(lo-self.state[:3]-delta,self.state[:3]+delta-hi),0)))
                              for _,_,lo,hi in self.parts+self.unknown)
                accepted.append({'goal_offset_m':delta.tolist(),'goal_tolerance_m':.003,
                    'route_length_m':float(np.linalg.norm(delta)),
                    'full_exit':proof['exit_goal_full_halo_disjoint'] and clearance>.003+1e-10,
                    'goal_full_halo_clearance_m':clearance,
                    'source':'initial_uncertainty_box_faces','geometry_sha256':self.geometry_sha256,
                    'requires_explicit_model_choice':True,'clearance_certified':False})
        return sorted(accepted,key=lambda x:(not x['full_exit'],x['route_length_m']))[:max_count]
