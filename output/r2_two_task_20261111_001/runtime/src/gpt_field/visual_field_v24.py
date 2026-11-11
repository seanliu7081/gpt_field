"""Fixed-goal conditional uncertainty egress with nominal-core separation."""
from __future__ import annotations
import numpy as np
from .visual_field_v23 import MetricFieldExecutorV23,_DirectedEgressValidity


class _CoreSeparatingValidity(_DirectedEgressValidity):
    def segment_valid(self,start,end):
        return super().segment_valid(start,end) and self.owner.core_separation_contract(start,end)['accepted']


class MetricFieldExecutorV24(MetricFieldExecutorV23):
    def __init__(self):
        super().__init__()
        self._committed_core_separation_state=None

    def core_separation_contract(self,start,end):
        geometry=self.egress_geometry
        result={'accepted':False,'reason':'egress_core_separation_unavailable','parts':[]}
        if geometry is None:return result
        start,end=np.asarray(start,float),np.asarray(end,float)
        for identity,part,lo,hi in geometry.cores:
            if (identity,part) not in geometry.initial_parts:continue
            origin=geometry.state[:3];closest=np.clip(origin,lo,hi);offset=origin-closest
            norm=float(np.linalg.norm(offset))
            if norm<=1e-10:
                result['reason']='egress_nominal_core_origin_not_separated';return result
            normal=offset/norm
            # The geometry and its support-plane normals remain anchored to the
            # original model hypothesis. A rejected initial command may however
            # be followed by an acknowledged pose hold with residual motion.
            # Before any egress action commits, compare against the actual
            # candidate start; the first exact acknowledgment freezes that
            # minimum for the entire physical episode, including later reseeds.
            baseline=(start if self._committed_core_separation_state is None
                else self._committed_core_separation_state[:3])
            initial=float(np.dot(baseline-closest,normal))
            current=float(np.dot(start-closest,normal));change=float(np.dot(end-start,normal))
            row={'id':identity,'tool_part':part,'normal_xyz':normal.tolist(),
                'original_support_plane_distance_m':norm,
                'minimum_committed_support_plane_distance_m':initial,
                'separation_baseline_committed':self._committed_core_separation_state is not None,
                'current_support_plane_distance_m':current,
                'proposed_distance_change_m':change}
            result['parts'].append(row)
            if current<initial-1e-10 or change< -1e-10:
                result.update(reason='egress_nominal_core_separation_decreases',blocked_part=[identity,part]);return result
        result.update(accepted=bool(result['parts']),reason='fixed_nominal_core_support_planes_nondecreasing')
        return result

    def egress_candidates(self,state8):
        return [row for row in super().egress_candidates(state8)
            if self.core_separation_contract(np.asarray(state8)[:3],np.asarray(state8)[:3]+row['goal_offset_m'])['accepted']]

    def egress_contract(self,state8,delta_xyz):
        result=super().egress_contract(state8,delta_xyz)
        if not self._egress_requested():return result
        state,delta=np.asarray(state8,float),np.asarray(delta_xyz,float)
        step=self.core_separation_contract(state[:3],state[:3]+delta)
        entire=self.core_separation_contract(state[:3],self.goal)
        result['nominal_core_separation_nondecreasing']=step['accepted'] and entire['accepted']
        result['nominal_core_separation']={'candidate':step,'complete_goal':entire}
        if result.get('accepted') and not result['nominal_core_separation_nondecreasing']:
            result.update(accepted=False,reason=step['reason'] if not step['accepted'] else entire['reason'])
        return result

    def _prepare_egress(self,state):
        reason=super()._prepare_egress(state)
        if reason is None:self.checked=_CoreSeparatingValidity(self)
        return reason

    def observe(self,before_state8,action,after_state8):
        first_ack=(self.pending is not None and self._egress_requested()
            and self._committed_core_separation_state is None)
        before=np.asarray(before_state8,float).copy()
        # Exact pending action/state validation happens before super returns.
        # Provisional previews and downstream veto holds cannot commit a floor.
        super().observe(before_state8,action,after_state8)
        if first_ack:self._committed_core_separation_state=before
        return self.summary()

    def summary(self):
        result=super().summary()
        result.update(version='fixed-route-conditional-uncertainty-field-v24',
            egress_nominal_core_separation='original_fixed_support_planes',
            egress_nominal_core_minimum_committed=self._committed_core_separation_state is not None,
            uncertainty_egress_is_clearance_proof=False)
        return result
