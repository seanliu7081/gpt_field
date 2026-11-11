"""Candidate-bound model-margin evidence for normal empty-tool Field transit.

These distances describe retained model geometry and known robot envelopes.
They never certify real scene clearance and do not themselves authorize motion,
clear image warnings, or change physical exploration/travel accounting.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import numpy as np

from .visual_field_v25 import MetricFieldExecutorV25
from .visual_scene_guard_v17 import SceneCoverageGuardV17, segment_box_interval


def _quadratic(start, delta, low, high, middle):
    point = start+middle*delta
    offset = np.where(point < low, low-start, np.where(point > high, start-high, 0.))
    slope = np.where(point < low, -delta, np.where(point > high, delta, 0.))
    return np.array([slope@slope, 2.*offset@slope, offset@offset], float)


def _roots(coefficients, low, high):
    a, b, c = coefficients
    if abs(a) < 1e-18:
        values = [] if abs(b) < 1e-18 else [-c/b]
    else:
        discriminant = b*b-4.*a*c
        values = [] if discriminant < -1e-20 else [(-b-math.sqrt(max(0.,discriminant)))/(2.*a),
                                                    (-b+math.sqrt(max(0.,discriminant)))/(2.*a)]
    return [float(t) for t in values if low+1e-12 < t < high-1e-12]


def full_margin_union_segment_proof(start, end, boxes):
    """Exact piecewise-quadratic distance to a finite AABB union along a segment.

    Pairwise polynomial crossings split the lower envelope. This detects a
    mid-segment approach even when both endpoints are equally far, and does not
    require every nonclosest robot part to move away. Numeric tolerances apply
    only to floating-point comparisons, not to obstacle inflation.
    """
    start, end = np.asarray(start,float), np.asarray(end,float)
    if start.shape!=(3,) or end.shape!=(3,) or not np.isfinite(np.r_[start,end]).all() or not boxes:
        raise ValueError('Finite start/end and at least one configuration box required')
    bounds=[]
    for row in boxes:
        low,high=np.asarray(row['lower_xyz'],float),np.asarray(row['upper_xyz'],float)
        if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite(np.r_[low,high]).all() or np.any(low>high):
            raise ValueError('Invalid configuration bounds')
        bounds.append((low,high))
    delta=end-start
    breaks={0.,1.}
    for low,high in bounds:
        for axis in range(3):
            if abs(delta[axis])>1e-15:
                breaks.update(float(t) for t in ((low[axis]-start[axis])/delta[axis],(high[axis]-start[axis])/delta[axis]) if 0.<t<1.)
    breaks=sorted(breaks)
    def distances(point):
        return np.array([np.linalg.norm(np.maximum(np.maximum(low-point,point-high),0.)) for low,high in bounds])
    start_gaps,end_gaps=distances(start),distances(end)
    minimum=float(min(start_gaps)); minimum_t=0.; minimum_part=int(np.argmin(start_gaps))
    nondecreasing=True; min_derivative=float('inf'); falling_intervals=[]
    for left,right in zip(breaks[:-1],breaks[1:]):
        coefficients=np.array([_quadratic(start,delta,low,high,(left+right)*.5) for low,high in bounds])
        divisions={left,right}
        for i in range(len(bounds)):
            for j in range(i+1,len(bounds)):
                divisions.update(_roots(coefficients[i]-coefficients[j],left,right))
        divisions=sorted(divisions)
        for a,b in zip(divisions[:-1],divisions[1:]):
            mid=(a+b)*.5
            active=int(np.argmin(coefficients@np.array([mid*mid,mid,1.])))
            quadratic,linear,constant=coefficients[active]
            derivative=min(2.*quadratic*a+linear,2.*quadratic*b+linear)
            min_derivative=min(min_derivative,float(derivative))
            if derivative < -1e-12:
                nondecreasing=False
                falling_intervals.append({'parameter_interval':[float(a),float(b)],'tool_part':boxes[active]['tool_part']})
            candidates=[a,b]
            if quadratic>1e-18:candidates.append(float(np.clip(-linear/(2.*quadratic),a,b)))
            for t in candidates:
                distance=math.sqrt(max(0.,quadratic*t*t+linear*t+constant))
                if distance<minimum:
                    minimum,minimum_t,minimum_part=distance,float(t),active
    intersections=[row['tool_part'] for row,(low,high) in zip(boxes,bounds) if segment_box_interval(start,end,low,high) is not None]
    return {'start_gap_m':float(start_gaps.min()),'end_gap_m':float(end_gaps.min()),
        'minimum_segment_gap_m':float(minimum),'minimum_segment_parameter':minimum_t,
        'minimum_segment_part':boxes[minimum_part]['tool_part'],
        'end_minus_start_gap_m':float(end_gaps.min()-start_gaps.min()),
        'gap_nondecreasing':bool(nondecreasing),'minimum_squared_gap_derivative_m2':0. if not np.isfinite(min_derivative) else min_derivative,
        'decreasing_parameter_intervals':falling_intervals,
        'full_uncertainty_sweep_disjoint':not intersections,'intersected_parts':intersections,
        'configuration_part_count':len(boxes),
        'method':'piecewise_quadratic_lower_envelope_of_full_margin_configuration_box_distances',
        'numeric_squared_derivative_tolerance_m2':1e-12,
        'is_probability':False,'clearance_certified':False}


class MetricFieldExecutorV27(MetricFieldExecutorV25):
    """Construct bounded candidates before checking their geometric evidence."""
    def __init__(self):
        super().__init__()
        self.transit_reconciliation_geometry=None

    def configure(self,*args,**kwargs):
        self.transit_reconciliation_geometry=None
        return super().configure(*args,**kwargs)

    def select(self,state8,joint_position,*,command_cap_m=None,measured_cap_m=None):
        guard=self.translation_guard
        scoped=(self._route()=='normal' and guard is not None
            and bool(getattr(guard,'worsening',()))
            and (getattr(guard,'context',None) or {}).get('translation_mode')=='transit'
            and hasattr(guard,'reconciliation_command_limit_m'))
        effective=command_cap_m
        if scoped:
            # The guard's limit defines the actual candidate, not an unused
            # suffix that must first be proven admissible. This only bounds
            # planning/tracking: all exact-segment geometry, response, role,
            # and physical-budget checks still run in inherited selection.
            limit=float(guard.reconciliation_command_limit_m)
            effective=limit if effective is None else min(effective,limit)
        result=super().select(state8,joint_position,command_cap_m=effective,
            measured_cap_m=measured_cap_m)
        if scoped:
            result.setdefault('diagnostics',{})['transit_reconciliation_candidate_limit']={
                'requested_command_cap_m':command_cap_m,
                'effective_command_cap_m':effective,
                'applied_before_candidate_geometry':True,
                'admission_granted_by_limit':False,
                'physical_budget_replenished':False}
        return result

    def _transit_reconciliation_contract(self,state,start,end,*,base_contract=None):
        try:
            state,start,end=np.asarray(state,float),np.asarray(start,float),np.asarray(end,float)
        except (TypeError,ValueError):
            return {'version':'v27-current-normal-transit-model-margin','eligible':False,'reason':'invalid_state_or_segment','object_proofs':{},'clearance_certified':False}
        if state.shape!=(8,) or start.shape!=(3,) or end.shape!=(3,) or not np.isfinite(np.r_[state,start,end]).all():
            return {'version':'v27-current-normal-transit-model-margin','eligible':False,'reason':'invalid_state_or_segment','object_proofs':{},'clearance_certified':False}
        try:
            base=copy.deepcopy(base_contract) if base_contract is not None else self._retreat_geometry_contract(state,start,end)
        except (TypeError,ValueError,KeyError,OverflowError):
            return {'version':'v27-current-normal-transit-model-margin','eligible':False,'reason':'invalid_scene_or_payload_geometry','object_proofs':{},'clearance_certified':False}
        proof=copy.deepcopy(base)
        proof.update(version='v27-current-normal-transit-model-margin',eligible=False,
            reason='normal_transit_binding_unavailable',route_family=self._route(),
            fixed_origin_xyz_m=None if self.origin is None else np.asarray(self.origin).tolist(),
            fixed_goal_xyz_m=None if self.goal is None else np.asarray(self.goal).tolist(),
            object_proofs={},hard_segment_clear=False,physical_budget_replenished=False,
            warning_erased=False,geometry_is_hypothesis=True,clearance_certified=False)
        binding=self.recovery_route_binding or {}
        guard=self.translation_guard
        if (state.shape!=(8,) or start.shape!=(3,) or end.shape!=(3,)
                or not np.isfinite(np.r_[state,start,end]).all()
                or guard is None or self.intent is None or self.checked is None
                or self._route()!='normal'
                or not self._same(state,self.expected_state,(8,))
                or not self._same(state,getattr(guard,'state',None),(8,))
                or not self._same(start,state[:3],(3,))
                or not self._same(binding.get('origin_xyz'),self.origin,(3,))
                or not self._same(binding.get('goal_xyz'),self.goal,(3,))
                or not self._same(base.get('state8'),state,(8,))
                or not self._same(base.get('start_xyz_m'),start,(3,))
                or not self._same(base.get('end_xyz_m'),end,(3,))
                or not isinstance(base.get('observation_epoch'),int)
                or isinstance(base.get('observation_epoch'),bool)
                or base.get('observation_epoch')!=getattr(guard,'sweep_observation_epoch',None)):
            return proof
        material=self._transit_materialization or {}
        objects=material.get('objects') or []
        proof['full_scene_ids']=[row.get('id') for row in objects]
        proof['geometry_fingerprint']={'objects':[{'id':row.get('id'),'world_geometry':copy.deepcopy(row.get('world_geometry'))} for row in objects],
            'intent_uncertainty_m':self.intent.get('uncertainty_m'),'tool_radius_m':self.intent.get('tool_radius_m'),
            'unknown_regions':copy.deepcopy(self.intent.get('unknown_regions'))}
        actual_digest=hashlib.sha256(json.dumps(proof['geometry_fingerprint'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if actual_digest!=base.get('geometry_sha256'):
            proof['reason']='scene_geometry_fingerprint_changed';return proof
        proof['hard_segment_clear']=bool(self.checked.segment_valid(start,end))
        proof['unknown_region_count']=len(self.intent.get('unknown_regions') or [])
        proof['orientation_margin_rad']=.04
        proof['full_finger_range']=True
        try:
            payload=SceneCoverageGuardV17._carried_bounds(state,self,.04)
            for record in objects:
                identity=record.get('id')
                if not isinstance(identity,str) or not identity or identity in proof['object_proofs']:
                    proof['reason']='duplicate_or_invalid_scene_identity';return proof
                boxes=SceneCoverageGuardV17._sweep_boxes(record,state,{'field_intent':self.intent},.04,payload)
                if boxes is None:
                    proof['reason']='scene_model_geometry_unavailable';return proof
                item=full_margin_union_segment_proof(start,end,boxes)
                item['world_geometry']=copy.deepcopy(record['world_geometry'])
                item['object_geometry_sha256']=hashlib.sha256(json.dumps(item['world_geometry'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
                item['configuration_geometry_sha256']=hashlib.sha256(json.dumps([{'tool_part':b['tool_part'],'lower_xyz':np.asarray(b['lower_xyz']).tolist(),'upper_xyz':np.asarray(b['upper_xyz']).tolist()} for b in boxes],sort_keys=True,separators=(',',':')).encode()).hexdigest()
                proof['object_proofs'][identity]=item
        except (TypeError,ValueError,KeyError,OverflowError):
            proof['reason']='scene_model_geometry_invalid';return proof
        ids=base.get('scene_ids') or []
        if not objects or len(ids)!=len(set(ids)) or set(proof['object_proofs'])!=set(ids):
            proof['reason']='incomplete_scene_model_geometry';return proof
        required=(base.get('full_scene_sweep_accepted') is True,
            proof['hard_segment_clear'],base.get('outside_target_contact_band') is True,
            base.get('empty_tool') is True,base.get('target_contact')=='none',
            base.get('contact_or_visual_goal_dependency') is False,
            base.get('remaining_actions',0)>0,base.get('remaining_measured_travel_m',0)>0,
            all(row['full_uncertainty_sweep_disjoint'] for row in proof['object_proofs'].values()),
            float(np.linalg.norm(end-start))>1e-12)
        proof.update(eligible=bool(all(required)),reason='current_normal_transit_model_geometry' if all(required) else 'normal_transit_role_scene_or_budget_ineligible')
        return proof

    def nominal_regions_for_segment(self,start,end):
        records=super().nominal_regions_for_segment(start,end)
        guard=self.translation_guard
        if guard is None or guard.state is None:
            return records
        base=records[0].get('directional_retreat_contract') if records else None
        proof=self._transit_reconciliation_contract(np.asarray(guard.state,float),np.asarray(start,float),np.asarray(end,float),base_contract=base)
        for row in records:row['transit_reconciliation_contract']=copy.deepcopy(proof)
        self.transit_reconciliation_geometry=copy.deepcopy(proof)
        return records

    def summary(self):
        result=super().summary()
        result.update(version='candidate-bound-model-margin-field-v27',
            transit_reconciliation_geometry=copy.deepcopy(getattr(self,'transit_reconciliation_geometry',None)))
        return result
