"""Fixed conditional model-route exposure, not renewable visual clearance.

The V23 geometry, current tracks, domain and cumulative adverse evidence remain
mandatory. Raw causal robot-reference evidence replaces favorable box-gap
tranches. Actual acknowledged travel alone spends the finite episode budget.
"""
from __future__ import annotations
import copy
import numpy as np
from .visual_egress_evidence_v23 import EgressEvidenceV23, valid_egress_contract
from .visual_monitor_v4 import _orientation_distance


class RouteExposureV24(EgressEvidenceV23):
    confirmation_limit_m=.004

    def __init__(self):
        super().__init__()
        self.confirmation_candidate_id=None
        self.confirmation_start_travel_m=None
        self.last_reference_evidence=None
        self.last_exposure_mode='not_started'

    @staticmethod
    def _geometry_context(proof):
        return EgressEvidenceV23._geometry_context(proof)+(tuple(np.round(proof['complete_route_goal_xyz'],12)),)

    def _progress(self,current):
        result=super()._progress(current)
        # Keep cumulative adverse evidence and its immutable origins. Favorable
        # 2D motion is diagnostic; it does not grant physical travel credit.
        result['legacy_favorable_gap_supported']=result['supported']
        result['supported']=False
        result['reason']='egress_two_view_worsening' if result.get('worsening') else 'conditional_model_route_observation'
        return result

    def assess(self,proof,state8,delta_xyz,*,guard,monitor,inventory,translation_response=None):
        state,delta=np.asarray(state8,float),np.asarray(delta_xyz,float)
        def reject(reason,**detail):
            return dict(accepted=False,cap_m=0.,reason=reason,clearance_certified=False,**detail)
        if (not valid_egress_contract(proof,state,delta,guard.sweep_observation_epoch)
                or proof.get('nominal_core_separation_nondecreasing') is not True):
            return reject('route_exposure_current_geometry_contract_invalid')
        if self.committed_geometry_context is not None and self._geometry_context(proof)!=self.committed_geometry_context:
            return reject('route_exposure_original_geometry_or_goal_changed')
        current=self._snapshot(state,guard,monitor,inventory,proof['initial_halo_ids'])
        if current is None:return reject('route_exposure_current_two_view_support_unavailable')
        if self.baseline is not None and (_orientation_distance(self.baseline['state8'][3:6],state[3:6])>self.maximum_orientation_change_rad
                or np.max(abs(self.baseline['state8'][6:]-state[6:]))>.002):
            return reject('route_exposure_pose_domain_changed')
        if self.last_progress and self.last_progress.get('worsening'):
            return reject('route_exposure_observed_worsening')
        witness=getattr(guard,'reference_witness',None)
        evidence=None if witness is None else witness.raw_support_evidence(state,monitor)
        if (not isinstance(evidence,dict) or evidence.get('current_bound') is not True
                or evidence.get('reliable') is not True):
            return reject('route_exposure_current_raw_reference_unavailable')
        if evidence.get('contradicted'):
            return reject('route_exposure_raw_reference_contradicted',reference_evidence=copy.deepcopy(evidence))
        qualified=evidence.get('causal_supported') is True and evidence.get('evidence_kind')=='prospective'
        spent=max(self.spent_m,float(guard.observation_travel_m))
        remaining=self.total_limit_m-spent
        mode='conditional_model_route' if qualified else 'initial_reference_probe'
        candidate=evidence.get('selected_candidate_id')
        selection=getattr(guard,'reference_selection',None)
        explicit=(selection is not None and candidate==selection.get('candidate_id')
            and evidence.get('confirmation_required') is True
            and selection.get('track_identity')==id(guard._tool(monitor)))
        if not qualified:
            if self.confirmation_candidate_id is not None:
                if not explicit or candidate!=self.confirmation_candidate_id:
                    return reject('route_exposure_confirmation_cannot_be_renewed')
                remaining=min(remaining,self.confirmation_limit_m-(self.egress_travel_m-self.confirmation_start_travel_m))
                mode='single_explicit_reference_confirmation'
            elif explicit:
                remaining=min(remaining,self.confirmation_limit_m)
                mode='single_explicit_reference_confirmation'
            else:
                remaining=min(remaining,self.bootstrap_m-self.egress_travel_m)
        if remaining<=1e-9:
            return reject('route_exposure_reference_confirmation_or_total_budget_exhausted',reference_evidence=copy.deepcopy(evidence))
        norm=float(np.linalg.norm(delta));requested=min(norm,self.command_limit_m)
        if norm<=1e-12:return reject('route_exposure_requires_translation')
        conversion=None
        if translation_response is not None:
            conversion=translation_response.command_cap_for_measured_limit(delta/norm,remaining,requested,state)
            cap=min(requested,float(conversion['command_cap_m']))
        else:cap=min(requested,remaining,.002)
        if cap<self.minimum_command_m-1e-10:
            return reject('route_exposure_no_useful_command_within_remaining_budget',remaining_measured_m=remaining,
                reference_evidence=copy.deepcopy(evidence),reference_repair_review_required=not qualified)
        return {'accepted':True,'cap_m':cap,'reason':mode,'remaining_measured_m':remaining,
            'conversion':conversion,'reference_evidence':copy.deepcopy(evidence),'clearance_certified':False,
            'risk_authorization':'explicit_fixed_model_route_with_retained_hypotheses',
            'pending':{'before':current,'proof':copy.deepcopy(proof),'delta':delta.copy(),'command_cap_m':cap,
                'prior_shared_debt_m':float(guard.observation_travel_m),'exposure_mode':mode,
                'confirmation_candidate_id':candidate if mode=='single_explicit_reference_confirmation' else None,
                'reference_evidence':copy.deepcopy(evidence),'prior_egress_travel_m':self.egress_travel_m}}

    def observe(self,before_state8,action7,after_state8,*,pending,guard,monitor,inventory):
        previous=self.observations
        super().observe(before_state8,action7,after_state8,pending=pending,guard=guard,monitor=monitor,inventory=inventory)
        if self.observations==previous+1:
            self.last_exposure_mode=pending['exposure_mode']
            self.last_reference_evidence=copy.deepcopy(pending['reference_evidence'])
            if pending['exposure_mode']=='single_explicit_reference_confirmation' and self.confirmation_candidate_id is None:
                self.confirmation_candidate_id=pending['confirmation_candidate_id']
                self.confirmation_start_travel_m=pending['prior_egress_travel_m']
        return self.summary()

    def summary(self):
        result=super().summary()
        result.update(version='fixed-conditional-model-route-exposure-v24',
            risk_authorization='explicit_fixed_model_route_not_visual_clearance',
            positive_image_gap_required=False,physical_budget_is_renewable=False,
            original_full_goal_bound=self.committed_geometry_context is not None,
            last_exposure_mode=self.last_exposure_mode,
            reference_evidence=copy.deepcopy(self.last_reference_evidence),
            confirmation_candidate_id=self.confirmation_candidate_id,
            confirmation_limit_m=self.confirmation_limit_m,
            confirmation_used_m=0. if self.confirmation_start_travel_m is None else self.egress_travel_m-self.confirmation_start_travel_m)
        result.pop('observation_tranche_remaining_m',None)
        return result
