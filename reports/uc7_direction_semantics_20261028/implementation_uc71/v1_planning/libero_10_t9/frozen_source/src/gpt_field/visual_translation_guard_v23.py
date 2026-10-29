"""Contextual finite uncertainty egress; ordinary V21 admission is unchanged."""
from __future__ import annotations

import copy
import numpy as np

from .visual_translation_guard_v21 import ProjectedTranslationGuardV21
from .visual_egress_evidence_v23 import EgressEvidenceV23, valid_egress_contract


class ProjectedTranslationGuardV23(ProjectedTranslationGuardV21):
    version='bounded-observed-uncertainty-egress-v23'
    # These refusals concern the legacy unvalidated experiment allowance, not
    # contradictory RGB, lost identity, robot/scene domain or missing geometry.
    contextual_limit_reasons=frozenset({
        'translation_observation_budget_exhausted',
        'translation_recovery_direction_exhausted',
        'translation_recovery_experiment_limit',
        'translation_recovery_direction_not_distinct',
        'translation_direction_resume_requires_nominal_separation',
        'translation_new_experiment_requires_nominal_separation',
        'translation_recovery_direction_change_requires_review',
        'translation_measured_allowance_below_useful_command_floor'})

    def __init__(self):
        super().__init__()
        self.egress_evidence=EgressEvidenceV23()
        self.egress_inventory=None
        self.egress_monitor=None
        self._egress_pending=None
        self.last_egress_admission=None
        self._last_executed_was_egress=False

    def bind_egress_runtime(self, inventory, monitor):
        self.egress_inventory,self.egress_monitor=inventory,monitor

    def assess(self, delta_xyz, **kwargs):
        explicit=kwargs.pop('egress_contract',None)
        records=kwargs.get('nominal_regions') or []
        contracts=[r.get('uncertainty_egress_contract') for r in records]
        proof=explicit if explicit is not None else (contracts[0] if contracts else None)
        self._egress_pending=None
        base=super().assess(delta_xyz,**kwargs)
        self.last_egress_admission=None
        if proof is None or (explicit is None and isinstance(proof,dict)
                and proof.get('reason')=='egress_not_explicitly_selected'):
            return base
        command=np.asarray(delta_xyz,float)
        def refuse(reason):
            self._motion_classification_pending=self._experiment_pending=self._retreat_pending=None
            self.last_egress_admission={'accepted':False,'reason':reason,'clearance_certified':False}
            return {'accepted':False,'cap_m':0.,'need_refresh':True,'reason':reason,
                'diagnostics':self.summary()}
        if (not isinstance(proof,dict) or not contracts or any(c != proof for c in contracts)
                or set(proof.get('scene_ids',[])) != {r.get('id') for r in records if r.get('kind')!='unknown_regions'}
                or not valid_egress_contract(proof,self.state,command,self.sweep_observation_epoch)
                or proof['target_id'] != self.target_id):
            return refuse('uncertainty_egress_current_contract_invalid')
        if ((self.context or {}).get('translation_mode')!='separate'
                or (self.context or {}).get('purpose')!='reference_repair'
                or (self.context or {}).get('intended_contact_id') is not None
                or kwargs.get('nominal_clear') is not True):
            return refuse('uncertainty_egress_explicit_context_unavailable')
        monitor=kwargs.get('monitor') or self.egress_monitor
        if (monitor is None or monitor is not self.egress_monitor
                or (monitor.refs.get('external') or {}).get('moving_kind')!='tool'
                or not monitor._external_reliable()
                or monitor.cumulative_response_evidence().get('contradicted')
                or monitor.status in ('visual_response_contradiction','target_anchor_shifted')):
            return refuse('uncertainty_egress_reference_unavailable_or_contradicted')
        if not base['accepted'] and base['reason'] not in self.contextual_limit_reasons:
            return base
        evidence=self.egress_evidence.assess(proof,self.state,command,guard=self,monitor=monitor,
            inventory=self.egress_inventory,translation_response=kwargs.get('translation_response'))
        self.last_egress_admission={k:copy.deepcopy(v) for k,v in evidence.items() if k!='pending'}
        if not evidence['accepted']:
            return refuse(evidence['reason'])
        cap=float(evidence['cap_m'])
        # The new branch owns ONLY the contextual finite uncertainty allowance;
        # keep any stricter non-budget command cap from the ordinary guard.
        if base['accepted'] and base.get('cap_m') is not None:
            cap=min(cap,float(base['cap_m']))
        self._egress_pending={**evidence['pending'],'command_cap_m':cap}
        self._motion_classification_pending={'uncertain':True,
            'direction':(command/np.linalg.norm(command)).tolist(),'command_cap_m':cap}
        self._experiment_pending=self._retreat_pending=None
        result={'accepted':True,'cap_m':cap,'need_refresh':False,
            'reason':'uncertainty_egress_bounded_observation',
            'diagnostics':self.summary()}
        result['diagnostics']['ordinary_admission_reason']=base['reason']
        return result

    def observe_egress(self,before_state8,action7,after_state8):
        pending=self._egress_pending
        self._egress_pending=None
        if self.egress_inventory is None or self.egress_monitor is None:
            return self.egress_evidence.summary()
        return self.egress_evidence.observe(before_state8,action7,after_state8,pending=pending,
            guard=self,monitor=self.egress_monitor,inventory=self.egress_inventory)

    def observe(self,images,state8,action,*,monitor=None):
        command=np.asarray(action[:3],float)*.05
        norm=float(np.linalg.norm(command))
        pending=self._egress_pending
        if norm>1e-8:
            self._last_executed_was_egress=bool(pending is not None
                and np.allclose(command,pending['delta'],atol=1e-10,rtol=0)
                and norm<=pending['command_cap_m']+1e-9
                and np.allclose(self.state,pending['before']['state8'],atol=1e-9,rtol=0))
        old_direction=self._last_charged_experiment
        if self._last_executed_was_egress:
            # V23 has one persistent contextual ledger, not a new grant in an
            # unrelated legacy heading. Keep global measured uncertainty debt
            # charged by inherited observe, including residual zero holds.
            self._last_charged_experiment=None
        try:
            return super().observe(images,state8,action,monitor=monitor)
        finally:
            if self._last_executed_was_egress:
                self._last_charged_experiment=old_direction

    def summary(self):
        result=super().summary()
        ledger=getattr(self,'egress_evidence',None)
        result['uncertainty_egress']=None if ledger is None else ledger.summary()
        result['uncertainty_egress_admission']=copy.deepcopy(getattr(self,'last_egress_admission',None))
        return result
