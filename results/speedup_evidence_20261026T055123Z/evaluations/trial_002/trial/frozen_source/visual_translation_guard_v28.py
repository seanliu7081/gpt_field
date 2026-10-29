"""Finite acquisition of useful heading evidence under current Field geometry.

All metric geometry remains a model hypothesis. This version retains the
existing measured-travel ledgers and hard gates, but does not require a response
model on a new direction before allowing its first bounded observation.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_translation_guard_v27 import ProjectedTranslationGuardV27
from .visual_reverse_evidence_v28 import ReverseBoundaryEvidenceV28
from .visual_heading_evidence_v28 import HeadingEvidenceAcquisitionV28


class ProjectedTranslationGuardV28(ReverseBoundaryEvidenceV28, ProjectedTranslationGuardV27):
    version = 'finite-heading-and-comparable-reverse-evidence-v28'

    def __init__(self):
        super().__init__()
        self.heading_acquisition = HeadingEvidenceAcquisitionV28()
        self._heading_pending = None
        self._v28_current_transit_contract = None
        self.last_heading_acquisition = None
        self.last_reverse_qualification = None

    def _experiment_for(self, direction, *, nominal_away):
        contract = self._v28_current_transit_contract
        if contract is None or self.context.get('translation_mode') != 'transit':
            return super()._experiment_for(direction, nominal_away=nominal_away)
        # Only a CURRENT exact normal-transit contract may qualify this lookup.
        # No context is relabeled "separate" and no provisional row is committed.
        proposal = {'index': None, 'direction': direction.tolist(),
                    'support_generation': self._support_generation}
        if not self.experiments:
            return proposal, None
        angles = [self._angle(direction, row['direction']) for row in self.experiments]
        matched = int(np.argmin(angles))
        if angles[matched] <= self.compatible_direction_rad+1e-9:
            if self.experiments[matched]['measured_travel_m'] >= self.experiment_limit_m-1e-9:
                return None, 'translation_recovery_direction_exhausted'
            return {**proposal, 'index': matched}, None
        if min(angles) < self.fresh_direction_rad-1e-9:
            return None, 'translation_recovery_direction_not_distinct'
        if len(self.experiments) >= self.maximum_experiments:
            return None, 'translation_recovery_experiment_limit'
        previous = self.experiments[self._active_experiment()]
        checkpoint = self._finite_reverse_original_checkpoint
        checkpoint = self.information_checkpoint_m if checkpoint is None else checkpoint
        if (previous['measured_travel_m'] < checkpoint-1e-9
                and not previous.get('support_loss_observed', False)):
            return None, 'translation_recovery_direction_change_requires_review'
        return proposal, None

    def _v28_finish(self, base, reason, *, cap=0., accepted=False):
        if not accepted:
            self._motion_classification_pending = None
            self._experiment_pending = None
            self._retreat_pending = None
            self._heading_pending = None
        self._recovery_reason = reason
        self._v12_last_result = {'accepted': accepted, 'cap_m': cap,
            'need_refresh': not accepted, 'reason': reason}
        result = {**self._v12_last_result, 'diagnostics': self.summary()}
        result['diagnostics']['checked_proposal'] = copy.deepcopy(
            base.get('diagnostics', {}).get('checked_proposal'))
        return result

    def assess(self, delta_xyz, **kwargs):
        command = np.asarray(delta_xyz, float)
        size = float(np.linalg.norm(command))
        records = kwargs.get('nominal_regions') or []
        rotation = float(kwargs.get('rotation_rad', 0.))
        contract = (self._transit_contract(records, command, rotation)
                    if self.worsening and size > 1e-12 else None)
        previous = self._v28_current_transit_contract
        self._v28_current_transit_contract = contract
        self._heading_pending = None
        self.last_heading_acquisition = None
        self.last_reverse_qualification = None
        events = self.events
        try:
            base = super().assess(delta_xyz, **kwargs)
            reverse = self._finite_reverse_contract(records, command, rotation)
            if base.get('accepted') and reverse is not None and self.worsening:
                qualification = self.reverse_boundary_qualification()
                self.last_reverse_qualification = copy.deepcopy(qualification)
                if not qualification['qualified']:
                    return self._v28_finish(base, qualification['reason'])
                # Include the current qualification in accepted diagnostics.
                checked = copy.deepcopy(base.get('diagnostics', {}).get('checked_proposal'))
                base['diagnostics'] = {**self.summary(), 'checked_proposal': checked}
            if base.get('reason') not in (
                    'translation_projected_conflict_robot_response_unavailable',
                    'translation_projected_conflict_image_response_unavailable'):
                return base
            # These two reasons occur only AFTER V27's current geometry,
            # reference, domain, historical contradiction and boundary checks.
            if contract is None:
                return base
            experiment, error = self._experiment_for(command/size, nominal_away=False)
            if error:
                return self._v28_finish(base, error)
            snapshot = self._usable_snapshot()
            if snapshot is None or any(i not in snapshot['neighbors'] for i in self.worsening):
                return self._v28_finish(base, 'translation_heading_evidence_current_binding_invalid')
            binding = {**contract, 'warning_ids': sorted(self.worsening),
                'model_intent_key': getattr(kwargs.get('monitor'), 'intent_id', None),
                'representation_key': {i: snapshot['neighbors'][i]['response_representation']
                                       for i in sorted(self.worsening)}}
            response = kwargs.get('translation_response')
            predicted = None if response is None else response.predict(command, self.state)
            # An unsupported response does not invent measured motion. Querying
            # this heading gives diagnostics only; actual observers train later.
            query = (np.asarray(predicted['mean_delta_m'], float)
                     if predicted and predicted.get('supported')
                     and predicted.get('mean_delta_m') is not None else command)
            image_evidence = {i: self._neighbor_responses[i].evidence(query, self.state[:3])
                for i in self.worsening if i in self._neighbor_responses}
            used = (0. if experiment['index'] is None else
                    self.experiments[experiment['index']]['measured_travel_m'])
            assessment = self.heading_acquisition.assess(self.state, command,
                binding=binding, experiment=experiment['index'], heading_used_m=used,
                shared_used_m=self.observation_travel_m,
                intent_remaining_m=contract['remaining_measured_travel_m'],
                intent_remaining_actions=contract['remaining_actions'],
                robot_response=response, image_evidence=image_evidence)
            self.last_heading_acquisition = copy.deepcopy(assessment)
            reselect = assessment['reason'] == 'translation_heading_evidence_candidate_requires_cap_reselection'
            if not assessment.get('accepted') and not reselect:
                return self._v28_finish(base, assessment['reason'])
            cap = assessment['command_cap_m']
            if reselect and size-cap <= 1e-10:
                # The inherited selector's retry tolerance is 1e-10 m. Make
                # this uncommitted proposal strictly cross that threshold so
                # it cannot accidentally execute without an exact admission.
                cap = max(0., cap-2e-10)
            self.events = events
            self._retreat_pending = None
            # A capped proposal is not executable until the selector reruns all
            # current geometry checks on that exact smaller candidate.
            self._motion_classification_pending = None if reselect else {
                'uncertain': True, 'direction': (command/size).tolist(), 'command_cap_m': cap}
            self._experiment_pending = None if reselect else {**experiment, 'command_cap_m': cap}
            self._heading_pending = copy.deepcopy(assessment.get('pending'))
            return self._v28_finish(base, assessment['reason'], cap=cap, accepted=True)
        finally:
            self._v28_current_transit_contract = previous

    def observe(self, images, state8, action, *, monitor=None):
        before = None if self.state is None else self.state.copy()
        pending = self._heading_pending
        self._heading_pending = None
        super().observe(images, state8, action, monitor=monitor)
        if before is not None:
            self.heading_acquisition.observe(before, action, state8, pending=pending,
                experiment_index=self._last_charged_experiment)
        return self.summary()

    def summary(self):
        result = super().summary()
        if not hasattr(self, 'heading_acquisition'):
            return result
        result['heading_evidence_acquisition'] = {
            **self.heading_acquisition.summary(),
            'last_admission': copy.deepcopy(self.last_heading_acquisition)}
        result['reverse_boundary_qualification'] = copy.deepcopy(self.last_reverse_qualification)
        result['recovery_experiments']['normal_transit_heading_qualification'] = (
            'current exact model-selected full-scene margin-separating contract; '
            'same physical slots, measured debt and distinct-heading review checkpoint')
        return result
