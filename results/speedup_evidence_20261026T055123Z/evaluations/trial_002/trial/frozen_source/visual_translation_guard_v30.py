"""Qualified noncontact approach to a declared target, with every obstacle retained.

A shrinking target image gap is expected during attraction. It is not itself
an unexplained obstacle response, nor a clearance or grasp certificate. This
opt-in changes only target direction/boundary interpretation; existing exact
whole-scene, prospective response, fixed-intent and physical-debt checks apply.
"""
from __future__ import annotations
import copy
import numpy as np
from .visual_translation_guard_v29 import ProjectedTranslationGuardV29


class ProjectedTranslationGuardV30(ProjectedTranslationGuardV29):
    version = 'qualified-target-attraction-and-retained-obstacle-risk-v30'

    def __init__(self):
        super().__init__()
        self.target_approach_context = None
        self._target_approach_arguments = None
        self.last_target_approach_execution = None

    def configure(self, *args, **kwargs):
        self.target_approach_context = None
        self.last_target_approach_execution = None
        return super().configure(*args, **kwargs)

    def bind_target_approach_context(self, context):
        """Bind the current model role; this is neither evidence nor permission."""
        valid = (isinstance(context, dict)
            and context.get('controller') == 'visual_field'
            and context.get('phase') in ('approach', 'align')
            and isinstance(context.get('target_id'), str)
            and context.get('target_id') == self.target_id)
        self.target_approach_context = copy.deepcopy(context) if valid else None
        return {'bound': bool(valid), 'physical_budget_modified': False,
                'motion_or_clearance_authorized': False}

    def _target_approach_qualification(self, identity, contract):
        role = self.target_approach_context or {}
        arguments = self._target_approach_arguments
        info = {'target_id': identity, 'eligible': False,
            'historical_target_closing_retained': identity in self.boundary_closing_ids,
            'historical_warning_retained': identity in self.worsening,
            'physical_budget_replenished': False, 'clearance_certified': False}
        def fail(reason):
            info['reason'] = reason
            self.last_target_approach_execution = copy.deepcopy(info)
            return False
        if identity != self.target_id or arguments is None:
            return False
        if (role.get('target_id') != identity or role.get('controller') != 'visual_field'
                or role.get('phase') not in ('approach', 'align')
                or (self.context or {}).get('purpose') != 'task_translation'
                or (self.context or {}).get('translation_mode') != 'transit'
                or contract.get('route_family') != 'normal'
                or contract.get('target_id') != identity
                or contract.get('target_contact') != 'none'
                or contract.get('empty_tool') is not True
                or contract.get('contact_or_visual_goal_dependency') is not False
                or contract.get('outside_target_contact_band') is not True):
            return fail('translation_target_approach_role_or_contact_ineligible')
        command, kwargs = arguments
        try:
            start = np.asarray(contract['start_xyz_m'], float)
            end = np.asarray(contract['end_xyz_m'], float)
            goal = np.asarray(contract['fixed_goal_xyz_m'], float)
            if any(v.shape != (3,) for v in (start,end,goal)) or not np.isfinite(np.r_[start,end,goal]).all():
                raise ValueError('Invalid goal')
            delta = end-start
            proof = contract['object_proofs'][identity]
            # No past-the-goal action and no lateral detour masquerading as
            # target approach. Positive full-margin distance is independently
            # checked by the unchanged parent contract for the whole segment.
            progress = (np.dot(delta, goal-start) >= np.dot(delta,delta)-1e-12
                and np.linalg.norm(end-goal) < np.linalg.norm(start-goal)-1e-9
                and proof['end_gap_m'] < proof['start_gap_m']-1e-10)
            exact = np.allclose(delta, command, atol=1e-10, rtol=0.)
        except (KeyError,TypeError,ValueError):
            return fail('translation_target_approach_fixed_goal_unavailable')
        if not progress or not exact:
            return fail('translation_target_approach_lacks_fixed_goal_progress')
        monitor = kwargs.get('monitor')
        bank = self.reference_witness
        if monitor is None or bank is None:
            return fail('translation_target_approach_causal_reference_unavailable')
        raw = bank.raw_support_evidence(self.state, monitor)
        info['controlled_reference'] = copy.deepcopy(raw)
        if (raw.get('current_bound') is not True or raw.get('reliable') is not True
                or raw.get('causal_supported') is not True or raw.get('contradicted')
                or raw.get('confirmation_required') is not False
                or raw.get('evidence_kind') != 'prospective'):
            return fail('translation_target_approach_causal_reference_unavailable')
        response = kwargs.get('translation_response')
        predicted = None if response is None else response.predict(command, self.state)
        responsive = bool(predicted and predicted.get('supported') is True
            and predicted.get('responsive') is True)
        query = np.asarray(predicted['mean_delta_m'],float) if responsive else command
        witness = self._neighbor_responses.get(identity)
        evidence = None if witness is None else witness.evidence(query, self.state[:3])
        info['target_relative_response'] = copy.deepcopy(evidence)
        boundary = self.boundary_evidence.get(identity) or {}
        current = (boundary.get('comparable') is True
            and boundary.get('measurement_epoch') == self.neighbor_measurement_epochs.get(identity))
        # A finite observation may collect missing SAME-heading evidence, but
        # cannot explain away an already robust current closing interval.
        validated = bool(responsive and evidence and evidence.get('validated') is True)
        contradicted = bool(evidence and evidence.get('recent_measured_sample')
            and evidence.get('held_out_count', 0) >= 2
            and evidence.get('held_out_zero_sse', 0.) > .25
            and evidence.get('held_out_prediction_sse', 0.) >= .75*evidence['held_out_zero_sse'])
        if contradicted:
            return fail('translation_target_approach_relative_response_contradicted')
        if not current or boundary.get('classification') not in ('inconclusive','opening','closing'):
            return fail('translation_target_approach_current_boundary_unavailable')
        if boundary['classification'] == 'closing' and not validated:
            return fail('translation_target_approach_closing_requires_validated_response')
        info['response_stage'] = ('empirically_validated_fixed_intent' if validated
            else 'finite_existing_heading_acquisition_required')
        # V27/V28 below still enforce responsive robot + same-heading heldouts
        # before V29 supported execution. This hook grants no physical budget.
        info.update(eligible=True, reason='qualified_expected_noncontact_target_approach',
            target_start_gap_m=proof['start_gap_m'], target_end_gap_m=proof['end_gap_m'],
            target_minimum_segment_gap_m=proof['minimum_segment_gap_m'],
            target_contact_band_m=contract.get('target_band_m'),
            target_core_distance_lower_bound_m=contract.get('target_core_distance_lower_bound_m'),
            fixed_goal_xyz_m=goal.tolist(), mode='conditional_model_target_attraction')
        self.last_target_approach_execution = copy.deepcopy(info)
        return True

    def _warned_object_gap_allowed(self, identity, proof, values, contract):
        return (super()._warned_object_gap_allowed(identity, proof, values, contract)
            or self._target_approach_qualification(identity, contract))

    def _projected_boundary_allowed(self, identity, evidence, contract):
        if super()._projected_boundary_allowed(identity, evidence, contract):
            return True
        comparable = (evidence.get('comparable')
            and evidence.get('measurement_epoch') == self.neighbor_measurement_epochs.get(identity)
            and evidence.get('classification') in ('inconclusive','opening','closing'))
        return bool(comparable and self._target_approach_qualification(identity, contract))

    def assess(self, delta_xyz, **kwargs):
        previous = self._target_approach_arguments
        self._target_approach_arguments = (np.asarray(delta_xyz,float), kwargs)
        self.last_target_approach_execution = None
        try:
            result = super().assess(delta_xyz, **kwargs)
            info = self.last_target_approach_execution
            if info is not None:
                if result.get('accepted') and self._validated_pending is not None and info.get('eligible'):
                    self._validated_pending = {**self._validated_pending,
                        'target_approach_role': copy.deepcopy(self.target_approach_context)}
                self.last_target_approach_execution = {**info,
                    'actual_candidate_accepted': bool(result.get('accepted')),
                    'final_admission_reason': result.get('reason')}
                result.setdefault('diagnostics', {})['target_approach_execution'] = copy.deepcopy(
                    self.last_target_approach_execution)
            return result
        finally:
            self._target_approach_arguments = previous

    def _supported_acknowledgement_extra_valid(self, pending):
        return (super()._supported_acknowledgement_extra_valid(pending)
            and ('target_approach_role' not in pending
                 or pending['target_approach_role'] == self.target_approach_context))

    def summary(self):
        result = super().summary()
        result['target_approach_execution'] = copy.deepcopy(getattr(self,'last_target_approach_execution',None))
        return result
