"""Expose paid, observation-bound reference evidence at explicit control handoffs.

The inherited 32-frame upper bound and all extraction/registration tests remain
unchanged. A board is a presentation of recorded motion, not motion permission,
automatic semantic selection or a refund of physical exploration allowance.
"""
from __future__ import annotations

from .visual_reference_candidates_v24 import CausalReferenceCandidatesV24


class CausalReferenceCandidatesV28(CausalReferenceCandidatesV24):
    version = 'paid-control-handoff-reference-review-v28'
    additional_handoff_reasons = frozenset({
        'translation_projected_conflict_requires_current_transit_geometry',
        'translation_projected_conflict_prior_reverse_contradiction',
        'translation_projected_conflict_reference_contradicted',
        'translation_projected_conflict_boundary_unresolved_or_closing',
        'translation_projected_conflict_robot_response_unavailable',
        'translation_projected_conflict_image_response_unavailable',
        'translation_projected_conflict_measured_allowance_exhausted',
        'translation_reverse_observed_further_worsening',
        'translation_reverse_requires_current_empty_tool_full_sweep',
        'translation_reverse_boundary_unavailable',
        'translation_reverse_boundary_evidence_unavailable',
        'translation_reverse_boundary_closing',
        'translation_heading_evidence_current_binding_invalid',
        'translation_heading_evidence_budget_exhausted',
        'translation_heading_evidence_observed_robot_nonresponsive',
        'translation_heading_evidence_robot_response_unavailable',
        'translation_heading_evidence_geometry_or_domain_changed',
        'translation_heading_evidence_evidence_contradicted',
        'translation_heading_evidence_allowance_exhausted',
        'translation_heading_evidence_motion_stalled',
        'translation_heading_evidence_contradicted',
        'translation_heading_evidence_context_changed',
        'translation_heading_evidence_no_useful_command',
        'translation_heading_evidence_response_unavailable',
    })
    handoff_reasons = CausalReferenceCandidatesV24.handoff_reasons | additional_handoff_reasons

    def summary(self):
        result = super().summary()
        result['handoff_review_contract'] = {
            'evidence_source': 'already_acknowledged_RGB_and_robot_state_history',
            'maximum_history_frames': self.maximum_frames,
            'explicit_model_selection_required': True,
            'automatic_identity_or_motion_permission': False,
            'physical_budget_replenished': False,
        }
        handoff = result.get('last_handoff')
        if handoff and result.get('review_required'):
            result['reason'] = 'paid_motion_available_at_control_handoff'
            result['interpretation'] = (
                'Current control was interrupted. Recorded motion can support semantic '
                'reference review, but neither the interruption nor a candidate proves '
                'the previous feature was wrong, the new feature is distal tool, or a '
                'path is clear. Select explicitly from the current original RGB views; '
                'all measured physical debt and current risk remain.')
        return result
