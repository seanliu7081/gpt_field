"""Request paid-history repair at a typed reference handoff, not only failure.

A formally reliable image track can have an unusable measured response. This
adapter lets its existing repair_required signal request candidate extraction
at a real handoff. It grants no control, identity or new physical allowance.
"""
from __future__ import annotations

from .visual_reference_candidates_v30 import ReferenceReviewEvent, ReferenceReviewKind
from .visual_reference_review_v30 import (
    TypedReferenceReviewPolicyMixinV30,
    reference_review_event as legacy_reference_review_event,
)


def reference_review_event(monitor, *, handoff_present=False,
                           failed_control_handoff=False, detail_reason=None):
    """Preserve V30 events and add reliable-track response-repair handoffs."""
    event = legacy_reference_review_event(monitor, handoff_present=handoff_present,
        failed_control_handoff=failed_control_handoff, detail_reason=detail_reason)
    if event is not None or not handoff_present:
        return event
    external = (getattr(monitor, 'refs', {}) or {}).get('external')
    if not external:
        # Model omission is not observed tracking loss; no former feature is
        # silently restored and no stale board is claimed to be current.
        return None
    report = monitor.summary() if hasattr(monitor, 'summary') else {}
    if (report.get('reference_health') or {}).get('repair_required'):
        return ReferenceReviewEvent(ReferenceReviewKind.CONTROLLED_RESPONSE_UNRELIABLE,
                                    detail_reason=detail_reason)
    return None


class TypedReferenceReviewPolicyMixinV34(TypedReferenceReviewPolicyMixinV30):
    """Same call order and tickets as V30; dispatch from current typed health."""

    def reset(self):
        self.reference_review_dispatch = None
        return super().reset()

    def _review_paid_reference(self, observation, reason):
        failed = bool(getattr(self, '_typed_reference_control_failure', False))
        self._typed_reference_control_failure = False
        event = reference_review_event(self.monitor, handoff_present=reason is not None,
                                       failed_control_handoff=failed, detail_reason=reason)
        prediction = getattr(self, 'prediction', None) or {}
        declared = prediction.get('visual_references')
        explicitly_omitted = (isinstance(declared, dict) and 'external' in declared
                              and declared['external'] is None)
        self.reference_review_dispatch = {
            'handoff_present': reason is not None,
            'failed_control_handoff': failed,
            'event': None if event is None else event.kind.value,
            'external_reference_explicitly_omitted': explicitly_omitted,
            'dispatch_status': ('typed_review_requested' if event is not None
                else 'model_external_reference_omitted' if explicitly_omitted
                else 'no_current_review_event'),
            'reason_text_is_dispatch_key': False,
            'automatic_reference_restore': False,
            'physical_allowance_renewed': False,
        }
        if event is None:
            return
        report = self.reference_candidates.review(observation, self.monitor, event)
        self.reference_review_dispatch.update(
            extraction_status=report.get('extraction_status'),
            eligibility_reason=report.get('eligibility_reason'),
            candidate_count=len(report.get('candidates') or []))
        identity = report.get('review_id')
        if report.get('review_required') and identity is not None:
            if identity not in self._requested_reference_reviews:
                self._requested_reference_reviews.add(identity)
                if not self.refresh_reason:
                    self.refresh_reason = 'causal_controlled_reference_review'
                self.events.append({'step': self.step_index-1,
                    'reason': 'paid_observation_reference_review',
                    'review_event': event.kind.value, 'review_id': identity,
                    'physical_allowance_renewed': False})
