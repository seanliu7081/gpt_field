"""Policy adapter for typed reference review at actual runtime events."""
from __future__ import annotations

from .visual_reference_candidates_v30 import ReferenceReviewEvent, ReferenceReviewKind


def reference_review_event(monitor, *, handoff_present=False,
                           failed_control_handoff=False, detail_reason=None):
    """Classify current evidence; no reason-string membership controls dispatch.

    An unavailable stationary anchor is distinct from a lost moving feature:
    the latter can use paid history, while the former cannot validate its
    anchor-relative candidate histories. Risk events indicate a control
    interruption, never a claim of actual collision or wrong identity.
    """
    external = (getattr(monitor, "refs", {}) or {}).get("external")
    if not external:
        return None
    tracks = getattr(monitor, "tracks", {})
    moving, target = tracks.get("moving"), tracks.get("target")
    kind = None
    if target is None or not target.reliable:
        kind = ReferenceReviewKind.TARGET_REFERENCE_LOST
    elif moving is None or not moving.reliable:
        kind = ReferenceReviewKind.CONTROLLED_REFERENCE_LOST
    else:
        if not failed_control_handoff:
            return None
        report = monitor.summary() if hasattr(monitor, "summary") else {}
        health = report.get("reference_health") or {}
        if health.get("repair_required"):
            kind = ReferenceReviewKind.CONTROLLED_RESPONSE_UNRELIABLE
        elif failed_control_handoff:
            kind = ReferenceReviewKind.CONTROL_RISK_HANDOFF
    if kind is None:
        return None
    return ReferenceReviewEvent(kind, detail_reason=detail_reason)


class TypedReferenceReviewPolicyMixinV30:
    """Use existing V24 call sites without changing action or feedback order."""

    def reset(self):
        self._typed_reference_control_failure = False
        return super().reset()

    def _failure(self, reason):
        previous = (getattr(self, "failed_hypotheses", None) or [None])[-1]
        result = super()._failure(reason)
        current = (getattr(self, "failed_hypotheses", None) or [None])[-1]
        if current is not previous:
            # Existing policy code already distinguishes a failed control
            # hypothesis from normal arrival/budget refresh. Do not duplicate
            # its legacy strings here or manufacture a new failure counter.
            self._typed_reference_control_failure = True
        return result

    def _review_paid_reference(self, observation, reason):
        failed = bool(getattr(self, "_typed_reference_control_failure", False))
        self._typed_reference_control_failure = False
        event = reference_review_event(self.monitor, handoff_present=reason is not None,
                                       failed_control_handoff=failed, detail_reason=reason)
        if event is None:
            return
        report = self.reference_candidates.review(observation, self.monitor, event)
        identity = report.get("review_id")
        if report.get("review_required") and identity is not None:
            if identity not in self._requested_reference_reviews:
                self._requested_reference_reviews.add(identity)
                if not self.refresh_reason:
                    self.refresh_reason = "causal_controlled_reference_review"
                self.events.append({"step": self.step_index-1,
                                    "reason": "paid_observation_reference_review",
                                    "review_event": event.kind.value, "review_id": identity,
                                    "physical_allowance_renewed": False})
