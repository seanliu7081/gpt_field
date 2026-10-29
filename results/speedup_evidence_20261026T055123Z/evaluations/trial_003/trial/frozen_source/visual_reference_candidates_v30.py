"""Typed review of already acknowledged visual evidence, never motion permission.

Control stop text is retained for explanation but is not a dispatch key. The
same current-observation, anchor, domain and bounded-history tests still apply
when the controlled correspondence is lost. The lost track is not resurrected.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from .visual_monitor_v4 import _orientation_distance
from .visual_field import _gray
from .visual_reference_candidates_v24 import CausalReferenceCandidatesV24


class ReferenceReviewKind(str, Enum):
    CONTROLLED_REFERENCE_LOST = "controlled_reference_lost"
    CONTROLLED_RESPONSE_UNRELIABLE = "controlled_response_unreliable"
    TARGET_REFERENCE_LOST = "target_reference_lost"
    CONTROL_RISK_HANDOFF = "control_risk_handoff"
    STATIC_SUPPORT_REVIEW = "static_support_review"


@dataclass(frozen=True)
class ReferenceReviewEvent:
    kind: ReferenceReviewKind
    source: str = "current_control_handoff"
    detail_reason: str | None = None

    def __post_init__(self):
        if not isinstance(self.kind, ReferenceReviewKind):
            raise TypeError("Reference review requires a typed ReferenceReviewKind")

    def to_dict(self):
        result = asdict(self)
        result["kind"] = self.kind.value
        return result


class CausalReferenceCandidatesV30(CausalReferenceCandidatesV24):
    version = "typed-paid-reference-review-v30"

    def __init__(self):
        self._review_event = None
        self._extraction_status = "not_requested"
        self._eligibility_reason = None
        self._last_extraction_sequence = None
        self._last_extraction_path = None
        super().__init__()

    def configure(self, observation, monitor, motion_context):
        old_reference = self._reference_track
        super().configure(observation, monitor, motion_context)
        if self._reference_track is not old_reference:
            self._review_event = None
            self._extraction_status = "not_requested"
            self._eligibility_reason = None
            self._last_handoff = None
        return self.summary()

    def _build_candidates(self):
        # Called either by the inherited static-support trigger or typed review.
        # No estimator, geometry, pending action or physical ledger is touched.
        super()._build_candidates()
        self._last_extraction_sequence = self.sequence
        self._last_extraction_path = self.path_m
        self._extraction_status = "available" if self._candidates else "extracted_none"
        self._eligibility_reason = None

    def observe(self, before, after, monitor):
        count = self._build_count
        super().observe(before, after, monitor)
        if self._build_count != count:
            self._review_event = ReferenceReviewEvent(
                ReferenceReviewKind.STATIC_SUPPORT_REVIEW,
                source="acknowledged_passive_motion").to_dict()
        if self._extraction_status == "available" and not self._candidates:
            self._extraction_status = "ineligible"
            self._eligibility_reason = "previous_candidates_no_longer_current"
        return self.summary()

    def _eligibility(self, observation, monitor, event):
        if not self._matches(observation):
            return "observation_binding_mismatch"
        if event.kind == ReferenceReviewKind.TARGET_REFERENCE_LOST:
            return "stationary_anchor_review_required"
        if self.status != "observing":
            return "history_" + self.status
        if self._moving_kind != "tool":
            return "controlled_role_is_not_tool"
        track, anchor = self._live(monitor)
        if track is None or track is not self._reference_track:
            return "controlled_correspondence_binding_changed"
        if anchor is None or not anchor.reliable:
            return "stationary_anchor_unavailable"
        if (not np.array_equal(getattr(anchor, "previous_image", None), _gray(self.current.rgb))
                or self.current.anchor is None
                or not np.array_equal(anchor.position, self.current.anchor)):
            return "stationary_anchor_observation_binding_mismatch"
        state = np.asarray(getattr(monitor, "state", []), float)
        if state.shape != (8,) or not np.array_equal(state, self.current.state):
            return "monitor_state_binding_mismatch"
        if self._reference_origin_state is None:
            return "reference_origin_unavailable"
        if _orientation_distance(state[3:6], self._reference_origin_state[3:6]) > .035:
            return "orientation_domain_changed"
        if np.max(abs(state[6:] - self._reference_origin_state[6:])) > .002:
            return "finger_domain_changed"
        history = self._history()
        if len(history) < 4:
            return "insufficient_acknowledged_history"
        if self.path_m - self._reference_origin_path < .001:
            return "insufficient_paid_reference_motion"
        if any(not frame.anchor_reliable for frame in history):
            return "stationary_anchor_missing_in_retained_history"
        if (_orientation_distance(history[0].state[3:6], history[-1].state[3:6]) > .035
                or np.max(abs(history[0].state[6:] - history[-1].state[6:])) > .002):
            return "retained_history_domain_changed"
        return None

    def review(self, observation, monitor, event):
        if not isinstance(event, ReferenceReviewEvent):
            raise TypeError("review requires a ReferenceReviewEvent, not stop text")
        self._review_event = event.to_dict()
        reason = self._eligibility(observation, monitor, event)
        self._last_handoff = {"event": event.kind.value, "reason": event.detail_reason,
                              "sequence": self.sequence, "physical_debt_modified": False}
        if reason is not None:
            self._extraction_status, self._eligibility_reason = "ineligible", reason
            return self.summary()
        if not self._candidates:
            if (self._last_extraction_path is not None
                    and self.path_m - self._last_extraction_path < .001):
                # A hold/repeated request does not cause another expensive pass.
                # Keep an exact-current empty result distinct from no attempt.
                same = self.sequence == self._last_extraction_sequence
                self._extraction_status = "extracted_none" if same else "ineligible"
                self._eligibility_reason = (None if same else "insufficient_new_paid_motion_since_extraction")
                return self.summary()
            self._build_candidates()
        self._eligibility_reason = None
        if self._candidates:
            self._extraction_status = "available"
            if self._review_id is None:
                self._review_id = f"reference-{self.reference_generation}-handoff-{self.sequence}"
        return self.summary()

    def handoff(self, observation, monitor, event):
        """Compatibility method name with a deliberately typed-only contract."""
        return self.review(observation, monitor, event)

    def summary(self):
        result = super().summary()
        result.update(extraction_status=self._extraction_status,
                      eligibility_reason=self._eligibility_reason,
                      review_event=copy.deepcopy(self._review_event),
                      last_extraction_sequence=self._last_extraction_sequence,
                      last_extraction_path_m=self._last_extraction_path)
        if self._extraction_status == "ineligible":
            # Keep any historical candidates internally, but do not offer a
            # board as current when this observation/domain failed eligibility.
            result.update(review_required=False, review_recommended=False, candidates=[])
        if result["review_required"]:
            result["reason"] = "paid_motion_available_at_typed_reference_review"
        elif self._extraction_status != "not_requested":
            result["reason"] = self._eligibility_reason or "extraction_completed_no_qualified_candidates"
        result["handoff_review_contract"] = {
            "dispatch": "typed_current_reference_or_control_event",
            "stop_text_is_diagnostic_only": True,
            "evidence_source": "already_acknowledged_RGB_and_robot_state_history",
            "maximum_history_frames": self.maximum_frames,
            "explicit_model_selection_required": True,
            "automatic_identity_or_motion_permission": False,
            "physical_budget_replenished": False}
        return result

    def image(self, observation=None):
        if self._extraction_status == "ineligible":
            return None
        return super().image(observation)

    def resolve_candidate(self, candidate_id, observation):
        if self._extraction_status == "ineligible":
            raise ValueError("reference_candidate_review_ineligible")
        return super().resolve_candidate(candidate_id, observation)
