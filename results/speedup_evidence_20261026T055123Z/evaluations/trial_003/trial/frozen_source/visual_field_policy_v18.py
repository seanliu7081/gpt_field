"""Model-selected causal RGB references with early review and response reuse.

The candidate bank never selects a semantic identity or grants motion. Replay
initializes predictors only: the existing guards still require fresh evidence
and retain their physical travel, contact and scene constraints.
"""
from __future__ import annotations

import copy
import time

from .visual_field_policy_v17 import VisualFieldPolicyV17
from .visual_field_policy_v15 import recovery_options
from .visual_reference_candidates_v18 import CausalReferenceCandidatesV18
from .visual_reference_replay_v18 import warm_start_reference_response


class VisualFieldPolicyV18(VisualFieldPolicyV17):
    def reset(self):
        super().reset()
        self.reference_candidates = CausalReferenceCandidatesV18()
        self._resolved_reference_candidates = {}
        self._requested_reference_reviews = set()
        self.reference_replay = {"accepted": False, "reason": "not_requested"}
        self.client.reference_candidate_provider = self.reference_candidates.image
        self.client.reference_candidate_resolver = self._resolve_reference_candidate

    def _resolve_reference_candidate(self, identity, observation):
        resolved = self.reference_candidates.resolve_candidate(identity, observation)
        self._resolved_reference_candidates[identity] = copy.deepcopy(resolved)
        return resolved

    def _observable_feedback(self):
        result = super()._observable_feedback()
        bank = getattr(self, "reference_candidates", None)
        report = {} if bank is None else bank.summary()
        result["reference_repair"] = report
        result["reference_replay"] = copy.deepcopy(getattr(self, "reference_replay", None))
        if report.get("review_required") or report.get("review_recommended"):
            # This is a presentation-level reconciliation of two independent
            # observations. Do not change the live monitor's validity gates.
            health = result.get("visual", {}).get("reference_health")
            if isinstance(health, dict):
                health["candidate_review_required"] = bool(report.get("review_required"))
                health["recommended_action"] = (
                    "The current response remains unvalidated. Independent RGB motion "
                    "evidence requests review of this support, even if its appearance "
                    "track is reliable. Inspect the current numbered candidates and "
                    "both raw views; select a visible rigid distal-tool feature and "
                    "correct the whole-tool outline when justified. Candidate motion "
                    "alone does not identify a tool or certify clearance. Selecting a "
                    "reference never replenishes physical exploration allowance.")
        return result

    def _refresh(self, obs):
        self._resolved_reference_candidates.clear()
        latency = super()._refresh(obs)
        started = time.monotonic()
        raw = self.prediction.get("raw_wire_prediction") or {}
        identity = self.prediction.get("reference_candidate_id", raw.get("reference_candidate_id"))
        if identity is not None:
            resolved = self._resolved_reference_candidates.get(identity)
            self.reference_replay = (
                {"accepted": False, "reason": "candidate_not_resolved_for_current_response"}
                if resolved is None else warm_start_reference_response(
                    self.monitor, self.translation_guard, resolved, obs))
        else:
            self.reference_replay = {"accepted": False, "reason": "no_candidate_selected"}
        self.reference_candidates.configure(obs, self.monitor, self.effective_motion_context)
        # A response warm-start changes possible predictions, never the actual
        # admission contract. Refresh previews through those same contracts.
        self.recovery_options = recovery_options(
            self.local, self.translation_guard, obs.state,
            monitor=self.monitor, robot_response=self.robot_response)
        self._screen_recovery_options(obs.state)
        self.feedback = self._observable_feedback()
        elapsed = time.monotonic() - started
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics["policy_version"] = "visual-field-policy-v18"
        if decision.diagnostics.get("refreshed"):
            decision.diagnostics["reference_replay"] = copy.deepcopy(self.reference_replay)
        return decision

    def observe_transition(self, before, decision, after):
        row = super().observe_transition(before, decision, after)
        started = time.monotonic()
        report = self.reference_candidates.observe(before, after, self.monitor)
        review_id = report.get("review_id")
        if (report.get("review_required") and review_id is not None
                and review_id not in self._requested_reference_reviews):
            # A single review interrupts stale-reference Field reuse while
            # useful measured travel remains. Holds cannot issue fresh tickets.
            self._requested_reference_reviews.add(review_id)
            if not self.refresh_reason:
                self.refresh_reason = "causal_controlled_reference_review"
            self.recovery_options = recovery_options(
                self.local, self.translation_guard, after.state,
                monitor=self.monitor, robot_response=self.robot_response)
            self._screen_recovery_options(after.state)
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic() - started
        return row
