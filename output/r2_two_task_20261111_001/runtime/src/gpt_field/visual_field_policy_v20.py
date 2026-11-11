"""Separate fixed-world-goal tracking from image-response-dependent correction.

Weak image-response evidence never becomes validated here. It may provisionally
permit construction of a robot-state-controlled candidate; the V20 executor
must then admit that exact candidate through fresh whole-scene geometry.
"""
from __future__ import annotations

import copy
import time
import numpy as np

from .visual_field_policy_v19 import VisualFieldPolicyV19
from .visual_field_v20 import MetricFieldExecutorV20
from .visual_field_subscription_v20 import geometry_semantics


class VisualFieldPolicyV20(VisualFieldPolicyV19):
    def reset(self):
        super().reset()
        self.local = MetricFieldExecutorV20()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)
        self.local.scene_assessor = self._scene_assess
        self._transit_anchor_shifted = False
        self._anchor_before_observe = None
        self._transit_limit_deferred = False

    def _refresh(self, obs):
        self.local.proprioceptive_transit_request = None
        previous_anchor = self.monitor.tracks.get("target")
        latency = super()._refresh(obs)
        # Retaining the same stationary-anchor hypothesis cannot erase a
        # measured jump. Only explicit reseeding of a new track can resolve it.
        external = self.prediction.get("visual_references", {}).get("external") or {}
        if (external.get("update") == "seed"
                and self.monitor.tracks.get("target") is not previous_anchor):
            self._transit_anchor_shifted = False
        self._anchor_before_observe = None
        self._transit_limit_deferred = False
        return latency

    def _observe_anchor_shift(self):
        target = self.monitor.tracks.get("target")
        previous = getattr(self, "_anchor_before_observe", None)
        if (previous is not None and target is not None and target.reliable
                and np.linalg.norm(np.asarray(target.position) - previous) > 6.):
            # Existing weak-budget status can mask this base-monitor event.
            self._transit_anchor_shifted = True

    def _transit_eligibility(self, reason, state8):
        pred = self.prediction or {}
        monitor = self.monitor
        external = pred.get("visual_references", {}).get("external") or {}
        resolved = monitor.refs.get("external") or {}
        record = getattr(monitor, "_active_reference", None)
        if (reason != monitor.uncertainty_budget_reason or record is None
                or record.get("blocked_reason") != reason
                or pred.get("controller") != "visual_field" or not self.field_configured
                or pred.get("phase") not in ("approach", "transport", "recover")
                or external.get("moving_kind") != "tool" or resolved.get("moving_kind") != "tool"
                or pred.get("target_contact") != "none"
                or pred.get("contact_alignment") is not None
                or self.local.contact_refinement_enabled or self.local.goal_revisions
                or not monitor._external_reliable() or monitor.missing >= 2
                or monitor.reference_update_status == "external_retain_rejected"
                or monitor.status in ("visual_response_contradiction", "target_anchor_shifted")
                or self._transit_anchor_shifted
                or self.translation_guard.domain_invalid
                or (pred.get("motion_context") or {}).get("translation_mode") not in ("transit", "separate")
                or monitor.cumulative_response_evidence().get("contradicted")):
            return False
        carried = self.target_contact.carried_summary(state8)
        if carried.get("active") or carried.get("candidate_ids"):
            return False
        return monitor._response_evidence().get("state") == "weakly_observable"

    def _field_control_limits(self, obs):
        self.local.proprioceptive_transit_request = None
        self._transit_limit_deferred = False
        limits = super()._field_control_limits(obs)
        if not self._transit_eligibility(limits[1], obs.state):
            return limits
        self.local.proprioceptive_transit_request = {
            "version": "v20-current-proprioceptive-transit",
            "reason": limits[1], "state8": list(obs.state),
            "observation_epoch": self.translation_guard.sweep_observation_epoch,
            "goal_xyz": self.local.goal.tolist(),
            "reference_id": self.monitor._active_reference["id"],
            "target_id": self.prediction["target_id"],
            "anchor_shifted": False, "raw_external_role": "tool", "target_contact": "none",
        }
        self._transit_limit_deferred = True
        return None, None

    def _post_field_control_limit(self, after):
        self._observe_anchor_shift()
        limits = super()._post_field_control_limit(after)
        if self._transit_eligibility(limits[1], after.state):
            # Defer only the request refresh. Every next actual action must
            # receive a fresh bound candidate admission; no permission is cached.
            return None, None
        return limits

    def _allows_planar_correction(self):
        return (not getattr(self.local, "proprioceptive_transit_request", None)
                and super()._allows_planar_correction())

    def _refine_contact_goal(self, obs):
        if getattr(self.local, "proprioceptive_transit_request", None):
            return None
        return super()._refine_contact_goal(obs)

    def _observable_feedback(self):
        result = super()._observable_feedback()
        result["geometry_semantics"] = geometry_semantics()
        result["proprioceptive_transit"] = {
            "visual_response_limit_deferred_to_candidate": getattr(self, "_transit_limit_deferred", False),
            "target_anchor_shifted": getattr(self, "_transit_anchor_shifted", False),
            "candidate_admission": copy.deepcopy(getattr(self.local, "last_proprioceptive_transit_admission", None)),
            "image_response_validated_by_this_branch": False,
            "physical_budget_reset_by_this_branch": False,
        }
        return result

    def act(self, obs):
        # Requests are single-select provisional inputs, not reusable permission.
        self.local.proprioceptive_transit_request = None
        decision = super().act(obs)
        decision.diagnostics["policy_version"] = "visual-field-policy-v20"
        self.local.proprioceptive_transit_request = None
        return decision

    def observe_transition(self, before, decision, after):
        target = self.monitor.tracks.get("target")
        self._anchor_before_observe = (np.asarray(target.position).copy()
            if target is not None and target.reliable else None)
        row = super().observe_transition(before, decision, after)
        started = time.monotonic()
        self._observe_anchor_shift()
        self._anchor_before_observe = None
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic() - started
        return row
