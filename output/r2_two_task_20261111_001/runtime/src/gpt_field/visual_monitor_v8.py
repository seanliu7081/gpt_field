"""Separate an inconclusive RGB response from measured response contradiction.

Only recorded RGB and robot proprioception enter these checks. V7's persistent
correspondence budget and V6's independent payload/tool attachment remain intact.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from .visual_monitor_v4 import _orientation_distance
from .visual_monitor_v7 import RoleAwareVisualMonitorV7


class RoleAwareVisualMonitorV8(RoleAwareVisualMonitorV7):
    uncertainty_budget_reason = "controlled_reference_uncertainty_budget_exhausted"
    contradiction_reason = "controlled_reference_response_contradicted_change_correspondence"

    def __init__(self):
        super().__init__()
        self._prediction_window = deque(maxlen=self.recent_window_size)
        self._pending_prediction = None
        self._mapping_had_supported_response = False

    def _reset_observability(self):
        self._prediction_window.clear()
        self._pending_prediction = None
        self._mapping_had_supported_response = False

    def configure(self, images, references, state8, *, intent_id=None):
        old_epoch = self.epoch
        super().configure(images, references, state8, intent_id=intent_id)
        if self.epoch != old_epoch:
            self._reset_observability()
        return self.summary()

    @staticmethod
    def _extent(deltas):
        if not len(deltas):
            return 0.
        positions = np.vstack((np.zeros(2), np.cumsum(deltas, axis=0)))
        return float(np.linalg.norm(positions[:, None]-positions[None, :], axis=-1).max())

    def _pending_row(self):
        pending = self._pending_prediction
        if pending is None or not self._external_reliable():
            return None
        previous = pending["previous"]
        if len(previous) != 2:
            return None
        moving, target = self.tracks["moving"], self.tracks["target"]
        return {"delta": pending["delta"], "predicted": pending["predicted"],
                "observed": (moving.position-previous["moving"])-(target.position-previous["target"]),
                "noise": float(np.hypot(self._noise(moving), self._noise(target)))}

    def _response_evidence(self, *, include_pending=False):
        rows = list(self._prediction_window)
        if include_pending:
            pending = self._pending_row()
            if pending is not None:
                rows = (rows+[pending])[-self.recent_window_size:]
        valid = [r for r in rows if r["predicted"] is not None and np.linalg.norm(r["delta"]) > .0001]
        measured = np.array([r["observed"] for r in valid]).reshape(-1, 2)
        predicted = np.array([r["predicted"] for r in valid]).reshape(-1, 2)
        threshold = max(.35, 3*max((r["noise"] for r in rows), default=0.))
        pred_extent = self._extent(predicted)
        residual_extent = self._extent(measured-predicted)
        error = float(np.sum((measured-predicted)**2))
        zero_error = float(np.sum(measured**2))
        # Distinguishability is required for BOTH the expected response and
        # its disagreement. A small response below noise does not prove that
        # the selected physical part is wrong or that robot motion was unsafe.
        contradicted = (len(valid) >= 3 and pred_extent > threshold
                        and residual_extent > threshold and error > zero_error)
        record = self._active_reference
        blocked = None if record is None else record["blocked_reason"]
        if not self._external_reliable():
            state, reason = "unavailable", "correspondence_lost_or_not_supplied"
        elif blocked == self.uncertainty_budget_reason:
            state, reason = "weakly_observable", "inconclusive_response_travel_allowance_exhausted"
        elif blocked:
            state, reason = ("contradicted" if blocked == self.contradiction_reason else "unvalidated"), blocked
        elif self._mapping_had_supported_response and contradicted:
            state, reason = "contradicted", "distinguishable_held_out_response_disagreement"
        elif self.response_ready:
            state, reason = "supported", "recent_measured_predictive_response"
        elif self._mapping_had_supported_response:
            state, reason = "weakly_observable", "insufficient_distinguishable_response_without_contradiction"
        else:
            state, reason = "unvalidated", "no_supported_response_in_current_mapping"
        used = 0. if record is None else record["unvalidated_travel_m"]
        return {"state": state, "reason": reason, "window_transitions": len(rows),
                "held_out_prediction_transitions": len(valid),
                "measured_displacement_px": self._extent([r["observed"] for r in rows]),
                "predicted_displacement_px": pred_extent,
                "predicted_transition_observed_displacement_px": self._extent(measured),
                "residual_displacement_px": residual_extent,
                "noise_threshold_px": threshold,
                "held_out_prediction_squared_error": error,
                "held_out_zero_motion_squared_error": zero_error,
                "mapping_had_supported_response": self._mapping_had_supported_response,
                "unvalidated_travel_remaining_m": max(0., self.reference_travel_limit_m-used),
                "clearance_certified": False, "semantic_identity_certified": False}

    def _block_reference(self, reason):
        evidence = self._response_evidence(include_pending=True)
        record = self._active_reference
        unresponsive = reason == "controlled_reference_unresponsive_change_correspondence"
        budget = reason == "controlled_reference_budget_exhausted_change_correspondence"
        if evidence["state"] == "weakly_observable" and (unresponsive or budget):
            record["v8_inconclusive_response"] = True
            # The allowance is the SAME measured spatial debt used in V7,
            # never renewed by another intent, retain or same-feature seed.
            if record["unvalidated_travel_m"] < self.reference_travel_limit_m-.001:
                self.status = "response_weakly_observable"
                return
            reason = self.uncertainty_budget_reason
        elif evidence["state"] == "contradicted" and not record["blocked_reason"]:
            reason = self.contradiction_reason
        record["v8_rejection_metrics"] = {key: evidence[key] for key in (
            "held_out_prediction_transitions", "measured_displacement_px",
            "predicted_displacement_px", "residual_displacement_px", "noise_threshold_px",
            "held_out_prediction_squared_error", "held_out_zero_motion_squared_error")}
        super()._block_reference(reason)
        self._reset_observability()

    def observe(self, images, state8, action):
        state = np.asarray(state8, float)
        delta = state[:3]-self.state[:3]
        domain_changed = (_orientation_distance(state[3:6], self.state[3:6]) > .035
                          or np.max(np.abs(state[6:8]-self.state[6:8])) > .002)
        if domain_changed:
            self._reset_observability()
        else:
            self._pending_prediction = {"delta": delta.copy(),
                "predicted": self.response.predict(delta),
                "previous": {k: self.tracks[k].position.copy() for k in ("moving", "target")
                             if self.tracks.get(k) is not None and self.tracks[k].reliable}}
        super().observe(images, state8, action)
        row = self._pending_row()
        if row is not None:
            self._prediction_window.append(row)
        elif not self._external_reliable():
            self._reset_observability()
        self._pending_prediction = None
        if self.response_ready:
            self._mapping_had_supported_response = True
        evidence = self._response_evidence()
        record = self._active_reference
        if record is not None and not record["blocked_reason"]:
            if evidence["state"] == "contradicted":
                self._block_reference(self.contradiction_reason)
            elif evidence["state"] == "weakly_observable":
                self.status = "response_weakly_observable"
        return self.summary()

    def control_limit(self):
        cap, reason = super().control_limit()
        if (reason == "controlled_reference_budget_exhausted_change_correspondence"
                and self._response_evidence()["state"] == "weakly_observable"):
            return cap, self.uncertainty_budget_reason
        return cap, reason

    def summary(self):
        result = super().summary()
        result["version"] = "role-aware-response-observability-v8"
        evidence = self._response_evidence()
        result["response_evidence"] = evidence
        if self._active_reference is not None and self._active_reference["blocked_reason"]:
            evidence["rejection_metrics"] = self._active_reference.get("v8_rejection_metrics")
        health = result["reference_health"]
        if evidence["state"] == "weakly_observable":
            blocked = self._active_reference and self._active_reference["blocked_reason"]
            health["status"] = blocked or "weakly_observable"
            health["repair_required"] = bool(blocked)
            health["recommended_action"] = (
                "The current image response is inconclusive, not evidence of a wrong physical landmark. "
                "Do not yaw, retreat or reseed solely to repair weak observability. "
                + ("The existing measured travel allowance is exhausted. Choose a visually justified useful task correction or terminal interaction; retain/rename/same-feature reseed does not renew it. "
                   if blocked else "Retain the correspondence and continue useful task motion within the remaining measured travel allowance, or make a visually justified task correction. ")
                + "No clearance, grasp or semantic identity is certified.")
            if not blocked:
                health["rejected_current_uv"] = None
        return result
