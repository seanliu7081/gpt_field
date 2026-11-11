"""Displacement-based causal RGB evidence; measured travel stays in metres.

Small control frames need not each exceed image noise. Predictions are made
before their observations enter the fit, then evaluated over a recent moving
window. This is correspondence evidence, not obstacle or attachment evidence.
"""
from __future__ import annotations

import numpy as np

from .visual_monitor_v10 import RoleAwareVisualMonitorV10


class RoleAwareVisualMonitorV12(RoleAwareVisualMonitorV10):
    recent_window_size = 64
    reference_travel_limit_m = .030

    def cumulative_response_evidence(self, *, include_pending=False):
        rows = list(getattr(self, "_prediction_window", []))
        if include_pending:
            pending = self._pending_row()
            if pending is not None:
                rows = (rows+[pending])[-self.recent_window_size:]
        # Use the shortest recent suffix with a distinguishable displacement.
        # Old good motion cannot hide a newly stationary/wrong correspondence.
        selected = []
        for row in reversed(rows):
            selected.insert(0, row)
            valid = [r for r in selected if r["predicted"] is not None
                     and np.linalg.norm(r["delta"]) > .00005]
            # Endpoint evidence must resolve at least one image pixel; several
            # identical raster frames can precede a legitimate quantized change.
            threshold = 1.+3*max((r["noise"] for r in selected), default=0.)
            predicted = np.sum([r["predicted"] for r in valid], axis=0) if valid else np.zeros(2)
            observed = np.sum([r["observed"] for r in valid], axis=0) if valid else np.zeros(2)
            if (len(valid) >= 3 and sum(np.linalg.norm(r["delta"]) for r in valid) >= .002
                    and max(np.linalg.norm(predicted), np.linalg.norm(observed)) >= threshold):
                break
        valid = [r for r in selected if r["predicted"] is not None
                 and np.linalg.norm(r["delta"]) > .00005]
        threshold = 1.+3*max((r["noise"] for r in selected), default=0.)
        predicted = np.sum([r["predicted"] for r in valid], axis=0) if valid else np.zeros(2)
        observed = np.sum([r["observed"] for r in valid], axis=0) if valid else np.zeros(2)
        measured = float(sum(np.linalg.norm(r["delta"]) for r in valid))
        residual = float(np.linalg.norm(predicted-observed))
        zero = float(np.dot(observed, observed))
        ratio = None if zero <= 1e-12 else residual**2/zero
        enough = len(valid) >= 3 and measured >= .002
        supported = (enough and np.linalg.norm(observed) >= threshold
                     and np.linalg.norm(predicted) >= threshold*.5
                     and ratio is not None and ratio < .75 and residual <= threshold)
        contradicted = (enough and np.linalg.norm(predicted) >= threshold
                        and residual > threshold and residual**2 > zero)
        return {"supported": bool(supported), "contradicted": bool(contradicted),
                "transitions": len(selected), "held_out_transitions": len(valid),
                "measured_path_m": measured, "predicted_displacement_px": predicted.tolist(),
                "observed_displacement_px": observed.tolist(), "noise_threshold_px": threshold,
                "residual_px": residual, "squared_error_to_zero_ratio": ratio,
                "causal_predictions": True, "clearance_certified": False}

    @property
    def response_ready(self):
        record = getattr(self, "_active_reference", None)
        if record and record["blocked_reason"]:
            return False
        evidence = self.cumulative_response_evidence(include_pending=True)
        if evidence["contradicted"] and getattr(self, "_mapping_had_supported_response", False):
            return False
        return bool(super().response_ready or (self._external_reliable()
            and self.response.matrix is not None and evidence["supported"]))

    def _response_evidence(self, *, include_pending=False):
        evidence = super()._response_evidence(include_pending=include_pending)
        record = getattr(self, "_active_reference", None)
        if not self._external_reliable() or (record and record["blocked_reason"]):
            return evidence
        cumulative = self.cumulative_response_evidence(include_pending=include_pending)
        # Per-frame raster jitter may dominate squared error even when causal
        # endpoint predictions are correct. Classify contradiction only after
        # recent motion resolves one pixel PLUS the endpoint noise allowance.
        if self._mapping_had_supported_response and cumulative["contradicted"]:
            evidence.update(state="contradicted", reason="resolved_cumulative_response_disagreement")
        elif self.response_ready:
            evidence.update(state="supported", reason="recent_measured_predictive_response")
        else:
            evidence.update(state="weakly_observable", reason="cumulative_response_not_yet_resolved")
        evidence["cumulative_response"] = cumulative
        return evidence

    def observe(self, images, state8, action):
        super().observe(images, state8, action)
        evidence = self.cumulative_response_evidence()
        record = self._active_reference
        if (record and not record["blocked_reason"] and self._mapping_had_supported_response
                and evidence["contradicted"]):
            self._block_reference(self.contradiction_reason)
        return self.summary()

    def measured_control_limit(self):
        """Return a remaining MEASURED travel allowance, not a command cap."""
        record = self._active_reference
        if record and record["blocked_reason"]:
            return 0., record["blocked_reason"]
        if self.missing >= 2 or not self._external_reliable():
            return 0., "external_reference_unavailable"
        if self.status in ("visual_response_contradiction", "target_anchor_shifted"):
            return 0., self.status
        if self.response_ready:
            return None, None
        used = 0. if record is None else record["unvalidated_travel_m"]
        remaining = self.reference_travel_limit_m-used
        if remaining <= .001:
            return 0., self.uncertainty_budget_reason
        return remaining, None

    def summary(self):
        result = super().summary()
        result.update(version="role-aware-cumulative-response-v12",
                      cumulative_response=self.cumulative_response_evidence(),
                      reference_travel_limit_m=self.reference_travel_limit_m)
        return result
