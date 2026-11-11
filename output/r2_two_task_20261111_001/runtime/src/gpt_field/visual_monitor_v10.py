"""Displacement-adaptive RGB attachment and finite inconclusive cold starts.

All evidence is from independent image correspondences and measured robot
motion. No object pose, depth, contact signal or calibrated camera is consumed.
"""
from __future__ import annotations

from collections import deque
import copy
import numpy as np

from .visual_monitor_v6 import WindowedAttachmentEvidence
from .visual_monitor_v5 import RoleAwareVisualMonitorV5
from .visual_monitor_v8 import RoleAwareVisualMonitorV8


class AdaptiveAttachmentEvidence(WindowedAttachmentEvidence):
    """Accumulate fresh displacement, not a fixed number of control frames.

    A successful window is consumed exactly once. Later unresolved motion may
    use only a finite allowance until another *fresh* window supports co-motion.
    Its 5--15 mm bound is set from the last observed pixel/robot response, never
    enlarged while evidence is uncertain. Stationary frames, requests and
    labels create no evidence.
    The finite history and command floor cannot enlarge that spatial allowance.
    """

    window_size = 64
    minimum_useful_command_m = .001
    minimum_uncertain_travel_m = .005
    maximum_uncertain_travel_m = .015
    handoff_reason = "attachment_observation_budget_below_useful_motion_floor"

    def __init__(self):
        super().__init__()
        self.window = deque(maxlen=self.window_size)
        self.last_rejection_metrics = None
        self.uncertain_travel_limit_m = self.minimum_uncertain_travel_m
        self.allowance_evidence = None

    def reset(self, state="unobserved", reason="new_correspondence"):
        super().reset(state, reason)
        self.last_rejection_metrics = None
        self.uncertain_travel_limit_m = self.minimum_uncertain_travel_m
        self.allowance_evidence = None

    def _reject(self, reason):
        metrics = copy.deepcopy(self.metrics)
        self.reset("contradicted", reason)
        self.last_rejection_metrics = metrics

    def observe(self, payload_delta, tool_delta, anchor_delta, *, measured_distance_m,
                separation_px, noise_px=(.08, .08, .08), reliable=True,
                distinct_features=True, opening=False, finger_change_m=0.,
                rotation_rad=0.):
        distance = float(measured_distance_m)
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("Measured attachment travel must be finite and nonnegative")
        if opening:
            self.reset("released", "open_command")
            return
        if finger_change_m > .002:
            self.reset("unobserved", "finger_configuration_changed")
            return
        if not reliable or any(x is None for x in (payload_delta, tool_delta, anchor_delta)):
            self.reset("correspondence_lost", "payload_tool_or_anchor_correspondence_lost")
            return
        if not distinct_features or separation_px < 8:
            self._reject("payload_and_tool_features_not_independent")
            return
        if rotation_rad > .035:
            self.reset("unobserved", "orientation_domain_changed")
            return
        payload, tool, anchor = [np.asarray(x, float) for x in
                                 (payload_delta, tool_delta, anchor_delta)]
        noise = np.maximum(np.asarray(noise_px, float), .08)
        if (any(x.shape != (2,) or not np.isfinite(x).all() for x in (payload, tool, anchor))
                or noise.shape != (3,) or not np.isfinite(noise).all()):
            raise ValueError("Attachment image deltas/noise must be finite vectors")
        pair_noise = float(np.sqrt(2*(noise[0]**2+noise[1]**2)))
        if np.linalg.norm(payload-tool) > max(2.5, 5*pair_noise):
            self._reject("abrupt_payload_tool_differential_motion")
            return
        self.window.append((payload-anchor, tool-anchor, distance, noise, anchor))
        moving = np.sum([r[0] for r in self.window], axis=0)
        controlled = np.sum([r[1] for r in self.window], axis=0)
        anchor_motion = np.sum([r[4] for r in self.window], axis=0)
        moving_norm, controlled_norm = map(float, (np.linalg.norm(moving), np.linalg.norm(controlled)))
        residual = float(np.linalg.norm(moving-controlled))
        window_noise = np.max([r[3] for r in self.window], axis=0)
        motion_noise = float(np.sqrt(2*(max(window_noise[:2])**2+window_noise[2]**2)))
        pair_noise = float(np.sqrt(2*(window_noise[0]**2+window_noise[1]**2)))
        resolved = max(.75, 3*motion_noise)
        measured = float(sum(r[2] for r in self.window))
        moving_transitions = sum(r[2] > 1e-7 for r in self.window)
        cosine = float(np.dot(moving, controlled)/max(1e-12, moving_norm*controlled_norm))
        self.metrics = {"window_transitions": len(self.window), "moving_transitions": moving_transitions,
                        "window_measured_path_m": measured,
                        "payload_motion_relative_anchor_px": moving.tolist(),
                        "tool_motion_relative_anchor_px": controlled.tolist(),
                        "window_anchor_motion_px": float(np.linalg.norm(anchor_motion)),
                        "payload_motion_px": moving_norm, "tool_motion_px": controlled_norm,
                        "payload_tool_differential_px": residual, "motion_threshold_px": resolved,
                        "pair_noise_px": pair_noise, "motion_cosine": cosine,
                        "evidence_window": "fresh_since_last_resolved_co_motion",
                        "history_limit_transitions": self.window_size}
        if np.linalg.norm(anchor_motion) > max(1., 3*np.sqrt(2)*window_noise[2]):
            self._uncertain(distance, "stationary_anchor_motion_uncertain")
            return
        shorter, longer = min(moving_norm, controlled_norm), max(moving_norm, controlled_norm)
        if len(self.window) >= 3 and longer >= resolved:
            if (shorter < .25*longer and residual > max(.7, 3*pair_noise)
                    or cosine < .2 and shorter >= resolved
                    or residual > max(1.2, 4*pair_noise, .5*shorter)):
                self._reject("adaptive_payload_tool_motion_disagrees")
                return
        supported = (moving_transitions >= 3 and measured >= .001
                     and shorter >= resolved and cosine >= .85
                     and residual <= max(.7, 3*pair_noise, .35*shorter))
        if supported:
            self.samples += 1
            self.state, self.reason = "observed", "adaptive_measured_co_motion"
            self.had_observed_evidence = True
            self.uncertain_travel_m = 0.
            self.last_rejection_metrics = None
            # A fixed 5 mm allowance starved the recorded V9 carry: the pair
            # needed 6--9 mm to resolve ~1.7 pixels. Only an ACTUAL fresh
            # successful window sets this estimate; uncertainty, new intents,
            # labels and increasing noise cannot grow it. It is a finite
            # observation allowance, not a physical attachment guarantee.
            gain = shorter/measured
            requested = 1.5*resolved/gain+self.minimum_useful_command_m
            self.uncertain_travel_limit_m = float(np.clip(requested,
                self.minimum_uncertain_travel_m, self.maximum_uncertain_travel_m))
            self.allowance_evidence = {"source": "last_fresh_observed_co_motion_window",
                "smaller_image_gain_px_per_m": gain, "motion_threshold_px": resolved,
                "measured_path_m": measured, "formula": "clip(1.5*threshold/gain+0.001,0.005,0.015)",
                "fixed_until_new_observed_window": True}
            # Do not keep old good motion in a window used to test later slip.
            self.window.clear()
        elif distance < .0001 and longer < resolved and self.observed:
            self.reason = "stationary_correspondence_after_observed_motion"
        elif self.had_observed_evidence:
            self._uncertain(distance, "accumulating_fresh_displacement_evidence")
        else:
            self.state, self.reason = "unobserved", "awaiting_measured_co_motion"

    def control_limit(self):
        if self.observed:
            return None, None
        if self.state == "uncertain" and self.had_observed_evidence:
            remaining = self.uncertain_travel_limit_m-self.uncertain_travel_m
            if remaining <= 0:
                return 0., "uncertain_attachment_travel_exhausted"
            if remaining < self.minimum_useful_command_m:
                return 0., self.handoff_reason
            return min(.003, remaining), None
        return 0., "payload_attachment_"+self.state

    def summary(self):
        result = super().summary()
        result.update(evidence_method="displacement_adaptive_fresh_windows",
            max_history_transitions=self.window_size,
            minimum_useful_command_m=self.minimum_useful_command_m,
            accumulator_transitions=len(self.window),
            uncertain_travel_bounds_m=[self.minimum_uncertain_travel_m, self.maximum_uncertain_travel_m],
            allowance_evidence=copy.deepcopy(self.allowance_evidence),
            rejection_metrics=copy.deepcopy(self.last_rejection_metrics))
        return result


class RoleAwareVisualMonitorV10(RoleAwareVisualMonitorV8):
    """V8 correspondences, adaptive attachment, and honest cold-start ambiguity."""

    def __init__(self):
        super().__init__()
        self.attachment = AdaptiveAttachmentEvidence()
        self._adaptive_recovery_supported = 0

    def configure(self, images, references, state8, *, intent_id=None):
        epoch = self.epoch
        super().configure(images, references, state8, intent_id=intent_id)
        if epoch != self.epoch:
            self._adaptive_recovery_supported = 0
        return self.summary()

    def _block_reference(self, reason):
        was_blocked = self._active_reference and self._active_reference["blocked_reason"]
        super()._block_reference(reason)
        if not was_blocked and self._active_reference and self._active_reference["blocked_reason"]:
            self._adaptive_recovery_supported = 0

    def observe(self, images, state8, action):
        super().observe(images, state8, action)
        record = self._active_reference
        payload = (self.refs.get("external") or {}).get("moving_kind") == "payload"
        if record and record["blocked_reason"] and payload:
            row = self._recovery_motion[-1] if self._recovery_motion else None
            # V7 already resets the predictor and held-out ledger at rejection.
            # Its original payload rescue required three consecutively observed
            # windows, incompatible with consuming fresh adaptive windows.
            # Finite previously observed pair support may bridge those windows;
            # every recovery transition still needs NEW predictive image motion.
            supported = (row is not None and np.linalg.norm(row[0]) > .0001
                and np.linalg.norm(row[1]) > row[2]
                and self._recent_motion_supported(self._recovery_motion)
                and RoleAwareVisualMonitorV5.response_ready.fget(self)
                and self.attachment.had_observed_evidence
                and self.attachment.control_limit()[1] is None)
            self._adaptive_recovery_supported = self._adaptive_recovery_supported+1 if supported else 0
            if self._adaptive_recovery_supported >= 3:
                record["blocked_reason"] = None
                record["recoveries"] = record.get("recoveries", 0)+1
                self._recovery_supported_transitions = self._adaptive_recovery_supported
                if self.response_ready:
                    record["unvalidated_travel_m"] = 0.
                    record["validated_transitions"] += 1
                    self._mapping_had_supported_response = True
                    self.status = "response_validated"
        return self.summary()

    def _response_evidence(self, *, include_pending=False):
        evidence = super()._response_evidence(include_pending=include_pending)
        record = self._active_reference
        if (evidence["state"] == "unvalidated" and self._external_reliable()
                and not (record and record["blocked_reason"])):
            # No validated response model supplies a distinguishable expected
            # image motion at cold start. A stationary-looking patch alone is
            # insufficient to call its physical identity wrong at 6 mm.
            # V8's existing hook still charges the SAME persistent 15 mm debt.
            evidence.update(state="weakly_observable", reason="cold_start_response_inconclusive")
        return evidence

    def summary(self):
        result = super().summary()
        result["version"] = "role-aware-adaptive-displacement-evidence-v10"
        result["attachment"]["decision_contract"] = (
            "Fresh displacement must support independent payload/tool motion relative to a stable anchor. "
            "Weak motion after support retains only a finite measured-travel allowance; a new request or "
            "stationary hold does not renew it. A sub-mm remaining allowance requests a new task decision, "
            "never a larger command. Co-motion is not physical attachment certification.")
        return result
