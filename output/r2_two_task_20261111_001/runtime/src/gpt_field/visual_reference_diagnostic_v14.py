"""Passive raw-RGB support diagnostics for a declared moving tool reference.

Numerical tracking reliability is not evidence that a patch belongs to the
controlled tool. This diagnostic compares actual pixels while acknowledged EEF
travel accumulates. Unchanged pixels can also result from weak projection, so
stasis requests a review; it never rejects motion, assigns semantic identity,
switches references, changes travel allowances, or establishes attachment.
"""
from __future__ import annotations

import copy
import numpy as np

from .observation import decode_image


class MovingReferenceDiagnosticV14:
    version = "passive-moving-reference-support-v14"
    minimum_measured_travel_m = .004
    maximum_mean_rgb_change = 1.5
    changed_pixel_threshold = 6.
    maximum_changed_pixel_fraction = .05

    def __init__(self):
        self.track = None
        self.status = "not_configured"
        self._pixels = self._mask = self._state = None
        self.image_shape = None
        self.support_bounds_px = None
        self.support_pixels = self.features = self.transitions = 0
        self.measured_travel_m = 0.
        self.net_delta = np.zeros(3)
        self.origin_xyz = None
        self.max_mean_rgb_change = self.max_changed_fraction = self.max_rgb_change = 0.
        self.last_mean_rgb_change = self.last_changed_fraction = self.last_rgb_change = None
        self.identity = None
        self.last_update = None

    @staticmethod
    def _data(observation):
        if isinstance(observation, dict):
            image = observation["observation/image"]
            state = observation["observation/state"]
        else:
            image, state = observation.image, observation.state
        state = np.asarray(state, float)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Reference diagnostic needs finite measured state8")
        rgb = np.asarray(decode_image(image))
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Reference diagnostic needs RGB uint8")
        return rgb, state

    @staticmethod
    def _moving_tool(monitor):
        external = (getattr(monitor, "refs", {}) or {}).get("external") or {}
        if external.get("moving_kind") != "tool":
            return None, "not_a_moving_tool_reference"
        track = (getattr(monitor, "tracks", {}) or {}).get("moving")
        if track is None or not getattr(track, "reliable", False):
            return None, "moving_tool_correspondence_unavailable"
        return track, None

    def _unavailable(self, reason):
        self.status = reason
        self.last_update = reason
        return self.summary()

    def configure(self, observation, monitor):
        track, reason = self._moving_tool(monitor)
        if reason:
            return self._unavailable(reason)
        rgb, state = self._data(observation)
        if track is self.track and self._pixels is not None:
            # Retaining a reference or revising its whole-tool envelope does
            # not replace its evidence baseline. This stores no control debt.
            if rgb.shape != self.image_shape:
                return self._unavailable("reference_support_image_shape_changed")
            if not np.allclose(state, self._state, atol=1e-9, rtol=0.):
                return self._unavailable("reference_diagnostic_transition_gap")
            self.last_update = "same_correspondence_retained"
            return self.summary()
        points = np.asarray(getattr(track, "points", []), float)
        if points.ndim != 2 or points.shape[1:] != (2,) or len(points) < 3 or not np.isfinite(points).all():
            return self._unavailable("reference_descriptor_support_unavailable")
        # Existing tracker descriptors are 7x7. Audit their finite CURRENT
        # support, not an invented robot mask or an expanded whole-scene ROI.
        mask = np.zeros(rgb.shape[:2], bool)
        for x, y in np.rint(points).astype(int):
            if x < 3 or y < 3 or x+3 >= rgb.shape[1] or y+3 >= rgb.shape[0]:
                return self._unavailable("reference_descriptor_support_outside_view")
            mask[y-3:y+4, x-3:x+4] = True
        rows, cols = np.where(mask)
        self.track, self.identity = track, getattr(track, "ident", None)
        self._mask, self._pixels, self._state = mask, rgb[mask].copy(), state.copy()
        self.image_shape = rgb.shape
        self.support_bounds_px = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
        self.support_pixels, self.features = int(mask.sum()), len(points)
        self.origin_xyz = state[:3].copy()
        self.measured_travel_m, self.net_delta, self.transitions = 0., np.zeros(3), 0
        self.max_mean_rgb_change = self.max_changed_fraction = self.max_rgb_change = 0.
        self.last_mean_rgb_change = self.last_changed_fraction = self.last_rgb_change = None
        self.status, self.last_update = "accumulating_motion_evidence", "new_correspondence_baseline"
        return self.summary()

    def observe(self, before, after, monitor):
        track, reason = self._moving_tool(monitor)
        if reason:
            return self._unavailable(reason)
        if track is not self.track or self._pixels is None:
            return self._unavailable("reference_correspondence_changed_requires_configuration")
        before_rgb, before_state = self._data(before)
        rgb, state = self._data(after)
        if rgb.shape != self.image_shape or before_rgb.shape != self.image_shape:
            return self._unavailable("reference_support_image_shape_changed")
        if not np.allclose(before_state, self._state, atol=1e-9, rtol=0.):
            return self._unavailable("reference_diagnostic_transition_gap")
        difference = np.abs(rgb[self._mask].astype(float)-self._pixels)
        mean = float(difference.mean())
        changed = float(np.mean(np.max(difference, axis=1) > self.changed_pixel_threshold))
        maximum = float(difference.max(initial=0.))
        self.last_mean_rgb_change, self.last_changed_fraction, self.last_rgb_change = mean, changed, maximum
        # Preserve any earlier support change, including out-and-back motion.
        # Memory stays constant; old movement is never mistaken for stasis just
        # because the latest raster happens to resemble its seed frame again.
        self.max_mean_rgb_change = max(self.max_mean_rgb_change, mean)
        self.max_changed_fraction = max(self.max_changed_fraction, changed)
        self.max_rgb_change = max(self.max_rgb_change, maximum)
        self.measured_travel_m += float(np.linalg.norm(state[:3]-before_state[:3]))
        self.net_delta = state[:3]-self.origin_xyz
        self.transitions += 1
        self._state = state.copy()
        static = (self.max_mean_rgb_change <= self.maximum_mean_rgb_change
                  and self.max_changed_fraction <= self.maximum_changed_pixel_fraction)
        if self.measured_travel_m < self.minimum_measured_travel_m-1e-12:
            self.status = "accumulating_motion_evidence"
        else:
            self.status = "reference_support_stasis" if static else "raw_support_changed"
        self.last_update = "acknowledged_rgb_robot_transition"
        return self.summary()

    def summary(self):
        review = self.status == "reference_support_stasis"
        return {"version": self.version, "status": self.status,
            "review_recommended": review, "identity_claim": self.identity,
            "interpretation": "inconclusive" if review else "passive_observation",
            "recommended_action": (
                "Re-examine whether the current support is a visible distal moving tool part. "
                "If needed, repair its correspondence from current RGB, or request an explicit "
                "different nominal-clear Field separation/observation direction within the "
                "existing shared allowance. Stasis alone does not establish wrong identity; "
                "weak projection is possible. Retain/rename must not renew control debt."
                if review else None),
            "measured_travel_m": self.measured_travel_m,
            "net_measured_delta_m": self.net_delta.tolist(), "transitions": self.transitions,
            "review_after_measured_travel_m": self.minimum_measured_travel_m,
            "support": {"kind": "fixed_rgb_union_of_existing_7x7_descriptor_footprints",
                "bounds_px": copy.deepcopy(self.support_bounds_px), "pixel_count": self.support_pixels,
                "initial_feature_count": self.features, "is_robot_segmentation_mask": False},
            "raw_rgb_change": {"units": "uint8_levels_0_to_255", "latest_mean_absolute": self.last_mean_rgb_change,
                "maximum_mean_absolute": self.max_mean_rgb_change, "latest_maximum_absolute": self.last_rgb_change,
                "maximum_absolute": self.max_rgb_change, "latest_changed_pixel_fraction": self.last_changed_fraction,
                "maximum_changed_pixel_fraction": self.max_changed_fraction,
                "changed_pixel_threshold": self.changed_pixel_threshold,
                "near_unchanged_mean_limit": self.maximum_mean_rgb_change,
                "near_unchanged_fraction_limit": self.maximum_changed_pixel_fraction},
            "last_update": self.last_update, "execution_veto": False,
            "automatic_reference_switch": False, "control_debt_modified": False,
            "semantic_correspondence_certified": False, "attachment_certified": False,
            "clearance_certified": False, "environment_inputs": []}
