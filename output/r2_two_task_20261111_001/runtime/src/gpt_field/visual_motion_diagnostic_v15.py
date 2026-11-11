"""Passive recent RGB change evidence, never a robot segmentation or control gate.

Only acknowledged RGB/robot transitions enter the bounded history. Changed
pixels are temporal evidence, not proof of a distal tool, causal object motion,
attachment or clearance. In particular shadows, camera motion, background and
proximal links can all change. No feature is selected or promoted here.
"""
from __future__ import annotations

import base64
import copy
import io
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .observation import decode_image


@dataclass(frozen=True)
class _Frame:
    rgb: np.ndarray
    state: np.ndarray
    path: float
    sequence: int


class VisualMotionDiagnosticV15:
    version = "passive-recent-rgb-motion-evidence-v15"
    minimum_measured_travel_m = .004
    keyframe_spacing_m = .001
    maximum_frames = 8
    changed_pixel_threshold = 12
    interpretation = ("Raw image changes can include shadows, appearance, camera motion, "
        "background and proximal robot links. They do not identify the distal tool. "
        "Unchanged support can also have weak projection. Inspect both original views; "
        "no reference is automatically selected or certified.")

    def __init__(self):
        self._keyframes = deque(maxlen=self.maximum_frames-1)
        self.current = None
        self.measured_path_m = 0.
        self.transitions = 0
        self.history_resets = 0
        self.status = "not_configured"
        self._moving = {}
        self._support_mask = None
        self._tool_track = self._roi_source = self._roi = None
        self._declared_roi = None
        self._roi_state = "unavailable"
        self._configured_sequence = None
        self._cached_image = None

    @staticmethod
    def _data(observation):
        image, state = ((observation["observation/image"], observation["observation/state"])
            if isinstance(observation, dict) else (observation.image, observation.state))
        state = np.asarray(state, float)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Motion diagnostic needs finite measured state8")
        rgb = np.asarray(decode_image(image))
        return rgb, state

    def _frames(self):
        frames = list(self._keyframes)
        if self.current is not None and (not frames or frames[-1] is not self.current):
            frames.append(self.current)
        return frames

    def _baseline(self):
        if self.current is None:
            return None
        eligible = [f for f in self._keyframes if
            self.current.path-f.path >= self.minimum_measured_travel_m-1e-12]
        return eligible[-1] if eligible else None

    def _start(self, rgb, state, *, gap=False):
        self._keyframes.clear()
        self.current = _Frame(rgb.copy(), state.copy(), self.measured_path_m, self.transitions)
        self._keyframes.append(self.current)
        if gap:
            self.history_resets += 1
        self.status = "accumulating_acknowledged_motion"
        self._cached_image = None

    @staticmethod
    def _box(value):
        try:
            box = np.asarray(value, float)
        except (ValueError, TypeError):
            return None
        if (box.shape != (4,) or not np.isfinite(box).all() or
                np.any(box < 0) or np.any(box > 1) or np.any(box[:2] >= box[2:])):
            return None
        return box

    def _read_support(self, monitor):
        ext = (getattr(monitor, "refs", {}) or {}).get("external") or {}
        track = (getattr(monitor, "tracks", {}) or {}).get("moving")
        self._moving = {"identity_claim": ext.get("moving_id"), "moving_kind": ext.get("moving_kind"),
            "reliable": bool(track is not None and getattr(track, "reliable", False)),
            "tracked_point_px": None, "descriptor_support_bounds_px": None,
            "descriptor_feature_count": 0, "descriptor_pixel_count": 0,
            "support_is_robot_mask": False, "support_inside_declared_tool_roi": None}
        self._support_mask = None
        if track is None or self.current is None:
            return
        position = np.asarray(getattr(track, "position", []), float)
        if position.shape == (2,) and np.isfinite(position).all():
            self._moving["tracked_point_px"] = position.tolist()
        points = np.asarray(getattr(track, "points", []), float)
        if points.ndim != 2 or points.shape[1:] != (2,) or not len(points) or not np.isfinite(points).all():
            return
        h, w = self.current.rgb.shape[:2]
        mask = np.zeros((h, w), bool)
        for x, y in np.rint(points).astype(int):
            x0, x1, y0, y1 = max(0, x-3), min(w, x+4), max(0, y-3), min(h, y+4)
            if x0 < x1 and y0 < y1:
                mask[y0:y1, x0:x1] = True
        ys, xs = np.where(mask)
        self._moving["descriptor_feature_count"] = len(points)
        if len(xs):
            bounds = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            self._moving.update(descriptor_support_bounds_px=bounds, descriptor_pixel_count=int(mask.sum()))
            self._support_mask = mask
            if self._roi is not None:
                self._moving["support_inside_declared_tool_roi"] = bool(
                    np.all(np.asarray(bounds[:2]) >= self._roi[:2]) and
                    np.all(np.asarray(bounds[2:]) <= self._roi[2:]))

    def _read_roi(self, monitor):
        if self._declared_roi is None or self.current is None:
            self._roi, self._roi_state = None, "unavailable"
            return
        ext = (getattr(monitor, "refs", {}) or {}).get("external") or {}
        track = (getattr(monitor, "tracks", {}) or {}).get(
            "tool" if ext.get("moving_kind") == "payload" else "moving")
        if (track is self._tool_track and track is not None and getattr(track, "reliable", False)
                and self._roi_source is not None):
            points = self._roi_source@track.matrix+track.translation
            if np.isfinite(points).all():
                self._roi = np.r_[points.min(axis=0), points.max(axis=0)]
                self._roi_state = "uncertified_affine_transport_of_model_rectangle"
                return
        self._roi = self._declared_roi*np.tile(np.array(self.current.rgb.shape[1::-1])-1, 2)
        self._roi_state = ("current_model_declaration" if self.current.sequence == self._configured_sequence
                           else "last_declared_rectangle_not_currently_tracked")

    def configure(self, observation, monitor, motion_context):
        rgb, state = self._data(observation)
        if self.current is None:
            self._start(rgb, state)
        elif (not np.array_equal(rgb, self.current.rgb) or
              not np.allclose(state, self.current.state, atol=1e-9, rtol=0.)):
            # A missing acknowledged interval is not reconstructed or charged.
            self._start(rgb, state, gap=True)
        self._declared_roi = self._box((motion_context or {}).get("tool_external_roi_uv"))
        self._configured_sequence = self.current.sequence
        ext = (getattr(monitor, "refs", {}) or {}).get("external") or {}
        self._tool_track = (getattr(monitor, "tracks", {}) or {}).get(
            "tool" if ext.get("moving_kind") == "payload" else "moving")
        self._roi_source = None
        if self._declared_roi is not None and self._tool_track is not None:
            box = self._declared_roi*np.tile(np.array(rgb.shape[1::-1])-1, 2)
            corners = np.array([box[:2], [box[2], box[1]], box[2:], [box[0], box[3]]])
            try:
                self._roi_source = (corners-self._tool_track.translation)@np.linalg.inv(self._tool_track.matrix)
            except (AttributeError, ValueError, np.linalg.LinAlgError):
                pass
        self._read_roi(monitor)
        self._read_support(monitor)
        self._cached_image = None
        return self.summary()

    def observe(self, before, after, monitor):
        before_rgb, before_state = self._data(before)
        rgb, state = self._data(after)
        if (self.current is None or not np.array_equal(before_rgb, self.current.rgb) or
                not np.allclose(before_state, self.current.state, atol=1e-9, rtol=0.)):
            self.status = "unacknowledged_transition_gap"
            self._cached_image = None
            return self.summary()
        if rgb.shape != before_rgb.shape:
            self.status = "image_shape_changed_requires_configuration"
            self._cached_image = None
            return self.summary()
        self.measured_path_m += float(np.linalg.norm(state[:3]-before_state[:3]))
        self.transitions += 1
        self.current = _Frame(rgb.copy(), state.copy(), self.measured_path_m, self.transitions)
        if self.current.path-self._keyframes[-1].path >= self.keyframe_spacing_m-1e-12:
            self._keyframes.append(self.current)
        self.status = "raw_motion_pair_available" if self._baseline() else "accumulating_acknowledged_motion"
        self._read_roi(monitor)
        self._read_support(monitor)
        self._cached_image = None
        return self.summary()

    def _metrics(self):
        old = self._baseline()
        if old is None:
            return None, None
        # int16 subtraction prevents uint8 wraparound from inventing changes.
        difference = np.abs(self.current.rgb.astype(np.int16)-old.rgb.astype(np.int16))
        changed = difference.max(axis=2) > self.changed_pixel_threshold
        roi_mask = np.zeros(changed.shape, bool)
        if self._roi is not None:
            h, w = changed.shape
            x0, y0 = np.maximum(np.floor(self._roi[:2]).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(self._roi[2:]).astype(int)+1, [w, h])
            if x0 < x1 and y0 < y1:
                roi_mask[y0:y1, x0:x1] = True
        support = self._support_mask
        stats = {"units": "uint8_levels_0_to_255", "changed_pixel_threshold": self.changed_pixel_threshold,
            "global_changed_pixel_count": int(changed.sum()),
            "global_changed_pixel_fraction": float(changed.mean()),
            "within_declared_tool_roi_changed_pixels": int((changed & roi_mask).sum()),
            "outside_declared_tool_roi_changed_pixels": int((changed & ~roi_mask).sum()),
            "current_descriptor_support_mean_absolute_change": None if support is None else float(difference[support].mean()),
            "current_descriptor_support_changed_pixel_count": None if support is None else int(changed[support].sum()),
            "support_comparison": "fixed CURRENT descriptor pixels in both actual rasters; not tracked optical-flow identity"}
        return changed, stats

    @staticmethod
    def _orientation_change(before, after):
        def quaternion(rotvec):
            angle = float(np.linalg.norm(rotvec))
            scale = np.sin(angle/2)/angle if angle > 1e-15 else .5
            return np.r_[np.asarray(rotvec)*scale, np.cos(angle/2)]
        dot = abs(float(quaternion(before)@quaternion(after)))
        return float(2*np.arccos(np.clip(dot, 0., 1.)))

    def summary(self):
        old = self._baseline()
        available = old is not None and self.status == "raw_motion_pair_available"
        _, stats = self._metrics() if available else (None, None)
        pair = None
        if available:
            delta = self.current.state[:3]-old.state[:3]
            length = float(np.linalg.norm(delta))
            pair = {"prior_transition": old.sequence, "current_transition": self.current.sequence,
                "acknowledged_path_m": self.current.path-old.path, "net_delta_xyz_m": delta.tolist(),
                "net_displacement_m": length,
                "net_direction_xyz": (delta/length).tolist() if length > 1e-9 else None,
                "orientation_change_rad": self._orientation_change(old.state[3:6], self.current.state[3:6]),
                "finger_state_change": (self.current.state[6:8]-old.state[6:8]).tolist(),
                "frames_in_window": sum(f.sequence >= old.sequence for f in self._frames())}
        return {"version": self.version, "status": self.status, "diagnostic_available": available,
            "minimum_acknowledged_path_m": self.minimum_measured_travel_m,
            "retained_frame_count": len(self._frames()), "maximum_retained_frames": self.maximum_frames,
            "keyframe_spacing_m": self.keyframe_spacing_m, "measured_path_m": self.measured_path_m,
            "acknowledged_transitions": self.transitions, "history_resets_for_unobserved_gap": self.history_resets,
            "pair": pair, "raw_rgb_change": stats, "current_moving_reference": copy.deepcopy(self._moving),
            "current_descriptor_support_bounds_px": self._moving.get("descriptor_support_bounds_px"),
            "whole_tool_roi": {"declared_uv": None if self._declared_roi is None else self._declared_roi.tolist(),
                "display_bounds_px": None if self._roi is None else self._roi.tolist(), "status": self._roi_state,
                "is_segmentation": False, "transport_is_identity_certificate": False},
            "interpretation": self.interpretation, "source": "acknowledged_external_RGB_and_measured_robot_state",
            "automatic_candidate_selection": False, "automatic_reference_promotion": False,
            "execution_veto": False, "control_debt_modified": False,
            "semantic_correspondence_certified": False, "attachment_certified": False,
            "clearance_certified": False, "environment_inputs": []}

    def image(self, current_observation=None):
        """One derived image for an existing handoff; never trigger a request."""
        report = self.summary()
        if not report["diagnostic_available"]:
            return None
        if current_observation is not None:
            rgb, state = self._data(current_observation)
            if (not np.array_equal(rgb, self.current.rgb) or
                    not np.allclose(state, self.current.state, atol=1e-9, rtol=0.)):
                return None
        if self._cached_image is not None:
            return self._cached_image
        old = self._baseline()
        changed, _ = self._metrics()
        interior = changed.copy()
        interior[1:] &= changed[:-1]
        interior[:-1] &= changed[1:]
        interior[:, 1:] &= changed[:, :-1]
        interior[:, :-1] &= changed[:, 1:]
        boundary = changed & ~interior
        marked = self.current.rgb.copy()
        marked[boundary] = (255, 210, 20)
        annotated = Image.fromarray(marked)
        draw = ImageDraw.Draw(annotated)
        if self._roi is not None:
            draw.rectangle(tuple(self._roi), outline=(30, 220, 255), width=1)
        bounds = self._moving.get("descriptor_support_bounds_px")
        if bounds is not None:
            draw.rectangle(tuple(bounds), outline=(255, 60, 220), width=1)
        point = self._moving.get("tracked_point_px")
        if point is not None:
            x, y = point
            draw.line((x-4, y, x+4, y), fill=(255, 60, 220), width=1)
            draw.line((x, y-4, x, y+4), fill=(255, 60, 220), width=1)
            draw.text((min(annotated.width-9, x+5), max(0, y-10)), "M", fill=(255, 60, 220))
        w, h = annotated.size
        canvas = Image.new("RGB", (w*3, h+98), (15, 18, 22))
        for x, image, label in ((0, Image.fromarray(old.rgb), "PRIOR raw RGB"),
                                (w, Image.fromarray(self.current.rgb), "CURRENT raw RGB"),
                                (w*2, annotated, "CURRENT + change outlines")):
            canvas.paste(image, (x, 18))
            ImageDraw.Draw(canvas).text((x+4, 3), label, fill=(245, 245, 245))
        pair = report["pair"]
        lines = [f"Measured path {pair['acknowledged_path_m']*1000:.2f} mm; net XYZ (mm): "
                 +", ".join(f"{x*1000:+.2f}" for x in pair["net_delta_xyz_m"]),
            "Yellow: raw changed regions, including old/new footprints; not object identity.",
            "Magenta M/box: current claimed moving reference + actual descriptor support.",
            "Cyan: model whole-tool ROI / tracker transport claim; may be inconsistent.",
            "Changes may be shadows, background, proximal arm or camera motion. NO semantic/clearance proof."]
        draw = ImageDraw.Draw(canvas)
        for i, line in enumerate(lines):
            draw.text((4, h+22+14*i), line, fill=(245, 245, 245))
        stream = io.BytesIO()
        canvas.resize((canvas.width*2, canvas.height*2), Image.Resampling.NEAREST).save(stream, format="PNG")
        self._cached_image = "data:image/png;base64,"+base64.b64encode(stream.getvalue()).decode("ascii")
        return self._cached_image
