"""Local RGB consistency of declared world boxes, never a geometry certificate.

Only the synchronized public external RGB and known fixed-camera calibration are
read. A reference is installed explicitly when the model supplies geometry; a
matching image never advances that reference or repairs an invalidated shape.
An accepted independent RGB geometry update may explicitly replace its reference.
No depth, segmentation, simulator truth, optical recentering or model call is
used. Projection and texture thresholds describe image observability only.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from collections.abc import Mapping

import cv2
import numpy as np

VISUAL_WATCH_VERSION = "agent-speedup-v3-visual-watch-7"
_SIGNS = np.asarray([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)


@dataclass(eq=False)
class _Frame:
    rgb: np.ndarray
    texture: np.ndarray
    calibration: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    focal: np.ndarray
    principal: np.ndarray
    scale: np.ndarray


@dataclass
class _Reference:
    frame: _Frame
    mask: np.ndarray
    occlusion: np.ndarray
    signature: tuple
    step: int
    usable: bool
    reason: str


def _array(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("invalid finite vector or matrix")
    return result


class VisualGeometryWatch:
    """Small cached fixed-camera appearance checks against model-update images.

    ``world_boxes`` and ``occluder_boxes`` use the geometry memory's OBB shape:
    ``center`` (3), ``rotation`` (3x3), ``half`` (3), optional ``uncertainty``.
    The caller supplies measured hand and estimated held-payload occluders in
    world coordinates. Masking is deliberately conservative, irrespective of
    relative depth: an excluded pixel cannot establish object visibility.
    """

    def __init__(self, *, max_edge=160, min_pixels=48, min_visible_fraction=.60):
        if not 32 <= int(max_edge) <= 512 or int(min_pixels) < 8 or not 0 < min_visible_fraction <= 1:
            raise ValueError("invalid visual-watch image limits")
        self.max_edge = int(max_edge)
        self.min_pixels = int(min_pixels)
        self.min_visible_fraction = float(min_visible_fraction)
        self._references = {}
        self._frames = OrderedDict()
        self._difference_cache = {}
        self._difference_frame = None
        self._projection_cache = OrderedDict()
        self.compute_counts = {"projections": 0, "projection_cache_hits": 0}

    def forget(self, ids):
        if isinstance(ids, str):
            ids = [ids]
        for ident in ids:
            self._references.pop(ident, None)

    def _evidence(self, ident, step, status, reason, **metrics):
        return {"version": VISUAL_WATCH_VERSION, "entity_id": ident,
                "observation_step": int(step), "status": status, "reason": reason,
                "source": "public_fixed_external_rgb_local_appearance",
                "reference_policy": "explicit_geometry_evidence_update_only",
                "signature_policy": "nominal_pose_and_shape_excludes_risk_padding",
                "monitoring_available": status in ("reference", "consistent", "changed"),
                "self_occluded": reason == "expected_self_occlusion",
                "geometry_verified": False, "free_space_verified": False,
                "contact_verified": False, "unmodeled_obstacle_detection_complete": False,
                "computation_policy": "reuse_exact_calibration_and_shape_projection; compare_current_rgb",
                "observation_age_refreshed": False, "compute_counts": dict(self.compute_counts),
                **metrics}

    def _frame(self, view):
        # StepView is immutable. Holding the object prevents id reuse; two live
        # cached views also cover common before/after calls without re-decoding.
        key = id(view)
        if key in self._frames and self._frames[key][0] is view:
            self._frames.move_to_end(key)
            return self._frames[key][1]
        result = self._read_frame(view)
        self._frames[key] = (view, result)
        while len(self._frames) > 2:
            self._frames.popitem(last=False)
        return result

    def _read_frame(self, view):
        geometry = getattr(getattr(view, "observation", None), "geometry", None)
        try:
            if not isinstance(geometry, Mapping):
                return None, "missing_calibration"
            cam = geometry["cameras"]["external"]
            if cam.get("pose_source") != "fixed_world_camera":
                return None, "camera_not_fixed"
            width, height = geometry["image"]["width"], geometry["image"]["height"]
            if type(width) is not int or type(height) is not int or not (16 <= width <= 8192 and 16 <= height <= 8192):
                raise ValueError("invalid calibrated image size")
            position = _array(cam["position_world_m"], (3,))
            rotation = _array(cam["rotation_world_from_camera"], (3, 3))
            focal = _array(cam["focal_length_px"], (2,))
            principal = _array(cam["principal_point_px"], (2,))
            if (np.any(focal <= 0) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
                raise ValueError("invalid camera transform")
            calibration = np.concatenate(([width, height], position, rotation.ravel(), focal, principal))
        except (KeyError, AttributeError, TypeError, ValueError):
            return None, "missing_or_invalid_calibration"
        try:
            rgb = np.asarray(view.external_rgb_upright)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                return None, "unreadable_image"
            if rgb.shape[:2] != (height, width):
                return None, "image_calibration_mismatch"
            factor = min(1., self.max_edge / max(width, height))
            size = (max(1, round(width * factor)), max(1, round(height * factor)))
            small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32)
            # Avoid treating subpixel antialiasing as a local scene change.
            small = cv2.GaussianBlur(small, (3, 3), .6)
            dx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3) / 8
            dy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3) / 8
            texture = np.maximum(np.abs(dx), np.abs(dy)).max(axis=2)
            return _Frame(small, texture, calibration, position, rotation, focal, principal,
                          np.array(size) / [width, height]), None
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError, cv2.error):
            return None, "unreadable_image"

    @staticmethod
    def _box_values(boxes):
        values, signature = [], []
        for box in boxes:
            center, rotation = _array(box["center"], (3,)), _array(box["rotation"], (3, 3))
            half = _array(box["half"], (3,))
            uncertainty = float(box.get("uncertainty", 0.))
            if (np.any(half <= 0) or not np.isfinite(uncertainty) or uncertainty < 0
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6)):
                raise ValueError("invalid world OBB")
            values.append((center, rotation, half + uncertainty))
            # Safety padding can change while the nominal estimate and RGB stay
            # identical. It affects the conservative ROI, not shape identity.
            # assess compares only the common reference/current pixels and never
            # treats an expanded ROI as newly observed geometry or free space.
            signature.extend(np.concatenate((center, rotation.ravel(), half)).tolist())
        return values, tuple(signature)

    @staticmethod
    def _project(values, frame, *, occluder=False):
        height, width = frame.rgb.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        visibility = 1.
        image_hull = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], np.float32)
        for center, rotation, half in values:
            corners = (_SIGNS * half) @ rotation.T + center
            camera = (corners - frame.position) @ frame.rotation
            depth = -camera[:, 2]
            if np.all(depth <= 1e-4):
                if not occluder:
                    visibility = 0.
                continue
            if np.any(depth <= 1e-4):
                return None, 0., "projection_crosses_camera_plane"
            uv = np.column_stack((frame.principal[0] + frame.focal[0]*camera[:, 0]/depth,
                                  frame.principal[1] - frame.focal[1]*camera[:, 1]/depth))
            uv = (uv + .5) * frame.scale - .5
            if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 1e7:
                return None, 0., "invalid_projection"
            hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
            area = float(cv2.contourArea(hull))
            if area < 1e-6:
                if not occluder:
                    visibility = 0.
                continue
            clipped_area, clipped = cv2.intersectConvexConvex(hull, image_hull)
            if not occluder:
                visibility = min(visibility, max(0., float(clipped_area)) / area)
            if clipped is not None and clipped_area > 0:
                cv2.fillConvexPoly(mask, np.round(clipped).astype(np.int32), 1)
        if occluder and np.any(mask):
            # Cover edges, downsampling and small known-tool projection error.
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        return mask.astype(bool), visibility, None

    def _region(self, world_boxes, occluder_boxes, frame):
        try:
            values, signature = self._box_values(world_boxes)
            occluders, _ = self._box_values(occluder_boxes)
            if not values:
                return None, "missing_geometry"
            mask, visible_fraction, reason = self._cached_projection(values, frame)
            if reason:
                return None, reason
            if mask.sum() < self.min_pixels:
                return None, "out_of_view_or_too_small"
            if visible_fraction < self.min_visible_fraction:
                return None, "insufficient_projection_visibility"
            occlusion, _, reason = self._cached_projection(occluders, frame, occluder=True)
            if reason:
                return None, "occluder_projection_unavailable"
            return (mask, occlusion, signature), None
        except (KeyError, TypeError, ValueError, cv2.error):
            return None, "invalid_geometry_projection"

    def _cached_projection(self, values, frame, *, occluder=False):
        # Projection depends on calibration and geometry, not RGB. Reusing the
        # exact mask never reuses an appearance verdict or advances a baseline.
        key = (occluder, tuple(frame.calibration), tuple(frame.scale),
               tuple(tuple(np.concatenate((center, rotation.ravel(), half))) for center, rotation, half in values))
        if key in self._projection_cache:
            self.compute_counts["projection_cache_hits"] += 1
            self._projection_cache.move_to_end(key)
            return self._projection_cache[key]
        self.compute_counts["projections"] += 1
        result = self._project(values, frame, occluder=occluder)
        self._projection_cache[key] = result
        while len(self._projection_cache) > 128:
            self._projection_cache.popitem(last=False)
        return result

    def _quality(self, frame, mask, occlusion):
        visible = mask & ~occlusion
        count = int(visible.sum())
        fraction = count / max(1, int(mask.sum()))
        metrics = {"visible_pixels": count, "visible_fraction": round(fraction, 5)}
        if count < self.min_pixels or fraction < self.min_visible_fraction:
            return visible, "expected_self_occlusion", metrics
        texture_fraction = float(np.mean(frame.texture[visible] > 4.))
        contrast = float(np.max(np.std(frame.rgb[visible], axis=0)))
        metrics.update(texture_fraction=round(texture_fraction, 5), contrast=round(contrast, 4))
        if texture_fraction < .04 or contrast < 4.:
            return visible, "low_texture", metrics
        return visible, None, metrics

    def capture(self, entity_id, world_boxes, view, step, *, occluder_boxes=()):
        """Replace this reference only after an explicit accepted geometry update."""
        self.forget([entity_id])
        frame, reason = self._frame(view)
        if frame is None:
            return self._evidence(entity_id, step, "unverified", reason)
        region, reason = self._region(world_boxes, occluder_boxes, frame)
        if region is None:
            return self._evidence(entity_id, step, "unverified", reason)
        mask, occlusion, signature = region
        _, reason, metrics = self._quality(frame, mask, occlusion)
        self._references[entity_id] = _Reference(frame, mask, occlusion, signature,
                                                int(step), reason is None, reason)
        return self._evidence(entity_id, step, "unverified" if reason else "reference",
                              reason or "geometry_update_reference_captured", reference_step=int(step), **metrics)

    def _difference(self, frame, reference):
        if self._difference_frame is not frame:
            self._difference_frame = frame
            self._difference_cache.clear()
        key = id(reference)
        if key not in self._difference_cache:
            self._difference_cache[key] = np.max(np.abs(frame.rgb - reference.rgb), axis=2)
        return self._difference_cache[key]

    def assess(self, entity_id, world_boxes, view, step, *, occluder_boxes=()):
        """Return advisory local appearance evidence without updating a baseline."""
        frame, reason = self._frame(view)
        ref = self._references.get(entity_id)
        fields = {} if ref is None else {"reference_step": ref.step}
        if frame is None:
            return self._evidence(entity_id, step, "unverified", reason, **fields)
        if ref is None:
            return self._evidence(entity_id, step, "unverified", "no_reference")
        if not np.allclose(frame.calibration, ref.frame.calibration, atol=1e-7, rtol=0):
            return self._evidence(entity_id, step, "unverified", "camera_moved", **fields)
        region, reason = self._region(world_boxes, occluder_boxes, frame)
        if region is None:
            return self._evidence(entity_id, step, "unverified", reason, **fields)
        mask, occlusion, signature = region
        if signature != ref.signature:
            return self._evidence(entity_id, step, "unverified", "geometry_changed_since_reference", **fields)
        common = mask & ref.mask
        union_occlusion = occlusion | ref.occlusion
        visible, reason, metrics = self._quality(frame, common, union_occlusion)
        if reason:
            return self._evidence(entity_id, step, "unverified", reason, **fields, **metrics)
        _, reason, _ = self._quality(ref.frame, common, union_occlusion)
        if reason or not ref.usable:
            return self._evidence(entity_id, step, "unverified", reason or ref.reason, **fields, **metrics)
        difference = self._difference(frame, ref.frame)
        # Broad changes cannot be assigned to this entity. They may reflect
        # illumination, unreported camera motion, or scene-wide changes.
        background = ~union_occlusion
        broad_fraction = float(np.mean(difference[background] > 24.)) if background.any() else 0.
        quadrants = [part for row in np.array_split(background, 2, axis=0)
                     for part in np.array_split(row, 2, axis=1)]
        differences = [part for row in np.array_split(difference, 2, axis=0)
                       for part in np.array_split(row, 2, axis=1)]
        broadly_changed = all(m.sum() >= self.min_pixels and np.mean(d[m] > 24.) > .45
                              for d, m in zip(differences, quadrants))
        if broad_fraction > .60 and broadly_changed:
            return self._evidence(entity_id, step, "unverified", "widespread_scene_or_camera_change",
                                  **fields, **metrics, broad_changed_fraction=round(broad_fraction, 5))
        delta = difference[visible]
        changed_fraction, mean_delta = float(np.mean(delta > 24.)), float(np.mean(delta))
        changed = changed_fraction > .18 and mean_delta > 10.
        return self._evidence(entity_id, step, "changed" if changed else "consistent",
                              "local_appearance_changed" if changed else "local_appearance_consistent",
                              **fields, **metrics, changed_pixel_fraction=round(changed_fraction, 5),
                              mean_rgb_delta=round(mean_delta, 4))
