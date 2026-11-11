"""Calibrated, partial surface evidence from RGB model depth predictions.

The geometry is *not* measured depth, an object model, or a collision certificate.
Cross-view agreement thresholds are engineering acceptance thresholds, not metric
error bounds. Invalid depth, occluded space, and unseen space remain unknown.

``build_geometry(frame, depths, regions)`` accepts original-raster boolean target
masks. Depth rasters may be resized only by the documented pixel-centre affine:
``u_depth + .5 = (u_original + .5) * width_depth / width_original`` (likewise v).
Cropping, changed extrinsics, and undocumented intrinsic changes are rejected.
Adapters with other preprocessing must resample depth onto the original rays.

All valid scene surfaces are retained privately, including low-score and
cross-view-conflicting hypotheses. Only the JSON representation is subsampled.
No model score is interpreted as a calibrated probability or used to clear space.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .types import CameraFrame, DepthEstimate, VisionFrame, finite_scalar


@dataclass(frozen=True)
class GeometryConfig:
    """Generic acceptance and output limits; none is an accuracy guarantee."""

    tolerance_abs_m: float = 0.02
    tolerance_rel: float = 0.02
    min_crossview_support: int = 2
    min_baseline_m: float = 0.001
    mask_agreement_min: float = 0.9
    min_target_points: int = 3
    max_serialized_points: int = 2048
    path_sample_spacing_m: float = 0.01
    path_surface_margin_m: float = 0.005
    max_path_samples: int = 4096

    def __post_init__(self):
        for name in ("tolerance_abs_m", "tolerance_rel", "min_baseline_m",
                     "path_surface_margin_m"):
            object.__setattr__(self, name, finite_scalar(getattr(self, name), name, minimum=0))
        spacing = finite_scalar(self.path_sample_spacing_m, "path_sample_spacing_m", minimum=0)
        if spacing == 0:
            raise ValueError("path_sample_spacing_m must be positive")
        object.__setattr__(self, "path_sample_spacing_m", spacing)
        agreement = finite_scalar(self.mask_agreement_min, "mask_agreement_min", minimum=0)
        if agreement > 1:
            raise ValueError("mask_agreement_min must be <= 1")
        object.__setattr__(self, "mask_agreement_min", agreement)
        for name, minimum in (("min_crossview_support", 2), ("min_target_points", 3),
                              ("max_serialized_points", 1), ("max_path_samples", 2)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


def _bounded_indices(count: int, limit: int) -> np.ndarray:
    return np.arange(count) if count <= limit else np.linspace(0, count - 1, limit, dtype=int)


def _points(value, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=float)
    if result.shape == (3,):
        result = result.reshape(1, 3)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape N x 3")
    return result


def _project(points: np.ndarray, intrinsics: np.ndarray, world_to_camera: np.ndarray):
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    z = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        homogeneous = camera @ intrinsics.T
        uv = homogeneous[:, :2] / z[:, None]
    return uv, z


def _pixel_indices(uv: np.ndarray, shape: tuple[int, int]):
    height, width = shape
    # Only the region between observed pixel centres is usable. In particular,
    # clipping an out-of-image projection to a border pixel cannot clear space.
    inside = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
              & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
    safe = np.where(inside[:, None], uv, 0)
    pixels = np.floor(safe + 0.5).astype(np.int64)
    return pixels[:, 1], pixels[:, 0], inside


def _tolerance(first: np.ndarray, second: np.ndarray, config: GeometryConfig):
    return config.tolerance_abs_m + config.tolerance_rel * np.maximum(np.abs(first), np.abs(second))


@dataclass
class _View:
    camera: CameraFrame
    estimate: DepthEstimate
    valid: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    points_world: np.ndarray
    support: np.ndarray
    contradictions: np.ndarray
    occlusions: np.ndarray
    support_map: np.ndarray
    contradiction_map: np.ndarray

    @property
    def center(self):
        matrix = self.camera.world_to_camera
        return -matrix[:3, :3].T @ matrix[:3, 3]


def _sample(points: np.ndarray, view: _View, config: GeometryConfig):
    uv, z = _project(points, view.estimate.intrinsics, view.estimate.world_to_camera)
    rows, cols, inside = _pixel_indices(uv, view.valid.shape)
    valid = inside & np.isfinite(z) & (z > 0) & view.valid[rows, cols]
    depth = view.estimate.depth_m[rows, cols]
    with np.errstate(invalid="ignore", over="ignore"):
        difference = z - depth
        tolerance = _tolerance(z, depth, config)
        surface = valid & (np.abs(difference) <= tolerance)
        # A point behind a visible surface is occluded, not contradicted.
        occluded = valid & (difference > tolerance)
        front = valid & (difference < -tolerance)
    return rows, cols, surface, occluded, front, valid


def _independent(first: _View, second: _View, config: GeometryConfig) -> bool:
    return float(np.linalg.norm(first.center - second.center)) > config.min_baseline_m


def _make_view(camera: CameraFrame, estimate: DepthEstimate) -> _View:
    if estimate.view != camera.name:
        raise ValueError("depth mapping key, depth view, and camera name must agree")
    if not np.array_equal(estimate.world_to_camera, camera.world_to_camera):
        raise ValueError("depth extrinsics must equal the supplied camera transform")
    raw_h, raw_w = camera.rgb.shape[:2]
    height, width = estimate.depth_m.shape
    sx, sy = width / raw_w, height / raw_h
    affine = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    expected_k = affine @ camera.intrinsics
    if not np.allclose(estimate.intrinsics, expected_k, rtol=1e-7, atol=1e-8):
        raise ValueError("depth intrinsics do not match documented pixel-centre resize")
    valid = np.isfinite(estimate.depth_m) & (estimate.depth_m > 0)
    rows, cols = np.nonzero(valid)
    pixels = np.column_stack((cols, rows, np.ones(len(rows))))
    camera_points = (pixels @ np.linalg.inv(estimate.intrinsics).T) * estimate.depth_m[rows, cols, None]
    transform = estimate.world_to_camera
    world = (camera_points - transform[:3, 3]) @ transform[:3, :3]
    # Extremely large finite inputs can overflow a transformation. They are not
    # valid geometry and must not create free-space observations.
    finite = np.isfinite(world).all(axis=1)
    if not finite.all():
        valid[rows[~finite], cols[~finite]] = False
        rows, cols, world = rows[finite], cols[finite], world[finite]
    n = len(world)
    return _View(camera, estimate, valid, rows, cols, world,
                 np.ones(n, dtype=np.int32), np.zeros(n, dtype=np.int32),
                 np.zeros(n, dtype=np.int32), np.zeros(valid.shape, dtype=np.int32),
                 np.zeros(valid.shape, dtype=np.int32))


def _mask_on_depth(mask: np.ndarray, view: _View) -> np.ndarray:
    raw_h, raw_w = mask.shape
    height, width = view.valid.shape
    raw_u = (view.cols + 0.5) * raw_w / width - 0.5
    raw_v = (view.rows + 0.5) * raw_h / height - 0.5
    rows, cols, inside = _pixel_indices(np.column_stack((raw_u, raw_v)), mask.shape)
    return inside & mask[rows, cols]


def _mask_at_world(points: np.ndarray, mask: np.ndarray, view: _View):
    uv, z = _project(points, view.camera.intrinsics, view.camera.world_to_camera)
    rows, cols, inside = _pixel_indices(uv, mask.shape)
    return inside & (z > 0) & mask[rows, cols]


def _normal(points: np.ndarray, camera_center: np.ndarray):
    if len(points) < 3:
        return None, None
    centered = points - points.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    if values[1] <= 1e-12 or max(values[0], 0) > 0.15 * values[1]:
        return None, None
    normal = vectors[:, 0]
    if np.dot(normal, camera_center - points.mean(axis=0)) < 0:
        normal = -normal
    return normal.tolist(), float(np.sqrt(max(values[0], 0)))


def _object_geometry(key: str, masks: Mapping[str, np.ndarray], views: Mapping[str, _View],
                     config: GeometryConfig) -> dict:
    hypotheses = 0
    supported = []
    pair_evidence = []
    rejected = False
    for name, mask in masks.items():
        if name not in views:
            continue
        view = views[name]
        selected = _mask_on_depth(mask, view)
        points = view.points_world[selected]
        hypotheses += len(points)
        target_support = np.ones(len(points), dtype=np.int32)
        for other_name, other_mask in masks.items():
            if other_name == name or other_name not in views:
                continue
            other = views[other_name]
            if not _independent(view, other, config):
                continue
            _, _, surface, occluded, front, _ = _sample(points, other, config)
            in_mask = _mask_at_world(points, other_mask, other)
            matches = surface & in_mask
            target_support += matches
            comparable = int(surface.sum())
            matched = int(matches.sum())
            ratio = matched / comparable if comparable else None
            # Association evidence is checked in each direction. Disjoint
            # selected instances cannot be fused into one target centroid.
            pair_rejected = comparable > 0 and ratio < config.mask_agreement_min
            rejected |= pair_rejected
            pair_evidence.append({"source_view": name, "other_view": other_name,
                                  "comparable_points": comparable, "mask_matches": matched,
                                  "mask_agreement_fraction": ratio,
                                  "occluded_points": int(occluded.sum()),
                                  "contradicted_points": int(front.sum()),
                                  "association_rejected": bool(pair_rejected)})
        eligible = ((target_support >= config.min_crossview_support)
                    & (view.contradictions[selected] == 0))
        supported.append(points[eligible])
    points = np.concatenate(supported, axis=0) if supported else np.empty((0, 3))
    if rejected:
        points = np.empty((0, 3))
    enough = len(points) >= config.min_target_points
    normal, residual = _normal(points, next(iter(views.values())).center) if enough else (None, None)
    selected = _bounded_indices(len(points), config.max_serialized_points)
    return {
        "target_key": key,
        "status": "supported_surface_hypothesis" if enough else "unknown",
        "association_status": ("rejected_mixed_instance" if rejected else
                               "crossview_supported" if enough else "unknown"),
        "surface_hypothesis_count": int(hypotheses),
        "supported_point_count": int(len(points)),
        "centroid_world_m": points.mean(axis=0).tolist() if enough else None,
        "bounds_world_m": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()} if enough else None,
        "normal_world": normal,
        "normal_kind": "planar_supported_surface_fit" if normal is not None else None,
        "normal_fit_rms_m": residual,
        "supported_surface_points_world": points[selected].tolist(),
        "crossview_mask_evidence": pair_evidence,
        "bounds_are_complete_object_extent": False,
        "error_bound_m": None,
        "scores_calibrated": False,
        "unobserved_geometry": "unknown",
        "quality": {"error_bound_m": None, "uncertainty_calibrated": False,
                    "required_crossview_support": config.min_crossview_support,
                    "interpretation": "partial_surface_hypothesis_not_complete_object_geometry"},
    }


@dataclass
class GeometryEvidence:
    """Snapshot whose query results are partial model evidence, never a proof."""

    frame_id: int
    captured_at_s: float
    object_geometry: dict
    config: GeometryConfig
    _views: dict[str, _View] = field(repr=False)

    def _age(self, now_s: float, max_age_s: float):
        now = finite_scalar(now_s, "now_s", minimum=0)
        maximum = finite_scalar(max_age_s, "max_age_s", minimum=0)
        age = now - self.captured_at_s
        return age, age < -1e-6 or age > maximum

    def to_dict(self) -> dict:
        """Return bounded, JSON-safe evidence; private clouds are not subsampled."""
        all_points = [view.points_world for view in self._views.values()]
        points = np.concatenate(all_points) if all_points else np.empty((0, 3))
        support = np.concatenate([view.support for view in self._views.values()]) if all_points else np.empty(0, dtype=int)
        contradictory = np.concatenate([view.contradictions for view in self._views.values()]) if all_points else np.empty(0, dtype=int)
        selected = _bounded_indices(len(points), self.config.max_serialized_points)
        views = {}
        for name, view in self._views.items():
            score = view.estimate.score_map
            finite_scores = score[np.isfinite(score)] if score is not None else np.empty(0)
            views[name] = {
                "source": view.estimate.source,
                "original_image_shape": list(view.camera.rgb.shape[:2]),
                "depth_shape": list(view.valid.shape),
                "intrinsics": view.estimate.intrinsics.tolist(),
                "world_to_camera": view.estimate.world_to_camera.tolist(),
                "valid_depth_pixels": int(view.valid.sum()),
                "invalid_depth_pixels": int((~view.valid).sum()),
                "crossview_supported_points": int(((view.support >= self.config.min_crossview_support)
                                                   & (view.contradictions == 0)).sum()),
                "contradicted_points": int((view.contradictions > 0).sum()),
                "occluded_in_other_views_points": int((view.occlusions > 0).sum()),
                "uncalibrated_score_range": [float(finite_scores.min()), float(finite_scores.max())] if len(finite_scores) else None,
            }
        return {
            "frame_id": self.frame_id, "captured_at_s": self.captured_at_s,
            "coordinate_frame": "world", "length_unit": "metre",
            "source": "rgb_model_surface_hypotheses_with_input_camera_calibration",
            "surface_hypothesis_count": int(len(points)),
            "surface_points_world": points[selected].tolist(),
            "surface_crossview_support_counts": support[selected].tolist(),
            "surface_contradiction_counts": contradictory[selected].tolist(),
            "serialized_points_subsampled": len(selected) < len(points),
            "views": views, "object_geometry": deepcopy(self.object_geometry),
            "crossview_acceptance_thresholds": {
                "absolute_m": self.config.tolerance_abs_m,
                "relative": self.config.tolerance_rel,
                "meaning": "engineering_acceptance_not_accuracy_bound",
            },
            "error_bound_m": None, "scores_calibrated": False,
            "occluded_and_unseen_space": "unknown",
            "collision_free_certified": False,
        }

    def query(self, points_world, now_s: float, max_age_s: float) -> list[dict]:
        """Classify each point as surface/estimated_free/unknown/stale.

        Any fresh observed surface hypothesis wins over free-space votes, even
        when its score/support is low. Free-space evidence requires independent
        views, each terminating at a cross-view-supported, uncontradicted depth.
        """
        points = _points(points_world, "points_world")
        age, stale = self._age(now_s, max_age_s)
        if stale:
            return [{"state": "stale", "age_s": age, "reason": "future_timestamp" if age < 0 else "evidence_too_old"}
                    for _ in points]
        observed = [[] for _ in points]
        free = [[] for _ in points]
        occluded = [[] for _ in points]
        for name, view in self._views.items():
            rows, cols, surface, behind, front, _ = _sample(points, view, self.config)
            supported_depth = ((view.support_map[rows, cols] >= self.config.min_crossview_support)
                               & (view.contradiction_map[rows, cols] == 0))
            for index in np.flatnonzero(surface):
                observed[index].append(name)
            for index in np.flatnonzero(front & supported_depth):
                free[index].append(name)
            for index in np.flatnonzero(behind):
                occluded[index].append(name)
        results = []
        for index, point in enumerate(points):
            independent = []
            for name in free[index]:
                if all(_independent(self._views[name], self._views[other], self.config) for other in independent):
                    independent.append(name)
            if observed[index]:
                state, reason = "surface", "observed_surface_hypothesis"
            elif not np.isfinite(point).all():
                state, reason = "unknown", "invalid_query_point"
            elif occluded[index]:
                state, reason = "unknown", "behind_observed_surface"
            elif len(independent) >= self.config.min_crossview_support:
                state, reason = "estimated_free", "crossview_supported_rays"
            else:
                state, reason = "unknown", "insufficient_supported_visible_rays"
            results.append({"state": state, "reason": reason, "age_s": age,
                            "surface_views": observed[index], "free_evidence_views": independent,
                            "occluded_views": occluded[index], "error_bound_m": None})
        return results

    def assess_path(self, polyline, radius_m: float, now_s: float, max_age_s: float) -> dict:
        """Check a sampled swept tube, not full-arm or continuous safety.

        Distances to *all* retained surface hypotheses are checked against every
        segment. Unknown coverage remains unknown; estimated_clear only describes
        finite ray samples and does not certify the intervening volume.
        """
        path = _points(polyline, "polyline")
        if not len(path) or not np.isfinite(path).all():
            raise ValueError("polyline must contain finite points")
        radius = finite_scalar(radius_m, "radius_m", minimum=0)
        age, stale = self._age(now_s, max_age_s)
        base = {"age_s": age, "radius_m": radius, "full_arm_checked": False,
                "continuous_clearance_certified": False, "error_bound_m": None}
        if stale:
            return dict(base, status="stale", reason="evidence_too_old_or_future")
        segments = list(zip(path[:-1], path[1:])) or [(path[0], path[0])]
        limit = radius + self.config.path_surface_margin_m
        # Low-support/low-score/contradicted surfaces still block. Public-cloud
        # subsampling cannot hide a retained obstacle from this check.
        for name, view in self._views.items():
            for start, end in segments:
                delta = end - start
                denominator = float(delta @ delta)
                for offset in range(0, len(view.points_world), 65536):
                    points = view.points_world[offset:offset + 65536]
                    fraction = np.clip((points - start) @ delta / denominator, 0, 1) if denominator > 0 else np.zeros(len(points))
                    distance = np.linalg.norm(points - (start + fraction[:, None] * delta), axis=1)
                    hit = np.flatnonzero(distance <= limit)
                    if len(hit):
                        i = int(hit[0])
                        return dict(base, status="blocked", reason="observed_surface_intersects_tube",
                                    obstacle_view=name, obstacle_point_world=points[i].tolist(),
                                    surface_distance_m=float(distance[i]),
                                    design_margin_m=self.config.path_surface_margin_m)
        samples = []
        count = 0
        for start, end in segments:
            n = max(2, int(np.ceil(np.linalg.norm(end - start) / self.config.path_sample_spacing_m)) + 1)
            count += n
            if count > self.config.max_path_samples:
                return dict(base, status="unknown", reason="path_sampling_limit")
            samples.append(np.linspace(start, end, n))
        centers = np.concatenate(samples)
        directions = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                               if x or y or z], dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        offsets = np.vstack((np.zeros((1, 3)), radius * directions)) if radius else np.zeros((1, 3))
        probes = (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        result = self.query(probes, now_s, max_age_s)
        states = [item["state"] for item in result]
        if "surface" in states:
            return dict(base, status="blocked", reason="surface_at_sampled_tube_point", samples_checked=len(result))
        if any(state != "estimated_free" for state in states):
            return dict(base, status="unknown", reason="incomplete_visible_supported_coverage", samples_checked=len(result))
        return dict(base, status="estimated_clear", reason="sampled_tube_has_supported_ray_evidence",
                    samples_checked=len(result), unsampled_volume="not_certified")


def build_geometry(frame: VisionFrame, depths: Mapping[str, DepthEstimate],
                   regions: Mapping[str, Mapping[str, np.ndarray]],
                   config: GeometryConfig | None = None) -> GeometryEvidence:
    """Build global scene and per-target partial geometry from permitted inputs."""
    if not isinstance(frame, VisionFrame):
        raise ValueError("frame must be a VisionFrame")
    config = config if config is not None else GeometryConfig()
    if not isinstance(config, GeometryConfig):
        raise ValueError("config must be a GeometryConfig")
    cameras = {camera.name: camera for camera in frame.cameras}
    if set(depths) - set(cameras):
        raise ValueError("depth has an unknown camera")
    views = {}
    for name, estimate in depths.items():
        if not isinstance(estimate, DepthEstimate):
            raise ValueError("depth values must be DepthEstimate instances")
        views[name] = _make_view(cameras[name], estimate)
    validated_masks = {}
    for key, masks in regions.items():
        if not isinstance(key, str) or not key:
            raise ValueError("target key must be a nonempty string")
        validated_masks[key] = {}
        for name, mask in masks.items():
            if name not in cameras:
                raise ValueError("target mask has an unknown camera")
            value = np.asarray(mask)
            if value.dtype != np.bool_ or value.shape != cameras[name].rgb.shape[:2]:
                raise ValueError("target masks must be boolean at original RGB resolution")
            validated_masks[key][name] = value.copy()
    for name, view in views.items():
        for other_name, other in views.items():
            if name == other_name:
                continue
            _, _, surface, occluded, front, _ = _sample(view.points_world, other, config)
            if _independent(view, other, config):
                view.support += surface
            view.contradictions += front
            view.occlusions += occluded
        view.support_map[view.rows, view.cols] = view.support
        view.contradiction_map[view.rows, view.cols] = view.contradictions
    objects = {key: _object_geometry(key, masks, views, config) for key, masks in validated_masks.items()}
    return GeometryEvidence(frame.frame_id, frame.captured_at_s, objects, config, views)
