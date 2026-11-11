"""Observation-only orchestration, explicit ambiguity, and dated feedback.

No method commands a robot, authorizes a grasp, or reads evaluator information.
SAM detections are associated locally using conservative visibility/appearance
checks. This is not SAM's video tracker or material-point motion estimation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Sequence

import numpy as np

from . import VERSION
from .geometry import GeometryConfig, build_geometry
from .types import Detection, TargetSpec, VisionFrame, finite_scalar


@dataclass(frozen=True)
class EngineConfig:
    max_age_s: float = 1.0
    min_region_pixels: int = 8
    max_identity_displacement_m: float = 0.10
    max_appearance_distance: float = 0.35
    min_mask_iou: float = 0.05
    geometry_change_m: float = 0.01
    geometry: GeometryConfig = field(default_factory=GeometryConfig)

    def __post_init__(self):
        for name in ("max_age_s", "max_identity_displacement_m", "geometry_change_m"):
            if finite_scalar(getattr(self, name), name, minimum=0) == 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_appearance_distance", "min_mask_iou"):
            value = finite_scalar(getattr(self, name), name, minimum=0)
            if value > 1:
                raise ValueError(f"{name} must be <= 1")
        if type(self.min_region_pixels) is not int or self.min_region_pixels < 1:
            raise ValueError("min_region_pixels must be positive")
        if not isinstance(self.geometry, GeometryConfig):
            raise ValueError("geometry must be GeometryConfig")


def _appearance(frame, regions):
    # A weak association feature, NOT a semantic identity or surface correspondence.
    result = {}
    for camera in frame.cameras:
        mask = regions.get(camera.name)
        if mask is not None and np.any(mask):
            pixels = camera.rgb[mask]
            hist = np.concatenate([np.histogram(pixels[:, c], bins=8, range=(0, 256))[0]
                                   for c in range(3)]).astype(float)
            result[camera.name] = hist / hist.sum()
    return result


def _iou(a, b):
    if a.shape != b.shape:
        return 0.0
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def _centre(geometry):
    value = geometry.get("centroid_world_m")
    if value is None:
        return None
    centre = np.asarray(value, float)
    return centre if centre.shape == (3,) and np.isfinite(centre).all() else None


@dataclass
class VisionSnapshot:
    data: dict
    geometry: object

    def to_dict(self):
        return deepcopy(self.data)

    def summary(self):
        """Bounded executive feedback, separate from dense planner geometry."""
        return {
            "version": self.data["version"], "frame_id": self.data["frame_id"],
            "captured_at_s": self.data["captured_at_s"],
            "generated_at_s": self.data["generated_at_s"],
            "status": self.data["status"], "age_s": self.data["age_s"],
            "objects": [{key: deepcopy(obj[key]) for key in (
                "key", "track_id", "status", "identity_evidence", "centroid_world_m",
                "relative_to_hand_m", "interaction")}
                for obj in self.data["objects"]],
            "events": deepcopy(self.data["events"]),
            "limitations": deepcopy(self.data["limitations"]),
        }


class VisionEngine:
    def __init__(self, segmenter, depth_estimator, config=None):
        self.segmenter = segmenter
        self.depth_estimator = depth_estimator
        self.config = config or EngineConfig()
        if not isinstance(self.config, EngineConfig):
            raise TypeError("config must be EngineConfig")
        self._last_frame = None
        self._tracks = {}
        self._next_track = 1
        self.latest = None
        self._clock_anchor = None

    def reset(self):
        """Begin a new episode; no old image, geometry, or identity survives."""
        self._last_frame = None
        self._tracks.clear()
        self.latest = None
        self._clock_anchor = None
        # IDs remain monotonically allocated so reset never aliases an old track.

    def _select(self, frame, targets, detections):
        cameras = {camera.name: camera for camera in frame.cameras}
        by_key = {target.key: {view: [] for view in cameras} for target in targets}
        for detection in detections:
            if not isinstance(detection, Detection):
                raise TypeError("segmenter must return validated Detection objects")
            if detection.prompt_key not in by_key or detection.view not in cameras:
                raise ValueError("segmenter returned an unexpected target or camera")
            if detection.mask.shape != cameras[detection.view].rgb.shape[:2]:
                raise ValueError("segmentation masks must use the original camera raster")
            if np.count_nonzero(detection.mask) >= self.config.min_region_pixels:
                by_key[detection.prompt_key][detection.view].append(detection)

        selected, reports = {}, {}
        for target in targets:
            prior = self._tracks.get(target.key)
            first_seed = prior is None or prior["text"] != target.text
            matches, counts, ambiguous = {}, {}, False
            for view, candidates in by_key[target.key].items():
                if first_seed and view in target.seed_pixels:
                    u, v = target.seed_pixels[view]
                    camera = cameras[view]
                    if not (0 <= u < camera.rgb.shape[1] and 0 <= v < camera.rgb.shape[0]):
                        raise ValueError(f"seed pixel for {target.key}/{view} lies outside image")
                    candidates = [d for d in candidates if d.mask[int(v), int(u)]]
                counts[view] = len(candidates)
                if len(candidates) > 1:
                    ambiguous = True
                elif len(candidates) == 1:
                    matches[view] = candidates[0].mask
            if set(target.seed_pixels) - set(cameras):
                raise ValueError("seed refers to an unknown camera")
            status = "identity_ambiguous" if ambiguous else (
                "visible" if len(matches) == len(cameras) else "partially_visible" if matches else "not_visible")
            # Do not combine masks of different possible instances into one 3D target.
            selected[target.key] = {} if ambiguous else matches
            reports[target.key] = {"visibility": status, "candidate_counts": counts}
        return selected, reports

    def process(self, frame: VisionFrame, targets: Sequence[TargetSpec], *, now_s=None):
        if not isinstance(frame, VisionFrame):
            raise TypeError("frame must be VisionFrame")
        targets = tuple(targets)
        if any(not isinstance(t, TargetSpec) for t in targets) or len({t.key for t in targets}) != len(targets):
            raise ValueError("targets must have unique validated keys")
        if self._last_frame is not None:
            if frame.frame_id <= self._last_frame.frame_id or frame.captured_at_s <= self._last_frame.captured_at_s:
                raise ValueError("frames must have strictly increasing IDs and sensor timestamps")
        now = frame.captured_at_s if now_s is None else finite_scalar(now_s, "now_s", minimum=0)
        if now < frame.captured_at_s:
            raise ValueError("now_s cannot predate the sensor frame")
        start = perf_counter()
        # Backends stay resident. Failures propagate; never use yesterday's geometry
        # or synthetic substitutes while reporting that inference succeeded.
        self.latest = None
        self._clock_anchor = None
        try:
            detections = tuple(self.segmenter.segment(frame, targets))
            segmented = perf_counter()
            selected, reports = self._select(frame, targets, detections)
            depths = self.depth_estimator.estimate(frame)
            estimated = perf_counter()
            geometry = build_geometry(frame, depths, selected, self.config.geometry)
            geometry_data = geometry.to_dict()
            reconstructed = perf_counter()
        except Exception:
            self._tracks.clear()
            raise

        generated_at = now + perf_counter() - start
        age = generated_at - frame.captured_at_s
        stale = age > self.config.max_age_s
        objects, events, next_tracks = [], [], {}
        for target in targets:
            prior = self._tracks.get(target.key)
            report = deepcopy(reports[target.key])
            geom = geometry.object_geometry.get(target.key, {})
            if geom.get("association_status") == "rejected_mixed_instance":
                report["visibility"] = "identity_ambiguous"
            centre = _centre(geom)
            regions = selected[target.key]
            appearance = _appearance(frame, regions)
            observable = report["visibility"] == "visible" and centre is not None and not stale
            identity = "unverified"
            track_id = None
            status = "stale" if stale else report["visibility"]
            if observable:
                track_fresh = (prior is not None and self._last_frame is not None
                               and frame.captured_at_s - self._last_frame.captured_at_s <= self.config.max_age_s)
                if prior is not None and prior.get("valid") and not track_fresh:
                    events.append({"type": "track_expired", "key": target.key,
                                   "old_track_id": prior["track_id"]})
                if prior is not None and prior.get("valid") and track_fresh and prior["text"] == target.text:
                    views = set(appearance) & set(prior["appearance"])
                    appearance_ok = len(views) >= 2 and all(
                        np.abs(appearance[v] - prior["appearance"][v]).sum() / 2 <= self.config.max_appearance_distance
                        for v in views)
                    position_ok = np.linalg.norm(centre - prior["centre"]) <= self.config.max_identity_displacement_m
                    # Pixel overlap is evidence only in a stationary calibrated view.
                    # When every camera moves, this first version cannot establish
                    # material correspondence; similar colours alone do not keep an ID.
                    previous_cameras = {c.name: c for c in self._last_frame.cameras}
                    stable_views = {c.name for c in frame.cameras
                                    if c.name in previous_cameras
                                    and np.allclose(c.world_to_camera, previous_cameras[c.name].world_to_camera,
                                                    atol=1e-7, rtol=0)
                                    and np.allclose(c.intrinsics, previous_cameras[c.name].intrinsics,
                                                    atol=1e-7, rtol=0)}
                    mask_support = sum(_iou(regions[v], prior["regions"][v]) > self.config.min_mask_iou
                                       for v in set(regions) & set(prior["regions"]) & stable_views)
                    if appearance_ok and position_ok and mask_support:
                        track_id = prior["track_id"]
                        identity = "crossview_geometry_appearance_and_stationary_mask_continuity"
                        status = "tracked_estimate"
                        change = float(np.linalg.norm(centre - prior["centre"]))
                        if change >= self.config.geometry_change_m:
                            events.append({"type": "geometry_changed", "key": target.key,
                                           "estimated_centroid_change_m": change,
                                           "interpretation": "visible_surface_estimate_changed; not rigid-body motion proof"})
                    else:
                        status = "identity_ambiguous"
                        events.append({"type": "identity_ambiguous", "key": target.key,
                                       "appearance_consistent": appearance_ok,
                                       "position_consistent": bool(position_ok), "mask_support_views": mask_support,
                                       "stationary_camera_count": len(stable_views)})
                else:
                    track_id = f"vision-{self._next_track}"
                    self._next_track += 1
                    identity = "unique_prompt_regions_with_crossview_geometry"
                    status = "new_candidate" if prior is None else "reacquired_candidate"
                    events.append({"type": status, "key": target.key, "track_id": track_id,
                                   "same_identity_as_old_track": None})
            if not observable or status == "identity_ambiguous":
                if report["visibility"] == "identity_ambiguous":
                    events.append({"type": "identity_ambiguous", "key": target.key,
                                   "candidate_counts": report["candidate_counts"],
                                   "geometry_association_status": geom.get("association_status")})
                if prior is not None and prior.get("valid"):
                    events.append({"type": "track_lost", "key": target.key, "old_track_id": prior["track_id"]})
                events.append({"type": "need_observation", "key": target.key,
                               "reason": "stale" if stale else report["visibility"] if centre is not None
                               else "insufficient_crossview_geometry"})
            relative = None
            if centre is not None and frame.hand_world is not None:
                relative = (frame.hand_world[:3, :3].T @ (centre - frame.hand_world[:3, 3])).tolist()
            objects.append({
                "key": target.key, "text": target.text, "track_id": track_id, "status": status,
                "identity_evidence": identity, "candidate_counts": report["candidate_counts"],
                "centroid_world_m": None if centre is None else centre.tolist(),
                "relative_to_hand_m": relative, "geometry": deepcopy(geom),
                "interaction": {"attachment": "unknown", "slip": "unknown", "release": "unknown",
                                "reason": "no_material_point_motion_or_contact_evidence"},
            })
            next_tracks[target.key] = {"valid": track_id is not None, "track_id": track_id,
                                       "centre": centre, "regions": regions, "appearance": appearance,
                                       "text": target.text}

        # Include association/serialization work in freshness, not just inference.
        generated_at = now + perf_counter() - start
        age = generated_at - frame.captured_at_s
        if age > self.config.max_age_s and not stale:
            stale = True
            events = [e for e in events if e["type"] not in
                      {"new_candidate", "reacquired_candidate", "geometry_changed"}]
            for obj in objects:
                obj.update(status="stale", track_id=None, identity_evidence="stale_observation")
                next_tracks[obj["key"]].update(valid=False, track_id=None)
                prior = self._tracks.get(obj["key"])
                if prior is not None and prior.get("valid") and not any(
                        e["type"] == "track_lost" and e["key"] == obj["key"] for e in events):
                    events.append({"type": "track_lost", "key": obj["key"], "old_track_id": prior["track_id"]})
                events.append({"type": "need_observation", "key": obj["key"], "reason": "stale"})
        if stale:
            events.append({"type": "stale_observation", "age_s": age})
        for key in self._tracks.keys() - next_tracks.keys():
            events.append({"type": "target_removed", "key": key,
                           "meaning": "no longer tracked; does not clear scene occupancy"})
        data = {
            "version": VERSION, "frame_id": frame.frame_id,
            "captured_at_s": frame.captured_at_s, "generated_at_s": generated_at,
            "age_s": age, "max_age_s": self.config.max_age_s,
            "status": "stale" if stale else "estimated",
            "objects": objects, "events": events, "geometry": geometry_data,
            "timing_s": {"segmentation": segmented-start, "depth": estimated-segmented,
                         "geometry": reconstructed-estimated, "total": perf_counter()-start},
            "config": asdict(self.config),
            "limitations": ["model_estimates_not_collision_certificates", "unseen_space_is_unknown",
                            "scores_not_calibrated_probabilities", "identity_association_is_a_hypothesis",
                            "attachment_slip_release_not_measured", "robot_geometry_not_removed_without_projection",
                            "no_robot_actions_or_task_success_judgment"],
        }
        self.latest = VisionSnapshot(data, geometry)
        self._tracks = next_tracks
        self._last_frame = frame
        self._clock_anchor = (now, start)
        return self.latest

    def query_path(self, polyline, radius_m, *, now_s=None):
        """Inspect estimated corridor geometry, never grant robot execution."""
        if self.latest is None:
            raise RuntimeError("no current successful observation")
        start = perf_counter()
        now = (self._clock_anchor[0] + start - self._clock_anchor[1] if now_s is None
               else finite_scalar(now_s, "now_s", minimum=0))
        if now < self.latest.data["generated_at_s"]:
            raise ValueError("query time cannot predate the generated observation")
        result = self.latest.geometry.assess_path(polyline, radius_m, now, self.config.max_age_s)
        age = now + perf_counter() - start - self.latest.data["captured_at_s"]
        result["age_s"] = age
        if age > self.config.max_age_s:
            result.update(status="stale", reason="evidence_too_old_at_query_completion")
        return result
