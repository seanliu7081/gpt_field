"""Resident, latest-wins perception on dated public observations.

Only this worker calls SAM/DA3.  Published snapshots are never revised in place.
The control thread may submit at every step, but a slow inference does not cause
another refresh merely because its own wall time exceeded a freshness threshold.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from threading import Condition, Thread
from time import perf_counter

import numpy as np

from ..foundation_harness.perception import HarnessPerception, _normals
from ..foundation_harness.types import ObjectGeometry, SceneSnapshot, jsonable
from ..foundation_vision.bridge import frame_from_step_view
from ..foundation_vision.geometry import build_geometry, _mask_on_depth, _project, _pixel_indices, _sample
from ..foundation_vision.types import TargetSpec


@dataclass(frozen=True)
class SceneConfig:
    refresh_steps: int = 10
    min_refresh_interval_s: float = 0.5
    voxel_m: float = 0.005
    max_scene_points: int = 30000
    max_object_points: int = 4096
    retention_s: float = 15.0
    self_surface_band_m: float = 0.006
    observed_sigma_floor_m: float = 0.005
    automatic_robot_queries: bool = True

    def __post_init__(self):
        if self.refresh_steps < 1 or self.max_scene_points < 16 or self.max_object_points < 3:
            raise ValueError("invalid scene cadence or storage bound")
        if min(self.voxel_m, self.retention_s, self.self_surface_band_m,
               self.observed_sigma_floor_m) <= 0 or self.min_refresh_interval_s < 0:
            raise ValueError("scene distances/times must be positive")


@dataclass(frozen=True)
class _Job:
    token: int
    frame: object
    state: np.ndarray
    joints: np.ndarray | None
    targets: tuple
    target_key: str


def _target_key(targets):
    items = deepcopy(targets)
    for target in items:
        if not any(v is not None for v in target.get("region_pixels", {}).values()):
            target.pop("frame_id", None)
    return sha256(json.dumps(jsonable(items), sort_keys=True).encode()).hexdigest()


def _box_mask(box, shape):
    values = np.asarray(box, float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("image region requires four finite pixel coordinates")
    x0, y0, x1, y1 = values
    if x1 <= x0 or y1 <= y0:
        raise ValueError("image region must have positive extent")
    height, width = shape
    x0, x1 = np.clip([int(np.floor(x0)), int(np.ceil(x1))], 0, width)
    y0, y1 = np.clip([int(np.floor(y0)), int(np.ceil(y1))], 0, height)
    mask = np.zeros(shape, bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _voxel_sample(points, owners, voxel, limit):
    if not len(points):
        return np.empty(0, int), np.empty(0), np.empty(0, int), voxel
    # Never let a sampled point of one identity replace another identity.
    points = np.asarray(points)
    _, owner_index = np.unique(np.asarray(owners, str), return_inverse=True)
    if len(np.unique(owner_index)) > limit:
        raise ValueError("scene identity count exceeds representation budget")
    origin = points.min(axis=0)
    while True:
        cells = np.floor((points-origin) / voxel).astype(np.int64)
        _, indices, inverse = np.unique(np.column_stack((cells, owner_index)), axis=0,
                                         return_index=True, return_inverse=True)
        if len(indices) <= limit:
            break
        # Coarsen the representation instead of dropping occupied cells. Every
        # original point is covered by its representative plus the radius below.
        voxel *= max(1.25, np.sqrt(len(indices) / limit))
    radius = np.zeros(len(indices))
    np.maximum.at(radius, inverse, np.linalg.norm(points-points[indices[inverse]], axis=1))
    return indices, radius, inverse, voxel


def _voxel_sigma(sigma, inverse, radius):
    result = np.zeros(len(radius))
    np.maximum.at(result, inverse, sigma)
    return result+radius


def scene_summary(snapshot, now_s=None):
    """Small current-scene context; dense covariance/clouds never enter Astra."""
    if snapshot is None:
        return {"status": "pending", "objects": [], "geometry_is_predicted": True}
    now = perf_counter() if now_s is None else now_s
    objects = []
    for obj in snapshot.objects.values():
        points = np.asarray(obj.points)
        objects.append({"id": obj.object_id, "query": obj.query, "role": obj.evidence.get("role"),
                        "parent_id": obj.evidence.get("parent_id"), "status": obj.status,
                        "visible_bounds_m": [points.min(0).tolist(), points.max(0).tolist()] if len(points) else None,
                        "sigma_m": obj.evidence.get("sigma_m"),
                        "observed_frame_id": obj.evidence.get("observed_frame_id"),
                        "complete_extent": False})
    return {"revision": snapshot.evidence.get("revision"), "frame_id": snapshot.frame_id,
            "status": snapshot.evidence.get("perception_status"),
            "age_s": max(0., now-snapshot.captured_at_s), "objects": objects,
            "geometry_is_predicted": True, "unknown_space_is_free": False,
            "self_filter": snapshot.evidence.get("self_filter", {}),
            "point_count": len(snapshot.scene_points)}


class _FramePerception(HarnessPerception):
    """Reuse backend adapters and partial-view geometry, not old scheduling."""

    def __init__(self, engine, config):
        super().__init__(engine, None)
        self.config = config
        self._source_frame = -1
        self._region_tracks = {}
        self._raw_targets = {}

    def _track_mask(self, target, name, camera, prior):
        """Material-point flow moves a previously selected region, not its box."""
        import cv2
        old_rgb, old_mask, old_frame = prior
        if old_frame == self._source_frame:
            return old_mask.copy(), "same_source_region"
        old_gray = cv2.cvtColor(old_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2GRAY)
        features = cv2.goodFeaturesToTrack(old_gray, 100, .01, 3,
                                           mask=old_mask.astype(np.uint8)*255)
        if features is None or len(features) < 4:
            return None, "region_features_unavailable"
        moved, ok, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, features, None)
        if moved is None:
            return None, "region_tracking_lost"
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, old_gray, moved, None)
        if back is None:
            return None, "region_tracking_lost"
        good = ok.ravel().astype(bool) & reverse.ravel().astype(bool)
        good &= np.linalg.norm(back[:, 0]-features[:, 0], axis=1) < 1.5
        if np.count_nonzero(good) < 4:
            return None, "region_tracking_lost"
        affine, inliers = cv2.estimateAffinePartial2D(features[good, 0], moved[good, 0],
                                                    method=cv2.RANSAC, ransacReprojThreshold=2.)
        if affine is None or inliers is None or int(inliers.sum()) < 4:
            return None, "region_motion_unresolved"
        scale = float(np.linalg.norm(affine[:, 0]))
        if not .65 <= scale <= 1.5:
            return None, "region_scale_change_requires_observation"
        warped = cv2.warpAffine(old_mask.astype(np.uint8), affine,
                                (gray.shape[1], gray.shape[0]), flags=cv2.INTER_NEAREST) > 0
        return warped, "tracked_local_region"

    def _select(self, target):
        raw = self._raw_targets[target["id"]]
        masks, reasons, selected = {}, {}, {}
        for camera in self._frame.cameras:
            name, key = camera.name, (target["id"], camera.name)
            candidates = [r for r in self._candidate_records.values()
                          if r["description"]["target_id"] == target["id"]
                          and r["description"]["view"] == name]
            source_box = raw.get("region_pixels", {}).get(name)
            fresh_roi = source_box is not None and raw.get("frame_id") == self._source_frame
            prior = self._region_tracks.get(key)
            region, reason = None, "no_region_binding"
            if fresh_roi:
                region = _box_mask(source_box, camera.rgb.shape[:2])
                reason = "source_frame_pixel_region"
            elif prior is not None:
                region, reason = self._track_mask(target, name, camera, prior)
            elif source_box is not None:
                reason = "stale_pixel_region_not_applied"

            chosen = None
            if region is not None:
                ranked = []
                for candidate in candidates:
                    overlap = np.count_nonzero(candidate["mask"] & region)
                    if overlap:
                        ranked.append((overlap / max(1, np.count_nonzero(region)), candidate))
                ranked.sort(key=lambda pair: pair[0], reverse=True)
                if ranked and (len(ranked) == 1 or ranked[0][0] > ranked[1][0] + .05):
                    chosen = ranked[0][1]
                    # A part ROI stays a part ROI even when SAM returns the body.
                    mask = chosen["mask"] & region
                    reason += ";sam_intersection"
                else:
                    mask = region
                    reason += ";rgb_region_depth_hypothesis"
            elif raw.get("role") == "robot":
                mask = np.logical_or.reduce([r["mask"] for r in candidates]) if candidates else None
                reason = "robot_query_union" if mask is not None else "robot_not_segmented"
            elif source_box is not None or raw.get("role") == "region":
                # A named part may not silently become its entire parent's mask.
                mask = None
            elif len(candidates) == 1 and prior is None:
                chosen = candidates[0]
                mask, reason = chosen["mask"], "unique_query_instance_hypothesis"
            else:
                mask = None
                reason = "instance_ambiguous" if candidates else "instance_unobserved"
            if mask is not None and np.count_nonzero(mask) >= self.engine.config.min_region_pixels:
                masks[name] = mask
                self._region_tracks[key] = (camera.rgb, mask.copy(), self._source_frame)
                if chosen is not None:
                    selected[name] = chosen["description"]["candidate_id"]
            reasons[name] = reason
        return masks, reasons, selected

    def process(self, job):
        self._frame = job.frame
        self._frame_id = self._source_frame = job.frame.frame_id
        self._observation_index = job.frame.frame_id
        self._raw_targets = {t["id"]: t for t in job.targets}
        targets = [{"id": t["id"], "query": t["query"], "parent_id": t.get("parent_id"),
                    "selections": {}, "region_pixels": {}} for t in job.targets]
        specs = [TargetSpec(t["id"], t["query"]) for t in job.targets]
        times, started, segmentation_error = {}, perf_counter(), None
        try:
            detections = tuple(self.engine.segmenter.segment(job.frame, specs)) if specs else ()
        except Exception as exc:
            detections = ()
            segmentation_error = type(exc).__name__ + ": " + str(exc)
        times["segmentation_s"] = perf_counter()-started
        self._record_candidates(detections, targets)
        started = perf_counter()
        self._depths = self.engine.depth_estimator.estimate(job.frame)
        times["depth_s"] = perf_counter()-started
        started = perf_counter()
        self._geometry = build_geometry(job.frame, self._depths, {}, self.engine.config.geometry)
        self._normal_maps = {name: _normals(view) for name, view in self._geometry._views.items()}
        times["geometry_s"] = perf_counter()-started
        snapshot = self._assemble(targets)
        snapshot.evidence["timings_s"] = times
        if segmentation_error is not None:
            snapshot.evidence["segmentation_error"] = segmentation_error
        for target in job.targets:
            obj = snapshot.objects[target["id"]]
            obj.evidence.update(role=target.get("role", "object"), parent_id=target.get("parent_id"),
                                observed_frame_id=job.frame.frame_id,
                                region_identity="source_region_then_material_flow",
                                source_region_frame=target.get("frame_id"))
        return snapshot


class SceneService:
    """One inference owner, one in-flight and at most one pending observation."""

    def __init__(self, engine, output_dir=None, kinematics=None, config=None):
        self.config = config or SceneConfig()
        self.engine, self.kinematics = engine, kinematics
        self.output_dir = Path(output_dir) if output_dir else None
        self._backend = _FramePerception(engine, self.config)
        self._cv = Condition()
        self._pending = self._latest = None
        self._closed = False
        self._next_token = self._done_token = 0
        self._last_source_frame = -self.config.refresh_steps
        self._last_submitted_s = -float("inf")
        self._target_key = None
        self._seen = deque(maxlen=4)
        self._observation_number = -1
        self._invalidated = set()
        self._objects = {}
        self._unlabelled_memory = None
        self._robot = None
        self.errors = []
        self.stats = {"submitted": 0, "published": 0, "coalesced": 0, "cadence_skipped": 0,
                      "failed": 0, "worker_s": 0., "max_pending": 1}
        self._thread = Thread(target=self._loop, name="foundation-vision-v1", daemon=True)
        self._thread.start()

    def submit(self, view, targets, *, force=False, frame_id=None):
        observation = getattr(view, "observation", view)
        now = perf_counter()
        with self._cv:
            old_capture = next(((stamp, fid) for prior, stamp, fid in self._seen if prior is observation), None)
            if old_capture is None:
                self._observation_number += 1
            source_id = int(frame_id if frame_id is not None else
                            old_capture[1] if old_capture is not None else self._observation_number)
            at = old_capture[0] if old_capture is not None else now
            if old_capture is None:
                self._seen.append((observation, at, source_id))
        parsed = deepcopy(list(targets))
        ids = [t["id"] for t in parsed]
        if len(ids) != len(set(ids)):
            raise ValueError("target IDs must be unique")
        for target in parsed:
            target.setdefault("role", "object")
            target.setdefault("parent_id", None)
            target.setdefault("frame_id", source_id)
            target.setdefault("region_pixels", {})
            if not target.get("id") or not target.get("query"):
                raise ValueError("target id/query required")
        key = _target_key(parsed)
        with self._cv:
            if self._closed:
                raise RuntimeError("scene service is closed")
            changed = key != self._target_key
            due = source_id-self._last_source_frame >= self.config.refresh_steps
            if not force and not changed and (not due or now-self._last_submitted_s < self.config.min_refresh_interval_s):
                self.stats["cadence_skipped"] += 1
                return self._next_token
            # The timestamp above remains unchanged when a slow model later
            # selects an ROI from exactly the same observation object.
        frame = frame_from_step_view(view, observation.geometry, source_id, at)
        state = np.array(view.state, float, copy=True)
        joints = getattr(observation, "joint_position", None)
        joints = None if joints is None else np.array(joints, float, copy=True)
        if self.config.automatic_robot_queries:
            for name, query in (("__robot_arm", "robot arm"), ("__robot_gripper", "robot gripper")):
                if name not in ids:
                    parsed.append({"id": name, "query": query, "role": "robot", "parent_id": None,
                                   "frame_id": source_id, "region_pixels": {}})
        with self._cv:
            self._next_token += 1
            token = self._next_token
            if self._pending is not None:
                self.stats["coalesced"] += 1
            self._pending = _Job(token, frame, state, joints, tuple(parsed), key)
            self._last_source_frame, self._last_submitted_s, self._target_key = source_id, now, key
            self.stats["submitted"] += 1
            self._cv.notify_all()
            return token

    def latest(self):
        with self._cv:
            return self._latest

    def invalidate(self, object_ids, reason="object_moved"):
        """Invalidate static memory after a robot event, without inference.

        Already published snapshots retain their original provenance. Consumers
        must use their current attached geometry rather than a static old copy.
        """
        with self._cv:
            self._invalidated.update(str(value) for value in object_ids if value)
            self.stats["last_invalidation"] = str(reason)

    def wait(self, token, timeout_s=10.):
        deadline = perf_counter()+max(0., float(timeout_s))
        with self._cv:
            while self._done_token < token and not self._closed:
                remaining = deadline-perf_counter()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)
            if self._latest is not None and self._latest.evidence["request_token"] >= token:
                return self._latest
            return None

    def close(self, timeout_s=30.):
        with self._cv:
            self._closed = True
            self._pending = None
            self._cv.notify_all()
        self._thread.join(timeout=max(0., timeout_s))
        if self._thread.is_alive():
            raise TimeoutError("perception worker did not finish within close timeout")

    def _loop(self):
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                job, self._pending = self._pending, None
            start = perf_counter()
            try:
                snapshot = self._backend.process(job)
                self._prepare(snapshot, job)
                self._save(snapshot)
                with self._cv:
                    self._latest = snapshot
                    self.stats["published"] += 1
            except Exception as exc:
                with self._cv:
                    self.stats["failed"] += 1
                    self.errors.append({"token": job.token, "frame_id": job.frame.frame_id,
                                        "type": type(exc).__name__, "message": str(exc), "at_s": perf_counter()})
                    self.errors[:] = self.errors[-16:]
            finally:
                with self._cv:
                    self._done_token = job.token
                    self.stats["worker_s"] += perf_counter()-start
                    self._cv.notify_all()

    def _self_filter(self, snapshot, points, owners, source_indices, state, joints):
        keep = np.ones(len(points), bool)
        evidence = {"removed": 0, "robot_query_points": 0, "status": "unavailable"}
        if joints is None or not len(points):
            return keep, evidence
        from ..foundation_harness.robot_geometry import RobotCollisionModel
        import cv2
        if self.kinematics is None:
            from ..robodojo_direct_sim import PandaKinematics
            self.kinematics = PandaKinematics.from_xml()
        if self._robot is None:
            self._robot = RobotCollisionModel(self.kinematics, include_visual=True)
        bound = self.kinematics.bind(state, joints)
        shapes = self._robot.shapes(bound, joints, state[6:8])
        roles = {key: obj.evidence.get("role") for key, obj in snapshot.objects.items()}
        robot_owned = np.array([roles.get(str(owner)) == "robot" for owner in owners])
        protected = (owners != "") & ~robot_owned
        slices = snapshot.evidence["scene_point_view_slices"]
        for camera in snapshot.frame.cameras:
            interval = slices.get(camera.name)
            if interval is None:
                continue
            ids = np.flatnonzero((source_indices >= interval[0]) & (source_indices < interval[1]))
            if not len(ids):
                continue
            view = snapshot.geometry._views[camera.name]
            local_indices = source_indices[ids]-interval[0]
            robot_mask = np.zeros(camera.rgb.shape[:2], bool)
            for obj in snapshot.objects.values():
                if obj.evidence.get("role") == "robot" and camera.name in obj.masks:
                    robot_mask |= obj.masks[camera.name]
            semantic = _mask_on_depth(robot_mask, view)[local_indices]
            evidence["robot_query_points"] += int(semantic.sum())
            origin = -camera.world_to_camera[:3, :3].T @ camera.world_to_camera[:3, 3]
            uv, _ = _project(points[ids], camera.intrinsics, camera.world_to_camera)
            rows, cols, raster = _pixel_indices(uv, robot_mask.shape)
            measured = np.linalg.norm(points[ids]-origin, axis=1)
            for shape in shapes:
                vertices = shape.surface_points() @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
                if np.all(vertices[:, 2] <= 0):
                    continue
                inside_camera = shape.contains(origin[None], 1e-7)[0]
                if inside_camera:
                    # A convex camera enclosure cannot establish which rays hit
                    # its physical aperture. Only semantic robot pixels qualify.
                    silhouette = semantic.copy()
                else:
                    visible = vertices[vertices[:, 2] > 1e-5]
                    projected = visible @ camera.intrinsics.T
                    polygon = cv2.convexHull(np.float32(projected[:, :2]/projected[:, 2:3]))
                    mask = np.zeros(robot_mask.shape, np.uint8)
                    cv2.fillConvexPoly(mask, np.int32(np.rint(np.clip(polygon, -100000, 100000))), 1)
                    silhouette = raster & mask[rows, cols].astype(bool)
                candidates = np.flatnonzero(silhouette & ~protected[ids] & keep[ids])
                for begin in range(0, len(candidates), 1024):
                    chosen = candidates[begin:begin+1024]
                    entry = shape.ray_entry_distances(origin, points[ids[chosen]])
                    finite = np.isfinite(entry)
                    # Foreground surfaces survive even inside a projected robot
                    # outline, including points that SAM mislabeled as robot.
                    foreground = measured[chosen] < entry-self.config.self_surface_band_m
                    coincident = np.abs(measured[chosen]-entry) <= self.config.self_surface_band_m
                    own = (semantic[chosen] & ~foreground) | coincident
                    keep[ids[chosen[finite & own]]] = False
        evidence.update(status="captured_pose_model_and_robot_masks", removed=int((~keep).sum()),
                        foreground_and_nonrobot_owned_preserved=True,
                        convex_silhouette_is_not_visibility_truth=True)
        return keep, evidence

    def _prepare(self, snapshot, job):
        points = np.asarray(snapshot.scene_points, float)
        owners = np.asarray(snapshot.scene_point_object_ids, str)
        finite = np.flatnonzero(np.isfinite(points).all(axis=1))
        selected, radius, inverse, _ = _voxel_sample(points[finite], owners[finite], self.config.voxel_m,
                                                     max(1, len(finite)))
        indices = finite[selected]
        sampled, sampled_owners = points[indices], owners[indices]
        keep, self_evidence = self._self_filter(snapshot, sampled, sampled_owners, indices, job.state, job.joints)
        sampled, sampled_owners = sampled[keep], sampled_owners[keep]
        # A retained robot-labelled surface is unresolved self geometry, not a
        # trustworthy external obstacle merely because SAM gave it an owner.
        robot_ids = {key for key, obj in snapshot.objects.items() if obj.evidence.get("role") == "robot"}
        unresolved_robot = np.isin(sampled_owners, list(robot_ids))
        source_support = np.concatenate([v.support for v in snapshot.geometry._views.values()])
        source_conflict = np.concatenate([v.contradictions for v in snapshot.geometry._views.values()])
        # These are engineering uncertainty hypotheses, explicitly uncalibrated.
        raw_sigma = np.full(len(finite), self.config.observed_sigma_floor_m)
        raw_sigma[source_support[finite] < 2] = max(self.config.observed_sigma_floor_m,
                                                    self.engine.config.geometry.tolerance_abs_m)
        raw_sigma[source_conflict[finite] > 0] *= 2
        sigma = _voxel_sigma(raw_sigma, inverse, radius)[keep]
        sigma[unresolved_robot] *= 2
        now = snapshot.captured_at_s
        with self._cv:
            invalidated, self._invalidated = self._invalidated, set()
        for object_id in invalidated:
            self._objects.pop(object_id, None)
        # Retain anonymous surfaces only while unobserved or occluded. A new
        # visible surface behind an old point provides free-ray evidence that
        # invalidates that point. Named/held objects have their own lifecycle.
        retained_count = 0
        if self._unlabelled_memory is not None:
            old_points, old_sigma, old_seen = self._unlabelled_memory
            retain = now-old_seen <= self.config.retention_s
            for depth_view in snapshot.geometry._views.values():
                _, _, surface, _, free_ray, valid = _sample(old_points, depth_view, snapshot.geometry.config)
                retain &= ~(free_ray & valid) & ~surface
            old_points, old_sigma, old_seen = old_points[retain], old_sigma[retain], old_seen[retain]
            retained_count = len(old_points)
            if retained_count:
                sampled = np.concatenate((sampled, old_points))
                sampled_owners = np.concatenate((sampled_owners, np.full(retained_count, "", dtype=sampled_owners.dtype)))
                sigma = np.concatenate((sigma, old_sigma + self.config.voxel_m))
        else:
            old_seen = np.empty(0)
        memory = sampled_owners == ""
        memory_times = np.r_[np.full(len(sampled)-retained_count, now), old_seen]
        mids = np.flatnonzero(memory)
        selected, radius, inverse, _ = _voxel_sample(sampled[mids], sampled_owners[mids], self.config.voxel_m,
                                                      self.config.max_scene_points)
        memory_sigma = _voxel_sigma(sigma[mids], inverse, radius)
        mids = mids[selected]
        self._unlabelled_memory = (sampled[mids].copy(), memory_sigma, memory_times[mids].copy())
        remembered = {}
        for object_id, obj in snapshot.objects.items():
            if object_id in robot_ids:
                continue
            prior = self._objects.get(object_id)
            if len(obj.points):
                selected, radius, _, object_voxel = _voxel_sample(obj.points, np.full(len(obj.points), object_id),
                                                                  self.config.voxel_m, self.config.max_object_points)
                obj.points = np.asarray(obj.points[selected]).copy()
                obj.normals = np.asarray(obj.normals[selected]).copy()
                error = self.config.observed_sigma_floor_m
                if obj.status != "supported":
                    error = max(error, self.engine.config.geometry.tolerance_abs_m)
                error += float(radius.max(initial=0.))
                obj.evidence.update(sigma_m=[error]*3, uncertainty_calibrated=False,
                                    observed_at_s=now, last_seen_frame=job.frame.frame_id,
                                    geometry_age_s=0., complete_object_extent=False,
                                    voxel_m=object_voxel, representation_radius_m=float(radius.max(initial=0.)))
                # Plane is a local surface estimate, never an evaluator table.
                if len(obj.points) >= 6 and obj.evidence.get("role") == "support":
                    centre = np.median(obj.points, axis=0)
                    _, _, axes = np.linalg.svd(obj.points-centre, full_matrices=False)
                    normal = axes[-1]
                    if normal[2] < 0:
                        normal = -normal
                    residual = np.abs((obj.points-centre) @ normal)
                    obj.evidence["support_plane"] = {"normal": normal.tolist(), "point": centre.tolist(),
                        "sigma_m": max(error, float(np.quantile(residual, .9)))}
                remembered[object_id] = obj
            elif prior is not None and now-prior.evidence.get("observed_at_s", now) <= self.config.retention_s:
                retained = deepcopy(prior)
                retained.status = "unobserved"
                retained.masks = {}
                retained.evidence.update(geometry_age_s=max(0., now-prior.evidence.get("observed_at_s", now)),
                                         unobserved_reason=obj.status,
                                         observed_frame_id=prior.evidence.get("observed_frame_id"))
                remembered[object_id] = retained
            else:
                remembered[object_id] = obj
        # Re-insert dated named geometry only while occluded/unobserved; do not
        # downgrade missing parts to their parent's new whole-object geometry.
        for obj in remembered.values():
            if obj.status == "unobserved" and len(obj.points):
                sampled = np.concatenate((sampled, obj.points))
                sampled_owners = np.concatenate((sampled_owners.astype(object),
                                                 np.full(len(obj.points), obj.object_id, object)))
                sigma = np.r_[sigma, np.full(len(obj.points), max(obj.evidence.get("sigma_m", [.02])))]
        final_ids, radius, inverse, scene_voxel = _voxel_sample(sampled, sampled_owners,
                                               self.config.voxel_m, self.config.max_scene_points)
        sigma = _voxel_sigma(sigma, inverse, radius)
        sampled, sampled_owners = sampled[final_ids], sampled_owners[final_ids]
        # Targets remain in the contract catalog; retained hypotheses are dated.
        snapshot.objects = remembered
        self._objects = deepcopy(remembered)
        snapshot.scene_points = np.asarray(sampled)
        snapshot.scene_point_object_ids = np.asarray(sampled_owners, str)
        snapshot.evidence.pop("scene_point_view_slices", None)
        snapshot.evidence.update(revision=job.token, request_token=job.token, source_frame_id=job.frame.frame_id,
            targets_revision=job.target_key, captured_state8=job.state.tolist(),
            captured_joints=None if job.joints is None else job.joints.tolist(),
            point_sigma_m=sigma, self_filter=self_evidence,
            scene_voxel_m=scene_voxel, representation_radius_m=float(radius.max(initial=0.)),
            representation_radius_included_in_sigma=True,
            unresolved_robot_points=int(unresolved_robot.sum()),
            retained_unlabelled_points=retained_count, invalidated_object_ids=sorted(invalidated),
            perception_status="degraded" if "segmentation_error" in snapshot.evidence else "ok",
            source_point_count=len(points), scene_point_count=len(sampled),
            scene_points_include_robot_surfaces=bool(unresolved_robot.any()),
            all_observed_surfaces_retained_regardless_crossview_strength=True,
            unknown_space_is_free=False, uncertainty_calibrated=False)
        snapshot.generated_at_s = perf_counter()

    def _save(self, snapshot):
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"scene_{snapshot.evidence['revision']:06d}_frame_{snapshot.frame_id:06d}"
        arrays = {"scene_points": snapshot.scene_points.astype(np.float32),
                  "scene_owners": snapshot.scene_point_object_ids,
                  "point_sigma_m": snapshot.evidence["point_sigma_m"]}
        for i, camera in enumerate(snapshot.frame.cameras):
            arrays[f"camera_{i}_rgb"] = camera.rgb
            arrays[f"camera_{i}_intrinsics"] = camera.intrinsics
            arrays[f"camera_{i}_world_to_camera"] = camera.world_to_camera
        for i, obj in enumerate(snapshot.objects.values()):
            arrays[f"object_{i}_points"] = obj.points
            arrays[f"object_{i}_normals"] = obj.normals
        path = self.output_dir/(prefix+".npz")
        np.savez_compressed(path, **arrays)
        # No large cloud, point sigma list or repeated old geometry in text logs.
        metadata = {key: value for key, value in snapshot.evidence.items() if key != "point_sigma_m"}
        row = {"frame_id": snapshot.frame_id, "captured_at_s": snapshot.captured_at_s,
               "generated_at_s": snapshot.generated_at_s, "arrays": path.name,
               "objects": [{"id": obj.object_id, "status": obj.status, "evidence": obj.evidence}
                           for obj in snapshot.objects.values()], "evidence": metadata}
        with (self.output_dir/"snapshots.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(row), allow_nan=False)+"\n")
