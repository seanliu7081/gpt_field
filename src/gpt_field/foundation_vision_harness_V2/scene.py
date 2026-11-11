"""Dated visual evidence with separate surfaces and disputed measurements.

Identity disagreement is not a larger solid obstacle. The motion layer receives
unresolved geometry separately and decides whether its *next* movement needs a
fresh observation. All positional bounds below are engineering hypotheses, not
calibrated probabilities or collision certificates.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import json
from time import perf_counter

import numpy as np

from ..foundation_vision_harness_V1.scene import (
    SceneConfig as _V1Config, SceneService as _V1Service, _FramePerception as _V1Perception,
    _voxel_sample, scene_summary as _v1_summary,
)
from ..foundation_harness.perception import _normals
from ..foundation_harness.types import jsonable
from ..foundation_vision.geometry import build_geometry, _mask_on_depth, _project, _pixel_indices, _sample
from ..foundation_vision.types import TargetSpec


@dataclass(frozen=True)
class SceneConfig(_V1Config):
    depth_boundary_radius_px: int = 2
    image_border_margin_px: int = 2

    def __post_init__(self):
        super().__post_init__()
        if (type(self.depth_boundary_radius_px) is not int or self.depth_boundary_radius_px < 1
                or type(self.image_border_margin_px) is not int or self.image_border_margin_px < 0):
            raise ValueError("invalid measurement qualification neighbourhood")


def scene_summary(snapshot, now_s=None):
    """Only compact measurements/identity counts enter the language-model input."""
    result = _v1_summary(snapshot, now_s)
    if snapshot is not None:
        result.update(identity_counts=snapshot.evidence.get("identity_counts", {}),
                      identity_policy="qualified external surfaces; credible identity-unresolved surfaces; disputed measurements are not occupancy",
                      measurement_qualification=snapshot.evidence.get("measurement_qualification", {}),
                      uncertainty_calibrated=False)
        for obj, summary in zip(snapshot.objects.values(), result["objects"]):
            summary["position_error_m"] = obj.evidence.get("position_error_m")
            summary["representation_radius_m"] = obj.evidence.get("representation_radius_m")
    return result


class _FramePerception(_V1Perception):
    """Reuse material-region association; cache inference on immutable sources."""

    def __init__(self, engine, config):
        super().__init__(engine, config)
        self._frame_cache = OrderedDict()

    def process(self, job):
        self._frame = job.frame
        self._frame_id = self._source_frame = job.frame.frame_id
        self._observation_index = job.frame.frame_id
        self._raw_targets = {t["id"]: t for t in job.targets}
        targets = [{"id": t["id"], "query": t["query"], "parent_id": t.get("parent_id"),
                    "selections": {}, "region_pixels": {}} for t in job.targets]
        # A later selection on this source image changes regions, not its depth.
        key = (job.frame.frame_id, job.frame.captured_at_s)
        cached = self._frame_cache.setdefault(key, {"segments": {}})
        self._frame_cache.move_to_end(key)
        while len(self._frame_cache) > 2:
            self._frame_cache.popitem(last=False)
        specs = [TargetSpec(t["id"], t["query"]) for t in job.targets]
        queries = tuple((s.key, s.text) for s in specs)
        times, started = {}, perf_counter()
        segmentation_error = None
        missing = [spec for spec, query in zip(specs, queries) if query not in cached["segments"]]
        segmentation_hit = not missing
        if missing:
            try:
                added = tuple(self.engine.segmenter.segment(job.frame, missing))
                for spec in missing:
                    cached["segments"][(spec.key, spec.text)] = tuple(d for d in added if d.prompt_key == spec.key)
            except Exception as exc:
                segmentation_error = type(exc).__name__ + ": " + str(exc)
        detections = tuple(d for query in queries for d in cached["segments"].get(query, ()))
        times["segmentation_s"] = perf_counter()-started
        self._record_candidates(detections, targets)
        started = perf_counter()
        depth_hit = "depths" in cached
        if not depth_hit:
            cached["depths"] = self.engine.depth_estimator.estimate(job.frame)
        self._depths = cached["depths"]
        times["depth_s"] = perf_counter()-started
        started = perf_counter()
        if "geometry" not in cached:
            cached["geometry"] = build_geometry(job.frame, self._depths, {}, self.engine.config.geometry)
            cached["normals"] = {name: _normals(view) for name, view in cached["geometry"]._views.items()}
        self._geometry, self._normal_maps = cached["geometry"], cached["normals"]
        times["geometry_s"] = perf_counter()-started
        snapshot = self._assemble(targets)
        snapshot.evidence.update(timings_s=times, inference_cache={"depth_hit": depth_hit,
                                  "segmentation_hit": segmentation_hit, "source_key": list(key)})
        if segmentation_error:
            snapshot.evidence["segmentation_error"] = segmentation_error
        for target in job.targets:
            snapshot.objects[target["id"]].evidence.update(
                role=target.get("role", "object"), parent_id=target.get("parent_id"),
                observed_frame_id=job.frame.frame_id, region_identity="source_region_then_material_flow",
                source_region_frame=target.get("frame_id"))
        return snapshot


def _group_max(values, inverse, count):
    result = np.zeros(count, float)
    np.maximum.at(result, inverse, values)
    return result


def _depth_measurement_issues(view, config, geometry_config):
    """Reject contradictory rays and low-trust one-view image boundaries.

    Interior single-view surfaces remain usable. A second agreeing view can
    support a depth boundary; neither a text label nor an inflated sigma can.
    These are measurement flags, not a declaration that their space is free.
    """
    depth = np.asarray(view.estimate.depth_m)
    valid = np.asarray(view.valid)
    radius = getattr(config, "depth_boundary_radius_px", 2)
    border = getattr(config, "image_border_margin_px", 2)
    padded_low = np.pad(np.where(valid, depth, np.inf), radius, constant_values=np.inf)
    padded_high = np.pad(np.where(valid, depth, -np.inf), radius, constant_values=-np.inf)
    low, high = np.full(depth.shape, np.inf), np.full(depth.shape, -np.inf)
    height, width = depth.shape
    for dr in range(2*radius+1):
        for dc in range(2*radius+1):
            low = np.minimum(low, padded_low[dr:dr+height, dc:dc+width])
            high = np.maximum(high, padded_high[dr:dr+height, dc:dc+width])
    tolerance = geometry_config.tolerance_abs_m + geometry_config.tolerance_rel*np.maximum(np.abs(low), np.abs(high))
    discontinuity = valid & ((high-low) > tolerance)
    edge = np.zeros(depth.shape, bool)
    if border:
        edge[:border] = edge[-border:] = True
        edge[:, :border] = edge[:, -border:] = True
    issues = np.full(len(view.points_world), "", dtype="<U40")
    single = np.asarray(view.support) < 2
    issues[single & discontinuity[view.rows, view.cols]] = "single_view_depth_discontinuity"
    issues[single & edge[view.rows, view.cols]] = "single_view_image_border"
    issues[np.asarray(view.contradictions) > 0] = "crossview_depth_conflict"
    qualified = np.zeros(depth.shape, bool)
    qualified[view.rows, view.cols] = issues == ""
    return issues, qualified


def _measurement_evidence(snapshot, config):
    issues, semantic, view_ids, qualified = [], [], [], {}
    robots = [obj for obj in snapshot.objects.values() if obj.evidence.get("role") == "robot"]
    for ordinal, (name, view) in enumerate(snapshot.geometry._views.items()):
        flags, qualified[name] = _depth_measurement_issues(view, config, snapshot.geometry.config)
        mask = np.zeros(view.camera.rgb.shape[:2], bool)
        for obj in robots:
            if name in obj.masks:
                mask |= obj.masks[name]
        issues.append(flags)
        semantic.append(_mask_on_depth(mask, view))
        view_ids.append(np.full(len(flags), ordinal, int))
    return np.concatenate(issues), np.concatenate(semantic), np.concatenate(view_ids), qualified


def _measurement_groups(owners, support, contradictions, issues, semantic, view_ids):
    """Voxel representatives may only replace observations with equal evidence."""
    _, identities = np.unique(np.asarray(owners, str), return_inverse=True)
    _, flags = np.unique(np.asarray(issues, str), return_inverse=True)
    _, groups = np.unique(np.column_stack((identities, support, contradictions, flags,
                                          np.asarray(semantic, int), view_ids)), axis=0, return_inverse=True)
    return groups.astype(str)


def _memory_keep(points, geometry, qualified, *, replace_surface):
    """Only qualified current rays may invalidate previously credible surfaces."""
    keep = np.ones(len(points), bool)
    for name, view in geometry._views.items():
        rows, cols, surface, _, front, valid = _sample(points, view, geometry.config)
        trusted_ray = valid & qualified[name][rows, cols]
        keep &= ~(trusted_ray & (front | (surface if replace_surface else False)))
    return keep


def _qualified_object_points(obj, geometry, qualified):
    """Keep object/memory clouds consistent with qualification of source rays."""
    keep = np.zeros(len(obj.points), bool)
    for span in obj.evidence.get("point_support_ranges", []):
        name = span["view"]
        if name not in geometry._views:
            continue
        start, end = span["start"], span["end"]
        view = geometry._views[name]
        rows, cols, surface, _, _, valid = _sample(obj.points[start:end], view, geometry.config)
        keep[start:end] |= valid & surface & qualified[name][rows, cols]
    # Compatibility with scene providers that do not expose source ranges:
    # require at least one current, agreeing qualified ray.
    if not obj.evidence.get("point_support_ranges"):
        for name, view in geometry._views.items():
            rows, cols, surface, _, _, valid = _sample(obj.points, view, geometry.config)
            keep |= valid & surface & qualified[name][rows, cols]
    return keep & _memory_keep(obj.points, geometry, qualified, replace_surface=False)


def classify_ray_evidence(owners, robot_ids, semantic_robot, ray_residual_m,
                          support, contradictions, position_error_m, surface_band_m,
                          measurement_issues=None):
    """Classify signed measured-minus-frontmost-FK depth evidence.

    This function contains no object/task vocabulary. A SAM robot label alone
    never deletes a foreground surface. Names protect identity, not invalid
    geometry. Disputed measurements are separate from unresolved occupancy.
    """
    owners = np.asarray(owners, str)
    robot_owned = np.isin(owners, list(robot_ids))
    protected = (owners != "") & ~robot_owned
    semantic = np.asarray(semantic_robot, bool) | robot_owned
    residual = np.asarray(ray_residual_m, float)
    projected = np.isfinite(residual)
    consistent = np.asarray(contradictions) == 0
    corroborated = (np.asarray(support) >= 2) & consistent
    error = np.asarray(position_error_m, float)
    foreground = projected & (residual < -np.maximum(error, surface_band_m))
    near_surface = projected & (np.abs(residual) <= surface_band_m)
    semantic_agrees = semantic & projected & (residual >= -surface_band_m) & (residual <= error + surface_band_m)
    kinds = np.full(len(owners), "external", dtype="<U10")
    reasons = np.full(len(owners), "observed_external_surface", dtype="<U40")
    uncertain = ~protected & (semantic | (projected & (np.abs(residual) <= error + surface_band_m)) |
                              (projected & ~corroborated & ~foreground))
    kinds[uncertain] = "unresolved"
    reasons[uncertain] = "unlabelled_robot_projection"
    reasons[uncertain & semantic & ~projected] = "robot_label_outside_model"
    reasons[uncertain & semantic & projected] = "robot_depth_conflict"
    own = ~protected & consistent & (near_surface | (semantic_agrees & ~foreground))
    kinds[own], reasons[own] = "self", "captured_fk_surface_and_ray"
    # Depth agreement is evidence of a foreground surface, not proof SAM's
    # identity was correct. Keep it external under an anonymous owner.
    foreground_supported = ~protected & semantic & foreground & corroborated
    kinds[foreground_supported], reasons[foreground_supported] = "external", "corroborated_foreground"
    clean_owners = owners.copy()
    clean_owners[foreground_supported & robot_owned] = ""
    reasons[protected] = "named_external_surface"
    issues = (np.full(len(owners), "", dtype="<U40") if measurement_issues is None
              else np.asarray(measurement_issues, str))
    # A consistent FK match can identify self at an image border. It does not
    # turn an otherwise weak ray into an external or unresolved solid surface.
    low_trust = (issues != "") & (kinds != "self")
    kinds[low_trust], reasons[low_trust] = "disputed", issues[low_trust]
    contradictory = ~consistent
    kinds[contradictory], reasons[contradictory] = "disputed", "crossview_depth_conflict"
    return kinds, reasons, clean_owners


class SceneService(_V1Service):
    """V1 latest-wins worker API, with V2 evidence ownership and inference reuse."""

    def __init__(self, engine, output_dir=None, kinematics=None, config=None):
        super().__init__(engine, output_dir, kinematics, config or SceneConfig())
        # The worker is waiting for its first submit; no process-global patching.
        self._backend = _FramePerception(engine, self.config)

    def _ray_evidence(self, snapshot, points, source_indices, state, joints):
        semantic = np.zeros(len(points), bool)
        residual = np.full(len(points), np.nan)
        detail = {"status": "body_model_unavailable", "source": "captured_public_joints_and_model"}
        if not len(points):
            return semantic, residual, detail
        slices = snapshot.evidence.get("scene_point_view_slices", {})
        robot_objects = [o for o in snapshot.objects.values() if o.evidence.get("role") == "robot"]
        for camera in snapshot.frame.cameras:
            interval = slices.get(camera.name)
            if interval is None:
                continue
            ids = np.flatnonzero((source_indices >= interval[0]) & (source_indices < interval[1]))
            mask = np.zeros(camera.rgb.shape[:2], bool)
            for obj in robot_objects:
                if camera.name in obj.masks:
                    mask |= obj.masks[camera.name]
            if len(ids):
                semantic[ids] = _mask_on_depth(mask, snapshot.geometry._views[camera.name])[source_indices[ids]-interval[0]]
        if joints is None:
            return semantic, residual, detail
        from ..foundation_harness.robot_geometry import RobotCollisionModel
        import cv2
        if self.kinematics is None:
            from ..robodojo_direct_sim import PandaKinematics
            self.kinematics = PandaKinematics.from_xml()
        if self._robot is None:
            self._robot = RobotCollisionModel(self.kinematics, include_visual=True)
        bound = self.kinematics.bind(state, joints)
        shapes = self._robot.shapes(bound, joints, state[6:8])
        for camera in snapshot.frame.cameras:
            interval = slices.get(camera.name)
            if interval is None:
                continue
            ids = np.flatnonzero((source_indices >= interval[0]) & (source_indices < interval[1]))
            if not len(ids):
                continue
            origin = -camera.world_to_camera[:3, :3].T @ camera.world_to_camera[:3, 3]
            uv, _ = _project(points[ids], camera.intrinsics, camera.world_to_camera)
            rows, cols, raster = _pixel_indices(uv, camera.rgb.shape[:2])
            entry = np.full(len(ids), np.inf)
            for shape in shapes:
                # An enclosure containing the camera does not model its aperture.
                if shape.contains(origin[None], 1e-7)[0]:
                    continue
                vertices = shape.surface_points() @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
                visible = vertices[vertices[:, 2] > 1e-5]
                if len(visible) < 3:
                    continue
                projected = visible @ camera.intrinsics.T
                polygon = cv2.convexHull(np.float32(projected[:, :2]/projected[:, 2:3]))
                mask = np.zeros(camera.rgb.shape[:2], np.uint8)
                cv2.fillConvexPoly(mask, np.int32(np.rint(np.clip(polygon, -100000, 100000))), 1)
                candidates = np.flatnonzero(raster & mask[rows, cols].astype(bool))
                for begin in range(0, len(candidates), 1024):
                    chosen = candidates[begin:begin+1024]
                    hit = shape.ray_entry_distances(origin, points[ids[chosen]])
                    entry[chosen] = np.minimum(entry[chosen], hit)
            finite = np.isfinite(entry)
            residual[ids[finite]] = np.linalg.norm(points[ids[finite]]-origin, axis=1)-entry[finite]
        detail.update(status="captured_fk_frontmost_ray", ray_intersections=int(np.isfinite(residual).sum()),
                      convex_model_visibility_is_provisional=True)
        return semantic, residual, detail

    def _prepare(self, snapshot, job):
        points = np.asarray(snapshot.scene_points, float)
        owners = np.asarray(snapshot.scene_point_object_ids, str)
        source_support = np.concatenate([v.support for v in snapshot.geometry._views.values()])
        source_conflict = np.concatenate([v.contradictions for v in snapshot.geometry._views.values()])
        issues, raw_semantic, view_ids, qualified = _measurement_evidence(snapshot, self.config)
        finite = np.flatnonzero(np.isfinite(points).all(axis=1))
        groups = _measurement_groups(owners[finite], source_support[finite], source_conflict[finite],
                                     issues[finite], raw_semantic[finite], view_ids[finite])
        selected, radius, inverse, _ = _voxel_sample(points[finite], groups, self.config.voxel_m,
                                                   max(1, len(finite)))
        indices = finite[selected]
        sampled, original_owners = points[indices], owners[indices]
        raw_error = np.where(source_support[finite] >= 2, self.config.observed_sigma_floor_m,
                             max(self.config.observed_sigma_floor_m, self.engine.config.geometry.tolerance_abs_m))
        error = _group_max(raw_error, inverse, len(selected))
        semantic, residual, self_detail = self._ray_evidence(snapshot, sampled, indices, job.state, job.joints)
        robot_ids = {key for key, obj in snapshot.objects.items() if obj.evidence.get("role") == "robot"}
        kinds, reasons, clean_owners = classify_ray_evidence(
            original_owners, robot_ids, semantic, residual, source_support[indices],
            source_conflict[indices], error, self.config.self_surface_band_m, issues[indices])
        separate = {}
        for name in ("self", "unresolved", "disputed"):
            subset = kinds == name
            separate.update({name+"_points": sampled[subset], name+"_owners": original_owners[subset],
                             name+"_reason": reasons[subset], name+"_source_indices": indices[subset],
                             name+"_position_error_m": error[subset],
                             name+"_representation_radius_m": radius[subset],
                             name+"_ray_residual_m": residual[subset],
                             name+"_support": source_support[indices[subset]],
                             name+"_contradictions": source_conflict[indices[subset]]})
        # Retain all disputed source rays, not only their voxel representatives.
        disputed_groups = np.flatnonzero(kinds == "disputed")
        disputed_ids = np.flatnonzero(np.isin(inverse, disputed_groups))
        order = disputed_ids[np.argsort(inverse[disputed_ids], kind="stable")]
        lengths = np.bincount(inverse, minlength=len(indices))[disputed_groups]
        separate.update(disputed_source_group_indices=finite[order],
                        disputed_source_group_offsets=np.r_[0, np.cumsum(lengths)])
        external = kinds == "external"
        sampled, owners = sampled[external], clean_owners[external]
        position, representation = error[external], radius[external]
        original_owner = original_owners[external]
        source_index = indices[external]
        reason = reasons[external]
        support = source_support[indices[external]]
        frames = np.full(len(sampled), job.frame.frame_id, int)
        seen = np.full(len(sampled), snapshot.captured_at_s)
        now = snapshot.captured_at_s
        with self._cv:
            invalidated, self._invalidated = self._invalidated, set()
        for object_id in invalidated:
            self._objects.pop(object_id, None)
        retained_count = 0
        if self._unlabelled_memory is not None:
            old = self._unlabelled_memory
            retain = now-old["seen"] <= self.config.retention_s
            retain &= _memory_keep(old["points"], snapshot.geometry, qualified, replace_surface=True)
            retained_count = int(retain.sum())
            if retained_count:
                sampled = np.r_[sampled, old["points"][retain]]
                owners = np.r_[owners, np.full(retained_count, "")]
                original_owner = np.r_[original_owner, np.full(retained_count, "")]
                position = np.r_[position, old["position"][retain]]
                representation = np.r_[representation, old["representation"][retain]]
                frames, seen = np.r_[frames, old["frames"][retain]], np.r_[seen, old["seen"][retain]]
                source_index = np.r_[source_index, np.full(retained_count, -1, int)]
                reason = np.r_[reason, np.full(retained_count, "dated_external_surface")]
                support = np.r_[support, old.get("support", np.ones(len(old["points"]), int))[retain]]
        remembered = {}
        for object_id, obj in snapshot.objects.items():
            if object_id in robot_ids:
                continue
            prior = self._objects.get(object_id)
            if len(obj.points):
                accepted = _qualified_object_points(obj, snapshot.geometry, qualified)
                rejected = int(np.count_nonzero(~accepted))
                obj.points, obj.normals = obj.points[accepted], obj.normals[accepted]
                obj.evidence.update(disputed_measurement_points=rejected,
                    measurement_qualification="current qualified source rays; disputed rays do not define extent")
                if rejected:
                    obj.evidence.pop("point_support_ranges", None)
                if not len(obj.points):
                    obj.status = "conflict"
            if len(obj.points):
                selected, extent, _, voxel = _voxel_sample(obj.points, np.full(len(obj.points), object_id),
                                                         self.config.voxel_m, self.config.max_object_points)
                obj.points, obj.normals = obj.points[selected].copy(), obj.normals[selected].copy()
                bound = self.config.observed_sigma_floor_m if obj.status == "supported" else max(
                    self.config.observed_sigma_floor_m, self.engine.config.geometry.tolerance_abs_m)
                rep = float(extent.max(initial=0.))
                obj.evidence.update(position_error_m=bound, representation_radius_m=rep, sigma_m=[bound+rep]*3,
                    uncertainty_calibrated=False, error_provenance="crossview_engineering_bound",
                    observed_at_s=now, last_seen_frame=job.frame.frame_id, geometry_age_s=0.,
                    complete_object_extent=False, voxel_m=voxel)
                if len(obj.points) >= 6 and obj.evidence.get("role") == "support":
                    centre = np.median(obj.points, axis=0)
                    _, _, axes = np.linalg.svd(obj.points-centre, full_matrices=False)
                    normal = axes[-1] * (1 if axes[-1, 2] >= 0 else -1)
                    obj.evidence["support_plane"] = {"normal": normal.tolist(), "point": centre.tolist(),
                        "sigma_m": max(bound+rep, float(np.quantile(np.abs((obj.points-centre) @ normal), .9)))}
            elif prior is not None and now-prior.evidence.get("observed_at_s", now) <= self.config.retention_s:
                missing_status = obj.status
                obj = deepcopy(prior)
                retain = _memory_keep(obj.points, snapshot.geometry, qualified, replace_surface=False)
                obj.points, obj.normals = obj.points[retain], obj.normals[retain]
                obj.status, obj.masks = "unobserved", {}
                obj.evidence.update(geometry_age_s=max(0., now-prior.evidence["observed_at_s"]),
                                    unobserved_reason=missing_status)
            remembered[object_id] = obj
            if obj.status == "unobserved" and len(obj.points):
                n = len(obj.points)
                sampled, owners = np.r_[sampled, obj.points], np.r_[owners, np.full(n, object_id)]
                original_owner = np.r_[original_owner, np.full(n, object_id)]
                position = np.r_[position, np.full(n, obj.evidence["position_error_m"])]
                representation = np.r_[representation, np.full(n, obj.evidence["representation_radius_m"])]
                frames = np.r_[frames, np.full(n, obj.evidence["observed_frame_id"], int)]
                seen = np.r_[seen, np.full(n, obj.evidence["observed_at_s"])]
                source_index = np.r_[source_index, np.full(n, -1, int)]
                reason = np.r_[reason, np.full(n, "dated_external_surface")]
                support = np.r_[support, np.full(n, 1, int)]
        # Do not promote one-view or dated evidence by merging it with stronger
        # observations in the final storage representation.
        groups = _measurement_groups(owners, support, np.zeros(len(sampled), int), reason,
                                     np.zeros(len(sampled), bool), frames)
        chosen, radius, inverse, voxel = _voxel_sample(sampled, groups, self.config.voxel_m,
                                                       self.config.max_scene_points)
        position = _group_max(position, inverse, len(chosen))
        representation = _group_max(representation, inverse, len(chosen))+radius
        # Conservatively date merged memory by its oldest contributing source.
        merged_seen, merged_frames = np.full(len(chosen), np.inf), np.full(len(chosen), np.iinfo(np.int64).max, np.int64)
        np.minimum.at(merged_seen, inverse, seen)
        np.minimum.at(merged_frames, inverse, frames)
        sampled, owners = sampled[chosen], owners[chosen]
        anonymous = owners == ""
        self._unlabelled_memory = {"points": sampled[anonymous].copy(), "position": position[anonymous].copy(),
            "representation": representation[anonymous].copy(), "seen": merged_seen[anonymous],
            "frames": merged_frames[anonymous], "support": support[chosen][anonymous].copy()}
        snapshot.objects = remembered
        self._objects = deepcopy(remembered)
        snapshot.scene_points, snapshot.scene_point_object_ids = sampled, owners
        snapshot.evidence.update(separate)
        counts = {name: int(np.count_nonzero(kinds == name)) for name in ("external", "self", "unresolved", "disputed")}
        self_detail.update(removed=counts["self"], robot_query_points=int(semantic.sum()),
                           credible_foreground_and_nonrobot_identity_preserved=True,
                           named_identity_does_not_override_measurement_conflict=True)
        snapshot.evidence.update(revision=job.token, request_token=job.token, source_frame_id=job.frame.frame_id,
            targets_revision=job.target_key, captured_state8=job.state.tolist(),
            captured_joints=None if job.joints is None else job.joints.tolist(),
            point_sigma_m=position+representation, point_position_error_m=position,
            point_representation_radius_m=representation, point_identity=np.full(len(sampled), "external"),
            point_reason=reason[chosen], point_source_indices=source_index[chosen],
            point_support=support[chosen], point_contradictions=np.zeros(len(sampled), int),
            point_original_owners=original_owner[chosen], point_observed_frame_id=merged_frames,
            point_observed_at_s=merged_seen, self_filter=self_detail, identity_counts=counts,
            scene_voxel_m=voxel, representation_radius_m=float(representation.max(initial=0.)),
            representation_radius_included_in_sigma=True, unresolved_robot_points=counts["unresolved"],
            retained_unlabelled_points=retained_count, invalidated_object_ids=sorted(invalidated),
            perception_status="degraded" if "segmentation_error" in snapshot.evidence else "ok",
            source_point_count=len(points), scene_point_count=len(sampled),
            scene_points_include_robot_surfaces=False, unknown_space_is_free=False,
            measurement_qualification={"disputed_points": counts["disputed"],
                "disputed_raw_measurements": len(order), "disputed_is_occupancy": False,
                "unseen_and_rejected_rays": "unknown; no free-space certificate",
                "single_view_interior_surfaces_allowed": True,
                "voxel_groups_preserve_measurement_evidence": True,
                "memory_invalidation_requires_qualified_current_ray": True,
                "depth_boundary_radius_px": getattr(self.config, "depth_boundary_radius_px", 2),
                "image_border_margin_px": getattr(self.config, "image_border_margin_px", 2)},
            uncertainty_calibrated=False, error_provenance="crossview_engineering_bound_not_calibrated_sigma",
            identity_policy="qualified_external_in_scene_points; credible_identity_unknown_in_unresolved; disputed_measurements_are_not_solids")
        snapshot.generated_at_s = perf_counter()

    def _save(self, snapshot):
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"scene_{snapshot.evidence['revision']:06d}_frame_{snapshot.frame_id:06d}"
        path = self.output_dir/(prefix+".npz")
        if path.exists():
            raise FileExistsError(f"refusing to replace scene artifact: {path}")
        arrays = {"scene_points": snapshot.scene_points.astype(np.float32), "scene_owners": snapshot.scene_point_object_ids}
        arrays.update({key: value for key, value in snapshot.evidence.items() if isinstance(value, np.ndarray)})
        for i, camera in enumerate(snapshot.frame.cameras):
            arrays.update({f"camera_{i}_rgb": camera.rgb, f"camera_{i}_intrinsics": camera.intrinsics,
                           f"camera_{i}_world_to_camera": camera.world_to_camera})
            estimate = self._backend._depths.get(camera.name)
            if estimate is not None:
                arrays[f"camera_{i}_depth_m"] = estimate.depth_m.astype(np.float32)
                arrays[f"camera_{i}_depth_intrinsics"] = estimate.intrinsics
        masks = []
        for i, record in enumerate(self._backend._candidate_records.values()):
            key = f"candidate_{i}_mask_bits"
            arrays[key] = np.packbits(record["mask"])
            masks.append({**record["description"], "mask_bits_key": key, "shape": list(record["mask"].shape)})
        for i, obj in enumerate(snapshot.objects.values()):
            arrays[f"object_{i}_points"], arrays[f"object_{i}_normals"] = obj.points, obj.normals
        np.savez_compressed(path, **arrays)
        metadata = {key: value for key, value in snapshot.evidence.items() if not isinstance(value, np.ndarray)}
        row = {"frame_id": snapshot.frame_id, "captured_at_s": snapshot.captured_at_s,
            "generated_at_s": snapshot.generated_at_s, "arrays": path.name,
            "cameras": [c.name for c in snapshot.frame.cameras], "candidate_masks": masks,
            "objects": [{"id": o.object_id, "query": o.query, "status": o.status, "evidence": o.evidence}
                        for o in snapshot.objects.values()], "evidence": metadata}
        with (self.output_dir/"snapshots.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(row), allow_nan=False)+"\n")
