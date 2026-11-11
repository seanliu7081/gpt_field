"""Lift a named visible contact point into a grip target without inventing a grasp.

The model owns material selection, approach direction and orientation. This
module changes position from the selected pixel's local predicted depth, tool
contact offset and requested standoff. It never replaces that material point
with a whole-object centre. Scene/model uncertainty remains explicit evidence.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from ..foundation_harness.types import jsonable
from ..foundation_vision.geometry import _independent, _project, _sample
from ..robodojo_direct_sim import quaternion_matrix


@dataclass(frozen=True)
class GroundingConfig:
    max_correction_m: float = .05
    max_reference_motion_m: float = .05
    max_position_error_m: float = .04
    patch_radius_px: int = 2
    min_patch_samples: int = 3
    min_patch_spread_m: float = .008
    tracking_radius_px: int = 35


class GroundingUnavailable(ValueError):
    def __init__(self, reason, **evidence):
        super().__init__(reason)
        self.reason, self.evidence = reason, evidence


def _vector(value, n, name):
    vector = np.asarray(value, float)
    if vector.shape != (n,) or not np.isfinite(vector).all():
        raise GroundingUnavailable("invalid_grounding_reference", field=name)
    return vector


def _region(scene, region_id, view_name):
    if scene is None or scene.geometry is None or scene.frame is None:
        raise GroundingUnavailable("contact_scene_unavailable")
    obj = scene.objects.get(region_id)
    if (obj is None or obj.status in {"missing", "unobserved", "stale", "ambiguous", "conflict"}
            or obj.evidence.get("observed_frame_id", scene.frame_id) != scene.frame_id):
        raise GroundingUnavailable("contact_region_unobserved", region=region_id)
    if obj.evidence.get("role") == "robot":
        raise GroundingUnavailable("robot_is_not_contact_material")
    camera = next((c for c in scene.frame.cameras if c.name == view_name), None)
    depth_view = scene.geometry._views.get(view_name)
    mask = obj.masks.get(view_name)
    if camera is None or depth_view is None or mask is None or np.shape(mask) != camera.rgb.shape[:2]:
        raise GroundingUnavailable("contact_view_unobserved", region=region_id, view=view_name)
    return obj, camera, depth_view, np.asarray(mask, bool)


def _pixel_index(pixel, shape):
    h, w = shape
    if not (0 <= pixel[0] <= w-1 and 0 <= pixel[1] <= h-1):
        raise GroundingUnavailable("contact_pixel_outside_image", pixel=pixel.tolist())
    col, row = np.floor(pixel+.5).astype(int)
    return row, col


def _lift(scene, region_id, name, pixel, config):
    obj, camera, view, mask = _region(scene, region_id, name)
    row, col = _pixel_index(pixel, mask.shape)
    if not mask[row, col]:
        raise GroundingUnavailable("contact_pixel_outside_selected_region", region=region_id,
                                   pixel=pixel.tolist(), view=name)
    raw_h, raw_w = mask.shape
    h, w = view.estimate.depth_m.shape
    depth_pixel = (pixel+.5)*np.array([w/raw_w, h/raw_h])-.5
    dr, dc = _pixel_index(depth_pixel, (h, w))
    if not view.valid[dr, dc]:
        raise GroundingUnavailable("contact_depth_unavailable", view=name, pixel=pixel.tolist())
    if view.contradiction_map[dr, dc] > 0:
        raise GroundingUnavailable("contact_depth_crossview_conflict", view=name)
    radius = config.patch_radius_px
    rr, cc = np.mgrid[max(0, dr-radius):min(h, dr+radius+1),
                      max(0, dc-radius):min(w, dc+radius+1)]
    raw_col = np.floor((cc+.5)*raw_w/w).astype(int).clip(0, raw_w-1)
    raw_row = np.floor((rr+.5)*raw_h/h).astype(int).clip(0, raw_h-1)
    good = view.valid[rr, cc] & mask[raw_row, raw_col] & (view.contradiction_map[rr, cc] == 0)
    rr, cc = rr[good], cc[good]
    if len(rr) < config.min_patch_samples:
        raise GroundingUnavailable("contact_depth_patch_insufficient", samples=len(rr))
    depths = view.estimate.depth_m[rr, cc]
    error = float(np.max(np.asarray(obj.evidence.get("position_error_m", obj.evidence.get("sigma_m", .02)))))
    if not np.isfinite(error) or error > config.max_position_error_m:
        raise GroundingUnavailable("contact_position_error_unresolved", position_error_m=error)
    spread = float(np.quantile(depths, .9)-np.quantile(depths, .1))
    if spread > max(config.min_patch_spread_m, 2*error):
        raise GroundingUnavailable("contact_depth_discontinuity", patch_spread_m=spread)
    depth = float(np.median(depths))
    ray = np.linalg.solve(camera.intrinsics, np.r_[pixel, 1.])
    point = (ray*depth-camera.world_to_camera[:3, 3]) @ camera.world_to_camera[:3, :3]
    supported_views = [name]
    for other_name, other in scene.geometry._views.items():
        if other_name == name or not _independent(view, other, scene.geometry.config):
            continue
        rows, cols, surface, occluded, front, valid = _sample(point[None], other, scene.geometry.config)
        if valid[0] and front[0]:
            raise GroundingUnavailable("contact_depth_crossview_conflict", view=other_name)
        other_mask = obj.masks.get(other_name)
        if valid[0] and surface[0] and other_mask is not None:
            uv, _ = _project(point[None], other.camera.intrinsics, other.camera.world_to_camera)
            other_row, other_col = _pixel_index(uv[0], other_mask.shape)
            if not other_mask[other_row, other_col]:
                raise GroundingUnavailable("contact_identity_crossview_conflict", view=other_name)
            supported_views.append(other_name)
    patch_pixels = np.column_stack((cc, rr, np.ones(len(rr))))
    patch_camera = (patch_pixels @ np.linalg.inv(view.estimate.intrinsics).T)*depths[:, None]
    patch_world = (patch_camera-view.estimate.world_to_camera[:3, 3]) @ view.estimate.world_to_camera[:3, :3]
    return point, patch_world, obj, {"region": region_id, "view": name, "pixel": pixel.tolist(),
        "point_world_m": point.tolist(), "position_error_m": error, "patch_spread_m": spread,
        "patch_samples": len(rr), "supporting_views": supported_views,
        "depth_source": "predicted_DA3_local_masked_patch", "uncertainty_is_calibrated": False}


def _track_pixel(source, current, region_id, name, pixel, config):
    """Keep the selected visible material point; never shift by body centroid."""
    _, old_camera, _, old_mask = _region(source, region_id, name)
    _, camera, _, mask = _region(current, region_id, name)
    if any(source.objects[region_id].evidence.get(key) != current.objects[region_id].evidence.get(key)
           for key in ("parent_id", "role")):
        raise GroundingUnavailable("contact_region_identity_changed")
    if source.frame_id == current.frame_id:
        if (source.captured_at_s != current.captured_at_s
                or not np.array_equal(old_camera.world_to_camera, camera.world_to_camera)):
            raise GroundingUnavailable("contact_source_frame_mismatch")
        return pixel.copy(), {"method": "same_source_frame"}
    if current.frame_id < source.frame_id or current.captured_at_s < source.captured_at_s:
        raise GroundingUnavailable("contact_observation_older_than_reference")
    import cv2
    old_gray = cv2.cvtColor(old_camera.rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2GRAY)
    row, col = _pixel_index(pixel, old_mask.shape)
    h, w = old_mask.shape
    radius = config.tracking_radius_px
    window = np.zeros_like(old_mask)
    window[max(0, row-radius):min(h, row+radius+1), max(0, col-radius):min(w, col+radius+1)] = True
    local = window & old_mask
    # In a fixed camera, unchanged local material appearance in the still-bound
    # region supports using the same ray even when the rest of the image moves.
    if (mask.shape == old_mask.shape and mask[row, col]
            and np.allclose(old_camera.world_to_camera, camera.world_to_camera, atol=1e-8, rtol=0)
            and np.allclose(old_camera.intrinsics, camera.intrinsics, atol=1e-8, rtol=0)
            and np.any(local) and np.max(np.abs(old_gray[local].astype(float)-gray[local])) <= 2):
        return pixel.copy(), {"method": "fixed_camera_unchanged_local_region"}
    features = cv2.goodFeaturesToTrack(old_gray, maxCorners=50, qualityLevel=.01,
                                      minDistance=3, mask=np.uint8(local)*255)
    if features is None or len(features) < 4:
        raise GroundingUnavailable("contact_material_correspondence_unavailable")
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, features, None, winSize=(21, 21), maxLevel=3)
    if moved is None or ok is None:
        raise GroundingUnavailable("contact_material_correspondence_lost")
    back, bok, _ = cv2.calcOpticalFlowPyrLK(gray, old_gray, moved, None, winSize=(21, 21), maxLevel=3)
    if back is None or bok is None:
        raise GroundingUnavailable("contact_material_correspondence_lost")
    good = ok[:, 0].astype(bool) & bok[:, 0].astype(bool)
    good &= np.isfinite(moved[:, 0]).all(1) & (np.linalg.norm(back[:, 0]-features[:, 0], axis=1) < 1.5)
    if np.count_nonzero(good) < 4:
        raise GroundingUnavailable("contact_material_correspondence_lost")
    affine, inliers = cv2.estimateAffinePartial2D(features[good, 0], moved[good, 0],
        method=cv2.RANSAC, ransacReprojThreshold=2.)
    if affine is None or inliers is None or int(inliers.sum()) < 4:
        raise GroundingUnavailable("contact_local_motion_unresolved")
    scale = float(np.linalg.norm(affine[:, 0]))
    if not .65 <= scale <= 1.5:
        raise GroundingUnavailable("contact_local_scale_unresolved", scale=scale)
    tracked = affine @ np.r_[pixel, 1.]
    row, col = _pixel_index(tracked, mask.shape)
    if not mask[row, col]:
        raise GroundingUnavailable("tracked_contact_outside_selected_region")
    return tracked, {"method": "local_material_affine_flow", "inliers": int(inliers.sum()), "scale": scale}


def _payload_offset(held, rotation, obj, patch):
    if held is None or held.get("extent_unknown"):
        raise GroundingUnavailable("payload_contact_extent_unavailable")
    points = np.asarray(held.get("points_in_gripper", []), float).reshape(-1, 3)
    points = points[np.isfinite(points).all(1)]
    if len(points) < 8:
        raise GroundingUnavailable("payload_contact_extent_unavailable")
    normal = (obj.evidence.get("support_plane") or {}).get("normal")
    if normal is None:
        _, values, axes = np.linalg.svd(patch-patch.mean(0), full_matrices=False)
        if len(values) < 3 or values[1] < 1e-6 or values[-1] > .3*values[1]:
            raise GroundingUnavailable("placement_support_plane_unresolved")
        normal = axes[-1]
    normal = _vector(normal, 3, "support_normal")
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if normal[2] < 0:
        normal = -normal
    if normal[2] < .5:
        raise GroundingUnavailable("placement_support_not_upward", normal=normal.tolist())
    heights = (points @ rotation.T) @ normal
    contact = points[heights <= heights.min()+.002]
    # This is the observed bottom support patch, not the whole object's centre.
    return contact.mean(0), {"source": "observed_payload_support_patch", "points": len(contact),
        "normal_world": normal.tolist(), "complete_object_shape": bool(held.get("geometry_complete", False)),
        "payload_object": held.get("object_id")}


def ground_candidate(candidate, segment, source_scene, current_scene, *, held=None, config=None):
    """Return grounded/nominal/unresolved without an inference or robot action.

    ``candidate`` has already resolved relative displacement and null rotation
    from the measured segment entry. ``grounding.frame_id`` is supplied by the
    protocol normalizer from the request frame, never guessed from current data.
    The caller must handle unresolved evidence before dispatching this candidate.
    """
    config = config or GroundingConfig()
    resolved, reference = deepcopy(candidate), segment.get("grounding")
    if reference is None:
        return {"status": "nominal", "candidate": resolved, "reason": "no_contact_reference", "evidence": {}}
    evidence = {"orientation_owner": "astra", "whole_object_centroid_used": False,
                "initial_contact_position_correction": True}
    try:
        if candidate.get("skill") != "move_to_pose":
            raise GroundingUnavailable("contact_grounding_requires_resolved_motion")
        if source_scene is None or reference.get("frame_id") != source_scene.frame_id:
            raise GroundingUnavailable("contact_source_frame_mismatch", reference_frame=reference.get("frame_id"),
                                       source_frame=None if source_scene is None else source_scene.frame_id)
        region, name = reference["region"], reference["view"]
        pixel = _vector(reference["pixel"], 2, "pixel")
        source_point, _, source_obj, source_evidence = _lift(source_scene, region, name, pixel, config)
        if (segment.get("contact_mode") in {"grasp", "articulated"}
                and (segment.get("object") is None or segment["object"] not in
                     {region, source_obj.evidence.get("parent_id")})):
            raise GroundingUnavailable("contact_material_object_mismatch", region=region,
                                       expected_object=segment.get("object"))
        current_pixel, track = _track_pixel(source_scene, current_scene, region, name, pixel, config)
        point, patch, obj, measurement = _lift(current_scene, region, name, current_pixel, config)
        evidence.update(source=source_evidence, measurement=measurement, correspondence=track,
                        source_frame=source_scene.frame_id, observed_frame=current_scene.frame_id)
        moved = float(np.linalg.norm(point-source_point))
        if moved > config.max_reference_motion_m:
            raise GroundingUnavailable("contact_reference_changed", displacement_m=moved)
        nominal = None if candidate.get("xyz_m") is None else _vector(candidate["xyz_m"], 3, "candidate_position")
        quat = _vector(candidate["quaternion_wxyz"], 4, "candidate_orientation")
        if np.linalg.norm(quat) < 1e-8:
            raise GroundingUnavailable("candidate_orientation_unresolved")
        rotation = quaternion_matrix(quat)
        standoff = float(reference["standoff_m"])
        if not np.isfinite(standoff) or not 0 <= standoff <= .20:
            raise GroundingUnavailable("invalid_contact_standoff")
        if reference["reference"] == "payload_support":
            if segment.get("contact_mode") != "place":
                raise GroundingUnavailable("payload_support_requires_placement")
            if held is not None and segment.get("object") not in {None, held.get("object_id")}:
                raise GroundingUnavailable("placement_payload_object_mismatch",
                    expected_object=segment["object"], observed_payload=held.get("object_id"))
            contact_offset, payload_evidence = _payload_offset(held, rotation, obj, patch)
            evidence["payload_contact"] = payload_evidence
        elif reference["reference"] == "tool":
            contact_offset = _vector(reference["offset_hand_m"], 3, "offset_hand_m")
        else:
            raise GroundingUnavailable("unknown_contact_reference")
        goal = point-rotation @ contact_offset-rotation[:, 2]*standoff
        correction = None if nominal is None else goal-nominal
        evidence.update(nominal_xyz_m=None if nominal is None else nominal.tolist(), corrected_xyz_m=goal.tolist(),
            correction_world_m=None if correction is None else correction.tolist(),
            correction_norm_m=None if correction is None else float(np.linalg.norm(correction)),
            position_source="material_point_projection" if nominal is None else "corrected_metric_candidate",
            nominal_correction_unavailable=nominal is None,
            contact_offset_hand_m=contact_offset.tolist(), standoff_m=standoff,
            orientation_changed=False, observed_contact_is_physical_contact=False)
        if correction is not None and np.linalg.norm(correction) > config.max_correction_m:
            raise GroundingUnavailable("contact_correction_exceeds_local_scope", limit_m=config.max_correction_m)
        resolved["xyz_m"] = goal.tolist()
        return {"status": "grounded", "candidate": resolved, "reason": "selected_material_point_lifted",
                "evidence": jsonable(evidence)}
    except (GroundingUnavailable, KeyError, ValueError, np.linalg.LinAlgError) as exc:
        evidence.update(getattr(exc, "evidence", {"message": str(exc)}))
        return {"status": "unresolved", "candidate": None,
                "reason": getattr(exc, "reason", "contact_geometry_unavailable"), "evidence": jsonable(evidence)}
