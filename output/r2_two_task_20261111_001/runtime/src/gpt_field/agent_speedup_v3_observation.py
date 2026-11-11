"""Bounded, model-geometry-checked viewing proposals for a specific evidence gap.

Camera calibration and robot/scene estimates are the only inputs. A proposal is
an information-gathering hypothesis, never a free-space or visibility proof.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .astranomy_agentic_motion import measured_pose
from .agent_speedup_v2_user_check_fragments import PROFILES, MAX_SEGMENT_STEPS


def _project(point, origin, rotation, camera):
    local = (np.asarray(point)-origin) @ rotation
    depth = -float(local[2])
    if depth <= 1e-6:
        return None
    fx, fy = camera["focal_length_px"]
    cx, cy = camera["principal_point_px"]
    return np.asarray([cx+fx*local[0]/depth, cy-fy*local[1]/depth])


def _occluded(origin, point, boxes):
    """Conservative segment test against estimated non-free scene boxes."""
    for box in boxes:
        start = (origin-box["center"]) @ box["rotation"]
        delta = (point-origin) @ box["rotation"]
        half = box["half"]+box["uncertainty"]
        low, high = 0., 1.
        for axis in range(3):
            if abs(delta[axis]) < 1e-12:
                if abs(start[axis]) > half[axis]:
                    low, high = 1., 0.
                    break
            else:
                a, b = (-half[axis]-start[axis])/delta[axis], (half[axis]-start[axis])/delta[axis]
                low, high = max(low, min(a, b)), min(high, max(a, b))
        if low < high and high > 1e-6 and low < 1.-1e-4:
            return True
    return False


def propose_evidence_views(geometry, view, context, assessment, step, attempted=()):
    result = {"source": "calibrated_camera_and_model_geometry", "candidates": [],
              "reason": "no_spatial_coverage_witness", "visibility_is_measured": False}
    witnesses = assessment.get("coverage_witnesses") or []
    focus_id = None
    tracking_status = assessment.get("tracking_status")
    tracking_gap = (tracking_status in ("mismatch", "lost") or
                    "measured_tracking_mismatch" in (assessment.get("blocking_reasons") or []))
    if not witnesses and (assessment.get("blocking_reasons") or tracking_gap):
        # Lean geometry need not manufacture a free-space boundary. Focus a
        # real blocker using its existing public-image estimate instead.
        ids = [(assessment.get("limiting_pair") or {}).get("obstacle_id")]
        if tracking_gap:
            ids.extend((context.get("target_id"), context.get("held_object_id")))
        for reason in assessment.get("blocking_reasons") or []:
            if isinstance(reason, str):
                if reason.startswith(("obstacle_core:", "changed_geometry:")):
                    ids.append(reason.split(":", 2)[1])
                elif reason == "attachment_verification_required":
                    ids.extend((context.get("held_object_id"), context.get("target_id")))
        for ident in ids:
            entity = geometry.entities.get(ident)
            if entity is not None and entity.get("position_m") is not None:
                focus_id = ident
                witnesses = [{"world_point_m": list(entity["position_m"]),
                              "uncertainty_m": entity.get("uncertainty_m"), "entity_id": ident}]
                result["focus_source"] = "blocking_entity_estimate_not_a_visible_material_point"
                result["focus_entity_id"] = ident
                break
        if not witnesses and tracking_gap:
            # This is the robot's observed endpoint, not an inferred contact
            # point or a new free-space boundary. A useful, admitted view is
            # still required below; unavailable visibility remains a stop.
            witnesses = [{"world_point_m": np.asarray(view.position, float).tolist(),
                          "uncertainty_m": None, "body_part": "observed_grip_site"}]
            result["focus_source"] = "observed_robot_endpoint_after_tracking_anomaly"
    if not witnesses:
        return result
    metadata = getattr(getattr(view, "observation", None), "geometry", None) or {}
    camera = metadata.get("cameras", {}).get("wrist")
    if not camera:
        result["reason"] = "synchronized_wrist_camera_unavailable"
        return result
    try:
        origin = np.asarray(camera["position_world_m"], float)
        rotation = np.asarray(camera["rotation_world_from_camera"], float)
        width, height = metadata["image"]["width"], metadata["image"]["height"]
        point = np.asarray(witnesses[0]["world_point_m"], float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("invalid witness")
        start_uv = _project(point, origin, rotation, camera)
        if start_uv is None:
            result["reason"] = "witness_behind_wrist_camera"
            return result
    except (KeyError, TypeError, ValueError):
        result["reason"] = "incomplete_camera_or_witness_metadata"
        return result
    pose = measured_pose(view.state)
    bodies, payload_unknown = geometry._local_bodies(view, context, set())
    # The geometry layer owns admission with retained payload hypotheses. A
    # missing identity/coverage certificate alone is not a second blanket veto;
    # an actually unbounded payload still fails its preview_motion assessment.
    result["payload_geometry_unknown"] = bool(payload_unknown)
    radius = max((float(np.linalg.norm(b["half"])) for b in bodies), default=0.)
    # A no_free_region witness has no boundary estimate, hence null rather
    # than zero uncertainty. Use known estimates only for proposing a viewing
    # baseline; absent values never shrink collision or coverage margins.
    known_uncertainties = [float(w["uncertainty_m"]) for w in witnesses
                           if isinstance(w.get("uncertainty_m"), (int, float))
                           and math.isfinite(w["uncertainty_m"]) and w["uncertainty_m"] >= 0.]
    uncertainty = max(known_uncertainties, default=0.)
    result["boundary_uncertainty_available"] = bool(known_uncertainties)
    # Travel derives from the observed gap/body scale and camera depth; no task
    # name, waypoint or prescribed scene displacement participates.
    depth = float(np.linalg.norm(point-origin))
    scale = min(PROFILES["transit"]["target_translation_m"],
                max(2*uncertainty, radius, depth*.08))
    directions = [rotation[:, 0], -rotation[:, 0], rotation[:, 1], -rotation[:, 1]]
    boxes = [box for ident, entity in geometry.entities.items()
             if ident != focus_id
             if entity["role"] not in ("free_region", "destination_region") and entity["attachment"] != "held"
             for box in geometry._entity_boxes(entity)]
    observe_context = {**deepcopy(context), "intent": "observe", "allowed_contact_ids": [],
                       "commanded_gripper": context.get("commanded_gripper", "open"),
                       "controller_translation_limit_m": PROFILES["transit"]["action_translation_m"],
                       "controller_rotation_limit_rad": .1}
    # A failed prior target supplies the reason to acquire a view, not evidence
    # that every new direction has already failed. The new primitive must build
    # its own tracking history, while exact-command geometry checks still run.
    observe_context.pop("tracking", None)
    # Choose an admitted useful baseline for each lateral direction. Admission
    # is not evidence that the predicted view or physical clearance is certain.
    candidates = []
    rejected = {"already_attempted": 0, "visibility": 0, "motion": 0}
    for direction in directions:
        for fraction in (1., .5):
            offset = direction*scale*fraction
            target = np.asarray(view.position)+offset
            if any(np.linalg.norm(target-np.asarray(p)) < PROFILES["transit"]["position_tolerance_m"] for p in attempted):
                rejected["already_attempted"] += 1
                continue
            new_origin = origin+offset
            uv = _project(point, new_origin, rotation, camera)
            if (uv is None or not 0 <= uv[0] < width or not 0 <= uv[1] < height
                    or np.linalg.norm(uv-start_uv) < 8. or _occluded(new_origin, point, boxes)):
                rejected["visibility"] += 1
                continue
            ray0, ray1 = point-origin, point-new_origin
            parallax = math.acos(float(np.clip(np.dot(ray0, ray1)/max(np.linalg.norm(ray0)*np.linalg.norm(ray1), 1e-12), -1., 1.)))
            checked = geometry.preview_motion(view, target, pose[3:7], observe_context, step)
            admitted = (checked["execution_allowed"] is True if "execution_allowed" in checked else
                        checked.get("status") == "supported" and checked.get("allowed_fraction", 0.) >= 1.-1e-9)
            if not admitted:
                rejected["motion"] += 1
                continue
            candidates.append({"xyz_m": target.tolist(), "quaternion_wxyz": list(pose[3:7]),
                "witness_world_point_m": point.tolist(), "body_part": witnesses[0].get("body_part"),
                "expected_pixel_shift": float(np.linalg.norm(uv-start_uv)),
                "predicted_parallax_rad": parallax, "travel_m": float(np.linalg.norm(offset)),
                "score": parallax/max(float(np.linalg.norm(offset)), 1e-9),
                "path_status": checked.get("status", "unknown"), "execution_allowed": True,
                "execution_status": checked.get("execution_status", checked.get("status", "unknown")),
                "blocking_reasons": deepcopy(checked.get("blocking_reasons", [])),
                "advisories": deepcopy(checked.get("advisories", [])),
                "physical_clearance_verified": False, "requires_fresh_visual_interpretation": True,
                "execution": {"segments": [{"skill": "move_to_pose", "xyz_m": target.tolist(),
                    "quaternion_wxyz": list(pose[3:7]), "gripper": "keep", "profile": "transit",
                    "max_steps": MAX_SEGMENT_STEPS, "observation_after": True}]}})
            break
    candidates.sort(key=lambda item: (-item["score"], item["travel_m"]))
    reason = ("supported_view_candidates" if all(c["path_status"] == "supported" for c in candidates)
              else "admissible_view_candidates") if candidates else "no_admissible_informative_view"
    result.update(candidates=candidates[:3], reason=reason,
                  rejected=rejected, witness_world_point_m=point.tolist(),
                  original_intent_is_not_resumed=True)
    return result
