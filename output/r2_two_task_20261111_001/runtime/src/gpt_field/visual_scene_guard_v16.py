"""Pure checks on materialized scene coverage and observable contact handoff.

Inventory geometry and semantic image boxes remain model hypotheses. This
module neither creates occupancy nor executes, clips, repairs or acknowledges
commands. It does not modify the existing travel budgets. Continuous hard
collision checks and per-part contact permissions remain executor-owned.
"""
from __future__ import annotations

import numpy as np

from .robot_tool_geometry import world_aabbs


def _vector(value, size):
    try:
        result = np.asarray(value, float)
    except (ValueError, TypeError):
        return None
    return result if result.shape == (size,) and np.isfinite(result).all() else None


def _roi(value):
    box = _vector(value, 4)
    return box if box is not None and np.all((box >= 0) & (box <= 1)) and np.all(box[:2] < box[2:]) else None


def _point_distance(point, boxes):
    return min(float(np.linalg.norm(np.maximum(np.maximum(lo-point, point-hi), 0.))) for lo, hi in boxes)


def _overlap(first, second, margin=0.):
    return bool(np.all(first[2:]+margin >= second[:2]) and np.all(second[2:]+margin >= first[:2]))


class SceneCoverageGuardV16:
    version = "persistent-scene-coverage-and-contact-handoff-v16"
    image_margin_uv = .02
    minimum_contact_band_m = .03

    @staticmethod
    def _cube(state, field, is_field):
        if is_field and field is not None and getattr(field, "origin", None) is not None:
            center = np.asarray(field.origin, float)
            half = float(getattr(getattr(field, "grid", None), "half_width_m", .08))
        else:
            # Direct cannot borrow a previous Field origin or clearance claim.
            center, half = state[:3], .08
        return center-half, center+half

    @staticmethod
    def _configuration_boxes(record, state, field, prediction, *, full):
        geometry = record.get("world_geometry")
        if not isinstance(geometry, dict):
            return None
        center = _vector(geometry.get("center_xyz_m"), 3)
        half = _vector(geometry.get("half_extents_m"), 3)
        try:
            sigma = float(geometry.get("uncertainty_m"))
        except (ValueError, TypeError):
            return None
        if center is None or half is None or np.any(half <= 0) or not np.isfinite(sigma) or sigma < 0:
            return None
        intent = prediction.get("field_intent") or {}
        # The Direct case intentionally uses current known robot geometry and
        # this prediction; it never treats a cached Field scene as clearance.
        sigma += max(0., float(intent.get("uncertainty_m", 0.)))
        margin = sigma if full else 0.
        low, high = center-half-margin, center+half+margin
        parts = world_aabbs(state, orientation_margin_rad=.04, full_finger_range=True)
        result = [(low-np.asarray(part["offset_max_xyz"]), high-np.asarray(part["offset_min_xyz"])) for part in parts]
        radius = max(0., float(intent.get("tool_radius_m", .045)))
        result.append((low-radius, high+radius))
        return result

    @staticmethod
    def _alignment_current(contact_alignment, target_id):
        summary = contact_alignment.summary() if hasattr(contact_alignment, "summary") else contact_alignment
        if not isinstance(summary, dict) or not summary.get("active") or summary.get("target_id") != target_id:
            return False
        points = summary.get("current_points_px") or {}
        return all(_vector(points.get(key), 2) is not None
                   for key in ("tool", "external_target", "wrist_target", "wrist_pinch"))

    def assess(self, state8, delta_xyz, *, materialization, field=None, monitor=None,
               prediction, contact_alignment=None, requires_scene_coverage=False):
        state, delta = _vector(state8, 8), _vector(delta_xyz, 3)
        if state is None or delta is None:
            raise ValueError("scene guard requires finite state8 and command delta_xyz")
        details = {"version": self.version, "coverage_issues": [], "unresolved_far_image_ids": [],
                   "metric_only_disjoint_ids": [], "near_contact": None,
                   "clearance_certified": False, "task_success_certified": False,
                   "budgets_modified": False, "command_modified": False,
                   "environment_truth_inputs": False}

        def result(reason=None):
            return {"accepted": reason is None, "reason": reason or "scene_coverage_and_contact_handoff_passed",
                    "need_refresh": reason is not None, "diagnostics": details}

        if np.linalg.norm(delta) < 1e-12 and not requires_scene_coverage:
            value = result();value["reason"] = "pose_hold_no_scene_clearance_claim"
            return value
        if not isinstance(materialization, dict) or not isinstance(materialization.get("objects"), list):
            return result("scene_inventory_materialization_unavailable")
        is_field = prediction.get("controller") == "visual_field"
        target_id = prediction.get("target_id")
        excluded = materialization.get("controlled_payload_excluded_id")
        metric_ids = set(materialization.get("applied_metric_ids", []))
        image_ids = set(materialization.get("applied_image_ids", []))
        disjoint_claims = set(materialization.get("geometrically_disjoint_ids", []))
        cube_low, cube_high = self._cube(state, field, is_field)
        details["local_cube"] = {"lower_xyz": cube_low.tolist(), "upper_xyz": cube_high.tolist()}
        regions = getattr(field, "region_geometry", []) if is_field else []
        actual_metric_ids = {r.get("id") for r in regions}
        guard = getattr(field, "translation_guard", None)
        actual_image_ids = {n.get("id") for n in getattr(guard, "neighbors", [])}
        # A currently tracked box is preferred over a raw declaration.
        tool = None
        scale = getattr(guard, "scale", None)
        if getattr(guard, "tool_box", None) is not None and scale is not None:
            tool = _roi(np.asarray(guard.tool_box)/np.tile(scale, 2))
        if tool is None:
            tool = _roi((prediction.get("motion_context") or {}).get("tool_external_roi_uv"))
        target_rois = []
        tracks = getattr(monitor, "tracks", {}) or {}
        target_track = tracks.get("target")
        if target_track is not None and getattr(target_track, "reliable", False):
            target_rois.append(_roi(target_track.summary().get("current_roi_uv")))
        for record in materialization["objects"]:
            if record.get("id") == target_id:
                target_rois.append(_roi((record.get("views", {}).get("external") or {}).get("current_roi_uv")))
        target_rois = [r for r in target_rois if r is not None]
        phase = prediction.get("phase")
        approaching = phase in ("approach", "align", "recover")
        for record in materialization["objects"]:
            identity = record.get("id")
            if identity == excluded:
                continue  # Confirmed controlled payload remains separately constrained.
            view = (record.get("views") or {}).get("external") or {}
            box = _roi(view.get("current_roi_uv"))
            # A fresh model rectangle is current evidence even if its texture
            # cannot support tracking. It is not a clearance certificate;
            # downstream near-risk tracking checks remain in force. A stale
            # rectangle has current_roi_uv=None and cannot use this branch.
            image_current = box is not None
            if box is None:
                box = _roi(view.get("last_known_roi_uv"))
            if image_current and (identity not in image_ids or (guard is not None and identity not in actual_image_ids)):
                # Being guessed metric-remote cannot remove an available
                # image region from the risk observer's inventory.
                details["coverage_issues"].append({"id": identity, "reason": "scene_image_risk_coverage_missing_after_materialization"})
            bounds = self._configuration_boxes(record, state, field, prediction, full=True)
            local_geometry = bounds is not None and any(np.all(hi >= cube_low) and np.all(lo <= cube_high) for lo, hi in bounds)
            disjoint = bounds is not None and not local_geometry
            if identity in disjoint_claims and not disjoint:
                details["coverage_issues"].append({"id": identity, "reason": "scene_disjoint_claim_not_supported_by_full_bounds"})
            if bounds is None:
                near_tool = box is not None and tool is not None and _overlap(box, tool, self.image_margin_uv)
                near_target = bool(approaching and box is not None and any(_overlap(box, r, self.image_margin_uv) for r in target_rois))
                if box is None or near_tool or near_target or identity == target_id:
                    details["coverage_issues"].append({"id": identity, "reason": "scene_visible_region_needs_metric_localization",
                        "near_tool_image": near_tool, "near_approach_target_image": near_target})
                else:
                    details["unresolved_far_image_ids"].append(identity)
                continue
            if is_field and not disjoint and (identity not in metric_ids or identity not in actual_metric_ids):
                details["coverage_issues"].append({"id": identity, "reason": "scene_metric_coverage_missing_after_materialization"})
            if disjoint:
                details["metric_only_disjoint_ids"].append(identity)
                continue
            if not image_current:
                details["coverage_issues"].append({"id": identity, "reason": "scene_local_metric_region_lacks_current_image_support"})
        if details["coverage_issues"]:
            return result(details["coverage_issues"][0]["reason"])
        if is_field and approaching and field is not None:
            target = getattr(field, "target_configuration", None) or {}
            parts = target.get("core_parts") or target.get("tool_core_parts") or []
            boxes = [(np.asarray(p["lower_xyz"], float), np.asarray(p["upper_xyz"], float)) for p in parts]
            if target.get("target_id") == target_id and boxes:
                start_distance = _point_distance(state[:3], boxes)
                end_distance = _point_distance(state[:3]+delta, boxes)
                band = max(self.minimum_contact_band_m, float(target.get("sigma_m", 0.)))
                toward = end_distance < start_distance-1e-7
                current = self._alignment_current(contact_alignment, target_id)
                details["near_contact"] = {"target_id": target_id, "start_core_distance_m": start_distance,
                    "end_core_distance_m": end_distance, "handoff_band_m": band, "moving_toward_core": toward,
                    "current_two_view_descriptor": current, "learned_response_required": False,
                    "permission_is_grasp_or_clearance": False}
                if toward and end_distance <= band+1e-9 and not current:
                    details["recommended_action"] = "Supply current visible two-view contact/pinch descriptors, or select an explicitly justified lateral/away recovery; never infer contact readiness from a metric waypoint."
                    return result("near_target_approach_requires_current_two_view_contact")
        return result()


VisualSceneGuardV16 = SceneCoverageGuardV16
