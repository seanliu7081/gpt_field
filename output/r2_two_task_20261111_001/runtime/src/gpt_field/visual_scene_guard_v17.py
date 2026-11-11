"""Current-command scene coverage using conservative known-tool sweeps.

The scene is a remembered model hypothesis, never certified occupancy or
clearance. Partial image support does not reduce its uncertainty. This pure
guard checks one proposed command without changing tracking or travel budgets.
"""
from __future__ import annotations

import numpy as np

from .robot_tool_geometry import world_aabbs
from .visual_scene_guard_v16 import SceneCoverageGuardV16, _vector, _roi, _overlap, _point_distance


def segment_box_interval(start, end, low, high):
    """Exact closed-segment/slab intersection, including grazing and a hold."""
    start, end, low, high = (np.asarray(value, float) for value in (start, end, low, high))
    lower, upper = 0., 1.
    for axis, movement in enumerate(end-start):
        if abs(movement) < 1e-15:
            if start[axis] < low[axis] or start[axis] > high[axis]:
                return None
            continue
        times = sorted(((low[axis]-start[axis])/movement, (high[axis]-start[axis])/movement))
        lower, upper = max(lower, times[0]), min(upper, times[1])
        if lower > upper:
            return None
    return [float(lower), float(upper)]


class SceneCoverageGuardV17(SceneCoverageGuardV16):
    version = "persistent-scene-command-sweep-and-contact-handoff-v17"

    @staticmethod
    def _rotation_margin(prediction, action7, rotation_margin_rad):
        margin = .04
        if action7 is not None:
            action = _vector(action7, 7)
            if action is None:
                raise ValueError("action7 must be finite")
            margin = max(margin, float(np.linalg.norm(action[3:6]))*.5)
        elif prediction.get("controller") == "direct":
            for action in prediction.get("direct_actions") or []:
                value = _vector(action, 7)
                if value is None:
                    raise ValueError("predicted Direct actions must be finite action7")
                margin = max(margin, float(np.linalg.norm(value[3:6]))*.5)
        if rotation_margin_rad is not None:
            value = float(rotation_margin_rad)
            if not np.isfinite(value) or value < 0:
                raise ValueError("rotation_margin_rad must be finite and nonnegative")
            margin = max(margin, value)
        # Any rotation in SO(3) lies within pi of the measured orientation.
        return min(margin, float(np.pi))

    @staticmethod
    def _sweep_boxes(record, state, prediction, orientation_margin, payload=None):
        geometry = record.get("world_geometry")
        if not isinstance(geometry, dict):
            return None
        center, half = _vector(geometry.get("center_xyz_m"), 3), _vector(geometry.get("half_extents_m"), 3)
        try:
            region_sigma = float(geometry.get("uncertainty_m"))
            intent = prediction.get("field_intent") or {}
            intent_sigma = float(intent.get("uncertainty_m", 0.))
            radius = float(intent.get("tool_radius_m", .045))
        except (TypeError, ValueError):
            return None
        if (center is None or half is None or np.any(half <= 0)
                or not np.isfinite([region_sigma, intent_sigma, radius]).all()
                or min(region_sigma, intent_sigma, radius) < 0):
            return None
        sigma = region_sigma+intent_sigma
        low, high = center-half-sigma, center+half+sigma
        parts = world_aabbs(state, orientation_margin_rad=orientation_margin, full_finger_range=True)
        boxes = [{"tool_part": part["name"],
                  "lower_xyz": low-np.asarray(part["offset_max_xyz"]),
                  "upper_xyz": high-np.asarray(part["offset_min_xyz"])} for part in parts]
        boxes.append({"tool_part": "declared_tool_or_payload_extent", "lower_xyz": low-radius, "upper_xyz": high+radius})
        if payload is not None:
            boxes.append({"tool_part": "observed_carried_payload_hypothesis",
                "lower_xyz": low-payload[1], "upper_xyz": high-payload[0]})
        return boxes

    @staticmethod
    def _carried_bounds(state, field, margin):
        geometry = getattr(field, "contact_geometry", None)
        if not hasattr(geometry, "carried_summary"):
            return None
        carried = geometry.carried_summary(state)
        if not carried.get("active"):
            return None
        world, local = carried.get("nominal_world") or {}, carried.get("nominal_hand_frame") or {}
        lo, hi = _vector(world.get("lower_xyz"), 3), _vector(world.get("upper_xyz"), 3)
        local_lo, local_hi = _vector(local.get("lower_xyz"), 3), _vector(local.get("upper_xyz"), 3)
        sigma = float(carried.get("uncertainty_m", 0.))
        if any(value is None for value in (lo, hi, local_lo, local_hi)) or not np.isfinite(sigma) or sigma < 0:
            raise ValueError("Active observed payload lacks valid retained bounds")
        radius = float(np.linalg.norm(np.maximum(abs(local_lo), abs(local_hi))))
        expansion = sigma+2*radius*np.sin(margin/2)
        return lo-state[:3]-expansion, hi-state[:3]+expansion

    def assess(self, state8, delta_xyz, *, materialization, field=None, monitor=None,
               prediction, contact_alignment=None, requires_scene_coverage=False,
               action7=None, rotation_margin_rad=None):
        state, delta = _vector(state8, 8), _vector(delta_xyz, 3)
        if state is None or delta is None:
            raise ValueError("scene guard requires finite state8 and command delta_xyz")
        margin = self._rotation_margin(prediction, action7, rotation_margin_rad)
        details = {"version": self.version, "coverage_issues": [], "unresolved_far_image_ids": [],
                   "metric_only_disjoint_ids": [], "sweep_disjoint_ids": [], "sweep_checks": [],
                   "checked_metric_ids": [], "checked_image_ids": [], "near_contact": None,
                   "proposed_delta_xyz_m": delta.tolist(), "start_xyz_m": state[:3].tolist(),
                   "end_xyz_m": (state[:3]+delta).tolist(), "orientation_margin_rad": margin,
                   "full_finger_range": True, "coverage_scope": "actual_commanded_whole_tool_sweep",
                   "clearance_certified": False, "task_success_certified": False,
                   "budgets_modified": False, "command_modified": False, "environment_truth_inputs": False}

        def result(reason=None):
            details["checked_metric_count"] = len(details["checked_metric_ids"])
            details["checked_image_count"] = len(details["checked_image_ids"])
            details["sweep_disjoint_count"] = len(details["sweep_disjoint_ids"])
            return {"accepted": reason is None, "reason": reason or "scene_command_sweep_and_contact_handoff_passed",
                    "need_refresh": reason is not None, "diagnostics": details}

        pose_changes = action7 is not None and (np.linalg.norm(np.asarray(action7)[3:6]) > 1e-8 or abs(action7[6]) > 1e-12)
        if np.linalg.norm(delta) < 1e-12 and not requires_scene_coverage and not pose_changes:
            value = result(); value["reason"] = "pose_hold_no_scene_clearance_claim"
            return value
        if not isinstance(materialization, dict) or not isinstance(materialization.get("objects"), list):
            return result("scene_inventory_materialization_unavailable")
        payload = self._carried_bounds(state, field, margin)
        details["carried_payload_hypothesis_in_sweep"] = payload is not None
        details["payload_extent_source"] = ("retained_preclosure_geometry_with_locally_observed_attachment"
            if payload is not None else "declared_tool_or_payload_extent_only_no_observed_payload_bounds")
        is_field = prediction.get("controller") == "visual_field"
        target_id = prediction.get("target_id")
        excluded = materialization.get("controlled_payload_excluded_id")
        metric_ids, image_ids = set(materialization.get("applied_metric_ids", [])), set(materialization.get("applied_image_ids", []))
        actual_metric_ids = {r.get("id") for r in getattr(field, "region_geometry", [])} if is_field else set()
        guard = getattr(field, "translation_guard", None)
        actual_image_ids = {n.get("id") for n in getattr(guard, "neighbors", [])}
        tool, scale = None, getattr(guard, "scale", None)
        if getattr(guard, "tool_box", None) is not None and scale is not None:
            tool = _roi(np.asarray(guard.tool_box)/np.tile(scale, 2))
        if tool is None:
            tool = _roi((prediction.get("motion_context") or {}).get("tool_external_roi_uv"))
        target_rois = []
        target_track = (getattr(monitor, "tracks", {}) or {}).get("target")
        if target_track is not None and getattr(target_track, "reliable", False):
            target_rois.append(_roi(target_track.summary().get("current_roi_uv")))
        for record in materialization["objects"]:
            if record.get("id") == target_id:
                view = (record.get("views") or {}).get("external") or {}
                target_rois.extend([_roi(view.get("current_roi_uv")), _roi(view.get("current_visible_roi_uv"))])
        target_rois = [roi for roi in target_rois if roi is not None]
        approaching = prediction.get("phase") in ("approach", "align", "recover")
        for record in materialization["objects"]:
            identity = record.get("id")
            if identity == excluded:
                continue  # Confirmed payload has its independent carried constraint.
            view = (record.get("views") or {}).get("external") or {}
            full_box = _roi(view.get("current_roi_uv"))
            partial_box = _roi(view.get("current_visible_roi_uv")) if view.get("partial_current_support") else None
            boundary_limited = (view.get("extent_visibility") == "boundary_limited"
                or bool(view.get("extent_touches_image_boundary"))
                or full_box is not None and (np.any(full_box[:2] <= 0.) or np.any(full_box[2:] >= 1.)))
            full_support = full_box is not None and not boundary_limited and not view.get("partial_current_support")
            current_box = full_box if full_box is not None else partial_box
            box = current_box if current_box is not None else _roi(view.get("last_known_roi_uv"))
            if current_box is not None:
                details["checked_image_ids"].append(identity)
                if identity not in image_ids or guard is not None and identity not in actual_image_ids:
                    details["coverage_issues"].append({"id": identity, "reason": "scene_image_risk_coverage_missing_after_materialization"})
            boxes = self._sweep_boxes(record, state, prediction, margin, payload)
            if boxes is None:
                near_tool = box is not None and tool is not None and _overlap(box, tool, self.image_margin_uv)
                near_target = bool(approaching and box is not None and any(_overlap(box, r, self.image_margin_uv) for r in target_rois))
                if box is None or near_tool or near_target or identity == target_id:
                    details["coverage_issues"].append({"id": identity, "reason": "scene_visible_region_needs_metric_localization",
                        "near_tool_image": near_tool, "near_approach_target_image": near_target})
                else:
                    details["unresolved_far_image_ids"].append(identity)
                continue
            details["checked_metric_ids"].append(identity)
            if is_field and (identity not in metric_ids or identity not in actual_metric_ids):
                details["coverage_issues"].append({"id": identity, "reason": "scene_metric_coverage_missing_after_materialization"})
            intersections = []
            for bounds in boxes:
                interval = segment_box_interval(state[:3], state[:3]+delta, bounds["lower_xyz"], bounds["upper_xyz"])
                if interval is not None:
                    intersections.append({"tool_part": bounds["tool_part"], "segment_parameter_interval": interval,
                        "lower_xyz": bounds["lower_xyz"].tolist(), "upper_xyz": bounds["upper_xyz"].tolist()})
            disjoint = not intersections
            support = ("full_current" if full_support else "partial_current" if partial_box is not None
                       else "boundary_limited" if boundary_limited and full_box is not None else "missing_or_stale")
            details["sweep_checks"].append({"id": identity, "image_support": support,
                "full_uncertainty_sweep_disjoint": disjoint, "configuration_part_count": len(boxes),
                "intersected_parts": intersections, "model_hypothesis_not_clearance_certificate": True})
            if disjoint:
                details["sweep_disjoint_ids"].append(identity)
                if not full_support:
                    details["metric_only_disjoint_ids"].append(identity)
            elif not full_support:
                details["coverage_issues"].append({"id": identity,
                    "reason": "scene_incomplete_image_support_intersects_candidate_uncertainty_sweep",
                    "image_support": support, "intersected_tool_parts": [item["tool_part"] for item in intersections]})
        if details["coverage_issues"]:
            return result(details["coverage_issues"][0]["reason"])
        # Keep contact handoff separate from obstacle coverage and retain the
        # target's per-part permissions supplied by the actual executor.
        if is_field and approaching and field is not None:
            target = getattr(field, "target_configuration", None) or {}
            parts = target.get("core_parts") or target.get("tool_core_parts") or []
            boxes = [(np.asarray(p["lower_xyz"], float), np.asarray(p["upper_xyz"], float)) for p in parts]
            if target.get("target_id") == target_id and boxes:
                start_distance, end_distance = _point_distance(state[:3], boxes), _point_distance(state[:3]+delta, boxes)
                band = max(self.minimum_contact_band_m, float(target.get("sigma_m", 0.)))
                toward = end_distance < start_distance-1e-7
                current = self._alignment_current(contact_alignment, target_id)
                details["near_contact"] = {"target_id": target_id, "start_core_distance_m": start_distance,
                    "end_core_distance_m": end_distance, "handoff_band_m": band, "moving_toward_core": toward,
                    "current_two_view_descriptor": current, "learned_response_required": False,
                    "permission_is_grasp_or_clearance": False}
                if toward and end_distance <= band+1e-9 and not current:
                    details["recommended_action"] = "Supply current visible two-view contact/pinch descriptors or an explicitly justified lateral/away recovery."
                    return result("near_target_approach_requires_current_two_view_contact")
        return result()


VisualSceneGuardV17 = SceneCoverageGuardV17
