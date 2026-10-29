"""Robot-tool configuration obstacles and observed translation admission.

The robot envelope comes from fixed robot assets, while scene geometry remains
an uncertain model hypothesis. Image admission is an additional observation
boundary, never a calibrated scene or a collision-clearance certificate.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import numpy as np

from .field import GridField
from .robot_tool_geometry import world_aabbs
from .visual_field_v6 import _AxisConstrainedField
from .visual_field_v9 import RiskRegionDistancePlanner, _distance_to_box
from .visual_field_v10 import MetricFieldExecutorV10


class ToolEnvelopeRiskPlanner(RiskRegionDistancePlanner):
    def _density(self, points, field, compact):
        points = np.asarray(points, float)
        density = np.ones(points.shape[:-1], float)
        for region in self.risk_regions:
            distances = [_distance_to_box(points, part["lower_xyz"], part["upper_xyz"])
                         for part in region["parts"]]
            distance = np.minimum.reduce(distances)
            # One risk contribution per scene object, not one per robot part.
            density += self.risk_weight*np.exp(-.5*(distance/region["sigma_m"])**2)
        return density

    def plan(self, *args, **kwargs):
        result = super().plan(*args, **kwargs)
        return replace(result, search_metadata={**result.search_metadata,
            "risk_density": "1 + weight * sum_per_object(exp(-0.5*(distance_to_union_of_tool_configuration_boxes/sigma)^2))",
            "robot_envelope": "known_robot_collision_parts_union_model_declared_tool_extent",
            "scene_geometry_is_still_model_hypothesis": True})


def _bounds(parts):
    return {"lower_xyz": np.min([p["lower_xyz"] for p in parts], axis=0).tolist(),
            "upper_xyz": np.max([p["upper_xyz"] for p in parts], axis=0).tolist()}


def _inside(point, parts):
    return any(np.all(point >= np.asarray(p["lower_xyz"])-1e-12)
               and np.all(point <= np.asarray(p["upper_xyz"])+1e-12) for p in parts)


class MetricFieldExecutorV11(MetricFieldExecutorV10):
    def __init__(self):
        super().__init__()
        self.tool_parts = []
        self.translation_guard = self.visual_monitor = None
        self.last_translation_admission = None

    def bind_translation_guard(self, guard, monitor):
        self.translation_guard, self.visual_monitor = guard, monitor

    def configure(self, state8, intent, *, scene_assessment=None, subgoal_origin_xyz=None):
        super().configure(state8, intent, scene_assessment=scene_assessment,
                          subgoal_origin_xyz=subgoal_origin_xyz)
        self.last_translation_admission = None
        self.tool_parts = world_aabbs(state8, orientation_margin_rad=self.MAX_DRIFT_RAD,
                                      full_finger_range=True)
        offsets = [{"name": p["name"], "lower": np.asarray(p["offset_min_xyz"]),
                    "upper": np.asarray(p["offset_max_xyz"])} for p in self.tool_parts]
        radius = self.intent["tool_radius_m"]
        offsets.append({"name": "model_declared_tool_or_payload_extent",
                        "lower": np.full(3, -radius), "upper": np.full(3, radius)})
        self.boxes, self.regions, self.region_geometry, self.risk_regions = [], [], [], []
        points = self.grid.coordinates(self.origin)
        masks = {key: np.zeros(self.grid.count, bool) for key in ("obstacles", "unknown_regions")}
        for group in masks:
            for region in self.intent[group]:
                low = self.origin+np.asarray(region["lower_offset_m"])
                high = self.origin+np.asarray(region["upper_offset_m"])
                sigma = self.intent["uncertainty_m"]+region["uncertainty_m"]
                core_parts, halo_parts = [], []
                soft = self.soft_uncertainty_enabled and group == "obstacles"
                mode = "hard_robot_parts_soft_uncertainty" if soft else "hard_robot_parts_full_uncertainty"
                for part in offsets:
                    # Configuration-space Minkowski difference: EEF must not
                    # place ANY known tool part inside a scene obstacle box.
                    core_low, core_high = low-part["upper"], high-part["lower"]
                    core = {"tool_part": part["name"], "lower_xyz": core_low.tolist(),
                            "upper_xyz": core_high.tolist()}
                    halo = {"tool_part": part["name"], "lower_xyz": (core_low-sigma).tolist(),
                            "upper_xyz": (core_high+sigma).tolist()}
                    core_parts.append(core); halo_parts.append(halo)
                    hard = core if soft else halo
                    hard_low, hard_high = np.asarray(hard["lower_xyz"]), np.asarray(hard["upper_xyz"])
                    self.boxes.append((hard_low, hard_high))
                    masks[group] |= np.all((points >= hard_low) & (points <= hard_high), axis=1)
                geometry = {"id": region["id"], "kind": group, "geometry_mode": mode,
                    "nominal": {"lower_xyz": low.tolist(), "upper_xyz": high.tolist()},
                    "tool_core": _bounds(core_parts), "full_halo": _bounds(halo_parts),
                    "tool_core_parts": core_parts, "full_halo_parts": halo_parts,
                    "union_bounds_are_display_only": True,
                    "margins_m": {"tool_radius": radius, "intent_uncertainty": self.intent["uncertainty_m"],
                        "region_uncertainty": region["uncertainty_m"], "combined_uncertainty": sigma},
                    "hard_collision_hypothesis": True, "certified_occupied": False}
                self.region_geometry.append(geometry)
                self.regions.append({"id": region["id"], "kind": group, "geometry_mode": mode,
                    "hard_part_count": len(core_parts), "bounds_semantics": "union_of_configuration_boxes"})
                if soft:
                    self.risk_regions.append({"id": region["id"], "parts": core_parts, "sigma_m": sigma})
        prediction = replace(self.field.prediction,
            blocked_indices=tuple(np.flatnonzero(masks["obstacles"]).tolist()),
            unknown_indices=tuple(np.flatnonzero(masks["unknown_regions"] & ~masks["obstacles"]).tolist()))
        self.field = GridField(self.grid, tuple(self.origin), prediction)
        if self.locked_axes:
            self.field = _AxisConstrainedField(self.field, self.locked_axes, self.origin)
            self.field = _AxisConstrainedField(self.field, self.locked_axes, self.subgoal_origin)
        self.planner = ToolEnvelopeRiskPlanner(self.boxes, self.risk_regions)
        return self.summary()

    def _nominal_away(self, point, delta):
        if self.translation_guard is None:
            return False
        identities = set(self.translation_guard.summary().get("conflicting_obstacle_ids", []))
        if not identities:
            return False
        changes = []
        for identity in identities:
            region = next((r for r in self.region_geometry if r["id"] == identity), None)
            if region is None:
                return False
            parts = region["tool_core_parts"]
            distance = lambda p: min(float(_distance_to_box(p, b["lower_xyz"], b["upper_xyz"])) for b in parts)
            changes.append(distance(point+delta)-distance(point))
        return bool(changes and min(changes) >= -1e-7 and max(changes) > 1e-6)

    def select(self, state8, joint_position, *, command_cap_m=None):
        count = self.orientation_goal_refresh_commands
        old_rotation = self.last_orientation_command_rad.copy()
        result = super().select(state8, joint_position, command_cap_m=command_cap_m)
        if not result["accepted"] or self.translation_guard is None:
            return result
        point = np.asarray(state8[:3], float)
        def assess(value):
            delta = np.asarray(value["action"][:3])*.05
            return self.translation_guard.assess(delta, monitor=self.visual_monitor,
                nominal_away=self._nominal_away(point, delta),
                rotation_rad=float(np.linalg.norm(value["action"][3:6])*.5))
        admission = assess(result)
        cap = admission.get("cap_m")
        norm = float(np.linalg.norm(result["action"][:3])*.05)
        if admission["accepted"] and cap is not None and cap < norm-1e-12:
            self.pending = None
            self.orientation_goal_refresh_commands = count
            self.last_orientation_command_rad = old_rotation.copy()
            if cap < self.MIN_USEFUL_COMMAND_M:
                admission = {**admission, "accepted": False, "reason": "translation_allowance_below_useful_motion_floor"}
            else:
                effective_cap = cap if command_cap_m is None else min(cap, command_cap_m)
                result = super().select(state8, joint_position, command_cap_m=effective_cap)
                if result["accepted"]:
                    admission = assess(result)
                    if (admission["accepted"] and admission.get("cap_m") is not None
                            and np.linalg.norm(result["action"][:3])*.05 > admission["cap_m"]+1e-12):
                        admission = {**admission, "accepted": False,
                                     "reason": "translation_admission_requires_shorter_replan"}
                else:
                    self.last_translation_admission = copy.deepcopy(admission)
                    return result
        self.last_translation_admission = copy.deepcopy(admission)
        if not admission["accepted"]:
            self.pending = None
            self.orientation_goal_refresh_commands = count
            self.last_orientation_command_rad = old_rotation
            return self._stop(admission["reason"], {"translation_admission": copy.deepcopy(admission)})
        result["diagnostics"]["translation_admission"] = copy.deepcopy(admission)
        result["diagnostics"]["robot_tool_envelope"] = self._tool_summary()
        return result

    def _tool_summary(self):
        return {"source": "fixed_robot_collision_assets_and_measured_robot_state_only",
            "parts": copy.deepcopy(getattr(self, "tool_parts", [])),
            "orientation_margin_rad": self.MAX_DRIFT_RAD, "full_finger_range": True,
            "model_tool_payload_extent_preserved": True, "scene_geometry_certified": False,
            "full_arm_included": False}

    def summary(self):
        result = super().summary()
        if self.intent is None:
            return result
        for item in result.get("region_feasibility", []):
            if "tool_core_parts" not in item:
                continue
            for label, parts_key in (("tool_core", "tool_core_parts"), ("full_halo", "full_halo_parts")):
                parts = item[parts_key]
                item[label]["contains_current"] = _inside(self.expected_state[:3], parts)
                item[label]["contains_goal"] = _inside(self.goal, parts)
                item[label]["current_distance_to_union_m"] = min(
                    float(_distance_to_box(self.expected_state[:3], b["lower_xyz"], b["upper_xyz"])) for b in parts)
            item["current_only_in_uncertainty_halo"] = item["full_halo"]["contains_current"] and not item["tool_core"]["contains_current"]
        result.update(version="robot-envelope-observed-translation-field-v11",
            robot_tool_envelope=self._tool_summary(),
            translation_admission=copy.deepcopy(getattr(self, "last_translation_admission", None)))
        result["risk_density"]["formula"] = "1 + 8 * sum_per_object(exp(-0.5*(distance_to_union_of_tool_configuration_boxes/sigma)^2))"
        return result
