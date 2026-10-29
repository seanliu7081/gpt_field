"""Opt-in hard object cores and soft geometric-uncertainty traversal costs.

All geometry remains an RGB-derived model hypothesis. A supported visible
object's nominal box plus tool radius is excluded; its uncertainty is a soft
cost outside that core. Genuinely unknown regions retain full hard inflation.
Neither the core nor the risk density certifies physical occupancy or safety.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import math

import numpy as np

from .distance_planner import DistanceFieldPlanner
from .field import GridField
from .planner import PlannerConfig
from .visual_field_v4 import _vector
from .visual_field_v6 import _AxisConstrainedField, validate_metric_intent_v6
from .visual_field_v8 import MetricFieldExecutorV8


RISK_WEIGHT = 8.


def _valid_roi(value):
    try:
        roi = _vector(value, 4, "image ROI")
    except (ValueError, TypeError):
        return False
    return bool(np.all((roi >= 0.) & (roi <= 1.))
                and roi[0] < roi[2] and roi[1] < roi[3])


def _assessed_visible_geometry(context, intent):
    """Missing or inconsistent support always falls back to strict geometry."""
    if not isinstance(context, dict):
        return False, "missing_scene_assessment"
    assessment = context.get("obstacle_assessment")
    if assessment not in ("present", "none_relevant"):
        return False, "unobservable_or_invalid_scene_assessment"
    if not _valid_roi(context.get("tool_external_roi_uv")):
        return False, "missing_or_invalid_tool_image_support"
    neighbors = context.get("nearby_obstacles")
    if not isinstance(neighbors, list):
        return False, "invalid_neighbor_list"
    identities, localized = [], set()
    for neighbor in neighbors:
        if not isinstance(neighbor, dict) or not isinstance(neighbor.get("id"), str) or not neighbor["id"].strip():
            return False, "invalid_neighbor_identity"
        identities.append(neighbor["id"])
        if _valid_roi(neighbor.get("external_roi_uv")):
            localized.add(neighbor["id"])
    if len(set(identities)) != len(identities):
        return False, "duplicate_neighbor_identity"
    visible = {box["id"] for box in intent["obstacles"]}
    all_regions = visible | {box["id"] for box in intent["unknown_regions"]}
    if not set(identities).issubset(all_regions):
        return False, "neighbor_without_metric_hypothesis"
    if assessment == "present" and not identities:
        return False, "present_without_neighbors"
    if assessment == "none_relevant" and (identities or visible):
        return False, "none_relevant_conflicts_with_visible_obstacles"
    if not visible.issubset(localized):
        return False, "visible_obstacle_without_localized_image_support"
    return True, "explicit_localized_rgb_assessment"


def _distance_to_box(points, low, high):
    points = np.asarray(points, dtype=float)
    outside = np.maximum(np.maximum(np.asarray(low)-points, points-np.asarray(high)), 0.)
    return np.linalg.norm(outside, axis=-1)


class RiskRegionDistancePlanner(DistanceFieldPlanner):
    """Weighted Dijkstra with continuous hard boxes and finite box halos."""

    def __init__(self, hard_boxes, risk_regions, *, risk_weight=RISK_WEIGHT):
        if isinstance(risk_weight, bool) or not isinstance(risk_weight, (int, float)) or not math.isfinite(risk_weight) or risk_weight < 0:
            raise ValueError("risk_weight must be finite and nonnegative")
        self.hard_boxes = [(np.asarray(low, float).copy(), np.asarray(high, float).copy())
                           for low, high in hard_boxes]
        self.risk_regions = copy.deepcopy(risk_regions)
        self.risk_weight = float(risk_weight)
        super().__init__(PlannerConfig(horizon=10))

    def _boxes(self, field, compact):
        return self.hard_boxes

    def _density(self, points, field, compact):
        points = np.asarray(points, dtype=float)
        density = np.ones(points.shape[:-1], dtype=float)
        for region in self.risk_regions:
            distance = _distance_to_box(points, region["lower_xyz"], region["upper_xyz"])
            density += self.risk_weight * np.exp(-.5*(distance/region["sigma_m"])**2)
        return density

    def _key(self, field, goal, tolerance, compact):
        base = super()._key(field, goal, tolerance, compact)
        # Include continuous geometry even when it is thinner than the grid and
        # therefore changes no vertex mask. Sigma and weight also affect cost.
        payload = {"base": base, "risk_weight": self.risk_weight,
                   "hard_boxes": [[low.tolist(), high.tolist()] for low, high in self.hard_boxes],
                   "risk_regions": self.risk_regions,
                   "density_contract": "one_plus_weighted_gaussian_distance_to_tool_core_v1"}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def plan(self, *args, **kwargs):
        result = super().plan(*args, **kwargs)
        weighted = bool(self.risk_regions and self.risk_weight > 0.)
        metadata = {**(result.search_metadata or {}),
                    "cost_unit": "weighted_m" if weighted else "m",
                    "risk_density": "1 + weight * sum(exp(-0.5*(distance_to_tool_core/sigma)^2))",
                    "risk_weight": self.risk_weight,
                    "risk_region_count": len(self.risk_regions),
                    "risk_is_probability": False,
                    "geometry_is_model_hypothesis": True}
        return replace(result, search_metadata=metadata)


class MetricFieldExecutorV9(MetricFieldExecutorV8):
    """V8 measured tracking with explicitly authorized uncertainty soft costs.

    ``configure(..., scene_assessment=None)`` is strict by default on every call.
    The policy may opt in only with the model's localized RGB motion context.
    No region is removed and the current point is never carved out of a mask.
    """

    def __init__(self):
        super().__init__()
        self.soft_uncertainty_enabled = False
        self.scene_assessment_reason = "not_configured"
        self.risk_regions = []
        self.region_geometry = []

    def configure(self, state8, intent, *, scene_assessment=None):
        checked = validate_metric_intent_v6(intent)
        enabled, reason = _assessed_visible_geometry(scene_assessment, checked)
        self.soft_uncertainty_enabled = enabled
        self.scene_assessment_reason = reason
        self.risk_regions, self.region_geometry = [], []
        # Preserve orientation, budgets, tracker state and validated intent.
        # Rebuild only this opt-in version's hard masks and risk-aware planner.
        super().configure(state8, checked)
        self.boxes, self.regions = [], []
        masks = {name: np.zeros(self.grid.count, dtype=bool) for name in ("obstacles", "unknown_regions")}
        points = self.grid.coordinates(self.origin)
        tool_radius = self.intent["tool_radius_m"]
        for group in masks:
            for region in self.intent[group]:
                nominal_low = self.origin+np.asarray(region["lower_offset_m"])
                nominal_high = self.origin+np.asarray(region["upper_offset_m"])
                sigma = self.intent["uncertainty_m"]+region["uncertainty_m"]
                core_low, core_high = nominal_low-tool_radius, nominal_high+tool_radius
                full_low, full_high = core_low-sigma, core_high+sigma
                soft = enabled and group == "obstacles"
                low, high = (core_low, core_high) if soft else (full_low, full_high)
                mode = ("hard_tool_core_soft_uncertainty" if soft else
                        "hard_full_unknown" if group == "unknown_regions" else "hard_full_strict_assessment")
                self.boxes.append((low.copy(), high.copy()))
                self.regions.append({"id": region["id"], "kind": group,
                    "inflation_m": tool_radius if soft else tool_radius+sigma,
                    "lower_xyz": low.tolist(), "upper_xyz": high.tolist(), "geometry_mode": mode})
                self.region_geometry.append({"id": region["id"], "kind": group, "geometry_mode": mode,
                    "nominal": {"lower_xyz": nominal_low.tolist(), "upper_xyz": nominal_high.tolist()},
                    "tool_core": {"lower_xyz": core_low.tolist(), "upper_xyz": core_high.tolist()},
                    "full_halo": {"lower_xyz": full_low.tolist(), "upper_xyz": full_high.tolist()},
                    "margins_m": {"tool_radius": tool_radius, "intent_uncertainty": self.intent["uncertainty_m"],
                        "region_uncertainty": region["uncertainty_m"], "combined_uncertainty": sigma,
                        "full_hard_inflation_if_strict": tool_radius+sigma},
                    "hard_collision_hypothesis": True, "certified_occupied": False})
                masks[group] |= np.all((points >= low) & (points <= high), axis=1)
                if soft:
                    self.risk_regions.append({"id": region["id"], "lower_xyz": core_low.tolist(),
                                              "upper_xyz": core_high.tolist(), "sigma_m": sigma})
        prediction = replace(self.field.prediction,
            blocked_indices=tuple(np.flatnonzero(masks["obstacles"]).tolist()),
            unknown_indices=tuple(np.flatnonzero(masks["unknown_regions"] & ~masks["obstacles"]).tolist()))
        self.field = GridField(self.grid, tuple(self.origin), prediction)
        if self.locked_axes:
            self.field = _AxisConstrainedField(self.field, self.locked_axes, self.origin)
        self.planner = RiskRegionDistancePlanner(self.boxes, self.risk_regions)
        return self.summary()

    def summary(self):
        result = super().summary()
        if self.intent is None:
            return result
        feasibility = []
        for geometry in self.region_geometry:
            record = copy.deepcopy(geometry)
            for name in ("nominal", "tool_core", "full_halo"):
                low, high = np.asarray(record[name]["lower_xyz"]), np.asarray(record[name]["upper_xyz"])
                record[name]["contains_current"] = bool(np.all(self.expected_state[:3] >= low-1e-12)
                    and np.all(self.expected_state[:3] <= high+1e-12))
                record[name]["contains_goal"] = bool(np.all(self.goal >= low-1e-12) and np.all(self.goal <= high+1e-12))
            record["current_only_in_uncertainty_halo"] = (record["full_halo"]["contains_current"]
                                                           and not record["tool_core"]["contains_current"])
            feasibility.append(record)
        result.update(version="metric-hypothesis-risk-distance-executor-v9",
            field_semantics="weighted_graph_cost_on_model_hypotheses_with_hard_cores_and_optional_uncertainty_risk",
            soft_uncertainty_enabled=self.soft_uncertainty_enabled,
            scene_assessment_reason=self.scene_assessment_reason,
            risk_density={"formula": "1 + 8 * sum(exp(-0.5*(distance_to_tool_core/sigma)^2))",
                          "sigma": "intent_uncertainty_m + region_uncertainty_m",
                          "minimum": 1., "region_count": len(self.risk_regions),
                          "unknown_regions": "always_hard_full_inflation",
                          "is_probability": False, "clearance_certified": False},
            region_feasibility=feasibility)
        return result
