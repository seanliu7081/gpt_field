"""Target-contact configuration obstacles and bounded visual Field retargets.

Targets stay represented when other obstacle lists are empty. Permissions apply
to explicit robot parts, never to an entire target object. All scene geometry is
still a model hypothesis; this module supplies no calibrated clearance claim.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import numpy as np

from .field import GridField
from .visual_field_v4 import _vector
from .visual_field_v6 import _AxisConstrainedField, _LOCKED_AXES
from .visual_field_v10 import SubgoalLedger
from .visual_field_v11 import ToolEnvelopeRiskPlanner, _bounds
from .visual_field_v12 import MetricFieldExecutorV12


class ContactSubgoalLedger(SubgoalLedger):
    """An explicit two-view contact objective permits bounded 3D refinement."""

    def enable_contact_refinement(self, enabled):
        if self.active is not None:
            self.active["contact_refinement_enabled"] = bool(enabled)
            self.active.setdefault("contact_correction_travel_m", 0.)
            self.active.setdefault("contact_correction_count", 0)

    def check_contact_correction(self, goal_xyz):
        item = self.active
        if item is None or not item.get("contact_refinement_enabled"):
            return False
        try:
            proposed = _vector(goal_xyz, 3, "goal_xyz")
        except ValueError:
            return False
        old, original = np.asarray(item["world_goal_xyz"]), np.asarray(item["model_world_goal_xyz"])
        locked = list(_LOCKED_AXES[item["motion_axes"]])
        shift = float(np.linalg.norm(proposed-old))
        return bool(np.allclose(proposed[locked], old[locked], atol=1e-12, rtol=0.)
            and shift <= .006+1e-12 and np.linalg.norm(proposed-original) <= .030+1e-12
            and item.get("contact_correction_travel_m", 0.)+shift <= .060+1e-12)

    def record_contact_correction(self, goal_xyz):
        if not self.check_contact_correction(goal_xyz):
            return False
        item = self.active
        old = np.asarray(item["world_goal_xyz"])
        proposed = np.asarray(goal_xyz, float)
        shift = float(np.linalg.norm(proposed-old))
        if shift <= 1e-12:
            return True
        item["world_goal_xyz"] = proposed.tolist()
        item["contact_correction_travel_m"] += shift
        item["contact_correction_count"] += 1
        item["last_contact_correction"] = {
            "previous_world_goal_xyz": old.tolist(), "world_goal_xyz": proposed.tolist(),
            "shift_m": shift, "source": "causally_supported_two_view_contact_alignment",
            "task_success_certified": False}
        # Only a validated new local objective can reopen numerical arrival.
        # Neither its visual alignment nor its later arrival implies a grasp.
        item["local_complete"] = False
        item["completion_reason"] = None
        return True


class MetricFieldExecutorV13(MetricFieldExecutorV12):
    def __init__(self):
        super().__init__()
        self.contact_geometry = None
        self.target_configuration = None
        self.contact_refinement_enabled = False

    def configure(self, state8, intent, *, scene_assessment=None, subgoal_origin_xyz=None):
        self.target_configuration = None
        super().configure(state8, intent, scene_assessment=scene_assessment,
                          subgoal_origin_xyz=subgoal_origin_xyz)
        if self.contact_geometry is None:
            return self.summary()
        target = self.contact_geometry.configuration(state8,
            orientation_margin_rad=self.MAX_DRIFT_RAD, full_finger_range=True)
        self.target_configuration = copy.deepcopy(target)
        parts = target.get("core_parts") or []
        if not parts:
            return self.summary()
        halos = target["uncertainty_halo_parts"]
        # The primary target is explicitly localized by its own current model
        # geometry contract. Unavailable retained geometry stays fully inflated.
        soft = bool(target["available"] and scene_assessment
                    and scene_assessment.get("obstacle_assessment") != "unobservable"
                    and scene_assessment.get("tool_external_roi_uv") is not None)
        hard_parts = parts if soft else halos
        points = self.grid.coordinates(self.origin)
        mask = np.zeros(self.grid.count, bool)
        for part in hard_parts:
            lo, hi = np.asarray(part["lower_xyz"]), np.asarray(part["upper_xyz"])
            self.boxes.append((lo, hi))
            mask |= np.all((points >= lo) & (points <= hi), axis=1)
        sigma = max(float(target["sigma_m"]), .001)
        geometry = {"id": target["target_id"], "kind": "interaction_target",
            "geometry_mode": "part_specific_target_contact",
            "nominal": copy.deepcopy(target["nominal"]),
            "tool_core": _bounds(parts), "full_halo": _bounds(halos),
            "tool_core_parts": copy.deepcopy(parts), "full_halo_parts": copy.deepcopy(halos),
            "union_bounds_are_display_only": True,
            "margins_m": {"tool_radius": self.intent["tool_radius_m"],
                "intent_uncertainty": 0., "region_uncertainty": sigma, "combined_uncertainty": sigma},
            "allowed_contact": target["effective_allowed_contact"],
            "hard_collision_hypothesis": True, "certified_occupied": False}
        self.region_geometry.append(geometry)
        self.regions.append({"id": target["target_id"], "kind": "interaction_target",
            "geometry_mode": geometry["geometry_mode"], "hard_part_count": len(parts),
            "exempt_parts": copy.deepcopy(target["exempt_parts"]),
            "bounds_semantics": "union_of_forbidden_robot_part_configuration_boxes"})
        if soft:
            self.risk_regions.append({"id": target["target_id"], "parts": copy.deepcopy(parts), "sigma_m": sigma})
        # Axis exclusions restrict graph vertices, not obstacle interpolation.
        # Merge the target into the underlying scene field so an acknowledged
        # subgrid lateral drift inside the exact axis corridor stays valid.
        # Existing obstacle and unknown masks remain untouched below.
        obstacle_field = self.field
        while isinstance(obstacle_field, _AxisConstrainedField):
            obstacle_field = obstacle_field._obstacle_field
        blocked = set(obstacle_field.prediction.blocked_indices) | set(np.flatnonzero(mask).tolist())
        pred = replace(obstacle_field.prediction, blocked_indices=tuple(sorted(blocked)),
            unknown_indices=tuple(i for i in obstacle_field.prediction.unknown_indices if i not in blocked))
        self.field = GridField(self.grid, tuple(self.origin), pred)
        if self.locked_axes:
            self.field = _AxisConstrainedField(self.field, self.locked_axes, self.origin)
            self.field = _AxisConstrainedField(self.field, self.locked_axes, self.subgoal_origin)
        self.planner = ToolEnvelopeRiskPlanner(self.boxes, self.risk_regions)
        return self.summary()

    def retarget_contact(self, goal_xyz):
        if (not self.contact_refinement_enabled or self.intent is None or self.pending is not None
                or self.stop_reason is not None):
            return False
        try:
            goal = _vector(goal_xyz, 3, "goal_xyz")
        except ValueError:
            return False
        locked = list(_LOCKED_AXES[self.intent["motion_axes"]])
        shift = float(np.linalg.norm(goal-self.goal))
        if (not np.allclose(goal[locked], self.goal[locked], atol=1e-12, rtol=0.)
                or shift > .006+1e-12 or self.goal_revision_travel_m+shift > .06+1e-12
                or np.linalg.norm(goal-self.original_goal) > .03+1e-12
                or not self.checked.segment_valid(goal, goal)):
            return False
        self.goal = goal.copy()
        self.goal_revision_travel_m += shift
        self.goal_revisions += 1
        self.compact["goal_offset_m"] = (goal-self.origin).tolist()
        self.tracker.reset()
        self.planner.reset_cache()
        return True

    def summary(self):
        result = super().summary()
        result.update(version="part-contact-visual-goal-field-v13",
            target_contact_geometry=copy.deepcopy(getattr(self, "target_configuration", None)),
            contact_refinement_enabled=getattr(self, "contact_refinement_enabled", False))
        return result
