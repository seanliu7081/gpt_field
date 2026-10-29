"""Bounded metric-hypothesis distance-field execution using measured robot pose.

The scene boxes are model hypotheses from RGB, not measured geometry. Inflated
blocked and unknown regions are retained throughout an intent. This controller
does not certify robot-link clearance, grasp state, or task completion.
"""
from __future__ import annotations

import copy
import math
from numbers import Real

import numpy as np

from .action import TRANSLATION_SCALE_M, _quaternion_product, _rotvec_quaternion
from .distance_planner import DistanceFieldPlanner
from .field import GridField
from .planner import PlannerConfig
from .schema import FieldPrediction, GridSpec
from .tracking import PathTracker
from .visual_field import PANDA_JOINT_LOWER, PANDA_JOINT_UPPER


def _vector(value, length, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} finite numbers")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) for x in value):
        raise ValueError(f"{name} must contain real numbers")
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def _number(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low},{high}]")
    return float(value)


def validate_metric_intent(intent, *, action_limit=60):
    """Independent execution-bound validation; no silent clipping or carving."""
    keys = {"goal_offset_m", "orientation_target_xyzw", "gripper_command", "goal_tolerance_m",
            "max_actions", "max_travel_m", "command_limit_m", "tool_radius_m", "uncertainty_m",
            "obstacles", "unknown_regions"}
    if not isinstance(intent, dict) or set(intent) != keys:
        raise ValueError("Metric field intent has missing or unexpected fields")
    goal = _vector(intent["goal_offset_m"], 3, "goal_offset_m")
    if np.any(abs(goal) > .08):
        raise ValueError("goal_offset_m must remain within the fixed 16 cm grid")
    orientation = _vector(intent["orientation_target_xyzw"], 4, "orientation_target_xyzw")
    if abs(np.linalg.norm(orientation)-1.) > .001:
        raise ValueError("orientation_target_xyzw must be a unit quaternion")
    if intent["gripper_command"] != "hold":
        raise ValueError("Metric Field only permits gripper hold; use Direct for contact")
    if type(intent["max_actions"]) is not int or not 10 <= intent["max_actions"] <= action_limit:
        raise ValueError(f"max_actions must be an integer in [10,{action_limit}]")
    for name, low, high in (("goal_tolerance_m", .003, .01), ("max_travel_m", .005, .16),
                            ("command_limit_m", .002, .02), ("tool_radius_m", .005, .08),
                            ("uncertainty_m", .001, .04)):
        _number(intent[name], low, high, name)
    ids = set()
    for name in ("obstacles", "unknown_regions"):
        if not isinstance(intent[name], list) or len(intent[name]) > 8:
            raise ValueError(f"{name} must contain at most eight regions")
        for region in intent[name]:
            if not isinstance(region, dict) or set(region) != {"id", "lower_offset_m", "upper_offset_m", "uncertainty_m"}:
                raise ValueError("Region has missing or unexpected fields")
            if not isinstance(region["id"], str) or not region["id"].strip() or region["id"] in ids:
                raise ValueError("Region identities must be nonempty and unique")
            ids.add(region["id"])
            low = _vector(region["lower_offset_m"], 3, "lower_offset_m")
            high = _vector(region["upper_offset_m"], 3, "upper_offset_m")
            # Wire centers and half-extents are each bounded by 0.5 m. Their
            # exact sum/difference can reach 1 m; retain the full hypothesis.
            if np.any(low >= high) or np.any(abs(low) > 1.) or np.any(abs(high) > 1.):
                raise ValueError("Region bounds must be ordered within [-1,1]")
            _number(region["uncertainty_m"], .0, .1, "region uncertainty_m")
    return copy.deepcopy(intent)


class _RegionPlanner(DistanceFieldPlanner):
    """Use all v4 regions without changing the older compact schema's limits."""
    def __init__(self, boxes):
        self.boxes = boxes
        super().__init__(PlannerConfig(horizon=10))

    def _boxes(self, field, compact):
        return self.boxes


class _CheckedField:
    def __init__(self, owner):
        self.owner = owner

    def query(self, point):
        return self.owner.field.query(point) if self.segment_valid(point, point) else None

    def segment_valid(self, start, end):
        return self.owner.planner._valid_segment(self.owner.field, np.asarray(start), np.asarray(end), self.owner.boxes)


class MetricFieldExecutorV4:
    """One model intent, a cached distance field and measured-pose path tracking.

    ``select`` does not consume an action budget. Acknowledging its accepted
    action with ``observe`` does. Rotation and incremental gripper output are
    always zero. All arrival flags concern this local subgoal, never the task.
    """
    def __init__(self):
        self.intent = None
        self.pending = None
        self.stop_reason = None

    _validate_intent = staticmethod(validate_metric_intent)

    def configure(self, state8, intent):
        state = _vector(state8, 8, "state8")
        self.intent = self._validate_intent(intent)
        self.origin, self.goal = state[:3].copy(), state[:3]+self.intent["goal_offset_m"]
        self.original_goal = self.goal.copy()
        self.goal_revision_travel_m = 0.
        self.goal_revisions = 0
        self.expected_state = state.copy()
        self.grid = GridSpec(size=33, spacing_m=.005)
        self.boxes, self.regions = [], []
        for group in ("obstacles", "unknown_regions"):
            for region in self.intent[group]:
                margin = self.intent["tool_radius_m"]+self.intent["uncertainty_m"]+region["uncertainty_m"]
                low = self.origin+np.asarray(region["lower_offset_m"])-margin
                high = self.origin+np.asarray(region["upper_offset_m"])+margin
                self.boxes.append((low, high))
                self.regions.append({"id": region["id"], "kind": group, "inflation_m": margin,
                                     "lower_xyz": low.tolist(), "upper_xyz": high.tolist()})
        points = self.grid.coordinates(self.origin)
        masks = {key: np.zeros(self.grid.count, dtype=bool) for key in ("obstacles", "unknown_regions")}
        for region, (low, high) in zip(self.regions, self.boxes):
            masks[region["kind"]] |= np.all((points >= low) & (points <= high), axis=1)
        # Potentials are display-only: the planner solves its own distance grid.
        prediction = FieldPrediction(stage="bounded_metric_intent", orientation_target_xyzw=tuple(self.intent["orientation_target_xyzw"]),
            gripper_command="hold", motion_mode="move", values=tuple(np.linalg.norm(points-self.goal, axis=1)),
            blocked_indices=tuple(np.flatnonzero(masks["obstacles"]).tolist()),
            unknown_indices=tuple(np.flatnonzero(masks["unknown_regions"] & ~masks["obstacles"]).tolist()))
        self.field = GridField(self.grid, tuple(self.origin), prediction)
        self.planner = _RegionPlanner(self.boxes)
        self.checked = _CheckedField(self)
        self.compact = {"stage": "bounded_metric_intent", "orientation_target_xyzw": self.intent["orientation_target_xyzw"],
            "gripper_command": "hold", "motion_mode": "move", "goal_offset_m": self.intent["goal_offset_m"],
            "attraction_weights": [1., 1., 1.], "repulsors": [], "blocked_boxes": [], "unknown_boxes": []}
        self.tracker = PathTracker(lookahead_m=self.intent["command_limit_m"], max_command_m=self.intent["command_limit_m"],
                                   endpoint_tolerance_m=self.intent["goal_tolerance_m"]*.5)
        self.actions = self.nonzero_actions = self.effective_actions = self.plan_calls = self.cache_hits = 0
        self.measured_travel_m = self.commanded_travel_m = self.observed_overshoot_margin_m = 0.
        self.motion_window, self.last_plan = [], None
        self.pending, self.stop_reason = None, None
        return self.summary()

    def retarget(self, goal_xyz):
        """Apply a caller-validated small XY correction, preserving every bound.

        This method supplies no visual evidence itself. The outer monitor must
        establish the same target identity and validated planar response first.
        """
        if self.intent is None or self.pending is not None or self.stop_reason is not None:
            return False
        try:
            goal = _vector(goal_xyz, 3, "goal_xyz")
        except ValueError:
            return False
        shift = float(np.linalg.norm(goal-self.goal))
        if (abs(goal[2]-self.original_goal[2]) > 1e-12 or shift > .003+1e-12
                or self.goal_revision_travel_m+shift > .01+1e-12
                or np.linalg.norm(goal-self.original_goal) > .01+1e-12
                or not self.checked.segment_valid(goal, goal)):
            return False
        self.goal = goal.copy()
        self.goal_revision_travel_m += shift
        self.goal_revisions += 1
        self.compact["goal_offset_m"] = (goal-self.origin).tolist()
        self.tracker.reset()
        self.planner.reset_cache()
        return True

    def _orientation_error(self, state):
        target = np.asarray(self.intent["orientation_target_xyzw"], float)
        target /= np.linalg.norm(target)
        current = _rotvec_quaternion(state[3:6])
        error = _quaternion_product(target, np.r_[-current[:3], current[3]])
        return float(2*math.atan2(np.linalg.norm(error[:3]), abs(error[3])))

    def _result(self, status, *, accepted=False, action=None, details=None):
        return {"action": [0.]*7 if action is None else [float(x) for x in action], "accepted": bool(accepted),
                "status": status, "need_refresh": not accepted,
                "diagnostics": {**self.summary(), **(details or {})}}

    def _stop(self, reason, details=None):
        self.stop_reason = reason
        return self._result(reason, details=details)

    def select(self, state8, joint_position, *, command_cap_m=None):
        if self.intent is None:
            raise RuntimeError("configure is required before select")
        if self.pending is not None:
            raise RuntimeError("observe must acknowledge the previous accepted action")
        state, q = _vector(state8, 8, "state8"), _vector(joint_position, 7, "joint_position")
        if self.stop_reason is not None:
            return self._result(self.stop_reason)
        if not np.allclose(state, self.expected_state, rtol=0., atol=1e-9):
            return self._stop("unacknowledged_state_change")
        if np.any(q < np.asarray(PANDA_JOINT_LOWER)+.005) or np.any(q > np.asarray(PANDA_JOINT_UPPER)-.005):
            return self._stop("measured_joint_limit")
        if self._orientation_error(state) > .04:
            return self._stop("orientation_requires_direct")
        if not self.checked.segment_valid(state[:3], state[:3]):
            return self._stop("invalid_measured_start")
        if not self.checked.segment_valid(self.goal, self.goal):
            return self._stop("invalid_goal_hypothesis")
        if np.linalg.norm(state[:3]-self.goal) <= self.intent["goal_tolerance_m"]:
            return self._stop("measured_local_goal_arrival")
        if self.actions >= self.intent["max_actions"]:
            return self._stop("intent_action_budget")
        remaining = self.intent["max_travel_m"]-self.measured_travel_m-self.observed_overshoot_margin_m
        if remaining <= 1e-6:
            return self._stop("intent_travel_budget")
        cap = self.intent["command_limit_m"]
        if command_cap_m is not None:
            cap = min(cap, _number(command_cap_m, 1e-6, .02, "command_cap_m"))
        cap = min(cap, remaining)
        self.tracker.max_command_m, self.tracker.lookahead_m = cap, cap
        if self.tracker.positions is None or self.tracker.endpoint_reached(state[:3]):
            self.last_plan = self.planner.plan(self.field, state[:3], goal_xyz=self.goal,
                compact_prediction=self.compact, goal_tolerance_m=self.intent["goal_tolerance_m"]*.5)
            self.plan_calls += 1
            self.cache_hits += int(bool((self.last_plan.search_metadata or {}).get("cache_hit")))
            if self.last_plan.status != "move" or self.last_plan.selected_index is None:
                return self._stop("field_"+(self.last_plan.reason or self.last_plan.status))
            self.tracker.reset(self.last_plan.candidates[self.last_plan.selected_index].positions)
        delta, tracking = self.tracker.command(state[:3], self.checked,
            self.planner.config.workspace_min, self.planner.config.workspace_max)
        delta, point = np.asarray(delta), state[:3]
        norm = float(np.linalg.norm(delta))
        if tracking["status"] != "tracking" or norm <= 1e-9:
            return self._stop("field_"+tracking["status"], {"tracking": tracking})
        if self.measured_travel_m+norm+self.observed_overshoot_margin_m > self.intent["max_travel_m"]+1e-10:
            return self._stop("prospective_travel_budget")
        if np.linalg.norm(point+delta-self.origin)+self.observed_overshoot_margin_m > self.intent["max_travel_m"]+1e-10:
            return self._stop("prospective_radius_budget")
        if not self.checked.segment_valid(point, point+delta):
            return self._stop("invalid_command_segment")
        action = np.r_[delta/TRANSLATION_SCALE_M, 0., 0., 0., 0.]
        self.pending = state.copy(), action.copy()
        return self._result("metric_field_tracking", accepted=True, action=action,
            details={"tracking": tracking, "command_delta_m": delta.tolist(), "plan": self.last_plan.to_dict()})

    def observe(self, before_state8, action, after_state8):
        before, command, after = _vector(before_state8, 8, "before_state8"), _vector(action, 7, "action"), _vector(after_state8, 8, "after_state8")
        if self.pending is None:
            # The outer policy may acknowledge a rejected zero command. It
            # consumes no Field budget and cannot reopen a rejected intent.
            if np.linalg.norm(command) > 1e-12:
                raise RuntimeError("No accepted Field command to acknowledge")
            self.expected_state = after.copy()
            return self.summary()
        expected, selected = self.pending
        if not np.allclose(before, expected, rtol=0., atol=1e-9) or not np.allclose(command, selected, rtol=0., atol=1e-9):
            raise ValueError("Acknowledged transition must match the selected Field command")
        self.pending, self.expected_state = None, after.copy()
        measured = float(np.linalg.norm(after[:3]-before[:3]))
        commanded = float(np.linalg.norm(command[:3])*TRANSLATION_SCALE_M)
        self.actions += 1
        self.nonzero_actions += int(commanded > 1e-9)
        self.effective_actions += int(commanded > 1e-9 and measured > .0001)
        self.measured_travel_m += measured
        self.commanded_travel_m += commanded
        self.observed_overshoot_margin_m = max(self.observed_overshoot_margin_m, measured-commanded, 0.)
        self.motion_window = (self.motion_window+[(commanded, measured)])[-5:]
        if not self.checked.segment_valid(before[:3], after[:3]):
            self.stop_reason = "measured_segment_outside_hypothesis"
        elif (self.measured_travel_m > self.intent["max_travel_m"]+1e-9
              or np.linalg.norm(after[:3]-self.origin) > self.intent["max_travel_m"]+1e-9):
            self.stop_reason = "measured_travel_budget_exceeded"
        elif (len(self.motion_window) == 5 and sum(v[0] for v in self.motion_window) >= .005
              and sum(v[1] for v in self.motion_window) <= .0005):
            self.stop_reason = "measured_motion_stall"
        return self.summary()

    def summary(self):
        if self.intent is None:
            return {"configured": False, "clearance_certified": False, "task_success_certified": False}
        return {"configured": True, "version": "metric-hypothesis-distance-executor-v4",
            "field_semantics": "model_hypothesis_inflated_regions_graph_distance_m",
            "grid_size": 33, "grid_spacing_m": .005, "origin_xyz": self.origin.tolist(), "goal_xyz": self.goal.tolist(),
            "executed_actions": self.actions, "nonzero_actions": self.nonzero_actions, "effective_moving_actions": self.effective_actions,
            "max_actions": self.intent["max_actions"], "max_travel_m": self.intent["max_travel_m"],
            "measured_travel_m": self.measured_travel_m, "commanded_travel_m": self.commanded_travel_m,
            "measured_radius_m": float(np.linalg.norm(self.expected_state[:3]-self.origin)),
            "observed_overshoot_margin_m": self.observed_overshoot_margin_m,
            "plan_calls": self.plan_calls, "distance_cache_hits": self.cache_hits,
            "goal_revisions": self.goal_revisions, "goal_revision_travel_m": self.goal_revision_travel_m,
            "inflated_regions": copy.deepcopy(self.regions), "stop_reason": self.stop_reason,
            "clearance_certified": False, "task_success_certified": False,
            "limitations": ["model scene hypotheses, no supplied depth or camera calibration",
                "EEF segments only; no full-arm collision certificate", "current joint limits only, no joint-path prediction"]}
