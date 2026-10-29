"""Axis-constrained field motion and explicitly requested local gripper settling.

Only measured robot state and model hypotheses enter these executors. Neither
finger settling nor local goal arrival establishes attachment or task success.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np

from .action import _quaternion_product, _rotvec_quaternion
from .field import GridField
from .visual_field import PANDA_JOINT_LOWER, PANDA_JOINT_UPPER
from .visual_field_v4 import _vector, validate_metric_intent
from .visual_field_v5 import MetricFieldExecutorV5


LOCKED_AXIS_TOLERANCE_M = .003
_LOCKED_AXES = {"xyz": (), "xy": (2,), "z": (0, 1)}


def validate_metric_intent_v6(intent):
    """Require a declared motion subspace; never discard a forbidden goal axis."""
    if not isinstance(intent, dict) or "motion_axes" not in intent:
        raise ValueError("V6 field intent requires motion_axes")
    axes = intent["motion_axes"]
    if not isinstance(axes, str) or axes not in _LOCKED_AXES:
        raise ValueError("motion_axes must be xyz, xy, or z")
    checked = validate_metric_intent(
        {key: value for key, value in intent.items() if key != "motion_axes"},
        action_limit=120,
    )
    if any(abs(checked["goal_offset_m"][axis]) > 1e-12 for axis in _LOCKED_AXES[axes]):
        raise ValueError("goal_offset_m must be zero on locked motion axes")
    checked["motion_axes"] = axes
    return checked


class _AxisConstrainedField(GridField):
    """Restrict graph nodes and continuous segments without erasing obstacles.

    With 5 mm lattice spacing the 3 mm corridor retains one plane (xy) or one
    line (z) of graph vertices. Actual measured poses may lie between vertices;
    their connectors are checked against the original obstacle interpolation
    and the exact convex axis corridor, rather than rejected for requiring an
    off-plane interpolation corner. This exception applies only to the axis
    restriction, never to obstacle or unknown masks.
    """

    def __init__(self, obstacle_field, locked_axes, origin):
        points = obstacle_field.grid.coordinates(obstacle_field.center)
        outside = np.any(
            abs(points[:, list(locked_axes)]-np.asarray(origin)[list(locked_axes)])
            > LOCKED_AXIS_TOLERANCE_M+1e-12,
            axis=1,
        )
        blocked = set(obstacle_field.prediction.blocked_indices)
        blocked.update(np.flatnonzero(outside).tolist())
        prediction = replace(obstacle_field.prediction,
            blocked_indices=tuple(sorted(blocked)),
            unknown_indices=tuple(index for index in obstacle_field.prediction.unknown_indices
                                  if index not in blocked))
        super().__init__(obstacle_field.grid, obstacle_field.center, prediction)
        object.__setattr__(self, "_obstacle_field", obstacle_field)
        object.__setattr__(self, "_locked_axes", tuple(locked_axes))
        object.__setattr__(self, "_axis_origin", np.asarray(origin).copy())

    def inside_corridor(self, position):
        try:
            point = _vector(position, 3, "position")
        except ValueError:
            return False
        axes = list(self._locked_axes)
        return bool(np.all(abs(point[axes]-self._axis_origin[axes])
                           <= LOCKED_AXIS_TOLERANCE_M+1e-12))

    def query(self, position):
        return self._obstacle_field.query(position) if self.inside_corridor(position) else None

    def segment_valid(self, start, end):
        # The corridor is convex: both endpoints inside implies every point
        # of a straight segment is inside. Obstacle checks remain continuous.
        return (self.inside_corridor(start) and self.inside_corridor(end)
                and self._obstacle_field.segment_valid(start, end))


class MetricFieldExecutorV6(MetricFieldExecutorV5):
    """V5 distance planning restricted to model-authorized translational axes."""

    _validate_intent = staticmethod(validate_metric_intent_v6)

    def configure(self, state8, intent):
        super().configure(state8, intent)
        self.locked_axes = _LOCKED_AXES[self.intent["motion_axes"]]
        if self.locked_axes:
            self.field = _AxisConstrainedField(self.field, self.locked_axes, self.origin)
        return self.summary()

    def _outside_corridor(self, position):
        axes = list(_LOCKED_AXES[self.intent["motion_axes"]])
        return bool(np.any(abs(np.asarray(position)[axes]-self.origin[axes])
                           > LOCKED_AXIS_TOLERANCE_M+1e-12))

    def select(self, state8, joint_position, *, command_cap_m=None):
        # Preserve acknowledgement/state validation before interpreting drift.
        if self.intent is not None and self.pending is None and self.stop_reason is None:
            state = _vector(state8, 8, "state8")
            if (np.allclose(state, self.expected_state, rtol=0., atol=1e-9)
                    and self._outside_corridor(state[:3])):
                return self._stop("locked_axis_pose_drift")
        return super().select(state8, joint_position, command_cap_m=command_cap_m)

    def observe(self, before_state8, action, after_state8):
        super().observe(before_state8, action, after_state8)
        if self._outside_corridor(self.expected_state[:3]):
            self.stop_reason = "locked_axis_pose_drift"
        return self.summary()

    def retarget(self, goal_xyz):
        if self.intent is None:
            return False
        try:
            goal = _vector(goal_xyz, 3, "goal_xyz")
        except ValueError:
            return False
        axes = list(_LOCKED_AXES[self.intent["motion_axes"]])
        if np.any(abs(goal[axes]-self.origin[axes]) > 1e-12):
            return False
        # Retain V5's small, caller-validated XY correction contract. A z-only
        # intent therefore cannot acquire a lateral target through retargeting.
        return super().retarget(goal)

    def summary(self):
        result = super().summary()
        if self.intent is not None:
            result.update(version="metric-hypothesis-distance-executor-v6",
                motion_axes=self.intent["motion_axes"],
                locked_axis_tolerance_m=LOCKED_AXIS_TOLERANCE_M,
                locked_axis_origin_xyz=self.origin.tolist())
        return result


def validate_gripper_settle_intent(intent):
    if not isinstance(intent, dict) or set(intent) != {"command", "max_actions"}:
        raise ValueError("Gripper settle intent requires command and max_actions")
    if intent["command"] not in ("open", "close"):
        raise ValueError("Gripper settle requires an explicit open or close command")
    if type(intent["max_actions"]) is not int or not 10 <= intent["max_actions"] <= 30:
        raise ValueError("Gripper max_actions must be an integer in [10,30]")
    return copy.deepcopy(intent)


class GripperSettleExecutor:
    """Bounded open/close at the measured pose, without automatic grasp claims.

    The existing incremental controller receives zero translation and rotation;
    it holds the current measured pose each control step. We monitor cumulative
    drift from the original intent pose and stop instead of pursuing a moving
    hold target. The caller supplies actions at the existing 20 Hz control rate.
    """

    MIN_ACTIONS = 5
    STABLE_CHANGES = 3
    FINGER_STABILITY_M = .0002
    MAX_POSE_DRIFT_M = .005
    MAX_ORIENTATION_DRIFT_RAD = .04

    def __init__(self):
        self.intent = None
        self.pending = None
        self.stop_reason = None

    def configure(self, state8, intent):
        state = _vector(state8, 8, "state8")
        checked = validate_gripper_settle_intent(intent)
        self.intent = checked
        self.origin_state, self.expected_state = state.copy(), state.copy()
        self.origin_orientation = _rotvec_quaternion(state[3:6])
        self.actions = self.stable_changes = 0
        self.last_finger_change_m = None
        self.max_measured_pose_drift_m = self.max_measured_orientation_drift_rad = 0.
        self.pending, self.stop_reason = None, None
        return self.summary()

    def _pose_drift(self, state):
        position = float(np.linalg.norm(state[:3]-self.origin_state[:3]))
        current = _rotvec_quaternion(state[3:6])
        origin = self.origin_orientation
        difference = _quaternion_product(current, np.r_[-origin[:3], origin[3]])
        orientation = float(2*math.atan2(np.linalg.norm(difference[:3]), abs(difference[3])))
        return position, orientation

    def _result(self, status, *, action=None, accepted=False):
        return {"action": [0.]*7 if action is None else [float(value) for value in action],
                "accepted": bool(accepted), "status": status, "need_refresh": not accepted,
                "diagnostics": self.summary()}

    def _stop(self, reason):
        self.stop_reason = reason
        return self._result(reason)

    def select(self, state8, joint_position):
        if self.intent is None:
            raise RuntimeError("configure is required before select")
        if self.pending is not None:
            raise RuntimeError("observe must acknowledge the previous accepted action")
        state = _vector(state8, 8, "state8")
        joints = _vector(joint_position, 7, "joint_position")
        if self.stop_reason is not None:
            return self._result(self.stop_reason)
        if not np.allclose(state, self.expected_state, rtol=0., atol=1e-9):
            return self._stop("unacknowledged_state_change")
        if (np.any(joints < np.asarray(PANDA_JOINT_LOWER)+.005)
                or np.any(joints > np.asarray(PANDA_JOINT_UPPER)-.005)):
            return self._stop("measured_joint_limit")
        position, orientation = self._pose_drift(state)
        if position > self.MAX_POSE_DRIFT_M:
            return self._stop("gripper_pose_drift")
        if orientation > self.MAX_ORIENTATION_DRIFT_RAD:
            return self._stop("gripper_orientation_drift")
        if self.actions >= self.MIN_ACTIONS and self.stable_changes >= self.STABLE_CHANGES:
            return self._stop("gripper_settled")
        if self.actions >= self.intent["max_actions"]:
            return self._stop("gripper_action_budget")
        action = np.r_[np.zeros(6), 1. if self.intent["command"] == "close" else -1.]
        self.pending = state.copy(), action.copy()
        return self._result("gripper_settling", action=action, accepted=True)

    def observe(self, before_state8, action, after_state8):
        before = _vector(before_state8, 8, "before_state8")
        command = _vector(action, 7, "action")
        after = _vector(after_state8, 8, "after_state8")
        if self.intent is None:
            raise RuntimeError("configure is required before observe")
        if self.pending is None:
            if np.linalg.norm(command) > 1e-12:
                raise RuntimeError("No accepted gripper command to acknowledge")
            self.expected_state = after.copy()
            return self.summary()
        expected, selected = self.pending
        if (not np.allclose(before, expected, rtol=0., atol=1e-9)
                or not np.allclose(command, selected, rtol=0., atol=1e-9)):
            raise ValueError("Acknowledged transition must match the selected gripper command")
        self.pending, self.expected_state = None, after.copy()
        self.actions += 1
        self.last_finger_change_m = float(np.max(abs(after[6:8]-before[6:8])))
        self.stable_changes = (self.stable_changes+1
                               if self.last_finger_change_m < self.FINGER_STABILITY_M else 0)
        position, orientation = self._pose_drift(after)
        self.max_measured_pose_drift_m = max(position, self.max_measured_pose_drift_m)
        self.max_measured_orientation_drift_rad = max(orientation, self.max_measured_orientation_drift_rad)
        if position > self.MAX_POSE_DRIFT_M:
            self.stop_reason = "gripper_pose_drift"
        elif orientation > self.MAX_ORIENTATION_DRIFT_RAD:
            self.stop_reason = "gripper_orientation_drift"
        elif self.actions >= self.MIN_ACTIONS and self.stable_changes >= self.STABLE_CHANGES:
            self.stop_reason = "gripper_settled"
        elif self.actions >= self.intent["max_actions"]:
            self.stop_reason = "gripper_action_budget"
        return self.summary()

    def summary(self):
        result = {"configured": self.intent is not None,
                  "version": "bounded-gripper-settle-executor-v6",
                  "grasp_certified": False, "task_success_certified": False,
                  "clearance_certified": False}
        if self.intent is not None:
            result.update(gripper_command=self.intent["command"],
                max_actions=self.intent["max_actions"], executed_actions=self.actions,
                consecutive_stable_finger_changes=self.stable_changes,
                last_finger_change_m=self.last_finger_change_m,
                measured_pose_drift_m=self._pose_drift(self.expected_state)[0],
                measured_orientation_drift_rad=self._pose_drift(self.expected_state)[1],
                max_measured_pose_drift_m=self.max_measured_pose_drift_m,
                max_measured_orientation_drift_rad=self.max_measured_orientation_drift_rad,
                stop_reason=self.stop_reason,
                limitations=["finger settling is not proof of attachment or release",
                             "zero pose commands cannot certify obstacle-free contact"])
        return result
