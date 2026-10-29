"""Opt-in measured absolute orientation holding during V6 distance tracking.

The installed OSC_POSE controller preserves its previous orientation goal when
all rotation commands are zero. After Direct reorientation that goal can still
be ahead of the measured pose. V8 explicitly replaces it with a bounded world-
frame correction toward the orientation measured when this intent was started.
No task orientation, object state, or camera calibration is introduced.
"""
from __future__ import annotations

import math

import numpy as np

from .action import ROTATION_SCALE_RAD, _quaternion_product, _rotvec_quaternion
from .visual_field_v4 import _vector
from .visual_field_v6 import GripperSettleExecutor, MetricFieldExecutorV6, validate_metric_intent_v6


OSC_GOAL_REFRESH_RAD = 1e-9
ORIENTATION_HOLD_MAX_CORRECTION_RAD = .02


def measured_current_pose_hold_action():
    """Zero translation/grip plus the minimal explicit OSC orientation reset."""
    return [0., 0., 0., OSC_GOAL_REFRESH_RAD / ROTATION_SCALE_RAD, 0., 0., 0.]


def _acknowledgement_command(executor, before_state8, action):
    """Recognize only the explicit rejected-command hold; consume no budget."""
    if executor.intent is None:
        raise RuntimeError("configure is required before observe")
    command = _vector(action, 7, "action")
    if executor.pending is None and np.array_equal(command, measured_current_pose_hold_action()):
        before = _vector(before_state8, 8, "before_state8")
        if not np.allclose(before, executor.expected_state, rtol=0., atol=1e-9):
            raise ValueError("Rejected pose-hold acknowledgement must match the measured state")
        return np.zeros(7)
    return command


def world_orientation_error_rotvec(target_xyzw, measured_rotvec):
    """Shortest log(R_target R_measured.T), in the OSC world-axis convention."""
    target = _vector(target_xyzw, 4, "target_xyzw")
    norm = float(np.linalg.norm(target))
    if norm < 1e-8:
        raise ValueError("target_xyzw must have nonzero norm")
    target = target / norm
    current = _rotvec_quaternion(_vector(measured_rotvec, 3, "measured_rotvec"))
    error = _quaternion_product(target, np.r_[-current[:3], current[3]])
    if error[3] < 0.:
        error = -error
    sine = float(np.linalg.norm(error[:3]))
    if sine < 1e-15:
        return np.zeros(3)
    return error[:3] * (2 * math.atan2(sine, float(error[3])) / sine)


def _bounded_hold_correction(target_xyzw, measured_rotvec):
    """Shared Field/gripper correction under the same fixed OSC contract."""
    error = world_orientation_error_rotvec(target_xyzw, measured_rotvec)
    correction = error.copy()
    magnitude = float(np.linalg.norm(correction))
    if magnitude > ORIENTATION_HOLD_MAX_CORRECTION_RAD:
        correction *= ORIENTATION_HOLD_MAX_CORRECTION_RAD / magnitude
    reset_goal = magnitude < OSC_GOAL_REFRESH_RAD
    if reset_goal:
        correction = np.array([OSC_GOAL_REFRESH_RAD, 0., 0.])
    return correction, error, reset_goal


class MetricFieldExecutorV8(MetricFieldExecutorV6):
    """Preserve V6 planning/guards, replacing only accepted rotation commands.

    The .04 rad inherited drift stop is unchanged. Corrections have a .02 rad
    vector-norm cap (not a per-axis cap), and are acknowledged as part of the
    exact selected command. They are not a collision-clearance certificate.
    """

    MAX_CORRECTION_RAD = ORIENTATION_HOLD_MAX_CORRECTION_RAD
    MAX_DRIFT_RAD = .04
    HOLD_GAIN = 1.
    # OSC ignores exactly-zero orientation commands instead of updating its
    # old goal. An explicitly logged 1 nanoradian command clears that stale
    # goal at initial/exact alignment; it is not a meaningful reorientation.
    OSC_GOAL_REFRESH_RAD = OSC_GOAL_REFRESH_RAD

    def __init__(self):
        super().__init__()
        self.hold_orientation = None
        self.max_orientation_error_rad = 0.
        self.last_orientation_command_rad = np.zeros(3)
        self.orientation_goal_refresh_commands = 0

    def configure(self, state8, intent):
        state = _vector(state8, 8, "state8")
        checked = validate_metric_intent_v6(intent)
        measured = _rotvec_quaternion(state[3:6])
        if np.linalg.norm(world_orientation_error_rotvec(
                checked["orientation_target_xyzw"], state[3:6])) > 1e-6:
            raise ValueError("V8 orientation target must match the measured refresh pose")
        # The hold target is measured locally, even if the equivalent supplied
        # quaternion has the opposite sign or small serialization roundoff.
        checked["orientation_target_xyzw"] = measured.tolist()
        self.hold_orientation = measured.copy()
        self.max_orientation_error_rad = 0.
        self.last_orientation_command_rad = np.zeros(3)
        self.orientation_goal_refresh_commands = 0
        return super().configure(state, checked)

    def select(self, state8, joint_position, *, command_cap_m=None):
        result = super().select(state8, joint_position, command_cap_m=command_cap_m)
        if not result["accepted"]:
            return result
        state = _vector(state8, 8, "state8")
        correction, error, reset_goal = _bounded_hold_correction(self.hold_orientation, state[3:6])
        magnitude = float(np.linalg.norm(error))
        action = np.asarray(result["action"], dtype=float)
        action[3:6] = correction / ROTATION_SCALE_RAD
        self.pending = state.copy(), action.copy()
        self.last_orientation_command_rad = correction.copy()
        self.max_orientation_error_rad = max(self.max_orientation_error_rad, magnitude)
        self.orientation_goal_refresh_commands += int(reset_goal)
        result["action"] = action.tolist()
        result["diagnostics"].update(self.summary())
        result["diagnostics"]["orientation_hold"].update(
            error_world_rotvec_rad=error.tolist(),
            goal_refresh_epsilon_used=reset_goal,
        )
        return result

    def observe(self, before_state8, action, after_state8):
        acknowledged = _acknowledgement_command(self, before_state8, action)
        super().observe(before_state8, acknowledged, after_state8)
        drift = float(np.linalg.norm(world_orientation_error_rotvec(
            self.hold_orientation, self.expected_state[3:6])))
        self.max_orientation_error_rad = max(self.max_orientation_error_rad, drift)
        if drift > self.MAX_DRIFT_RAD and self.stop_reason is None:
            self.stop_reason = "orientation_requires_direct"
        return self.summary()

    def summary(self):
        result = super().summary()
        if self.intent is not None:
            result.update(version="metric-hypothesis-distance-executor-v8")
            result["orientation_hold"] = {
                "target_source": "measured_orientation_at_intent_refresh",
                "target_xyzw": self.hold_orientation.tolist(),
                "error_frame": "world_log_R_target_R_current_transpose",
                "gain": self.HOLD_GAIN,
                "max_correction_rad": self.MAX_CORRECTION_RAD,
                "hard_drift_limit_rad": self.MAX_DRIFT_RAD,
                "max_measured_error_rad": self.max_orientation_error_rad,
                "last_command_world_rotvec_rad": self.last_orientation_command_rad.tolist(),
                "osc_goal_refresh_epsilon_rad": self.OSC_GOAL_REFRESH_RAD,
                "osc_goal_refresh_commands": self.orientation_goal_refresh_commands,
                "clearance_certified": False,
            }
        return result


class GripperSettleExecutorV8(GripperSettleExecutor):
    """V6 explicit finger settling with the measured orientation held locally.

    Zero XYZ and explicit +/-1 grip remain unchanged, as do action budgets,
    finger stability, joint and pose-drift guards. A settled gripper still does
    not establish attachment. Rotation only corrects toward the configure pose.
    """

    def __init__(self):
        super().__init__()
        self.last_orientation_command_rad = np.zeros(3)
        self.orientation_goal_refresh_commands = 0

    def configure(self, state8, intent):
        self.last_orientation_command_rad = np.zeros(3)
        self.orientation_goal_refresh_commands = 0
        return super().configure(state8, intent)

    def select(self, state8, joint_position):
        result = super().select(state8, joint_position)
        if not result["accepted"]:
            return result
        state = _vector(state8, 8, "state8")
        correction, error, reset_goal = _bounded_hold_correction(self.origin_orientation, state[3:6])
        action = np.asarray(result["action"], dtype=float)
        action[3:6] = correction / ROTATION_SCALE_RAD
        self.pending = state.copy(), action.copy()
        self.last_orientation_command_rad = correction.copy()
        self.orientation_goal_refresh_commands += int(reset_goal)
        result["action"] = action.tolist()
        result["diagnostics"].update(self.summary())
        result["diagnostics"]["orientation_hold"].update(
            error_world_rotvec_rad=error.tolist(), goal_refresh_epsilon_used=reset_goal)
        return result

    def observe(self, before_state8, action, after_state8):
        acknowledged = _acknowledgement_command(self, before_state8, action)
        return super().observe(before_state8, acknowledged, after_state8)

    def summary(self):
        result = super().summary()
        if self.intent is not None:
            result.update(version="bounded-gripper-settle-executor-v8")
            result["orientation_hold"] = {
                "target_source": "measured_orientation_at_intent_refresh",
                "target_xyzw": self.origin_orientation.tolist(),
                "error_frame": "world_log_R_target_R_current_transpose",
                "gain": 1.,
                "max_correction_rad": ORIENTATION_HOLD_MAX_CORRECTION_RAD,
                "hard_drift_limit_rad": self.MAX_ORIENTATION_DRIFT_RAD,
                "last_command_world_rotvec_rad": self.last_orientation_command_rad.tolist(),
                "osc_goal_refresh_epsilon_rad": OSC_GOAL_REFRESH_RAD,
                "osc_goal_refresh_commands": self.orientation_goal_refresh_commands,
                "clearance_certified": False,
            }
        return result

