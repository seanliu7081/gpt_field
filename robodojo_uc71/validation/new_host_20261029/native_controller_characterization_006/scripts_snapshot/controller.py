"""Candidate robot-only differential controller; not calibrated on native ARX.

No scene/object state, collision world, task recipe or reward is accepted here.
The caller supplies measured robot state and a model-authored world TCP target.
Native tracking and geometry validation are required before any approved test.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import numpy as np


class ControllerError(RuntimeError):
    """Generic local status code suitable for nonprivileged feedback."""


# Native metre-scale link poses/quaternions are float32 (about 1e-7 resolution).
# These numerical hold floors absorb pose quantization; operational arrival and
# settling remain the caller's separate, much larger measured tolerances.
POSITION_NUMERICAL_RESOLUTION_M = 1e-6
ORIENTATION_NUMERICAL_RESOLUTION_RAD = 1e-6


@dataclass(frozen=True)
class ControllerConfig:
    control_hz: float = 25.0
    transit_translation_speed_m_s: float = 0.4
    contact_translation_speed_m_s: float = 0.1
    angular_speed_rad_s: float = 2.0
    joint_target_lead_rad: float = 0.05
    damping: float = 0.02
    calibration_status: str = "uncalibrated_candidate"

    def validate(self) -> None:
        values = [self.control_hz, self.transit_translation_speed_m_s,
                  self.contact_translation_speed_m_s, self.angular_speed_rad_s,
                  self.joint_target_lead_rad, self.damping]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
            raise ControllerError("invalid_controller_configuration")
        if self.control_hz != 25 or self.calibration_status != "uncalibrated_candidate":
            raise ControllerError("unsupported_controller_configuration")


@dataclass(frozen=True)
class ArmState:
    """Measured robot-only state; Jacobian rows are world [vx,vy,vz,wx,wy,wz].

    link6_world_jacobian is 6x6 for the six measured arm joints. The TCP offset
    is the declared, fixed link6-to-TCP translation in link6 coordinates. TCP
    orientation is link6 orientation; no task/object frame is involved.
    """
    joint_positions: object
    joint_velocities: object
    link6_position: object
    link6_quaternion_wxyz: object
    link6_world_jacobian: object
    joint_lower: object
    joint_upper: object
    joint_velocity_limits: object
    tcp_offset_link6: object


@dataclass(frozen=True)
class JointTarget:
    joint_positions: tuple[float, ...]
    status: str
    position_error_m: float
    orientation_error_rad: float
    predicted_translation_m: tuple[float, ...]
    predicted_rotation_rad: tuple[float, ...]
    calibration_status: str = "uncalibrated_candidate"


@dataclass(frozen=True)
class DualArmCommand:
    joint_positions: dict[str, tuple[float, ...]]
    gripper_commands: dict[str, float]
    active_arm: str
    active_result: JointTarget
    calibration_status: str = "uncalibrated_candidate"


def _array(value: object, shape: tuple[int, ...]) -> np.ndarray:
    try:
        source = np.asarray(value)
        if source.dtype.kind not in "iuf":
            raise ControllerError("invalid_robot_state")
        result = source.astype(float)
    except (ValueError, TypeError, OverflowError):
        raise ControllerError("invalid_robot_state") from None
    if result.shape != shape or not np.isfinite(result).all():
        raise ControllerError("invalid_robot_state")
    return result.copy()


def _quat(value: object) -> np.ndarray:
    result = _array(value, (4,))
    norm = float(np.linalg.norm(result))
    if norm < 1e-12 or abs(norm - 1.0) > 1e-3:
        raise ControllerError("invalid_orientation")
    return result / norm


def _validated(state: ArmState):
    if not isinstance(state, ArmState):
        raise ControllerError("invalid_robot_state")
    q = _array(state.joint_positions, (6,))
    dq = _array(state.joint_velocities, (6,))
    p = _array(state.link6_position, (3,))
    quat = _quat(state.link6_quaternion_wxyz)
    jacobian = _array(state.link6_world_jacobian, (6, 6))
    lower = _array(state.joint_lower, (6,))
    upper = _array(state.joint_upper, (6,))
    velocity = _array(state.joint_velocity_limits, (6,))
    offset = _array(state.tcp_offset_link6, (3,))
    if np.any(lower >= upper) or np.any(velocity <= 0):
        raise ControllerError("invalid_joint_limits")
    if np.any(q < lower - 1e-9) or np.any(q > upper + 1e-9):
        raise ControllerError("measured_joint_outside_limits")
    return q, dq, p, quat, jacobian, lower, upper, velocity, offset


def _rotation(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw])


def orientation_error_world(current_wxyz: object, target_wxyz: object) -> np.ndarray:
    """Shortest world-frame axis-angle vector from current to target."""
    current, target = _quat(current_wxyz), _quat(target_wxyz)
    error = _multiply(target, current * np.array([1., -1., -1., -1.]))
    # Resolve antipodal quaternion ambiguity, including exactly pi rotations.
    if error[0] < 0 or (abs(error[0]) < 1e-14 and tuple(error[1:]) < tuple(-error[1:])):
        error = -error
    sin_half = float(np.linalg.norm(error[1:]))
    if sin_half < 1e-12:
        return np.zeros(3)
    angle = 2 * math.atan2(sin_half, max(0., float(error[0])))
    return error[1:] * (angle / sin_half)


def measured_tcp_pose(state: ArmState) -> tuple[np.ndarray, np.ndarray]:
    _, _, position, quat, _, _, _, _, offset = _validated(state)
    return position + _rotation(quat) @ offset, quat


def tcp_world_jacobian(state: ArmState) -> np.ndarray:
    _, _, _, quat, jacobian, _, _, _, offset = _validated(state)
    r = _rotation(quat) @ offset
    # v_tcp = v_link6 + omega cross r, so the sign is -skew(r).
    skew = np.array([[0., -r[2], r[1]], [r[2], 0., -r[0]], [-r[1], r[0], 0.]])
    jacobian[:3] -= skew @ jacobian[3:]
    return jacobian


def _cap(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum else vector * (maximum / norm)


def joint_target(state: ArmState, target_position: object, target_quaternion_wxyz: object,
                 *, profile: str = "transit", config: ControllerConfig | None = None) -> JointTarget:
    config = config or ControllerConfig()
    config.validate()
    if profile not in {"transit", "contact"}:
        raise ControllerError("invalid_motion_profile")
    q, _, _, _, _, lower, upper, velocity, _ = _validated(state)
    position, quat = measured_tcp_pose(state)
    target = _array(target_position, (3,))
    rotation_error = orientation_error_world(quat, target_quaternion_wxyz)
    position_error = target - position
    position_norm, rotation_norm = float(np.linalg.norm(position_error)), float(np.linalg.norm(rotation_error))
    if (position_norm <= POSITION_NUMERICAL_RESOLUTION_M
            and rotation_norm <= ORIENTATION_NUMERICAL_RESOLUTION_RAD):
        return JointTarget(tuple(q), "at_target", position_norm, rotation_norm, (0.,)*3, (0.,)*3)
    speed = config.transit_translation_speed_m_s if profile == "transit" else config.contact_translation_speed_m_s
    translation_cap = speed / config.control_hz
    angular_cap = config.angular_speed_rad_s / config.control_hz
    desired = np.concatenate([_cap(position_error, translation_cap), _cap(rotation_error, angular_cap)])
    jacobian = tcp_world_jacobian(state)
    try:
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + config.damping**2 * np.eye(6), desired)
    except np.linalg.LinAlgError:
        raise ControllerError("differential_ik_failed") from None
    if not np.isfinite(delta).all():
        raise ControllerError("differential_ik_failed")
    lead = np.minimum(config.joint_target_lead_rad, velocity / config.control_hz)
    delta = np.clip(delta, -lead, lead)
    delta = np.clip(q + delta, lower, upper) - q
    # Independent joint clipping can distort the Cartesian direction; recheck
    # both physical displacement caps after applying the live joint limits.
    predicted = jacobian @ delta
    scale = min(1., translation_cap / max(float(np.linalg.norm(predicted[:3])), 1e-30),
                angular_cap / max(float(np.linalg.norm(predicted[3:])), 1e-30))
    delta *= scale
    predicted = jacobian @ delta
    if float(np.dot(desired, predicted)) <= 1e-14:
        raise ControllerError("no_cartesian_progress")
    return JointTarget(tuple(q + delta), "command", position_norm, rotation_norm,
                       tuple(predicted[:3]), tuple(predicted[3:]))


class DualArmController:
    """One active arm; inactive measured-q hold and separate latched grippers."""
    ARMS = frozenset({"left", "right"})

    def __init__(self, *, gripper_commands: Mapping[str, float], config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()
        self.config.validate()
        if set(gripper_commands) != self.ARMS:
            raise ControllerError("both_initial_gripper_commands_required")
        self._gripper_commands = self._grippers(gripper_commands)
        self._active_arm: str | None = None
        self._inactive_hold: np.ndarray | None = None

    @staticmethod
    def _grippers(commands: Mapping[str, float]) -> dict[str, float]:
        result = {}
        for arm, value in commands.items():
            if arm not in DualArmController.ARMS or type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ControllerError("invalid_gripper_command")
            result[arm] = float(value)
        return result

    def _states(self, states: Mapping[str, ArmState]) -> None:
        if not isinstance(states, Mapping) or set(states) != self.ARMS:
            raise ControllerError("both_measured_arm_states_required")
        for state in states.values():
            _validated(state)

    def handoff(self, active_arm: str, states: Mapping[str, ArmState]) -> None:
        if active_arm not in self.ARMS:
            raise ControllerError("invalid_active_arm")
        self._states(states)
        inactive = "right" if active_arm == "left" else "left"
        self._inactive_hold = _validated(states[inactive])[0]
        self._active_arm = active_arm

    def command(self, states: Mapping[str, ArmState], target_position: object,
                target_quaternion_wxyz: object, *, profile: str = "transit",
                gripper_commands: Mapping[str, float] | None = None) -> DualArmCommand:
        self._states(states)
        if self._active_arm is None or self._inactive_hold is None:
            raise ControllerError("arm_handoff_required")
        updates = self._grippers(gripper_commands or {})
        active = self._active_arm
        inactive = "right" if active == "left" else "left"
        result = joint_target(states[active], target_position, target_quaternion_wxyz,
                              profile=profile, config=self.config)
        q, _, _, _, _, lower, upper, velocity, _ = _validated(states[inactive])
        if np.any(self._inactive_hold < lower) or np.any(self._inactive_hold > upper):
            raise ControllerError("inactive_hold_outside_live_limits")
        lead = np.minimum(self.config.joint_target_lead_rad, velocity / self.config.control_hz)
        hold_target = q + np.clip(self._inactive_hold - q, -lead, lead)
        self._gripper_commands.update(updates)
        return DualArmCommand({active: result.joint_positions, inactive: tuple(hold_target)},
                              dict(self._gripper_commands), active, result)
