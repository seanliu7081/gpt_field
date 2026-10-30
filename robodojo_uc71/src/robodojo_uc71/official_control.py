"""Action boundary for the pinned RoboDojo joint and official EEF backends.

The official backend produces world-frame link6 poses, not joint increments.
RoboDojo's own take_action performs cuRobo IK for both arms and the subsequent
joint interpolation. This module neither solves IK nor alters native failures,
collision checks, gripper mapping, action accounting or the simulator cadence.
Its inactive-arm hold is Cartesian; native IK may choose different joint angles.
"""
from __future__ import annotations

from collections.abc import Mapping

from .controller import ArmState, ControllerError, _array, _quat, _rotation, measured_tcp_pose
from .schema import ARMS, quaternion, vector


LEGACY_JOINT = "legacy_joint"
OFFICIAL_EE = "official_ee"
CONTROL_BACKENDS = frozenset({LEGACY_JOINT, OFFICIAL_EE})
JOINT_ACTION_KEYS = frozenset(f"{arm}_{kind}" for arm in ARMS
                            for kind in ("arm_joint_state", "ee_joint_state"))
EE_ACTION_KEYS = frozenset(f"{arm}_{kind}" for arm in ARMS
                         for kind in ("ee_pose", "ee_joint_state"))


def validate_control_backend(value: str) -> str:
    if not isinstance(value, str) or value not in CONTROL_BACKENDS:
        raise ValueError("unsupported_control_backend")
    return value


def validate_control_action(action: dict, control_backend: str) -> dict:
    backend = validate_control_backend(control_backend)
    keys = EE_ACTION_KEYS if backend == OFFICIAL_EE else JOINT_ACTION_KEYS
    if not isinstance(action, dict) or set(action) != keys:
        raise ValueError("exact_official_ee_keys_required" if backend == OFFICIAL_EE
                         else "exact_joint_mode_keys_required")
    result = {}
    try:
        for key in sorted(keys):
            if key.endswith("ee_joint_state"):
                values = vector(action[key], 1, "native gripper action")
                if values[0] not in (0., 1.):
                    raise ValueError("binary_gripper_required")
            elif backend == OFFICIAL_EE:
                values = vector(action[key], 7, "native link6 pose")
                values[3:] = quaternion(values[3:], "native link6 pose")
            else:
                values = vector(action[key], 6, "native joint action")
            result[key] = values
    except (ValueError, TypeError, OverflowError):
        raise ValueError("invalid_native_ee_values" if backend == OFFICIAL_EE
                         else "invalid_native_joint_values") from None
    return result


def validate_official_ee_action(action: dict) -> dict:
    return validate_control_action(action, OFFICIAL_EE)


def validate_ik_status(value: Mapping) -> dict[str, str]:
    """Only native solver status, never native result/debug/collision details."""
    if (not isinstance(value, Mapping) or set(value) != set(ARMS)
            or any(not isinstance(v, str) or v not in ("Success", "Fail") for v in value.values())):
        raise ValueError("invalid_native_ik_status")
    return {arm: value[arm] for arm in ARMS}


def world_link6_target(state: ArmState, tcp_position, tcp_quaternion_wxyz) -> list[float]:
    """Convert a world TCP target to native link6, using the target orientation.

    ArmState's established calibration is a fixed translation in link6 with
    identical TCP/link6 orientations. The current measured orientation must not
    be used to rotate the offset when the model requests a new orientation.
    """
    measured_tcp_pose(state)  # Validate the robot measurement/calibration contract.
    target = _array(tcp_position, (3,))
    orientation = _quat(tcp_quaternion_wxyz)
    offset = _array(state.tcp_offset_link6, (3,))
    link6 = target - _rotation(orientation) @ offset
    return [*link6.tolist(), *orientation.tolist()]


class OfficialEEController:
    """Pack final world link6 targets for untouched native EEF dispatch.

    No legacy DLS speed, joint-increment, damping or velocity settings are used.
    Fragment-level target envelopes remain the caller's responsibility.
    """

    def __init__(self, *, gripper_commands: Mapping[str, float]):
        if not isinstance(gripper_commands, Mapping) or set(gripper_commands) != set(ARMS):
            raise ControllerError("both_initial_gripper_commands_required")
        self._gripper_commands = self._grippers(gripper_commands)
        self._active_arm = None
        self._inactive_hold = None

    @staticmethod
    def _grippers(commands: Mapping[str, float]) -> dict[str, float]:
        if not isinstance(commands, Mapping):
            raise ControllerError("invalid_gripper_command")
        result = {}
        for arm, value in commands.items():
            if arm not in ARMS or type(value) not in (int, float) or value not in (0., 1.):
                raise ControllerError("invalid_gripper_command")
            result[arm] = float(value)
        return result

    @staticmethod
    def _states(states: Mapping[str, ArmState]) -> None:
        if not isinstance(states, Mapping) or set(states) != set(ARMS):
            raise ControllerError("both_measured_arm_states_required")
        for state in states.values():
            measured_tcp_pose(state)

    def handoff(self, active_arm: str, states: Mapping[str, ArmState]) -> None:
        if active_arm not in ARMS:
            raise ControllerError("invalid_active_arm")
        self._states(states)
        inactive = "right" if active_arm == "left" else "left"
        state = states[inactive]
        self._inactive_hold = [*_array(state.link6_position, (3,)).tolist(),
                               *_quat(state.link6_quaternion_wxyz).tolist()]
        self._active_arm = active_arm

    def command(self, states: Mapping[str, ArmState], target_position,
                target_quaternion_wxyz, *, profile: str = "transit",
                gripper_commands: Mapping[str, float] | None = None) -> dict:
        self._states(states)
        if self._active_arm is None or self._inactive_hold is None:
            raise ControllerError("arm_handoff_required")
        if profile not in ("transit", "contact"):
            raise ControllerError("invalid_motion_profile")
        updates = self._grippers({} if gripper_commands is None else gripper_commands)
        active = self._active_arm
        inactive = "right" if active == "left" else "left"
        action = {
            f"{active}_ee_pose": world_link6_target(states[active], target_position, target_quaternion_wxyz),
            f"{inactive}_ee_pose": list(self._inactive_hold),
        }
        latches = {**self._gripper_commands, **updates}
        action.update({f"{arm}_ee_joint_state": [latches[arm]] for arm in ARMS})
        clean = validate_official_ee_action(action)
        self._gripper_commands = latches
        return clean
