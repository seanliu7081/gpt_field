"""Convert world-frame poses to the verified robosuite OSC_POSE interface.

The policy state is [position_xyz, orientation_rotvec_xyz, finger_qpos_2].
These constants are checked again against the live controller by the worker.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


TRANSLATION_SCALE_M = 0.05
ROTATION_SCALE_RAD = 0.5


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _rotvec_quaternion(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.r_[rotvec * (math.sin(angle / 2) / angle), math.cos(angle / 2)]


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lv, lw, rv, rw = left[:3], left[3], right[:3], right[3]
    return np.r_[lw * rv + rw * lv + np.cross(lv, rv), lw * rw - lv @ rv]


@dataclass
class ActionAdapter:
    """Keep gripper state and bound shortest world-frame rotation commands.

    Translation is rejected if a component exceeds the OSC physical limit;
    clipping it silently would invalidate the planner's proposed movement.
    ``hold`` repeats the prior open/close command, initially open (-1).
    """

    max_rotation_rad: float = 0.15
    _gripper: float = -1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_rotation_rad) or not 0 < self.max_rotation_rad <= ROTATION_SCALE_RAD:
            raise ValueError("max_rotation_rad must be in (0, 0.5]")
        self.reset()

    def reset(self) -> None:
        self._gripper = -1.0

    def adapt(
        self,
        delta_xyz: Sequence[float],
        orientation_target_xyzw: Sequence[float],
        gripper_command: str,
        state8: Sequence[float],
    ) -> tuple[float, ...]:
        delta = _vector(delta_xyz, 3, "delta_xyz")
        if np.any(np.abs(delta) > TRANSLATION_SCALE_M + 1e-12):
            raise ValueError("translation exceeds verified OSC limit of 0.05 m per axis")
        state = _vector(state8, 8, "state8")
        target = _vector(orientation_target_xyzw, 4, "orientation_target_xyzw")
        norm = float(np.linalg.norm(target))
        if norm < 1e-8:
            raise ValueError("target quaternion must have nonzero norm")
        target = target / norm
        current = _rotvec_quaternion(state[3:6])
        # Left multiplication: R_delta = R_target @ R_current.T (world frame).
        error = _quaternion_product(target, np.r_[-current[:3], current[3]])
        if error[3] < 0:
            error = -error
        sine = float(np.linalg.norm(error[:3]))
        if sine < 1e-12:
            rotation = np.zeros(3)
        else:
            angle = 2 * math.atan2(sine, float(error[3]))
            rotation = error[:3] / sine * min(angle, self.max_rotation_rad)
        if gripper_command == "open":
            self._gripper = -1.0
        elif gripper_command == "close":
            self._gripper = 1.0
        elif gripper_command != "hold":
            raise ValueError("gripper_command must be open, close, or hold")
        action = np.r_[delta / TRANSLATION_SCALE_M, rotation / ROTATION_SCALE_RAD, self._gripper]
        return tuple(float(v) for v in np.clip(action, -1.0, 1.0))
