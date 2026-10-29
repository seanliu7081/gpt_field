"""Explicit local gripper operation using only measured robot state.

This is an opt-in operation, never an interpretation of a Direct action chunk.
Finger settling ends the operation and requests another visual decision; it is
not evidence that an object was grasped, released, or successfully placed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .action import ActionAdapter, TRANSLATION_SCALE_M, _rotvec_quaternion


def _state(value: Sequence[float]) -> np.ndarray:
    try:
        if len(value) != 8 or any(
            isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, float, np.number))
            or isinstance(v, (complex, np.complexfloating))
            for v in value
        ):
            raise ValueError
        state = np.asarray(value, dtype=float)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("state8 must contain eight finite real robot values") from exc
    return state.copy()


@dataclass(frozen=True)
class GripperSettleConfig:
    min_close_actions: int = 15
    min_open_actions: int = 10
    max_actions: int = 30
    stable_feedback_steps: int = 4
    finger_stability_tolerance_m: float = 0.0002
    finger_motion_threshold_m: float = 0.0005
    max_pose_correction_m: float = 0.01

    def __post_init__(self) -> None:
        for name in ("min_close_actions", "min_open_actions", "max_actions", "stable_feedback_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max(self.min_close_actions, self.min_open_actions) > self.max_actions:
            raise ValueError("minimum operation actions cannot exceed max_actions")
        if self.stable_feedback_steps > self.max_actions:
            raise ValueError("stable_feedback_steps cannot exceed max_actions")
        for name in ("finger_stability_tolerance_m", "finger_motion_threshold_m", "max_pose_correction_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.finger_motion_threshold_m <= self.finger_stability_tolerance_m:
            raise ValueError("finger motion threshold must exceed stability tolerance")
        if self.max_pose_correction_m > TRANSLATION_SCALE_M:
            raise ValueError("pose correction must fit the verified OSC translation scale")


class LocalGripperSettle:
    """Hold the initial measured pose and repeat an explicit open/close intent.

    ``next_action(state8)`` must be followed by ``observe_transition`` with the
    matching action and measured after-state. No time passing or repeated reads
    count as executed physics steps. The start pose is fixed for the operation,
    so drift is corrected rather than redefining the desired pose each step.
    """

    def __init__(
        self,
        command: str,
        initial_state8: Sequence[float],
        *,
        config: GripperSettleConfig | None = None,
        adapter: ActionAdapter | None = None,
    ) -> None:
        if command not in ("open", "close"):
            raise ValueError("local gripper operation requires explicit open or close")
        self.command = command
        self.config = config or GripperSettleConfig()
        self.adapter = adapter if adapter is not None else ActionAdapter()
        self._initial_state = _state(initial_state8)
        self._last_state = self._initial_state.copy()
        self._orientation_target = _rotvec_quaternion(self._initial_state[3:6])
        self._finger_window = deque(
            [self._initial_state[6:8].copy()],
            maxlen=self.config.stable_feedback_steps + 1,
        )
        self._pending = None
        self.actions_executed = 0
        self.stop_reason = None
        self._max_finger_excursion_m = 0.0

    @property
    def min_actions(self) -> int:
        return (self.config.min_close_actions if self.command == "close"
                else self.config.min_open_actions)

    @property
    def done(self) -> bool:
        return self.stop_reason is not None

    @property
    def finger_motion_observed(self) -> bool:
        return self._max_finger_excursion_m >= self.config.finger_motion_threshold_m

    def _finger_range(self) -> float:
        return float(np.max(np.ptp(np.asarray(self._finger_window), axis=0)))

    @property
    def fingers_stable(self) -> bool:
        return (len(self._finger_window) == self.config.stable_feedback_steps + 1
                and self._finger_range() <= self.config.finger_stability_tolerance_m)

    def next_action(self, state8: Sequence[float]) -> tuple[float, ...]:
        if self.done:
            raise RuntimeError("gripper operation ended; a new model decision is required")
        if self._pending is not None:
            raise RuntimeError("observe_transition is required before the next gripper action")
        current = _state(state8)
        if not np.array_equal(current, self._last_state):
            raise ValueError("gripper action state must equal the preceding measured feedback")
        delta = self._initial_state[:3] - current[:3]
        distance = float(np.linalg.norm(delta))
        if distance > self.config.max_pose_correction_m:
            # Bound the vector norm without changing the correction direction.
            delta *= self.config.max_pose_correction_m / distance
        action = self.adapter.adapt(delta, self._orientation_target, self.command, current)
        self._pending = (current, action)
        return action

    def observe_transition(
        self,
        before_state8: Sequence[float],
        action: Sequence[float],
        after_state8: Sequence[float],
    ) -> dict:
        if self._pending is None:
            raise RuntimeError("no pending gripper action to observe")
        before = _state(before_state8)
        after = _state(after_state8)
        pending_before, pending_action = self._pending
        if not np.array_equal(before, pending_before):
            raise ValueError("gripper feedback before-state must match the issued action")
        if tuple(action) != pending_action:
            raise ValueError("gripper feedback action must match the issued action")
        self._pending = None
        self.actions_executed += 1
        self._last_state = after
        self._finger_window.append(after[6:8].copy())
        self._max_finger_excursion_m = max(
            self._max_finger_excursion_m,
            float(np.max(np.abs(after[6:8] - self._initial_state[6:8]))),
        )
        if self.actions_executed >= self.min_actions and self.finger_motion_observed and self.fingers_stable:
            self.stop_reason = "measured_fingers_stable"
        elif self.actions_executed >= self.config.max_actions:
            self.stop_reason = "action_budget_exhausted"
        return self.diagnostics()

    def diagnostics(self) -> dict:
        return {
            "execution_kind": "local_gripper_settle",
            "command": self.command,
            "actions_executed": self.actions_executed,
            "min_actions": self.min_actions,
            "max_actions": self.config.max_actions,
            "done": self.done,
            "stop_reason": self.stop_reason,
            "finger_motion_observed": self.finger_motion_observed,
            "max_finger_excursion_m": self._max_finger_excursion_m,
            "finger_motion_threshold_m": self.config.finger_motion_threshold_m,
            "fingers_stable": self.fingers_stable,
            "stable_feedback_steps": self.config.stable_feedback_steps,
            "observed_window_steps": len(self._finger_window) - 1,
            "finger_window_range_m": self._finger_range(),
            "finger_stability_tolerance_m": self.config.finger_stability_tolerance_m,
            "pose_target_xyz_m": self._initial_state[:3].tolist(),
            "orientation_target_xyzw": self._orientation_target.tolist(),
            "pose_error_m": float(np.linalg.norm(self._initial_state[:3] - self._last_state[:3])),
            "max_pose_correction_m": self.config.max_pose_correction_m,
            "grasp_success_certified": False,
            "task_success_certified": False,
        }
