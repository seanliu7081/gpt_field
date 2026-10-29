"""Astranomy Agentic short absolute targets on the existing LIBERO controller.

The official execute_eef_target interface repeats one complete absolute target
for 1--5 selected control steps, without interpolation or waiting for arrival.
Its requested gripper command acts on every step, including incomplete arm moves.
Reference: https://research.nvidia.com/labs/srl/projects/astranomy/

This adapter retains the LIBERO Panda physics and native 20 Hz OSC_POSE interface.
The paper uses 15 Hz DROID, per-step damped-least-squares IK and bounded joint
position increments. Here the local Cartesian error is bounded to 0.02 m / 0.1
rad and converted to OSC deltas; no 0.05-rad joint-increment guarantee is made.
Position is LIBERO's world grip_site, orientation is world right_hand, unlike
DROID's robot-root Robotiq mounting-flange pose. No scene geometry, grasp logic,
path planner, automatic retry or task-evaluation input is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Sequence

import numpy as np

from .action import ActionAdapter, ROTATION_SCALE_RAD, _quaternion_product, _rotvec_quaternion

MOTION_VERSION = "astranomy-agentic-short-target-libero-osc-v1"
CONTROL_HZ = 20
REFERENCE_CONTROL_HZ = 15
REFERENCE_URL = "https://research.nvidia.com/labs/srl/projects/astranomy/"
TARGET_FIELDS = ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
OSC_ORIENTATION_REFRESH_RAD = 1e-9
ADAPTATIONS = (
    "LIBERO Panda grip_site world position and right_hand world orientation replace the DROID Robotiq flange frame.",
    "Native 20 Hz OSC_POSE replaces the paper's 15 Hz per-step DLS joint-position controller.",
    "Cartesian command errors are capped; no per-joint 0.05-rad increment or scene collision guarantee is provided.",
)


class TargetValidationError(ValueError):
    """Malformed proposal; no action may be executed."""


class TargetLimitError(TargetValidationError):
    """Well-formed target outside the local envelope; return a tool rejection."""


def _vector(value, size: int, name: str) -> np.ndarray:
    try:
        raw = list(value)
    except (TypeError, ValueError) as exc:
        raise TargetValidationError(f"{name} must contain {size} finite real numbers") from exc
    if len(raw) != size or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                              or not math.isfinite(float(v)) for v in raw):
        raise TargetValidationError(f"{name} must contain {size} finite real numbers")
    return np.asarray(raw, dtype=float)


def _norm(value) -> float:
    return math.hypot(*(float(v) for v in value))


def _normalized_quaternion(wxyz) -> np.ndarray:
    q = _vector(wxyz, 4, "quaternion wxyz")
    # Rescaling first handles finite but unusually large components without
    # overflow, while retaining rejection of a zero quaternion.
    scale = float(np.max(np.abs(q)))
    if scale == 0.:
        raise TargetValidationError("quaternion must have nonzero norm")
    q = q/scale
    q = q/_norm(q)
    return q


def rotation_error_rad(target_wxyz, measured_rotvec) -> np.ndarray:
    """Shortest world-frame rotation from measured right_hand to the target."""
    q = _normalized_quaternion(target_wxyz)
    target = np.r_[q[1:], q[0]]
    current = _rotvec_quaternion(_vector(measured_rotvec, 3, "measured rotation"))
    error = _quaternion_product(target, np.r_[-current[:3], current[3]])
    if error[3] < 0.:
        error = -error
    sine = _norm(error[:3])
    if sine < 1e-12:
        return np.zeros(3)
    return error[:3]*(2*math.atan2(sine, float(error[3]))/sine)


def measured_pose(state8: Sequence[float]) -> list[float]:
    """Return [x,y,z,qw,qx,qy,qz,closure] from robot proprioception only.

    Target gripper is binary (0=open,1=closed). Measured closure is continuous,
    derived from the two finger joints and their known 0.08 m total opening.
    Neither closure nor finger spacing is interpreted as a successful grasp.
    """
    state = _vector(state8, 8, "state8")
    q = _rotvec_quaternion(state[3:6])
    closure = float(np.clip(1.-(state[6]-state[7])/.08, 0., 1.))
    return [float(v) for v in np.r_[state[:3], q[3], q[:3], closure]]


@dataclass(frozen=True)
class ShortTargetConfig:
    max_target_translation_m: float = .05
    max_target_rotation_rad: float = .35
    max_step_translation_m: float = .02
    max_step_rotation_rad: float = .1

    def __post_init__(self):
        ceilings = {"max_target_translation_m": .05, "max_target_rotation_rad": .35,
                    "max_step_translation_m": .02, "max_step_rotation_rad": .1}
        for key, ceiling in ceilings.items():
            value = getattr(self, key)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or not 0. < value <= ceiling):
                raise ValueError(f"{key} must be positive and no greater than {ceiling}")

    def to_dict(self):
        return {**asdict(self), "control_hz": CONTROL_HZ, "min_steps": 1, "max_steps": 5,
                "target_order": list(TARGET_FIELDS), "gripper_open": 0, "gripper_closed": 1,
                "execution": "repeat_absolute_target_with_measured_error_bounded_OSC",
                "reference_control_hz": REFERENCE_CONTROL_HZ, "adaptations": list(ADAPTATIONS)}


def _target_values(target, steps: int) -> tuple[float, ...]:
    if type(steps) is not int or not 1 <= steps <= 5:
        raise TargetValidationError("steps must be an integer from 1 through 5")
    try:
        raw = list(target)
    except (TypeError, ValueError) as exc:
        raise TargetValidationError("target must be the complete 8D absolute pose") from exc
    if len(raw) != 8:
        raise TargetValidationError("target must be the complete 8D absolute pose")
    position = _vector(raw[:3], 3, "target position")
    q = _normalized_quaternion(raw[3:7])
    grip = raw[7]
    # The Agentic report calls grippers booleans; accept bool or numeric 0/1,
    # normalize to an integer, and reject intermediate apertures explicitly.
    if not isinstance(grip, (Real, bool, np.bool_)) or grip not in (0, 1):
        raise TargetValidationError("gripper must be binary: 0=open, 1=closed")
    return tuple(float(v) for v in np.r_[position, q, int(grip)])


def validate_target(state8, target, steps: int, *, config=None) -> tuple[float, ...]:
    """Normalize a complete pose and reject requests outside the short horizon.

    ``TargetValidationError`` indicates malformed fields. Its more specific
    ``TargetLimitError`` means a valid proposal exceeded the geometric envelope
    and should become a zero-action tool result before another model decision.
    """
    config = config or ShortTargetConfig()
    state = _vector(state8, 8, "state8")
    normalized = _target_values(target, steps)
    with np.errstate(over="ignore", invalid="ignore"):
        translation = np.asarray(normalized[:3])-state[:3]
    distance = _norm(translation)
    angle = _norm(rotation_error_rad(normalized[3:7], state[3:6]))
    if distance > config.max_target_translation_m+1e-12:
        raise TargetLimitError(f"target translation {distance:.6g} m exceeds "
                              f"{config.max_target_translation_m:.6g} m from the latest measured pose")
    if angle > config.max_target_rotation_rad+1e-12:
        raise TargetLimitError(f"target rotation {angle:.6g} rad exceeds "
                              f"{config.max_target_rotation_rad:.6g} rad from the latest measured pose")
    return normalized


def rejected_feedback(state8, target, steps: int, detail: str) -> dict:
    """Robot-only zero-action result for a well-formed, geometrically rejected target.

    Malformed schemas still raise TargetValidationError. No state is advanced,
    and the requested gripper command has not been applied.
    """
    state = _vector(state8, 8, "state8")
    normalized = _target_values(target, steps)
    if not isinstance(detail, str) or not detail.strip():
        raise ValueError("rejected target requires a reason")
    distance = _norm(np.asarray(normalized[:3])-state[:3])
    if not math.isfinite(distance):
        raise TargetValidationError("requested displacement cannot be represented as a finite metric")
    return {"status": "rejected", "requested_target": list(normalized),
            "measured_pose": measured_pose(state), "position_error_m": distance,
            "orientation_error_rad": _norm(rotation_error_rad(normalized[3:7], state[3:6])),
            "requested_steps": steps, "executed_steps": 0, "simulation_time_s": 0.,
            "control_hz": CONTROL_HZ, "gripper_command": int(normalized[7]),
            "detail": detail + "; no arm or gripper action was executed"}


class ShortTargetExecutor:
    """One 1--5-step target hold with explicit one-action acknowledgements.

    No constructor action, IK preflight, interpolation, arrival gate, trailing
    settling step or retry occurs. Exactly the chosen count is issued unless
    the host independently interrupts the segment.
    """

    def __init__(self, state8, target, steps: int, *, config=None):
        self.config = config or ShortTargetConfig()
        self.target = validate_target(state8, target, steps, config=self.config)
        self.requested_steps = steps
        self.executed_steps = 0
        self.expected_state = _vector(state8, 8, "state8").copy()
        self.pending = None
        self.done = False
        self.status = "running"
        self.detail = "fixed target segment pending; arrival is not a termination condition"
        self.adapter = ActionAdapter(max_rotation_rad=self.config.max_step_rotation_rad)

    def next_action(self, state8):
        if self.done:
            raise RuntimeError("short target segment is already finished")
        if self.pending is not None:
            raise RuntimeError("observe_transition must acknowledge the previous action")
        state = _vector(state8, 8, "state8")
        if not np.allclose(state, self.expected_state, atol=1e-9, rtol=0.):
            raise ValueError("action state differs from the latest acknowledged measured state")
        delta = np.asarray(self.target[:3])-state[:3]
        distance = _norm(delta)
        if distance > self.config.max_step_translation_m:
            delta *= self.config.max_step_translation_m/distance
        wxyz = self.target[3:7]
        action = list(self.adapter.adapt(delta, [*wxyz[1:], wxyz[0]],
                                        "close" if self.target[7] == 1. else "open", state))
        # LIBERO's unchanged OSC otherwise retains its previous orientation goal
        # on an exactly-zero rotation; this numerical refresh prevents that.
        if not any(action[3:6]):
            action[3] = OSC_ORIENTATION_REFRESH_RAD/ROTATION_SCALE_RAD
        issued = tuple(float(v) for v in action)
        self.pending = (state.copy(), issued)
        return issued

    def observe_transition(self, before_state8, action, after_state8):
        if self.pending is None:
            raise RuntimeError("no issued action awaits acknowledgement")
        before = _vector(before_state8, 8, "before_state8")
        after = _vector(after_state8, 8, "after_state8")
        command = _vector(action, 7, "action")
        expected, issued = self.pending
        if not np.allclose(before, expected, atol=1e-9, rtol=0.) or not np.allclose(
                command, issued, atol=1e-12, rtol=0.):
            raise ValueError("acknowledgement does not match the issued state and action")
        self.pending = None
        self.expected_state = after.copy()
        self.executed_steps += 1
        if self.executed_steps == self.requested_steps:
            self.done = True
            self.status = "segment_complete"
            self.detail = "selected control steps executed; inspect measured pose errors; " \
                          "segment completion does not establish arrival, grasp or task success"
        return self.feedback(after)

    def feedback(self, state8=None):
        state = self.expected_state if state8 is None else _vector(state8, 8, "state8")
        return {"status": self.status, "requested_target": list(self.target),
                "measured_pose": measured_pose(state),
                "position_error_m": _norm(np.asarray(self.target[:3])-state[:3]),
                "orientation_error_rad": _norm(rotation_error_rad(self.target[3:7], state[3:6])),
                "requested_steps": self.requested_steps, "executed_steps": self.executed_steps,
                "simulation_time_s": self.executed_steps/CONTROL_HZ, "control_hz": CONTROL_HZ,
                "gripper_command": int(self.target[7]), "detail": self.detail}

    def interrupt(self, detail="host interrupted the short target segment"):
        self.pending = None
        self.done = True
        self.status = "interrupted"
        self.detail = str(detail)
