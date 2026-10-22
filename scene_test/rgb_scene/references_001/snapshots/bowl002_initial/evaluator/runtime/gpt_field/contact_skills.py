"""Geometry helpers and non-visual contact primitives for agent_SpeedUp_v1.

Every primitive is a closed loop on measured state8 only: anchored pose holds,
guarded tool-axis moves with stall / finger-touch / gap-jump events, settle
closes that record their gap curve, rotation in place, and free-space waypoint
reach on the fine compact field. Orientation is always held toward an explicit
measured target and never commanded with an exactly-zero rotation (robosuite
osc.py:259-260 keeps a stale goal then). Nothing here certifies a grasp or task
success; closes are classified against the online empty-close curve and holds
are verified by scripted motion. No simulator, object or camera truth is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from .action import (ActionAdapter, ROTATION_SCALE_RAD, TRANSLATION_SCALE_M, _quaternion_product,
                     _rotvec_quaternion)
from .fine_distance_planner import FINE_GRID, FineCompactField, FineDistancePlanner
from .observation import PolicyObservation, decode_image
from .planner import PlannerConfig
from .robot_tool_geometry import _rotation_matrix
from .speed_first_gripper import GripperSettleConfig, LocalGripperSettle
from .tracking import PathTracker
from .visual_field_v8 import OSC_GOAL_REFRESH_RAD, world_orientation_error_rotvec

CONTACT_SKILLS_VERSION = "agent-speedup-v1-contact-skills-2"
OPEN_GAP_MM_DEFAULT = 77.4
# Measured from recorded runs: constant-close empty curve from fully open
# (k = actions since the close began), linear between the recorded knots.
_EMPTY_CURVE_KNOTS_MM = ((0, 77.4), (5, 50.7), (10, 8.4), (15, 2.5), (20, 1.88), (25, 1.5), (30, 1.2))
DEFAULT_EMPTY_CURVE_MM = tuple(round(float(v), 4) for v in np.interp(
    np.arange(1, 31), [k for k, _ in _EMPTY_CURVE_KNOTS_MM], [g for _, g in _EMPTY_CURVE_KNOTS_MM]))
DEFAULT_INWARD_NOISE_MM = 0.02
GRIPPERS = ("open", "close")


# ---------------------------------------------------------------- utilities
def _finite(value, size: int, name: str) -> np.ndarray:
    try:
        result = np.array(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain {size} finite numbers") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _unit(value, size: int, name: str) -> np.ndarray:
    result = _finite(value, size, name)
    norm = float(np.linalg.norm(result))
    if norm < 1e-9:
        raise ValueError(f"{name} must have nonzero norm")
    return result / norm


def _real(value, name: str, *, minimum: float = 0.0, maximum: float | None = None, strict: bool = True) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if value < minimum or (strict and value == minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _count(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _gripper(value) -> str:
    if value not in GRIPPERS:
        raise ValueError("gripper must be explicitly 'open' or 'close'")
    return value


def _capped(vector: np.ndarray, cap: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if cap <= 0:
        return np.zeros(3)
    return vector * (cap / norm) if norm > cap else vector


def _json(value):
    """Plain JSON-safe copy; non-finite floats become None."""
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{type(value).__name__} is not JSON-safe")


def _xyz(vector) -> list[float]:
    return [float(v) for v in vector]


# ------------------------------------------------------------ geometry API
def rotation_matrix(rotvec) -> np.ndarray:
    """3x3 matrix whose columns are the right_hand body axes in world."""
    return _rotation_matrix(_finite(rotvec, 3, "rotvec"))


def matrix_to_quaternion_xyzw(R) -> np.ndarray:
    matrix = np.array(R, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("R must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0:
        s = 2.0 * math.sqrt(1.0 + trace)
        q = [(matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s,
             (matrix[1, 0] - matrix[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(matrix)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k]))
        if s < 1e-12:
            raise ValueError("R is not a rotation matrix")
        q = [0.0] * 4
        q[i] = 0.25 * s
        q[j] = (matrix[j, i] + matrix[i, j]) / s
        q[k] = (matrix[k, i] + matrix[i, k]) / s
        q[3] = (matrix[k, j] - matrix[j, k]) / s
    q = np.asarray(q, dtype=float)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("R is not a rotation matrix")
    q /= norm
    return q if q[3] >= 0 else -q


def axes_to_quaternion_xyzw(jaw, approach) -> np.ndarray:
    """right_hand frame: R[:,1]=jaw, R[:,2]=approach, R[:,0]=jaw x approach.

    The approach axis is kept exactly; the jaw is Gram-Schmidt orthogonalized
    against it.
    """
    z = _unit(approach, 3, "approach")
    y = _finite(jaw, 3, "jaw")
    y = y - float(y @ z) * z
    norm = float(np.linalg.norm(y))
    if norm < 1e-6:
        raise ValueError("jaw must not be parallel to the approach axis")
    y /= norm
    return matrix_to_quaternion_xyzw(np.column_stack([np.cross(y, z), y, z]))


def quaternion_angle_rad(q_a, q_b) -> float:
    """Geodesic angle between two orientations, invariant to quaternion sign."""
    a, b = _unit(q_a, 4, "q_a"), _unit(q_b, 4, "q_b")
    error = _quaternion_product(a, np.r_[-b[:3], b[3]])
    return float(2 * math.atan2(float(np.linalg.norm(error[:3])), abs(float(error[3]))))


def finger_gap_mm(state8) -> float:
    state = _finite(state8, 8, "state8")
    return float((state[6] - state[7]) * 1000)


def _inward_mm(before: "StepView", after: "StepView") -> tuple[float, float]:
    """Per-step inward finger motion: finger 1 when q1 falls, finger 2 when q2 rises."""
    return (float((before.state[6] - after.state[6]) * 1000), float((after.state[7] - before.state[7]) * 1000))


class HoldingActionAdapter(ActionAdapter):
    """ActionAdapter that never emits an exactly-zero rotation.

    robosuite OSC refreshes its orientation goal only when a rotation component
    is nonzero (osc.py:259-260); a 1 nrad x-rotation re-bases it on the measured
    pose (visual_field_v8.OSC_GOAL_REFRESH_RAD).
    """

    def adapt(self, delta_xyz, orientation_target_xyzw, gripper_command, state8):
        action = list(super().adapt(delta_xyz, orientation_target_xyzw, gripper_command, state8))
        if not any(action[3:6]):
            action[3] = OSC_GOAL_REFRESH_RAD / ROTATION_SCALE_RAD
        return tuple(action)


def _refreshed(action) -> tuple[float, ...]:
    try:
        values = [float(v) for v in action]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("primitive produced a malformed action") from exc
    if len(values) != 7 or not all(math.isfinite(v) and abs(v) <= 1.0 for v in values):
        raise RuntimeError("primitive produced an action outside the normalized OSC range")
    if not any(values[3:6]):
        values[3] = OSC_GOAL_REFRESH_RAD / ROTATION_SCALE_RAD
    return tuple(values)


# ------------------------------------------------------------- calibration
@dataclass(frozen=True)
class GripperCalibration:
    open_gap_mm: float
    empty_curve_mm: tuple[float, ...]
    inward_noise_mm: float
    source: str

    def __post_init__(self) -> None:
        curve = tuple(float(v) for v in self.empty_curve_mm)
        if not curve or not all(math.isfinite(v) for v in curve):
            raise ValueError("empty_curve_mm must be a nonempty sequence of finite gaps")
        object.__setattr__(self, "empty_curve_mm", curve)
        object.__setattr__(self, "open_gap_mm", _real(self.open_gap_mm, "open_gap_mm"))
        object.__setattr__(self, "inward_noise_mm", _real(self.inward_noise_mm, "inward_noise_mm", strict=False))
        if self.source not in ("online", "default_recorded"):
            raise ValueError("source must be 'online' or 'default_recorded'")

    @property
    def touch_threshold_mm(self) -> float:
        return max(0.10, 3 * self.inward_noise_mm)

    def empty_gap_at(self, k: int) -> float:
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
            raise ValueError("k must be an integer action count")
        return self.empty_curve_mm[min(max(int(k), 1), len(self.empty_curve_mm)) - 1]

    @property
    def empty_floor_mm(self) -> float:
        return min(self.empty_curve_mm)

    def to_dict(self) -> dict:
        return {"open_gap_mm": self.open_gap_mm, "empty_curve_mm": list(self.empty_curve_mm),
                "inward_noise_mm": self.inward_noise_mm, "source": self.source,
                "touch_threshold_mm": self.touch_threshold_mm, "empty_floor_mm": self.empty_floor_mm,
                "empty_k15_mm": self.empty_gap_at(15)}

    @classmethod
    def from_dict(cls, data: dict) -> "GripperCalibration":
        return cls(data["open_gap_mm"], tuple(data["empty_curve_mm"]), data["inward_noise_mm"], data["source"])

    @classmethod
    def default(cls) -> "GripperCalibration":
        return cls(OPEN_GAP_MM_DEFAULT, DEFAULT_EMPTY_CURVE_MM, DEFAULT_INWARD_NOISE_MM, "default_recorded")


# ---------------------------------------------------------------- step view
def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class StepView:
    """Immutable per-step snapshot; built once per observation by the policy."""

    __slots__ = ("_observation", "_state", "_R", "_position", "_gap_mm", "_quaternion", "_cache")

    def __init__(self, observation: PolicyObservation):
        state = _finite(getattr(observation, "state", None), 8, "observation.state")
        self._observation = observation
        self._state = _readonly(state)
        self._R = _readonly(_rotation_matrix(state[3:6]))
        self._position = _readonly(state[:3].copy())
        self._gap_mm = float((state[6] - state[7]) * 1000)
        self._quaternion = _readonly(_rotvec_quaternion(state[3:6].copy()))
        self._cache = {}

    observation = property(lambda self: self._observation)
    state = property(lambda self: self._state)
    R = property(lambda self: self._R)
    position = property(lambda self: self._position)
    gap_mm = property(lambda self: self._gap_mm)
    quaternion_xyzw = property(lambda self: self._quaternion)

    def _image(self, key: str, url: str) -> np.ndarray:
        if key not in self._cache:
            self._cache[key] = _readonly(np.ascontiguousarray(np.asarray(decode_image(url))[:, ::-1]))
        return self._cache[key]

    @property
    def wrist_rgb_upright(self) -> np.ndarray:
        return self._image("wrist", self._observation.wrist_image)

    @property
    def external_rgb_upright(self) -> np.ndarray:
        return self._image("external", self._observation.image)

    @property
    def wrist_gray_upright(self) -> np.ndarray:
        if "wrist_gray" not in self._cache:
            rgb = self.wrist_rgb_upright.astype(float)
            self._cache["wrist_gray"] = _readonly(rgb @ np.array([0.299, 0.587, 0.114]))
        return self._cache["wrist_gray"]

    def __repr__(self) -> str:
        return f"StepView(position={_xyz(self._position)}, gap_mm={self._gap_mm:.2f})"


def _view_state(view) -> np.ndarray:
    if not isinstance(view, StepView):
        raise TypeError("primitives take StepView snapshots")
    return view.state


# ------------------------------------------------------------ primitive base
class Primitive:
    """Base. Subclasses implement _next(view) and _observe(before, action, after).

    ``next_action`` must be followed by ``observe_transition`` with the exact
    issued action and the before-snapshot it was computed from. A primitive
    may already be done at construction (zero actions). Subclasses end with
    ``self._finish(status, reason, evidence)``, at construction or inside
    ``_observe``; ``_next`` always returns an action (the base re-adds the
    OSC goal refresh if a rotation is exactly zero).
    """

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("primitive name must be a nonempty string")
        self.name = name
        self.actions_executed = 0
        self._pending = None
        self._status = None
        self._reason = None
        self._evidence = {}

    @property
    def done(self) -> bool:
        return self._status is not None

    @property
    def status(self) -> str | None:
        return self._status

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @property
    def pointing_request(self) -> dict | None:
        return None

    def _finish(self, status: str, reason: str, evidence: dict) -> None:
        if status not in ("succeeded", "failed"):
            raise ValueError("status must be succeeded or failed")
        self._status, self._reason, self._evidence = status, reason, evidence

    def next_action(self, view: StepView) -> tuple[float, ...]:
        if self.done:
            raise RuntimeError(f"primitive {self.name} has finished; start the next primitive")
        if self._pending is not None:
            raise RuntimeError("observe_transition is required before the next primitive action")
        state = _view_state(view).copy()
        action = _refreshed(self._next(view))
        self._pending = (state, action)
        return action

    def observe_transition(self, before: StepView, action, after: StepView) -> dict:
        if self._pending is None:
            raise RuntimeError("no pending primitive action to observe")
        state, pending = self._pending
        try:
            same = len(action) == 7 and tuple(float(v) for v in action) == pending
        except (TypeError, ValueError):
            same = False
        if not same:
            raise ValueError("primitive feedback action must match the exact pending action")
        if not np.array_equal(_view_state(before), state):
            raise ValueError("primitive feedback before-state must match the state used for next_action")
        _view_state(after)
        self._pending = None
        self.actions_executed += 1
        details = self._observe(before, pending, after) or {}
        feedback = {"type": "primitive_step", "primitive": self.name, "step": self.actions_executed}
        feedback.update(details)
        feedback.update(done=self.done, status=self._status, reason=self._reason)
        return _json(feedback)

    def outcome(self) -> dict:
        if not self.done:
            raise RuntimeError(f"primitive {self.name} has not finished")
        return _json({"primitive": self.name, "status": self._status, "reason": self._reason,
                      "evidence": self._evidence, "actions": self.actions_executed})

    def _next(self, view: StepView):
        raise NotImplementedError

    def _observe(self, before: StepView, action: tuple[float, ...], after: StepView) -> dict:
        raise NotImplementedError


# ------------------------------------------------------- gripper primitives
class GripperSettlePrimitive(Primitive):
    """LocalGripperSettle behind the Primitive protocol, recording the gap curve.

    States are passed through unchanged, so the settle's exact-equality and
    identical-action checks still apply. Settling is not a grasp certificate.
    """

    def __init__(self, command: str, view: StepView, *, config: GripperSettleConfig | None = None,
                 name: str | None = None):
        super().__init__(name or f"gripper_{command}")
        self.settle = LocalGripperSettle(command, _view_state(view), config=config, adapter=HoldingActionAdapter())
        self.command = command
        self.start_gap_mm = view.gap_mm
        self.gap_curve_mm: list[float] = []
        self.settle_k: int | None = None
        self.last_diagnostics = self.settle.diagnostics()

    def _next(self, view):
        return self.settle.next_action(view.state)

    def _observe(self, before, action, after):
        diagnostics = self.settle.observe_transition(before.state, action, after.state)
        self.last_diagnostics = diagnostics
        self.gap_curve_mm.append(after.gap_mm)
        if self.settle.done:
            if self.settle.stop_reason == "measured_fingers_stable":
                self.settle_k = self.settle.actions_executed
            self._finish("succeeded", self.settle.stop_reason, self._evidence_now())
        return {"command": self.command, "gap_mm": after.gap_mm,
                "settle_actions": diagnostics["actions_executed"],
                "fingers_stable": diagnostics["fingers_stable"],
                "finger_motion_observed": diagnostics["finger_motion_observed"],
                "pose_error_m": diagnostics["pose_error_m"]}

    def _evidence_now(self) -> dict:
        d = self.last_diagnostics
        return {"command": self.command, "gap_curve_mm": list(self.gap_curve_mm),
                "start_gap_mm": self.start_gap_mm,
                "final_gap_mm": self.gap_curve_mm[-1] if self.gap_curve_mm else self.start_gap_mm,
                "settle_k": self.settle_k, "stop_reason": self.settle.stop_reason,
                "pose_error_m": d["pose_error_m"], "min_actions": d["min_actions"],
                "max_actions": d["max_actions"], "max_pose_correction_m": d["max_pose_correction_m"]}


class HoldPrimitive(Primitive):
    """Hold the measured start pose with a constant gripper command."""

    MAX_CORRECTION_M = 0.01

    def __init__(self, view: StepView, *, steps: int, gripper: str, name: str = "hold"):
        super().__init__(name)
        _view_state(view)
        self.steps = _count(steps, "steps")
        self.gripper = _gripper(gripper)
        self.anchor_xyz = view.position.copy()
        self.orientation_xyzw = view.quaternion_xyzw.copy()
        self.adapter = HoldingActionAdapter()
        self.inward_steps_mm: list[float] = []
        self.gaps_mm: list[float] = []

    @property
    def inward_noise_mm(self) -> float:
        return float(np.std(self.inward_steps_mm)) if self.inward_steps_mm else 0.0

    def _next(self, view):
        delta = _capped(self.anchor_xyz - view.position, self.MAX_CORRECTION_M)
        return self.adapter.adapt(delta, self.orientation_xyzw, self.gripper, view.state)

    def _observe(self, before, action, after):
        inward = _inward_mm(before, after)
        self.inward_steps_mm.extend(inward)
        self.gaps_mm.append(after.gap_mm)
        if self.actions_executed >= self.steps:
            self._finish("succeeded", "hold_complete", {
                "steps": self.actions_executed, "gripper": self.gripper,
                "inward_steps_mm": list(self.inward_steps_mm), "inward_noise_mm": self.inward_noise_mm,
                "max_inward_mm": max(self.inward_steps_mm), "mean_gap_mm": float(np.mean(self.gaps_mm)),
                "pose_error_m": float(np.linalg.norm(after.position - self.anchor_xyz))})
        return {"inward_mm": list(inward), "gap_mm": after.gap_mm}


class Calibrate(Primitive):
    """Episode-start free-air close + open + open hold; zero model calls.

    The online empty-close curve is used only if the close moved the fingers
    at least 20 mm; otherwise the recorded default curve is returned.
    """

    MIN_CLOSING_MOTION_MM = 20.0

    def __init__(self, view: StepView, *, name: str = "calibrate"):
        super().__init__(name)
        self.start_gap_mm = view.gap_mm
        self.close = GripperSettlePrimitive(
            "close", view, config=GripperSettleConfig(min_close_actions=30, max_actions=30), name=f"{name}_close")
        self.open: GripperSettlePrimitive | None = None
        self.hold: HoldPrimitive | None = None
        self.current: Primitive = self.close
        self.stage = "close"
        self.stage_outcomes: list[dict] = []
        self.calibration: GripperCalibration | None = None

    def _next(self, view):
        return self.current.next_action(view)

    def _observe(self, before, action, after):
        stage = self.stage
        feedback = self.current.observe_transition(before, action, after)
        if self.current.done:
            self.stage_outcomes.append(self.current.outcome())
            if self.current is self.close:
                self.open = GripperSettlePrimitive(
                    "open", after, config=GripperSettleConfig(min_open_actions=15, max_actions=30),
                    name=f"{self.name}_open")
                self.current, self.stage = self.open, "open"
            elif self.current is self.open:
                self.hold = HoldPrimitive(after, steps=10, gripper="open", name=f"{self.name}_hold")
                self.current, self.stage = self.hold, "hold"
            else:
                self._conclude()
        return {"stage": stage, "gap_mm": after.gap_mm, "stage_done": feedback["done"]}

    def _conclude(self) -> None:
        curve = list(self.close.gap_curve_mm)
        closing = self.start_gap_mm - min(curve) if curve else 0.0
        online = len(curve) >= 15 and closing >= self.MIN_CLOSING_MOTION_MM
        if online:
            self.calibration = GripperCalibration(float(np.mean(self.hold.gaps_mm)), tuple(curve),
                                                  self.hold.inward_noise_mm, "online")
        else:
            self.calibration = GripperCalibration.default()
        self._finish("succeeded", "calibrated_online" if online else "calibrated_default", {
            "calibration": self.calibration.to_dict(), "online": online, "closing_motion_mm": closing,
            "start_gap_mm": self.start_gap_mm, "measured_close_curve_mm": curve,
            "close_stop_reason": self.close.settle.stop_reason, "close_settle_k": self.close.settle_k,
            "open_stop_reason": self.open.settle.stop_reason, "open_settle_k": self.open.settle_k,
            "open_hold_inward_steps_mm": list(self.hold.inward_steps_mm),
            "open_hold_inward_noise_mm": self.hold.inward_noise_mm})


def classify_close(settle: GripperSettlePrimitive, calibration: GripperCalibration, *, margin_mm: float = 0.5) -> dict:
    """Object present if gap(k=15) exceeds the empty curve at the same k plus a margin.

    A shorter curve uses its last sample and the empty curve at that k (equal to
    the k=15 comparison whenever the close ran 15 or more actions).
    """
    if not isinstance(settle, GripperSettlePrimitive) or settle.command != "close":
        raise ValueError("classify_close needs a close GripperSettlePrimitive")
    if not isinstance(calibration, GripperCalibration):
        raise ValueError("calibration must be a GripperCalibration")
    margin = _real(margin_mm, "margin_mm", strict=False)
    if not settle.gap_curve_mm:
        raise ValueError("the close has not executed any action")
    k = min(15, len(settle.gap_curve_mm))
    gap, empty = settle.gap_curve_mm[k - 1], calibration.empty_gap_at(k)
    return _json({"object_present": bool(gap > empty + margin), "gap_k15_mm": gap, "empty_k15_mm": empty,
                  "margin_mm": margin, "k": k, "settle_k": settle.settle_k,
                  "final_gap_mm": settle.gap_curve_mm[-1], "calibration_source": calibration.source})


# ----------------------------------------------------------- motion primitives
class AnchoredMove(Primitive):
    """Straight guarded move along a frozen world axis with an anchored lateral hold.

    Events: travel reached, contact stall (realized/commanded ratio collapse
    against this move's own free ratio), finger touch (open fingers only), and
    single-step gap jumps. All thresholds come from this move or Calibrate.
    """

    DEFAULT_FREE_RATIO = 0.26
    FREE_RATIO_BOUNDS = (0.18, 0.40)
    # Real-LIBERO smokes 005/006: with a 3-step window at 0.3x the stall fired 7-8 steps (~9-10 mm) after
    # first contact. A 2-step window at 0.6x fires ~4 steps earlier; recorded free-motion, from-rest, lift
    # and drawer-pull ratios (>= 0.20 after step 2) stay above 0.6 x 0.26.
    STALL_FRACTION = 0.6
    STALL_WINDOW = 2
    STALL_MIN_STEPS = 3
    STALL_MAX_ROTATION_RAD = 0.05
    TRAVEL_TOLERANCE_M = 0.0005
    BUDGET_STEP_M = 0.0015
    MAX_ROTATION_RAD = 0.15

    def __init__(self, view: StepView, *, axis, travel_m: float, gripper: str, command_m: float = 0.010,
                 stop_on_contact: bool = False, touch_threshold_mm: float | None = None,
                 stop_on_gap_jump: bool = False, gap_jump_mm: float = 0.2,
                 lateral_cap_m: float = 0.003, orientation_xyzw=None, max_steps: int | None = None,
                 lateral_target=None, gap_rise_mm: float | None = None, name: str = "anchored_move"):
        super().__init__(name)
        _view_state(view)
        self.axis = _readonly(_unit(axis, 3, "axis"))
        # Held object meeting a support: a lip-held bowl slides up in the pinch instead of stalling the hand,
        # which opens the fingers (live 001: +0.17 then +1.9 mm at contact; free lowering drifted <= 0.09 mm).
        self.gap_rise_mm = None if gap_rise_mm is None else _real(gap_rise_mm, "gap_rise_mm")
        self.travel_m = _real(travel_m, "travel_m")
        self.gripper = _gripper(gripper)
        self.command_m = _real(command_m, "command_m", maximum=TRANSLATION_SCALE_M)
        self.lateral_cap_m = _real(lateral_cap_m, "lateral_cap_m", strict=False, maximum=TRANSLATION_SCALE_M)
        if self.command_m + self.lateral_cap_m > TRANSLATION_SCALE_M:
            raise ValueError("command_m + lateral_cap_m must fit the 0.05 m OSC translation limit")
        self.stop_on_contact, self.stop_on_gap_jump = bool(stop_on_contact), bool(stop_on_gap_jump)
        self.touch_threshold_mm = (None if touch_threshold_mm is None
                                   else _real(touch_threshold_mm, "touch_threshold_mm"))
        self.gap_jump_mm = _real(gap_jump_mm, "gap_jump_mm")
        self.orientation_xyzw = (view.quaternion_xyzw.copy() if orientation_xyzw is None
                                 else _unit(orientation_xyzw, 4, "orientation_xyzw"))
        self.max_steps = (math.ceil(self.travel_m / self.BUDGET_STEP_M) + 10 if max_steps is None
                          else _count(max_steps, "max_steps"))
        self.start_xyz = view.position.copy()
        self.anchor_xyz = (self.start_xyz.copy() if lateral_target is None
                           else _finite(lateral_target, 3, "lateral_target"))
        self.end_xyz = self.start_xyz.copy()
        self.gap_start_mm = self.gap_end_mm = view.gap_mm
        self.adapter = HoldingActionAdapter(max_rotation_rad=self.MAX_ROTATION_RAD)
        self.ratios: list[float] = []
        self.free_ratio = self.DEFAULT_FREE_RATIO
        self.free_ratio_measured = False
        self.travel_measured_m = 0.0
        self.commanded_travel_m = 0.0
        self.max_lateral_error_m = 0.0
        self.contact: str | None = None
        self.finger_deflected: str | None = None
        self.contact_xyz: list[float] | None = None

    def lateral_offset(self, position) -> np.ndarray:
        """Perpendicular offset of a position from the anchor line."""
        d = np.asarray(position, dtype=float) - self.anchor_xyz
        return d - float(d @ self.axis) * self.axis

    def _next(self, view):
        lateral = _capped(-self.lateral_offset(view.position), self.lateral_cap_m)
        return self.adapter.adapt(self.command_m * self.axis + lateral, self.orientation_xyzw,
                                  self.gripper, view.state)

    # hooks for subclasses -------------------------------------------------
    def _pre_event(self, before: StepView, after: StepView) -> str | None:
        return None

    def _resolve(self, event: str) -> tuple[str, str]:
        return ("failed" if event == "budget" else "succeeded"), event

    def _extra_evidence(self) -> dict:
        return {}

    # ---------------------------------------------------------------------
    def _observe(self, before, action, after):
        commanded = float((np.asarray(action[:3]) * TRANSLATION_SCALE_M) @ self.axis)
        realized = float((after.position - before.position) @ self.axis)
        ratio = realized / commanded if commanded > 1e-9 else 0.0
        self.ratios.append(ratio)
        self.commanded_travel_m += commanded
        if not self.free_ratio_measured and len(self.ratios) >= 5:
            low, high = self.FREE_RATIO_BOUNDS
            self.free_ratio = float(np.clip(np.median(self.ratios[2:5]), low, high))
            self.free_ratio_measured = True
        rotation = quaternion_angle_rad(before.quaternion_xyzw, after.quaternion_xyzw)
        inward = _inward_mm(before, after)
        gap_step = after.gap_mm - before.gap_mm
        self.travel_measured_m = float((after.position - self.start_xyz) @ self.axis)
        lateral_error = float(np.linalg.norm(self.lateral_offset(after.position)))
        self.max_lateral_error_m = max(self.max_lateral_error_m, lateral_error)
        self.end_xyz, self.gap_end_mm = after.position.copy(), after.gap_mm
        window = self.ratios[-self.STALL_WINDOW:]

        event = self._pre_event(before, after)
        if (event is None and self.stop_on_contact and self.gripper == "open"
                and self.touch_threshold_mm is not None and max(inward) > self.touch_threshold_mm):
            event, self.contact = "contact_touch", "touch"
            self.finger_deflected = "finger_1" if inward[0] >= inward[1] else "finger_2"
        if (event is None and self.stop_on_contact and len(self.ratios) >= self.STALL_MIN_STEPS
                and rotation < self.STALL_MAX_ROTATION_RAD
                and float(np.mean(window)) < self.STALL_FRACTION * self.free_ratio):
            event, self.contact = "contact_stall", "stall"
        if event is None and self.stop_on_gap_jump and abs(gap_step) >= self.gap_jump_mm:
            event = "gap_jump"
        if event is None and self.gap_rise_mm is not None and after.gap_mm - self.gap_start_mm >= self.gap_rise_mm:
            event = "gap_rise"
        if event is None and self.travel_measured_m >= self.travel_m - self.TRAVEL_TOLERANCE_M:
            event = "travel_reached"
        if event is None and self.actions_executed >= self.max_steps:
            event = "budget"
        if event is not None:
            if self.contact is not None:
                self.contact_xyz = _xyz(after.position)
            status, reason = self._resolve(event)
            self._finish(status, reason, self._evidence_now(event))
        return {"travel_m": self.travel_measured_m, "ratio": ratio, "rotation_step_rad": rotation,
                "inward_mm": list(inward), "gap_mm": after.gap_mm, "gap_step_mm": gap_step,
                "lateral_error_m": lateral_error, "free_ratio": self.free_ratio}

    def _evidence_now(self, event: str | None = None) -> dict:
        window = self.ratios[-self.STALL_WINDOW:]
        evidence = {"event": event, "axis": _xyz(self.axis), "travel_m": self.travel_measured_m,
                    "requested_travel_m": self.travel_m, "commanded_travel_m": self.commanded_travel_m,
                    "steps": self.actions_executed, "contact": self.contact,
                    "finger_deflected": self.finger_deflected, "contact_xyz": self.contact_xyz,
                    "free_ratio": self.free_ratio, "free_ratio_measured": self.free_ratio_measured,
                    "ratio_window": list(window), "max_lateral_error_m": self.max_lateral_error_m,
                    "start_xyz": _xyz(self.start_xyz), "end_xyz": _xyz(self.end_xyz),
                    "gap_start_mm": self.gap_start_mm, "gap_end_mm": self.gap_end_mm,
                    "gripper": self.gripper, "stop_on_contact": self.stop_on_contact,
                    "touch_threshold_mm": self.touch_threshold_mm, "max_steps": self.max_steps,
                    "gap_rise_mm": self.gap_rise_mm}
        evidence.update(self._extra_evidence())
        return evidence


class VerifyHold(AnchoredMove):
    """Scripted closed-gripper move; the gap must stay above the empty floor.

    A stall is a jam (hold_jam). The optional wrist monitor is secondary
    evidence only in v1 and never changes the status.
    """

    def __init__(self, view: StepView, *, axis, travel_m: float, calibration: GripperCalibration,
                 hold_margin_mm: float = 1.0, monitor=None, name: str = "verify_hold",
                 command_m: float = 0.010, lateral_cap_m: float = 0.003, max_steps: int | None = None):
        if not isinstance(calibration, GripperCalibration):
            raise ValueError("calibration must be a GripperCalibration")
        self.calibration = calibration
        self.hold_margin_mm = _real(hold_margin_mm, "hold_margin_mm", strict=False)
        self.hold_floor_mm = calibration.empty_floor_mm + self.hold_margin_mm
        self.monitor = monitor
        self.monitor_last: dict | None = None
        self.monitor_disagrees = False
        super().__init__(view, axis=axis, travel_m=travel_m, gripper="close", command_m=command_m,
                         stop_on_contact=True, lateral_cap_m=lateral_cap_m, max_steps=max_steps, name=name)
        self.min_gap_mm = view.gap_mm
        if view.gap_mm < self.hold_floor_mm:
            self._finish("failed", "hold_lost", self._evidence_now("hold_lost"))

    def _pre_event(self, before, after):
        if self.monitor is not None:
            result = self.monitor.observe(after)
            self.monitor_last = _json(result) if isinstance(result, dict) else None
            if (isinstance(result, dict) and result.get("available") is True and result.get("ok") is False):
                self.monitor_disagrees = True
        self.min_gap_mm = min(self.min_gap_mm, after.gap_mm)
        return "hold_lost" if after.gap_mm < self.hold_floor_mm else None

    def _resolve(self, event):
        return {"hold_lost": ("failed", "hold_lost"), "contact_stall": ("failed", "hold_jam"),
                "travel_reached": ("succeeded", "hold_pass")}.get(event, ("failed", event))

    def _extra_evidence(self):
        return {"hold_floor_mm": self.hold_floor_mm, "hold_margin_mm": self.hold_margin_mm,
                "min_gap_mm": self.min_gap_mm, "calibration_source": self.calibration.source,
                "monitor": self.monitor_last, "monitor_disagrees": self.monitor_disagrees}


class Orient(Primitive):
    """Rotate in place toward a target quaternion with an anchored xyz hold."""

    SETTLED_STEP_RAD = math.radians(0.1)

    MAX_CORRECTION_M = 0.01
    MAX_ROTATION_RAD = 0.15
    STALL_STEPS = 12
    STALL_MIN_DECREASE_RAD = math.radians(0.5)
    BUDGET_DEG_PER_STEP = 1.8
    BUDGET_EXTRA_STEPS = 15

    def __init__(self, view: StepView, *, target_xyzw, tolerance_rad: float = math.radians(3.0),
                 name: str = "orient", max_steps: int | None = None, gripper: str | None = None,
                 budget_tolerance_rad: float | None = None, max_pose_drift_m: float | None = None):
        super().__init__(name)
        _view_state(view)
        target = _unit(target_xyzw, 4, "target_xyzw")
        self.target_xyzw = target if target[3] >= 0 else -target
        self.tolerance_rad = _real(tolerance_rad, "tolerance_rad")
        # Optional looser acceptance once the budget is spent (live 001: four restores ended at 3.2-3.8 deg).
        self.budget_tolerance_rad = (self.tolerance_rad if budget_tolerance_rad is None
                                     else max(self.tolerance_rad, _real(budget_tolerance_rad, "budget_tolerance_rad")))
        # Optional fail-fast when the anchored position cannot be held: beyond the top-down envelope the
        # rotation drags the hand back and down (live 001: 37-121 mm; a feasible 90 deg turn drifted 5.5 mm).
        self.max_pose_drift_m = (None if max_pose_drift_m is None
                                 else _real(max_pose_drift_m, "max_pose_drift_m"))
        self.start_xyz = view.position.copy()
        self.gripper = ("open" if view.gap_mm >= 60 else "close") if gripper is None else _gripper(gripper)
        self.anchor_xyz = view.position.copy()
        self.adapter = HoldingActionAdapter(max_rotation_rad=self.MAX_ROTATION_RAD)
        self.initial_angle_rad = quaternion_angle_rad(view.quaternion_xyzw, self.target_xyzw)
        self.angles_rad = [self.initial_angle_rad]
        self.max_steps = (math.ceil(math.degrees(self.initial_angle_rad) / self.BUDGET_DEG_PER_STEP)
                          + self.BUDGET_EXTRA_STEPS if max_steps is None else _count(max_steps, "max_steps"))
        self.pose_error_m = 0.0
        self.end_xyz = self.start_xyz.copy()
        if self.initial_angle_rad <= self.tolerance_rad:
            self._finish("succeeded", "orientation_reached", self._evidence_now())

    def _next(self, view):
        delta = _capped(self.anchor_xyz - view.position, self.MAX_CORRECTION_M)
        return self.adapter.adapt(delta, self.target_xyzw, self.gripper, view.state)

    def _observe(self, before, action, after):
        angle = quaternion_angle_rad(after.quaternion_xyzw, self.target_xyzw)
        self.angles_rad.append(angle)
        self.pose_error_m = float(np.linalg.norm(after.position - self.anchor_xyz))
        self.end_xyz = after.position.copy()
        steps = self.actions_executed
        # The real arm keeps rotating for several steps after the command shrinks (recorded 0.2-0.5 deg/step);
        # succeed only once settled, or the next Align's probe frames are all rotated and discarded.
        settled = quaternion_angle_rad(before.quaternion_xyzw, after.quaternion_xyzw) <= self.SETTLED_STEP_RAD
        if angle <= self.tolerance_rad and (settled or steps >= self.max_steps):
            self._finish("succeeded", "orientation_reached", self._evidence_now())
        elif self.max_pose_drift_m is not None and self.pose_error_m > self.max_pose_drift_m:
            self._finish("failed", "pose_drift", self._evidence_now())
        elif steps >= self.max_steps and angle <= self.budget_tolerance_rad:
            self._finish("succeeded", "orientation_reached_loose", self._evidence_now())
        elif (steps >= self.STALL_STEPS
              and self.angles_rad[-1 - self.STALL_STEPS] - angle < self.STALL_MIN_DECREASE_RAD):
            self._finish("failed", "rotation_stall", self._evidence_now())
        elif steps >= self.max_steps:
            self._finish("failed", "budget", self._evidence_now())
        return {"angle_deg": math.degrees(angle), "pose_error_m": self.pose_error_m}

    def _evidence_now(self) -> dict:
        return {"initial_angle_deg": math.degrees(self.initial_angle_rad),
                "final_angle_deg": math.degrees(self.angles_rad[-1]), "steps": self.actions_executed,
                "target_xyzw": _xyz(self.target_xyzw), "tolerance_deg": math.degrees(self.tolerance_rad),
                "budget_steps": self.max_steps, "gripper": self.gripper, "pose_error_m": self.pose_error_m,
                "start_xyz": _xyz(self.start_xyz), "end_xyz": _xyz(self.end_xyz),
                "max_pose_drift_m": self.max_pose_drift_m}


class WaypointReach(Primitive):
    """Free-space reach to a world offset on the fine compact field (no repulsors).

    Same machinery as the speed_first Field block: analytic plan + PathTracker,
    replanning every 10 actions or at the path endpoint, and an explicit 5 mm /
    0.05 rad arrival check (the planner halves the tolerance, so its own
    goal_reached would not fire). Orientation is held at ``orientation_xyzw``
    (default: the measured start quaternion). Goals outside the +-0.08 m cube
    stop with zero motion (field_boundary_reached).

    Measured-only stop events (live run 001): reach_limit when the orientation
    error to the held target is >= 0.05 rad and has grown over the last 3 steps
    while the goal error shrank by less than 10% of the commanded motion (beyond
    the arm's top-down envelope the OSC gives up orientation and the hand pitches;
    outward motion realizes 1-2.5% of the command there, free motion ~25%; a move
    back from full stretch rotates briefly while progressing normally, real smoke
    011);
    uncontrolled_motion for a single-step displacement above 20 mm (free-space
    maximum recorded 12.7 mm); and, when holding (``hold_floor_mm``/``gap_ref_mm``),
    hold_lost below the empty-close floor and held_object_contact when the finger
    gap rises (a held object pressed against a support or obstacle opens the
    pinch). A budget ending is reported by its mechanism: orientation_drift,
    slow_arrival or blocked; leaving the planning grid is left_planning_grid.
    """

    GOAL_TOLERANCE_M = 0.005
    GOAL_ANGLE_RAD = 0.05
    BUDGET = 30
    REPLAN_EVERY = 10
    NO_PROGRESS_WINDOW = 5
    NO_PROGRESS_COMMANDED_M = 0.005
    NO_PROGRESS_MEASURED_M = 0.0005
    REACH_LIMIT_ANGLE_RAD = 0.05
    REACH_LIMIT_RISING_STEPS = 3
    REACH_LIMIT_PROGRESS_RATIO = 0.10
    UNCONTROLLED_STEP_M = 0.020
    GAP_RISE_MM = 1.0          # live 001: clean carries deviated <= 0.55 mm; contact episodes +1.3..+1.7 mm first
    GAP_STEP_RISE_MM = 0.4     # free-carry noise <= 0.149 mm/step
    SLOW_ARRIVAL_PROGRESS = 0.8
    ORIENTATION_DRIFT_ERROR_M = 0.015
    _INVALID_TRACKING = ("invalid_measured_start", "no_valid_reference")

    def __init__(self, view: StepView, offset_m, *, gripper: str, name: str = "reach",
                 max_steps: int | None = None, orientation_xyzw=None, hold_floor_mm: float | None = None,
                 gap_ref_mm: float | None = None, reach_limit: bool = True):
        super().__init__(name)
        _view_state(view)
        self.gripper = _gripper(gripper)
        self.offset_m = _finite(offset_m, 3, "offset_m")
        self.max_steps = self.BUDGET if max_steps is None else _count(max_steps, "max_steps")
        self.start_xyz = view.position.copy()
        self.goal_xyz = self.start_xyz + self.offset_m
        self.end_xyz = self.start_xyz.copy()
        self.orientation_xyzw = (view.quaternion_xyzw.copy() if orientation_xyzw is None
                                 else _unit(orientation_xyzw, 4, "orientation_xyzw"))
        self.hold_floor_mm = None if hold_floor_mm is None else _real(hold_floor_mm, "hold_floor_mm", strict=False)
        self.gap_ref_mm = None if gap_ref_mm is None else _real(gap_ref_mm, "gap_ref_mm", strict=False)
        self.reach_limit = bool(reach_limit)
        self.gap_start_mm = self.gap_end_mm = view.gap_mm
        # Baseline for held contact: the verified-grasp gap while the hold is intact, else this move's start
        # gap (a move that starts already wedged must still see a further rise).
        self.gap_baseline_mm = (self.gap_ref_mm if self.gap_ref_mm is not None
                                and abs(view.gap_mm - self.gap_ref_mm) <= self.GAP_RISE_MM else view.gap_mm)
        self.max_gap_rise_mm = 0.0
        self.max_step_m = 0.0
        self.adapter = HoldingActionAdapter()
        self.window: list[tuple[float, float]] = []
        self.since_plan = 0
        self.replans = 0
        self.plan = None
        self.plan_status = self.plan_reason = None
        self.last_tracking: dict | None = None
        self.start_angle_rad = float(np.linalg.norm(
            world_orientation_error_rotvec(self.orientation_xyzw, view.state[3:6])))
        self.final_angle_rad = self.start_angle_rad
        self.angles_rad = [self.start_angle_rad]
        self.goal_errors_m = [float(np.linalg.norm(self.offset_m))]
        self.commanded_m: list[float] = []
        self._prepared = None
        self._invalid_next = False
        self.field = self.planner = self.tracker = None
        if float(np.linalg.norm(self.offset_m)) <= self.GOAL_TOLERANCE_M:
            self._finish("succeeded", "goal_reached", self._evidence_now("goal_reached"))
            return
        if np.any(np.abs(self.offset_m) > FINE_GRID.half_width_m + 1e-12):
            self.plan_status, self.plan_reason = "needs_observation", "goal_outside_known_grid"
            self._finish("failed", "field_boundary_reached", self._evidence_now("field_boundary_reached"))
            return
        self.compact = {"stage": "agent_reach", "orientation_target_xyzw": _xyz(self.orientation_xyzw),
                        "gripper_command": self.gripper, "motion_mode": "move",
                        "goal_offset_m": _xyz(self.offset_m), "attraction_weights": [1.0, 1.0, 1.0],
                        "repulsors": [], "blocked_boxes": [], "unknown_boxes": []}
        self.field = FineCompactField(FINE_GRID, _xyz(self.start_xyz), self.compact)
        self.planner = FineDistancePlanner(PlannerConfig(horizon=10))
        self.tracker = PathTracker(endpoint_tolerance_m=.0025)
        self._replan(view.position)
        if not self.done:
            self._prepare(view)

    def _replan(self, position) -> None:
        self.plan = self.planner.plan_analytic(self.field, _xyz(position), goal_xyz=_xyz(self.goal_xyz),
                                               compact_prediction=self.compact,
                                               goal_tolerance_m=self.GOAL_TOLERANCE_M)
        self.replans += 1
        self.since_plan = 0
        self.plan_status, self.plan_reason = self.plan.status, self.plan.reason
        selected = self.plan.selected_index
        self.tracker.reset(None if selected is None else self.plan.candidates[selected].positions)
        if self.plan.status == "needs_observation":
            self._finish("failed", "field_boundary_reached", self._evidence_now("field_boundary_reached"))
        elif self.plan.status == "no_valid_candidates":
            self._finish("failed", "no_valid_plan", self._evidence_now("no_valid_plan"))

    def _command(self, position):
        config = self.planner.config
        return self.tracker.command(_xyz(position), self.field, config.workspace_min, config.workspace_max)

    def _prepare(self, view) -> None:
        delta, tracking = self._command(view.position)
        self.last_tracking = tracking
        if tracking["status"] in self._INVALID_TRACKING:
            # The measured hand left the +-0.08 m planning grid (overshoot or an uncommanded excursion);
            # the commanded direction itself was valid (live 001: all six cases).
            self._finish("failed", "left_planning_grid", self._evidence_now("left_planning_grid"))
        else:
            self._prepared = (view.state.copy(), delta)

    def _next(self, view):
        if self._prepared is not None and np.array_equal(self._prepared[0], view.state):
            delta = self._prepared[1]
        else:
            delta, tracking = self._command(view.position)
            self.last_tracking = tracking
            if tracking["status"] in self._INVALID_TRACKING:
                delta, self._invalid_next = (0.0, 0.0, 0.0), True
        self._prepared = None
        return self.adapter.adapt(delta, self.orientation_xyzw, self.gripper, view.state)

    @property
    def progress_fraction(self) -> float:
        """Along-goal progress of the measured displacement as a fraction of the commanded offset."""
        norm = float(np.linalg.norm(self.offset_m))
        if norm <= 1e-9:
            return 1.0
        return float((self.end_xyz - self.start_xyz) @ self.offset_m) / (norm * norm)

    def _reach_limited(self) -> bool:
        if not self.reach_limit or len(self.angles_rad) <= self.REACH_LIMIT_RISING_STEPS:
            return False
        steps = self.REACH_LIMIT_RISING_STEPS
        recent = self.angles_rad[-1 - steps:]
        rising = all(later > earlier for earlier, later in zip(recent, recent[1:]))
        progress = self.goal_errors_m[-1 - steps] - self.goal_errors_m[-1]
        stalled = progress < self.REACH_LIMIT_PROGRESS_RATIO * sum(self.commanded_m[-steps:])
        return rising and stalled and recent[-1] >= self.REACH_LIMIT_ANGLE_RAD

    def _budget_reason(self, error: float) -> str:
        if error <= self.ORIENTATION_DRIFT_ERROR_M and self.final_angle_rad > self.GOAL_ANGLE_RAD:
            return "orientation_drift"
        return "slow_arrival" if self.progress_fraction >= self.SLOW_ARRIVAL_PROGRESS else "blocked"

    def _observe(self, before, action, after):
        self.since_plan += 1
        commanded = float(np.linalg.norm(np.asarray(action[:3]) * TRANSLATION_SCALE_M))
        measured = float(np.linalg.norm(after.position - before.position))
        self.max_step_m = max(self.max_step_m, measured)
        self.window = (self.window + [(commanded, measured)])[-self.NO_PROGRESS_WINDOW:]
        self.end_xyz = after.position.copy()
        self.gap_end_mm = after.gap_mm
        error = float(np.linalg.norm(after.position - self.goal_xyz))
        self.final_angle_rad = float(np.linalg.norm(
            world_orientation_error_rotvec(self.orientation_xyzw, after.state[3:6])))
        self.angles_rad.append(self.final_angle_rad)
        self.goal_errors_m.append(error)
        self.commanded_m.append(commanded)
        rise = after.gap_mm - self.gap_baseline_mm
        self.max_gap_rise_mm = max(self.max_gap_rise_mm, rise)
        held = self.gap_ref_mm is not None
        if self._invalid_next:
            self._finish("failed", "left_planning_grid", self._evidence_now("left_planning_grid"))
        elif self.hold_floor_mm is not None and after.gap_mm < self.hold_floor_mm:
            self._finish("failed", "hold_lost", self._evidence_now("hold_lost"))
        elif held and (rise >= self.GAP_RISE_MM or after.gap_mm - before.gap_mm >= self.GAP_STEP_RISE_MM):
            self._finish("failed", "held_object_contact", self._evidence_now("held_object_contact"))
        elif error <= self.GOAL_TOLERANCE_M and self.final_angle_rad <= self.GOAL_ANGLE_RAD:
            self._finish("succeeded", "goal_reached", self._evidence_now("goal_reached"))
        elif measured > self.UNCONTROLLED_STEP_M:
            self._finish("failed", "uncontrolled_motion", self._evidence_now("uncontrolled_motion"))
        elif self._reach_limited():
            self._finish("failed", "reach_limit", self._evidence_now("reach_limit"))
        elif (len(self.window) == self.NO_PROGRESS_WINDOW
              and sum(c for c, _ in self.window) >= self.NO_PROGRESS_COMMANDED_M
              and sum(m for _, m in self.window) <= self.NO_PROGRESS_MEASURED_M):
            self._finish("failed", "no_progress", self._evidence_now("no_progress"))
        elif self.actions_executed >= self.max_steps:
            reason = self._budget_reason(error)
            self._finish("failed", reason, self._evidence_now(reason))
        else:
            if self.since_plan >= self.REPLAN_EVERY or self.tracker.endpoint_reached(_xyz(after.position)):
                self._replan(after.position)
            if not self.done:
                self._prepare(after)
        return {"goal_error_m": error, "commanded_norm_m": commanded, "measured_norm_m": measured,
                "orientation_error_rad": self.final_angle_rad, "gap_mm": after.gap_mm,
                "tracking_status": None if self.last_tracking is None else self.last_tracking.get("status")}

    def _evidence_now(self, reason: str) -> dict:
        remaining = self.goal_xyz - self.end_xyz
        horizontal = np.array([remaining[0], remaining[1], 0.0])
        blocked = (None if float(np.linalg.norm(horizontal)) < 1e-6
                   else _xyz(horizontal / float(np.linalg.norm(horizontal))))
        return {"commanded_offset_m": _xyz(self.offset_m),
                "measured_displacement_m": _xyz(self.end_xyz - self.start_xyz),
                "final_error_m": float(np.linalg.norm(self.end_xyz - self.goal_xyz)),
                "progress_fraction": self.progress_fraction,
                "steps": self.actions_executed, "reason": reason, "start_xyz": _xyz(self.start_xyz),
                "goal_xyz": _xyz(self.goal_xyz), "end_xyz": _xyz(self.end_xyz),
                "final_angle_rad": self.final_angle_rad, "final_angle_deg": math.degrees(self.final_angle_rad),
                "start_angle_deg": math.degrees(self.start_angle_rad),
                "blocked_direction_xy": blocked if reason in ("reach_limit", "blocked", "no_progress") else None,
                "max_step_m": self.max_step_m, "gap_start_mm": self.gap_start_mm, "gap_end_mm": self.gap_end_mm,
                "max_gap_rise_mm": self.max_gap_rise_mm if self.gap_ref_mm is not None else None,
                "gap_baseline_mm": self.gap_baseline_mm if self.gap_ref_mm is not None else None,
                "hold_floor_mm": self.hold_floor_mm,
                "plan_status": self.plan_status, "plan_reason": self.plan_reason, "replans": self.replans,
                "gripper": self.gripper, "max_steps": self.max_steps}
