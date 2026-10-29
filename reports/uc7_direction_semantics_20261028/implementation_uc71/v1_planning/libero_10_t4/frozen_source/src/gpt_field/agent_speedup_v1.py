"""agent_SpeedUp_v1 policy: model-supervised macro skills with sensor-grounded contact.

The hosted model chooses macro skills and discrete parameters, and selects
numbered marks when a skill asks for a target point. Each macro skill runs a
fixed sequence of local closed-loop primitives (contact_skills, wrist_servo)
at the simulator rate with no choice calls in between. The model never writes
contact quaternions, millimetre offsets or descent depths, and the gripper
closes only after the strategy's approach and depth steps report their sensor
events. See agent_SpeedUp_v1.md for the design and constant labels.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable

import numpy as np

from .agent_speedup_v1_client import APPROACHES, SIDES, SKILLS, STRATEGIES, TASK_TARGETS
from .astra import SchemaError
from .contact_skills import (AnchoredMove, Calibrate, GripperCalibration, GripperSettlePrimitive,
                             HoldPrimitive, Orient, StepView, VerifyHold, WaypointReach,
                             axes_to_quaternion_xyzw, classify_close, quaternion_angle_rad)
from .external_axes import EXTERNAL_AXES_VERSION, fit_external_axes, hand_image_motion, image_shift_for
from .mark_prompting import (DEFAULT_MARGIN_CELLS, FINGERTIP_EDGE, WIDE_MARGIN_CELLS, coarse_overlay, encode_png_data_url,
                             fine_overlay, fine_point_pixel, gravity_quarter_turns, rotate_quarter, unrotate_pixel)
from .policy import Decision
from .speed_first_gripper import GripperSettleConfig
from .visual_field_v8 import measured_current_pose_hold_action
from .wrist_servo import GRIP_AXIS_OFFSET_M, Align, WristPatchMonitor, bar_point_check

POLICY_VERSION = "agent-speedup-v1-policy-2"
LEDGER_LIMIT = 12
OUTCOME_HISTORY_LIMIT = 50
MAX_DECISIONS_PER_ACT = 2
REJECTION_STORM_LIMIT = 3        # after this many consecutive rejections allow one decision per act
MIN_MOVE_OFFSET_M = 0.005        # WaypointReach treats smaller offsets as already reached
MAX_TRANSITIONS_PER_ACT = 24
# Design defaults (agent_SpeedUp_v1.md labels them "design default"; tune in Phase 3b).
RIM_CLEARANCE_M = 0.004          # fingertip clearance after the floor stall
BAR_CLEARANCE_M = 0.005          # robot-only retreat from the cabinet-face stall (R-3 alternative: 0.010)
BAR_UNDER_FINGER_RETREAT_M = 0.010
LADDER_SEGMENT_M = 0.020
LADDER_SETTLE_STEPS = 3
FIRST_SETTLE_MAX_STEPS = 14     # quiet-based settle before the first descent segment (<= 0.3 mm, <= 0.03 deg)
FINGERTIP_BAND_ROW_FRACTION = 410.0 / 512.0   # upright wrist rows below this show the robot's own fingertips
POST_ORIENT_HOLD_STEPS = 4      # residual rotation after Orient (recorded 0.2-0.5 deg/step) settles
APPROACH_BOUND_M = 0.300          # real-LIBERO smoke: a high hover exhausted 0.15 m; the table/support stops it
RETRY_SHIFT_M = 0.010
LIFT_TEST_M = 0.025
PULL_TEST_M = 0.020
RETREAT_M = 0.030
LARGE_TURN_RETREAT_M = 0.050
LOWER_BOUND_M = 0.150            # minimum lowering; with a measured pick height the bound is relative (below)
# Live run 001 fixes (agent_SpeedUp_v1.md §17.1); all measured-only (state8 / the robot's own motion).
PLACE_BELOW_PICK_M = 0.030       # lower to at most 3 cm below the grasp's floor-touch height
PLACE_LOWER_MAX_M = 0.300
PLACE_FAST_ABOVE_PICK_M = 0.050  # fast lowering (20 mm command) ends 5 cm above the pick height
PLACE_FAST_COMMAND_M = 0.020
PLACE_GAP_RISE_MM = 0.30         # held object meets a support: the pinch opens (free lowering drift <= 0.09 mm)
SUPPORT_EVENTS = ("contact_stall", "contact_touch", "gap_rise")
HELD_CONTACT_BACKOFF_M = 0.010
MAX_HELD_REORIENT_RAD = math.radians(15.0)   # never swing a held object back by more than this mid-move
RESTORE_INWARD_M = 0.060         # restore orientation only after moving back toward the robot ...
RESTORE_INWARD_MAX_RAD = math.radians(60.0)  # ... when the tilt is a reach-limit pitch (not a frontal hand)
RESTORE_CLEAR_ABOVE_PICK_M = 0.060
RESTORE_DEFAULT_LIFT_M = 0.030
RESTORE_BUDGET_TOLERANCE_RAD = math.radians(5.0)
ORIENT_MAX_DRIFT_M = 0.020
PROBE_MOVES = ((0.040, 0.0, 0.0), (-0.040, 0.040, 0.0), (0.0, -0.040, 0.0))   # external-axes probes, net zero
REACH_LIMIT_MEMORY = 4
REACH_LIMIT_RADIUS_M = 0.030
REACH_LIMIT_PUSH_M = 0.010
REACH_LIMIT_MAX_LIFT_M = 0.010
LOOP_WARNING_FAILURES = 5
PROCEDURAL_REASONS = ("orientation_restored",)   # status failed, but not counted as a failure
RECENT_DECISIONS = 3
SAME_STALL_HEIGHT_M = 0.003      # drawer live 002: 13 stalls at grip z 1.092-1.104 on the open drawer front
# One outward retry after blocked_above_rim, bounded by the robot's fingers: the wall must stay between them, so the
# shift stays below half the ~78 mm opening minus the pad and the pointing spread (live 002 rim points 49-57 mm from
# the axis). A second 3.5 cm shift was dropped: it was validated only with oracle rim points, and it puts live points
# past the ~86 mm straddle limit. Outward because the side is chosen with free space outside the container (prompt).
BLOCKED_OUTWARD_SHIFTS_M = (0.025,)
GRASP_LOOP_FAILURES = 3          # per-target: drawer live 001 reached 12 failed grasps between successful reaches
# ROBOT mount fact (ruling R-5, granted by the user 2026-10-17; verified in both LIBERO scenes): Panda base and
# shoulder (joint-2 anchor). With a frontal (horizontal-approach) hand the wrist point grip - 0.204 R2 tracks the
# arm's stretch (corr 0.9986); full stretch is ~0.804 m and drawer live 001 was trapped at the elbow singularity
# from 0.788 on (unrecoverable). Hovers, grasp starts and outward goals beyond 0.77 m are refused; a frontal move
# drifting past 0.785 m stops. Top-down moves keep the reach_limit handling (the mapping differs there).
ROBOT_BASE_XYZ = (-0.66, 0.0, 0.912)
SHOULDER_XYZ = (-0.66, 0.0, 1.245)
WRIST_FROM_GRIP_M = 0.204
FRONTAL_STRETCH_LIMIT_M = 0.77
FRONTAL_STRETCH_STOP_M = 0.785
FRONTAL_MAX_VERTICAL = 0.5       # |approach . z| below this: a frontal (horizontal-approach) hand
ORIENT_RETRY_LIFT_M = 0.040
HANDLE_RELEASE_LIFT_M = 0.050
HOLD_DEGRADED_MM = 3.0
SUMMARY_DECIMALS = 4
ORIENT_REASONS = {"budget": "orient_budget", "rotation_stall": "orient_stall", "pose_drift": "pose_drift_reach_limit"}
LARGE_TURN_RAD = math.radians(95.0)
SETPOINT_ROW_FRACTION = 380.0 / 512.0
OPEN_TOLERANCE_MM = 2.0
HOLD_MARGIN_MM = 1.0
WORLD_AXES = {"world_-y": (0., -1., 0.), "world_+y": (0., 1., 0.),
              "world_-x": (-1., 0., 0.), "world_+x": (1., 0., 0.)}
START_JAW, START_APPROACH = (0., -1., 0.), (0., 0., -1.)
GRASP_SETTLE = GripperSettleConfig(max_pose_correction_m=0.03)
OPEN_SETTLE = GripperSettleConfig(min_open_actions=15)
RULINGS_ASSUMED = {"R-1_online_image_jacobian": True, "R-2_wrist_mount_facts": True,
                   "R-3_evaluator_tuned_constants": False, "R-4_episode_continuation_signal": False,
                   # live run 001 fix; --external-axes off disables it
                   "R-1x_external_image_axes_from_own_motion": True,
                   # granted by the user 2026-10-17: robot base / shoulder position (mount fact) for the reach guard
                   "R-5_robot_base_position": True}
# "measured": external-image -> world direction map fitted from the robot's own probe motion (R-1 extended
# to the external camera, assumed); "off": no external mapping is given to the model.
EXTERNAL_AXES_MODES = ("measured", "off")


def jsonable(value: Any) -> Any:
    """Plain JSON-safe copy: numpy to Python, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _vec(values, digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in values]


def _rounded(value: Any, digits: int = SUMMARY_DECIMALS) -> Any:
    """Round every float in a JSON-safe structure (the ledger repeats in every request)."""
    if isinstance(value, dict):
        return {key: _rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item, digits) for item in value]
    if isinstance(value, float):
        return round(value, digits)
    return value


def world_up_in_wrist_image(R) -> str:
    """Upright wrist-image direction of world up (image right = R1, up = R0), or out_of_image_plane."""
    right, up = float(np.asarray(R)[2, 1]), float(np.asarray(R)[2, 0])
    if math.hypot(right, up) < 0.7:
        return "out_of_image_plane"
    if abs(right) >= abs(up):
        return "right" if right > 0 else "left"
    return "up" if up > 0 else "down"


def wrist_to_shoulder_m(position, approach_axis) -> float:
    """Distance of the wrist point (grip - 0.204 m along the approach axis) from the arm's shoulder (R-5)."""
    wrist = np.asarray(position, float) - WRIST_FROM_GRIP_M * np.asarray(approach_axis, float)
    return float(np.linalg.norm(wrist - np.asarray(SHOULDER_XYZ)))


def robot_base_direction(position) -> list[float]:
    """Horizontal unit vector from the grip point toward the robot base (R-5)."""
    d = np.asarray(ROBOT_BASE_XYZ, float) - np.asarray(position, float)
    d[2] = 0.0
    return _vec(d / max(float(np.linalg.norm(d)), 1e-9), 3)


def tilt_from_straight_down_deg(R) -> float:
    """Angle between the approach axis (fingers' pointing direction) and world -z."""
    return math.degrees(math.acos(float(np.clip(-np.asarray(R)[2, 2], -1.0, 1.0))))


def _target_name(target: str | None) -> str:
    return (target or "target").replace("_", " ")


def rim_target_phrase(target: str, side: str) -> str:
    name = _target_name(target)
    return (f"the point on the visible top edge (rim) of the {name} at its far "
            f"{'left' if side == 'wrist_left' else 'right'} side, on the horizontal line through the {name}'s centre")


def bar_target_phrase(target: str) -> str:
    # Any visible point on the right bar between its posts works: the along-bar move then brings the grip axis
    # there. The bar centre is often hidden behind the fingertips (real drawer smoke 008).
    name = _target_name(target)
    return (f"a clearly visible point on the {name}'s bar (the horizontal handle bar; not the handle of another "
            "drawer) between its two mounting posts, preferably mid-way; never a part hidden behind the robot's "
            "fingertips; 0 if that bar is not visible")


def rim_pointing_context(target: str, side: str, target_description: str | None) -> dict:
    context = {"strategy": "rim_pinch", "target": target, "side": side}
    if target_description and target in ("black_bowl", "bowl"):
        context["target_appearance"] = target_description
    return context


def bar_pointing_context(target: str, approach_name: str) -> dict:
    return {"strategy": "frontal_bar_pinch", "target": target, "approach": approach_name}


class _Context:
    """Mutable state shared between the policy and its active macro generator."""

    def __init__(self, policy: "AgentSpeedUpV1Policy", skill: str, params: dict):
        self.policy, self.skill, self.params = policy, skill, params
        self.view: StepView | None = None
        self.target_phrase: str | None = None
        self.pointing_context: dict = {}
        self.last_pointing: dict | None = None
        self.macro_steps = 0


class AgentSpeedUpV1Policy:
    """Event-driven supervisor loop over local macro skills (runner contract: reset/act/observe_transition)."""

    grid = None

    def __init__(self, client, *, local_retries: int = 2, external_axes: str = "measured"):
        if type(local_retries) is not int or not 0 <= local_retries <= 5:
            raise ValueError("local_retries must be an integer in [0, 5]")
        if external_axes not in EXTERNAL_AXES_MODES:
            raise ValueError(f"external_axes must be one of {EXTERNAL_AXES_MODES}")
        self.client = client
        self.local_retries = local_retries
        self.external_axes = external_axes
        self.task_key = getattr(client, "task_key", "bowl_plate")
        if self.task_key not in TASK_TARGETS:
            raise ValueError("unsupported task for agent_SpeedUp_v1")
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.step_index = 0
        self.calibration: GripperCalibration | None = None
        self.ledger: list[dict] = []
        self.outcomes: list[dict] = []
        self._macro = None
        self._context: _Context | None = None
        self._primitive = None
        self._pending = None
        self._expected_observation = None
        self._cached_view: tuple[Any, StepView] | None = None
        self._holding = False
        self._gripper_intent = "open"
        self._pulled_m = 0.0
        self._held_strategy = None
        self._outcome_counter = 0
        self._consecutive_rejections = 0
        self._start_quaternion = None
        self._last_outcome: dict | None = None
        self._grasp_quaternion = None        # measured hand orientation at the verified grasp (held moves keep it)
        self._pick_floor_z: float | None = None   # grip z at the rim pinch's floor-touch stall
        self._hold_gap_ref_mm: float | None = None
        self._reach_limits: list[dict] = []
        self._target_history: dict | None = None
        self._external_axes_fit: dict | None = None
        self._failed_in_a_row = 0
        self._recent_failures: list[str] = []
        self._pointing_rounds: dict[int, list[str]] = {}
        self._recent_decisions: list[dict] = []
        self._grasp_failures_by_target: dict[str, int] = {}
        self._contact_heights: dict[str, list[float]] = {}   # grip z where rim-pinch descents stopped, per target
        self._held_away_world: list[float] | None = None   # out from under a known overhang (after blocked_above_rim)
        self._counts = {"decisions": 0, "rejected_decisions": 0, "pointing_coarse": 0, "pointing_fine": 0,
                        "grasp_with_calls": 0, "grasp_attempts": 0, "local_retries_used": 0, "hold_actions": 0}
        self._macro_counts: dict[str, dict[str, int]] = {}
        self._primitive_actions: dict[str, int] = {}
        self._start_macro("calibrate", {}, self._calibrate())

    @property
    def holding(self) -> bool:
        return self._holding

    def agent_stats(self) -> dict:
        return jsonable({
            "policy_version": POLICY_VERSION, "step": getattr(self, "step_index", 0),
            "calibration": self.calibration.to_dict() if getattr(self, "calibration", None) else None,
            "counts": dict(getattr(self, "_counts", {})), "macros": copy.deepcopy(getattr(self, "_macro_counts", {})),
            "primitive_actions": dict(getattr(self, "_primitive_actions", {})),
            "holding": getattr(self, "_holding", False), "gripper_intent": getattr(self, "_gripper_intent", "open"),
            "active_macro": self._context.skill if getattr(self, "_context", None) else None,
            "active_primitive": getattr(getattr(self, "_primitive", None), "name", None),
            "ledger": list(getattr(self, "ledger", [])), "last_outcome": getattr(self, "_last_outcome", None),
            "local_retries": getattr(self, "local_retries", None), "rulings_assumed": dict(RULINGS_ASSUMED),
            "external_axes_mode": getattr(self, "external_axes", None),
            "external_axes": getattr(self, "_external_axes_fit", None),
            "reach_limits": list(getattr(self, "_reach_limits", [])),
            "target_history": getattr(self, "_target_history", None),
            "pick_floor_z": getattr(self, "_pick_floor_z", None),
        })

    # ------------------------------------------------------------------ runner contract
    def act(self, observation) -> Decision:
        if self._pending is not None:
            raise RuntimeError("observe_transition must acknowledge the pending action before act")
        if self._expected_observation is not None and observation != self._expected_observation:
            raise ValueError("act requires the observation from the preceding transition")
        view = self._view(observation)
        if self._start_quaternion is None:
            self._start_quaternion = view.quaternion_xyzw.copy()
        calls: list[dict] = []
        pointing: list[dict] = []
        refreshed, routing, boundary_outcome = False, None, None
        outcomes_in_act, routings_in_act = [], []
        decisions = 0
        decision_limit = 1 if self._consecutive_rejections >= REJECTION_STORM_LIMIT else MAX_DECISIONS_PER_ACT
        action = None
        for _ in range(MAX_TRANSITIONS_PER_ACT):
            if self._macro is None:
                if decisions >= decision_limit:
                    break
                decisions += 1
                decision = self._decide(view, calls)
                self._recent_decisions = (self._recent_decisions + [{
                    "step": self.step_index, "skill": decision.get("skill"),
                    "params": {k: v for k, v in (decision.get("params") or {}).items() if v is not None},
                    "situation": str(decision.get("situation") or "")[:100], "outcome": None}])[-RECENT_DECISIONS:]
                refreshed = True
                routing = {"skill": decision["skill"], "params": decision["params"], "situation": decision["situation"]}
                routings_in_act.append(routing)
                rejection = self._begin_decision(decision, view)
                if rejection is not None:
                    boundary_outcome = self._finish(rejection)
                    outcomes_in_act.append(boundary_outcome)
                continue
            if self._primitive is None or self._primitive.done:
                ended = self._advance(view)
                if ended is not None:
                    boundary_outcome = ended
                    outcomes_in_act.append(ended)
                continue
            request = self._primitive.pointing_request
            if request is not None:
                if len(pointing) >= 2:
                    break
                pointing.append(self._point(view, request, calls))
                continue
            action = tuple(float(v) for v in self._primitive.next_action(view))
            break
        primitive = self._primitive if action is not None else None
        if action is None:
            action = tuple(float(v) for v in measured_current_pose_hold_action())
            self._counts["hold_actions"] += 1
        api_latency = sum(call["latency_s"] for call in calls) if calls else None
        last_call = calls[-1] if calls else {}
        details = jsonable({
            "step": self.step_index, "controller": "agent_skill", "refreshed": refreshed, "routing": routing,
            "api_latency_s": api_latency, "api_calls_in_act": [call["kind"] for call in calls],
            "api_request_key": last_call.get("request_key"), "api_request_log": last_call.get("request_log"),
            "execution_kind": f"agent_skill:{primitive.name if primitive is not None else 'hold'}",
            "macro": self._context.skill if self._context is not None else None,
            "primitive": primitive.name if primitive is not None else "hold",
            "macro_params": self._context.params if self._context is not None else None,
            "update_steps": 1, "history_stats": self.client.history_stats(), "feedback": None,
            "plan_refreshed": False, "outcome": boundary_outcome, "outcomes": outcomes_in_act,
            "routings": routings_in_act, "pointing": pointing,
            "macro_step": self._context.macro_steps if self._context is not None else None,
            "holding": self._holding, "finger_gap_mm": view.gap_mm,
        })
        decision_obj = Decision(action, details)
        self._pending = (observation, decision_obj, view, primitive)
        self.step_index += 1
        return decision_obj

    def observe_transition(self, before, decision, after) -> dict:
        if self._pending is None:
            raise RuntimeError("observe_transition requires a pending action")
        pending_obs, pending_decision, before_view, primitive = self._pending
        if before != pending_obs or decision is not pending_decision:
            raise ValueError("observe_transition requires the exact pending decision and observation")
        action = np.asarray(decision.action, float)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1 + 1e-9):
            raise ValueError("executed action must contain seven finite values in [-1, 1]")
        after_view = self._view(after)
        primitive_feedback = None
        if primitive is not None:
            primitive_feedback = primitive.observe_transition(before_view, decision.action, after_view)
            self._primitive_actions[primitive.name] = self._primitive_actions.get(primitive.name, 0) + 1
            if self._context is not None:
                self._context.macro_steps += 1
        if action[6] > 0:
            self._gripper_intent = "close"
        elif action[6] < 0:
            self._gripper_intent = "open"
        if self._holding and self.calibration is not None and after_view.gap_mm < self.calibration.empty_floor_mm + HOLD_MARGIN_MM:
            self._holding = False
            self._note_lost(after_view, "finger_gap_below_empty_floor")
        delta = after_view.position - before_view.position
        feedback = jsonable({
            "type": "agent_transition", "step": int(decision.diagnostics.get("step", self.step_index - 1)),
            "before_state": list(before_view.state), "after_state": list(after_view.state),
            "normalized_action": action.tolist(),
            "measured_translation_delta_m": delta.tolist(),
            "measured_translation_norm_m": float(np.linalg.norm(delta)),
            "commanded_translation_norm_m": float(np.linalg.norm(action[:3] * 0.05)),
            "translation_tracking_error_m": float(np.linalg.norm(delta - action[:3] * 0.05)),
            "gripper_joint_delta": (after_view.state[6:8] - before_view.state[6:8]).tolist(),
            "finger_gap_mm": after_view.gap_mm, "gripper_command": float(action[6]),
            "rolling_progress": {}, "macro": self._context.skill if self._context is not None else None,
            "primitive": primitive.name if primitive is not None else "hold",
            "primitive_feedback": primitive_feedback,
        })
        self._expected_observation = after
        self._pending = None
        return feedback

    # ------------------------------------------------------------------ helpers
    def _view(self, observation) -> StepView:
        if self._cached_view is not None and self._cached_view[0] is observation:
            return self._cached_view[1]
        view = StepView(observation)
        self._cached_view = (observation, view)
        return view

    def _timed_call(self, kind: str, method: Callable, request: dict, calls: list) -> dict:
        started = time.monotonic()
        try:
            return method(request)
        finally:
            calls.append({"kind": kind, "latency_s": time.monotonic() - started,
                          "request_key": getattr(self.client, "last_request_key", None),
                          "request_log": getattr(self.client, "last_log_path", None)})

    def _gripper_state(self, view: StepView) -> str:
        calibration = self.calibration or GripperCalibration.default()
        if view.gap_mm >= calibration.open_gap_mm - 3.0:
            return "open"
        if view.gap_mm <= calibration.empty_floor_mm + HOLD_MARGIN_MM:
            return "closed_empty"
        return "closed_on_object" if self._holding else "partially_closed"

    def _state_summary(self, view: StepView) -> dict:
        R = view.R
        holding = self._holding
        rim_held = holding and self._held_strategy == "rim_pinch" and self._pick_floor_z is not None
        hold_quality = None
        if holding and self._hold_gap_ref_mm is not None:
            hold_quality = {"finger_gap_at_grasp_mm": self._hold_gap_ref_mm,
                            "degraded": bool(view.gap_mm > self._hold_gap_ref_mm + HOLD_DEGRADED_MM)}
        last = self._compact(self._last_outcome) if self._last_outcome else None
        return _rounded(jsonable({
            "task": view.observation.prompt, "task_key": self.task_key,
            "target_description": getattr(self.client, "target_description", None),
            "step": self.step_index,
            "end_effector_position_m": _vec(view.position, 4),
            "tool_axes_world": {"jaw_axis": _vec(R[:, 1], 3), "approach_axis": _vec(R[:, 2], 3),
                                "wrist_image_right": _vec(R[:, 1], 3), "wrist_image_up": _vec(R[:, 0], 3)},
            "approach_tilt_from_straight_down_deg": round(tilt_from_straight_down_deg(R), 1),
            "world_up_in_wrist_image": world_up_in_wrist_image(R),
            "robot_base_direction_world": robot_base_direction(view.position),
            "frontal_arm_reach": ({"wrist_to_shoulder_m": round(wrist_to_shoulder_m(view.position, R[:, 2]), 3),
                                   "limit_m": FRONTAL_STRETCH_LIMIT_M} if self._frontal(view) else None),
            "external_image_axes_world": self._external_axes_summary(),
            "finger_gap_mm": round(view.gap_mm, 2), "gripper_state": self._gripper_state(view),
            "holding_object": holding, "held_with_strategy": self._held_strategy if holding else None,
            "pulled_since_grasp_m": round(self._pulled_m, 4) if holding and self._held_strategy == "frontal_bar_pinch" else None,
            "held_height_above_pick_m": round(float(view.position[2]) - self._pick_floor_z, 3) if rim_held else None,
            "hold_quality": hold_quality,
            "target_history": self._target_history,
            "reach_limits": [limit for limit in self._reach_limits if limit.get("hand") == self._hand_kind(view)][-3:]
                            or None,
            "loop_warning": self._loop_warning(),
            "consecutive_rejected_decisions": self._consecutive_rejections,
            "last_outcome": last,
            "ledger": self.ledger[-LEDGER_LIMIT:],
            "recent_decisions": self._recent_decisions[:-1] if self._recent_decisions and
                                self._recent_decisions[-1]["outcome"] is None else self._recent_decisions,
            "counts": {"decisions": self._counts["decisions"], "grasp_with_calls": self._counts["grasp_with_calls"],
                       "grasp_attempts_reaching_align": self._counts["grasp_attempts"],
                       "local_retries_used": self._counts["local_retries_used"],
                       "failed_in_a_row": self._failed_in_a_row},
            "limits": {"max_waypoint_offset_m_per_axis": 0.075, "pull_travel_range_m": [0.02, 0.16],
                       "place_lowering": ("down to 3 cm below the pick height" if self._pick_floor_z is not None
                                          else f"at most {LOWER_BOUND_M:g} m")},
        }))

    @staticmethod
    def _frontal(view: StepView) -> bool:
        return abs(float(view.R[2, 2])) < FRONTAL_MAX_VERTICAL

    def _hand_kind(self, view: StepView) -> str:
        return "turned" if self._frontal(view) else "top_down"

    def _stretch_detail(self, distance: float, position) -> str:
        return (f"the turned hand's wrist would be {distance:.3f} m from the arm's shoulder (limit "
                f"{FRONTAL_STRETCH_LIMIT_M:g} m): the arm would lock at full stretch; move toward the robot base "
                f"(world {robot_base_direction(position)}) and approach from the robot side")

    def _external_axes_summary(self) -> dict | None:
        fit = self._external_axes_fit
        if self.external_axes != "measured" or fit is None:
            return None
        if not fit.get("available"):
            return {"available": False, "reason": fit.get("reason")}
        return {"image_right": fit["image_right_world"], "image_up": fit["image_up_world"],
                "source": "measured from the hand's own probe moves at episode start"}

    def _loop_warning(self) -> str | None:
        repeated = {t: n for t, n in self._grasp_failures_by_target.items() if n >= GRASP_LOOP_FAILURES}
        if repeated:
            target, count = max(repeated.items(), key=lambda item: item[1])
            stalls = self._contact_heights.get(target, [])
            if stalls:   # rim pinch: re-hovering does not change the aligned rim point
                heights = ", ".join(f"{z:.3f}" for z in stalls[-4:])
                return (f"grasp_with on {target} failed {count} times since its last success; the executor aligns "
                        f"on the same rim point from any hover, and the descents stopped at grip z {heights}: "
                        "change the side, or clear what the hand body hits (it is about 20 cm long along the "
                        "jaw), rather than the hover")
            return (f"grasp_with on {target} failed {count} times since its last success; the hovers and points "
                    "tried are not working: change the approach (e.g. a fresh hover level with and in front of "
                    "the target, judged in the external image), not small shifts of the same hover")
        if self._failed_in_a_row < LOOP_WARNING_FAILURES:
            return None
        recent = ", ".join(self._recent_failures[-LOOP_WARNING_FAILURES:])
        return (f"the last {self._failed_in_a_row} skills failed or were rejected ({recent}); repeating similar "
                "choices is not working, so change what you do and use the reasons and evidence")

    def _note_lost(self, view: StepView, cause: str) -> None:
        history = self._target_history
        if history is not None and history.get("lost_at_step") is None and history.get("released_at_step") is None:
            history.update(lost_at_step=self.step_index, lost_eef_xyz=_vec(view.position, 3), lost_cause=cause)

    def _note_reach_limit(self, evidence: dict) -> None:
        """Remember a workspace edge hit by a horizontal (or downward) push, for the outward-push guard.

        Lifts near the edge also lose orientation (real smoke 012: a move back and 3 cm up), but there the
        blocked part is the height, not the horizontal direction, so they are reported and not remembered."""
        direction = evidence.get("blocked_direction_xy")
        end = evidence.get("end_xyz")
        commanded = np.asarray(evidence.get("commanded_offset_m") or (0.0, 0.0, 0.0), float)
        if direction is None or end is None or commanded[2] > REACH_LIMIT_MAX_LIFT_M:
            return
        horizontal = commanded[:2]
        if float(np.linalg.norm(horizontal)) < REACH_LIMIT_PUSH_M or \
                float(horizontal @ np.asarray(direction[:2], float)) < 0.5 * float(np.linalg.norm(horizontal)):
            return
        view = self._context.view if self._context is not None else None
        self._reach_limits = (self._reach_limits + [{
            "step": self.step_index, "eef_xyz": _vec(end, 3), "blocked_direction_world": _vec(direction, 3),
            "hand": "turned" if view is not None and self._frontal(view) else "top_down",
            "orientation_error_deg": round(float(evidence.get("final_angle_deg") or 0.0), 1)}])[-REACH_LIMIT_MEMORY:]

    def _decide(self, view: StepView, calls: list) -> dict:
        request = {"state_summary": self._state_summary(view),
                   "external_image": encode_png_data_url(view.external_rgb_upright),
                   "wrist_image": encode_png_data_url(view.wrist_rgb_upright)}
        self._counts["decisions"] += 1
        return self._timed_call("decision", self.client.decide, request, calls)

    def _point(self, view: StepView, request: dict, calls: list) -> dict:
        """Coarse/fine pointing on the wrist image, turned so that world up is at the top.

        With a frontal hand (jaw vertical) the upright wrist view is rotated 90 deg; in drawer live run 001 the
        pointing model, told about a "horizontal" bar in an "UPRIGHT" view, hit the top handle in 1 of 11
        episodes (wine-rack post, cabinet side edge, own finger instead). The turn comes from the measured hand
        rotation only; answers are mapped back to the upright wrist image. Bar answers must also pass a
        zero-shot orientation check (one re-ask with feedback, then pointed_not_a_bar)."""
        primitive, context = self._primitive, self._context
        phrase = context.target_phrase if context is not None else "the target point"
        base = {"target_phrase": phrase, "context": dict(context.pointing_context) if context else {}}
        bar = base["context"].get("strategy") == "frontal_bar_pinch"
        rounds = self._pointing_rounds.setdefault(id(primitive), [])
        if rounds:
            base["context"]["previous_answer_rejected"] = rounds[-1]
        margin = WIDE_MARGIN_CELLS if bar else DEFAULT_MARGIN_CELLS
        turns = gravity_quarter_turns(float(view.R[2, 1]), float(view.R[2, 0]))
        rgb = view.wrist_rgb_upright
        width = rgb.shape[1]
        if turns:
            rgb = rotate_quarter(rgb, turns)
            base["context"]["image_orientation"] = (
                f"turned so that world up is at the top of this image; the robot's fingertips are at its "
                f"{FINGERTIP_EDGE[turns]} edge")
        if request.get("stage") == "coarse":
            image = coarse_overlay(rgb)
            self._counts["pointing_coarse"] += 1
            try:
                reply = self._timed_call("pointing_coarse", self.client.point,
                                         {"stage": "coarse", "image": image, **base}, calls)
            except SchemaError:
                reply = {"cell_id": None, "schema_error": True}
            primitive.provide_coarse(reply.get("cell_id"))
            return jsonable({"stage": "coarse", "cell_id": reply.get("cell_id"), "target_phrase": phrase,
                             "quarter_turns": turns})
        cell_id = int(request["cell_id"])
        image, origin = fine_overlay(rgb, cell_id, margin_cells=margin)
        base["context"]["cell_id"] = cell_id
        self._counts["pointing_fine"] += 1
        try:
            reply = self._timed_call("pointing_fine", self.client.point,
                                     {"stage": "fine", "image": image, **base}, calls)
        except SchemaError:
            reply = {"point_id": None, "schema_error": True}
        point_id = reply.get("point_id")
        uv = None if point_id is None else unrotate_pixel(
            fine_point_pixel(origin, int(point_id), width, margin), turns, width)
        result = {"stage": "fine", "cell_id": cell_id, "point_id": point_id, "quarter_turns": turns,
                  "uv_upright": None if uv is None else list(uv), "bar_check": None, "pointing_round": len(rounds) + 1}
        if uv is not None and bar:
            check = bar_point_check(view.wrist_gray_upright, uv, view.R)
            if uv[1] > FINGERTIP_BAND_ROW_FRACTION * view.wrist_gray_upright.shape[0]:
                check = {**check, "ok": False, "fingertip_band": True}   # live 773: an answer on the robot's own hand
            result["bar_check"] = check
            if not check["ok"]:
                # Re-ask once from the COARSE stage (drawer verify: 9 of 10 coarse cells held no handle pixel, so a
                # fine re-ask in the same crop cannot recover); then fail with pointed_not_a_bar.
                if not rounds:
                    rounds.append(f"your previous point was not on a bar that runs left-right in this image "
                                  f"({'on the robot' if check.get('fingertip_band') else 'a different structure'}); "
                                  "find the handle bar itself, or answer 0 if it is not visible")
                    primitive.restart_pointing()
                    return jsonable(result)
                primitive.fail_pointing("pointed_not_a_bar", detail="two pointing rounds failed the bar check",
                                        bar_check=check)
                return jsonable(result)
        primitive.provide_point(uv, view)
        if context is not None:
            context.last_pointing = result
        return jsonable(result)

    # ------------------------------------------------------------------ macro plumbing
    def _start_macro(self, skill: str, params: dict, generator) -> None:
        self._context = _Context(self, skill, params)
        self._macro, self._primitive = generator, None
        self._generator_started = False

    def _advance(self, view: StepView) -> dict | None:
        """Start or continue the active macro; returns its outcome if it ended."""
        self._context.view = view
        try:
            if not self._generator_started:
                self._generator_started = True
                factory = next(self._macro)
            else:
                finished = self._primitive
                factory = self._macro.send((finished.outcome(), finished))
        except StopIteration as stop:
            return self._finish(stop.value or {"status": "succeeded", "reason": "macro_complete"})
        self._primitive = factory(view)
        return None

    def _finish(self, outcome: dict) -> dict:
        skill = self._context.skill if self._context is not None else outcome.get("skill", "unknown")
        params = self._context.params if self._context is not None else outcome.get("params")
        view = self._context.view if self._context is not None else None
        pose = None if view is None else {"eef_xyz": _vec(view.position), "jaw_axis": _vec(view.R[:, 1]),
                                          "approach_axis": _vec(view.R[:, 2]), "finger_gap_mm": round(view.gap_mm, 3),
                                          "tilt_from_straight_down_deg": round(tilt_from_straight_down_deg(view.R), 1)}
        self._outcome_counter += 1
        outcome = jsonable({"skill": skill, "params": params, **{k: v for k, v in outcome.items() if k != "params"},
                            "pose": pose, "n": self._outcome_counter, "step": self.step_index})
        status = outcome.get("status", "unknown")
        if skill != "calibrate" and self._recent_decisions and self._recent_decisions[-1]["outcome"] is None:
            self._recent_decisions[-1]["outcome"] = f"{status}:{outcome.get('reason')}"
        self._consecutive_rejections = self._consecutive_rejections + 1 if status == "rejected" else 0
        # orientation_restored is a procedural re-ask (the side is chosen again in the top-down view), not a failed
        # grasp: drawer live 002 counted it, so the loop warning fired after 2 real failures and stayed on.
        procedural = outcome.get("reason") in PROCEDURAL_REASONS
        if skill == "grasp_with" and not procedural:
            target = (params or {}).get("target") or "?"
            if status == "succeeded":
                self._grasp_failures_by_target[target] = 0
            elif status == "failed":
                self._grasp_failures_by_target[target] = self._grasp_failures_by_target.get(target, 0) + 1
        if skill != "calibrate" and not procedural:
            if status in ("failed", "rejected"):
                self._failed_in_a_row += 1
                self._recent_failures = (self._recent_failures + [f"{skill}:{outcome.get('reason')}"])[-8:]
            elif status == "succeeded":
                self._failed_in_a_row, self._recent_failures = 0, []
        bucket = self._macro_counts.setdefault(skill, {})
        bucket[status] = bucket.get(status, 0) + 1
        self._last_outcome = outcome
        self.outcomes = (self.outcomes + [outcome])[-OUTCOME_HISTORY_LIMIT:]
        if skill != "calibrate" or status != "succeeded":
            self.ledger = (self.ledger + [self._ledger_line(outcome)])[-LEDGER_LIMIT:]
        self._macro, self._context, self._primitive = None, None, None
        return outcome

    def _ledger_line(self, outcome: dict) -> dict:
        evidence = outcome.get("evidence") or {}
        line = {"n": outcome.get("n"), "step": outcome.get("step"), "skill": outcome.get("skill"),
                "params": outcome.get("params"), "status": outcome.get("status"), "reason": outcome.get("reason")}
        for key in ("measured_displacement_m", "progress_fraction", "final_angle_deg", "attempts", "attempt_reasons",
                    "final_gap_mm", "travel_m", "pulled_total_m", "max_gap_rise_mm"):
            if evidence.get(key) is not None:
                line[key] = evidence[key]
        orient = evidence.get("orient_start") or evidence.get("orient")
        if isinstance(orient, dict) and orient.get("evidence"):
            line["orient"] = {key: orient["evidence"].get(key) for key in ("final_angle_deg", "pose_error_m")}
        if "measured_displacement_m" in evidence:
            shift = image_shift_for(self._external_axes_fit, evidence["measured_displacement_m"])
            if shift is not None:
                line["external_image_shift_px_right_down"] = shift
        pose = outcome.get("pose")
        if pose is not None:
            line["end_xyz"] = _vec(pose["eef_xyz"], 3)
            line["tilt_from_straight_down_deg"] = pose.get("tilt_from_straight_down_deg")
        if outcome.get("detail"):
            line["detail"] = outcome["detail"]
        return _rounded(jsonable(line))

    def _compact(self, outcome: dict) -> dict:
        keep = ("skill", "params", "status", "reason", "detail", "step")
        compact = {key: outcome[key] for key in keep if key in outcome}
        evidence = outcome.get("evidence") or {}
        compact["evidence"] = {key: evidence[key] for key in (
            "measured_displacement_m", "final_error_m", "progress_fraction", "final_angle_deg", "attempts",
            "final_gap_mm", "travel_m", "attempt_reasons", "pulled_total_m", "max_gap_rise_mm") if key in evidence}
        details = evidence.get("details")
        if details:
            last = details[-1]
            close, hold = last.get("close") or {}, last.get("hold") or {}
            align = last.get("align") or {}
            compact["evidence"]["last_attempt"] = {
                "align": align.get("reason"), "align_detail": align.get("detail"),
                "pointed_uv_upright_wrist": align.get("pointed_uv_upright"),
                "align_residual_mm_est": align.get("residual_mm_est"),
                "contact": (last.get("contact") or {}).get("reason"),
                "finger_deflected": (last.get("contact") or {}).get("finger_deflected"),
                "descended_m": last.get("descended_m"),
                "contact_grip_z": (last.get("contact") or {}).get("contact_grip_z"),
                "gap_k15_mm": close.get("gap_k15_mm"), "empty_k15_mm": close.get("empty_k15_mm"),
                "settle_k": close.get("settle_k"), "hold": hold.get("reason"),
                "monitor_disagrees": (hold.get("evidence") or {}).get("monitor_disagrees")}
        return _rounded(jsonable(compact))

    # ------------------------------------------------------------------ decisions
    def _begin_decision(self, decision: dict, view: StepView) -> dict | None:
        skill, params = decision.get("skill"), dict(decision.get("params") or {})
        compact_params = {key: value for key, value in params.items() if value is not None}

        def reject(reason, detail):
            self._counts["rejected_decisions"] += 1
            self._context = _Context(self, skill or "invalid", compact_params)
            return {"status": "rejected", "reason": reason, "detail": detail, "params": compact_params}

        if skill not in SKILLS:
            return reject("invalid_parameters", "unknown skill")
        offset = params.get("waypoint_offset_m")
        segments = [offset]
        if params.get("more_waypoints_m"):
            if skill not in ("reach", "transport"):
                return reject("invalid_parameters", "more_waypoints_m is only for reach and transport")
            segments = [offset, *params["more_waypoints_m"]]
        if skill in ("reach", "transport", "place_release"):
            for segment in segments:
                if (segment is None or len(segment) != 3
                        or any(not math.isfinite(float(v)) or abs(float(v)) > 0.075 + 1e-9 for v in segment)):
                    return reject("invalid_parameters", f"{skill} waypoints need three values within ±0.075 m each")
                if skill in ("reach", "transport") and float(np.linalg.norm(np.asarray(segment, float))) <= MIN_MOVE_OFFSET_M:
                    return reject("invalid_parameters", f"{skill} waypoints must be longer than 5 mm (they would not move)")
        if skill in ("transport", "place_release", "pull") and not self._holding:
            return reject("precondition_unmet", f"{skill} requires a verified grasp; none is held")
        if skill in ("reach", "transport") and self._frontal(view):
            now, goal = wrist_to_shoulder_m(view.position, view.R[:, 2]), view.position.copy()
            for segment in segments:
                goal = goal + np.asarray([float(v) for v in segment])
                then = wrist_to_shoulder_m(goal, view.R[:, 2])
                if then > FRONTAL_STRETCH_LIMIT_M and then > now + 0.002:
                    return reject("beyond_frontal_reach", self._stretch_detail(then, view.position))
                now = max(now, then)
        if skill in ("reach", "transport", "place_release") and self._reach_limits:
            push_xy = np.asarray([float(v) for v in offset[:2]])
            # only limits met with the same hand orientation (drawer live 002: a top-down limit blocked the turned
            # hand's along-bar move)
            for limit in reversed([l for l in self._reach_limits if l.get("hand") == self._hand_kind(view)]):
                near = (float(np.linalg.norm(view.position[:2] - np.asarray(limit["eef_xyz"][:2]))) <= REACH_LIMIT_RADIUS_M
                        and abs(float(view.position[2]) - limit["eef_xyz"][2]) <= REACH_LIMIT_RADIUS_M)
                push = float(push_xy @ np.asarray(limit["blocked_direction_world"][:2]))
                if near and push > REACH_LIMIT_PUSH_M:
                    return reject("beyond_recorded_reach_limit",
                                  f"the arm hit its workspace limit near here at step {limit['step']} moving toward "
                                  f"world {limit['blocked_direction_world'][:2]}; this offset pushes {push:.3f} m further "
                                  "that way. Move the other way (back toward the robot) or bring the object closer")
        if skill == "grasp_with":
            self._counts["grasp_with_calls"] += 1
            strategy, target = params.get("strategy"), params.get("target")
            if self._holding:
                return reject("precondition_unmet", "an object is still held; release it with place_release "
                                                    "(waypoint_offset_m [0, 0, 0] releases in place); retreat keeps the grasp")
            if strategy not in STRATEGIES or target not in TASK_TARGETS[self.task_key]:
                return reject("invalid_parameters", "grasp_with requires target and strategy")
            if (strategy == "rim_pinch") == ("handle" in target):
                return reject("invalid_parameters", "use frontal_bar_pinch for handles and rim_pinch for bowls/containers")
            if strategy == "rim_pinch" and params.get("side") not in SIDES:
                return reject("invalid_parameters", "rim_pinch requires side wrist_left or wrist_right")
            if strategy == "frontal_bar_pinch" and params.get("approach") not in APPROACHES:
                return reject("invalid_parameters", "frontal_bar_pinch requires a world approach direction")
        if skill == "pull" or (skill == "grasp_with" and params.get("max_travel_m") is not None):
            travel = params.get("max_travel_m")
            if travel is None or not 0.02 - 1e-9 <= float(travel) <= 0.16 + 1e-9:
                return reject("invalid_parameters", "pull requires max_travel_m in [0.02, 0.16]")
            if skill == "grasp_with" and params.get("strategy") != "frontal_bar_pinch":
                return reject("invalid_parameters", "max_travel_m with grasp_with is only for frontal_bar_pinch (grasp then pull)")
        generator = {
            "reach": lambda: self._reach_path(segments),
            "transport": lambda: self._transport_path(segments),
            "place_release": lambda: self._place_release(offset),
            "retreat": lambda: self._retreat(),
            "pull": lambda: self._pull(float(params["max_travel_m"])),
            "grasp_with": lambda: (self._grasp_rim(params["target"], params["side"])
                                   if params["strategy"] == "rim_pinch"
                                   else self._grasp_bar(params["target"], params["approach"],
                                                        pull_m=params.get("max_travel_m"))),
        }[skill]()
        self._start_macro(skill, compact_params, generator)
        return None

    # ------------------------------------------------------------------ macros
    def _calibrate(self):
        _, primitive = yield (lambda v: Calibrate(v))
        self.calibration = getattr(primitive, "calibration", None) or GripperCalibration.default()
        evidence = self.calibration.to_dict()
        if self.external_axes == "measured":
            evidence["external_axes"] = yield from self._probe_external_axes()
        return {"status": "succeeded", "reason": "calibrated", "evidence": evidence}

    def _probe_external_axes(self):
        """Short pure-translation probe moves (net zero); fit the external-image direction map from flow."""
        samples = []
        for offset in PROBE_MOVES:
            before = self._context.view
            outcome, _ = yield (lambda v, o=offset: WaypointReach(v, list(o), gripper="open", name="axes_probe"))
            after = self._context.view
            try:
                motion = hand_image_motion(before.external_rgb_upright, after.external_rgb_upright)
            except ValueError:
                motion = None
            samples.append({"commanded_m": list(offset), "reach_reason": outcome["reason"],
                            "world_displacement_m": _vec(after.position - before.position),
                            "image_shift_px": None if motion is None else motion["image_shift_px"],
                            "flow_pixels": None if motion is None else motion["pixels"]})
        fit = fit_external_axes(samples)
        fit["probes"] = samples
        self._external_axes_fit = jsonable(fit)
        return self._external_axes_fit

    def _hold_floor_mm(self) -> float:
        return (self.calibration or GripperCalibration.default()).empty_floor_mm + HOLD_MARGIN_MM

    def _held_reach(self, view: StepView, offset, name: str):
        """WaypointReach while holding: per-step hold/contact checks, grasp orientation kept when close to it."""
        orientation = None
        if self._grasp_quaternion is not None and \
                quaternion_angle_rad(view.quaternion_xyzw, self._grasp_quaternion) <= MAX_HELD_REORIENT_RAD:
            orientation = self._grasp_quaternion
        return WaypointReach(view, list(offset), gripper="close", name=name, orientation_xyzw=orientation,
                             hold_floor_mm=self._hold_floor_mm(),
                             gap_ref_mm=self._hold_gap_ref_mm if self._hold_gap_ref_mm is not None else view.gap_mm)

    def _move_detail(self, reason: str, evidence: dict) -> str | None:
        blocked = evidence.get("blocked_direction_xy")
        toward = f" toward world {_vec(blocked, 2)}" if blocked else ""
        if reason == "reach_limit":
            return (f"the hand could no longer keep its orientation moving this way (error "
                    f"{evidence.get('final_angle_deg', 0.0):.0f} deg and growing) after "
                    f"{evidence.get('progress_fraction', 0.0):.0%} of the move: workspace edge or contact; do not "
                    f"push further{toward}")
        if reason == "frontal_stretch_limit":
            return self._stretch_detail(float(evidence.get("wrist_to_shoulder_m") or 0.0),
                                        evidence.get("end_xyz") or self._context.view.position)
        if reason == "finger_contact":
            return (f"an open finger touched something (finger gap changed "
                    f"{evidence.get('finger_gap_change_mm', 0.0):+.1f} mm) after "
                    f"{evidence.get('progress_fraction', 0.0):.0%} of the move; go around it (up first)")
        if reason == "uncontrolled_motion":
            return "the arm jumped more than 2 cm in one step (it is near its workspace edge); stopped"
        if reason == "left_planning_grid":
            return "the measured hand overshot the planned move (left the local planning region); stopped"
        if reason == "blocked":
            return (f"made only {evidence.get('progress_fraction', 0.0):.0%} of the move{toward} (contact or the "
                    "arm cannot move this way in this orientation)")
        if reason == "orientation_drift":
            return "position reached but the hand orientation drifted (workspace edge); stopped"
        return None

    def _move_result(self, outcome: dict, *, record_limit: bool = True) -> dict:
        reason, evidence = outcome["reason"], outcome.get("evidence", {})
        if reason == "reach_limit" and record_limit:
            self._note_reach_limit(evidence)
        result = {"status": outcome["status"], "reason": reason, "evidence": evidence}
        detail = self._move_detail(reason, evidence)
        if detail:
            result["detail"] = detail
        return result

    def _after_held_move(self, outcome: dict, offset=None):
        """Resolve a held WaypointReach: loss, held-object contact (back off 1 cm) or the plain move result."""
        evidence = outcome.get("evidence", {})
        view = self._context.view
        if outcome["reason"] == "hold_lost" or not self._holding or view.gap_mm < self._hold_floor_mm():
            self._holding = False
            self._note_lost(view, "lost_during_move")
            return {"status": "failed", "reason": "lost_in_transport", "evidence": evidence,
                    "detail": "the finger gap fell to the empty-close floor during the move: the object dropped"}
        if outcome["reason"] == "held_object_contact":
            # Lift off a support (live 001), but after hitting something while rising (an overhang: drawer 002 executor
            # lens F4) back off against the move; lifting again repeated the contact until the bowl dropped.
            # Executor verify (from 799): straight down needed 6 contacts to scrape past; 5 mm down plus 10 mm out from
            # under the overhang needed 2, then lifted clean.
            rising = offset is not None and float(offset[2]) > 0.0 and float(np.linalg.norm(offset)) > 1e-9
            away = self._held_away_world
            if rising and away is not None:
                vec = np.array([0.010 * away[0], 0.010 * away[1], -0.005])
                backed = f"moved 1 cm along world {away} (out from under the overhang) and 5 mm down"
            elif rising:
                vec = -np.asarray(offset, float) / float(np.linalg.norm(offset)) * HELD_CONTACT_BACKOFF_M
                backed = "moved back 1 cm against the move"
            else:
                vec, backed = np.array([0.0, 0.0, HELD_CONTACT_BACKOFF_M]), "lifted 1 cm"
            travel = float(np.linalg.norm(vec))
            axis = tuple(vec / travel)
            backoff, _ = yield (lambda v: AnchoredMove(v, axis=axis, travel_m=travel, gripper="close",
                                                       name="contact_backoff"))
            evidence = {**evidence, "backoff": backoff}
            if not self._holding:
                self._note_lost(self._context.view, "lost_during_contact_backoff")
                return {"status": "failed", "reason": "lost_in_transport", "evidence": evidence,
                        "detail": "the held object hit something and dropped"}
            return {"status": "failed", "reason": "held_object_contact", "evidence": evidence,
                    "detail": f"the held object pressed against something (finger gap rose "
                              f"{evidence.get('max_gap_rise_mm') or 0.0:.1f} mm) after "
                              f"{evidence.get('progress_fraction', 0.0):.0%} of the move; the hand was {backed}"}
        return self._move_result(outcome)

    def _reach(self, offset):
        # A frontal-hand stall is the arm's elbow singularity (drawer verify: every failed frontal reach from step
        # 531 started and ended at full stretch), not controller coupling: report it honestly and remember it.
        if self._holding:
            return (yield from self._transport(offset))
        gripper = "close" if self._gripper_intent == "close" else "open"
        outcome, _ = yield (lambda v: WaypointReach(v, list(offset), gripper=gripper, wrist_sphere=self._wrist_sphere(v)))
        return self._move_result(outcome)

    def _wrist_sphere(self, view: StepView):
        return (SHOULDER_XYZ, WRIST_FROM_GRIP_M, FRONTAL_STRETCH_STOP_M) if self._frontal(view) else None

    def _reach_path(self, segments):
        """Up to three waypoints in one decision (drawer live 002: 18 decisions to step around the rack)."""
        return (yield from self._path(segments, self._reach))

    def _transport_path(self, segments):
        return (yield from self._path(segments, self._transport))

    def _path(self, segments, move):
        if len(segments) == 1:
            return (yield from move(segments[0]))
        start = self._context.view.position.copy()
        done = []
        for index, segment in enumerate(segments):
            result = yield from move(segment)
            done.append({"segment": index + 1, "reason": result["reason"],
                         "measured_displacement_m": (result.get("evidence") or {}).get("measured_displacement_m")})
            if result["status"] != "succeeded":
                evidence = {**(result.get("evidence") or {}), "segments": done,
                            "measured_displacement_m": _vec(self._context.view.position - start)}
                detail = f"waypoint {index + 1} of {len(segments)} ended {result['reason']}"
                if result.get("detail"):
                    detail += ": " + result["detail"]
                return {**result, "evidence": evidence, "detail": detail}
        last = dict(result)
        last["evidence"] = {**(last.get("evidence") or {}), "segments": done,
                            "measured_displacement_m": _vec(self._context.view.position - start)}
        return last

    def _transport(self, offset):
        outcome, _ = yield (lambda v: self._held_reach(v, offset, "transport"))
        return (yield from self._after_held_move(outcome, offset))

    def _release_and_back_off(self, evidence: dict, reason: str):
        outcome, _ = yield (lambda v: GripperSettlePrimitive("open", v, config=OPEN_SETTLE, name="release"))
        evidence["release"] = outcome
        if self._target_history is not None and self._target_history.get("lost_at_step") is None:
            self._target_history.update(released_at_step=self.step_index,
                                        released_eef_xyz=_vec(self._context.view.position, 3))
        self._holding, self._pulled_m, self._held_strategy = False, 0.0, None
        self._grasp_quaternion = self._hold_gap_ref_mm = None
        self._held_away_world = None
        outcome, _ = yield (lambda v: AnchoredMove(v, axis=-v.R[:, 2], travel_m=RETREAT_M, gripper="open", name="retreat"))
        evidence["retreat"] = outcome
        return {"status": "succeeded", "reason": reason, "evidence": evidence}

    def _place_release(self, offset):
        evidence = {}
        in_place = not any(abs(float(v)) > 1e-9 for v in offset)
        if in_place and self._held_strategy == "frontal_bar_pinch":
            # Releasing a handle: open where it is and back away; never push down on the mechanism. Then lift
            # clear and turn top-down in the same decision, so the next grasp sees a top-down view (drawer live
            # 002 spent two more decisions: a 7.5 cm lift and grasp_with -> orientation_restored).
            result = yield from self._release_and_back_off(evidence, "released_handle")
            yield (lambda v: AnchoredMove(v, axis=(0., 0., 1.), travel_m=HANDLE_RELEASE_LIFT_M, gripper="open",
                                          name="release_lift"))
            restored = yield from self._restore_start_orientation()
            if restored is not None:
                result["evidence"]["orient_start"] = restored
                if restored["status"] != "succeeded":
                    result["detail"] = ("released the handle; turning the hand top-down failed "
                                        f"({ORIENT_REASONS.get(restored['reason'], restored['reason'])}); grasp_with "
                                        "will try again")
                else:
                    result["detail"] = "released the handle, lifted and turned the hand top-down"
            return result
        # The hover never descends while holding (live 001: a -35 mm hover pressed the bowl into its support for
        # 15 blind steps); any downward part is left to the guarded lowering.
        hover = [float(offset[0]), float(offset[1]), max(0.0, float(offset[2]))]
        if float(np.linalg.norm(hover)) > MIN_MOVE_OFFSET_M:
            outcome, _ = yield (lambda v: self._held_reach(v, hover, "place_hover"))
            evidence["reach"] = outcome
            if outcome["status"] != "succeeded":
                result = yield from self._after_held_move(outcome, hover)
                if result["reason"] == "lost_in_transport":
                    return {**result, "evidence": evidence}
                return {"status": "failed", "reason": f"hover_{result['reason']}", "evidence": evidence,
                        **({"detail": result["detail"]} if result.get("detail") else {})}
        view = self._context.view
        floor = self._pick_floor_z if self._held_strategy == "rim_pinch" else None
        if floor is not None:
            bound = float(np.clip(view.position[2] - floor + PLACE_BELOW_PICK_M, LOWER_BOUND_M, PLACE_LOWER_MAX_M))
            fast = max(0.0, float(view.position[2]) - (floor + PLACE_FAST_ABOVE_PICK_M))
        else:
            bound, fast = LOWER_BOUND_M, 0.0
        evidence["lower_bound_m"] = bound
        lowered, contact = 0.0, None
        if fast > 0.010:
            # Fast segment above the pick height; an event here only hands over to the slow segment (a false
            # gap rise in mid-air must never release the object).
            outcome, _ = yield (lambda v: AnchoredMove(
                v, axis=(0., 0., -1.), travel_m=min(fast, bound), gripper="close", command_m=PLACE_FAST_COMMAND_M,
                stop_on_contact=True, gap_rise_mm=PLACE_GAP_RISE_MM, name="lower_fast"))
            evidence["lower_fast"] = outcome
            lowered += max(float(outcome.get("evidence", {}).get("travel_m", 0.0)), 0.0)
        remaining = bound - lowered
        if remaining > 0.002 and self._holding:
            outcome, _ = yield (lambda v: AnchoredMove(
                v, axis=(0., 0., -1.), travel_m=remaining, gripper="close", stop_on_contact=True,
                gap_rise_mm=PLACE_GAP_RISE_MM, name="lower"))
            evidence["lower"] = outcome
            lowered += max(float(outcome.get("evidence", {}).get("travel_m", 0.0)), 0.0)
            if outcome["reason"] in SUPPORT_EVENTS:
                contact = outcome
        view = self._context.view
        if not self._holding or view.gap_mm < self._hold_floor_mm():
            self._holding = False
            self._note_lost(view, "lost_while_lowering")
            return {"status": "failed", "reason": "lost_in_transport", "evidence": evidence,
                    "detail": "the object dropped while lowering"}
        if contact is None:
            above = None if floor is None else float(view.position[2]) - floor
            return {"status": "failed", "reason": "no_support_contact_within_bound", "evidence": evidence,
                    "detail": (f"lowered {lowered:.3f} m without feeling a support"
                               + ("" if above is None else f"; the grip point is {above:.3f} m above the pick height"))}
        return (yield from self._release_and_back_off(evidence, "released"))

    def _retreat(self):
        gripper = "close" if self._holding or self._gripper_intent == "close" else "open"
        outcome, _ = yield (lambda v: AnchoredMove(v, axis=-v.R[:, 2], travel_m=RETREAT_M, gripper=gripper, name="retreat"))
        return {"status": outcome["status"], "reason": outcome["reason"], "evidence": outcome.get("evidence", {})}

    def _pull(self, max_travel):
        remaining = max(0.005, max_travel - self._pulled_m)
        outcome, _ = yield (lambda v: AnchoredMove(v, axis=-v.R[:, 2], travel_m=remaining, gripper="close",
                                                   stop_on_contact=True, stop_on_gap_jump=True, name="pull"))
        travel = float(outcome.get("evidence", {}).get("travel_m", 0.0))
        self._pulled_m += max(travel, 0.0)
        reason = {"travel_reached": "pull_bound", "contact_stall": "pull_stall", "contact_touch": "pull_stall",
                  "gap_jump": "pull_gap_jump"}.get(outcome["reason"], outcome["reason"])
        view = self._context.view
        if not self._holding or (view is not None and view.gap_mm < self._hold_floor_mm()):
            self._holding = False
            self._note_lost(view, "lost_while_pulling")
            reason = "pull_lost_grip"
        status = "succeeded" if reason in ("pull_bound", "pull_stall", "pull_gap_jump") else "failed"
        return {"status": status, "reason": reason,
                "evidence": {**outcome.get("evidence", {}), "pulled_total_m": self._pulled_m}}

    def _ensure_open(self):
        view = self._context.view
        calibration = self.calibration or GripperCalibration.default()
        if view is None or view.gap_mm < calibration.open_gap_mm - OPEN_TOLERANCE_MM:
            yield (lambda v: GripperSettlePrimitive("open", v, config=OPEN_SETTLE, name="open"))

    def _restore_start_orientation(self):
        """Rotate back to the measured episode-start (top-down) orientation.

        Large turns (a frontal hand) retreat along the approach first. A moderate tilt is the forced pitch of the
        workspace edge (live 001), so the hand first moves back toward the robot (opposite the approach axis's
        horizontal part) and up, where top-down is feasible; rotating in place there dragged the hand 4-12 cm
        back and down onto the table. Orient fails fast on pose drift and accepts 5 deg at budget.
        """
        view = self._context.view
        target = self._start_quaternion if self._start_quaternion is not None else \
            axes_to_quaternion_xyzw(START_JAW, START_APPROACH)
        angle = quaternion_angle_rad(view.quaternion_xyzw, target)
        if angle <= math.radians(10.0):
            return None
        pre = None
        if angle > LARGE_TURN_RAD:
            yield (lambda v: AnchoredMove(v, axis=-v.R[:, 2], travel_m=LARGE_TURN_RETREAT_M, gripper="open",
                                          name="turn_retreat"))
        elif angle <= RESTORE_INWARD_MAX_RAD:
            horizontal = np.array([view.R[0, 2], view.R[1, 2]])
            if float(np.linalg.norm(horizontal)) > math.sin(math.radians(10.0)):
                inward = -horizontal / float(np.linalg.norm(horizontal)) * RESTORE_INWARD_M
                z = float(view.position[2])
                lift = (RESTORE_DEFAULT_LIFT_M if self._pick_floor_z is None
                        else max(0.0, self._pick_floor_z + RESTORE_CLEAR_ABOVE_PICK_M - z))
                offset = [float(inward[0]), float(inward[1]), min(lift, 0.075)]
                pre, _ = yield (lambda v: WaypointReach(v, offset, gripper="open", name="restore_inward",
                                                        reach_limit=False))
        outcome, _ = yield (lambda v: Orient(v, target_xyzw=target, name="orient_start",
                                             budget_tolerance_rad=RESTORE_BUDGET_TOLERANCE_RAD,
                                             max_pose_drift_m=ORIENT_MAX_DRIFT_M))
        yield (lambda v: HoldPrimitive(v, steps=POST_ORIENT_HOLD_STEPS, gripper="open", name="orient_settle"))
        if pre is not None:
            outcome = {**outcome, "pre_move": {"reason": pre["reason"],
                                               "measured_displacement_m": pre.get("evidence", {}).get("measured_displacement_m")}}
        return outcome

    def _orient_failure(self, restored: dict, key: str = "orient_start") -> dict:
        evidence = restored.get("evidence", {})
        reason = ORIENT_REASONS.get(restored["reason"], restored["reason"])
        return {"status": "failed", "reason": reason, "evidence": {key: restored},
                "detail": (f"could not turn the hand to the grasp orientation here (still "
                           f"{evidence.get('final_angle_deg', 0.0):.0f} deg off after {evidence.get('steps', 0)} steps, "
                           f"hand pushed {100 * float(evidence.get('pose_error_m') or 0.0):.0f} cm): contact with a "
                           "nearby object or the workspace edge; move clear of nearby objects and up (a turned hand "
                           "reaches about 10 cm around the grip point), then try again")}
    def _descend_ladder(self, align: Align, offset, axis, calibration, attempt_evidence, axes: str = "both",
                        hold_xyzw=None, along_xyz=None):
        """Guarded approach along ``axis`` with Align refreshes every LADDER_SEGMENT_M until cutoff.

        across_bar refreshes re-probe kappa with one short move along R1: the kappa transfer needs the mark's
        radial offset, which a bar point aligned on the grip axis does not have (drawer live run 001). They
        also hold the strategy orientation and the attempt's along-bar point, so neither drifts over segments
        (closed-loop replays: +30 mm along the bar during the descent)."""
        track, jacobian = align.track, align.jacobian
        descended, cutoff, contact = 0.0, False, None
        bar = axes == "across_bar"

        def lateral_target(v):
            if not bar or along_xyz is None:
                return None
            along = v.R[:, 0]
            return v.position + float((np.asarray(along_xyz) - v.position) @ along) * along

        # Settle until still before the first segment as well: an Align can end converged while the arm still
        # carries lateral momentum (real drawer smoke 011: from-rest ratio 0.04, false contact_stall after 3 mm).
        yield (lambda v: HoldPrimitive(v, steps=FIRST_SETTLE_MAX_STEPS, gripper="open", name="ladder_settle",
                                       until_quiet=True, orientation_xyzw=hold_xyzw))
        while descended < APPROACH_BOUND_M - 0.002:
            segment = APPROACH_BOUND_M - descended if cutoff else min(LADDER_SEGMENT_M, APPROACH_BOUND_M - descended)
            outcome, _ = yield (lambda v, s=segment: AnchoredMove(
                v, axis=axis, travel_m=s, gripper="open", stop_on_contact=True,
                touch_threshold_mm=calibration.touch_threshold_mm, name="guarded_approach",
                orientation_xyzw=hold_xyzw, lateral_target=lateral_target(v), arm_stall_after_motion=bar))
            descended += max(float(outcome.get("evidence", {}).get("travel_m", 0.0)), 0.0)
            if outcome["reason"] in ("contact_stall", "contact_touch"):
                contact = outcome
                break
            if outcome["status"] != "succeeded" or cutoff:
                if outcome["status"] != "succeeded":
                    attempt_evidence["approach_failure"] = outcome["reason"]
                    break
                continue
            # OSC lag keeps the hand drifting along the approach for a few steps; settle before the
            # refresh probe so its lateral-only Jacobian fit is not corrupted (real-LIBERO smoke).
            yield (lambda v: HoldPrimitive(v, steps=LADDER_SETTLE_STEPS, gripper="open", name="ladder_settle"))
            refresh, refreshed_align = yield (lambda v: Align(
                v, offset_m=offset, gripper="open", track=track, jacobian=jacobian, reprobe=bar,
                allow_pointing=False, setpoint_row_limit=SETPOINT_ROW_FRACTION * v.wrist_rgb_upright.shape[0],
                name="align_refresh", axes=axes, hold_xyzw=hold_xyzw if bar else None,
                along_point_xyz=along_xyz if bar else None))
            # reprobe=False: on real rims the +/-2 mm probe is ill-conditioned along the edge (smoke 005:
            # kappa 2836 vs ~950 px/m); the depth-free kappa transfer plus servo similarity fit is used instead.
            attempt_evidence.setdefault("ladder", []).append(
                {"reason": refresh["reason"], "residual_px": refresh.get("evidence", {}).get("residual_px")})
            if refresh["status"] != "succeeded" or refresh["reason"] == "setpoint_cutoff":
                cutoff = True
            else:
                track, jacobian = refreshed_align.track, refreshed_align.jacobian
        attempt_evidence["descended_m"] = descended
        return contact

    def _grasp_attempt_close(self, calibration, lift_axis, lift_m, attempt_evidence):
        """Settle close, classify against the online empty curve, then the scripted hold test."""
        outcome, settle = yield (lambda v: GripperSettlePrimitive("close", v, config=GRASP_SETTLE, name="grasp"))
        classified = classify_close(settle, calibration)
        attempt_evidence["close"] = classified
        if not classified["object_present"]:
            return "empty_close"
        outcome, _ = yield (lambda v: VerifyHold(v, axis=lift_axis, travel_m=lift_m, calibration=calibration,
                                                 hold_margin_mm=HOLD_MARGIN_MM, monitor=WristPatchMonitor(v)))
        attempt_evidence["hold"] = {"reason": outcome["reason"], "evidence": outcome.get("evidence", {})}
        return outcome["reason"]

    def _return_to_hover(self, hover_xyz, approach):
        yield from self._ensure_open()
        yield (lambda v: AnchoredMove(
            v, axis=-np.asarray(approach, float), gripper="open", lateral_target=hover_xyz,
            travel_m=max(0.005, float((np.asarray(hover_xyz) - v.position) @ (-np.asarray(approach, float)))),
            name="retreat_to_hover"))

    def _note_grasp(self, target: str) -> None:
        view = self._context.view
        self._grasp_quaternion = view.quaternion_xyzw.copy()
        self._hold_gap_ref_mm = float(view.gap_mm)
        self._target_history = {"held_target": target, "grasped_at_step": self.step_index,
                                "grasp_eef_xyz": _vec(view.position, 3), "lost_at_step": None}

    def _grasp_rim(self, target, side):
        """Straddle-and-floor-touch rim pinch at the start orientation (agent_SpeedUp_v1.md §7.1)."""
        context, calibration = self._context, self.calibration or GripperCalibration.default()
        center_sign = 1.0 if side == "wrist_left" else -1.0
        context.target_phrase = rim_target_phrase(target, side)
        context.pointing_context = rim_pointing_context(target, side, getattr(self.client, "target_description", None))
        yield from self._ensure_open()
        restored = yield from self._restore_start_orientation()
        if restored is not None:
            if restored["status"] != "succeeded":
                return self._orient_failure(restored)
            # The side was chosen in the previous (turned) wrist view; ask again in the top-down view.
            return {"status": "failed", "reason": "orientation_restored",
                    "detail": "the hand is now top-down; choose side from this wrist image and call grasp_with again",
                    "evidence": {"orient_start": restored}}
        attempts, reasons, shift = [], [], 0.0
        hover_track, jacobian, hover_xyz, approach = None, None, None, None
        outward = list(BLOCKED_OUTWARD_SHIFTS_M)
        blocked_seen = False
        for attempt in range(self.local_retries + 1):
            self._counts["grasp_attempts"] += 1
            if attempt:
                self._counts["local_retries_used"] += 1
            evidence: dict = {"attempt": attempt + 1, "shift_m": shift}
            # shift > 0 moves the grip toward the container centre, < 0 away from it (after blocked_above_rim)
            offset = (GRIP_AXIS_OFFSET_M[0] - shift * center_sign, GRIP_AXIS_OFFSET_M[1])
            track = copy.deepcopy(hover_track) if hover_track is not None else None
            outcome, align = yield (lambda v, o=offset, t=track, j=jacobian: Align(
                v, offset_m=o, gripper="open", track=t, jacobian=j, reprobe=j is None, allow_pointing=True,
                name="align"))
            evidence["align"] = {"reason": outcome["reason"], "residual_px": outcome.get("evidence", {}).get("residual_px"),
                                 "residual_mm_est": outcome.get("evidence", {}).get("residual_mm_est"),
                                 "detail": outcome.get("evidence", {}).get("detail")}
            if outcome["status"] != "succeeded":
                reasons.append(outcome["reason"])
                attempts.append(evidence)
                break
            view = context.view
            hover_track, jacobian = copy.deepcopy(align.track), align.jacobian
            hover_xyz = view.position.copy()
            if approach is None:
                approach = view.R[:, 2].copy()
            contact = yield from self._descend_ladder(align, offset, approach, calibration, evidence)
            if contact is None:
                reasons.append(evidence.get("approach_failure") or "no_contact_within_bound")
                attempts.append(evidence)
                yield from self._return_to_hover(hover_xyz, approach)
                break
            ladder = evidence.get("ladder") or []
            if ladder and all(entry.get("reason") == "aligned" for entry in ladder):
                # Contact before the ladder's setpoint cutoff: the aligned rim mark is still far below the fingertips,
                # so this is not the inner fingertip on the container floor. 10/10 recorded hold_pass attempts reached
                # the cutoff first; 13/13 drawer live 002 stalls (the palm on the open drawer front) did not. Do not
                # close; retry with the grip moved AWAY from the centre (the hand body extends ~10 cm both ways).
                contact_end = contact.get("evidence", {}).get("end_xyz")
                evidence["contact"] = {"reason": "blocked_above_rim",
                                       "contact_grip_z": None if contact_end is None else round(float(contact_end[2]), 4),
                                       "descent_start_z": round(float(hover_xyz[2]), 4)}
                if contact_end is not None:
                    self._contact_heights.setdefault(target, []).append(float(contact_end[2]))
                reasons.append("blocked_above_rim")
                attempts.append(evidence)
                blocked_seen = True
                yield from self._return_to_hover(hover_xyz, approach)
                if not outward:
                    break
                shift = -outward.pop(0)
                continue
            contact_xyz = contact.get("evidence", {}).get("contact_xyz") or contact.get("evidence", {}).get("end_xyz")
            evidence["contact"] = {"reason": contact["reason"],
                                   "finger_deflected": contact.get("evidence", {}).get("finger_deflected"),
                                   "contact_grip_z": None if contact_xyz is None else round(float(contact_xyz[2]), 4),
                                   "descent_start_z": round(float(hover_xyz[2]), 4)}
            if contact_xyz is not None:
                self._contact_heights.setdefault(target, []).append(float(contact_xyz[2]))
            yield (lambda v: AnchoredMove(v, axis=-approach, travel_m=RIM_CLEARANCE_M, gripper="open", name="clearance"))
            result = yield from self._grasp_attempt_close(calibration, (0., 0., 1.), LIFT_TEST_M, evidence)
            reasons.append(result)
            attempts.append(evidence)
            if result == "hold_pass":
                self._holding, self._pulled_m, self._held_strategy = True, 0.0, "rim_pinch"
                if contact_xyz is not None:
                    self._pick_floor_z = float(contact_xyz[2])
                self._note_grasp(target)
                success = {"status": "succeeded", "reason": "hold_pass",
                           "evidence": {"attempts": len(attempts), "attempt_reasons": reasons, "details": attempts}}
                if blocked_seen:
                    # Geometry lens G6 (drawer 002): the container's far side can still lie under what blocked the
                    # hand; a straight lift hit the drawer and dropped the bowl. The blocked side is the centre side.
                    away = -center_sign * np.asarray(self._context.view.R[:, 1], float)
                    away[2] = 0.0
                    away = [v + 0.0 for v in _vec(away / max(float(np.linalg.norm(away)), 1e-9), 2)]   # no -0.0
                    success["evidence"]["away_from_blocked_side_world"] = away
                    self._held_away_world = away
                    success["detail"] = (
                        "held after an outward-shifted retry (the hand had landed on something above the container's "
                        f"far side, which may overhang it): before lifting more than about 5 cm, move it about 4 cm "
                        f"horizontally along world {away} (away from the blocked side)")
                return success
            shift = (-outward.pop(0)) if (blocked_seen and outward) else shift + RETRY_SHIFT_M
            yield from self._return_to_hover(hover_xyz, approach)
        result = {"status": "failed", "reason": reasons[-1] if reasons else "no_attempt",
                  "evidence": {"attempts": len(attempts), "attempt_reasons": reasons, "details": attempts}}
        if result["reason"] == "blocked_above_rim":
            stop = [(a.get("contact") or {}).get("contact_grip_z") for a in attempts]
            stop = [z for z in stop if z is not None]
            result["detail"] = (
                "the hand stopped on something before the fingers reached the rim"
                + (f" (grip z {', '.join(f'{z:.3f}' for z in stop)})" if stop else "")
                + "; the hand body extends about 10 cm along the jaw on both sides of the grip and 3 cm across, and "
                "outward-shifted retries did not clear it: clear that space (move the object away, or open the "
                "obstruction less) or grasp from a freer side")
        heights = [(a.get("contact") or {}).get("contact_grip_z") for a in attempts]
        heights = [z for z in heights if z is not None]
        if result["reason"] == "empty_close" and heights:
            same = len(heights) >= 2 and max(heights) - min(heights) <= SAME_STALL_HEIGHT_M
            descended = [(a.get("descended_m") or 0.0) for a in attempts]
            result["detail"] = (
                f"the descent{'s' if len(heights) > 1 else ''} stopped at grip z "
                f"{', '.join(f'{z:.3f}' for z in heights)} after {', '.join(f'{d:.2f}' for d in descended)} m and the "
                "fingers closed on nothing" + (
                    "; the same height every time means the hand body is landing on something above the container "
                    "(it is about 20 cm long along the jaw): change the side, or clear what it hits" if same else
                    "; the fingers may have missed the wall"))
        return result

    def _grasp_bar(self, target, approach_name, pull_m=None):
        """Frontal bar pinch: horizontal approach, vertical jaw, cabinet-face stall (§7.2)."""
        context, calibration = self._context, self.calibration or GripperCalibration.default()
        approach = np.asarray(WORLD_AXES[approach_name], float)
        context.target_phrase = bar_target_phrase(target)
        context.pointing_context = bar_pointing_context(target, approach_name)
        view = context.view
        predicted = wrist_to_shoulder_m(view.position, approach)
        if predicted > FRONTAL_STRETCH_LIMIT_M:
            # drawer live 001: hovers at 0.788-0.804 trapped the arm at the elbow singularity for half the run
            return {"status": "failed", "reason": "hover_beyond_frontal_reach",
                    "evidence": {"wrist_to_shoulder_m": predicted, "limit_m": FRONTAL_STRETCH_LIMIT_M},
                    "detail": self._stretch_detail(predicted, view.position)}
        yield from self._ensure_open()
        view = context.view

        def frontal(v):
            # Vertical jaw, either sign: take the one needing the smaller rotation (the parallel gripper is
            # symmetric). At the top-down start R[:,1]·z is numerical noise, so a sign test would flip 90°/178°.
            options = [axes_to_quaternion_xyzw((0., 0., sign), approach) for sign in (1.0, -1.0)]
            return min(options, key=lambda q: quaternion_angle_rad(v.quaternion_xyzw, q))

        if quaternion_angle_rad(view.quaternion_xyzw, frontal(view)) > LARGE_TURN_RAD:
            yield (lambda v: AnchoredMove(v, axis=-v.R[:, 2], travel_m=LARGE_TURN_RETREAT_M, gripper="open",
                                          name="turn_retreat"))
            restored = yield from self._restore_start_orientation()
            if restored is not None and restored["status"] != "succeeded":
                return self._orient_failure(restored)
        frontal_q = frontal(context.view)
        outcome, _ = yield (lambda v: Orient(v, target_xyzw=frontal_q, name="orient", max_pose_drift_m=ORIENT_MAX_DRIFT_M))
        if outcome["status"] != "succeeded" and outcome["reason"] == "pose_drift":
            # drawer live 001 step 187: the turning fingers dragged along the wine rack and pressed on a bottle;
            # lift clear and retry once before reporting
            yield (lambda v: AnchoredMove(v, axis=(0., 0., 1.), travel_m=ORIENT_RETRY_LIFT_M, gripper="open",
                                          name="orient_lift"))
            outcome, _ = yield (lambda v: Orient(v, target_xyzw=frontal_q, name="orient",
                                                 max_pose_drift_m=ORIENT_MAX_DRIFT_M))
        if outcome["status"] != "succeeded":
            return self._orient_failure(outcome, "orient")

        yield (lambda v: HoldPrimitive(v, steps=POST_ORIENT_HOLD_STEPS, gripper="open", name="orient_settle"))
        attempts, reasons = [], []
        hover_track, jacobian, hover_xyz = None, None, None
        offset = GRIP_AXIS_OFFSET_M
        for attempt in range(self.local_retries + 1):
            self._counts["grasp_attempts"] += 1
            if attempt:
                self._counts["local_retries_used"] += 1
            evidence: dict = {"attempt": attempt + 1}
            track = copy.deepcopy(hover_track) if hover_track is not None else None
            # A bar is aligned only across its length: NCC cannot localize a point along a uniform bar
            # (drawer live run 001: probe_failed x7, servo stalls), and a pinch needs only the bar between
            # the fingers. The along-bar position is set once from the pointed pixel.
            reachable = lambda xyz: wrist_to_shoulder_m(xyz, approach) <= FRONTAL_STRETCH_LIMIT_M
            outcome, align = yield (lambda v, t=track, j=jacobian: Align(
                v, offset_m=offset, gripper="open", track=t, jacobian=j, reprobe=j is None, allow_pointing=True,
                name="align", axes="across_bar", along_target_ok=reachable))
            evidence["align"] = {"reason": outcome["reason"], "residual_px": outcome.get("evidence", {}).get("residual_px"),
                                 "residual_mm_est": outcome.get("evidence", {}).get("residual_mm_est"),
                                 "detail": outcome.get("evidence", {}).get("detail"),
                                 "along_record": outcome.get("evidence", {}).get("along_record"),
                                 "pointed_uv_upright": (outcome.get("evidence", {}).get("pointed_uv") or [None])[-1]}
            if outcome["status"] != "succeeded":
                reasons.append(outcome["reason"])
                attempts.append(evidence)
                shift = (evidence["align"].get("along_record") or {}).get("shift_m")
                along_goal = context.view.position + (shift or 0.0) * context.view.R[:, 0]
                if (outcome["reason"] in ("align_blocked", "approach_drift") and shift and abs(shift) > 0.010
                        and attempt < self.local_retries and reachable(along_goal)):
                    # From hovers past the bar's end the along-bar move inside Align yaws the hand and drifts it
                    # back (live-state replays 555/583). Make the measured along move as a plain reach (toward the
                    # robot-side half of the bar it also lowers the arm's stretch), then re-point and align.
                    move = [float(np.clip(x, -0.075, 0.075)) for x in shift * context.view.R[:, 0]]
                    reach, _ = yield (lambda v, m=move: WaypointReach(v, m, gripper="open", name="along_bar_reach"))
                    evidence["along_bar_reach"] = {"reason": reach["reason"], "offset_m": move,
                                                   "measured_displacement_m":
                                                       reach.get("evidence", {}).get("measured_displacement_m")}
                    hover_track = jacobian = None
                    continue
                break
            view = context.view
            hover_track, jacobian, hover_xyz = copy.deepcopy(align.track), align.jacobian, view.position.copy()
            # Descend along the measured camera axis (Orient leaves up to its 3 deg tolerance of tilt).
            approach = view.R[:, 2].copy()
            # Hold the measured orientation (not the exact frontal target): rotating toward the target while the
            # template is pinned along the bar made the track jump to the bar's edge (live-state replay t7).
            contact = yield from self._descend_ladder(align, offset, approach, calibration, evidence,
                                                      axes="across_bar", along_xyz=view.position.copy())
            if contact is None:
                reasons.append(evidence.get("approach_failure") or "no_contact_within_bound")
                attempts.append(evidence)
                yield from self._return_to_hover(hover_xyz, approach)
                break
            finger = contact.get("evidence", {}).get("finger_deflected")
            # A single-finger deflection is recorded but not treated as "bar under a finger": on real LIBERO
            # (drawer smoke 003) a fingertip deflected when the open fingers met the cabinet face with the bar
            # correctly between them. A real miss shows up as an empty close or a failed hold test.
            evidence["contact"] = {"reason": contact["reason"], "finger_deflected": finger}
            yield (lambda v: AnchoredMove(v, axis=-approach, travel_m=BAR_CLEARANCE_M, gripper="open", name="clearance"))
            result = yield from self._grasp_attempt_close(calibration, tuple(-approach), PULL_TEST_M, evidence)
            reasons.append(result)
            attempts.append(evidence)
            if result == "hold_pass":
                self._holding, self._held_strategy = True, "frontal_bar_pinch"
                self._pulled_m = PULL_TEST_M
                self._note_grasp(target)
                grasped = {"attempts": len(attempts), "attempt_reasons": reasons, "details": attempts}
                if pull_m is None:
                    return {"status": "succeeded", "reason": "hold_pass", "evidence": grasped}
                # grasp then pull in one decision (speed: drawer live 002 spent a decision on the pull)
                pulled = yield from self._pull(float(pull_m))
                return {"status": pulled["status"], "reason": f"hold_pass_then_{pulled['reason']}",
                        "evidence": {**grasped, "pull": pulled, "pulled_total_m": self._pulled_m},
                        "detail": f"grasped the handle and pulled {self._pulled_m:.3f} m in total ({pulled['reason']})"}
            yield from self._return_to_hover(hover_xyz, approach)
            hover_track = None
        result = {"status": "failed", "reason": reasons[-1] if reasons else "no_attempt",
                  "evidence": {"attempts": len(attempts), "attempt_reasons": reasons, "details": attempts}}
        if result["reason"] in ("mark_not_visible", "pointed_not_a_bar", "textureless_mark"):
            down = _vec(-context.view.R[:, 0], 2)
            result["detail"] = (
                "the handle bar was not found in the wrist view. If it is near the fingertips it is hidden behind "
                f"them: move toward wrist-image-down (world {down}) by 3-5 cm; if it is outside the view, re-hover "
                "level with and 6-10 cm in front of the handle (judge it in the external image), then try again")
        shift = ((attempts[-1].get("align") or {}).get("along_record") or {}).get("shift_m") if attempts else None
        if result["reason"] == "bar_point_beyond_reach" and shift:
            goal = context.view.position + shift * context.view.R[:, 0]
            result["detail"] = (f"the pointed handle point is {abs(shift):.3f} m along the bar, beyond the arm's reach "
                                "for a turned hand; " + self._stretch_detail(
                                    wrist_to_shoulder_m(goal, approach), context.view.position).split(": ", 1)[1])
        elif result["reason"] in ("align_blocked", "approach_drift", "bar_point_far_along") and shift:
            # drawer live-state replays u2/u9: from hovers past the bar's end the along-bar move yaws the hand
            toward = _vec(np.sign(shift) * context.view.R[:, 0], 2)
            result["detail"] = (
                f"the pointed handle point is {abs(shift):.3f} m along the bar toward world {toward} from the grip "
                "axis and the hand could not get there while aligning; reach that way first (same height), then "
                "call grasp_with again")
        elif result["reason"] == "hover_off_bar_across":
            up = _vec(context.view.R[:, 1], 2)
            result["detail"] = (
                "the hover is more than 4 cm across the bar from the grip axis; re-hover level with the handle "
                f"(wrist-image right is world {up}) and 6-10 cm in front of it")
        return result
