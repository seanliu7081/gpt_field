"""Operation-directed local control, opt-in over the unchanged v3 baseline.

The model chooses every operation, goal and verification vector. Local code
executes only that finite intent, maintains observable evidence, and returns
on arrival, invalidity, stagnation or expiry. Neither tracking nor a settled
gripper certifies grasp or benchmark completion.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from .action import _rotvec_quaternion
from .astra import SchemaError, _canonical
from .direct_history import _allowed
from .fine_distance_planner import FineCompactField, FineDistancePlanner
from .hybrid_subscription import HYBRID_KEYS
from .operation_contract import (OPERATION_VERSION, OperationEvidenceLedger, legacy_context,
    operation_instructions, operation_schema, validate_operation,
    validate_visual_intent, visual_intent_schema)
from .operation_visual import OperationVisualFeedback
from .planner import CandidateRecord, PlanResult
from .policy import Decision
from .speed_first_v3 import (ChatGPTPlanSpeedFirstV3Client, FINE_GRID,
    SpeedFirstV3Policy, speed_first_v3_schema)


SPEED_FIRST_V4_VERSION = OPERATION_VERSION


def _planner_mode(value):
    if value not in ("field", "waypoint"):
        raise ValueError("planner_mode must be field or waypoint")
    return value


def speed_first_v4_schema(task_key="bowl_plate"):
    schema = speed_first_v3_schema(task_key, attempt_memory=True, pregrasp_feedback=True)
    schema["required"].remove("control_context")
    del schema["properties"]["control_context"]
    schema["properties"].update(operation=operation_schema(), visual_intent=visual_intent_schema())
    schema["required"].extend(("operation", "visual_intent"))
    return schema


def speed_first_v4_instructions(task_key="bowl_plate", image_history=2, *,
                               robot_geometry_guidance=False, target_description=None,
                               planner_mode="field"):
    mode = _planner_mode(planner_mode)
    text = operation_instructions(task_key, image_history,
        robot_geometry_guidance=robot_geometry_guidance,
        target_description=target_description, attempt_memory=True)
    text += (
        "\nExecution contract: every ordinary Direct response has local_action_budget=5 "
        "and allow_local_continuation=false. A Field operation may request 1 to 30 actions; "
        "a budget above 30, at most 60, additionally requires allow_local_continuation=true. "
        "At action 30 continuation is permitted only with measured progress toward the "
        "same nominal goal and recent forward progress; this never creates a new waypoint. "
        "Explicit gripper-only open/close uses budget=30 and allow_local_continuation=false. "
        "For grasp_close with an explicit verify_motion use budget=60 and "
        "allow_local_continuation=true: close at the current pose first, then ONLY if measured "
        "finger motion settles, execute the supplied <=3 cm world XYZ offset from the fresh "
        "post-closure measured pose with the same orientation and closed gripper, for at most "
        "30 further actions. This does not assert a grasp. If fingers do not settle, request "
        "a new decision without the verification movement. No automatic release follows. "
        "Only LIBERO success ends the episode. Both optional visual correction engines are "
        "limited to 6 mm TOTAL from the immutable model-selected nominal goal. Failed visual "
        "feedback restores the nominal goal and continues normal control. No repeated probe "
        "or wait action is inserted. current_operation_feedback and operation_evidence are "
        "dated local measurements/claims, never hidden object state or benchmark feedback.")
    if mode == "waypoint":
        text += (
            "\nPLANNER ABLATION: waypoint mode follows the same selected goal using only a "
            "straight local path and the same tracking, limits, feedback and predicted "
            "validity checks. It does not minimize field density or search a detour. A "
            "blocked straight segment returns to you; repulsor costs do not alter the line.")
    else:
        text += "\nPLANNER MODE: field; retain lazy 1 mm weighted distance planning and local feedback tracking."
    return SPEED_FIRST_V4_VERSION + ". " + text


class ChatGPTPlanSpeedFirstV4Client(ChatGPTPlanSpeedFirstV3Client):
    speed_first_v4 = True

    def __init__(self, config=None, *, planner_mode="field", pregrasp_feedback=True,
                 attempt_memory=True, **kwargs):
        if pregrasp_feedback is not True or attempt_memory is not True:
            raise ValueError("V4 requires pregrasp_feedback=True and attempt_memory=True")
        self.planner_mode = _planner_mode(planner_mode)
        super().__init__(config, pregrasp_feedback=True, attempt_memory=True, **kwargs)

    @property
    def prompt_version(self):
        return SPEED_FIRST_V4_VERSION + "+" + self.planner_mode

    def reset_history(self):
        super().reset_history()
        self.operation_ledger = OperationEvidenceLedger()
        self.latest_operation_feedback = None

    def _reset_prediction_diagnostics(self):
        super()._reset_prediction_diagnostics()
        self.last_operation = self.last_visual_intent = None

    def build_payload(self, observation, grid=None, mode="hybrid"):
        with self._history_lock:
            payload = super().build_payload(observation, grid, mode)
            payload["instructions"] = speed_first_v4_instructions(self.task_spec.key, self.image_history,
                robot_geometry_guidance=self.robot_geometry_guidance,
                target_description=self.target_description, planner_mode=self.planner_mode)
            payload["text"]["format"] = {"type": "json_schema", "strict": True,
                "name": "astra_speed_first_operation_hybrid_v4",
                "schema": speed_first_v4_schema(self.task_spec.key)}
            allowed = _allowed(observation, self.task_prompt)
            current = {"type": "current_operation_feedback",
                "operation_evidence": self.operation_ledger.presentation(
                    self._observation_count, current_state=allowed.state),
                "execution": copy.deepcopy(self.latest_operation_feedback)}
            # Parent's pending complete source transcript is already captured;
            # these current snapshots never recursively enter accepted history.
            payload["input"].append({"role": "user", "content": [
                {"type": "input_text", "text": _canonical(current)}]})
            return payload

    def validate_response(self, data, grid):
        required = HYBRID_KEYS | {"gripper_operation", "interaction_review", "operation", "visual_intent"}
        if grid != FINE_GRID or not isinstance(data, dict) or set(data) != required:
            raise SchemaError("V4 requires the complete typed operation wrapper without control_context")
        raw = copy.deepcopy(data)
        try:
            operation = validate_operation(raw["operation"])
            visual = validate_visual_intent(raw["visual_intent"], operation)
        except (ValueError, TypeError) as exc:
            raise SchemaError(str(exc)) from exc
        if operation["target_id"] != raw["target_region"]:
            raise SchemaError("operation.target_id must match target_region")
        kind, budget = operation["kind"], operation["local_action_budget"]
        gripper = raw["gripper_operation"]
        if kind in ("grasp_close", "release"):
            command = "close" if kind == "grasp_close" else "open"
            if gripper != {"command": command}:
                raise SchemaError("grasp_close/release requires the matching explicit gripper operation")
            verify = operation["verify_motion"] is not None
            if budget != (60 if verify else 30) or operation["allow_local_continuation"] != verify:
                raise SchemaError("gripper operation budget must be 30, or explicitly authorized close+verify 60")
        elif gripper is not None:
            raise SchemaError("Only grasp_close/release may request gripper_operation")
        elif raw["controller"] == "contact_direct":
            if budget != 5 or operation["allow_local_continuation"]:
                raise SchemaError("Ordinary Direct remains exactly five actions without continuation")
        elif raw["controller"] == "coarse_field":
            if kind == "direct":
                raise SchemaError("operation.kind=direct requires contact_direct")
            if budget > 30 and not operation["allow_local_continuation"]:
                raise SchemaError("Field continuation beyond 30 requires explicit permission")
        # Applicability belongs to the operation, including when its optional
        # visual reference is unavailable. Null must not erase the hand intent.
        if kind in ("pregrasp", "place_align"):
            field = raw.get("field")
            direct = raw.get("direct")
            if raw["controller"] == "coarse_field" and isinstance(field, dict):
                if kind == "pregrasp" and field.get("gripper_command") != "open":
                    raise SchemaError("Pregrasp operation requires explicit open-gripper Field")
                if kind == "place_align" and field.get("gripper_command") == "open":
                    raise SchemaError("Held-object place_align cannot open the gripper")
            elif raw["controller"] == "contact_direct" and isinstance(direct, dict):
                actions = direct.get("actions")
                if isinstance(actions, list) and all(isinstance(a, list) and len(a) == 7 for a in actions):
                    grips = [a[6] for a in actions]
                    if all(type(g) in (int, float) for g in grips):
                        if kind == "pregrasp" and any(g >= 0 for g in grips):
                            raise SchemaError("Pregrasp operation requires open commands in all Direct actions")
                        if kind == "place_align" and any(g < 0 for g in grips):
                            raise SchemaError("Held-object place_align cannot execute open commands")
        # Cross-turn operation identity is also part of atomic response
        # validation, never an exception after accepted model history commits.
        try:
            ledger_candidate = copy.deepcopy(self.operation_ledger)
            ledger_candidate.accept(operation, visual, self._observation_count)
        except (ValueError, TypeError) as exc:
            raise SchemaError(str(exc)) from exc
        adapted = copy.deepcopy(raw)
        del adapted["operation"], adapted["visual_intent"]
        context = legacy_context(operation, visual, task_key=self.task_spec.key)
        if context["phase"] not in self.task_spec.control_phases:
            context["phase"] = "approach"  # implementation-only label; never model semantics
        adapted["control_context"] = context
        # This last validation mutates diagnostics only after all fields validate.
        result = super().validate_response(adapted, grid)
        self.last_operation = copy.deepcopy(operation)
        self.last_visual_intent = copy.deepcopy(visual)
        self.last_compact_prediction = raw
        return result

    def _feedback_action_limit(self, controller):
        if self.last_operation is not None:
            return self.last_operation["local_action_budget"]
        return super()._feedback_action_limit(controller)


class WaypointAblationPlanner(FineDistancePlanner):
    """Straight goal tracking with identical bounds/masks, no field-cost search."""

    def plan_analytic(self, field, position, *, goal_xyz, compact_prediction, goal_tolerance_m=.005):
        start, goal = np.asarray(position, dtype=float), np.asarray(goal_xyz, dtype=float)
        tolerance = float(goal_tolerance_m) / 2.
        distance = float(np.linalg.norm(goal - start))
        search = dict(method="waypoint_straight_line_ablation", uses_field_cost=False,
            searches_detours=False, goal_xyz=goal.tolist(), goal_distance_m=distance,
            goal_tolerance_m=tolerance, goal_reached=False, cache_hit=False,
            candidate_semantics="one_straight_goal_path", distance_semantics="euclidean_goal_error",
            segment_count=0, task_success_certified=False)
        def stop(status, reason):
            return PlanResult((0., 0., 0.), status, 0, 0, reason=reason, search_metadata=search)
        if field.prediction.motion_mode == "hold":
            return stop("hold", "explicit_motion_hold")
        boxes = self._boxes(field, compact_prediction)
        if not self._valid_segment(field, start, start, boxes):
            return stop("no_valid_candidates", "invalid_start")
        if np.any(goal < field.lower) or np.any(goal > field.upper):
            return stop("needs_observation", "goal_outside_known_grid")
        if np.any(goal < self.config.workspace_min) or np.any(goal > self.config.workspace_max):
            return stop("no_valid_candidates", "goal_outside_workspace")
        if distance <= tolerance:
            search["goal_reached"] = True
            return stop("goal_reached", "measured_goal_within_tolerance")
        length = min(distance - tolerance, self.config.horizon * self.config.step_m)
        count = max(1, math.ceil(length / self.config.step_m))
        arc = np.minimum(np.arange(count + 1) * self.config.step_m, length)
        points = start + arc[:, None] * (goal - start) / distance
        if not all(self._valid_segment(field, a, b, boxes) for a, b in zip(points[:-1], points[1:])):
            return stop("no_valid_candidates", "straight_path_blocked_no_detour_search")
        candidate = CandidateRecord(("waypoint_line",) * count,
            tuple(tuple(float(v) for v in point) for point in points), length, True,
            None, None)
        search.update(segment_count=count, endpoint_xyz=points[-1].tolist(),
            endpoint_in_goal_region=distance - length <= tolerance + 1e-12)
        return PlanResult(tuple(points[1] - start), "move", 1, 1, (candidate,), 0,
                          search_metadata=search)


class SpeedFirstV4Policy(SpeedFirstV3Policy):
    def __init__(self, client, planner_mode="field"):
        if not getattr(client, "speed_first_v4", False):
            raise ValueError("V4 policy requires its typed operation client")
        self.planner_mode = _planner_mode(planner_mode)
        if self.planner_mode != client.planner_mode:
            raise ValueError("Client and policy planner_mode must match")
        super().__init__(client)
        if self.planner_mode == "waypoint":
            self._field_policy.planner = WaypointAblationPlanner(self.planner.config)
        self.method = SPEED_FIRST_V4_VERSION

    def reset(self):
        super().reset()
        self.operation = None
        self.outcome = OperationVisualFeedback()
        self.outcome_feedback = {"status": "not_configured", "verification_evidence": "unknown",
                                 "task_success_certified": False}
        self._operation_start = 0
        self._operation_positions = []
        self._verify_pending = False
        self._verification_active = False
        self._operation_last_event = None
        self._operation_visual_since_act = 0.
        self._operation_stats = dict(operations=0, continuation_requests=0, continuations_granted=0,
            continuations_denied=0, continuation_actions=0, close_verify_requested=0,
            close_verify_started=0, close_verify_skipped_unsettled=0, verification_actions=0,
            ordinary_direct_actions=0, field_actions=0, gripper_actions=0,
            visual_seconds=0., visual_errors=0, operation_seconds=0.)

    def _outcome_call(self, function):
        began = time.monotonic()
        try:
            value = function()
            if not isinstance(value, dict):
                value = self.outcome.snapshot()
            if not isinstance(value, dict):
                raise TypeError("Operation visual snapshot must be a dictionary")
        except Exception as exc:
            self._operation_stats["visual_errors"] += 1
            value = {"status": "local_feedback_error", "error_type": type(exc).__name__,
                "verification_evidence": "unknown", "task_success_certified": False,
                "fallback": "continue_explicit_operation"}
        elapsed = time.monotonic() - began
        self._operation_stats["visual_seconds"] += elapsed
        self.local_visual_total_s += elapsed
        self._operation_visual_since_act += elapsed
        self.outcome_feedback = copy.deepcopy(value)
        return value

    def operation_stats(self):
        return {"version": SPEED_FIRST_V4_VERSION, "planner_mode": self.planner_mode,
            **copy.deepcopy(self._operation_stats), "current": self.current_operation_feedback()}

    def current_operation_feedback(self):
        return {"operation": copy.deepcopy(self.operation),
            "actions_executed": max(0, self.step_index - self._operation_start - int(self._pending is not None)),
            "verification_active": self._verification_active,
            "last_event": copy.deepcopy(self._operation_last_event),
            "visual_evidence": copy.deepcopy(self.outcome_feedback),
            "grasp_certified": False, "task_success_certified": False}

    def _validate_envelope(self, value):
        result = super()._validate_envelope(value)
        operation = copy.deepcopy(self.client.last_operation)
        visual = copy.deepcopy(self.client.last_visual_intent)
        self.client.operation_ledger.accept(operation, visual, self.client._observation_count - 1,
                                           state8=self._current_observation.state)
        self.operation = operation
        self._operation_start = self.step_index
        self._operation_positions = [np.asarray(self._current_observation.state[:3], dtype=float)]
        self._verification_active = self._verify_pending = False
        self._operation_last_event = None
        self._operation_stats["operations"] += 1
        self._operation_stats["continuation_requests"] += int(
            value["controller"] == "coarse_field" and operation["local_action_budget"] > 30)
        self._operation_stats["close_verify_requested"] += int(operation["verify_motion"] is not None)
        self.field_update_steps = min(30, operation["local_action_budget"])
        self._field_policy.update_steps = self.field_update_steps
        self._outcome_call(lambda: self.outcome.configure(self._current_observation, operation, visual))
        if operation["kind"] == "verify_grasp":
            self._outcome_call(lambda: self.outcome.begin_verification(self._current_observation))
        return result

    def _local_alignment_active(self):
        return bool(self.operation and self.operation["kind"] == "place_align"
                    and super()._local_alignment_active())

    def _pregrasp_alignment_active(self):
        return bool(self.operation and self.operation["kind"] == "pregrasp"
                    and super()._pregrasp_alignment_active())

    def _maybe_update_goal(self, observation):
        if self.pregrasp is not None and self._pregrasp_requested:
            return super()._maybe_update_goal(observation)
        if not self.operation or self.operation["kind"] != "place_align":
            return
        # Reuse the v3 estimator but enforce one total 6 mm neighborhood and
        # restore nominal on loss. Never clip a proposal into apparent validity.
        original = self.visual.corrected_goal
        def bounded(obs, nominal):
            try:
                goal, diagnostic = original(obs, nominal)
            except Exception as exc:
                return None, {"status": "local_feedback_error", "error_type": type(exc).__name__,
                    "calibration_valid": False, "proposal_available": False,
                    "task_success_certified": False}
            if goal is not None and (np.asarray(goal).shape != (3,)
                    or not np.isfinite(np.asarray(goal, dtype=float)).all()
                    or np.linalg.norm(np.asarray(goal) - nominal) > .006 + 1e-12):
                return None, {**diagnostic, "proposal_available": False,
                    "correction_rejected_reason": "outside_nominal_6mm_neighborhood"}
            return goal, diagnostic
        self.visual.corrected_goal = bounded
        try:
            super()._maybe_update_goal(observation)
        finally:
            self.visual.corrected_goal = original
        if (self._local_goal_active and (not self.local_feedback.get("calibration_valid")
                or not self.local_feedback.get("proposal_available")
                or self.local_feedback.get("correction_rejected_reason"))):
            self._set_pregrasp_goal(self._nominal_goal, restoring=True)
            self._local_goal_active = False
            self.local_feedback = {**self.local_feedback, "nominal_goal_restored": True,
                                  "fallback": "resume_model_nominal_goal"}

    def _event(self, reason):
        if (reason == "field_budget_exhausted" and self.operation
                and self.controller == "coarse_field" and not self._verification_active
                and self.step_index - self._operation_start == 30
                and self.operation["local_action_budget"] > 30):
            goal = self._nominal_goal
            positions = self._operation_positions
            progress = (float(np.linalg.norm(positions[0] - goal) - np.linalg.norm(positions[-1] - goal))
                        if goal is not None and len(positions) > 1 else 0.)
            recent = (float(np.linalg.norm(positions[-6] - goal) - np.linalg.norm(positions[-1] - goal))
                      if goal is not None and len(positions) >= 6 else 0.)
            allowed = (self.operation["allow_local_continuation"] and progress >= .0025 and recent >= .0005
                       and self._field_cache.effective_compact_field["motion_mode"] == "move")
            self._operation_last_event = dict(reason="local_continuation_granted" if allowed else
                "local_continuation_denied", step=self.step_index, measured_goal_progress_m=progress,
                recent_measured_goal_progress_m=recent, task_success_certified=False)
            if allowed:
                self._chunk_length = self.operation["local_action_budget"]
                self._field_policy.update_steps = self._chunk_length
                self._field_policy.clear_pending_plan()
                self._operation_stats["continuations_granted"] += 1
                return
            self._operation_stats["continuations_denied"] += 1
        super()._event(reason)
        if self._field_refresh_event is not None:
            self._operation_last_event = {"reason": reason, "step": self.step_index,
                                         "task_success_certified": False}

    def _begin_verification(self, observation):
        """Execute only the already accepted vector, based on fresh measured pose."""
        motion = self.operation["verify_motion"]
        remaining = self.operation["local_action_budget"] - (self.step_index - self._operation_start)
        if motion is None or remaining < 1:
            raise RuntimeError("No finite model-authorized verification motion remains")
        compact = dict(stage="explicit_close_verification", motion_mode="move", gripper_command="close",
            orientation_target_xyzw=_rotvec_quaternion(np.asarray(observation.state[3:6])).tolist(),
            goal_offset_m=list(motion["offset_xyz_m"]), attraction_weights=[1., 1., 1.],
            repulsors=[], blocked_boxes=[], unknown_boxes=[])
        self._verify_pending = False
        self._verification_active = True
        self.gripper_operation = None
        self._field_refresh_event = None
        self.controller = "coarse_field"
        self._chunk_start = self.step_index
        self._chunk_length = min(30, remaining)
        self._field_policy.update_steps = self._chunk_length
        self._field_policy.clear_pending_plan()
        self.planner.reset_cache()
        self._field_motion_window = []
        self._field_cache.prepare(observation, compact, compact)
        self.prediction = dict(controller="coarse_field", target_region=self.operation["target_id"],
            near_target=False, routing_reason="Explicit model-selected close verification vector",
            field=copy.deepcopy(compact), direct=None)
        self.routing = {key: self.prediction[key] for key in
            ("controller", "target_region", "near_target", "routing_reason")}
        self.control_context = {"phase": "verify", "visual_reference": None}
        self._nominal_goal = np.asarray(observation.state[:3]) + np.asarray(motion["offset_xyz_m"])
        self._pregrasp_requested = self._pregrasp_goal_active = self._local_goal_active = False
        self._operation_stats["close_verify_started"] += 1
        self._outcome_call(lambda: self.outcome.begin_verification(observation))

    def act(self, observation):
        # Validate the same feedback boundary before changing an execution phase.
        allowed = _allowed(observation.to_dict(), self.client.task_prompt)
        if self._pending is not None:
            raise RuntimeError("observe_transition is required before the next action")
        if self._expected_observation is not None and allowed != self._expected_observation:
            raise ValueError("Next observation must equal the preceding transition's after observation")
        if self._verify_pending:
            self._begin_verification(allowed)
        self.client.latest_operation_feedback = self.current_operation_feedback()
        began = time.monotonic()
        decision = super().act(allowed)
        details = dict(decision.diagnostics)
        if self._verification_active:
            details["execution_kind"] = "local_close_verification"
            details["chunk_step"] = self.step_index - self._operation_start - 1
            details["hybrid_prediction"] = copy.deepcopy(self.client.last_compact_prediction)
        details.update(speed_first_v4=True, operation=copy.deepcopy(self.operation),
            operation_feedback=self.current_operation_feedback(), planner_mode=self.planner_mode,
            operation_visual_latency_s=self._operation_visual_since_act,
            operation_diagnostics=self.operation_stats())
        self._operation_visual_since_act = 0.
        # Timer excludes the already independently recorded model pipeline time.
        self._operation_stats["operation_seconds"] += max(0., time.monotonic() - began
            - (details.get("api_latency_s") or 0.))
        decision = Decision(decision.action, details)
        self._pending = (self._pending[0], decision)
        return decision

    def observe_transition(self, before, decision, after):
        before = _allowed(before.to_dict(), self.client.task_prompt)
        after = _allowed(after.to_dict(), self.client.task_prompt)
        if self._pending is None or before != self._pending[0] or decision is not self._pending[1]:
            raise ValueError("Feedback must correspond to the exact pending observation and decision")
        self._operation_positions.append(np.asarray(after.state[:3], dtype=float))
        self._outcome_call(lambda: self.outcome.observe(before, after))
        feedback = super().observe_transition(before, decision, after)
        self.client.operation_ledger.record_feedback(before.state, after.state, self.step_index - 1,
                                                     visual_diagnostics=self.outcome_feedback)
        kind = decision.diagnostics["execution_kind"]
        counter = ("verification_actions" if kind == "local_close_verification" else
                   "gripper_actions" if kind == "local_gripper_settle" else
                   "field_actions" if kind == "field_tracking" else "ordinary_direct_actions")
        self._operation_stats[counter] += 1
        if (self.controller == "coarse_field" and not self._verification_active
                and self.step_index - self._operation_start > 30):
            self._operation_stats["continuation_actions"] += 1
        if (self.gripper_operation is not None and self.gripper_operation.done
                and self.operation["verify_motion"] is not None):
            if self.gripper_operation.stop_reason == "measured_fingers_stable":
                self._verify_pending = True
                self._operation_last_event = {"reason": "explicit_verification_ready", "step": self.step_index}
            else:
                self._operation_stats["close_verify_skipped_unsettled"] += 1
                self._operation_last_event = {"reason": "verification_skipped_fingers_unsettled",
                                             "step": self.step_index}
        self.client.latest_operation_feedback = self.current_operation_feedback()
        return feedback
