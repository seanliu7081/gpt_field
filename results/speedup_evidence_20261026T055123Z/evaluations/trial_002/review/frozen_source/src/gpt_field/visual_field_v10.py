"""Persistent local world goals and finite, useful Field execution.

A local goal is a model proposal, not a task-success predicate. Geometry,
orientation holding and measured-motion bounds remain those of V9/V8. Neither
small commands nor reference repair can silently advance or replace a goal.
"""
from __future__ import annotations

import copy
import numpy as np

from .visual_field_v4 import _number, _vector
from .visual_field_v6 import LOCKED_AXIS_TOLERANCE_M, _AxisConstrainedField, _LOCKED_AXES, validate_metric_intent_v6
from .visual_field_v9 import MetricFieldExecutorV9


MIN_USEFUL_COMMAND_M = .001


class SubgoalLedger:
    """Track one explicitly identified model goal across local controllers.

    ``apply(state8, prediction)`` returns a rewritten intent and executor
    context. A rejected update never replaces the stored goal. ``observe``
    accepts robot state only and latches local arrival; it never asserts task
    completion. ``none`` preserves existing state, while ``reset`` is reserved
    for an explicit new task/episode.
    """

    def __init__(self, task_id=None):
        self.reset(task_id)

    def reset(self, task_id=None):
        self.task_id = task_id
        self.active = None
        self.last_update = None
        self.completed = []
        self.revision_count = 0
        self._last_xyz = None
        return self.summary()

    @staticmethod
    def _active_axes(axes):
        return [i for i in range(3) if i not in _LOCKED_AXES[axes]]

    def observe(self, state8):
        xyz = _vector(state8, 8, "state8")[:3]
        if self.active is not None:
            item = self.active
            axes, locked = self._active_axes(item["motion_axes"]), list(_LOCKED_AXES[item["motion_axes"]])
            goal, origin = np.asarray(item["world_goal_xyz"]), np.asarray(item["origin_xyz"])
            error = float(np.linalg.norm((xyz-goal)[axes]))
            corridor = bool(np.all(abs((xyz-origin)[locked]) <= LOCKED_AXIS_TOLERANCE_M+1e-12))
            item["measured_active_axis_error_m"] = error
            item["locked_axis_drift_m"] = float(np.max(abs((xyz-origin)[locked]), initial=0.))
            if not item["local_complete"] and corridor:
                arrival = error <= item["goal_tolerance_m"]
                reason = "measured_local_goal_arrival" if arrival else None
                if not arrival and self._last_xyz is not None:
                    previous = self._last_xyz
                    previous_inside = np.all(abs((previous-origin)[locked]) <= LOCKED_AXIS_TOLERANCE_M+1e-12)
                    delta = (xyz-previous)[axes]
                    fraction = float(np.clip(np.dot((goal-previous)[axes], delta)/max(1e-30, np.dot(delta, delta)), 0., 1.))
                    closest = previous[axes]+fraction*delta
                    if previous_inside and np.linalg.norm(closest-goal[axes]) <= item["goal_tolerance_m"]:
                        arrival, reason = True, "measured_transition_crossed_local_goal_region"
                if arrival:
                    item["local_complete"], item["completion_reason"] = True, reason
            item["last_measured_xyz"] = xyz.tolist()
        self._last_xyz = xyz.copy()
        return self.summary()

    def _result(self, accepted, status, *, intent=None, proposed=None, request=None):
        self.last_update = {"accepted": bool(accepted), "status": status,
                            "request": copy.deepcopy(request), "proposed_world_goal_xyz": proposed,
                            "stored_world_goal_xyz": None if self.active is None else self.active["world_goal_xyz"].copy()}
        return {"accepted": bool(accepted), "status": status,
                "field_intent": copy.deepcopy(intent),
                "executor_context": ({"subgoal_origin_xyz": self.active["origin_xyz"].copy()}
                                     if intent is not None and self.active is not None else {}),
                "feedback": self.summary()}

    def apply(self, state8, prediction):
        state = _vector(state8, 8, "state8")
        if not isinstance(prediction, dict):
            raise ValueError("prediction must be an object")
        request = prediction.get("subgoal_update")
        if not isinstance(request, dict) or set(request) != {"action", "subgoal_id", "reason"}:
            raise ValueError("subgoal_update requires action, subgoal_id and reason")
        action, ident, reason = request["action"], request["subgoal_id"], request["reason"]
        if action not in ("new", "retain", "replace", "finish", "none") or not isinstance(reason, str):
            raise ValueError("invalid subgoal update")
        if action == "none":
            if ident is not None:
                raise ValueError("none requires null subgoal_id")
        elif not isinstance(ident, str) or not ident.strip():
            raise ValueError("subgoal update requires a nonempty ID")
        if action == "replace" and not reason.strip():
            raise ValueError("replace requires an explicit reason")
        controller = prediction.get("controller")
        if controller not in ("visual_field", "direct", "gripper"):
            raise ValueError("invalid controller for subgoal update")
        field = controller == "visual_field"
        if field and action not in ("new", "retain", "replace"):
            raise ValueError("Field requires new, retain or replace")
        if not field and action in ("new", "replace"):
            raise ValueError("Only Field creates or replaces a metric subgoal")
        checked = validate_metric_intent_v6(prediction.get("field_intent")) if field else None
        proposed = None if checked is None else (state[:3]+checked["goal_offset_m"]).tolist()
        self.observe(state)
        reject = lambda status: self._result(False, status, proposed=proposed, request=request)
        if action == "none":
            return self._result(True, "subgoal_preserved_without_update", request=request)
        if action in ("retain", "replace", "finish"):
            if self.active is None:
                return reject("subgoal_missing")
            if ident != self.active["subgoal_id"]:
                return reject("subgoal_identity_mismatch")
        if action == "finish":
            if not self.active["local_complete"]:
                return reject("subgoal_finish_before_measured_arrival")
            self.completed.append(copy.deepcopy(self.active))
            self.active = None
            return self._result(True, "subgoal_finished_locally", request=request)
        if action == "retain" and not field:
            return self._result(True, "subgoal_retained_during_local_control", request=request)
        if action == "new" and self.active is not None and not self.active["local_complete"]:
            return reject("subgoal_active_requires_retain_or_explicit_replace")
        if action in ("new", "replace"):
            phase, target = prediction.get("phase"), prediction.get("target_id")
            if any(not isinstance(v, str) or not v.strip() for v in (phase, target)):
                raise ValueError("Field subgoal requires phase and target_id")
            if action == "new" and self.active is not None and ident == self.active["subgoal_id"]:
                return reject("new_subgoal_requires_new_identity")
            if self.active is not None:
                previous = copy.deepcopy(self.active)
                previous["superseded_reason"] = reason if action == "replace" else "new_goal_after_local_arrival"
                self.completed.append(previous)
            self.revision_count += int(action == "replace")
            self.active = {"subgoal_id": ident, "phase": phase, "target_id": target,
                           "motion_axes": checked["motion_axes"], "origin_xyz": state[:3].tolist(),
                           "world_goal_xyz": proposed.copy(), "model_world_goal_xyz": proposed.copy(),
                           "visual_correction_count": 0, "visual_correction_travel_m": 0.,
                           "last_visual_correction": None,
                           "goal_tolerance_m": checked["goal_tolerance_m"],
                           "local_complete": False, "completion_reason": None,
                           "revision_reason": reason, "task_success_certified": False}
            self._last_xyz = state[:3].copy()
            self.observe(state)
        else:
            item = self.active
            if any(prediction.get(k) != item[k] for k in ("phase", "target_id")) or checked["motion_axes"] != item["motion_axes"]:
                return reject("subgoal_semantics_changed_requires_explicit_replace")
        item = self.active
        locked = list(_LOCKED_AXES[item["motion_axes"]])
        if np.any(abs((state[:3]-item["origin_xyz"])[locked]) > LOCKED_AXIS_TOLERANCE_M+1e-12):
            return reject("subgoal_locked_axis_drift_requires_replan")
        if item["local_complete"]:
            return reject("subgoal_local_complete")
        offset = np.asarray(item["world_goal_xyz"])-state[:3]
        # Inactive axes hold the measured resume pose, while the executor also
        # preserves the ORIGINAL corridor. Never command a forbidden component.
        offset[locked] = 0.
        if np.any(abs(offset) > .08+1e-12):
            return reject("subgoal_outside_local_grid_requires_replan")
        resumed = copy.deepcopy(checked)
        # Remove sub-picometer subtraction roundoff at the fixed grid boundary;
        # no geometric margin or out-of-grid goal is clipped.
        resumed["goal_offset_m"] = np.round(offset, 12).tolist()
        resumed["goal_tolerance_m"] = item["goal_tolerance_m"]
        return self._result(True, "subgoal_"+action, intent=resumed, proposed=proposed, request=request)

    def record_visual_correction(self, goal_xyz, *, reason="validated_local_visual_correction"):
        """Record a separately validated local correction after retarget succeeds.

        This method does not validate RGB itself. The caller must have already
        obtained an observed correction and a successful bounded ``retarget``.
        Failed ledger recording requires discarding that unsynchronized plan.
        Original locked axes and the model goal are retained for auditing.
        """
        if self.active is None or self.active["local_complete"]:
            return False
        try:
            proposed = _vector(goal_xyz, 3, "goal_xyz")
        except ValueError:
            return False
        if not isinstance(reason, str) or not reason.strip():
            return False
        item = self.active
        active, locked = self._active_axes(item["motion_axes"]), list(_LOCKED_AXES[item["motion_axes"]])
        old, model = np.asarray(item["world_goal_xyz"]), np.asarray(item["model_world_goal_xyz"])
        original = np.asarray(item["origin_xyz"])
        # The actual executor goal holds the current inactive coordinate after
        # repair. Accept that representation, but never write it into the
        # original stored corridor or treat it as a visual axis correction.
        held = self._last_xyz if self._last_xyz is not None else original
        if np.any(abs((proposed-original)[locked]) > LOCKED_AXIS_TOLERANCE_M+1e-12):
            return False
        if (not np.allclose(proposed[locked], held[locked], atol=1e-12, rtol=0.)
                and not np.allclose(proposed[locked], old[locked], atol=1e-12, rtol=0.)):
            return False
        corrected = old.copy(); corrected[active] = proposed[active]
        shift = float(np.linalg.norm(corrected-old))
        if (shift > .003+1e-12 or item["visual_correction_travel_m"]+shift > .01+1e-12
                or np.linalg.norm(corrected-model) > .01+1e-12):
            return False
        if shift <= 1e-12:
            return True
        item["world_goal_xyz"] = corrected.tolist()
        item["visual_correction_count"] += 1
        item["visual_correction_travel_m"] += shift
        item["last_visual_correction"] = {"reason": reason, "previous_world_goal_xyz": old.tolist(),
            "proposed_executor_goal_xyz": proposed.tolist(), "stored_world_goal_xyz": corrected.tolist(),
            "shift_m": shift, "task_success_certified": False}
        return True

    def summary(self):
        return {"version": "persistent-world-subgoal-v10", "task_id": self.task_id,
                "active": copy.deepcopy(self.active), "last_update": copy.deepcopy(self.last_update),
                "completed_or_superseded_count": len(self.completed), "explicit_revisions": self.revision_count,
                "local_complete": bool(self.active and self.active["local_complete"]),
                "task_success_certified": False,
                "semantics": "Measured arrival is local goal progress only, never grasp or task success."}


class MetricFieldExecutorV10(MetricFieldExecutorV9):
    """Retain V9 geometry and V8 hold, with bounded command-starvation exit."""

    MIN_USEFUL_COMMAND_M = MIN_USEFUL_COMMAND_M

    def __init__(self):
        super().__init__()
        self.subgoal_origin = None
        self.starvation_diagnostics = None

    def configure(self, state8, intent, *, scene_assessment=None, subgoal_origin_xyz=None):
        state = _vector(state8, 8, "state8")
        checked = validate_metric_intent_v6(intent)
        original = state[:3].copy() if subgoal_origin_xyz is None else _vector(subgoal_origin_xyz, 3, "subgoal_origin_xyz")
        locked = list(_LOCKED_AXES[checked["motion_axes"]])
        if np.any(abs((state[:3]-original)[locked]) > LOCKED_AXIS_TOLERANCE_M+1e-12):
            raise ValueError("Stored subgoal locked-axis corridor requires explicit replanning")
        self.subgoal_origin = original.copy()
        self.starvation_diagnostics = None
        super().configure(state, checked, scene_assessment=scene_assessment)
        if locked:
            self.field = _AxisConstrainedField(self.field, locked, original)
        return self.summary()

    def _outside_corridor(self, position):
        axes = list(_LOCKED_AXES[self.intent["motion_axes"]])
        origin = self.origin if self.subgoal_origin is None else self.subgoal_origin
        return bool(np.any(abs((np.asarray(position)-origin)[axes]) > LOCKED_AXIS_TOLERANCE_M+1e-12))

    def select(self, state8, joint_position, *, command_cap_m=None):
        # Let inherited validation and stronger stop reasons win. If it selects
        # a below-floor action, discard its pending command before returning a
        # rejection; select itself never consumes measured action budgets.
        if command_cap_m is not None:
            cap = _number(command_cap_m, 0., .02, "command_cap_m")
            if cap < 1e-6:
                # The base executor accepts positive caps only. A zero monitor
                # allowance remains a zero-action handoff, never rounded upward.
                if self.intent is None:
                    raise RuntimeError("configure is required before select")
                if self.pending is not None:
                    raise RuntimeError("observe must acknowledge the previous accepted action")
                state = _vector(state8, 8, "state8"); _vector(joint_position, 7, "joint_position")
                if self.stop_reason is not None:
                    return self._result(self.stop_reason)
                if not np.allclose(state, self.expected_state, atol=1e-9, rtol=0.):
                    return self._stop("unacknowledged_state_change")
                self.starvation_diagnostics = {"effective_cap_m": cap, "minimum_useful_command_m": self.MIN_USEFUL_COMMAND_M}
                return self._stop("field_command_below_useful_motion_floor")
        previous_orientation_count = self.orientation_goal_refresh_commands
        previous_orientation_command = self.last_orientation_command_rad.copy()
        result = super().select(state8, joint_position, command_cap_m=command_cap_m)
        if not result["accepted"]:
            return result
        tracking = result["diagnostics"].get("tracking", {})
        cap = tracking.get("max_command_m", self.intent["command_limit_m"])
        if cap < self.MIN_USEFUL_COMMAND_M-1e-12:
            self.pending = None
            self.orientation_goal_refresh_commands = previous_orientation_count
            self.last_orientation_command_rad = previous_orientation_command
            self.starvation_diagnostics = {"effective_cap_m": cap,
                "minimum_useful_command_m": self.MIN_USEFUL_COMMAND_M,
                "discarded_command_norm_m": float(np.linalg.norm(result["action"][:3])*.05),
                "command_rounded_up": False}
            return self._stop("field_command_below_useful_motion_floor")
        return result

    def summary(self):
        result = super().summary()
        if self.intent is not None:
            result.update(version="persistent-subgoal-distance-executor-v10",
                subgoal_locked_axis_origin_xyz=None if self.subgoal_origin is None else self.subgoal_origin.tolist(),
                starvation_guard={"minimum_useful_command_m": self.MIN_USEFUL_COMMAND_M,
                                  "details": copy.deepcopy(self.starvation_diagnostics),
                                  "command_rounded_up": False})
        return result
