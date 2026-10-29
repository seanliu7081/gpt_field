"""Astra selects field planning or direct control at configured chunk boundaries.

Only the routing and orchestration are new. Coarse actions pass through the
original FieldPolicy; direct actions are executed exactly as validated. Both
branches share one transcript, step counter and observable-feedback window.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from .action import _quaternion_product, _rotvec_quaternion
from .direct_history import TASK, DirectHistoryPolicy, _allowed
from .hybrid_subscription import (
    HYBRID_CHUNK_SIZE, HYBRID_FIELD_UPDATE_STEPS, HYBRID_GRID, HYBRID_PLANNER_HORIZON,
    validate_hybrid_prediction,
)
from .observation import PolicyObservation
from .policy import Decision, FieldPolicy
from .planner import PlannerConfig
from .schema import GridSpec
from .field_grounding import VisualGoalConsistencyChecker


class ModelTaskComplete(Exception):
    """An accepted model completion decision, with no proposed robot action."""

    def __init__(self, details, prediction):
        self.details = copy.deepcopy(details)
        self.prediction = copy.deepcopy(prediction)
        super().__init__(details["completion_reason"])


class _FieldCache:
    """Supply one already validated field to the frozen planner without I/O."""

    def __init__(self, client, grid):
        self.client = client
        self.grid = grid
        self.clear()

    def clear(self):
        self.observation = self.prediction = None
        self.effective_compact_field = None
        self.consumed = False

    def prepare(self, observation, prediction, effective_compact_field=None):
        self.observation = observation
        self.prediction = copy.deepcopy(prediction)
        self.effective_compact_field = copy.deepcopy(effective_compact_field)
        self.consumed = False

    @property
    def last_request_key(self):
        return self.client.last_request_key

    @property
    def last_log_path(self):
        return self.client.last_log_path

    def predict(self, observation, grid, mode="field"):
        if mode != "field" or grid != self.grid:
            raise ValueError("Cached coarse predictions require the configured hybrid grid")
        if self.consumed or self.prediction is None:
            raise RuntimeError("A fresh hybrid field is required before planning")
        if PolicyObservation.from_dict(observation) != self.observation:
            raise ValueError("Field must be centered at its actual generation observation")
        self.consumed = True
        return copy.deepcopy(self.prediction)


class HybridFieldDirectPolicy:
    """Reversible mode selection, committed for one configured action chunk.

Routing is Astra's visual judgment, not an oracle proximity/contact predicate.
No forced closure, grasp controller, automatic recovery, or reward input exists.
"""

    def __init__(self, client):
        self.client = client
        self.field_improvements = getattr(client, "field_improvements", False)
        self.grid = getattr(client, "grid", HYBRID_GRID)
        self.method = "hybrid_field_direct"
        self.direct_action_steps = getattr(client, "chunk_size", HYBRID_CHUNK_SIZE)
        self.field_update_steps = getattr(
            client, "field_update_steps",
            self.direct_action_steps if self.grid == GridSpec() else HYBRID_FIELD_UPDATE_STEPS,
        )
        self.update_steps = self.field_update_steps
        if type(self.direct_action_steps) is not int or self.direct_action_steps < 1:
            raise ValueError("Direct action chunk length must be a positive integer")
        planner_horizon = getattr(client, "planner_horizon",
                                  3 if self.grid == GridSpec() else HYBRID_PLANNER_HORIZON)
        default_execution = (planner_horizon if self.grid == HYBRID_GRID
                             and self.field_update_steps != self.direct_action_steps
                             and planner_horizon != 3 else 1)
        self.field_execution_steps = getattr(client, "field_execution_steps", default_execution)
        self._field_cache = _FieldCache(client, self.grid)
        self._field_policy = FieldPolicy(self._field_cache, method="field", grid=self.grid,
                                         update_steps=self.update_steps,
                                         planner_config=PlannerConfig(horizon=planner_horizon),
                                         execution_steps=self.field_execution_steps,
                                         execution_mode=getattr(client, "execution_mode", "fixed"),
                                         field_improvements=self.field_improvements)
        self.reset()

    @property
    def planner(self):
        return self._field_policy.planner

    @property
    def pending_field_event(self):
        """Observable evidence closing the current field block, if any."""
        return copy.deepcopy(self._field_refresh_event)

    @property
    def adapter(self):
        return self._field_policy.adapter

    @property
    def field(self):
        return self._field_policy.field if self.controller == "coarse_field" else None

    def reset(self):
        self.client.reset_history()
        self._field_cache.clear()
        self._field_policy.reset()
        self.step_index = 0
        self._chunk_start = 0
        self._chunk_index = -1
        self._chunk_length = 0
        self.controller = None
        self.prediction = None
        self.routing = None
        self.chunk = []
        self.latest_feedback = None
        self._pending = None
        self._expected_observation = None
        self._progress = []
        self._completion = None
        self._field_refresh_event = None
        self._last_field_check = None
        self._visual_goal_checker = VisualGoalConsistencyChecker() if self.field_improvements else None
        self._field_motion_window = []

    def act(self, observation: PolicyObservation) -> Decision:
        if self._completion is not None:
            raise self._completion
        allowed = _allowed(observation.to_dict(), getattr(self.client, "task_prompt", TASK))
        if self._pending is not None:
            raise RuntimeError("observe_transition is required before the next action")
        if self._expected_observation is not None and allowed != self._expected_observation:
            raise ValueError("Next observation must equal the preceding transition's after observation")
        step = self.step_index
        refresh_event = copy.deepcopy(self._field_refresh_event)
        refreshed = (self.controller is None or step - self._chunk_start == self._chunk_length
                     or refresh_event is not None)
        previous_controller = self.controller
        api_latency = None
        if refreshed:
            started = time.monotonic()
            envelope = self._validate_envelope(
                self.client.predict(allowed.to_dict(), self.grid, mode="hybrid"))
            api_latency = time.monotonic() - started
            if (getattr(self.client, "stop_on_model_complete", False)
                    and self.client.last_task_complete):
                details = {
                    "step": step, "method": self.method, "refreshed": True,
                    "task_complete": True, "completion_reason": self.client.last_completion_reason,
                    "controller": envelope["controller"], "previous_controller": previous_controller,
                    "routing": {key: copy.deepcopy(envelope[key]) for key in
                                ("controller", "target_region", "near_target", "routing_reason")},
                    "routing_is_model_prediction": getattr(self.client, "controller_mode", "hybrid") == "hybrid",
                    "hybrid_prediction": copy.deepcopy(envelope),
                    "compact_prediction": copy.deepcopy(self.client.last_compact_prediction),
                    "field_representation": self.client.field_representation,
                    "field_expansion_latency_s": self.client.last_field_expansion_s,
                    "api_latency_s": api_latency,
                    "api_request_key": self.client.last_request_key,
                    "api_request_log": self.client.last_log_path,
                    "history_stats": self.client.history_stats(),
                    "feedback": copy.deepcopy(self.latest_feedback),
                    "requested_reasoning_effort": self.client.reasoning_effort,
                    "status": "model_declared_complete", "stage": "model declared task complete",
                }
                self._completion = ModelTaskComplete(details, envelope)
                raise self._completion
            self.prediction = copy.deepcopy(envelope)
            self._field_refresh_event = None
            self._field_motion_window = []
            self.controller = envelope["controller"]
            self._chunk_start = step
            self._chunk_index += 1
            self._chunk_length = (self.field_update_steps if self.controller == "coarse_field"
                                  else self.direct_action_steps)
            self.routing = {key: copy.deepcopy(envelope[key]) for key in
                            ("controller", "target_region", "near_target", "routing_reason")}
            if self.controller == "coarse_field":
                self._field_cache.prepare(allowed, envelope["field"],
                    getattr(self.client, "last_effective_compact_field", None))
                if self.field_improvements:
                    raw_field = self.client.last_compact_prediction["field"]
                    self._last_field_check = self._visual_goal_checker.configure(
                        allowed, raw_field["goal_reference"])
                self.chunk = []
            else:
                self._field_policy.clear_pending_plan()
                self.chunk = copy.deepcopy(envelope["direct"]["actions"])

        chunk_step = step - self._chunk_start
        if self.controller == "coarse_field":
            # A field block may start at any global step after direct control.
            # Its local step zero forces a freshly centered field. Each local
            # plan commits its configured primitive prefix within that field.
            self._field_policy.step_index = chunk_step
            original = self._field_policy.act(allowed)
            action, details = original.action, dict(original.diagnostics)
            if "plan_start_step" in details:
                details["plan_start_step"] += self._chunk_start
        elif self.controller == "contact_direct":
            if not self.chunk:
                raise RuntimeError("Direct chunk exhausted before its next refresh boundary")
            action = tuple(float(value) for value in self.chunk.pop(0))
            details = {"status": "direct_action", "stage": "contact_direct:" + self.routing["target_region"]}
            if refreshed:
                details["prediction"] = copy.deepcopy(self.prediction["direct"])
        else:
            raise RuntimeError("No validated controller has been selected")

        details.update({"step": step, "method": self.method, "refreshed": refreshed,
                        "update_steps": self._chunk_length,
                        "field_update_steps": self.field_update_steps,
                        "direct_action_steps": self.direct_action_steps,
                        "controller": self.controller, "routing": copy.deepcopy(self.routing),
                        "routing_decision_step": self._chunk_start,
                        "routing_is_model_prediction": getattr(self.client, "controller_mode", "hybrid") == "hybrid",
                        "previous_controller": previous_controller,
                        "mode_changed": previous_controller is not None and previous_controller != self.controller,
                        "chunk_index": self._chunk_index, "chunk_step": chunk_step,
                        "field_refreshed": refreshed and self.controller == "coarse_field",
                        "direct_chunk_refreshed": refreshed and self.controller == "contact_direct",
                        "history_stats": self.client.history_stats(),
                        "feedback": copy.deepcopy(self.latest_feedback)})
        if refreshed:
            details.update({"hybrid_prediction": copy.deepcopy(self.prediction),
                            "api_latency_s": api_latency,
                            "api_request_key": self.client.last_request_key,
                            "api_request_log": self.client.last_log_path})
        if hasattr(self.client, "field_representation"):
            details["field_representation"] = self.client.field_representation
            if refreshed:
                details["compact_prediction"] = copy.deepcopy(self.client.last_compact_prediction)
                details["field_expansion_latency_s"] = self.client.last_field_expansion_s
        if self.field_improvements:
            details["field_improvements"] = True
            details["field_refresh_event"] = refresh_event
            details["visual_goal_check"] = copy.deepcopy(self._last_field_check)
            if refreshed:
                details["goal_grounding"] = copy.deepcopy(self.client.last_goal_grounding)
                details["effective_compact_field"] = copy.deepcopy(self.client.last_effective_compact_field)
        decision = Decision(tuple(action), details)
        self._pending = (allowed, decision)
        self.step_index += 1
        return decision

    def _validate_envelope(self, value):
        """Representation boundary; opt-in compact policies can avoid dense expansion."""
        return validate_hybrid_prediction(value, self.grid, chunk_size=self.direct_action_steps,
            target_regions=getattr(self.client, "target_regions", ("black_bowl", "plate")))

    def observe_transition(self, before, decision, after):
        feedback = DirectHistoryPolicy.observe_transition(self, before, decision, after)
        # A direct zero is executed as zero, while the last nonzero open/close
        # intent remains available to a later field's "hold" command.
        grip = decision.action[6]
        if grip != 0:
            self.adapter._gripper = 1.0 if grip > 0 else -1.0
        if self.field_improvements and self.controller == "coarse_field":
            self._check_field_progress(after, decision, feedback)
        return feedback

    def _check_field_progress(self, after, decision, feedback):
        """Request a new visual decision; never claim success or end an episode."""
        self._last_field_check = self._visual_goal_checker.observe(after)
        compact = self._field_cache.effective_compact_field
        if compact["motion_mode"] == "hold":
            return
        reason = None
        if self._last_field_check.get("refresh_requested"):
            reason = "visual_divergence"
        goal = np.asarray(self._field_policy.field.center) + np.asarray(compact["goal_offset_m"])
        position = np.asarray(after.state[:3])
        config = self.planner.config
        if reason is None and (self.field.query(position) is None
                or np.any(position < config.workspace_min) or np.any(position > config.workspace_max)):
            reason = "invalid_direction"
        current = _rotvec_quaternion(np.asarray(after.state[3:6]))
        target = np.asarray(compact["orientation_target_xyzw"])
        target = target / np.linalg.norm(target)
        rotation_error = _quaternion_product(target, np.r_[-current[:3], current[3]])
        orientation_error_rad = 2 * np.arctan2(np.linalg.norm(rotation_error[:3]), abs(rotation_error[3]))
        if (reason is None and np.linalg.norm(position - goal) <= 0.005
                and orientation_error_rad <= 0.05):
            reason = "goal_reached"
        plan = decision.diagnostics["plan"]
        search = plan.get("search") or {}
        if (reason is None and search.get("boundary_clipped")
                and self._field_policy.tracker.endpoint_reached(after.state[:3])):
            reason = "field_boundary_reached"
        tracking_status = decision.diagnostics.get("tracking", {}).get("status")
        if reason is None and (tracking_status in ("invalid_measured_start", "no_valid_reference")
                or (plan["selected_index"] is None and plan["status"] != "goal_reached")):
            reason = "invalid_direction"
        self._field_motion_window.append((feedback["commanded_translation_norm_m"],
                                          feedback["measured_translation_norm_m"]))
        self._field_motion_window = self._field_motion_window[-5:]
        if (reason is None and len(self._field_motion_window) == 5
                and sum(item[0] for item in self._field_motion_window) >= 0.005
                and sum(item[1] for item in self._field_motion_window) <= 0.0005):
            reason = "no_progress"
        if reason is not None:
            event = {"type": "field_control_event", "step": self.step_index,
                     "chunk_index": self._chunk_index, "reason": reason}
            if reason == "visual_divergence":
                event["visual_check"] = {key: self._last_field_check[key] for key in
                    ("status", "current_distance_px", "baseline_distance_px")
                    if key in self._last_field_check}
            self.client.record_field_event(event)
            self._field_refresh_event = event
            self._field_policy.clear_pending_plan()
