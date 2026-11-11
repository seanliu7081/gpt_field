"""Small, opt-in distance-planning baseline for completion-speed experiments.

Reuse the early distance planner and measured-pose follower. The model owns
task progression and contact actions; no pixel objective, reference acquisition,
automatic view recovery, or V36 execution-admission machinery is installed.
Collision evaluation remains outside the policy.
"""

from __future__ import annotations

import copy
from pathlib import Path
import random
import time

import numpy as np

from .action import _quaternion_product, _rotvec_quaternion
from .astra import APIError, SchemaError
from .chatgpt_plan import _failure_metadata
from .direct_history import DirectHistoryPolicy
from .distance_planner import DistanceFieldPlanner
from .distance_policy import _ContinuousValidityField
from .distance_subscription import ChatGPTPlanDistanceClient, distance_output_schema
from .distance_tasks import resolve_distance_task
from .hybrid_policy import HybridFieldDirectPolicy
from .hybrid_subscription import HYBRID_GRID
from .policy import Decision
from .tracking import PathTracker


SPEED_FIRST_VERSION = "speed-first-distance-hybrid-v1"
OVERLOAD_RETRY_POLICY = "same-observation-overload-backoff-v1"


def speed_first_instructions(task_key="bowl_plate", image_history=2):
    task = resolve_distance_task(task_key)
    return (
        f"{SPEED_FIRST_VERSION}. Task: {task.prompt}. This is ZERO-SHOT: no demonstration, "
        "recorded solution, object poses, depth or camera calibration is provided. Use the current "
        "external and wrist RGB views and measured robot state. State is [end-effector xyz meters, "
        "axis-angle xyz radians, two finger joint positions]. World-frame actions are "
        "[dx,dy,dz,rx,ry,rz,gripper] in [-1,1]; translation 1 means 0.05 m, rotation 1 means "
        "0.5 radians, +1 gripper closes and -1 opens. Physics advances only with executed actions "
        "at 20 Hz; waiting for inference does not settle the fingers. "
        "Optimize time to completing the task. Use coarse_field for free-space approach and "
        "transport. Use contact_direct for alignment near interaction, grasping, articulated "
        "interaction, insertion, release, and recovery after an unsuccessful attempt. Do not "
        "keep navigating merely to obtain a better view once interaction is appropriate. "
        "Local waypoint arrival is not grasp readiness or task completion. Review both RGB views "
        "at each refresh; revise an incorrect goal and move toward the actual interaction. "
        "Do not optimize image centering by moving away from the object. "
        "Return controller, target_region, near_target, routing_reason, field, direct and "
        f"control_context. target_region is one of {', '.join(task.target_regions)}. near_target "
        "is true exactly for contact_direct. Set the inactive branch to null. Give a short "
        "observable routing reason. control_context contains phase and visual_reference=null; "
        f"phase is one of {', '.join(task.control_phases)}. "
        "For coarse_field output stage, unit orientation_target_xyzw, gripper_command "
        "open/close/hold, motion_mode move/hold, goal_offset_m, attraction_weights, repulsors, "
        "blocked_boxes, unknown_boxes and goal_reference=null. The goal is an end-effector "
        "waypoint relative to its CURRENT measured position. Account for the held object's "
        "offset when selecting a transport waypoint. Choose a nearby waypoint inside the "
        "16 cm cube, preferably within +/-0.075 m on each axis; a distant task target needs "
        "successive waypoints. The 33 cubed grid has 0.005 m spacing. Local Dijkstra computes "
        "26-neighbor shortest path cost to the waypoint; cost is edge length times the average "
        "density, where density is 1 plus nonnegative Gaussian repulsors. attraction_weights "
        "remain for schema compatibility. Provide up to four repulsors, each center_offset_m, "
        "radius_m and positive weight, and up to four boxes per kind with lower_offset_m and "
        "upper_offset_m. Prefer soft repulsors for uncertain obstacle estimates; reserve hard "
        "blocked/unknown boxes for regions you intend the planner to exclude. Geometry is a "
        "visual estimate, not measured clearance. An empty list is allowed; do not fabricate "
        "geometry. In this simulation protocol, collisions are recorded separately and do not "
        "themselves fail the episode; repeated attempts are permitted. "
        "The follower uses measured pose, 3 cm lookahead and bounded commands. A local path "
        "contains at most ten segments of at most 1 cm; it replans locally at most every ten "
        "executed actions. Field refreshes after at most 30 actions, or earlier at waypoint "
        "arrival (5 mm and 0.05 rad), invalid path, boundary or measured stall. It does not "
        "require image-feature validation, change goals from pixel error, or automatically "
        "raise the arm for another view. Use motion_mode=hold only for purposeful settling or "
        "observation; a Field hold hands back after five actions. "
        "For contact_direct output exactly five consecutive seven-number actions. They execute "
        "unchanged until an external episode stop; there is no local visual veto or replacement. "
        "Gripper hold in Field repeats the most recent nonzero command, initially open. Direct "
        "zero remains zero. After closure, keep the pose while the measured fingers settle; "
        "for a freely lifted object, test a short lift and check that it follows the gripper. "
        "If it does not, reopen, correct the observed misalignment and retry. An articulated "
        "handle requires movement consistent with its mechanism, not a lifting test. Before "
        "release, align the held object with its destination in both views. Continue correcting "
        "an incomplete placement; your statement of completion cannot terminate the episode. "
        f"The latest {image_history} live image pairs and block feedback summaries are supplied. "
        "Historical offsets refer to their original observations. Only the evaluator's actual "
        "LIBERO success completes this experiment; evaluator signals are not policy inputs."
    )


def speed_first_output_schema(task_key="bowl_plate"):
    schema = distance_output_schema(task_key=task_key)
    schema["properties"]["control_context"]["properties"]["visual_reference"] = {"type": "null"}
    for branch in schema["properties"]["field"]["anyOf"]:
        if branch.get("type") == "object":
            branch["properties"]["goal_reference"] = {"type": "null"}
    return schema


class ChatGPTPlanSpeedFirstClient(ChatGPTPlanDistanceClient):
    """Early compact wire format and subscription history, without visual goals."""

    speed_first = True

    def __init__(self, config=None, **kwargs):
        if kwargs.get("demonstration") is not None or kwargs.get("zero_shot_collision_free", False):
            raise ValueError("Speed-first control requires zero-shot, collision-record-only evaluation")
        if config is not None and config.model != "gpt-6-astra":
            raise ValueError("Speed-first control requires gpt-6-astra")
        if type(kwargs.get("chunk_size", 5)) is not int or kwargs.get("chunk_size", 5) != 5:
            raise ValueError("Speed-first Direct requires exactly five actions")
        super().__init__(config, **kwargs)
        if self.grid != HYBRID_GRID or self.field_update_steps != 30:
            raise ValueError("Speed-first baseline requires a 33 cubed / 5 mm grid and 30 Field actions")
        if self.planner_horizon != 10 or self.field_execution_steps != 10:
            raise ValueError("Speed-first baseline requires ten path segments and ten tracking actions per plan")

    @property
    def prompt_version(self):
        return SPEED_FIRST_VERSION

    def reset_history(self):
        super().reset_history()
        self._overload_attempt = 1
        self._overload_retry_requests = 0
        self._overload_retry_wait_s = 0.

    def history_stats(self):
        return {**super().history_stats(),
                "overload_retry_policy": OVERLOAD_RETRY_POLICY,
                "overload_retry_requests": self._overload_retry_requests,
                "overload_retry_wait_s": self._overload_retry_wait_s}

    def _write_log(self, path, data):
        if isinstance(data, dict) and "payload" in data and "request_ordinal" in data:
            data = {**data, "attempt": self._overload_attempt,
                    "overload_retry_policy": OVERLOAD_RETRY_POLICY}
        elif isinstance(data, dict) and data.get("status") == "error":
            data = {**data, "error": "Subscription attempt failed; no API-key fallback",
                    "overload_retry_policy": OVERLOAD_RETRY_POLICY}
        super()._write_log(path, data)

    def predict(self, observation, grid=None, mode="hybrid"):
        # Only this opt-in profile retries a confirmed provider overload. Failed
        # streams supply no executable actions and never commit model history.
        # The policy's outer timer includes every attempt and the backoff waits.
        with self._history_lock:
            same_observation = copy.deepcopy(observation)
            initial_observation_count = self._observation_count
            self._overload_attempt = 1
            try:
                while True:
                    try:
                        return super().predict(same_observation, grid, mode)
                    except APIError as exc:
                        if (self._overload_attempt >= 3
                                or getattr(exc, "error_code", None) != "server_is_overloaded"
                                or getattr(exc, "http_status", None) not in (None, 503)
                                or self.last_response is not None
                                or self._observation_count != initial_observation_count):
                            raise
                        delay = 2. ** self._overload_attempt + random.uniform(0., 1.)
                        retry_after = _failure_metadata(exc).get("retry_after_s", 0.)
                        delay = max(delay, retry_after)
                        # Never violate a long server Retry-After by clamping it.
                        # End this request instead of an unbounded client wait.
                        if delay > 60.:
                            raise
                        self._reserve()  # Counts all attempts against existing limits.
                        if (self.config.max_wall_time_s is not None
                                and self.clock() - self._started + delay >= self.config.max_wall_time_s):
                            raise
                        record = {
                            "protocol": OVERLOAD_RETRY_POLICY, "status": "bounded_retry_scheduled",
                            **_failure_metadata(exc), "attempt": self._overload_attempt,
                            "next_attempt": self._overload_attempt + 1, "max_attempts": 3,
                            "wait_s": delay, "same_observation": True,
                            "observation_index": initial_observation_count,
                            "failed_request_key": self.last_request_key,
                            "failed_request_ordinal": self._api_calls,
                            "actions_executed_from_failed_response": 0,
                            "budget": self.budget_snapshot(),
                        }
                        path = (Path(self.last_log_path).with_suffix(".overload_retry.json")
                                if self.last_log_path is not None else None)
                        self._write_log(path, record)
                        started = self.clock()
                        try:
                            time.sleep(delay)
                        finally:
                            self._overload_retry_wait_s += max(0., self.clock() - started)
                        self._overload_attempt += 1
                        self._overload_retry_requests += 1
            finally:
                self._overload_attempt = 1

    def build_payload(self, observation, grid=None, mode="hybrid"):
        payload = super().build_payload(observation, grid, mode)
        payload["instructions"] = speed_first_instructions(self.task_spec.key, self.image_history)
        payload["text"]["format"] = {
            "type": "json_schema", "name": "astra_speed_first_distance_hybrid_v1", "strict": True,
            "schema": speed_first_output_schema(self.task_spec.key),
        }
        return payload

    def validate_response(self, data, grid):
        if isinstance(data, dict):
            context = data.get("control_context")
            field = data.get("field")
            if isinstance(context, dict) and context.get("visual_reference") is not None:
                raise SchemaError("Speed-first visual_reference must be null")
            if isinstance(field, dict) and field.get("goal_reference") is not None:
                raise SchemaError("Speed-first goal_reference must be null")
        return super().validate_response(data, grid)


class _DistancePlanner(DistanceFieldPlanner):
    def plan_analytic(self, field, position, *, goal_xyz, compact_prediction, goal_tolerance_m=.005):
        # Preserve the successful baseline's joint planner/tracker tolerance.
        return self.plan(field, position, goal_xyz=goal_xyz, compact_prediction=compact_prediction,
                         goal_tolerance_m=goal_tolerance_m / 2)


class _DistanceTracker(PathTracker):
    def __init__(self, owner):
        self.owner = owner
        super().__init__(endpoint_tolerance_m=.0025)

    def command(self, position, field, workspace_min, workspace_max):
        checked = _ContinuousValidityField(field, self.owner.planner,
                                           self.owner._field_cache.effective_compact_field)
        return super().command(position, checked, workspace_min, workspace_max)


class SpeedFirstHybridPolicy(HybridFieldDirectPolicy):
    """Model-routed distance navigation plus unchanged five-action interaction."""

    def __init__(self, client):
        if not getattr(client, "speed_first", False):
            raise ValueError("SpeedFirstHybridPolicy requires the speed-first subscription contract")
        super().__init__(client)
        self._field_policy.planner = _DistancePlanner(self.planner.config)
        self._field_policy.tracker = _DistanceTracker(self)
        self.method = SPEED_FIRST_VERSION

    def reset(self):
        super().reset()
        self.control_context = None
        self.local_visual_total_s = self.local_update_total_s = 0.

    def act(self, observation):
        refresh = (self.controller is None or self._field_refresh_event is not None
                   or self.step_index - self._chunk_start == self._chunk_length)
        if refresh:
            self.planner.reset_cache()
            self._field_motion_window = []
        decision = super().act(observation)
        details = dict(decision.diagnostics)
        if details["refreshed"]:
            self.control_context = copy.deepcopy(self.client.last_control_context)
        details.update(speed_first=True, distance_field_control=True,
                       control_context=copy.deepcopy(self.control_context),
                       local_visual_feedback={"status": "disabled_speed_first", "reliable": False,
                                              "task_success_certified": False},
                       local_visual_latency_s=0., local_update_latency_s=0.,
                       local_goal_updates=0, local_field_revision=0, field_reuse_extensions=0,
                       local_recovery_active=False, local_field_updated=False,
                       task_success_certified=False)
        if self.controller == "coarse_field":
            details["effective_compact_field"] = copy.deepcopy(self._field_cache.effective_compact_field)
            search = details.get("plan", {}).get("search") or {}
            if (details.get("plan_refreshed") and details["plan"]["status"] == "move"
                    and not search.get("cache_hit", True) and "settled_vertex_count" in search):
                distances = self.planner.distance_grid_m
                if distances is not None:
                    values = np.asarray(distances).reshape(-1)
                    finite = np.isfinite(values)
                    display = self.field.prediction.to_dict()
                    display["values"] = np.where(finite, values, 0.).tolist()
                    display["unknown_indices"] = sorted(set(display["unknown_indices"]) |
                        (set(np.flatnonzero(~finite).tolist()) - set(display["blocked_indices"])))
                    details["distance_field_display"] = display
        decision = Decision(decision.action, details)
        self._pending = (self._pending[0], decision)
        return decision

    def _event(self, reason):
        if self._field_refresh_event is not None:
            return
        event = {"type": "distance_control_event", "step": self.step_index,
                 "chunk_index": self._chunk_index, "reason": reason,
                 "feedback": {"status": "disabled_speed_first", "reliable": False,
                              "calibration_valid": False, "pixel_error_norm": None,
                              "local_goal_updates": 0, "task_success_certified": False}}
        self.client.record_control_event(event)
        self._field_refresh_event = event
        self._field_policy.clear_pending_plan()

    def _check_field_progress(self, after, decision, feedback):
        compact = self._field_cache.effective_compact_field
        if compact["motion_mode"] == "hold":
            if self.step_index - self._chunk_start >= 5:
                self._event("field_budget_exhausted")
            return
        plan = decision.diagnostics["plan"]
        if plan["status"] == "needs_observation":
            self._event("field_boundary_reached")
            return
        position = np.asarray(after.state[:3])
        checked = _ContinuousValidityField(self.field, self.planner, compact)
        if (plan["status"] == "no_valid_candidates" or checked.query(position) is None
                or np.any(position < self.planner.config.workspace_min)
                or np.any(position > self.planner.config.workspace_max)
                or decision.diagnostics.get("tracking", {}).get("status") in
                    ("invalid_measured_start", "no_valid_reference")):
            self._event("invalid_direction")
            return
        self._field_motion_window.append((feedback["commanded_translation_norm_m"],
                                          feedback["measured_translation_norm_m"]))
        self._field_motion_window = self._field_motion_window[-5:]
        if (len(self._field_motion_window) == 5
                and sum(x[0] for x in self._field_motion_window) >= .005
                and sum(x[1] for x in self._field_motion_window) <= .0005):
            self._event("no_progress")
            return
        goal = np.asarray(self.field.center) + np.asarray(compact["goal_offset_m"])
        current = _rotvec_quaternion(np.asarray(after.state[3:6]))
        target = np.asarray(compact["orientation_target_xyzw"], dtype=float)
        target = target / np.linalg.norm(target)
        error = _quaternion_product(target, np.r_[-current[:3], current[3]])
        angle = 2 * np.arctan2(np.linalg.norm(error[:3]), abs(error[3]))
        if np.linalg.norm(position - goal) <= .005 and angle <= .05:
            self._event("goal_reached")

    def observe_transition(self, before, decision, after):
        feedback = DirectHistoryPolicy.observe_transition(self, before, decision, after)
        if decision.action[6] != 0:
            self.adapter._gripper = 1. if decision.action[6] > 0 else -1.
        if self.controller == "coarse_field":
            self._check_field_progress(after, decision, feedback)
        if self._field_refresh_event is None and self.step_index - self._chunk_start == self._chunk_length:
            self._event("field_budget_exhausted")
        return feedback
