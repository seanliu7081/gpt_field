"""Opt-in RGB-feedback distance-field control; no evaluator or simulator inputs."""

from __future__ import annotations

import copy
import time

import numpy as np

from .action import _quaternion_product, _rotvec_quaternion
from .compact_field import expand_compact_field
from .direct_history import DirectHistoryPolicy
from .distance_planner import DistanceFieldPlanner
from .field import GridField
from .hybrid_policy import HybridFieldDirectPolicy
from .local_visual_feedback import LocalVisualFeedback
from .policy import Decision
from .planner import CandidateRecord, PlanResult
from .schema import FieldPrediction
from .tracking import PathTracker


class _TrackingDistancePlanner(DistanceFieldPlanner):
    def __init__(self, config, *, owner):
        self.owner = owner
        super().__init__(config)

    def plan_analytic(self, field, position, *, goal_xyz, compact_prediction, goal_tolerance_m=.005):
        if self.owner._recovery_active:
            return self._vertical_reobservation(field, position, goal_xyz, compact_prediction,
                                                goal_tolerance_m / 2)
        # Reserve half of the measured arrival tolerance for path tracking;
        # stopping near a goal-region endpoint must still be near the real goal.
        return self.plan(field, position, goal_xyz=goal_xyz, compact_prediction=compact_prediction,
                         goal_tolerance_m=goal_tolerance_m / 2)

    def _vertical_reobservation(self, field, position, goal_xyz, compact, tolerance):
        """Follow the separately authorized +Z observation motion, not a cost detour.

        The original/effective compact repulsors stay recorded and unchanged.
        Hard geometry and workspace checks remain active. The path starts at the
        actual position and never adds a lateral or downward geometric segment.
        """
        start = np.asarray(position, dtype=float)
        goal = np.asarray(goal_xyz, dtype=float)
        vertical_distance = max(0., float(goal[2] - start[2]))
        boxes = self._boxes(field, compact)
        search = {"method": "local_vertical_reobservation", "cache_hit": False,
                  "candidate_semantics": "one_explicit_vertical_reobservation_path",
                  "goal_xyz": goal.tolist(), "goal_tolerance_m": tolerance,
                  "goal_distance_m": float(np.linalg.norm(goal - start)),
                  "cost_unit": "m", "goal_reached": False, "endpoint_is_goal": False,
                  "endpoint_in_goal_region": False, "direction_invalid": False,
                  "boundary_clipped": False, "segment_count": 0,
                  "soft_repulsors_ignored_for_reobservation": bool(compact["repulsors"]),
                  "hard_geometry_checked": True,
                  "max_path_length_m": min(.02, self.config.horizon * self.config.step_m)}
        if not self._valid_segment(field, start, start, boxes):
            search.update(direction_invalid=True, path_termination="invalid_start")
            return PlanResult((0., 0., 0.), "no_valid_candidates", 0, 0,
                              reason="invalid_start", search_metadata=search)
        if vertical_distance <= tolerance:
            reached = bool(np.linalg.norm(goal - start) <= .005)
            search.update(goal_reached=reached, endpoint_in_goal_region=reached,
                          path_termination="reobservation_height_reached")
            return PlanResult((0., 0., 0.), "goal_reached" if reached else "hold", 0, 0,
                              reason="reobservation_height_reached", search_metadata=search)
        distance = min(vertical_distance, search["max_path_length_m"])
        arcs = [index * self.config.step_m for index in range(1, self.config.horizon + 1)
                if index * self.config.step_m < distance - 1e-12]
        arcs.append(distance)
        points = [start] + [start + np.array([0., 0., arc]) for arc in arcs]
        if not all(self._valid_segment(field, a, b, boxes) for a, b in zip(points[:-1], points[1:])):
            search.update(direction_invalid=True, path_termination="invalid_reobservation_segment")
            return PlanResult((0., 0., 0.), "no_valid_candidates", 0, 0,
                              reason="invalid_reobservation_segment", search_metadata=search)
        terminal = max(0., vertical_distance - distance)
        candidate = CandidateRecord(("vertical_reobservation",) * len(arcs),
            tuple(tuple(float(value) for value in point) for point in points),
            distance, True, terminal, max(0., vertical_distance - arcs[0]))
        search.update(segment_count=len(arcs), endpoint_xyz=points[-1].tolist(),
                      endpoint_is_goal=bool(np.linalg.norm(points[-1] - goal) <= 1e-12),
                      endpoint_in_goal_region=bool(np.linalg.norm(points[-1] - goal) <= .005),
                      path_termination="reobservation_height" if terminal <= 1e-12 else "horizon")
        return PlanResult(tuple(float(value) for value in points[1] - start), "move", 1, 1,
                          (candidate,), 0, vertical_distance, distance, search_metadata=search)


class _ContinuousValidityField:
    def __init__(self, field, planner, compact):
        self.base, self.planner = field, planner
        self.boxes = planner._boxes(field, compact)

    def query(self, point):
        point = np.asarray(point, dtype=float)
        if not self.planner._avoids_boxes(point, point, self.boxes)[0]:
            return None
        return self.base.query(point)

    def segment_valid(self, start, end):
        return self.planner._valid_segment(self.base, np.asarray(start), np.asarray(end), self.boxes)


class _DistanceTracker(PathTracker):
    def __init__(self, owner):
        self.owner = owner
        super().__init__(endpoint_tolerance_m=.0025)

    def command(self, position, field, workspace_min, workspace_max):
        checked = _ContinuousValidityField(field, self.owner.planner,
                                           self.owner._field_cache.effective_compact_field)
        return super().command(position, checked, workspace_min, workspace_max)


class DistanceHybridPolicy(HybridFieldDirectPolicy):
    """Reuse a bounded local map and update planar goals from measured RGB.

    Field reuse never expands the originally predicted known domain. A view
    recovery is an explicitly logged local command, not a model-predicted action
    or a task-success decision. Contact Direct actions remain unchanged.
    """

    max_field_reuse_steps = 120
    verification_steps = 5
    view_recovery_steps = 10

    def __init__(self, client):
        if not getattr(client, "distance_field_control", False):
            raise ValueError("DistanceHybridPolicy requires the distance control protocol")
        super().__init__(client)
        self._field_policy.planner = _TrackingDistancePlanner(self.planner.config, owner=self)
        self._field_policy.tracker = _DistanceTracker(self)
        # The outer policy decides when the model interval ends. At 30/60/90
        # ticks it may extend it; do not consume the one-use model cache again.
        self._field_policy.update_steps = self.max_field_reuse_steps

    def reset(self):
        super().reset()
        self.control_context = None
        self.local_visual = LocalVisualFeedback()
        self.local_feedback = {"status": "not_configured", "reliable": False,
                               "calibration_valid": False, "task_success_certified": False}
        self.local_revision = 0
        self.local_goal_updates = 0
        self._revision_pending = False
        self._local_update_seconds = 0.
        self._visual_seconds = 0.
        self.local_visual_total_s = self.local_update_total_s = 0.
        self._recent_motion = []
        self._visual_errors = []
        self._reuse_count = 0
        self._recovery_active = False
        self._recovery_started = None
        self._reobservation_used = False
        self._verification_observations = 0
        self._stable_verification_frames = 0
        self._previous_verify_points = None
        self._tracking_was_lost = False
        self._previous_grip = -1.
        if hasattr(self.planner, "reset_cache"):
            self.planner.reset_cache()

    def _phase(self):
        return (self.control_context or {}).get("phase")

    def _configure_visual(self, observation):
        context = self.client.last_control_context
        self.control_context = copy.deepcopy(context)
        reference = context.get("visual_reference")
        if reference is not None:
            reference = {"kind": reference["kind"], "visual_anchor": {
                "moving_point_uv": reference["moving_point_uv"],
                "target_point_uv": reference["target_point_uv"]}}
            # A released bowl is not rigidly coupled to end-effector motion.
            compact = self._field_cache.effective_compact_field
            grip = compact.get("gripper_command") if compact else None
            fingers = observation.state[6:8]
            stable_fingers = (self.latest_feedback is not None and
                max(abs(x) for x in self.latest_feedback["gripper_joint_delta"]) < .002)
            prior_closed = (self._previous_grip > 0 and abs(fingers[0] - fingers[1]) < .075
                            and stable_fingers and grip in ("close", "hold"))
            rigid = reference["kind"] == "end_effector" or prior_closed
            if context["phase"] == "align" and rigid and self.controller == "coarse_field":
                reference["local_alignment"] = {
                    "semantic": "same_rigid_point_desired_projection", "planar_alignment": True}
        self.local_feedback = self.local_visual.configure(observation, reference)
        self.local_revision = 0
        self.local_goal_updates = 0
        self._revision_pending = False
        self._recent_motion = []
        self._visual_errors = []
        self._reuse_count = 0
        self._recovery_active = False
        self._verification_observations = 0
        self._stable_verification_frames = 0
        self._previous_verify_points = None
        self._tracking_was_lost = False

    def _pixel_error(self):
        error = self.local_feedback.get("pixel_error")
        return None if error is None else float(np.linalg.norm(error))

    def _event(self, reason):
        if self._field_refresh_event is not None:
            return
        event = {"type": "distance_control_event", "step": self.step_index,
                 "chunk_index": self._chunk_index, "reason": reason,
                 "feedback": {"status": self.local_feedback["status"],
                     "reliable": bool(self.local_feedback.get("reliable")),
                     "calibration_valid": bool(self.local_feedback.get("calibration_valid")),
                     "pixel_error_norm": self._pixel_error(),
                     "local_goal_updates": self.local_goal_updates,
                     "task_success_certified": False}}
        self.client.record_control_event(event)
        self._field_refresh_event = event
        self._field_policy.clear_pending_plan()

    def _can_reuse(self):
        compact = self._field_cache.effective_compact_field
        return (self._chunk_length < self.max_field_reuse_steps
                and self._phase() in ("approach", "transport", "align")
                and compact["motion_mode"] == "move"
                and self.local_feedback.get("reliable") is True
                and len(self._recent_motion) >= 5
                and sum(x[1] for x in self._recent_motion) > .001
                and len(self._visual_errors) >= 5
                and self._visual_errors[-1] <= self._visual_errors[0] + 2.
                and not self._recovery_active)

    def act(self, observation):
        if (self.controller == "coarse_field" and self._field_refresh_event is None
                and self.step_index - self._chunk_start == self._chunk_length):
            if self._can_reuse():
                self._chunk_length = min(self.max_field_reuse_steps,
                                         self._chunk_length + self.field_update_steps)
                self._reuse_count += 1
            else:
                self._event("field_budget_exhausted")
        will_refresh = (self.controller is None or self._field_refresh_event is not None
                        or self.step_index - self._chunk_start == self._chunk_length)
        if will_refresh:
            self.planner.reset_cache()
            # A recovery override belongs only to its previous model block.
            # The fresh response is planned inside super().act(), before the
            # new visual context is configured below.
            self._recovery_active = False
        elif self._recovery_active:
            # Re-anchor the cheap vertical recovery to each measured pose;
            # tracking an older line would correct lateral drift or descend
            # after overshoot, outside this observation move's intent.
            self._field_policy.clear_pending_plan()
        decision = super().act(observation)
        details = dict(decision.diagnostics)
        if details["refreshed"]:
            began = time.monotonic()
            self._configure_visual(observation)
            elapsed = time.monotonic() - began
            self._visual_seconds += elapsed
            self.local_visual_total_s += elapsed
        details.update(distance_field_control=True,
                       control_context=copy.deepcopy(self.control_context),
                       local_visual_feedback=copy.deepcopy(self.local_feedback),
                       local_field_revision=self.local_revision,
                       local_goal_updates=self.local_goal_updates,
                       field_reuse_extensions=self._reuse_count,
                       local_recovery_active=self._recovery_active,
                       local_field_updated=self._revision_pending,
                       local_update_latency_s=self._local_update_seconds,
                       local_visual_latency_s=self._visual_seconds,
                       task_success_certified=False)
        if self.controller == "coarse_field":
            details["effective_compact_field"] = copy.deepcopy(self._field_cache.effective_compact_field)
            if self._revision_pending:
                details["field_prediction"] = self.field.prediction.to_dict()
            search = details.get("plan", {}).get("search") or {}
            if (details.get("plan_refreshed") and details["plan"]["status"] == "move"
                    and not search.get("cache_hit", True)
                    and "settled_vertex_count" in search):
                distances = self.planner.distance_grid_m
                if distances is not None:
                    # Render a separate finite display map. The planner keeps
                    # infinity for unreachable states; guidance masks stay fixed.
                    values = np.asarray(distances).reshape(-1)
                    finite = np.isfinite(values)
                    display = self.field.prediction.to_dict()
                    display["values"] = np.where(finite, values, 0.).tolist()
                    display["unknown_indices"] = sorted(set(display["unknown_indices"]) |
                        (set(np.flatnonzero(~finite).tolist()) - set(display["blocked_indices"])))
                    details["distance_field_display"] = display
        self._revision_pending = False
        self._local_update_seconds = self._visual_seconds = 0.
        decision = Decision(decision.action, details)
        # Parent feedback validates equality against the exact proposed decision.
        self._pending = (self._pending[0], decision)
        return decision

    def _set_local_goal(self, goal, *, recovery=False):
        began = time.monotonic()
        goal = np.asarray(goal, dtype=float)
        if (not self._segment_valid(goal, goal) or np.any(goal < self.planner.config.workspace_min)
                or np.any(goal > self.planner.config.workspace_max)):
            return False
        compact = copy.deepcopy(self._field_cache.effective_compact_field)
        compact["goal_offset_m"] = (goal - np.asarray(self.field.center)).tolist()
        compact["motion_mode"] = "move"
        if recovery:
            compact["gripper_command"] = "open"
        expanded = expand_compact_field(compact, self.grid)
        self._field_cache.effective_compact_field = compact
        self._field_policy.prediction = expanded
        self._field_policy.field = GridField(self.grid, self.field.center,
                                             FieldPrediction.from_dict(expanded, self.grid))
        self._field_policy.clear_pending_plan()
        self.local_revision += 1
        self.local_goal_updates += int(not recovery)
        self._revision_pending = True
        elapsed = time.monotonic() - began
        self._local_update_seconds += elapsed
        self.local_update_total_s += elapsed
        return True

    def _segment_valid(self, start, end):
        return _ContinuousValidityField(self.field, self.planner,
            self._field_cache.effective_compact_field).segment_valid(start, end)

    def _request_reobservation(self, after, reason):
        task_spec = getattr(self.client, "task_spec", None)
        allow_upward = getattr(task_spec, "allow_upward_reobservation", True)
        if allow_upward and not self._reobservation_used and self.adapter._gripper < 0:
            position = np.asarray(after.state[:3])
            goal = position + np.array([0., 0., .02])
            if self._segment_valid(position, goal) and self._set_local_goal(goal, recovery=True):
                self._reobservation_used = True
                self._recovery_active = True
                self._recovery_started = self.step_index
                self._recent_motion = []
                return
        self._event(reason)

    def _check_field_progress(self, after, decision, feedback):
        # LocalVisualFeedback is observed once in observe_transition, including
        # Direct. Do not run the legacy checker here or skip explicit holds.
        compact = self._field_cache.effective_compact_field
        goal = np.asarray(self.field.center) + np.asarray(compact["goal_offset_m"])
        position = np.asarray(after.state[:3])
        search = decision.diagnostics["plan"].get("search") or {}
        status = decision.diagnostics["plan"]["status"]
        if self._recovery_active:
            if (np.linalg.norm(position - goal) <= .005 or
                    self.step_index - self._recovery_started >= self.view_recovery_steps or
                    status in ("needs_observation", "no_valid_candidates")):
                self._event("placement_unverified")
            return
        placement_target = getattr(getattr(self.client, "task_spec", None), "placement_target", "plate")
        verifying = (self._phase() == "verify" or
                     compact["motion_mode"] == "hold" and self.routing["target_region"] == placement_target
                     and self.adapter._gripper < 0)
        if verifying:
            self._verification_observations += 1
            error = self._pixel_error()
            points = [self.local_feedback.get(key) for key in ("moving_point_px", "target_point_px")]
            valid = self.local_feedback.get("reliable") and error is not None and error <= 3.
            stable = (valid and all(point is not None for point in points) and
                      (self._previous_verify_points is None or
                       np.linalg.norm(np.asarray(points) - self._previous_verify_points, axis=1).max() <= 1.))
            self._stable_verification_frames = self._stable_verification_frames + 1 if stable else 0
            self._previous_verify_points = np.asarray(points) if stable else None
            if self._verification_observations >= self.verification_steps:
                if self._stable_verification_frames >= self.verification_steps:
                    self._event("placement_observation_consistent")
                else:
                    self._request_reobservation(after, "placement_misaligned" if
                        self.local_feedback.get("reliable") and error is not None and error > 3.
                        else "placement_unverified")
            return
        if compact["motion_mode"] == "hold":
            return
        if self._tracking_was_lost:
            self._event("visual_unavailable")
            return
        if status == "needs_observation":
            self._event("field_boundary_reached")
            return
        if (status == "no_valid_candidates" or not self._segment_valid(position, position)
                or np.any(position < self.planner.config.workspace_min)
                or np.any(position > self.planner.config.workspace_max)
                or decision.diagnostics.get("tracking", {}).get("status") in
                    ("invalid_measured_start", "no_valid_reference")):
            self._event("invalid_direction")
            return
        self._recent_motion.append((feedback["commanded_translation_norm_m"],
                                    feedback["measured_translation_norm_m"]))
        self._recent_motion = self._recent_motion[-10:]
        error = self._pixel_error()
        if error is not None:
            self._visual_errors = (self._visual_errors + [error])[-10:]
        if (len(self._recent_motion) >= 5 and sum(x[0] for x in self._recent_motion[-5:]) >= .005
                and sum(x[1] for x in self._recent_motion[-5:]) <= .0005):
            self._event("no_progress")
            return
        correction = self.local_feedback.get("goal_correction_xy_m")
        if self._phase() == "align" and self.local_feedback.get("calibration_valid") and correction is not None:
            proposed = np.array([position[0] + correction[0], position[1] + correction[1], goal[2]])
            if not self.local_visual.validate_metric_goal(proposed):
                self.local_feedback = self.local_visual.invalidate("proposed_goal_outside_calibration")
                self._event("visual_unavailable")
                return
            if np.linalg.norm(proposed - goal) > .002:
                if not self._set_local_goal(proposed):
                    self._event("local_goal_outside_field")
                    return
                goal = proposed
        elif self._phase() == "align" and self.step_index - self._chunk_start >= 10:
            # Rank-one motion or missing texture must not masquerade as a
            # calibrated metric update; ask for another view or Direct recovery.
            self._event("visual_unavailable")
            return
        current = _rotvec_quaternion(np.asarray(after.state[3:6]))
        target = np.asarray(compact["orientation_target_xyzw"], dtype=float)
        target /= np.linalg.norm(target)
        rotation = _quaternion_product(target, np.r_[-current[:3], current[3]])
        angle = 2 * np.arctan2(np.linalg.norm(rotation[:3]), abs(rotation[3]))
        if np.linalg.norm(position - goal) <= .005 and angle <= .05:
            self._event("goal_reached")

    def observe_transition(self, before, decision, after):
        feedback = DirectHistoryPolicy.observe_transition(self, before, decision, after)
        grip = decision.action[6]
        if grip != 0:
            self.adapter._gripper = 1. if grip > 0 else -1.
        # Re-arm one local view recovery after an actual new close/open cycle,
        # or a model-directed recovery motion; never repeatedly climb on hold.
        if grip < 0 and self._previous_grip > 0:
            self._reobservation_used = False
        if grip != 0:
            self._previous_grip = grip
        if (self.controller == "contact_direct" and self._phase() == "recover"
                and feedback["measured_translation_norm_m"] > .001):
            self._reobservation_used = False
        began = time.monotonic()
        was_reliable = self.local_feedback.get("reliable")
        was_calibrated = self.local_feedback.get("calibration_valid")
        self.local_feedback = self.local_visual.observe(after)
        self._tracking_was_lost = bool(was_reliable and not self.local_feedback.get("reliable") or
            was_calibrated and not self.local_feedback.get("calibration_valid") and self.local_goal_updates)
        elapsed = time.monotonic() - began
        self._visual_seconds += elapsed
        self.local_visual_total_s += elapsed
        if self.controller == "coarse_field":
            self._check_field_progress(after, decision, feedback)
            if (self._field_refresh_event is None
                    and self.step_index - self._chunk_start == self._chunk_length
                    and not self._can_reuse()):
                self._event("field_budget_exhausted")
        elif self.step_index - self._chunk_start == self.direct_action_steps:
            reason = ("placement_unverified" if self._phase() in ("release", "verify")
                      else "field_budget_exhausted")
            self._event(reason)
        return feedback
