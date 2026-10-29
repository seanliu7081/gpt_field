"""Multi-feature Field transport and model-directed Direct5 contact handoff.

No phase transition invents a grasp action or interprets image alignment as
success. GPT receives the current RGB and an explicit distinction between local
tracking availability, predicted path feasibility, and interaction readiness.
"""
import time

from .visual_field_policy_v4 import VisualFieldPolicyV4
from .visual_field_v5 import MetricFieldExecutorV5
from .visual_monitor_v5 import RoleAwareVisualMonitorV5


NORMAL_FIELD_ENDS = frozenset(("intent_action_budget", "intent_travel_budget",
                             "prospective_travel_budget", "prospective_radius_budget"))
REFERENCE_FAILURES = frozenset(("external_reference_unavailable", "controlled_feature_unresponsive",
                              "visual_response_contradiction", "target_anchor_shifted",
                              "visual_response_budget_exhausted"))


class VisualFieldPolicyV5(VisualFieldPolicyV4):
    def reset(self):
        super().reset()
        self.local = MetricFieldExecutorV5()
        self.monitor = RoleAwareVisualMonitorV5()
        self.reference_failure_events = []
        self.normal_field_refreshes = 0
        self.intent_refreshes = 0
        self.initial_visual_error_px = None
        self.visual_divergence_streak = 0

    def _refresh(self, obs):
        snapshot = getattr(self.client, "budget_snapshot", None)
        before_calls = None if snapshot is None else snapshot()["api_calls"]
        previous_calls = self.model_requests
        try:
            latency = super()._refresh(obs)
            self.intent_refreshes += 1
            self.initial_visual_error_px = (self.monitor.last_error
                if self._anchor_is_interaction_target() else None)
            self.visual_divergence_streak = 0
            return latency
        finally:
            if before_calls is not None:
                # One intent can need a bounded schema-repair request. Count
                # both requests, including an exception-ending request.
                self.model_requests = previous_calls + snapshot()["api_calls"] - before_calls

    @property
    def diagnostic_counters(self):
        result = super().diagnostic_counters
        result["intent_refreshes"] = getattr(self, "intent_refreshes", 0)
        return result

    def _allows_planar_correction(self):
        # The monitor still requires a validated rank-2 planar response, stable
        # height/orientation, a nearby metric goal, and a <=3 mm correction.
        # The planner rechecks the revised goal against every obstacle/unknown.
        return (self.prediction["phase"] in ("approach", "align")
                and self._anchor_is_interaction_target())

    def _anchor_is_interaction_target(self):
        prediction = self.prediction or {}
        external = prediction.get("visual_references", {}).get("external") or {}
        return bool(external.get("target_id") and external["target_id"] == prediction.get("target_id"))

    def _failure(self, reason):
        if reason in NORMAL_FIELD_ENDS:
            self.normal_field_refreshes += 1
            self.refresh_reason = reason
            return
        if reason in REFERENCE_FAILURES:
            self.reference_failure_events.append({"step": self.step_index,
                "target_id": self.prediction["target_id"], "phase": self.prediction["phase"],
                "reason": reason})
            self.reference_failure_events = self.reference_failure_events[-8:]
        super()._failure(reason)

    def _observable_feedback(self):
        result = super()._observable_feedback()
        prediction = self.prediction or {}
        external = prediction.get("visual_references", {}).get("external") or {}
        visual = result["visual"]
        same_target = self._anchor_is_interaction_target()
        visual.update(interaction_target_id=prediction.get("target_id"),
                      external_stationary_anchor_id=external.get("target_id"),
                      image_goal_targets_interaction=same_target,
                      anchor_relative_projection_error_px=visual["external_goal_error_px"])
        if not same_target:
            # Direct grasp/test motion can observe the payload relative to an
            # independent stationary landmark. That error must not be reported
            # as task-goal alignment or drive corrections to the metric goal.
            visual["external_goal_error_px"] = None
            visual["same_goal_image_improvement_px"] = None
        # Counters are available even during construction/reset. This advice
        # changes no action, pose, collision constraint or completion signal.
        result["controller_handoff"] = {
            "recent_reference_failures": list(getattr(self, "reference_failure_events", [])),
            "normal_field_refreshes": getattr(self, "normal_field_refreshes", 0),
            "interaction_readiness_certified": False,
            "decision_rule": (
                "Choose the next controller from current RGB and robot state. Use Field for "
                "remaining noncontact travel. For near-contact visual alignment or engagement, "
                "use Direct5 when the live views support the action. External tracking loss "
                "does not establish collision or require a retreat/calibration. If interaction "
                "geometry is still ambiguous, choose a changed informative view; do not keep "
                "reissuing the same failed feature and motion.")}
        return result

    def act(self, obs):
        decision = super().act(obs)
        decision.diagnostics["policy_version"] = "visual-field-policy-v5"
        return decision

    def observe_transition(self, before, decision, after):
        row = super().observe_transition(before, decision, after)
        started = time.monotonic()
        selection = decision.diagnostics.get("selection") or {}
        error = self.monitor.last_error if self._anchor_is_interaction_target() else None
        # For a straight, predicted-clear route, persistent movement AWAY from
        # the declared image objective contradicts that goal hypothesis. Do not
        # apply this to an obstacle detour, where image error may first increase.
        supported = (self.field_configured and self._anchor_is_interaction_target()
            and selection.get("accepted")
            and not self.refresh_reason and self.monitor.response_ready
            and self.initial_visual_error_px is not None and error is not None
            and self.local.measured_travel_m >= .01
            and self.local.checked.segment_valid(after.state[:3], self.goal))
        diverging = supported and error > self.initial_visual_error_px+4.
        self.visual_divergence_streak = self.visual_divergence_streak+1 if diverging else 0
        if self.visual_divergence_streak >= 3:
            self._failure("visual_goal_divergence")
        self.feedback = self._observable_feedback()
        row.update(self.feedback)
        row["visual_goal_check"] = {"initial_error_px": self.initial_visual_error_px,
            "current_error_px": error, "divergence_streak": self.visual_divergence_streak,
            "straight_route_check_supported": bool(supported), "task_success_certified": False}
        self.local_total_s += time.monotonic()-started
        return row
