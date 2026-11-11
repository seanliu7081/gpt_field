"""Metric Field with measured orientation hold and evidence-aware RGB reuse."""
from __future__ import annotations

import copy
import time

from .observation import decode_image
from .policy import Decision
from .visual_field_policy_v4 import VisualFieldPolicyV4
from .visual_field_policy_v7 import VisualFieldPolicyV7
from .visual_field_v8 import MetricFieldExecutorV8, GripperSettleExecutorV8, measured_current_pose_hold_action
from .visual_monitor_v8 import RoleAwareVisualMonitorV8
from .visual_rotation_guard_v8 import projected_rotation_risk


class VisualFieldPolicyV8(VisualFieldPolicyV7):
    def reset(self):
        super().reset()
        self.local = MetricFieldExecutorV8()
        self.monitor = RoleAwareVisualMonitorV8()
        self.gripper = GripperSettleExecutorV8()
        self.rotation_screen = None
        self.rotation_vetoes = 0
        self.scene_execution_limits = None

    @property
    def diagnostic_counters(self):
        result = super().diagnostic_counters
        result["direct_rotation_vetoes"] = getattr(self, "rotation_vetoes", 0)
        return result

    def _observable_feedback(self):
        result = super()._observable_feedback()
        result["projected_rotation_risk"] = copy.deepcopy(getattr(self, "rotation_screen", None))
        result["scene_execution_limits"] = copy.deepcopy(getattr(self, "scene_execution_limits", None))
        # Replace inherited advice: inconclusive response must not itself be a
        # request to spin the tool or permanently discard a good correspondence.
        result["controller_handoff"]["decision_rule"] = (
            "Keep Field for useful approach and transport. Local bounded orientation feedback "
            "holds its measured starting orientation; it does not command a new task pose. "
            "Read response_evidence: weak image response is inconclusive, not proof of a wrong "
            "feature or physical misalignment. Retain a reliable correspondence while finite "
            "uncertainty allowance remains; do not rotate just to repair tracking. A lost or "
            "contradicted reference may need a different visible seed. Near interaction choose "
            "a task-directed contact correction from current RGB, not a repeated observation "
            "routine. Before task rotation inspect the whole tool/payload and nearby obstacles "
            "in external RGB; zero EEF translation does not prevent fingertip swept contact. "
            "A projected-risk veto requires a changed action; it does not authorize deleting "
            "obstacles, shrinking the tool envelope or relabeling reference repair. "
            "Local arrival, settled fingers and image overlap never establish task completion.")
        return result

    def _refresh(self, obs):
        # V6 labels any empty Direct queue "complete". A veto deliberately
        # discards its queue, so retain the actual interruption reason for both
        # the flushed history block and the next model observation.
        if (self.prediction and self.prediction["controller"] == "direct"
                and self.rotation_screen and not self.rotation_screen["accepted"]
                and self.refresh_reason == "direct_chunk_complete"):
            self.refresh_reason = self.rotation_screen["reason"]
        latency = super()._refresh(obs)
        started = time.monotonic()
        self.rotation_screen = None
        self.scene_execution_limits = None
        context = self.prediction.get("motion_context") or {}
        if (self.prediction["controller"] == "visual_field"
                and context.get("obstacle_assessment") == "unobservable"):
            # Reliable feature motion cannot resolve unknown surrounding scene
            # geometry. Preserve all goals/regions, but bound this observation's
            # reuse explicitly rather than silently treating it as clear space.
            effective = copy.deepcopy(self.local.intent)
            effective["max_travel_m"] = min(effective["max_travel_m"], .015)
            effective["command_limit_m"] = min(effective["command_limit_m"], .005)
            self.local.configure(obs.state, effective)
            self.scene_execution_limits = {
                "reason": "model_declared_scene_geometry_unobservable",
                "max_travel_m": effective["max_travel_m"],
                "command_limit_m": effective["command_limit_m"],
                "goal_and_regions_preserved": True,
                "clearance_certified": False,
            }
        if self.prediction["controller"] == "direct":
            self.rotation_screen = projected_rotation_risk(
                self.prediction.get("motion_context"), self.chunk,
                image_size=decode_image(obs.image).size)
            self.rotation_screen["observation_step"] = self.step_index
        elapsed = time.monotonic()-started
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def act(self, obs):
        decision = super().act(obs)
        started = time.monotonic()
        if (decision.diagnostics["controller"] == "direct"
                and self.rotation_screen and not self.rotation_screen["accepted"]):
            # Parent has selected but not executed the command. Discard the
            # whole rejected block and request a new intent.
            self.chunk = []
            self.rotation_vetoes += 1
            reason = self.rotation_screen["reason"]
            # This post-selection veto is a motion-risk event, not a semantic
            # reference failure. Record the action index already allocated by
            # the parent without polluting its reference-repair ledger.
            VisualFieldPolicyV4._failure(self, reason)
            self.events[-1]["step"] = self.step_index-1
            diagnostics = copy.deepcopy(decision.diagnostics)
            diagnostics.update(selection={"action": [0.]*7, "accepted": False,
                "status": reason, "need_refresh": True,
                "diagnostics": copy.deepcopy(self.rotation_screen)}, refresh_reason=reason,
                rejected_model_action=list(decision.action), feedback=self._observable_feedback())
            decision = Decision((0.,)*7, diagnostics)
            self.pending = obs, decision
        selection = decision.diagnostics.get("selection")
        if selection is not None and not selection.get("accepted"):
            # Exactly-zero rotation preserves the installed OSC's previous
            # unfinished target. Explicitly set its target to the current
            # measured pose using the validated nanoradian refresh convention.
            # No requested motion from the rejected selection is executed.
            hold = tuple(measured_current_pose_hold_action())
            diagnostics = copy.deepcopy(decision.diagnostics)
            diagnostics["selection"]["action"] = list(hold)
            diagnostics["rejection_hold"] = {
                "purpose": "reset_persistent_osc_goal_to_current_measured_orientation",
                "orientation_refresh_rad": MetricFieldExecutorV8.OSC_GOAL_REFRESH_RAD,
                "requested_motion_executed": False,
            }
            decision = Decision(hold, diagnostics)
            self.pending = obs, decision
        decision.diagnostics["policy_version"] = "visual-field-policy-v8"
        decision.diagnostics["projected_rotation_risk"] = copy.deepcopy(self.rotation_screen)
        decision.diagnostics["scene_execution_limits"] = copy.deepcopy(self.scene_execution_limits)
        elapsed = time.monotonic()-started
        self.local_total_s += elapsed
        decision.diagnostics["local_planning_s"] += elapsed
        return decision
