"""Persistent observed correspondences and axis-constrained metric field control.

Spatial goals, contact decisions and gripper commands remain model directed.
Local execution never receives environment truth or declares task completion.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from .visual_field_policy import Decision, images
from .visual_field_policy_v5 import VisualFieldPolicyV5, NORMAL_FIELD_ENDS
from .visual_field_v6 import MetricFieldExecutorV6, GripperSettleExecutor
from .visual_monitor_v6 import RoleAwareVisualMonitorV6


class VisualFieldPolicyV6(VisualFieldPolicyV5):
    def reset(self):
        super().reset()
        self.local = MetricFieldExecutorV6()
        self.monitor = RoleAwareVisualMonitorV6()
        self.gripper = GripperSettleExecutor()
        self.gripper_actions = 0
        self.last_field_handoff = None

    def _refresh(self, obs):
        self._flush()
        self.client.observable_context = {
            "visual_feedback": self._observable_feedback(),
            "refresh_reason": self.refresh_reason,
            "failed_hypotheses": copy.deepcopy(self.failed_hypotheses[-6:]),
            "attempts_started": len(self.attempts),
            "recovery_episodes_detected": self.recovery_episodes,
            "efficiency_goal": "Retain reliable correspondences and reuse Field goals. Separate horizontal alignment from vertical approach; settle model-commanded gripper changes locally.",
            "task_success_certified": False,
        }
        snapshot = getattr(self.client, "budget_snapshot", None)
        before_calls = None if snapshot is None else snapshot()["api_calls"]
        started = time.monotonic()
        try:
            prediction = self.client.predict(obs.to_dict())
        finally:
            self.model_requests += 1 if before_calls is None else snapshot()["api_calls"]-before_calls
        latency = time.monotonic()-started
        started = time.monotonic()
        self.intent_refreshes += 1
        self.phase_previous = self.prediction["phase"] if self.prediction else None
        self.prediction = prediction
        self.chunk_index += 1
        self.block_start = self.step_index
        self._block_effective = 0
        self.monitor.configure(images(obs), prediction["visual_references"], obs.state,
                               intent_id=self.model_requests)
        self.initial_visual_error_px = self.monitor.last_error if self._anchor_is_interaction_target() else None
        self.visual_divergence_streak = 0
        self.refresh_reason = None
        self.field_configured = prediction["controller"] == "visual_field"
        self.chunk = []
        if self.field_configured:
            self.local.configure(obs.state, prediction["field_intent"])
            self.goal = self.local.goal.copy()
            self._block_initial_error = self._best_goal_error = float(np.linalg.norm(self.goal-obs.state[:3]))
        else:
            self.goal = None
            self._block_initial_error = self._best_goal_error = None
            if prediction["controller"] == "gripper":
                self.gripper.configure(obs.state, prediction["gripper_intent"])
            else:
                self.chunk = copy.deepcopy(prediction["direct_actions"])
        self.feedback = self._observable_feedback()
        self._refresh_local_s = time.monotonic()-started
        self.local_total_s += self._refresh_local_s
        return latency

    def _allows_planar_correction(self):
        return (super()._allows_planar_correction()
                and self.prediction["field_intent"]["motion_axes"] != "z")

    @property
    def diagnostic_counters(self):
        result = super().diagnostic_counters
        result["gripper_settle_actions"] = getattr(self, "gripper_actions", 0)
        return result

    def _observable_feedback(self):
        result = super()._observable_feedback()
        prediction = self.prediction or {}
        gripper = getattr(self, "gripper", None)
        result["gripper_execution"] = (gripper.summary() if gripper is not None
            and prediction.get("controller") == "gripper" else None)
        result["last_field_handoff"] = copy.deepcopy(getattr(self, "last_field_handoff", None))
        result["controller_handoff"]["decision_rule"] = (
            "Retain reliable visual references instead of re-guessing their coordinates. "
            "Use Field for approach, horizontal alignment and bounded vertical placement approach; "
            "vertical-only Field preserves its starting horizontal position. A model-commanded "
            "gripper block holds the tool and ends on measured finger settling or its finite budget. "
            "Settled fingers do not establish attachment. Direct5 remains for complex interaction "
            "or reorientation. Image overlap and stable support alone do not establish precise "
            "placement; assess object-relative target alignment before opening.")
        return result

    def act(self, obs):
        if self.pending is not None:
            raise RuntimeError("Transition feedback required before the next action")
        if obs.joint_position is None or obs.joint_velocity is None:
            raise ValueError("V6 requires measured joint position and velocity")
        if self.prediction and self.prediction["controller"] == "direct" and not self.chunk:
            self.refresh_reason = "direct_chunk_complete"
        refreshed = self.prediction is None or bool(self.refresh_reason)
        self._refresh_local_s = 0.
        api_s = self._refresh(obs) if refreshed else 0.
        started = time.monotonic()
        controller = self.prediction["controller"]
        selection = None
        if controller == "direct":
            action = tuple(self.chunk.pop(0))
        elif controller == "gripper":
            selection = self.gripper.select(obs.state, obs.joint_position)
            action = tuple(selection["action"])
            if selection.get("need_refresh"):
                self.refresh_reason = selection["status"]
        else:
            cap, reason = self.monitor.control_limit()
            external = self.prediction["visual_references"].get("external") or {}
            if external.get("moving_kind") == "payload":
                attachment_cap, attachment_reason = self.monitor.attachment_control_limit()
                reason = reason or attachment_reason
                if attachment_cap is not None:
                    cap = attachment_cap if cap is None else min(cap, attachment_cap)
            if reason:
                selection = self._rejected(reason)
            else:
                if self._allows_planar_correction() and self.step_index-self.last_correction_step >= 5:
                    proposed = self.monitor.planar_correction(self.goal)
                    if proposed is not None and self.local.retarget(proposed):
                        self.goal = proposed.copy()
                        self.visual_corrections += 1
                        self.last_correction_step = self.step_index
                        self._best_goal_error = float(np.linalg.norm(self.goal-np.asarray(obs.state[:3])))
                selection = self.local.select(obs.state, obs.joint_position, command_cap_m=cap)
            action = tuple(selection["action"])
            if selection.get("need_refresh"):
                if selection["status"] in ("goal_reached", "measured_goal_arrival", "measured_local_goal_arrival", "intent_complete"):
                    self.refresh_reason = selection["status"]
                else:
                    self._failure(selection["status"])
        local_s = time.monotonic()-started
        self.local_total_s += local_s
        decision = Decision(tuple(float(x) for x in action), {
            "step": self.step_index, "refreshed": refreshed, "controller": controller,
            "phase": self.prediction["phase"], "target_id": self.prediction["target_id"],
            "hypothesis_id": self.prediction["hypothesis_id"], "chunk_index": self.chunk_index,
            "chunk_step": self.step_index-self.block_start, "model_pipeline_s": api_s,
            "local_planning_s": local_s+self._refresh_local_s, "selection": selection,
            "feedback": self._observable_feedback(), "refresh_reason": self.refresh_reason,
            "policy_version": "visual-field-policy-v6",
            "prediction": copy.deepcopy(self.prediction) if refreshed else None,
            "request_key": self.client.last_request_key if refreshed else None,
            "task_success_certified": False,
        })
        self.pending = obs, decision
        self.step_index += 1
        return decision

    def observe_transition(self, before, decision, after):
        # Parent acknowledges every pending transition and updates the same RGB,
        # robot-state, counters and measured Field execution. A gripper block
        # has its own additional, pose-bounded executor; it is not Direct replay.
        row = super().observe_transition(before, decision, after)
        started = time.monotonic()
        if decision.diagnostics["controller"] == "gripper":
            self.gripper.observe(before.state, decision.action, after.state)
            self.gripper_actions += int(bool((decision.diagnostics.get("selection") or {}).get("accepted")))
            reason = self.gripper.summary().get("stop_reason")
            if reason:
                self.refresh_reason = reason
        if self.field_configured:
            self.last_field_handoff = {
                "target_id": self.prediction["target_id"], "phase": self.prediction["phase"],
                "motion_axes": self.prediction["field_intent"]["motion_axes"],
                "goal_xyz": self.goal.tolist(), "measured_xyz": list(after.state[:3]),
                "goal_error_m": float(np.linalg.norm(self.goal-np.asarray(after.state[:3]))),
                "reason": self.refresh_reason, "task_success_certified": False,
            }
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-started
        return row
