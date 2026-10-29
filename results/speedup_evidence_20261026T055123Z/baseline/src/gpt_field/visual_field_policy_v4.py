"""Persistent robot-space distance planning with role-aware visual feedback.

GPT chooses a bounded non-contact hypothesis or Direct5 contact actions. Local
planning executes the hypothesis against predicted geometry, acknowledges every
robot transition, and requests GPT on useful events rather than every action.
"""
from __future__ import annotations

import copy
import time
import numpy as np

from .policy import Decision
from .visual_field_policy import VisualFieldPolicy, images
from .visual_field_v4 import MetricFieldExecutorV4
from .visual_monitor_v4 import RoleAwareVisualMonitor


class VisualFieldPolicyV4(VisualFieldPolicy):
    def reset(self):
        super().reset()
        self.local = MetricFieldExecutorV4()
        self.monitor = RoleAwareVisualMonitor()
        self.field_configured = False
        self.field_effective_actions = 0
        self.field_measured_path_m = 0.
        self.field_nonzero_actions = 0
        self.field_goal_improvement_m = 0.
        self.model_requests = 0
        self.visual_corrections = 0
        self.last_correction_step = -5
        self.unchanged_recoveries = 0
        self._last_failure = None
        self._block_initial_error = None
        self._best_goal_error = None
        self._block_effective = 0
        self._block_baseline_path = 0.
        self._refresh_local_s = 0.

    @property
    def diagnostic_counters(self):
        return {"field_nonzero_actions": self.field_nonzero_actions,
            "field_effective_actions": self.field_effective_actions,
            "field_measured_path_m": self.field_measured_path_m,
            "field_goal_improvement_m": self.field_goal_improvement_m,
            "model_requests": self.model_requests, "visual_goal_corrections": self.visual_corrections,
            "repeated_same_failure": self.unchanged_recoveries,
            "field_effective_actions_per_request": self.field_effective_actions/max(1, self.model_requests),
            "requests_per_field_measured_cm": None if self.field_measured_path_m < 1e-6 else
                self.model_requests/(100*self.field_measured_path_m)}

    def _observable_feedback(self):
        return {"visual": self.monitor.summary(),
            "field": self.local.summary() if self.field_configured else None,
            "efficiency": self.diagnostic_counters,
            "same_failure_repeats": self.unchanged_recoveries,
            "task_success_certified": False}

    def _flush(self):
        if not self.block:
            return
        first, last = self.block[0], self.block[-1]
        self.client.record_block({"type": "visual_execution_block", "chunk_index": self.chunk_index,
            "steps": len(self.block), "first_step": first["step"], "last_step": last["step"],
            "before_state": first["before_state"], "after_state": last["after_state"],
            "joint_position": last["joint_position"],
            "measured_path_m": sum(row["measured_translation_m"] for row in self.block),
            "commanded_path_m": sum(row["commanded_translation_m"] for row in self.block),
            "phase": self.prediction["phase"], "target_id": self.prediction["target_id"],
            "hypothesis_id": self.prediction["hypothesis_id"], "controller": self.prediction["controller"],
            "end_visual_feedback": self._observable_feedback(), "refresh_reason": self.refresh_reason,
            "effective_field_actions": self._block_effective,
            "task_success_certified": False})
        self.block = []

    def _refresh(self, obs):
        self._flush()
        self.client.observable_context = {"visual_feedback": self._observable_feedback(),
            "refresh_reason": self.refresh_reason, "failed_hypotheses": copy.deepcopy(self.failed_hypotheses[-6:]),
            "attempts_started": len(self.attempts), "recovery_episodes_detected": self.recovery_episodes,
            "efficiency_goal": "Reuse a validated non-contact field; Direct5 is for interaction or changed observation.",
            "task_success_certified": False}
        started = time.monotonic()
        prediction = self.client.predict(obs.to_dict())
        latency = time.monotonic()-started
        started = time.monotonic()
        self.model_requests += 1
        self.phase_previous = self.prediction["phase"] if self.prediction else None
        self.prediction = prediction
        self.chunk_index += 1
        self.block_start = self.step_index
        self._block_effective = 0
        self.monitor.configure(images(obs), prediction["visual_references"], obs.state,
                               intent_id=self.model_requests)
        self.refresh_reason = None
        self.field_configured = prediction["controller"] == "visual_field"
        if self.field_configured:
            intent = prediction["field_intent"]
            self.local.configure(obs.state, intent)
            self.goal = np.asarray(obs.state[:3])+intent["goal_offset_m"]
            self._block_initial_error = self._best_goal_error = float(np.linalg.norm(self.goal-obs.state[:3]))
            self.chunk = []
        else:
            self.chunk = copy.deepcopy(prediction["direct_actions"])
            self.goal = None
            self._block_initial_error = self._best_goal_error = None
        self.feedback = self._observable_feedback()
        self._refresh_local_s = time.monotonic()-started
        self.local_total_s += self._refresh_local_s
        return latency

    def _failure(self, reason):
        signature = (self.prediction["target_id"], self.prediction["phase"], reason)
        self.unchanged_recoveries = self.unchanged_recoveries+1 if signature == self._last_failure else 1
        self._last_failure = signature
        event = {"step": self.step_index, "chunk_index": self.chunk_index,
            "reason": reason, "target_id": self.prediction["target_id"],
            "same_failure_repeats": self.unchanged_recoveries, "episode_stop": False}
        self.events.append(event)
        self.failed_hypotheses.append(event)
        self.recovery_episodes += 1
        self.refresh_reason = reason

    @staticmethod
    def _rejected(reason):
        return {"action": [0.]*7, "accepted": False, "need_refresh": True,
                "status": reason, "diagnostics": {"clearance_certified": False}}

    def _allows_planar_correction(self):
        return self.prediction["phase"] == "align"

    def act(self, obs):
        if self.pending is not None:
            raise RuntimeError("Transition feedback required before the next action")
        if obs.joint_position is None or obs.joint_velocity is None:
            raise ValueError("V4 requires measured joint position and velocity")
        if self.prediction and self.prediction["controller"] == "direct" and not self.chunk:
            self.refresh_reason = "direct_chunk_complete"
        refreshed = self.prediction is None or bool(self.refresh_reason)
        self._refresh_local_s = 0.
        api_s = self._refresh(obs) if refreshed else 0.
        started = time.monotonic()
        selection = None
        if self.prediction["controller"] == "direct":
            action = tuple(self.chunk.pop(0))
        else:
            cap, reason = self.monitor.control_limit()
            ext = self.prediction["visual_references"].get("external") or {}
            needs_attachment = self.prediction["phase"] == "transport" or (
                self.prediction["phase"] == "align" and ext.get("moving_kind") == "payload")
            if needs_attachment and not self.monitor.attachment_evidence:
                reason = "payload_co_motion_required_use_direct_test"
            if reason:
                selection = self._rejected(reason)
            else:
                if self._allows_planar_correction() and self.step_index-self.last_correction_step >= 5:
                    proposed = self.monitor.planar_correction(self.goal)
                    if proposed is not None and self.local.retarget(proposed):
                        self.goal = proposed.copy()
                        self.visual_corrections += 1
                        self.last_correction_step = self.step_index
                        # A revised goal cannot manufacture progress against the prior goal.
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
            "step": self.step_index, "refreshed": refreshed, "controller": self.prediction["controller"],
            "phase": self.prediction["phase"], "target_id": self.prediction["target_id"],
            "hypothesis_id": self.prediction["hypothesis_id"], "chunk_index": self.chunk_index,
            "chunk_step": self.step_index-self.block_start, "model_pipeline_s": api_s,
            "local_planning_s": local_s+self._refresh_local_s, "selection": selection, "feedback": self._observable_feedback(),
            "refresh_reason": self.refresh_reason, "policy_version": "visual-field-policy-v4",
            "prediction": copy.deepcopy(self.prediction) if refreshed else None,
            "request_key": self.client.last_request_key if refreshed else None,
            "task_success_certified": False})
        self.pending = obs, decision
        self.step_index += 1
        return decision

    def _post_field_control_limit(self, after):
        return self.monitor.control_limit()

    def observe_transition(self, before, decision, after):
        if self.pending is None or self.pending[0] != before or self.pending[1] is not decision:
            raise ValueError("Transition must acknowledge the pending decision")
        started = time.monotonic()
        selected = decision.diagnostics.get("selection") or {}
        if self.field_configured and selected.get("accepted"):
            self.local.observe(before.state, decision.action, after.state)
        self.monitor.observe(images(after), after.state, decision.action)
        measured = float(np.linalg.norm(np.subtract(after.state[:3], before.state[:3])))
        commanded = float(np.linalg.norm(decision.action[:3])*.05)
        if self.field_configured and selected.get("accepted") and commanded > 1e-9:
            self.field_nonzero_actions += 1
            self.field_measured_path_m += measured
            if measured >= .0001:
                self.field_effective_actions += 1
                self._block_effective += 1
            error = float(np.linalg.norm(self.goal-np.asarray(after.state[:3])))
            if error < self._best_goal_error:
                self.field_goal_improvement_m += self._best_goal_error-error
                self._best_goal_error = error
            cap, reason = self._post_field_control_limit(after)
            if reason:
                self._failure(reason)
            info = self.local.summary()
            local_reason = info.get("stop_reason")
            if local_reason:
                self.refresh_reason = local_reason
        grip = decision.action[6]
        if grip > 0 and self.last_gripper < 0:
            self.attempts.append({"step": self.step_index-1, "target_id": self.prediction["target_id"],
                "hypothesis_id": self.prediction["hypothesis_id"],
                "definition": "open-to-close command; not certified grasp"})
        if grip:
            self.last_gripper = 1. if grip > 0 else -1.
        self.feedback = self._observable_feedback()
        row = {"type": "observable_transition", "step": self.step_index-1,
            "before_state": list(before.state), "after_state": list(after.state),
            "joint_position": list(after.joint_position), "joint_velocity": list(after.joint_velocity),
            "action": list(decision.action), "measured_translation_m": measured,
            "commanded_translation_m": commanded, **copy.deepcopy(self.feedback)}
        self.block.append(row)
        self.pending = None
        self.local_total_s += time.monotonic()-started
        return row
