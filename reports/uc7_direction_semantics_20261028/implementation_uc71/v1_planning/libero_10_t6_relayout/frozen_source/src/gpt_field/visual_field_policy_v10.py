"""Field reuse with persistent world goals and displacement-aware evidence."""
from __future__ import annotations

import copy
import time
import numpy as np

from .observation import decode_image
from .policy import Decision
from .visual_field_policy import images
from .visual_field_policy_v5 import VisualFieldPolicyV5
from .visual_field_policy_v9 import VisualFieldPolicyV9
from .visual_field_v8 import measured_current_pose_hold_action
from .visual_field_v10 import MetricFieldExecutorV10, SubgoalLedger
from .visual_monitor_v10 import RoleAwareVisualMonitorV10
from .visual_rotation_guard_v8 import projected_rotation_risk


class VisualFieldPolicyV10(VisualFieldPolicyV9):
    def reset(self):
        super().reset()
        self.local = MetricFieldExecutorV10()
        self.monitor = RoleAwareVisualMonitorV10()
        self.subgoals = SubgoalLedger()
        self.subgoal_update_result = None
        self.subgoal_rejection = None

    def _observable_feedback(self):
        result = super()._observable_feedback()
        ledger = getattr(self, "subgoals", None)
        result["subgoal_ledger"] = None if ledger is None else ledger.summary()
        result["controller_handoff"]["decision_rule"] += (
            " Keep the stored world subgoal through reference or attachment repair; "
            "Direct motion can already reach it. Retain does not add a fresh relative lift. "
            "Change a wrong goal explicitly with replace and a reason; start a new stage "
            "after local arrival. Local arrival never establishes task completion. "
            "A below-useful-motion handoff requests a changed informative action, not "
            "repeated microscopic commands or renewed uncertainty debt.")
        return result

    def _refresh(self, obs):
        self.subgoals.observe(obs.state)
        self._flush()
        self.client.observable_context = {
            "visual_feedback": self._observable_feedback(),
            "refresh_reason": self.refresh_reason,
            "failed_hypotheses": copy.deepcopy(self.failed_hypotheses[-6:]),
            "attempts_started": len(self.attempts),
            "recovery_episodes_detected": self.recovery_episodes,
            "efficiency_goal": "Reuse Field for useful travel; preserve world goals across repairs and use measured arrival to advance local stages.",
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
        self.prediction = prediction  # Preserve the exact validated model output.
        self.chunk_index += 1
        self.block_start = self.step_index
        self._block_effective = 0
        self.monitor.configure(images(obs), prediction["visual_references"], obs.state,
                               intent_id=self.model_requests)
        self.initial_visual_error_px = self.monitor.last_error if self._anchor_is_interaction_target() else None
        self.visual_divergence_streak = 0
        self.refresh_reason = None
        self.rotation_screen = self.scene_execution_limits = None
        self.chunk = []
        self.goal = None
        self._block_initial_error = self._best_goal_error = None
        update = self.subgoal_update_result = self.subgoals.apply(obs.state, prediction)
        self.subgoal_rejection = None if update["accepted"] else update["status"]
        self.field_configured = prediction["controller"] == "visual_field" and update["accepted"]
        context = prediction.get("motion_context") or {}
        if self.field_configured:
            effective = copy.deepcopy(update["field_intent"])
            if context.get("obstacle_assessment") == "unobservable":
                effective["max_travel_m"] = min(effective["max_travel_m"], .015)
                effective["command_limit_m"] = min(effective["command_limit_m"], .005)
                self.scene_execution_limits = {
                    "reason": "model_declared_scene_geometry_unobservable",
                    "max_travel_m": effective["max_travel_m"],
                    "command_limit_m": effective["command_limit_m"],
                    "goal_and_regions_preserved": True, "clearance_certified": False,
                }
            self.local.configure(obs.state, effective, scene_assessment=copy.deepcopy(context),
                                 **update["executor_context"])
            self.goal = self.local.goal.copy()
            self._block_initial_error = self._best_goal_error = float(np.linalg.norm(self.goal-obs.state[:3]))
        elif update["accepted"]:
            if prediction["controller"] == "gripper":
                self.gripper.configure(obs.state, prediction["gripper_intent"])
            else:
                self.chunk = copy.deepcopy(prediction["direct_actions"])
                self.rotation_screen = projected_rotation_risk(context, self.chunk,
                    image_size=decode_image(obs.image).size)
                self.rotation_screen["observation_step"] = self.step_index
        self.feedback = self._observable_feedback()
        self._refresh_local_s = time.monotonic()-started
        self.local_total_s += self._refresh_local_s
        return latency

    def act(self, obs):
        if self.pending is not None:
            raise RuntimeError("Transition feedback required before the next action")
        if obs.joint_position is None or obs.joint_velocity is None:
            raise ValueError("V10 requires measured joint position and velocity")
        if self.prediction and self.prediction["controller"] == "direct" and not self.chunk and not self.refresh_reason:
            self.refresh_reason = "direct_chunk_complete"
        refreshed = self.prediction is None or bool(self.refresh_reason)
        self._refresh_local_s = 0.
        api_s = self._refresh(obs) if refreshed else 0.
        started = time.monotonic()
        controller = self.prediction["controller"]
        selection = None
        rejected_model_action = None
        if self.subgoal_rejection:
            selection = self._rejected(self.subgoal_rejection)
        elif controller == "direct":
            if self.rotation_screen and not self.rotation_screen["accepted"]:
                rejected_model_action = self.chunk[0]
                self.chunk = []
                self.rotation_vetoes += 1
                selection = self._rejected(self.rotation_screen["reason"])
            else:
                action = tuple(self.chunk.pop(0))
        elif controller == "gripper":
            selection = self.gripper.select(obs.state, obs.joint_position)
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
                        if self.subgoals.record_visual_correction(proposed):
                            self.goal = proposed.copy()
                            self.visual_corrections += 1
                            self.last_correction_step = self.step_index
                            self._best_goal_error = float(np.linalg.norm(self.goal-np.asarray(obs.state[:3])))
                        else:
                            self.local.retarget(self.goal)
                            selection = self._rejected("subgoal_visual_correction_requires_explicit_replan")
                if selection is None:
                    selection = self.local.select(obs.state, obs.joint_position, command_cap_m=cap)
        if selection is not None:
            action = tuple(selection["action"])
            if selection.get("need_refresh"):
                if selection["status"] in ("goal_reached", "measured_goal_arrival", "measured_local_goal_arrival", "intent_complete", "subgoal_local_complete"):
                    self.refresh_reason = selection["status"]
                else:
                    self._failure(selection["status"])
        rejection_hold = None
        if selection is not None and not selection.get("accepted"):
            action = tuple(measured_current_pose_hold_action())
            selection["action"] = list(action)
            rejection_hold = {
                "purpose": "reset_persistent_osc_goal_to_current_measured_orientation",
                "orientation_refresh_rad": self.local.OSC_GOAL_REFRESH_RAD,
                "requested_motion_executed": False,
            }
        local_s = time.monotonic()-started
        self.local_total_s += local_s
        diagnostics = {
            "step": self.step_index, "refreshed": refreshed, "controller": controller,
            "phase": self.prediction["phase"], "target_id": self.prediction["target_id"],
            "hypothesis_id": self.prediction["hypothesis_id"], "chunk_index": self.chunk_index,
            "chunk_step": self.step_index-self.block_start, "model_pipeline_s": api_s,
            "local_planning_s": local_s+self._refresh_local_s, "selection": selection,
            "feedback": self._observable_feedback(), "refresh_reason": self.refresh_reason,
            "policy_version": "visual-field-policy-v10",
            "prediction": copy.deepcopy(self.prediction) if refreshed else None,
            "request_key": self.client.last_request_key if refreshed else None,
            "task_success_certified": False,
            "subgoal_update": copy.deepcopy(self.subgoal_update_result) if refreshed else None,
            "projected_rotation_risk": copy.deepcopy(self.rotation_screen),
            "scene_execution_limits": copy.deepcopy(self.scene_execution_limits),
        }
        if rejection_hold:
            diagnostics["rejection_hold"] = rejection_hold
        if rejected_model_action is not None:
            diagnostics["rejected_model_action"] = list(rejected_model_action)
        decision = Decision(tuple(float(x) for x in action), diagnostics)
        self.pending = obs, decision
        self.step_index += 1
        return decision

    def observe_transition(self, before, decision, after):
        # An invalid ledger request never configures/executes a new gripper block.
        if self.subgoal_rejection and decision.diagnostics["controller"] == "gripper":
            row = VisualFieldPolicyV5.observe_transition(self, before, decision, after)
        else:
            row = super().observe_transition(before, decision, after)
        started = time.monotonic()
        self.subgoals.observe(after.state)
        if self.field_configured and self.subgoals.summary()["local_complete"] and not self.refresh_reason:
            self.refresh_reason = "subgoal_local_complete"
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-started
        return row
