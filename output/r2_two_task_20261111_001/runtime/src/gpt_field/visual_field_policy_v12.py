"""Field/Direct/gripper routing with measured-motion and image-risk feedback."""
from __future__ import annotations

import copy
import time
import numpy as np

from .policy import Decision
from .visual_field_policy import images
from .visual_field_policy_v10 import VisualFieldPolicyV10
from .visual_field_policy_v11 import VisualFieldPolicyV11
from .visual_field_v8 import measured_current_pose_hold_action
from .visual_field_v12 import MetricFieldExecutorV12
from .visual_monitor_v12 import RoleAwareVisualMonitorV12
from .visual_translation_guard_v12 import ProjectedTranslationGuardV12
from .robot_translation_response import MeasuredTranslationResponse


class VisualFieldPolicyV12(VisualFieldPolicyV11):
    def reset(self):
        super().reset()
        self.monitor = RoleAwareVisualMonitorV12()
        self.robot_response = MeasuredTranslationResponse()
        self.translation_guard = ProjectedTranslationGuardV12()
        self.local = MetricFieldExecutorV12()
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)

    def _observable_feedback(self):
        result = super()._observable_feedback()
        response = getattr(self, "robot_response", None)
        result["robot_translation_response"] = None if response is None else response.summary()
        result["controller_handoff"]["decision_rule"] = (
            "Reuse the persistent Field goal for useful task travel. Projected overlap is an "
            "uncertain observation, not proof of collision or clearance. Follow current measured "
            "risk trends and bounded exploration; never renew a consumed allowance by renaming. "
            "Current tool envelope revision is separate from feature identity. Preserve known "
            "neighbors and metric uncertainty. Actual displacement and commanded displacement "
            "have different units/contracts; local response translates between them. Advance "
            "stage on observed progress, not repeated relative-goal addition. Neither local "
            "arrival nor visual agreement establishes task success.")
        return result

    def _refresh(self, obs):
        latency = VisualFieldPolicyV10._refresh(self, obs)
        started = time.monotonic()
        self.translation_guard.configure(images(obs), self.prediction["motion_context"],
            obs.state, target_id=self.prediction["target_id"], monitor=self.monitor)
        self.feedback = self._observable_feedback()
        elapsed = time.monotonic()-started
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def act(self, obs):
        if self.pending is not None:
            raise RuntimeError("Transition feedback required before the next action")
        if obs.joint_position is None or obs.joint_velocity is None:
            raise ValueError("V12 requires measured joint position and velocity")
        if self.prediction and self.prediction["controller"] == "direct" and not self.chunk and not self.refresh_reason:
            self.refresh_reason = "direct_chunk_complete"
        refreshed = self.prediction is None or bool(self.refresh_reason)
        self._refresh_local_s = 0.
        api_s = self._refresh(obs) if refreshed else 0.
        started = time.monotonic()
        controller = self.prediction["controller"]
        selection = admission = rejected_model_action = None
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
                delta = np.asarray(action[:3])*.05
                # A stale Field hypothesis cannot license a new Direct repair.
                admission = self.translation_guard.assess(delta, monitor=self.monitor,
                    nominal_clear=False, nominal_away=False, nominal_regions=[],
                    direct_observation=True,
                    translation_response=self.robot_response,
                    rotation_rad=float(np.linalg.norm(action[3:6])*.5), orientation_hold=False)
                cap = admission.get("cap_m")
                if not admission["accepted"] or cap is not None and np.linalg.norm(delta) > cap+1e-12:
                    rejected_model_action = list(action)
                    self.chunk = []
                    selection = self._rejected(admission["reason"] if not admission["accepted"]
                        else "direct_translation_exceeds_observed_allowance")
                    selection["diagnostics"]["translation_admission"] = copy.deepcopy(admission)
        elif controller == "gripper":
            selection = self.gripper.select(obs.state, obs.joint_position)
        else:
            measured_cap, reason = self.monitor.measured_control_limit()
            command_cap = .005 if measured_cap is not None else None
            external = self.prediction["visual_references"].get("external") or {}
            if external.get("moving_kind") == "payload":
                # Attachment has its own evidence contract and conservative
                # command bound. An image-response fit cannot prove a grasp.
                attachment_cap, attachment_reason = self.monitor.attachment_control_limit()
                reason = reason or attachment_reason
                if attachment_cap is not None:
                    command_cap = attachment_cap if command_cap is None else min(command_cap, attachment_cap)
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
                    selection = self.local.select(obs.state, obs.joint_position,
                        command_cap_m=command_cap, measured_cap_m=measured_cap)
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
            rejection_hold = {"purpose": "reset_persistent_osc_goal_to_current_measured_orientation",
                "orientation_refresh_rad": self.local.OSC_GOAL_REFRESH_RAD, "requested_motion_executed": False}
            self.translation_vetoes += int("translation" in selection.get("status", ""))
        elapsed = time.monotonic()-started
        self.local_total_s += elapsed
        diagnostics = {"step": self.step_index, "refreshed": refreshed, "controller": controller,
            "phase": self.prediction["phase"], "target_id": self.prediction["target_id"],
            "hypothesis_id": self.prediction["hypothesis_id"], "chunk_index": self.chunk_index,
            "chunk_step": self.step_index-self.block_start, "model_pipeline_s": api_s,
            "local_planning_s": elapsed+self._refresh_local_s, "selection": selection,
            "feedback": self._observable_feedback(), "refresh_reason": self.refresh_reason,
            "policy_version": "visual-field-policy-v12",
            "prediction": copy.deepcopy(self.prediction) if refreshed else None,
            "request_key": self.client.last_request_key if refreshed else None,
            "task_success_certified": False,
            "subgoal_update": copy.deepcopy(self.subgoal_update_result) if refreshed else None,
            "projected_rotation_risk": copy.deepcopy(self.rotation_screen),
            "scene_execution_limits": copy.deepcopy(self.scene_execution_limits),
            "translation_admission": copy.deepcopy(admission)}
        if rejection_hold:
            diagnostics["rejection_hold"] = rejection_hold
        if rejected_model_action is not None:
            diagnostics["rejected_model_action"] = list(rejected_model_action)
        decision = Decision(tuple(float(x) for x in action), diagnostics)
        self.pending = obs, decision
        self.step_index += 1
        return decision

    def observe_transition(self, before, decision, after):
        row = VisualFieldPolicyV10.observe_transition(self, before, decision, after)
        started = time.monotonic()
        self.robot_response.observe(before.state, decision.action, after.state,
            before_joints=before.joint_position, after_joints=after.joint_position)
        self.translation_guard.observe(images(after), after.state, decision.action, monitor=self.monitor)
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-started
        return row
