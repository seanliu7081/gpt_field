"""Part-aware target contact and local two-view refinement through Field planning."""
from __future__ import annotations

import copy
import time
import numpy as np

from .policy import Decision
from .visual_field_policy import images
from .visual_field_policy_v10 import VisualFieldPolicyV10
from .visual_field_policy_v11 import VisualFieldPolicyV11
from .visual_field_v8 import measured_current_pose_hold_action
from .visual_field_v13 import MetricFieldExecutorV13, ContactSubgoalLedger
from .contact_geometry_v13 import TargetContactGeometry
from .visual_contact_alignment_v13 import VisualContactAlignmentV13
from .visual_monitor_v12 import RoleAwareVisualMonitorV12
from .visual_translation_guard_v12 import ProjectedTranslationGuardV12
from .robot_translation_response import MeasuredTranslationResponse


def compact_target_geometry(value):
    if value is None:
        return None
    result = {key: copy.deepcopy(item) for key, item in value.items()
              if key not in ("core_parts", "uncertainty_halo_parts", "tool_core_parts", "full_halo_parts")}
    result["constrained_part_names"] = [part["tool_part"] for part in value.get("core_parts", [])]
    result["part_coordinates_in_local_diagnostics"] = True
    return result


class VisualFieldPolicyV13(VisualFieldPolicyV11):
    def reset(self):
        super().reset()
        self.monitor = RoleAwareVisualMonitorV12()
        self.robot_response = MeasuredTranslationResponse()
        self.translation_guard = ProjectedTranslationGuardV12()
        self.local = MetricFieldExecutorV13()
        self.subgoals = ContactSubgoalLedger()
        self.target_contact = TargetContactGeometry()
        self.contact_alignment = VisualContactAlignmentV13()
        self.last_contact_refinement = None
        self.contact_geometry_vetoes = 0
        self.last_target_admission = None
        self.last_carried_admission = None
        self._last_grasp_candidate_id = None
        self._release_commanded = False
        self.local.bind_translation_guard(self.translation_guard, self.monitor, self.robot_response)

    def _observable_feedback(self):
        result = super()._observable_feedback()
        response = getattr(self, "robot_response", None)
        result["robot_translation_response"] = None if response is None else response.summary()
        geometry = getattr(self, "target_contact", None)
        alignment = getattr(self, "contact_alignment", None)
        result["target_contact_geometry"] = None if geometry is None else compact_target_geometry(geometry.summary())
        if isinstance(result.get("field"), dict):
            result["field"]["target_contact_geometry"] = compact_target_geometry(
                result["field"].get("target_contact_geometry"))
        for key in ("last_target_admission", "last_carried_admission"):
            admission = getattr(self, key, None)
            result[key] = None if admission is None else {
                name: copy.deepcopy(admission.get(name)) for name in ("accepted", "reason", "need_refresh")}
            if admission is not None:
                detail = admission.get("diagnostics", {})
                result[key]["intersections"] = copy.deepcopy(detail.get("intersections", []))
        result["contact_alignment"] = None if alignment is None else alignment.summary()
        result["last_contact_refinement"] = copy.deepcopy(getattr(self, "last_contact_refinement", None))
        result["controller_handoff"]["decision_rule"] = (
            "Reuse the persistent Field goal for useful task travel. Projected overlap is an "
            "uncertain observation, not proof of collision or clearance. Follow current measured "
            "risk trends and bounded exploration; never renew a consumed allowance by renaming. "
            "Current tool envelope revision is separate from feature identity. Preserve known "
            "neighbors and metric uncertainty. Actual displacement and commanded displacement "
            "have different units/contracts; local response translates between them. Advance "
            "stage on observed progress, not repeated relative-goal addition. Neither local "
            "arrival nor visual agreement establishes task success. Keep a selected local contact "
            "feature and the actual jaw pinch projection consistent across both views. A supported "
            "two-view contact objective can refine XYZ through Field; the image center is not the "
            "pinch point. Finger contact permission never permits palm contact with that object.")
        return result

    def _refresh(self, obs):
        # Parent first updates the exact model output and reference tracks. No
        # action is selected until the primary target is independently inserted.
        self.local.contact_geometry = None
        latency = VisualFieldPolicyV10._refresh(self, obs)
        started = time.monotonic()
        pred = self.prediction
        context = pred["motion_context"]
        intent = pred.get("field_intent") or {}
        self.target_contact.configure(obs.state, pred.get("target_geometry"),
            target_id=pred["target_id"], tool_radius_m=intent.get("tool_radius_m", .045),
            payload_observed=bool(self.monitor.attachment_evidence))
        self.local.contact_geometry = self.target_contact
        if self.field_configured:
            effective = copy.deepcopy(self.local.intent)
            if pred.get("target_geometry") is None:
                effective["max_travel_m"] = min(effective["max_travel_m"], .015)
                effective["command_limit_m"] = min(effective["command_limit_m"], .005)
            self.local.configure(obs.state, effective, scene_assessment=context,
                subgoal_origin_xyz=self.subgoals.active["origin_xyz"])
            self.goal = self.local.goal.copy()
        descriptor = pred.get("contact_alignment")
        # Closing cannot erase known incompatible alignment by omitting its
        # descriptor. Existing tracker/domain changes still invalidate evidence.
        requests_close = ((pred["controller"] == "gripper"
                           and pred["gripper_intent"]["command"] == "close")
                          or (pred["controller"] == "direct"
                              and any(action[6] > 0 for action in self.chunk)))
        retain_close = (descriptor is None and requests_close
            and self.contact_alignment.summary().get("target_id") == pred["target_id"])
        if not retain_close:
            self.contact_alignment.configure(descriptor, self.monitor, obs.state,
                tool_roi_uv=context.get("tool_external_roi_uv"))
        else:
            self.contact_alignment.configure({"update": "retain", "target_id": pred["target_id"],
                "external": None, "wrist": None}, self.monitor, obs.state,
                tool_roi_uv=context.get("tool_external_roi_uv"))
        self.local.contact_refinement_enabled = bool(descriptor and self.field_configured
                                                     and pred["phase"] in ("approach", "align"))
        self.subgoals.enable_contact_refinement(self.local.contact_refinement_enabled)
        self.last_contact_refinement = None
        self.translation_guard.configure(images(obs), context,
            obs.state, target_id=self.prediction["target_id"], monitor=self.monitor)
        self.feedback = self._observable_feedback()
        elapsed = time.monotonic()-started
        self._refresh_local_s += elapsed
        self.local_total_s += elapsed
        return latency

    def _anchor_is_interaction_target(self):
        # The older one-view goal offset can disagree with the actual two-view
        # pinch objective. Do not run both controllers against different goals.
        if (getattr(self, "prediction", None) or {}).get("contact_alignment"):
            return False
        return super()._anchor_is_interaction_target()

    def _refine_contact_goal(self, obs):
        if (not self.field_configured or not self.prediction.get("contact_alignment")
                or self.prediction["phase"] not in ("approach", "align")):
            return None
        # An observed terminal alignment must stop approach immediately. Only
        # changing a goal is rate-limited; review is never delayed by that limit.
        if self.contact_alignment.summary().get("ready_for_model_review"):
            return "contact_alignment_observed"
        if self.step_index-self.last_correction_step < 3:
            return None
        correction = self.contact_alignment.correction(self.goal,
            allowed_axes=self.prediction["field_intent"]["motion_axes"], max_correction_m=.006)
        self.last_contact_refinement = copy.deepcopy(correction)
        if correction["reason"] == "contact_alignment_observed":
            return "contact_alignment_observed"
        if not correction["accepted"]:
            return None  # Useful bounded nominal Field motion can collect evidence.
        proposed = np.asarray(correction["goal_xyz"], float)
        if not self.subgoals.check_contact_correction(proposed):
            self.last_contact_refinement["execution_status"] = "persistent_contact_refinement_bound"
            return "contact_refinement_requires_model_review"
        if not self.local.retarget_contact(proposed):
            self.last_contact_refinement["execution_status"] = "contact_goal_incompatible_with_field_geometry"
            return "contact_refinement_geometry_requires_model_review"
        if not self.subgoals.record_contact_correction(proposed):
            raise RuntimeError("Validated contact goal could not be recorded")
        self.goal = proposed.copy()
        self.visual_corrections += 1
        self.last_correction_step = self.step_index
        self._best_goal_error = float(np.linalg.norm(self.goal-np.asarray(obs.state[:3])))
        self.last_contact_refinement["execution_status"] = "field_retargeted"
        return None

    def _target_admission(self, obs, action):
        self.last_target_admission = self.target_contact.assess(obs.state, action,
            allow_separating=self.prediction["phase"] == "recover")
        return self.last_target_admission

    def _closing_alignment_reason(self):
        info = self.contact_alignment.summary()
        if not info["active"] or info["target_id"] != self.prediction["target_id"]:
            return None
        error, uncertainty = info["image_error_px"], info["error_component_uncertainty_px"]
        if error is not None and any(np.linalg.norm(error[i:i+2])-np.linalg.norm(uncertainty[i:i+2]) > 5.
                                     for i in (0, 2)):
            return "observed_contact_alignment_incompatible_with_close"
        return None

    def _field_control_limits(self, obs):
        """Default image-response limits; specialized controllers may add admission."""
        return self.monitor.measured_control_limit()

    def act(self, obs):
        if self.pending is not None:
            raise RuntimeError("Transition feedback required before the next action")
        if obs.joint_position is None or obs.joint_velocity is None:
            raise ValueError("V13 requires measured joint position and velocity")
        if self.prediction and self.prediction["controller"] == "direct" and not self.chunk and not self.refresh_reason:
            self.refresh_reason = "direct_chunk_complete"
        refreshed = self.prediction is None or bool(self.refresh_reason)
        self._refresh_local_s = 0.
        api_s = self._refresh(obs) if refreshed else 0.
        started = time.monotonic()
        controller = self.prediction["controller"]
        selection = admission = target_admission = rejected_model_action = None
        local_rotation_count = self.local.orientation_goal_refresh_commands
        local_rotation = self.local.last_orientation_command_rad.copy()
        gripper_rotation_count = self.gripper.orientation_goal_refresh_commands
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
                if selection is None:
                    target_admission = self._target_admission(obs, action)
                    reason = (self._closing_alignment_reason() if action[6] > 0 else None)
                    if not target_admission["accepted"] or reason:
                        rejected_model_action = list(action)
                        self.chunk = []
                        selection = self._rejected(reason or target_admission["reason"])
        elif controller == "gripper":
            selection = self.gripper.select(obs.state, obs.joint_position)
            if selection["accepted"]:
                target_admission = self._target_admission(obs, selection["action"])
                reason = (self._closing_alignment_reason()
                          if self.prediction["gripper_intent"]["command"] == "close" else None)
                if not target_admission["accepted"] or reason:
                    rejected_model_action = list(selection["action"])
                    # select is provisional; only observe consumes an action.
                    self.gripper.pending = None
                    self.gripper.orientation_goal_refresh_commands -= int(
                        selection["diagnostics"].get("orientation_hold", {}).get("goal_refresh_epsilon_used", False))
                    selection = self._rejected(reason or target_admission["reason"])
        else:
            measured_cap, reason = self._field_control_limits(obs)
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
                contact_reason = self._refine_contact_goal(obs)
                if contact_reason:
                    selection = self._rejected(contact_reason)
                if (selection is None and not self.prediction.get("contact_alignment")
                        and self._allows_planar_correction() and self.step_index-self.last_correction_step >= 5):
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
        if selection is None or selection.get("accepted"):
            proposed_action = tuple(selection["action"]) if selection is not None else action
            self.last_carried_admission = self.target_contact.assess_carried_payload(obs.state, proposed_action)
            if not self.last_carried_admission["accepted"]:
                rejected_model_action = list(proposed_action)
                self.chunk = []
                self.local.pending = None
                self.local.orientation_goal_refresh_commands = local_rotation_count
                self.local.last_orientation_command_rad = local_rotation
                self.gripper.pending = None
                self.gripper.orientation_goal_refresh_commands = gripper_rotation_count
                selection = self._rejected(self.last_carried_admission["reason"])
            elif proposed_action[6] > 0:
                captured = self.target_contact.remember_grasp_candidate(obs.state, self.prediction["target_id"])
                if captured["accepted"]:
                    self._last_grasp_candidate_id = self.prediction["target_id"]
                    self._release_commanded = False
        if selection is not None:
            action = tuple(selection["action"])
            if selection.get("need_refresh"):
                if selection["status"] in ("goal_reached", "measured_goal_arrival", "measured_local_goal_arrival", "intent_complete", "subgoal_local_complete", "contact_alignment_observed"):
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
            self.contact_geometry_vetoes += int(target_admission is not None and not target_admission["accepted"])
        elapsed = time.monotonic()-started
        self.local_total_s += elapsed
        diagnostics = {"step": self.step_index, "refreshed": refreshed, "controller": controller,
            "phase": self.prediction["phase"], "target_id": self.prediction["target_id"],
            "hypothesis_id": self.prediction["hypothesis_id"], "chunk_index": self.chunk_index,
            "chunk_step": self.step_index-self.block_start, "model_pipeline_s": api_s,
            "local_planning_s": elapsed+self._refresh_local_s, "selection": selection,
            "feedback": self._observable_feedback(), "refresh_reason": self.refresh_reason,
            "policy_version": "visual-field-policy-v13",
            "prediction": copy.deepcopy(self.prediction) if refreshed else None,
            "request_key": self.client.last_request_key if refreshed else None,
            "task_success_certified": False,
            "subgoal_update": copy.deepcopy(self.subgoal_update_result) if refreshed else None,
            "projected_rotation_risk": copy.deepcopy(self.rotation_screen),
            "scene_execution_limits": copy.deepcopy(self.scene_execution_limits),
            "translation_admission": copy.deepcopy(admission),
            "target_contact_admission": copy.deepcopy(target_admission),
            "carried_contact_admission": copy.deepcopy(self.last_carried_admission),
            "contact_refinement": copy.deepcopy(self.last_contact_refinement)}
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
        self.contact_alignment.observe(self.monitor, after.state, decision.action)
        external = self.monitor.refs.get("external") or {}
        if external.get("moving_kind") == "payload":
            payload_id = external["moving_id"]
            if self.monitor.attachment_evidence:
                self.target_contact.confirm_payload_attachment(after.state, payload_id, observed=True)
            elif (self._release_commanded and self.monitor.attachment.state == "contradicted"
                  and self.monitor.attachment.reason == "adaptive_payload_tool_motion_disagrees"):
                self.target_contact.release_payload(payload_id, observed_separation=True)
        if decision.action[6] < 0:
            self._release_commanded = True
            # Opening can discard a never-confirmed attempt; it cannot erase an
            # active carried hypothesis without observed tool/payload separation.
            if (self._last_grasp_candidate_id is not None
                    and np.linalg.norm(after.state[6:8]) > np.linalg.norm(before.state[6:8])+1e-5):
                self.target_contact.cancel_unconfirmed_candidate(self._last_grasp_candidate_id)
        # A supported visual correction may refine a just-reached approximate
        # metric goal; no new action/model call is needed solely for that arrival.
        if (self.field_configured and self.prediction.get("contact_alignment")
                and self.refresh_reason in ("subgoal_local_complete", "measured_local_goal_arrival")
                and self.local.stop_reason is None and self.local.pending is None):
            before_correction = self.visual_corrections
            contact_reason = self._refine_contact_goal(after)
            if contact_reason:
                self.refresh_reason = contact_reason
            elif self.visual_corrections > before_correction:
                self.refresh_reason = None
        self.feedback = self._observable_feedback()
        row.update(copy.deepcopy(self.feedback))
        self.local_total_s += time.monotonic()-started
        return row
