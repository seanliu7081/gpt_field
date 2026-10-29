"""Opt-in visual reliability policy; all evidence comes from RGB and robot feedback.

Stagnation changes the next observation request, never the episode stop rule.
Moving Direct5 contact actions are neither corrected nor shortened.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import time

import numpy as np

from .visual_field import PANDA_JOINT_LOWER, PANDA_JOINT_UPPER
from .visual_field_policy import VisualFieldPolicy, images
from .visual_field_v2 import VisualFieldControllerV2


RECOVERY_INSTRUCTION = (
    "Acquire the missing evidence instead of repeating hold/verify. Choose a small nonzero Direct "
    "translation or viewpoint change justified by current RGB, with stable orientation/gripper for "
    "response measurement, or correct an observable alignment/grasp hypothesis. Explain the expected "
    "measured/image difference. Keep probe translation norm <=0.1 normalized (5 mm target increment). "
    "Do not assume any global retreat direction is clear. If motion is not justified, identify the "
    "occluded/ambiguous landmark and revise references explicitly; renaming the same hypothesis or "
    "claiming completion supplies no evidence. No automatic task stop follows this request."
)


def compact_feedback(feedback):
    keys = ("version", "status", "tracks_reliable", "geometry_reliable", "action_to_joint", "action_to_eef",
            "eef_to_image", "eef_to_target_anchor", "projection_reset_reason", "attachment_epoch", "joint_to_image", "relative_feature_change_px", "object_tool_relative_change_px",
            "co_motion_evidence", "co_motion_sample_count", "prediction_failure", "observed_prediction_errors",
            "no_progress", "recovery", "clearance_certified", "task_success_certified", "limitations")
    result = {key: copy.deepcopy(feedback[key]) for key in keys if key in feedback}
    allowed = ("status", "reliable", "geometry_reliable", "moving_id", "target_id", "moving_kind",
               "moving_track_reliable", "target_track_reliable", "target_anchor_track_reliable",
               "tool_track_reliable", "moving_point_uv", "target_point_uv", "target_anchor_uv", "tool_point_uv",
               "image_shape_hw", "image_scale_px", "goal_error_px", "distance_cost_px", "obstacle_track_count",
               "obstacle_tracks", "target_offset_uv", "target_offset_px")
    result["views"] = {view: {key: copy.deepcopy(value[key]) for key in allowed if key in value}
                       for view, value in feedback.get("views", {}).items()}
    return result


def _cost(feedback):
    values = [feedback.get("views", {}).get(view, {}).get("distance_cost_px") for view in ("external", "wrist")]
    return sum(values) if all(value is not None for value in values) else None


class VisualFieldPolicyV2(VisualFieldPolicy):
    def reset(self):
        super().reset()
        self.local = VisualFieldControllerV2(joint_lower=PANDA_JOINT_LOWER, joint_upper=PANDA_JOINT_UPPER)
        self.task_stagnation = {"active": False, "reason": None, "zero_progress_blocks": 0,
            "unavailable_refreshes": 0, "repeated_hypothesis_blocks": 0,
            "evidence": {}, "next_action_requirement": None, "task_success_certified": False}
        self._block_initial_feedback = {}
        self._zero_chunk = False
        self._previous_hypothesis = None
        self._stagnation_events = 0
        self._zero_actions_skipped = 0
        self._attachment_gate_actions = 0
        self._attachment_epoch = None
        self._release_reference = None
        self._placement_previous = None
        self._placement_samples = 0
        self._placement_identity = None
        self.placement_evidence = self._uncertain_placement("no_post_release_observation")

    @staticmethod
    def _uncertain_placement(reason):
        return {"state": "uncertain", "reason": reason, "stable_observations": 0,
                "observed_image_alignment": False, "observed_gripper_separation": False,
                "observed_open_gripper": False, "support_3d_verified": False,
                "task_success_certified": False}

    @property
    def diagnostic_counters(self):
        return {"task_stagnation_events": self._stagnation_events,
                "unexecuted_zero_actions": self._zero_actions_skipped,
                "attachment_gate_actions": self._attachment_gate_actions,
                "placement_evidence_state": self.placement_evidence["state"]}

    def _activate(self, reason, evidence):
        new = not self.task_stagnation["active"] or self.task_stagnation["reason"] != reason
        self.task_stagnation.update(active=True, reason=reason, evidence=copy.deepcopy(evidence),
                                    next_action_requirement=RECOVERY_INSTRUCTION)
        if new:
            event = {"chunk_index": self.chunk_index, "step": self.step_index,
                "reason": reason, "target_id": self.prediction["target_id"] if self.prediction else None,
                "hypothesis_id": self.prediction["hypothesis_id"] if self.prediction else None,
                "evidence": copy.deepcopy(evidence), "episode_stop": False}
            self.events.append(event)
            self.failed_hypotheses.append(event)
            self.recovery_episodes += 1
            self._stagnation_events += 1

    def _clear_stagnation(self, reason):
        self.task_stagnation.update(active=False, reason=None, zero_progress_blocks=0,
            unavailable_refreshes=0, repeated_hypothesis_blocks=0,
            evidence={"cleared_by": reason, "task_progress_certified": False}, next_action_requirement=None)

    def _assess_block(self):
        if not self.block:
            return
        first, last = self.block[0], self.block[-1]
        net = float(np.linalg.norm(np.subtract(last["after_state"][:3], first["before_state"][:3])))
        rotation = float(np.linalg.norm(np.subtract(last["after_state"][3:6], first["before_state"][3:6])))
        fingers = float(np.linalg.norm(np.subtract(last["after_state"][6:], first["before_state"][6:])))
        initial_cost, cost = _cost(self._block_initial_feedback), _cost(self.feedback)
        visual_progress = (self.feedback.get("tracks_reliable", False) and initial_cost is not None
                           and cost is not None and initial_cost-cost >= 2.)
        evidence = {"chunk_index": self.chunk_index, "executed_steps": len(self.block),
                    "measured_net_m": net, "measured_rotation_delta_rad": rotation,
                    "finger_state_delta": fingers, "image_distance_reduction_px":
                        initial_cost-cost if initial_cost is not None and cost is not None else None,
                    "tracks_reliable": bool(self.feedback.get("tracks_reliable")),
                    "placement_state": self.placement_evidence["state"]}
        if net >= .003 or rotation >= .04 or visual_progress:
            self._clear_stagnation("measured_viewpoint_motion_or_image_descent")
        else:
            no_change = net <= .001 and rotation <= .015 and fingers <= .002 and not visual_progress
            self.task_stagnation["zero_progress_blocks"] = (
                self.task_stagnation["zero_progress_blocks"]+1 if no_change else 0)
            hypothesis = self.prediction["hypothesis_id"]
            self.task_stagnation["repeated_hypothesis_blocks"] = (
                self.task_stagnation["repeated_hypothesis_blocks"]+1
                if hypothesis == self._previous_hypothesis and no_change else 0)
            if self.task_stagnation["zero_progress_blocks"] >= 2:
                self._activate("repeated_observation_without_measured_progress", evidence)
            elif self.task_stagnation["unavailable_refreshes"] >= 2:
                self._activate("repeated_missing_visual_evidence", evidence)
        self._previous_hypothesis = self.prediction["hypothesis_id"]

    def _flush(self):
        if not self.block:
            return
        self._assess_block()
        first, last = self.block[0], self.block[-1]
        record = {"type": "visual_execution_block", "chunk_index": self.chunk_index,
            "steps": len(self.block), "first_step": first["step"], "last_step": last["step"],
            "before_state": first["before_state"], "after_state": last["after_state"],
            "joint_position": last["joint_position"],
            "measured_path_m": sum(r["measured_translation_m"] for r in self.block),
            "commanded_path_m": sum(r["commanded_translation_m"] for r in self.block),
            "measured_net_m": float(np.linalg.norm(np.subtract(last["after_state"][:3], first["before_state"][:3]))),
            "phase": self.prediction["phase"], "target_id": self.prediction["target_id"],
            "hypothesis_id": self.prediction["hypothesis_id"], "controller": self.prediction["controller"],
            "end_visual_feedback": compact_feedback(self.feedback), "refresh_reason": self.refresh_reason,
            "task_stagnation": copy.deepcopy(self.task_stagnation),
            "placement_evidence": copy.deepcopy(self.placement_evidence), "task_success_certified": False}
        self.client.record_block(record)
        self.block = []

    def _refresh(self, obs):
        self._flush()
        self.client.observable_context = {"visual_feedback": compact_feedback(self.feedback),
            "refresh_reason": self.refresh_reason,
            "failed_hypotheses": copy.deepcopy(self.failed_hypotheses[-8:]),
            "attempts_started": len(self.attempts), "recovery_episodes_detected": self.recovery_episodes,
            "task_stagnation": copy.deepcopy(self.task_stagnation),
            "placement_evidence": copy.deepcopy(self.placement_evidence),
            "attachment": {"co_motion_observed": self.attachment_evidence,
                "visible_payload_identity": self.attachment_identity, "grasp_certified": False},
            "stage_evidence": "RGB and measured robot response only; no evaluator information"}
        start = time.monotonic()
        prediction = self.client.predict(obs.to_dict())
        latency = time.monotonic()-start
        self.phase_previous = self.prediction["phase"] if self.prediction else None
        self.prediction = prediction
        self.chunk_index += 1
        self.block_start = self.step_index
        self.budget = 30
        local_start = time.monotonic()
        self.feedback = self.local.configure(images(obs), prediction["visual_references"], obs.joint_position, state8=obs.state)
        identity = self._payload_identity()
        epoch = self.feedback.get("attachment_epoch")
        reseeded = epoch is not None and epoch != self._attachment_epoch
        self._attachment_epoch = epoch
        if identity is None or identity != self.attachment_identity or reseeded:
            self.attachment_evidence = False
            self.attachment_identity = None
        if identity != self._placement_identity or reseeded:
            self._release_reference = self._placement_previous = None
            self._placement_samples = 0
            self.placement_evidence = self._uncertain_placement("payload_identity_missing_or_changed")
            self._placement_identity = identity
        self.initial_visual_cost = _cost(self.feedback)
        self._block_initial_feedback = copy.deepcopy(self.feedback)
        if not self.feedback.get("tracks_reliable"):
            self.task_stagnation["unavailable_refreshes"] += 1
            if self.task_stagnation["unavailable_refreshes"] >= 2:
                self._activate("repeated_missing_visual_evidence", {
                    "unavailable_refreshes": self.task_stagnation["unavailable_refreshes"],
                    "recovery": copy.deepcopy(self.feedback.get("recovery", {}))})
        else:
            self.task_stagnation["unavailable_refreshes"] = 0
        self.local_total_s += time.monotonic()-local_start
        self.refresh_reason = None
        if prediction["controller"] == "direct":
            self.chunk = copy.deepcopy(prediction["direct_actions"])
            self._zero_chunk = bool(np.all(np.asarray(self.chunk, dtype=float) == 0.))
            self.goal = None
        else:
            self.chunk = []
            self._zero_chunk = False
            self.goal = np.asarray(obs.state[:3], dtype=float) + prediction["field_intent"]["goal_offset_m"]
        return latency

    def act(self, obs):
        decision = super().act(obs)
        gate = None
        if self.prediction["controller"] == "visual_field":
            intent = self.prediction["field_intent"]
            closed_intent = intent["gripper_command"] == "close" or (
                intent["gripper_command"] == "hold" and self.last_gripper > 0)
            payload_motion = self.prediction["phase"] == "transport" or (
                self.prediction["phase"] == "align" and closed_intent)
            if payload_motion and (not self.attachment_evidence or self._payload_identity() is None):
                gate = "payload_co_motion_unverified_select_test_motion_or_regrasp"
                self.refresh_reason = gate
                self._attachment_gate_actions += 1
                self._activate("payload_transport_missing_attachment", {
                    "moving_point_must_be_payload": True, "co_motion_observed": False,
                    "required": "Independently visible payload and tool during measured Direct test motion"})
                decision = replace(decision, action=(0., 0., 0., 0., 0., 0., decision.action[6]))
        diagnostics = {**decision.diagnostics, "task_stagnation": copy.deepcopy(self.task_stagnation),
            "placement_evidence": copy.deepcopy(self.placement_evidence), "attachment_gate": gate,
            "refresh_reason": self.refresh_reason, "policy_version": "visual-field-policy-v2"}
        decision = replace(decision, diagnostics=diagnostics)
        self.pending = (obs, decision)
        return decision

    @staticmethod
    def _placement_features(feedback):
        values, identity = [], []
        for name in ("external", "wrist"):
            view = feedback.get("views", {}).get(name, {})
            if (not view.get("reliable") or view.get("moving_kind") != "payload"
                    or not view.get("tool_track_reliable") or view.get("goal_error_px") is None):
                return None
            scale = view.get("image_scale_px")
            if scale is None and view.get("image_shape_hw"):
                scale = [view["image_shape_hw"][1]-1, view["image_shape_hw"][0]-1]
            if scale is None or any(view.get(key) is None for key in ("moving_point_uv", "target_point_uv", "tool_point_uv")):
                return None
            moving, target, tool = [np.asarray(view[key], dtype=float)*scale
                                   for key in ("moving_point_uv", "target_point_uv", "tool_point_uv")]
            values.append({"relative_goal": moving-target, "relative_tool": moving-tool,
                           "goal_error_px": float(view["goal_error_px"])})
            identity.append((view.get("moving_id"), view.get("target_id")))
        return values, tuple(identity)

    def _placement_observe(self, before_feedback, before, decision, after):
        if decision.action[6] > 0:
            self._release_reference = self._placement_previous = None
            self._placement_samples = 0
            self.placement_evidence = self._uncertain_placement("gripper_reclosed_new_release_required")
            return
        features = self._placement_features(self.feedback)
        previous = self._placement_features(before_feedback)
        if decision.action[6] < 0 and self._release_reference is None and previous is not None:
            self._release_reference = previous
        if features is None or self._release_reference is None:
            self._placement_samples = 0
            self.placement_evidence = self._uncertain_placement("post_release_payload_target_or_tool_tracking_unavailable")
            return
        current, identity = features
        baseline, baseline_identity = self._release_reference
        if identity != baseline_identity:
            self._release_reference = self._placement_previous = None
            self._placement_samples = 0
            self.placement_evidence = self._uncertain_placement("placement_landmark_identity_changed")
            return
        spread = abs(after.state[6]-after.state[7])
        opened = decision.action[6] <= 0 and spread >= .035
        aligned = all(item["goal_error_px"] <= 3. for item in current)
        separation = min(float(np.linalg.norm(item["relative_tool"]-reference["relative_tool"]))
                         for item, reference in zip(current, baseline))
        stable = self._placement_previous is not None and all(
            np.linalg.norm(item["relative_goal"]-old["relative_goal"]) <= 1.5
            for item, old in zip(current, self._placement_previous))
        self._placement_samples = self._placement_samples+1 if opened and aligned and stable and separation >= 2. else 0
        self._placement_previous = current
        observed = self._placement_samples >= 3
        self.placement_evidence = {"state": "observed" if observed else "uncertain",
            "reason": "post_open_image_alignment_and_tool_separation" if observed else "need_stable_aligned_separated_observations",
            "stable_observations": self._placement_samples, "observation_interval_s": .05,
            "observed_image_alignment": aligned, "observed_gripper_separation": separation >= 2.,
            "observed_open_gripper": opened, "minimum_relative_tool_change_px": separation,
            "support_3d_verified": False, "task_success_certified": False,
            "limitation": "Image evidence only; no 3D support, containment or task-success certificate"}

    def observe_transition(self, before, decision, after):
        before_feedback = copy.deepcopy(self.feedback)
        row = super().observe_transition(before, decision, after)
        epoch = self.feedback.get("attachment_epoch")
        if epoch is not None and epoch != self._attachment_epoch:
            self.attachment_evidence = False
            self.attachment_identity = None
            self._release_reference = self._placement_previous = None
            self._placement_samples = 0
        self._attachment_epoch = epoch
        start = time.monotonic()
        self._placement_observe(before_feedback, before, decision, after)
        # A changed EEF view or tracked descent supplies new measurement, not task success.
        measured_rotation = float(np.linalg.norm(np.subtract(after.state[3:6], before.state[3:6])))
        old_cost, new_cost = _cost(before_feedback), _cost(self.feedback)
        if (row["measured_translation_m"] >= .003 or measured_rotation >= .04 or
                self.feedback.get("tracks_reliable") and old_cost is not None and new_cost is not None and old_cost-new_cost >= 2.):
            self._clear_stagnation("new_measured_motion_or_image_descent")
        skipped = 0
        if (self.prediction["controller"] == "direct" and self._zero_chunk
                and self.task_stagnation["active"] and self.chunk):
            # Only the *entire generated chunk* being exactly zero permits this.
            # Mixed/moving/gripper contact chunks retain all five original actions.
            skipped = len(self.chunk)
            self.chunk = []
            self._zero_actions_skipped += skipped
            self.refresh_reason = "persistent_zero_chunk_requires_informative_observation"
            self.events.append({"reason": self.refresh_reason, "step": self.step_index,
                "chunk_index": self.chunk_index, "unexecuted_zero_actions": skipped,
                "episode_stop": False})
        row.update(task_stagnation=copy.deepcopy(self.task_stagnation),
                   placement_evidence=copy.deepcopy(self.placement_evidence), unexecuted_zero_actions=skipped)
        self.local_total_s += time.monotonic()-start
        return row
