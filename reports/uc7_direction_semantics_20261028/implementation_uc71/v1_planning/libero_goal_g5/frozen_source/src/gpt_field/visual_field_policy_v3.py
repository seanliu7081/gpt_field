"""Persistent V3 semantic intent, typed blockers and measured information acquisition.

A local grant is a bounded observation permission, never an episode stop budget.
All evidence is derived from current RGB, measured robot state and executed actions.
"""
from __future__ import annotations

import copy
from collections import Counter
import time

import numpy as np

from .policy import Decision
from .visual_field import PANDA_JOINT_LOWER, PANDA_JOINT_UPPER
from .visual_field_policy import VisualFieldPolicy, images
from .visual_field_policy_v2 import VisualFieldPolicyV2, compact_feedback as compact_v2
from .visual_field_v3 import VisualFieldControllerV3, STRUCTURAL_CODES


def compact_feedback(feedback):
    result = compact_v2(feedback)
    for key in ("feasibility", "map_feasibility", "blockers", "view_models", "active_views", "unavailable_views"):
        if key in feedback:
            result[key] = copy.deepcopy(feedback[key])
    for name, view in feedback.get("views", {}).items():
        target = result.setdefault("views", {}).setdefault(name, {})
        for key in ("preflight", "landmark_epoch", "model_epoch", "goal_epoch", "goal_descriptor",
                    "eef_to_image", "eef_to_target_anchor", "moving_footprint_uv", "constraint_uncertainty"):
            if key in view:
                target[key] = copy.deepcopy(view[key])
    return result


def blocker_code(value):
    return value.get("code", value.get("category", "unclassified"))


def blocker_key(value):
    return "|".join(str(value.get(k) or "") for k in ("view", "code", "category", "feature_id", "region_id"))


def _view_identity(view):
    return (view.get("moving_id"), view.get("target_id"))


def _quality(feedback):
    result = {}
    for name, view in feedback.get("views", {}).items():
        models = feedback.get("view_models", {}).get(name, {})
        result[name] = {"identity": _view_identity(view), "landmark_epoch": view.get("landmark_epoch"),
            "model_epoch": view.get("model_epoch"), "goal_epoch": view.get("goal_epoch"),
            "goal_descriptor": copy.deepcopy(view.get("goal_descriptor")),
            "reliable": bool(view.get("reliable")), "goal_error_px": view.get("goal_error_px"),
            "models": {key: copy.deepcopy(view.get(key, models.get(key, {})))
                       for key in ("eef_to_image", "eef_to_target_anchor")}}
    return result


class ProgressLedger:
    """Current blockers and deterministic observable outcomes, independent of chat history."""
    def __init__(self):
        self.active = {}
        self.latest_quality = {}
        self.information_events = 0
        self.task_distance_events = 0
        self.resolved_counts = Counter()
        self.superseded_counts = Counter()
        self.probe_count = 0
        self.probe_directions = {}
        self.recent_outcomes = []
        self.last_gain = []
        self.best_error = {}

    def update(self, feedback, *, semantic=False):
        quality = _quality(feedback)
        incoming = {blocker_key(value): copy.deepcopy(value) for value in feedback.get("blockers", [])}
        gains = []
        for key, old in list(self.active.items()):
            if key in incoming:
                continue
            name = old.get("view")
            view, prior = quality.get(name, {}), self.latest_quality.get(name, {})
            if not view.get("reliable"):
                incoming[key] = old  # Missing observations do not clear uncertainty.
            elif (view.get("identity") != prior.get("identity")
                  or view.get("landmark_epoch") != prior.get("landmark_epoch")):
                self.superseded_counts[blocker_code(old)] += 1
                self.recent_outcomes.append({"code": blocker_code(old), "view": name,
                    "outcome": "superseded_by_reference_change_not_measured_resolution"})
            else:
                kind = "semantic_reference_correction" if semantic else "observed_blocker_removed"
                self.resolved_counts[blocker_code(old)] += 1
                self.recent_outcomes.append({"code": blocker_code(old), "view": name, "outcome": kind})
                if not semantic:
                    gains.append({"view": name, "reason": kind, "code": blocker_code(old)})
        for name, view in quality.items():
            prior = self.latest_quality.get(name)
            if prior is None or prior["identity"] != view["identity"] or semantic:
                continue
            if view["reliable"] and not prior["reliable"]:
                gains.append({"view": name, "reason": "same_identity_visibility_recovered"})
            if view["model_epoch"] == prior["model_epoch"]:
                for model, current in view["models"].items():
                    old = prior["models"][model]
                    improvement = []
                    if current.get("rank", 0) > old.get("rank", 0):
                        improvement.append("supported_rank_increased")
                    if min(current.get("validation_count", 0), 2) > min(old.get("validation_count", 0), 2):
                        improvement.append("new_prediction_validation")
                    a, b = old.get("residual"), current.get("residual")
                    if a is not None and b is not None and a-b > .15:
                        improvement.append("prediction_residual_reduced")
                    if improvement:
                        gains.append({"view": name, "reason": ",".join(improvement), "model": model})
            if (view["reliable"] and prior["reliable"] and view["landmark_epoch"] == prior["landmark_epoch"]
                    and view["goal_epoch"] == prior["goal_epoch"] and view["goal_descriptor"] == prior["goal_descriptor"]):
                key = (name, str(view["identity"]), view["landmark_epoch"], view["goal_epoch"], str(view["goal_descriptor"]))
                error, old = view["goal_error_px"], prior["goal_error_px"]
                if error is not None and old is not None:
                    best = self.best_error.get(key, old)
                    if best-error >= 1.5:
                        self.task_distance_events += 1
                        gains.append({"view": name, "reason": "same_goal_image_distance_decreased", "delta_px": best-error})
                        self.best_error[key] = error
                    else:
                        self.best_error.setdefault(key, best)
        self.information_events += sum(gain["reason"] != "same_goal_image_distance_decreased" for gain in gains)
        self.active, self.latest_quality, self.last_gain = incoming, quality, gains
        self.recent_outcomes = self.recent_outcomes[-12:]
        return gains

    def add_blocker(self, blocker):
        self.active[blocker_key(blocker)] = copy.deepcopy(blocker)

    def probe(self, action, before, after, gains, *, grant_index):
        self.probe_count += 1
        key = ",".join(f"{float(x):.4f}" for x in action[:3])
        entry = self.probe_directions.setdefault(key, {"count": 0, "informative_count": 0})
        entry["count"] += 1
        entry["last_probe_index"] = self.probe_count
        entry["informative_count"] += bool(gains)
        self.recent_outcomes.append({"outcome": "local_probe", "grant_index": grant_index,
            "normalized_translation": list(action[:3]),
            "measured_delta_m": np.subtract(after.state[:3], before.state[:3]).tolist(),
            "information_changes": copy.deepcopy(gains), "task_success_certified": False})
        self.recent_outcomes = self.recent_outcomes[-12:]

    def snapshot(self):
        ordered = sorted(self.probe_directions, key=lambda key: self.probe_directions[key]["last_probe_index"])
        recent = ordered[-24:]
        older = ordered[:-24]
        return {"version": "typed-per-view-progress-v3", "active_blockers": list(copy.deepcopy(self.active).values()),
            "information_events": self.information_events, "task_distance_events": self.task_distance_events,
            "resolved_counts": dict(self.resolved_counts), "superseded_counts": dict(self.superseded_counts),
            "probe_count": self.probe_count, "probe_directions": {key: copy.deepcopy(self.probe_directions[key]) for key in recent},
            "older_direction_totals": {"direction_count": len(older),
                "probe_count": sum(self.probe_directions[key]["count"] for key in older),
                "informative_count": sum(self.probe_directions[key]["informative_count"] for key in older)},
            "view_evidence": copy.deepcopy(self.latest_quality), "recent_outcomes": copy.deepcopy(self.recent_outcomes),
            "last_information_changes": copy.deepcopy(self.last_gain), "task_success_certified": False,
            "interpretation": "EEF movement alone never resolves a blocker; image progress is not 3D task success"}


class VisualFieldPolicyV3(VisualFieldPolicyV2):
    def reset(self):
        super().reset()
        self.local = VisualFieldControllerV3(joint_lower=PANDA_JOINT_LOWER, joint_upper=PANDA_JOINT_UPPER)
        self.progress = ProgressLedger()
        self._block_gain_start = (0, 0)
        self._no_information_blocks = 0
        self._grant = None
        self._grant_origin = None
        self._grant_steps = 0
        self._acquisition_actions = 0
        self._semantic_repair_requests = 0
        self._field_progress_start = (0, 0)
        self.local_mode = "semantic_observation"
        self._last_blocker_signature = None

    @property
    def diagnostic_counters(self):
        return {**super().diagnostic_counters, "local_information_probe_actions": self._acquisition_actions,
                "semantic_repair_requests": self._semantic_repair_requests,
                "information_events": self.progress.information_events,
                "task_distance_events": self.progress.task_distance_events,
                "blocks_without_information": self._no_information_blocks}

    def _counts(self):
        return self.progress.information_events, self.progress.task_distance_events

    def _stagnation(self):
        blockers = list(self.progress.active.values())
        structural = any(blocker_code(value) in STRUCTURAL_CODES for value in blockers)
        active = bool(blockers) or self._no_information_blocks >= 2
        reason = ("semantic_reference_repair_required" if structural else "typed_observation_evidence_required"
                  if blockers else "repeated_actions_without_information_or_image_progress" if active else None)
        signature = (reason, tuple(sorted(blocker_key(value) for value in blockers))) if active else None
        if active and signature != self._last_blocker_signature:
            self.events.append({"reason": reason, "step": self.step_index, "chunk_index": self.chunk_index,
                "blocker_codes": [blocker_code(value) for value in blockers], "episode_stop": False})
            self.recovery_episodes += 1
            self._stagnation_events += 1
        self._last_blocker_signature = signature
        self.task_stagnation = {"active": active, "reason": reason,
            "blocks_without_information": self._no_information_blocks,
            "active_blocker_count": len(blockers), "evidence": {"information_events": self.progress.information_events,
                "task_distance_events": self.progress.task_distance_events},
            "next_action_requirement": ("Repair the listed structural reference/footprint/goal contradictions; "
                "calibration probes cannot fix them." if structural else
                "Address each typed blocker using its remedy. Preserve healthy view models. Authorize local "
                "acquisition only for supported candidates; use Direct5 for contact or an initial unsupported measurement. "
                "Consult prior probe outcomes; physical movement alone is not new information." if active else None),
            "task_success_certified": False}
        return self.task_stagnation

    def _flush(self):
        if not self.block:
            return
        gained = self._counts() != self._block_gain_start
        self._no_information_blocks = 0 if gained else self._no_information_blocks+1
        self._stagnation()
        first, last = self.block[0], self.block[-1]
        self.client.record_block({"type": "visual_execution_block", "chunk_index": self.chunk_index,
            "steps": len(self.block), "first_step": first["step"], "last_step": last["step"],
            "before_state": first["before_state"], "after_state": last["after_state"],
            "joint_position": last["joint_position"],
            "measured_path_m": sum(row["measured_translation_m"] for row in self.block),
            "commanded_path_m": sum(row["commanded_translation_m"] for row in self.block),
            "measured_net_m": float(np.linalg.norm(np.subtract(last["after_state"][:3], first["before_state"][:3]))),
            "phase": self.prediction["phase"], "target_id": self.prediction["target_id"],
            "hypothesis_id": self.prediction["hypothesis_id"], "controller": self.prediction["controller"],
            "end_visual_feedback": compact_feedback(self.feedback), "refresh_reason": self.refresh_reason,
            "task_stagnation": copy.deepcopy(self.task_stagnation), "progress_ledger": self.progress.snapshot(),
            "placement_evidence": copy.deepcopy(self.placement_evidence), "task_success_certified": False})
        self.block = []

    def _refresh(self, obs):
        self._flush()
        self.client.observable_context = {"visual_feedback": compact_feedback(self.feedback),
            "refresh_reason": self.refresh_reason, "progress_ledger": self.progress.snapshot(),
            "task_stagnation": copy.deepcopy(self._stagnation()),
            "placement_evidence": copy.deepcopy(self.placement_evidence),
            "attachment": {"co_motion_observed": self.attachment_evidence,
                           "visible_payload_identity": self.attachment_identity, "grasp_certified": False},
            "attempts_started": len(self.attempts), "recovery_episodes_detected": self.recovery_episodes,
            "stage_evidence": "observable images and measured response only"}
        started = time.monotonic()
        prediction = self.client.predict(obs.to_dict())
        latency = time.monotonic()-started
        started = time.monotonic()
        self.phase_previous = self.prediction["phase"] if self.prediction else None
        self.prediction, self.chunk_index = prediction, self.chunk_index+1
        self.block_start, self.budget = self.step_index, 30
        self.feedback = self.local.configure(images(obs), prediction["visual_references"], obs.joint_position, state8=obs.state)
        self.progress.update(self.feedback, semantic=True)
        self._block_gain_start = self._field_progress_start = self._counts()
        identity, epoch = self._payload_identity(), self.feedback.get("attachment_epoch")
        reseeded = epoch is not None and epoch != self._attachment_epoch
        self._attachment_epoch = epoch
        if identity is None or identity != self.attachment_identity or reseeded:
            self.attachment_evidence, self.attachment_identity = False, None
        if identity != self._placement_identity or reseeded:
            self._release_reference = self._placement_previous = None
            self._placement_samples = 0
            self.placement_evidence = self._uncertain_placement("payload_identity_missing_or_changed")
            self._placement_identity = identity
        self._grant = copy.deepcopy(prediction.get("observation_intent"))
        self._grant_origin = np.asarray(obs.state[:3], dtype=float)
        self._grant_steps = 0
        self.refresh_reason = None
        if prediction["controller"] == "direct":
            self.chunk = copy.deepcopy(prediction["direct_actions"])
            self.goal, self.local_mode = None, "model_direct_contact_or_measurement"
        else:
            self.chunk = []
            self.goal = np.asarray(obs.state[:3], dtype=float)+prediction["field_intent"]["goal_offset_m"]
            self.local_mode = "persistent_field"
        self._stagnation()
        self.local_total_s += time.monotonic()-started
        return latency

    def _available_grant(self, obs):
        if self._grant is None or self._grant_steps >= self._grant["max_steps"]:
            return None
        if np.linalg.norm(np.asarray(obs.state[:3])-self._grant_origin) >= self._grant["max_displacement_m"]-1e-9:
            return None
        if any(blocker_code(value) in STRUCTURAL_CODES for value in self.progress.active.values()):
            return None
        return self._grant

    def _reject(self, selection, reason, action):
        return {**selection, "action": [0.]*7,
                "accepted": False, "status": reason, "need_refresh": True}

    def act(self, obs):
        if self.pending is not None:
            raise RuntimeError("Transition feedback required before next action")
        if obs.joint_position is None or obs.joint_velocity is None:
            raise ValueError("Measured joint feedback is required")
        if self.prediction and self.prediction["controller"] == "direct" and not self.chunk:
            self.refresh_reason = "direct_chunk_complete"
        if self.prediction and self.prediction["controller"] == "visual_field" and self.step_index-self.block_start >= self.budget:
            if self._counts() != self._field_progress_start and self.budget < 120 and not self.progress.active:
                self.budget += 30
                self._field_progress_start = self._counts()
            else:
                self.refresh_reason = "persistent_intent_refresh_boundary"
        refreshed = self.prediction is None or bool(self.refresh_reason)
        api_s = self._refresh(obs) if refreshed else 0.
        started, selection, gate = time.monotonic(), None, None
        if self.prediction["controller"] == "direct":
            action = tuple(self.chunk.pop(0))
        else:
            intent = self.prediction["field_intent"]
            delta = self.goal-np.asarray(obs.state[:3], dtype=float)
            norm = float(np.linalg.norm(delta))
            if norm > .03:
                delta *= .03/norm
            proposed = self.adapter.adapt(delta, intent["orientation_target_xyzw"], intent["gripper_command"], obs.state)
            grant = self._available_grant(obs)
            selection = self.local.select_action(proposed, q=obs.joint_position, observation_intent=grant)
            action = tuple(selection["action"])
            if selection.get("accepted") and selection.get("status") == "local_information_probe":
                direction_allowed = grant is not None and any(np.allclose(action[:3], direction, rtol=0., atol=1e-8)
                                                              for direction in grant["directions"])
                predicted = selection.get("diagnostics", {}).get("predicted_eef_delta_m")
                displacement = np.asarray(obs.state[:3])-self._grant_origin
                # A commanded target bound remains conservative fallback metadata; no physical safety certificate.
                estimated = np.asarray(predicted, float) if predicted is not None else np.asarray(action[:3])*.05
                uncertainty = max(0., float(selection.get("diagnostics", {}).get("prediction_residual_margin_m", 0.)))
                within = grant is not None and np.linalg.norm(displacement+estimated)+uncertainty <= grant["max_displacement_m"]+1e-9
                stable = np.linalg.norm(action[3:6]) <= 1e-12 and action[6] == 0.
                if not direction_allowed or not within or not stable:
                    selection = self._reject(selection, "observation_grant_boundary", action)
                    action = tuple(selection["action"])
                else:
                    self.local_mode = "local_information_acquisition"
                    self.adapter._gripper = self.last_gripper
            elif selection.get("accepted"):
                self.local_mode = "persistent_field"
            else:
                self.local_mode = "semantic_observation"
                # A rejected geometric intent grants neither motion nor a gripper
                # change. Contact remains an explicit model-directed Direct action.
                selection = self._reject(selection, selection["status"], action)
                action = tuple(selection["action"])
                self.adapter._gripper = self.last_gripper
            closed = intent["gripper_command"] == "close" or (intent["gripper_command"] == "hold" and self.last_gripper > 0)
            if (self.prediction["phase"] == "transport" or self.prediction["phase"] == "align" and closed) and (
                    not self.attachment_evidence or self._payload_identity() is None):
                gate = "payload_attachment_evidence_required"
                selection = self._reject(selection, gate, action)
                action = tuple(selection["action"])
                self._attachment_gate_actions += 1
            if selection.get("need_refresh") and not selection.get("accepted"):
                self.refresh_reason = "local_"+selection["status"]
                if selection["status"] == "semantic_correction_required":
                    self._semantic_repair_requests += 1
            if norm <= .005 and selection.get("status") != "local_information_probe":
                self.refresh_reason = self.refresh_reason or "measured_local_goal_arrival"
        local_s = time.monotonic()-started
        self.local_total_s += local_s
        diagnostics = {"step": self.step_index, "refreshed": refreshed,
            "controller": self.prediction["controller"], "phase": self.prediction["phase"],
            "target_id": self.prediction["target_id"], "hypothesis_id": self.prediction["hypothesis_id"],
            "chunk_index": self.chunk_index, "chunk_step": self.step_index-self.block_start,
            "model_pipeline_s": api_s, "local_planning_s": local_s, "local_mode": self.local_mode,
            "selection": selection, "feedback": copy.deepcopy(self.feedback), "attachment_gate": gate,
            "refresh_reason": self.refresh_reason, "task_stagnation": copy.deepcopy(self._stagnation()),
            "progress_ledger": self.progress.snapshot(), "placement_evidence": copy.deepcopy(self.placement_evidence),
            "acquisition_steps_used": self._grant_steps, "task_success_certified": False,
            "prediction": copy.deepcopy(self.prediction) if refreshed else None,
            "request_key": self.client.last_request_key if refreshed else None, "policy_version": "visual-field-policy-v3"}
        decision = Decision(tuple(float(x) for x in action), diagnostics)
        self.pending = obs, decision
        self.step_index += 1
        return decision

    def observe_transition(self, before, decision, after):
        old_feedback = copy.deepcopy(self.feedback)
        # V1 bookkeeping records actual actions, gripper attempts, and measured feedback;
        # it does not use V2's EEF-motion-based task-stagnation clearing.
        row = VisualFieldPolicy.observe_transition(self, before, decision, after)
        started = time.monotonic()
        gains = self.progress.update(self.feedback)
        epoch = self.feedback.get("attachment_epoch")
        if epoch is not None and epoch != self._attachment_epoch:
            self.attachment_evidence, self.attachment_identity = False, None
            self._release_reference = self._placement_previous = None
            self._placement_samples = 0
        self._attachment_epoch = epoch
        self._placement_observe(old_feedback, before, decision, after)
        selection = decision.diagnostics.get("selection") or {}
        if selection.get("accepted") and selection.get("status") == "local_information_probe":
            self._grant_steps += 1
            self._acquisition_actions += 1
            self.progress.probe(decision.action, before, after, gains, grant_index=self.chunk_index)
            measured_radius = float(np.linalg.norm(np.asarray(after.state[:3])-self._grant_origin))
            if measured_radius > self._grant["max_displacement_m"]+1e-9:
                self.refresh_reason = "observation_grant_measured_displacement_exceeded"
                self.events.append({"reason": self.refresh_reason, "step": self.step_index-1,
                    "chunk_index": self.chunk_index, "measured_displacement_m": measured_radius,
                    "authorized_displacement_m": self._grant["max_displacement_m"], "episode_stop": False})
            # The next action first tries the same field intent; an expired grant cannot renew itself.
            self.local_mode = "persistent_field"
        self._stagnation()
        row.update(task_stagnation=copy.deepcopy(self.task_stagnation), progress_ledger=self.progress.snapshot(),
                   placement_evidence=copy.deepcopy(self.placement_evidence), local_mode=self.local_mode,
                   acquisition_steps_used=self._grant_steps, unexecuted_zero_actions=0)
        self.local_total_s += time.monotonic()-started
        return row
