"""Persistent visual intent, local image-field planning and measured OSC feedback."""
from __future__ import annotations

import copy
import time

import numpy as np

from .action import ActionAdapter
from .observation import decode_image
from .policy import Decision
from .visual_field import VisualFieldController, PANDA_JOINT_LOWER, PANDA_JOINT_UPPER


def images(observation):
    return {"external": np.asarray(decode_image(observation.image)),
            "wrist": np.asarray(decode_image(observation.wrist_image))}


class VisualFieldPolicy:
    """The Direct arm observes identical local features but executes raw Direct5.

    No simulator object information or evaluator status is accepted by this class.
    Image-space risk is a heuristic, not a certificate of 3D collision avoidance.
    """
    def __init__(self, client):
        self.client = client
        self.reset()

    def reset(self):
        self.client.reset_history()
        self.local = VisualFieldController(joint_lower=PANDA_JOINT_LOWER, joint_upper=PANDA_JOINT_UPPER)
        self.adapter = ActionAdapter()
        self.step_index = 0
        self.prediction = None
        self.chunk = []
        self.chunk_index = -1
        self.block = []
        self.feedback = {}
        self.refresh_reason = "initial_observation"
        self.block_start = 0
        self.budget = 30
        self.goal = None
        self.pending = None
        self.failed_hypotheses = []
        self.attempts = []
        self.last_gripper = -1.
        self.recovery_episodes = 0
        self.events = []
        self.motion_history = []
        self.stall_active = False
        self.local_total_s = 0.
        self.phase_previous = None
        self.attachment_evidence = False
        self.attachment_identity = None
        self.initial_visual_cost = None

    def _flush(self):
        if not self.block:
            return
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
            "end_visual_feedback": copy.deepcopy(self.feedback),
            "refresh_reason": self.refresh_reason, "task_success_certified": False}
        self.client.record_block(record)
        self.block = []

    def _refresh(self, obs):
        self._flush()
        self.client.observable_context = {
            "visual_feedback": copy.deepcopy(self.feedback),
            "refresh_reason": self.refresh_reason,
            "failed_hypotheses": copy.deepcopy(self.failed_hypotheses[-8:]),
            "attempts_started": len(self.attempts),
            "recovery_episodes_detected": self.recovery_episodes,
            "stage_evidence": "visual evidence only; no evaluator information"}
        start = time.monotonic()
        prediction = self.client.predict(obs.to_dict())
        latency = time.monotonic() - start
        self.phase_previous = self.prediction["phase"] if self.prediction else None
        self.prediction = prediction
        self.chunk_index += 1
        self.block_start = self.step_index
        self.budget = 30
        local_start = time.monotonic()
        self.feedback = self.local.configure(images(obs), prediction["visual_references"],
                                            obs.joint_position, state8=obs.state)
        identity = self._payload_identity()
        if identity is None or identity != self.attachment_identity:
            self.attachment_evidence = False
            self.attachment_identity = None
        costs = [v.get("distance_cost_px") for v in self.feedback.get("views", {}).values()]
        self.initial_visual_cost = sum(costs) if len(costs) == 2 and all(v is not None for v in costs) else None
        self.local_total_s += time.monotonic() - local_start
        self.refresh_reason = None
        if prediction["controller"] == "direct":
            self.chunk = copy.deepcopy(prediction["direct_actions"])
            self.goal = None
        else:
            self.chunk = []
            self.goal = np.asarray(obs.state[:3]) + prediction["field_intent"]["goal_offset_m"]
        return latency

    def _payload_identity(self):
        views = self.feedback.get("views", {})
        if set(views) != {"external", "wrist"}:
            return None
        if any(v.get("moving_kind") != "payload" or not v.get("moving_track_reliable", v.get("reliable"))
               or not v.get("tool_track_reliable") for v in views.values()):
            return None
        return tuple(views[name]["moving_id"] for name in ("external", "wrist"))

    def act(self, obs):
        if self.pending is not None:
            raise RuntimeError("Transition feedback required before next action")
        if obs.joint_position is None:
            raise ValueError("Measured joint feedback is required")
        refreshed = self.prediction is None or bool(self.refresh_reason)
        if self.prediction and self.prediction["controller"] == "direct" and not self.chunk:
            self.refresh_reason = "direct_chunk_complete"
            refreshed = True
        if self.prediction and self.prediction["controller"] == "visual_field":
            if self.step_index - self.block_start >= self.budget:
                reliable = self.feedback.get("tracks_reliable", False) and self.feedback.get("geometry_reliable", False)
                recent = self.block[-10:]
                progress = sum(r["measured_translation_m"] for r in recent)
                costs = [v.get("distance_cost_px") for v in self.feedback.get("views", {}).values()]
                cost = sum(costs) if len(costs) == 2 and all(v is not None for v in costs) else None
                visual_progress = cost is not None and self.initial_visual_cost is not None and cost < self.initial_visual_cost - 1.
                if reliable and progress > .003 and visual_progress and self.budget < 120:
                    self.budget += 30
                    self.initial_visual_cost = cost
                else:
                    self.refresh_reason = "field_reuse_boundary"
                    refreshed = True
        api_s = self._refresh(obs) if refreshed else 0.
        start = time.monotonic()
        selection = None
        if self.prediction["controller"] == "direct":
            action = tuple(self.chunk.pop(0))
        else:
            intent = self.prediction["field_intent"]
            delta = self.goal - np.asarray(obs.state[:3])
            norm = float(np.linalg.norm(delta))
            if norm > .03:
                delta *= .03 / norm
            proposed = self.adapter.adapt(delta, intent["orientation_target_xyzw"],
                                          intent["gripper_command"], obs.state)
            selection = self.local.select_action(proposed, q=obs.joint_position)
            action = tuple(selection["action"])
            # An unavailable local predictor permits only its explicitly bounded bootstrap.
            # Lost identities and rejected candidates request a new semantic observation.
            if selection.get("need_refresh") and selection.get("status") not in ("bootstrap_unverified", "unsupported_direction_bootstrap"):
                self.refresh_reason = "local_" + selection["status"]
            if norm <= .005:
                self.refresh_reason = "measured_goal_arrival"
            # Model field transport needs evidence from the observed test lift. Direct contact
            # stays model-controlled and unmodified in both arms.
            if self.prediction["phase"] == "transport" and not self.attachment_evidence:
                action = (0., 0., 0., 0., 0., 0., float(action[6]))
                self.refresh_reason = "attachment_unverified_request_test_lift"
        planning_s = time.monotonic() - start
        self.local_total_s += planning_s
        diagnostics = {"step": self.step_index, "refreshed": refreshed,
            "controller": self.prediction["controller"], "phase": self.prediction["phase"],
            "target_id": self.prediction["target_id"], "hypothesis_id": self.prediction["hypothesis_id"],
            "chunk_index": self.chunk_index, "chunk_step": self.step_index - self.block_start,
            "model_pipeline_s": api_s, "local_planning_s": planning_s,
            "selection": selection, "feedback": copy.deepcopy(self.feedback),
            "refresh_reason": self.refresh_reason, "task_success_certified": False,
            "prediction": copy.deepcopy(self.prediction) if refreshed else None,
            "request_key": self.client.last_request_key if refreshed else None}
        decision = Decision(tuple(float(x) for x in action), diagnostics)
        self.pending = (obs, decision)
        self.step_index += 1
        return decision

    def observe_transition(self, before, decision, after):
        if self.pending is None or self.pending[0] != before or self.pending[1] is not decision:
            raise ValueError("Transition does not match the pending decision")
        start = time.monotonic()
        self.feedback = self.local.observe(images(after), after.joint_position,
                                           state8=after.state, executed_action=decision.action)
        identity = self._payload_identity()
        if identity is None or self.attachment_identity not in (None, identity):
            self.attachment_evidence = False
            self.attachment_identity = None
        if self.feedback.get("co_motion_evidence") and identity is not None:
            self.attachment_evidence = True
            self.attachment_identity = identity
        relative = self.feedback.get("object_tool_relative_change_px")
        if decision.action[6] < 0 or relative is not None and relative > 4.:
            self.attachment_evidence = False
            self.attachment_identity = None
        self.local_total_s += time.monotonic() - start
        measured = float(np.linalg.norm(np.subtract(after.state[:3], before.state[:3])))
        commanded = float(np.linalg.norm(decision.action[:3]) * .05)
        row = {"type": "observable_transition", "step": self.step_index - 1,
            "before_state": list(before.state), "after_state": list(after.state),
            "joint_position": list(after.joint_position), "joint_velocity": list(after.joint_velocity),
            "action": list(decision.action), "measured_translation_m": measured,
            "commanded_translation_m": commanded, "visual": copy.deepcopy(self.feedback)}
        self.block.append(row)
        self.motion_history = (self.motion_history + [row])[-8:]
        grip = decision.action[6]
        if grip > 0 and self.last_gripper < 0:
            self.attempts.append({"step": self.step_index - 1, "target_id": self.prediction["target_id"],
                "hypothesis_id": self.prediction["hypothesis_id"], "eef_position": list(before.state[:3]),
                "definition": "open-to-close command episode; not certified grasp"})
        if grip != 0:
            self.last_gripper = 1. if grip > 0 else -1.
            self.adapter._gripper = self.last_gripper
        recent = self.motion_history
        stalled = (len(recent) == 8 and sum(r["commanded_translation_m"] for r in recent) >= .015
                   and sum(r["measured_translation_m"] for r in recent) <= .001)
        if sum(r["measured_translation_m"] for r in recent) > .003:
            self.stall_active = False
        if stalled and not self.stall_active:
            self.stall_active = True
            event = {"chunk_index": self.chunk_index, "step": self.step_index,
                "reason": "commanded_motion_without_measured_progress", "target_id": self.prediction["target_id"],
                "hypothesis_id": self.prediction["hypothesis_id"]}
            self.events.append(event)
            self.failed_hypotheses.append(event)
            self.recovery_episodes += 1
            if self.prediction["controller"] == "visual_field":
                self.refresh_reason = "observed_no_progress_change_hypothesis"
        self.pending = None
        return row

    def finish(self):
        self._flush()
