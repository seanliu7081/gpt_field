"""Astra batches over measured fragments; optional vision has no action authority."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from time import monotonic

import numpy as np

from ..agent_speedup_v2_user_check_fragments import FragmentSequencePrimitive, fragment_capabilities
from ..agent_speedup_v2_user_check_routing import TransitObstacleMap
from ..astranomy_agentic_motion import measured_pose
from ..contact_skills import HoldingActionAdapter, StepView
from ..foundation_harness.types import jsonable
from ..policy import Decision
from . import VERSION, REVISION
from .protocol import ProtocolRepairError


@dataclass(frozen=True)
class HarnessConfig:
    history_items: int = 6


def _summary(segment):
    if segment is None:
        return None
    result = {key: deepcopy(segment.get(key)) for key in
              ("id", "object", "checkpoint", "checkpoint_object")}
    result["candidates"] = [{key: deepcopy(candidate.get(key)) for key in
        ("skill", "xyz_m", "quaternion_wxyz", "profile", "gripper")}
        for candidate in segment.get("candidates", [])]
    return result


def _compact_outcome(outcome):
    evidence = outcome.get("evidence", {})
    fields = ("position_error_m", "orientation_error_rad", "measured_pose_wxyz", "measured_gap_mm",
              "target_xyz_m", "target_quaternion_wxyz", "gripper_command", "detail", "handoff_kind",
              "scope", "parts", "joint_position_rad", "sample_fraction")
    result = {key: deepcopy(outcome.get(key)) for key in ("status", "reason", "actions")}
    result["evidence"] = {key: deepcopy(evidence[key]) for key in fields if key in evidence}
    # Retain the actual route/stall reason without copying hundreds of samples.
    result["evidence"]["segments"] = [{key: deepcopy(row[key]) for key in
        ("status", "reason", "detail", "routing", "stall_evidence", "gap_change_evidence") if key in row}
        for row in evidence.get("segments", [])]
    for row in result["evidence"]["segments"]:
        if "routing" in row:
            row["routing"] = {key: value for key, value in row["routing"].items() if key in
                ("status", "reason", "map_source", "map_revision", "obstacle_count",
                 "straight_path_valid", "path_length_m", "planning_s")}
    return result


class RoutedFragment(FragmentSequencePrimitive):
    """Use the real Astra map, preserving the exact dispatched-action contract."""
    def __init__(self, view, candidate, gripper, obstacle_map):
        super().__init__(view, {"segments": [deepcopy(candidate)]},
                         keep_gripper=gripper, obstacle_map=obstacle_map)
        self.adapter = HoldingActionAdapter(max_rotation_rad=.1)
        self.adapter._gripper = 1. if gripper == "close" else -1.

    def commit_action(self, action):
        if self._pending is None:
            raise RuntimeError("fragment has no pending action")
        action = tuple(float(x) for x in action)
        self._pending = (self._pending[0], action)
        self.gripper_command = "close" if action[6] > 0 else "open"
        self.adapter._gripper = action[6]
        return action

    def cancel_proposal(self):
        self._pending = None


class FoundationVisionHarnessPolicy:
    def __init__(self, client, *, output_dir=None, kinematics=None, vision_tools=None,
                 robot_guard=None, config=None):
        from .vision_tools import OnDemandVisionTools
        from .robot_guard import RobotGuard
        self.client = client
        self.config = config or HarnessConfig()
        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vision_tools = vision_tools if vision_tools is not None else OnDemandVisionTools(mode="off")
        self.robot_guard = robot_guard if robot_guard is not None else RobotGuard(kinematics)
        self.adapter = HoldingActionAdapter(max_rotation_rad=.1)
        self.reset()

    def reset(self):
        self.queue = deque()
        self.active = self.primitive = self.entry_state = None
        self.candidate_index = 0
        self.obstacles = TransitObstacleMap()
        self.gripper_command = "open"
        self.step_index = 0
        self.need_model = True
        self.terminal = self.last_view = self.pending_kind = None
        self.feedback = {"kind": "initial", "reason": "initial_observation"}
        self.history, self.path, self.tool_results, self.model_requests = [], [], [], []
        self.model_decisions = self.grasp_attempts = self.completed_segments = 0
        self.model_seconds = self.tool_seconds = self.guard_seconds = 0.
        self.started_at = monotonic()
        self._observation_received_at_s = None
        self._protocol_failure = None
        self._act_guard_seconds = 0.
        self._intended_grasp_object = None
        self._last_released_object = None

    def _log(self, event, **fields):
        if self.output_dir:
            with (self.output_dir / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(jsonable({"event": event, "step": self.step_index,
                    "online_s": monotonic()-self.started_at, **fields}), separators=(",", ":"))+"\n")

    def _terminal(self, kind, reason):
        self.terminal = {"kind": kind, "reason": reason, "step": self.step_index}
        self._log("terminal", **self.terminal)

    def _context(self, view):
        tools = deepcopy(self.tool_results[-2:])
        for result in tools:
            if result.get("captured_at_s") is not None:
                result["age_at_model_request_s"] = max(0., monotonic()-result["captured_at_s"])
        fragments = fragment_capabilities()
        return jsonable({"task_prompt": view.observation.prompt,
            "step": self.step_index, "frame_id": self.step_index,
            "current_segment": _summary(self.active),
            "queue_remaining": [_summary(s) for s in self.queue],
            "segment_history": self.history[-self.config.history_items:],
            "feedback": self.feedback, "gripper_command": self.gripper_command,
            "obstacle_map": self.obstacles.snapshot(), "tool_results": tools,
            "capabilities": {"skills": fragments["skills"], "profiles": fragments["profiles"],
                "minimum_gripper_steps": fragments["minimum_gripper_steps"],
                "vision": self.vision_tools.capabilities(),
                "batch_execution": "measured local continuation; visual checkpoints return to Astra",
                "grasp_evidence": "finger gap and RGB; stable gap does not establish holding",
                "environment_constraints": "Astra boxes for transit grip-site routing; not full-body certification",
                "robot_constraints": "known joints and sampled self geometry, independent of visual measurements"},
            "timing": {"wall_s": monotonic()-self.started_at, "model_s": self.model_seconds,
                       "tool_s": self.tool_seconds}})

    def _ask(self, view, calls):
        self.model_decisions += 1
        started = monotonic()
        status = "error"
        try:
            answer = self.client.request_harness(view, self._context(view))
            status = "accepted"
        except ProtocolRepairError as error:
            status = "protocol_repair"
            signature = str(error)
            self.feedback = {"kind": "protocol_error", "reason": str(error),
                             "repair": getattr(error, "evidence", {})}
            if signature == self._protocol_failure:
                self._terminal("blocked", "repeated_unrepaired_protocol")
            self._protocol_failure = signature
            self.need_model = True
            return
        finally:
            elapsed = monotonic()-started
            self.model_seconds += elapsed
            record = {"kind": "harness", "ordinal": self.model_decisions,
                      "status": status, "latency_s": elapsed, "step": self.step_index}
            self.model_requests.append(record)
            calls.append(record)
            self._log("model_request", **record)
        self._protocol_failure = None
        self._log("model_plan", answer=answer)
        if answer["status"] in {"finish", "blocked"}:
            self._terminal(answer["status"], answer["why"])
            return
        if answer["status"] == "measure":
            started = monotonic()
            result = self.vision_tools.measure(view, answer["measurements"], frame_id=self.step_index,
                                              captured_at_s=self._observation_received_at_s)
            self.tool_seconds += monotonic()-started
            self.tool_results.append(jsonable(result))
            self.feedback = {"kind": "measurement_result", "reason": "requested_measurements_returned",
                             "frame_id": result.get("frame_id"), "status": result.get("status"),
                             "request_ids": [item.get("id") for item in result.get("results", [])]}
            self._log("tool_result", result=result)
            self.need_model = True
            return
        self.obstacles.update(answer.get("obstacle_map"), self.step_index)
        self.queue = deque(deepcopy(answer["segments"]))
        self.active = self.primitive = self.entry_state = None
        self.need_model = False
        self.feedback = {"kind": "executing", "reason": answer["why"]}

    def _resolved_candidate(self):
        candidate = deepcopy(self.active["candidates"][self.candidate_index])
        # All alternatives share the segment-entry reference, not a moving base.
        if candidate["skill"] in {"move_relative", "approach"}:
            candidate["xyz_m"] = (self.entry_state[:3]+np.asarray(candidate["xyz_m"])).tolist()
            candidate["skill"] = "move_to_pose"
        if candidate["skill"] == "move_to_pose" and candidate["quaternion_wxyz"] is None:
            candidate["quaternion_wxyz"] = measured_pose(self.entry_state)[3:7]
        return candidate

    def _activate(self, view):
        if self.active is None:
            if not self.queue:
                self._handoff("batch_complete", {"completed_segments": self.completed_segments})
                return False
            self.active = self.queue.popleft()
            self.entry_state = view.state.copy()
            self.candidate_index = 0
        candidate = self._resolved_candidate()
        self.primitive = RoutedFragment(view, candidate, self.gripper_command, self.obstacles.snapshot())
        self._log("candidate_activated", segment_id=self.active["id"],
                  candidate=self.candidate_index, resolved=candidate,
                  obstacle_map_revision=self.obstacles.revision)
        if self.primitive.done:
            self._candidate_failed(self.primitive.outcome())
            return False
        return True

    def _candidate_failed(self, outcome):
        issued = 0 if self.primitive is None else self.primitive.actions_executed
        self._log("candidate_failed", segment=_summary(self.active),
                  candidate=self.candidate_index, outcome=outcome)
        if self.primitive and self.primitive.pending:
            self.primitive.cancel_proposal()
        # Alternatives are tried locally only before the segment has moved.
        # Interrupted motion needs a fresh semantic decision on the actual state.
        self.candidate_index += 1
        if issued == 0 and self.candidate_index < len(self.active["candidates"]):
            self.primitive = None
            return
        self._handoff(outcome.get("reason", "action_unavailable"), _compact_outcome(outcome))

    def _handoff(self, reason, evidence):
        self.feedback = {"kind": "execution_feedback", "reason": reason,
                         "segment": _summary(self.active), "evidence": deepcopy(evidence),
                         "remaining_segment_ids": [s["id"] for s in self.queue]}
        self._log("handoff", **self.feedback)
        if self.active:
            self.history.append({"segment": _summary(self.active), "status": "interrupted",
                                 "reason": reason, "step": self.step_index})
        self.active = self.primitive = self.entry_state = None
        self.need_model = True

    def _decision(self, action, calls, kind, evidence=None):
        self.pending_kind = kind
        return Decision(tuple(float(x) for x in action), jsonable({
            "version": VERSION, "controller": VERSION, "step": self.step_index,
            "phase": None if self.active is None else self.active["id"],
            "primitive": "measured_v2_fragment" if kind == "fragment" else "observation_hold",
            "api_calls": calls, "api_latency_s": sum(c["latency_s"] for c in calls),
            "terminal": self.terminal, "motion_evidence": evidence or {},
            "goal_xyz_m": None if self.primitive is None else self.primitive._target_xyz,
            "goal_quaternion_wxyz": None if self.primitive is None else self.primitive._target_wxyz,
            "control_timing": {"robot_guard_wall_s": self._act_guard_seconds},
            "queue_remaining": len(self.queue)}))

    def _hold(self, view, calls):
        return self._decision(self.adapter.adapt([0., 0., 0.], view.quaternion_xyzw,
                              self.gripper_command, view.state), calls, "observation_hold")

    def act(self, observation):
        if self.pending_kind is not None:
            raise RuntimeError("feedback is required before the next policy action")
        same_observation = self.last_view is not None and self.last_view.observation is observation
        view = self.last_view if same_observation else StepView(observation)
        if not same_observation:
            self._observation_received_at_s = monotonic()
        self.last_view = view
        calls = []
        self._act_guard_seconds = 0.
        if self.terminal:
            return self._hold(view, calls)
        if getattr(observation, "joint_position", None) is None or getattr(observation, "geometry", None) is None:
            self._terminal("blocked", "public_robot_state_or_calibration_missing")
            return self._hold(view, calls)
        if self.need_model:
            self._ask(view, calls)
        # At most three candidates per segment; each accepted action is returned
        # immediately. This bound never caps the episode or invokes model loops.
        for _ in range(3):
            if self.terminal or self.need_model:
                return self._hold(view, calls)
            if self.primitive is None and not self._activate(view):
                continue
            action = self.primitive.next_action(view)
            started = monotonic()
            try:
                result = self.robot_guard.filter(action, view, self.active)
            finally:
                elapsed = monotonic()-started
                self.guard_seconds += elapsed
                self._act_guard_seconds += elapsed
            if result["status"] in {"execute", "scaled"}:
                actual = self.primitive.commit_action(result["action"])
                return self._decision(actual, calls, "fragment", result.get("evidence"))
            self._candidate_failed({"status": "failed", "reason": result["reason"],
                                    "evidence": result.get("evidence", {})})
        return self._hold(view, calls)

    def observe_transition(self, before, decision, after):
        old = self.last_view if self.last_view is not None and self.last_view.observation is before else StepView(before)
        current = StepView(after)
        previous_gripper = self.gripper_command
        self.gripper_command = "close" if decision.action[6] > 0 else "open"
        if previous_gripper != self.gripper_command:
            if self.gripper_command == "close":
                self.grasp_attempts += 1
                self._intended_grasp_object = None if self.active is None else self.active.get("object")
            else:
                self._last_released_object = self._intended_grasp_object
                self._intended_grasp_object = None
        if self.pending_kind == "fragment":
            self.primitive.observe_transition(old, decision.action, current)
            self.robot_guard.observe_transition(old, decision.action, current)
        self.step_index += 1
        self.last_view = current
        self._observation_received_at_s = monotonic()
        self.path.append(current.position.tolist())
        self.path = self.path[-1000:]
        if self.pending_kind == "fragment" and self.primitive.done:
            outcome = self.primitive.outcome()
            brief = _compact_outcome(outcome)
            self._log("segment_result", segment=_summary(self.active), outcome=outcome)
            if outcome["status"] == "succeeded":
                self.completed_segments += 1
                segment = self.active
                self.history.append({"segment": _summary(segment), **brief, "step": self.step_index})
                self.active = self.primitive = self.entry_state = None
                checkpoint = segment.get("checkpoint", "none")
                if checkpoint != "none":
                    target = segment.get("checkpoint_object")
                    if target is None:
                        target = self._last_released_object if checkpoint == "released" else self._intended_grasp_object
                    self._handoff("visual_checkpoint", {"state": checkpoint, "object": target,
                        "measured_result": brief, "semantic_verdict": "awaiting_Astra_RGB_review"})
                elif not self.queue:
                    self._handoff("batch_complete", brief)
            else:
                self._handoff(outcome["reason"], brief)
        self.pending_kind = None
        return jsonable({"feedback": self.feedback, "measured_gap_mm": current.gap_mm,
                         "grasp_success": "unknown", "task_success": "unknown"})

    def planning_snapshot(self):
        from ..robodojo_direct_sim import quaternion_matrix
        poses, selected = [], []
        if self.primitive is not None:
            poses = [{"name": self.active["id"], "position": self.primitive._target_xyz,
                      "rotation": quaternion_matrix(self.primitive._target_wxyz)}]
            route = self.primitive._route
            selected = route.path if route is not None else [self.last_view.position, self.primitive._target_xyz]
        return jsonable({"step": self.step_index, "stage": None if self.active is None else self.active["id"],
            "status": self.terminal or ("awaiting_model" if self.need_model else "executing"),
            "scene_points": [], "scene_point_strong": [], "keyposes": poses,
            "selected_path": selected, "measured_path": self.path[-500:],
            "obstacle_boxes": self.obstacles.snapshot()["avoid_boxes"],
            "geometry_source": "astra_estimated_boxes", "geometry_is_prediction": True})

    def agent_stats(self):
        return jsonable({"version": VERSION, "revision": REVISION, "step": self.step_index,
            "terminal": self.terminal, "model_decisions": self.model_decisions,
            "grasp_attempts": self.grasp_attempts, "completed_segments": self.completed_segments,
            "queue_remaining": len(self.queue), "segment_history": self.history,
            "feedback": self.feedback, "config": asdict(self.config),
            "model_requests": self.model_requests, "vision_tools": self.vision_tools.stats(),
            "robot_guard": self.robot_guard.stats(),
            "timing": {"model_s": self.model_seconds, "vision_tool_s": self.tool_seconds,
                       "robot_guard_s": self.guard_seconds, "wall_s": monotonic()-self.started_at}})

    def close(self):
        self.vision_tools.close()
