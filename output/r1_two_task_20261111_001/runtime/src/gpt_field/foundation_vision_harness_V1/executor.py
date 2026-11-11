"""Operation-level execution on measured regions, RGB and robot proprioception.

Heavy perception runs independently. This module never imports an evaluator or
uses its task-success/contact channels. A terminal decision is consumed BEFORE
dispatch by the V1 runner.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

import numpy as np

from ..contact_skills import StepView, HoldingActionAdapter, matrix_to_quaternion_xyzw
from ..foundation_harness.types import jsonable
from ..foundation_harness.feedback import grip_pose, transform_points
from ..foundation_vision.bridge import frame_from_step_view
from ..policy import Decision
from . import VERSION
from .feedback import MaterialTrack, PayloadState


@dataclass(frozen=True)
class HarnessConfig:
    perception_wait_s: float = 15.
    visual_servo_limit_m: float = .02
    visual_servo_step_m: float = .002
    close_min_steps: int = 12
    open_min_steps: int = 8
    settle_window: int = 5
    gap_stable_mm: float = .6
    empty_gap_mm: float = 3.
    phase_action_budget: int = 180
    progress_window: int = 18
    control_period_s: float = .05
    # Local stage bounds do not impose an episode/action/model-call cap.


def _signature(value):
    return sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FoundationVisionHarnessPolicy:
    def __init__(self, client, *, vision_engine=None, output_dir=None, kinematics=None,
                 config=None, scene_service=None, grounder=None, planner=None):
        self.client = client
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or HarnessConfig()
        if scene_service is None:
            from .scene import SceneService
            scene_service = SceneService(vision_engine,
                output_dir=None if self.output_dir is None else self.output_dir / "vision",
                kinematics=kinematics)
        if grounder is None or planner is None:
            from .geometry import Grounder
            from .motion import RollingPlanner
            grounder = grounder or Grounder(kinematics=kinematics)
            planner = planner or RollingPlanner(kinematics=kinematics)
        self.service, self.grounder, self.planner = scene_service, grounder, planner
        self.adapter = HoldingActionAdapter(max_rotation_rad=.08)
        self.reset()

    def reset(self):
        self.step_index = 0
        self.targets = {}
        self.scene = None
        self.operations = deque()
        self.current_operation = None
        self.phase = None
        self.phases = deque()
        self.keypose = self.plan = None
        self.nominal_pose = None
        self.phase_actions = 0
        self.phase_gripper_actions = 0
        self.phase_entry = None
        self.phase_gap = deque(maxlen=self.config.settle_window)
        self.progress = deque(maxlen=self.config.progress_window)
        self.gripper_command = "open"
        self.payload = PayloadState()
        self.region_track = MaterialTrack()
        self.terminal = None
        self.need_model = True
        self.feedback = {"reason": "initial_observation"}
        self.operation_history = []
        self.completion_conditions = []
        self.model_decisions = self.grasp_attempts = 0
        self.model_seconds = self.planning_seconds = self.vision_wait_seconds = 0.
        self.control_seconds = self.tracking_seconds = 0.
        self.local_refreshes = set()
        self.semantic_failures = set()
        self.last_observe_signature = None
        self.measured_path = []
        self.last_view = None
        self.last_command = None
        self.last_request_token = None
        self.source_token = None
        self._phase_refresh_pending = False
        self._articulated = False
        self._pending = None
        self._ask_again_after_observe = False
        self._fresh_after_model = False
        self.started_at = monotonic()

    def close(self):
        self.service.close()

    def _log(self, event, **fields):
        if self.output_dir is not None:
            with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(jsonable({"event": event, "step": self.step_index,
                    "at_s": monotonic(), **fields}), ensure_ascii=False, allow_nan=False)+"\n")

    def _terminal(self, kind, reason, evidence=None):
        self.terminal = {"kind": kind, "reason": reason, "evidence": jsonable(evidence or {}),
                         "step": self.step_index, "wall_s": monotonic()-self.started_at}
        self._log("terminal", **self.terminal)

    def _refresh(self, view, *, force=False, wait=False):
        started = monotonic()
        token = self.service.submit(view, list(self.targets.values()), force=force, frame_id=self.step_index)
        self.last_request_token = token
        if wait:
            snapshot = self.service.wait(token, timeout_s=self.config.perception_wait_s)
            self.vision_wait_seconds += monotonic()-started
            if snapshot is None:
                raise RuntimeError("perception_publication_unavailable")
        else:
            snapshot = self.service.latest()
            failures = getattr(self.service, "errors", [])
            published = 0 if snapshot is None else snapshot.evidence.get("request_token", 0)
            if failures and failures[-1].get("token", 0) > published:
                raise RuntimeError("perception_worker_failed: "+str(failures[-1].get("message", "unknown")))
        if snapshot is not None:
            self.scene = snapshot
        return self.scene

    def _compact_scene(self):
        if self.scene is None:
            return {"status": "pending", "objects": [], "frame_id": None}
        # Never serialize point_sigma_m, point clouds or the growing raw evidence.
        objects = []
        for object_id, obj in self.scene.objects.items():
            points = np.asarray(obj.points).reshape(-1, 3)
            boxes = {}
            for name, mask in obj.masks.items():
                yy, xx = np.nonzero(mask)
                if len(xx):
                    boxes[name] = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
            objects.append({"id": object_id, "query": obj.query, "status": obj.status,
                "role": self.targets.get(object_id, {}).get("role"),
                "parent_id": self.targets.get(object_id, {}).get("parent_id"),
                "points": len(points), "boxes_pixels": boxes,
                "bounds_m": [points.min(0), points.max(0)] if len(points) else None,
                "sigma_m": obj.evidence.get("sigma_m"),
                "source_frame": obj.evidence.get("observed_frame_id", self.scene.frame_id)})
        return jsonable({"frame_id": self.scene.frame_id,
            "revision": self.scene.evidence.get("revision"),
            "age_s": max(0., monotonic()-self.scene.captured_at_s), "objects": objects,
            "surface_points": len(self.scene.scene_points),
            "status": self.scene.evidence.get("perception_status", "ok"),
            "geometry_is_partial_prediction": True})

    def _context(self, view):
        return jsonable({"task_prompt": view.observation.prompt, "step": self.step_index,
            "frame_id": self.step_index, "scene": self._compact_scene(),
            "current_operation": self.current_operation, "current_phase": self.phase,
            "operation_history": self.operation_history[-6:], "feedback": self.feedback,
            "held": self.payload.summary(), "gripper_command": self.gripper_command,
            "robot_joints": view.observation.joint_position,
            "completion_conditions": self.completion_conditions,
            "timing": {"wall_s": monotonic()-self.started_at, "model_s": self.model_seconds},
            "capabilities": {"execution": "region_grounded_next_segment_continuous_OSC",
                "contact": "explicit object and robot parts; observed material feedback",
                "perception": "asynchronous predicted geometry; incomplete surfaces retain uncertainty",
                "first_response": "inventory, frame-specific regions and an operation sequence together",
                "local_continuation": "alignment, closure, lift, transport, placement, release, retreat",
                "termination": "finish after released goal evidence; blocked for unrecoverable local fault"}})

    def _ask(self, view, calls):
        started = monotonic()
        answer = self.client.request_harness(view, self._context(view))
        latency = monotonic()-started
        self.model_decisions += 1
        self.model_seconds += latency
        calls.append({"kind": "harness", "latency_s": latency, "ok": True})
        self._log("semantic_plan", answer=answer, latency_s=latency)
        if answer["status"] == "blocked":
            self._terminal("blocked", "semantic_blocked", {"why": answer["why"]})
            return
        if answer["status"] == "finish":
            if self.payload.held is not None or self.gripper_command != "open":
                self._terminal("blocked", "finish_without_release", self.payload.summary())
            else:
                self._terminal("finish", "policy_completion_claim", {"why": answer["why"],
                    "completion_conditions": answer["completion_conditions"]})
            return
        for target in answer["targets"]:
            self.targets[target["id"]] = deepcopy(target)
        self.completion_conditions = list(answer["completion_conditions"])
        self._refresh(view, force=True, wait=True)
        if answer["status"] == "observe":
            signature = _signature({"targets": [{k: v for k, v in t.items() if k != "frame_id"}
                        for t in answer["targets"]], "state": np.round(view.state[:6], 3)})
            if signature == self.last_observe_signature:
                self._terminal("blocked", "repeated_observation_without_new_action", {"signature": signature})
                return
            self.last_observe_signature = signature
            self.need_model = True
            self.feedback = {"reason": "requested_regions_processed", "scene": self._compact_scene()}
            self._ask_again_after_observe = True
            return
        self.last_observe_signature = None
        self.operations = deque(deepcopy(answer["operations"]))
        self.current_operation = None
        self.phase = None
        self.phases.clear()
        self.keypose = self.plan = None
        self.need_model = False
        self._fresh_after_model = True
        if self.operations and _signature(self.operations[0]) in self.semantic_failures:
            self._terminal("blocked", "unchanged_failed_operation", self.feedback)

    def _begin_operation(self, view):
        if not self.operations:
            self.need_model = True
            return False
        self.current_operation = self.operations.popleft()
        op = self.current_operation
        kind = op["kind"]
        self._articulated = (kind == "contact_move" or (kind == "grasp" and self.operations
            and self.operations[0]["kind"] == "contact_move"
            and self.operations[0].get("object") == op.get("object")))
        held = self.payload.held
        if held is not None and op.get("object") not in (None, held["object_id"]):
            self._fault("holding_different_object", semantic=True)
            return False
        if kind == "pick_place":
            phases = (["transport", "preplace", "place", "release", "retreat", "release_check"]
                if held and held.get("confirmed") else
                ["approach", "align", "close", "lift", "hold_check", "transport", "preplace", "place",
                 "release", "retreat", "release_check"])
        elif kind == "grasp":
            phases = ["approach", "align", "close"] if self._articulated else ["approach", "align", "close", "lift", "hold_check"]
        elif kind == "place":
            if held is None or not held.get("confirmed"):
                self._fault("place_requires_observed_hold", semantic=True)
                return False
            phases = ["transport", "preplace", "place", "release", "retreat", "release_check"]
        elif kind == "release":
            phases = ["release", "retreat", "release_check"]
        elif kind == "observe":
            self._refresh(view, force=True, wait=True)
            self._finish_operation("observed")
            return False
        else:
            phases = [kind]
        self.phases = deque(phases)
        self.phase = None
        self._advance_phase(view)
        self._log("operation_started", operation=op, phases=phases)
        return True

    def _advance_phase(self, view):
        if self.phase is not None:
            self._log("stage_result", phase=self.phase, actions=self.phase_actions,
                      interaction=self.payload.report, outcome="local_condition_reached")
        if not self.phases:
            self._finish_operation("operation_completed")
            return
        self.phase = self.phases.popleft()
        self.keypose = self.plan = self.nominal_pose = None
        self.phase_actions = 0
        self.phase_gripper_actions = 0
        self.phase_entry = view.state.copy()
        self.phase_gap.clear()
        self.progress.clear()
        self.region_track.reset()
        self._phase_refresh_pending = self.phase in {"align", "close", "preplace", "place", "release_check"}

    def _finish_operation(self, status):
        result = {"operation": deepcopy(self.current_operation), "status": status,
                  "step": self.step_index, "interaction": deepcopy(self.payload.report)}
        self.operation_history.append(result)
        self.feedback = result
        self._log("operation_result", **result)
        self.current_operation = self.phase = self.keypose = self.plan = None
        self.phases.clear()
        self.need_model = not self.operations

    def _fault(self, reason, *, detail=None, semantic=False):
        self.feedback = {"reason": reason, "detail": jsonable(detail or {}),
            "operation": deepcopy(self.current_operation), "phase": self.phase,
            "interaction": deepcopy(self.payload.report)}
        self._log("operation_interrupted", **self.feedback)
        if semantic and self.current_operation is not None:
            signature = _signature(self.current_operation)
            if signature not in self.semantic_failures:
                self.semantic_failures.add(signature)
                self.operation_history.append({**self.feedback, "status": "incomplete"})
                self.current_operation = self.phase = self.keypose = self.plan = None
                self.operations.clear(); self.phases.clear()
                self.need_model = True
                return
        self._terminal("blocked", reason, self.feedback)

    def _ground(self, view):
        if self._phase_refresh_pending:
            self._refresh(view, force=True, wait=True)
            self._phase_refresh_pending = False
        started = monotonic()
        self.keypose = self.grounder.ground(self.current_operation, self.scene,
            view.state, view.observation.joint_position, phase=self.phase, held=self.payload.held)
        if self.phase in {"approach", "align"} and self.current_operation.get("grasp_region"):
            self.region_track.seed(self.scene, self.current_operation["grasp_region"])
        self._plan_current(view)
        self.nominal_pose = self.keypose.pose.copy()
        elapsed = monotonic()-started
        self.planning_seconds += elapsed
        self._log("operation_plan", phase=self.phase, latency_s=elapsed,
                  keypose=self.keypose.to_dict(),
                  plan=self.plan.to_dict() if hasattr(self.plan, "to_dict") else {"available": True})

    def _plan_current(self, view):
        self.plan = self.planner.plan(self.keypose, self.scene, view.state,
                                     view.observation.joint_position, held=self.payload.held)
        self.keypose = getattr(self.plan, "keypose", self.keypose)
        # Keep the selected local candidate fixed across rolling replans.
        # A later visual correction changes this selected pose, not an obsolete
        # first candidate left in the metadata.
        self.keypose.metadata["candidate_poses"] = [self.keypose.pose.tolist()]

    def _local_refresh_retry(self, view, reason, detail):
        key = (reason, _signature(self.current_operation), self.phase)
        if key in self.local_refreshes:
            self._fault(reason, detail=detail)
            return False
        self.local_refreshes.add(key)
        self._refresh(view, force=True, wait=True)
        # The phase target is held fixed: relative motions are never re-based.
        self._plan_current(view)
        self._log("local_replan", reason=reason, phase=self.phase)
        return True

    def _visual_correction(self, view):
        evidence = self.region_track.evidence
        if self.phase != "align" or evidence.get("status") != "tracked" or self.nominal_pose is None:
            return
        correction = np.asarray(evidence["delta_from_seed_m"], float)
        bound = min(self.config.visual_servo_limit_m, max(.004, 3*float(evidence.get("sigma_m", .005))))
        if np.linalg.norm(correction) > bound:
            self.feedback = {"reason": "visual_region_moved_beyond_local_bound", "tracking": evidence}
            return
        desired = self.nominal_pose[:3, 3]+correction
        change = desired-self.keypose.pose[:3, 3]
        if np.linalg.norm(change) < .001:
            return
        change *= min(1., self.config.visual_servo_step_m/max(1e-9, np.linalg.norm(change)))
        self.keypose.pose[:3, 3] += change
        self.keypose.metadata["candidate_poses"] = [self.keypose.pose.tolist()]
        self._plan_current(view)
        self._log("visual_retarget", correction_m=change, evidence=evidence)

    def _confirm_hold_from_scene(self, view):
        held = self.payload.held
        if held is None or self.payload.close_object is None:
            return False
        obj = self.scene.objects.get(held["object_id"])
        if not self._fresh_object(obj) or len(obj.points) < 8:
            return False
        observed = np.median(np.asarray(obj.points), axis=0)
        initial = self.payload.close_object[:3, 3]
        predicted = (grip_pose(view.state) @ np.linalg.inv(self.payload.close_grip)
                     @ self.payload.close_object)[:3, 3]
        moved = float(np.linalg.norm(predicted-initial))
        error = float(np.linalg.norm(observed-predicted))
        stationary_error = float(np.linalg.norm(observed-initial))
        sigma = float(np.max(np.asarray(obj.evidence.get("sigma_m", .01))))
        confirmed = (moved >= max(.008, sigma) and error <= max(.01, 2*sigma)
                     and stationary_error > error+.003 and view.gap_mm > self.config.empty_gap_mm)
        if confirmed:
            held["confirmed"] = True
            self.payload.report = {"state": "held_confirmed", "object_id": held["object_id"],
                "source": "reobserved_body_comotion", "error_m": error,
                "stationary_error_m": stationary_error, "physical_contact_measured": False}
        return confirmed

    def _fresh_object(self, obj):
        return (obj is not None and obj.status not in {"missing", "ambiguous", "conflict", "stale", "unobserved"}
                and obj.evidence.get("observed_frame_id", self.scene.frame_id) == self.scene.frame_id)

    def _check_phase(self, view):
        if self.phase == "hold_check":
            if self.payload.held is not None and self.payload.held.get("confirmed"):
                self._advance_phase(view)
                return True
            self._refresh(view, force=True, wait=True)
            if self._confirm_hold_from_scene(view):
                self._advance_phase(view)
            else:
                self._fault("grasp_not_established", semantic=True)
            return True
        if self.phase == "release_check":
            self._refresh(view, force=True, wait=True)
            released = self.payload.released
            if released is None:
                self._advance_phase(view)
                return True
            obj = self.scene.objects.get(released["object_id"])
            evidence = {"visual": self.payload.report, "relation": self.current_operation.get("relation")}
            observed = self._fresh_object(obj) and len(obj.points) >= 8
            if observed:
                centre = np.median(obj.points, axis=0)
                old = np.median(released["predicted_world_points"], axis=0)
                hand_delta = view.position-released["released_grip"][:3, 3]
                object_delta = centre-old
                sigma = float(np.max(np.asarray(obj.evidence.get("sigma_m", .01))))
                stationary_error = float(np.linalg.norm(object_delta))
                attached_error = float(np.linalg.norm(object_delta-hand_delta))
                evidence.update(stationary_error_m=stationary_error, attached_error_m=attached_error,
                                frame_id=self.scene.frame_id)
                observed &= stationary_error <= max(.02, 3*sigma) and attached_error > stationary_error+.003
            if self.payload.report.get("state") == "released_stationary":
                observed = True
            relation_check = getattr(self.grounder, "check_release_relation", None)
            relation = self.current_operation.get("relation")
            if relation is not None:
                relation_evidence = (relation_check(self.current_operation, self.scene, obj.points)
                    if callable(relation_check) and self._fresh_object(obj) else
                    {"status": "unknown", "reason": "fresh_relation_geometry_unavailable"})
                evidence["relation_evidence"] = relation_evidence
                observed &= relation_evidence.get("status") == "supported"
            if observed:
                self.payload.report = {"state": "release_observed", "object_id": released["object_id"],
                                       "evidence": evidence, "support_contact_measured": False}
                self._advance_phase(view)
            else:
                self._fault("release_relation_needs_visual_review", detail=evidence, semantic=True)
            return True
        return False

    def _hold(self, view, calls=()):
        action = self.adapter.adapt(np.zeros(3), view.quaternion_xyzw, self.gripper_command, view.state)
        return self._decision(action, calls, {"holding_position": True})

    def _decision(self, action, calls, evidence):
        self._pending = {"phase": self.phase, "operation": deepcopy(self.current_operation),
                         "evidence": evidence, "previous_command": self.gripper_command}
        return Decision(tuple(float(x) for x in action), jsonable({"version": VERSION,
            "controller": "foundation_vision_harness_V1", "step": self.step_index,
            "phase": self.phase or "semantic_decision", "primitive": "fvh_v1:"+str(self.phase),
            "api_calls": list(calls), "api_latency_s": sum(c["latency_s"] for c in calls),
            "terminal": self.terminal, "motion_evidence": evidence,
            "scene_revision": None if self.scene is None else self.scene.evidence.get("revision")}))

    def act(self, observation):
        started = monotonic()
        view = self.last_view if self.last_view is not None and self.last_view.observation is observation else StepView(observation)
        self.last_view = view
        calls = []
        if self.terminal is not None:
            return self._hold(view)
        try:
            if getattr(observation, "geometry", None) is None or observation.joint_position is None:
                self._terminal("blocked", "public_calibration_or_joint_feedback_missing")
                return self._hold(view)
            self._refresh(view)
            if self._fresh_after_model:
                # The response may have taken tens of seconds. Its ROIs were
                # resolved on that exact source image, then a real hold step
                # acquired a new observation before any task motion is sent.
                self._refresh(view, force=True, wait=True)
                self._fresh_after_model = False
            if self.need_model:
                self._ask(view, calls)
                if self.terminal is not None or self._ask_again_after_observe or self._fresh_after_model:
                    self._ask_again_after_observe = False
                    return self._hold(view, calls)
            # Only state transitions, not robot steps, can iterate here.
            for _ in range(16):
                if self.terminal is not None or self.need_model:
                    return self._hold(view, calls)
                if self.current_operation is None:
                    if not self._begin_operation(view):
                        continue
                if self._check_phase(view):
                    continue
                if self.scene is None:
                    self._refresh(view, force=True, wait=True)
                if self.keypose is None:
                    try:
                        self._ground(view)
                    except (ValueError, RuntimeError) as error:
                        reason = getattr(error, "reason", str(error))
                        self._fault(reason, detail=getattr(error, "evidence", {}),
                            semantic=any(word in reason for word in ("region", "identity", "reference", "support")))
                        return self._hold(view, calls)
                self._visual_correction(view)
                command = self.planner.command(self.plan, self.scene, view.state,
                    observation.joint_position, getattr(observation, "joint_velocity", None), held=self.payload.held)
                self.last_command = command
                if command.get("blocked"):
                    evidence = command.get("evidence", {})
                    reason = evidence.get("reason", "motion_blocked")
                    if self._local_refresh_retry(view, reason, evidence):
                        continue
                    return self._hold(view, calls)
                arrived = bool(command.get("arrived"))
                if arrived and self.phase not in {"close", "release"}:
                    self._advance_phase(view)
                    continue
                gripper = command.get("gripper", "keep")
                if self.phase == "close" and arrived:
                    gripper = "close"
                elif self.phase == "release" and arrived:
                    gripper = "open"
                if gripper == "keep":
                    gripper = self.gripper_command
                if self.phase in {"close", "release"} and arrived:
                    minimum = self.config.close_min_steps if self.phase == "close" else self.config.open_min_steps
                    stable = (len(self.phase_gap) == self.config.settle_window
                              and max(self.phase_gap)-min(self.phase_gap) <= self.config.gap_stable_mm)
                    if self.phase_gripper_actions >= minimum and stable:
                        if self.phase == "close" and view.gap_mm <= self.config.empty_gap_mm:
                            self.payload.held = None
                            self._fault("grasp_empty_close", semantic=True)
                        else:
                            self._advance_phase(view)
                        continue
                action = self.adapter.adapt(command["delta_xyz"], command["quaternion_xyzw"], gripper, view.state)
                return self._decision(action, calls, command.get("evidence", {}))
            self._terminal("blocked", "local_transition_cycle")
            return self._hold(view, calls)
        except Exception as error:
            self._terminal("blocked", "local_service_error", {"type": type(error).__name__, "message": str(error)})
            self._log("local_error", error_type=type(error).__name__, message=str(error))
            return self._hold(view, calls)
        finally:
            self.control_seconds += monotonic()-started

    def observe_transition(self, before, decision, after):
        before_view, view = StepView(before), StepView(after)
        pending = self._pending or {}
        phase = pending.get("phase")
        old_command = self.gripper_command
        self.gripper_command = "close" if decision.action[6] > 0 else "open"
        if phase == "close" and self.gripper_command == "close" and old_command != "close":
            self.grasp_attempts += 1
            if not self._articulated:
                self.payload.attach_candidate(self.scene, self.current_operation["object"], before_view,
                    keep_upright=self.current_operation.get("keep_upright", False))
            self._log("grasp_attempt", object_id=self.current_operation.get("object"), articulated=self._articulated)
        if self.gripper_command == "open" and old_command == "close":
            self.payload.released_at(before_view)
            invalidate = getattr(self.service, "invalidate", None)
            if callable(invalidate) and self.current_operation and self.current_operation.get("object"):
                invalidate([self.current_operation["object"]], reason="released_reobserve")
        self.step_index += 1
        self.last_view = view
        self.measured_path.append(view.position.tolist())
        if len(self.measured_path) > 1000:
            self.measured_path = self.measured_path[-1000:]
        started = monotonic()
        frame = frame_from_step_view(view, after.geometry, self.step_index, monotonic())
        self.region_track.update(frame)
        self.payload.observe(view, frame, self.gripper_command)
        self.tracking_seconds += monotonic()-started
        if phase == self.phase and phase is not None:
            self.phase_actions += 1
            expected = "close" if phase == "close" else "open" if phase == "release" else None
            if expected is not None and self.gripper_command == expected:
                self.phase_gripper_actions += 1
                self.phase_gap.append(view.gap_mm)
            if self.keypose is not None:
                distance = float(np.linalg.norm(view.position-self.keypose.pose[:3, 3]))
                angle = float(np.arccos(np.clip((np.trace(self.keypose.pose[:3, :3] @ view.R.T)-1)/2, -1, 1)))
                self.progress.append(distance+.05*angle)
                if self.phase_actions >= self.config.phase_action_budget:
                    self._fault("local_phase_budget", detail={"phase_actions": self.phase_actions})
                elif phase not in {"close", "release"} and len(self.progress) == self.config.progress_window:
                    stalled = self.progress[0]-min(self.progress) < .0008 and distance > .004
                    if stalled:
                        if self.current_operation.get("stop_on") == "contact" and phase == "contact_move":
                            # Progress is only evidence of blocked motion; no contact sensor is invented.
                            self.feedback = {"reason": "guarded_motion_stopped", "contact_measured": False}
                            self._advance_phase(view)
                        else:
                            self._fault("measured_motion_stalled", detail={"distance_m": distance}, semantic=True)
        if self.payload.report.get("state") in {"suspected_slip", "object_absent"} and self.phase not in {"close", "release"}:
            self._fault(self.payload.report["state"], detail=self.payload.report, semantic=True)
        self._pending = None
        feedback = jsonable({"phase": phase, "gripper": self.gripper_command,
            "interaction": self.payload.report, "terminal": self.terminal})
        self._log("feedback", measured_state=after.state, joint_position=after.joint_position, **feedback)
        return feedback

    def planning_snapshot(self):
        cloud = np.empty((0, 3)) if self.scene is None else np.asarray(self.scene.scene_points)
        if len(cloud) > 1600:
            cloud = cloud[np.linspace(0, len(cloud)-1, 1600, dtype=int)]
        keyposes = [] if self.keypose is None else [{"name": self.phase,
            "position": self.keypose.pose[:3, 3], "rotation": self.keypose.pose[:3, :3]}]
        waypoints = [] if self.plan is None else getattr(self.plan, "waypoints", [])
        positions = [np.asarray(pose)[:3, 3] for pose in waypoints]
        return jsonable({"step": self.step_index, "stage": self.phase,
            "status": self.terminal or ("awaiting_model" if self.need_model else "executing"),
            "object_id": (self.current_operation or {}).get("object"), "scene_points": cloud,
            "scene_point_strong": np.zeros(len(cloud), bool), "keyposes": keyposes,
            "selected_path": positions, "measured_path": self.measured_path[-500:],
            "geometry_is_prediction": True})

    def agent_stats(self):
        return jsonable({"version": VERSION, "step": self.step_index, "phase": self.phase,
            "terminal": self.terminal, "model_decisions": self.model_decisions,
            "grasp_attempts": self.grasp_attempts, "operation_history": self.operation_history,
            "feedback": self.feedback, "payload": self.payload.summary(),
            "config": asdict(self.config), "perception": getattr(self.service, "stats", {}),
            "timing": {"model_s": self.model_seconds, "perception_wait_s": self.vision_wait_seconds,
                "planning_s": self.planning_seconds, "act_s": self.control_seconds,
                "tracking_s": self.tracking_seconds, "wall_s": monotonic()-self.started_at}})
