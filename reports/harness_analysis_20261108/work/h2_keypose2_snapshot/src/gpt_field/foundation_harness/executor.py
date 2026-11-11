"""Event-driven operations executed on public RGB and measured robot state.

The external evaluator owns task success. Local stage budgets only bound a
single attempted motion; they are not episode time/action/model-call caps.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
import json

import numpy as np

from .. import agent_speedup_v2 as base
from ..contact_skills import Primitive, HoldingActionAdapter, matrix_to_quaternion_xyzw
from ..foundation_vision.bridge import frame_from_step_view
from ..policy import Decision
from ..robodojo_direct_sim import PandaKinematics
from .feedback import InteractionTracker, grip_pose
from .types import KeyPose, jsonable

POLICY_VERSION = "foundation-harness-keypose-2"


@dataclass(frozen=True)
class HarnessConfig:
    refresh_steps: int = 10
    max_geometry_age_s: float = 5.
    transit_step_m: float = .02
    contact_step_m: float = .003
    stage_steps: int = 180
    local_replans: int = 1


def _unused_episode(_ctx):
    if False:
        yield


class JointPathStage(Primitive):
    """Track the planned branch through short measured OSC targets."""
    def __init__(self, view, keypose, joint_path, bound, keep_gripper, config):
        super().__init__("foundation:" + keypose.name)
        self.keypose, self.config, self.bound = keypose, config, bound
        self.path = [np.asarray(q, float) for q in joint_path]
        self.poses = [bound.forward(q)[0] for q in self.path]
        if not self.poses:
            self.poses = [np.asarray(keypose.pose, float)]
        self.index = 0
        self.keep_gripper = keep_gripper
        self.commanded_gripper = keep_gripper
        self.adapter = HoldingActionAdapter(max_rotation_rad=.08)
        self._errors = deque(maxlen=18)
        self._gaps = deque(maxlen=5)
        self._grip_actions = 0
        self._arrival = 0
        self.entry = view.position.copy()

    def errors(self, view, pose=None):
        target = self.poses[self.index] if pose is None else pose
        angle = np.arccos(np.clip((np.trace(target[:3, :3] @ view.R.T)-1)/2, -1, 1))
        return float(np.linalg.norm(target[:3, 3]-view.position)), float(angle)

    def next_joint(self, view):
        # Advance only after the preceding waypoint was actually reached.
        while self.index < len(self.poses)-1:
            distance, angle = self.errors(view)
            if distance > .004 or angle > .055:
                break
            self.index += 1
        return self.path[min(self.index, len(self.path)-1)] if self.path else None

    def _next(self, view):
        self.next_joint(view)
        target = self.poses[self.index]
        delta = target[:3, 3]-view.position
        limit = self.config.contact_step_m if self.keypose.profile == "contact" else self.config.transit_step_m
        norm = np.linalg.norm(delta)
        if norm > limit:
            delta *= limit/norm
        distance, angle = self.errors(view, self.keypose.pose)
        at_final = self.index == len(self.poses)-1 and distance <= .003 and angle <= .05
        self.commanded_gripper = self.keypose.gripper if at_final and self.keypose.gripper != "keep" else self.keep_gripper
        return self.adapter.adapt(delta, matrix_to_quaternion_xyzw(target[:3, :3]),
                                  self.commanded_gripper, view.state)

    def _observe(self, before, action, after):
        distance, angle = self.errors(after, self.keypose.pose)
        waypoint_distance, waypoint_angle = self.errors(after)
        metric = waypoint_distance + .05*waypoint_angle
        self._errors.append((self.index, metric))
        at_pose = self.index == len(self.poses)-1 and distance <= .003 and angle <= .05
        self._arrival = self._arrival+1 if at_pose else 0
        if self.keypose.gripper != "keep" and self.commanded_gripper == self.keypose.gripper and at_pose:
            self._grip_actions += 1
            self._gaps.append(after.gap_mm)
            minimum = 12 if self.keypose.gripper == "close" else 8
            if self._grip_actions >= minimum and len(self._gaps) == 5 and max(self._gaps)-min(self._gaps) < .6:
                self._finish("succeeded", "gripper_settled_not_attachment", {"gap_mm": after.gap_mm})
        elif self.keypose.gripper == "keep" and self._arrival >= 2:
            self._finish("succeeded", "measured_pose_reached", {})
        if not self.done and self.actions_executed >= self.config.stage_steps:
            self._finish("failed", "local_stage_budget", {})
        if not self.done and len(self._errors) == 18 and len({x[0] for x in self._errors}) == 1:
            errors = [x[1] for x in self._errors]
            if errors[0]-min(errors[1:]) < .0008 and not at_pose:
                self._finish("failed", "motion_stalled", {})
        return {"position_error_m": distance, "orientation_error_rad": angle,
                "waypoint_index": self.index, "gripper": self.commanded_gripper,
                "goal": self.keypose.to_dict()}


class ObservationHold(Primitive):
    def __init__(self, view, command, reason):
        super().__init__("foundation:observation")
        self.pose, self.command, self.reason_text = grip_pose(view.state), command, reason
        self.adapter = HoldingActionAdapter(max_rotation_rad=.05)

    def _next(self, view):
        delta = self.pose[:3, 3]-view.position
        length = np.linalg.norm(delta)
        if length > .003:
            delta *= .003/length
        return self.adapter.adapt(delta, matrix_to_quaternion_xyzw(self.pose[:3, :3]), self.command, view.state)

    def _observe(self, before, action, after):
        self._finish("succeeded", self.reason_text, {})
        return {"reason": self.reason_text}


class FoundationHarnessPolicy(base.AgentSpeedUpV2Policy):
    def __init__(self, client, *, vision_engine, output_dir, kinematics=None, lock=None, config=None, **kwargs):
        from .perception import HarnessPerception
        self.harness_config = config or HarnessConfig(max_geometry_age_s=vision_engine.config.max_age_s)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.perception = HarnessPerception(vision_engine, output_dir=self.output_dir / "vision",
            refresh_steps=self.harness_config.refresh_steps, max_age_s=self.harness_config.max_geometry_age_s)
        self.kinematics = kinematics
        self.plan_lock = lock
        super().__init__(client, episode_factory=_unused_episode, **kwargs)

    def reset(self):
        super().reset()
        if hasattr(self.perception, "reset"):
            self.perception.reset()
        self.tracker = InteractionTracker()
        self.scene = None
        self.targets = []
        self.target_catalog = {}
        self.operations = deque()
        self.current_operation = None
        self.operation_plan = None
        self.stage_index = 0
        self.keypose_index = 0
        self.stage = None
        self.planner = None
        self.gripper_command = "open"
        self.need_model = True
        self.fresh_after_model = False
        self.force_vision = True
        self.feedback = {"reason": "initial_scene"}
        self.operation_history = []
        self.completion_conditions = []
        self.model_decisions = 0
        self.model_seconds = self.planning_seconds = self.vision_seconds = 0.
        self.local_replans = 0
        self.grasp_attempts = 0
        self.withheld_decisions = 0
        self.measured_path = []
        self._last_frame_id = None
        self.possible_payload = None
        self.motion_evidence = None

    @property
    def holding(self):
        held = self.tracker.held
        return held["object_id"] if held and held.get("confirmed") and self.gripper_command == "close" else None

    @property
    def relation_plan(self):
        # Semantic operation contracts have their own versioned schema; do not
        # pass them off as the legacy benchmark-independent relation grammar.
        return None

    def _log(self, event, **fields):
        row = jsonable({"event": event, "step": self.step_index, "at_s": perf_counter(), **fields})
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")

    def _scene_update(self, view):
        geometry = getattr(view.observation, "geometry", None)
        if geometry is None or getattr(view.observation, "joint_position", None) is None:
            raise ValueError("foundation harness requires public RGB calibration and measured arm joints")
        if not self.targets:
            from .client import bootstrap_targets
            self.targets = bootstrap_targets(view.observation.prompt)
        started = perf_counter()
        self.scene = self.perception.update(view, geometry, self.targets, force=self.force_vision)
        self.vision_seconds += perf_counter()-started
        self.force_vision = False
        if self.planner is None:
            from .motion import CandidatePlanner, MotionConfig
            self.kinematics = self.kinematics or PandaKinematics.from_xml()
            self.planner = CandidatePlanner(self.kinematics,
                MotionConfig(max_geometry_age_s=self.harness_config.max_geometry_age_s))
        return self.scene

    def _context(self, view):
        held = self.tracker.held
        return jsonable({"task_prompt": view.observation.prompt, "step": self.step_index,
            "scene": self.scene.summary(perf_counter()), "current_operation": self.current_operation,
            "operation_history": self.operation_history[-12:], "feedback": self.feedback,
            "held": None if held is None else {k: v for k, v in held.items() if k != "points_in_gripper"},
            "possible_payload": None if self.possible_payload is None else {
                k: v for k, v in self.possible_payload.items() if k != "points_in_gripper"},
            "gripper_command": self.gripper_command,
            "keypose_index": self.keypose_index, "motion_evidence": self.motion_evidence,
            "robot_joints": view.observation.joint_position,
            "capabilities": {"operations": ["pick_place", "grasp", "place", "move", "contact_move", "release", "observe"],
                "placement": "Astra chooses contact/support pose from RGB and partial geometry; local feedback reports outcome",
                "execution": "Astra keyposes; one local joint-path segment at a time; measured OSC feedback",
                "observe_after": "interrupts the remaining batch at this reached keypose and requests a new RGB-grounded plan",
                "unknown_space": "reported, traversed in bounded controller increments with renewed checks; not certified free",
                "completion": "local outcomes are evidence; only external evaluator determines task success",
                "vision_regions": "region_pixels restricts an explicitly selected candidate to a task-relevant surface",
                "geometry": "partial predicted surfaces, no unobserved shape guarantee"}})

    def _ask(self, view, calls):
        # Candidate IDs/regions must refer to the exact RGB pair sent in this
        # request, even if the previous local refresh was several steps ago.
        self.force_vision = True
        self._scene_update(view)
        started = perf_counter()
        answer = self.client.request_harness(view, self._context(view))
        elapsed = perf_counter()-started
        self.model_decisions += 1
        self.model_seconds += elapsed
        calls.append({"kind": "harness", "latency_s": elapsed, "ok": True})
        self._log("semantic_plan", answer=answer, latency_s=elapsed)
        known_targets = self.target_catalog
        known_targets.update({target["id"]: deepcopy(target) for target in self.targets})
        known_targets.update({target["id"]: deepcopy(target) for target in answer["targets"]})
        active_ids = {target["id"] for target in answer["targets"]}
        for operation in answer["operations"]:
            active_ids.update(operation[key] for key in ("object", "target", "grasp_region") if operation.get(key))
            region, parent = operation.get("grasp_region"), operation.get("object")
            if region and parent and region != parent and region in known_targets:
                known_targets[region]["parent_id"] = parent
        if self.tracker.object_id:
            active_ids.add(self.tracker.object_id)
        for object_id in list(active_ids):
            parent = known_targets.get(object_id, {}).get("parent_id")
            while parent:
                active_ids.add(parent)
                parent = known_targets.get(parent, {}).get("parent_id")
        self.targets = [target for key, target in known_targets.items() if key in active_ids] or self.targets
        self.operations = deque(answer["operations"])
        self.completion_conditions = answer["completion_conditions"]
        self.current_operation = None
        self.operation_plan = self.stage = None
        # Bind selections to the image actually shown to Astra, before a fresh
        # sensor observation. This does not refresh the old image timestamp.
        self.scene = self.perception.update(view, view.observation.geometry, self.targets)
        self.need_model = answer["status"] != "plan" or not self.operations
        self.fresh_after_model = True
        self.force_vision = True
        self.feedback = {"reason": "fresh_observation_after_semantic_decision", "model_status": answer["status"],
                         "completion_claim_is_evaluator_success": False}

    def _plan(self, view, *, operation=None):
        started = perf_counter()
        op = operation or self.current_operation
        model_keyposes = bool(op.get("keyposes"))
        if model_keyposes:
            op = {**op, "_keypose_index": self.keypose_index}
        try:
            plan = self.planner.plan(op, self.scene, view.state, view.observation.joint_position,
                                     held=self._collision_payload())
        except (ValueError, RuntimeError) as exc:
            self.planning_seconds += perf_counter()-started
            detail = {"reason": getattr(exc, "reason", str(exc)),
                      "evidence": deepcopy(getattr(exc, "evidence", {})),
                      "keypose_index": self.keypose_index}
            self._fail("planning_unavailable", detail=detail, allow_local=False)
            return False
        self.planning_seconds += perf_counter()-started
        self.operation_plan, self.stage_index, self.stage = plan, 0, None
        self._log("operation_plan", plan=plan.to_dict())
        if not plan.keyposes:
            self._finish_operation("observed", {"reason": "no_motion_requested"})
            return False
        return True

    def _fail(self, reason, *, detail=None, allow_local=True):
        self.feedback = {"reason": reason, "detail": detail, "operation": self.current_operation,
                         "interaction": deepcopy(self.tracker.report)}
        self._log("operation_interrupted", **self.feedback)
        self.stage = None
        self.operation_plan = None
        if allow_local and self.local_replans < self.harness_config.local_replans:
            self.local_replans += 1
            self.force_vision = True
        else:
            self.operation_history.append({**self.feedback, "status": "incomplete"})
            self.operations.clear()
            self.current_operation = None
            self.need_model = True

    def _finish_operation(self, status, evidence):
        outcome = {"operation": self.current_operation, "status": status, "evidence": evidence,
                   "step": self.step_index, "task_success": "not_available_to_policy"}
        self.operation_history.append(outcome)
        self._log("operation_result", **outcome)
        self.feedback = outcome
        self.current_operation = self.operation_plan = self.stage = None
        self.need_model = not self.operations

    def _completed_stage(self, view):
        stage = self.stage
        if stage.status != "succeeded":
            self._fail(stage.reason, detail=stage.outcome())
            return
        keypose = stage.keypose
        if self.current_operation.get("keyposes"):
            return self._completed_model_keypose(view, stage)
        # Standalone directed move/release uses the same feedback semantics as a
        # model keypose: reaching it does not certify attachment or task success.
        if keypose.gripper == "open":
            self.tracker.held = None
            self.possible_payload = None
            self.force_vision = True
        self._log("stage_result", stage=keypose.name, outcome=stage.outcome(), interaction=self.tracker.report)
        self._finish_operation("segment_completed", {"interaction": deepcopy(self.tracker.report),
                                                    "pose_source": "astra"})

    def _completed_model_keypose(self, view, stage):
        """Commit only the reached model waypoint, then plan from measured state."""
        keypose = stage.keypose
        if keypose.gripper == "close":
            proposed = keypose.metadata.get("payload_after")
            if proposed is not None:
                self.possible_payload = deepcopy(proposed)
                self.possible_payload["confirmed"] = False
        elif keypose.gripper == "open":
            # Opening releases the transport hypothesis immediately. Stationary
            # visual evidence is reported separately and is never fabricated.
            self.possible_payload = None
            self.tracker.held = None
        self._log("stage_result", stage=keypose.name, keypose_index=self.keypose_index,
                  outcome=stage.outcome(), interaction=self.tracker.report,
                  pose_source="astra", motion_evidence=self.motion_evidence)
        self.keypose_index += 1
        remaining = len(self.current_operation["keyposes"])-self.keypose_index
        observe_after = bool(keypose.metadata.get("observe_after"))
        self.operation_plan = self.stage = None
        self.stage_index = 0
        if observe_after or not remaining:
            evidence = {"reached_keyposes": self.keypose_index, "remaining_keyposes": remaining,
                        "interaction": deepcopy(self.tracker.report),
                        "grasp_confirmed": self.holding is not None,
                        "observe_after": observe_after, "pose_source": "astra",
                        "reason": "requested_observation" if observe_after else "provided_segment_finished"}
            self._finish_operation("segment_completed", evidence)
            if observe_after:
                self.operations.clear()
                self.need_model = True
            self.force_vision = True
            return
        # Close/open/contact transitions get new geometry; subsequent poses are
        # connected only now, so unavailable future geometry cannot block approach.
        if keypose.profile == "contact" or keypose.name == "pregrasp" or keypose.gripper != "keep":
            self.force_vision = True
            self._scene_update(view)
        self._plan(view)

    def _choose_stage(self, view):
        if self.current_operation is None and self.operations:
            self.current_operation = self.operations.popleft()
            self.keypose_index = 0
            self.local_replans = 0
        if self.current_operation is None:
            self.need_model = True
            return None
        if self.current_operation["kind"] == "observe" and not self.current_operation.get("keyposes"):
            self._finish_operation("observation_requested", deepcopy(self.tracker.report))
            self.operations.clear()
            self.need_model = self.force_vision = True
            return None
        if self.operation_plan is None and not self._plan(view):
            return None
        if self.stage is not None and self.stage.done:
            self._completed_stage(view)
            if self.stage is not None and self.stage.done:
                return None  # obtain another sensor frame for local evidence
        if self.operation_plan is None or self.current_operation is None:
            return None
        if self.stage is None:
            pose = self.operation_plan.keyposes[self.stage_index]
            path = self.operation_plan.joint_paths[self.stage_index]
            if pose.gripper == "close" and self.gripper_command != "close":
                self.grasp_attempts += 1
                object_id = self.current_operation.get("object")
                self.tracker.seed(self.scene, object_id, view)
                self.tracker.contact_mode = self.current_operation.get("contact_mode", "free_object")
                self.tracker.keep_upright = self.current_operation.get("keep_upright", False)
                self._log("grasp_attempt", object_id=object_id, attempt=self.grasp_attempts,
                          interaction=self.tracker.report)
            self.stage = JointPathStage(view, pose, path, self.planner.bound,
                                        self.gripper_command, self.harness_config)
        return self.stage

    def _collision_payload(self):
        if self.tracker.held is not None:
            return self.tracker.held
        # The first post-close lift still carries a possible payload. Its
        # geometric hypothesis is used for collision checking while visual
        # attachment evidence is pending; it does not set holding/confirmed.
        payload = self.possible_payload
        if self.operation_plan is not None:
            for pose in self.operation_plan.keyposes[:self.stage_index]:
                if "payload_after" in pose.metadata:
                    payload = pose.metadata["payload_after"]
                if pose.metadata.get("release"):
                    payload = None
        return payload

    def act(self, observation):
        if self._pending is not None:
            raise RuntimeError("observe_transition must acknowledge the pending action")
        if self._expected_observation is not None and observation is not self._expected_observation:
            raise ValueError("act requires the latest acknowledged public observation")
        view = self._view(observation)
        self.ctx.view, self.ctx.step = view, self.step_index
        calls = []
        self._scene_update(view)
        if self.fresh_after_model:
            self.fresh_after_model = False
        acting = None
        if self.need_model:
            self._ask(view, calls)
            acting = ObservationHold(view, self.gripper_command, "fresh_observation_after_model")
        elif perf_counter()-self.scene.captured_at_s > self.harness_config.max_geometry_age_s:
            self.force_vision = True
            acting = ObservationHold(view, self.gripper_command, "stale_geometry")
        else:
            acting = self._choose_stage(view)
            if acting is not None:
                targetq = acting.next_joint(view)
                if targetq is not None:
                    checked = self.planner.check_motion(observation.joint_position, targetq, self.scene,
                        view.state, held=self._collision_payload(), allowed_contacts=acting.keypose.allowed_contacts,
                        fingers=view.state[6:8],
                        contact_parts=acting.keypose.metadata.get("contact_parts"))
                    self.motion_evidence = checked
                    if not checked.get("safe", False):
                        self._fail("motion_check_blocked", detail=checked)
                        acting = None
                if acting is not None and acting.keypose.gripper != "keep":
                    distance, angle = acting.errors(view, acting.keypose.pose)
                    if distance <= .003 and angle <= .05:
                        finger_goal = acting.keypose.metadata.get("fingers", [.04, -.04] if acting.keypose.gripper == "open" else [0., 0.])
                        checked = self.planner.check_gripper(observation.joint_position, self.scene, view.state,
                            finger_goal, held=self._collision_payload(), allowed_contacts=acting.keypose.allowed_contacts,
                            incremental_closure=acting.keypose.metadata.get("incremental_closure", False),
                            contact_parts=acting.keypose.metadata.get("contact_parts"))
                        self.motion_evidence = checked
                        if not checked.get("safe", False):
                            self._fail("gripper_sweep_blocked", detail=checked)
                            acting = None
                if self.tracker.held and self.gripper_command == "close" and self.tracker.report.get("slip") == "suspected":
                    self._fail("held_object_slip", allow_local=False)
                    acting = None
        if acting is None:
            acting = ObservationHold(view, self.gripper_command, self.feedback.get("reason", "local_feedback"))
        elif not isinstance(acting, ObservationHold) and perf_counter()-self.scene.captured_at_s > self.harness_config.max_geometry_age_s:
            # Planning itself can be slow. Never stamp the old snapshot as fresh
            # or execute from it merely because it was fresh before planning.
            self.force_vision = True
            acting = ObservationHold(view, self.gripper_command, "geometry_aged_during_planning")
        self._primitive = acting
        action = tuple(acting.next_action(view))
        diagnostics = jsonable({"step": self.step_index, "controller": self.controller,
            "macro": "foundation_operation", "primitive": acting.name,
            "execution_kind": "foundation_harness:"+acting.name,
            "macro_step": 0 if self.stage is None else self.stage.actions_executed,
            "refreshed": bool(calls), "api_calls_in_act": [c["kind"] for c in calls],
            "api_latency_s": sum(c["latency_s"] for c in calls), "api_calls": calls,
            "api_request_key": getattr(self.client, "last_request_key", None) if calls else None,
            "api_request_log": str(getattr(self.client, "last_log_path", "")) if calls else None,
            "outcomes": [], "outcome": None, "v2_events": self._events.drain(), "pointing": [],
            "relation": {"phase": self._phase(), "do": (self.current_operation or {}).get("kind")},
            "observers": {}, "parts_uv": {}, "holding": self.holding, "finger_gap_mm": view.gap_mm,
            "feedback": self.feedback, "interaction": self.tracker.report,
            "scene_frame_id": self.scene.frame_id, "stage_index": self.stage_index,
            "keypose_index": self.keypose_index, "motion_evidence": self.motion_evidence,
            "goal": None if self.stage is None else self.stage.keypose.to_dict(),
            "update_steps": 1, "calibration_probe": False,
            "history_stats": self.client.history_stats()})
        decision = Decision(action, diagnostics)
        self._pending = (observation, decision, view, acting)
        self.step_index += 1
        return decision

    def observe_transition(self, before, decision, after):
        result = super().observe_transition(before, decision, after)
        view = self._view(after)
        self.gripper_command = "close" if decision.action[6] > 0 else "open"
        frame = frame_from_step_view(view, after.geometry, self.step_index, perf_counter())
        interaction = self.tracker.observe(view, frame, self.gripper_command)
        self.measured_path.append(view.position.tolist())
        result.update(execution_owner="foundation_harness", interaction=interaction,
                      task_success="not_available_to_policy")
        self._log("feedback", measured_state=view.state, joint_position=after.joint_position,
                  interaction=interaction, primitive=result.get("primitive_feedback"))
        return result

    def _phase(self):
        return self.stage.keypose.name if self.stage else "semantic_decision" if self.need_model else "local_planning"

    def planning_snapshot(self):
        points = np.empty((0, 3)) if self.scene is None else self.scene.scene_points
        strength = np.zeros(len(points), bool)
        planner_scene = (self.scene is not None and self.planner is not None
                         and getattr(self.planner, "_scene_key", None) == (id(self.scene), self.scene.frame_id))
        if planner_scene:
            points = self.planner._cloud
            strength = self.planner._strong
        indices = np.linspace(0, len(points)-1, min(1600, len(points)), dtype=int) if len(points) else []
        poses = [] if self.operation_plan is None else self.operation_plan.keyposes
        path = []
        if self.stage is not None:
            path = [pose[:3, 3] for pose in self.stage.poses[self.stage.index:]]
        return jsonable({"step": self.step_index, "stage": self._phase(),
            "object_id": (self.current_operation or {}).get("object"), "scene_points": points[indices],
            "scene_point_strong": strength[indices],
            "scene_display_source": "planner_after_self_mask" if planner_scene else "raw_pending_planner_preparation",
            "keyposes": [{"name": p.name, "position": p.pose[:3, 3], "rotation": p.pose[:3, :3]} for p in poses],
            "selected_path": path, "measured_path": self.measured_path[-500:],
            "status": "awaiting_model" if self.need_model else "executing", "unknown_geometry": True})

    def agent_stats(self):
        result = super().agent_stats()
        result.update(policy_version=POLICY_VERSION, execution_owner="foundation_harness",
                      phase=self._phase(), model_decisions=self.model_decisions,
                      operation_history=jsonable(self.operation_history), feedback=jsonable(self.feedback),
                      config=asdict(self.harness_config), withheld_decisions=self.withheld_decisions,
                      interaction=jsonable(self.tracker.report),
                      timing={"model_s": self.model_seconds, "vision_s": self.vision_seconds,
                              "planning_s": self.planning_seconds})
        result["counts"].update(agent_decisions=self.model_decisions, grasp_attempts=self.grasp_attempts)
        result["counts"]["calls_by_need"]["harness"] = self.model_decisions
        return result
