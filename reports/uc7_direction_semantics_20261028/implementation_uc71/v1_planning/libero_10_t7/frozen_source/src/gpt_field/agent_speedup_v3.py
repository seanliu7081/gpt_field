"""Reviewed model executive with persistent geometry and adaptive local motion."""
from __future__ import annotations

from copy import deepcopy
from uuid import uuid4
from time import monotonic

import numpy as np

from .agent_speedup_v2_user_check_agent import AgentFirstV2Policy
from .agent_speedup_v2_types import jsonable
from .agent_speedup_v3_fragments import (
    AdaptiveFragmentSequencePrimitive, AdaptiveLocalMotionPrimitive, adaptive_fragment_capabilities,
)
from .agent_speedup_v3_geometry import GeometryMemory, _box, obb_separation
from .agent_speedup_v3_local_geometry import LocalGeometryMemory
from .agent_speedup_v3_observation import propose_evidence_views
from .agent_speedup_v3_program import AdaptiveProgramPrimitive
from .agent_speedup_v3_operation import OperationRuntime
from .astranomy_agentic_motion import measured_pose
from .astra import _canonical

POLICY_VERSION = "agent-speedup-v3-11"


class EvidenceAcquisitionFailed(RuntimeError):
    """An explicit diagnostic stop, never a task-completion or transport error."""


def _protected_context(context):
    return {key: deepcopy(context.get(key)) for key in
            ("intent", "target_id", "held_object_id", "allowed_contact_ids")}


def _action_identity(action, context):
    return _canonical({"type": action["type"], "motion": action.get("motion"),
                       "execution": action.get("execution"), "macro": action.get("macro"),
                       "program": action.get("program"),
                       "context": _protected_context(context)})


def _blocker_identity(assessment):
    return _canonical({key: assessment.get(key) for key in
                       ("status", "reason", "missing_evidence_requests", "stale_evidence_ids",
                        "monitoring_unknown_ids", "payload_geometry_unknown",
                        "motion_support_revoked", "tracking_status")})


def _geometric_effect(assessment):
    """Compare actual path evidence, not a newly named duplicate or timestamp."""
    keys = ("status", "reason", "unknown_fraction", "allowed_fraction", "clearance_estimate_m")
    result = {key: assessment.get(key) for key in keys}
    result["witnesses"] = [{key: item.get(key) for key in
        ("body_part", "path_fraction", "world_point_m", "axis", "face", "padded_margin_m", "nominal_margin_m")}
        for item in assessment.get("coverage_witnesses", [])]
    result["prefix_fraction"] = (assessment.get("execution_prefix") or {}).get("fraction")
    def rounded(value):
        if isinstance(value, float):
            return round(value, 8)
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rounded(item) for item in value]
        return value
    return _canonical(rounded(result))


class AgentSpeedUpV3Policy(AgentFirstV2Policy):
    # The independent evaluator uses this established action protocol label.
    # Agent identity lives in policy_version and the reviewed runner manifest.
    controller = AgentFirstV2Policy.controller

    def reset(self):
        self._framework_sync_s = 0.
        self._framework_full_syncs = 0
        self._framework_light_syncs = 0
        self._evidence_view_session = None
        self._evidence_view_attempts = {}
        self._view_proposals = None
        self._view_proposal_key = None
        self._pending_intent = None
        self._active_intent_id = None
        self._intent_action = None
        self._repeat_guard = None
        self._evidence_pause = None
        self._continuation_events = []
        self._continuation_rejections = 0
        self._last_motion_tracking = {}
        self._local_evidence_resumes = set()
        self.shared_geometry = LocalGeometryMemory()
        self.operation_runtime = OperationRuntime(self.shared_geometry)
        self.interaction_monitor = self.operation_runtime.monitor
        self.motion_context = {"intent": "observe", "target_id": None, "held_object_id": None,
                               "allowed_contact_ids": [], "scene_inventory": {
                                   "reviewed": False, "support_ids": [],
                                   "nearby_obstacle_ids": [], "unresolved": True}}
        super().reset()

    def _capabilities(self):
        result = super()._capabilities()
        result["execute"] = adaptive_fragment_capabilities()
        result["program"] = {"contracts": ["operations: 1..3 pick_place with dependencies", "stages: 1..24 semantic goals"],
                             "schema_source": "response_format", "goals": "derived from current geometry, no task-specific distances",
                             "local_phase_transitions": True, "next_operation_gate": "accepted_visual_release",
                             "visual_interaction_evidence": "two-view material motion"}
        result["operation_runtime"] = {"shared_across": ["program", "execute", "motion", "recovery"],
            "verification_object": "persistently_bound_manipulated_object",
            "evidence_required": "current_two_view_motion_and_accepted_geometry_confirmation",
            "local_recovery": "bounded_admissible_candidates_with_cumulative_budget"}
        result["direct_motion"].update(geometry_prefix_checked=False,
                                      exact_command_guard=True, execution_policy="goal_local")
        result["native_macros"]["planning"] = "existing native controllers; observed transitions feed shared geometry"
        return result

    def _native_evidence(self):
        result = super()._native_evidence()
        result["shared_geometry"] = self.shared_geometry.summary(context=self.motion_context)
        result["operation_state"] = self.operation_runtime.snapshot()
        return result

    def _execution_state(self, view=None):
        result = super()._execution_state(view)
        primitive = getattr(self, "_primitive", None)
        if isinstance(primitive, AdaptiveProgramPrimitive):
            progress = primitive.progress()
            if progress.get("operation_contract"):
                progress["completed_stage_count"] = len(progress.pop("completed_stages", []))
                progress["remaining_stage_count"] = len(progress.pop("pending_stages", []))
                progress["remaining_operation_ids"] = [item["id"] for item in progress.pop("pending_operations", [])]
            result.update(active=not primitive.done, progress=progress, kind="semantic_program")
        return result

    def _context(self, view):
        self.shared_geometry.import_native(self.ctx.scene, getattr(self.ctx, "holding", None), view, self.step_index)
        result = super()._context(view)
        result["native_evidence"]["shared_geometry"] = self.shared_geometry.summary(view, self.motion_context)
        if self._pending_intent is not None:
            target = self._pending_intent["saved"]["current_target"]
            if target is not None:
                self._pending_intent["saved"]["whole_goal_assessment"] = self.shared_geometry.preview_motion(
                    view, target["xyz_m"], target["quaternion_wxyz"], self._pending_intent["motion_context"], self.step_index)
        result["native_evidence"]["current_decision"] = self._decision_core()
        if self._pending_intent is not None:
            result["native_evidence"]["current_decision"]["evidence_acquisition"] = self._propose_views(view)
        return jsonable(result)

    def _goal_assessment(self):
        assessment = self.shared_geometry.last_motion or {}
        if self._pending_intent is not None:
            saved = self._pending_intent["saved"]
            assessment = saved.get("whole_goal_assessment") or assessment.get("whole_goal_assessment") or assessment
        return deepcopy(assessment)

    def _decision_core(self):
        assessment = self._goal_assessment()
        missing = deepcopy(assessment.get("missing_evidence_requests") or [])
        requirements = []
        for request in missing:
            if isinstance(request, str):
                parts = request.split(":")
                ident = (parts[2] if parts[0] == "sweep_coverage" and len(parts) > 2 else
                         parts[1] if parts[0] == "obstacle_core" and len(parts) > 1 else request.partition(":")[2])
            else:
                ident = request.get("entity_id")
            entity = self.shared_geometry.entities.get(ident, {})
            requirements.append({"request": deepcopy(request), "entity_id": ident or None,
                "reason": assessment.get("reason"), "required_evidence": deepcopy(request),
                "entity_status": {key: deepcopy(entity.get(key)) for key in (
                    "role", "source", "observation_step", "stale", "stale_reason", "invalidation_kind",
                    "invalidation_origin_ids", "monitoring_unknown", "motion_support_revoked")}})
        pending = None
        if self._pending_intent is not None:
            saved = self._pending_intent["saved"]
            pending = {"intent_id": self._pending_intent["intent_id"], "kind": self._pending_intent["kind"],
                       "motion_context": deepcopy(self._pending_intent["motion_context"]),
                       **{key: deepcopy(saved.get(key)) for key in (
                           "current_target", "pending_segments", "motion", "gripper_command",
                           "remaining_chunk_steps", "remaining_watchdog_steps", "consumed_steps",
                           "program", "stage_index", "phase", "completed_stages", "remaining_stages",
                           "stage_consumed_steps", "missing_evidence_requests")}}
            if saved.get("operation_contract"):
                pending["program"] = deepcopy(saved["operation_contract"])
                pending["completed_stage_count"] = len(pending.pop("completed_stages", []))
                pending["remaining_stage_count"] = len(pending.pop("remaining_stages", []))
                pending["completed_operations"] = deepcopy(saved.get("completed_operations", []))
                completed = {item["id"] for item in pending["completed_operations"]}
                pending["remaining_operation_ids"] = [item["id"] for item in pending["program"]["operations"]
                                                       if item["id"] not in completed]
                stage = saved.get("stage_index", 0)
                mapping = saved.get("stage_operation_ids", [])
                pending["current_operation_id"] = mapping[stage] if stage < len(mapping) else None
        guard = self._repeat_guard or {}
        requested = deepcopy(self._intent_action)
        if pending and (pending.get("program") or {}).get("operations") and requested:
            requested["program"] = {"contract_source": "pending_intent.program"}
        return {"version": "v3-current-decision-2", "step": self.step_index,
                "operation_state": self.operation_runtime.snapshot(),
                "pending_intent": pending, "current_requested_action": requested,
                "motion_context": deepcopy(self.motion_context),
                "current_tracking": deepcopy(self._last_motion_tracking),
                "missing_evidence_requests": missing,
                "requirements": requirements, "requirements_truncated": False,
                "whole_goal_assessment": {key: deepcopy(assessment[key]) for key in (
                    "status", "reason", "unknown_fraction", "allowed_fraction", "clearance_estimate_m",
                    "coverage_witnesses", "missing_evidence_requests", "execution_prefix", "execution_status",
                    "execution_allowed", "blocking_reasons", "advisories", "execution_policy",
                    "stale_evidence_ids", "monitoring_unknown_ids", "tracking_status", "limiting_pair",
                    "sweep_samples", "interval_padding_m", "refinement", "sweep_refinement",
                    "xyz_m", "quaternion_wxyz", "first_uncovered_fraction", "scope",
                    "target_xyz_m", "target_quaternion_wxyz")
                    if key in assessment},
                "executable_prefix_assessment": deepcopy((self.shared_geometry.last_motion or {}).get("execution_prefix")),
                "blocker": {key: deepcopy(assessment.get(key)) for key in (
                    "status", "reason", "execution_allowed", "blocking_reasons", "advisories",
                    "evidence_ids", "stale_evidence_ids", "monitoring_unknown_ids",
                    "payload_geometry_unknown", "final_interaction_guard", "motion_support_revoked",
                    "tracking_status", "limiting_pair", "command_candidates", "selected_command_candidate")},
                "repeated_action_guard": {"new_evidence_required": bool(guard),
                    "consecutive_rejections": guard.get("rejections", 0),
                    "required_route": "acquire_evidence_or_change_intent" if guard else None,
                    "unchanged_retry_limit": 1},
                "evidence_pause": deepcopy(self._evidence_pause), "task_success": "not_available_to_policy"}

    def _preview_target(self):
        if self._pending_intent is not None:
            return self._pending_intent["saved"]["current_target"]
        primitive = getattr(self, "_primitive", None)
        if getattr(primitive, "_target_xyz", None) is not None:
            return {"xyz_m": primitive._target_xyz.tolist(), "quaternion_wxyz": list(primitive._target_wxyz)}
        if getattr(primitive, "original_target", None) is not None:
            return {"xyz_m": list(primitive.original_target[:3]), "quaternion_wxyz": list(primitive.original_target[3:7])}
        return None

    def _patch_changes_path(self, candidate, view, context):
        target = self._preview_target()
        if target is None:
            return False
        shape_keys = ("role", "parent_id", "position_m", "quaternion_wxyz", "boxes", "uncertainty_m", "coverage", "attachment", "anchors")
        def shape(entity):
            return _canonical({key: entity.get(key) for key in shape_keys})
        old_shapes = {shape(entity) for entity in self.shared_geometry.entities.values()}
        bodies, _ = candidate._local_bodies(view, context, set())
        radius = max(float(np.linalg.norm(body["center"])+np.linalg.norm(body["half"])+body["uncertainty"]) for body in bodies)
        start, end = np.asarray(view.position), np.asarray(target["xyz_m"])
        swept_bounds = _box((start+end)/2., np.eye(3), np.abs(end-start)/2.+radius, "pending", "broad_phase")
        changed = []
        for ident in set(self.shared_geometry.entities) | set(candidate.entities):
            old, new = self.shared_geometry.entities.get(ident), candidate.entities.get(ident)
            if old is not None and new is not None and shape(old) == shape(new):
                continue
            if old is None and new is not None and shape(new) in old_shapes:
                continue
            changed.extend((memory, entity) for memory, entity in
                           ((self.shared_geometry, old), (candidate, new)) if entity is not None)
        if not any(obb_separation(swept_bounds, box, padding_b=box["uncertainty"]) <= 0
                   for memory, entity in changed for box in memory._entity_boxes(entity)):
            return False
        before = self.shared_geometry.preview_motion(view, target["xyz_m"], target["quaternion_wxyz"], context, self.step_index)
        after = candidate.preview_motion(view, target["xyz_m"], target["quaternion_wxyz"], context, self.step_index)
        return _geometric_effect(before) != _geometric_effect(after)

    def _propose_views(self, view):
        pending = self._pending_intent
        if pending is None:
            return {"candidates": [], "reason": "no_pending_goal"}
        if pending["saved"].get("current_target") is None:
            return {"candidates": [], "reason": "semantic_goal_requires_geometry_before_view_planning",
                    "missing_evidence_requests": deepcopy(pending["saved"].get("missing_evidence_requests", []))}
        tracking = getattr(self, "_last_motion_tracking", {})
        key = (self.shared_geometry.revision, tuple(view.state), pending["intent_id"],
               tracking.get("status"), tracking.get("sample_count"))
        if key != self._view_proposal_key:
            target = pending["saved"]["current_target"]
            context = {**pending["motion_context"], "commanded_gripper": self.ctx.memory["user_check_gripper_command"]}
            assessment = self.shared_geometry.preview_motion(view, target["xyz_m"], target["quaternion_wxyz"], context, self.step_index)
            if tracking.get("status") in ("mismatch", "lost"):
                # A nominal path preview cannot erase an actual motion failure.
                # Carry it as the reason to observe, not as a permanent veto on
                # a newly chosen viewing motion's independent controller.
                assessment = {**assessment, "tracking_status": tracking["status"],
                    "failed_motion_tracking": deepcopy(tracking),
                    "failed_goal_xyz_m": list(target["xyz_m"]),
                    "blocking_reasons": [*assessment.get("blocking_reasons", []), "measured_tracking_mismatch"]}
            self._view_proposals = propose_evidence_views(self.shared_geometry, view, context, assessment,
                self.step_index, self._evidence_view_attempts.get(pending["intent_id"], []))
            self._view_proposal_key = key
        return deepcopy(self._view_proposals)

    def _acquire_evidence(self, answer, view):
        pending = self._pending_intent
        continuation = answer["continuation"]
        if pending is None or continuation["intent_id"] != pending["intent_id"]:
            return self._reject_continuation("fresh_matching_pending_intent_required")
        if _protected_context(answer["motion_context"]) != _protected_context(pending["motion_context"]):
            return self._reject_continuation("evidence_view_cannot_change_saved_intent_or_contacts")
        if any(answer["scene_update"].values()):
            return self._reject_continuation("patch_geometry_before_requesting_evidence_view")
        proposals = self._propose_views(view)
        attempts = self._evidence_view_attempts.setdefault(pending["intent_id"], [])
        if not proposals["candidates"] or len(attempts) >= 2:
            self._stop_for_evidence("targeted_view_unavailable: " + (
                "finite_view_attempt_budget_exhausted" if len(attempts) >= 2 else proposals["reason"]))
        selected = proposals["candidates"][0]
        self._evidence_view_session = {"pending": deepcopy(pending), "action": deepcopy(self._intent_action),
            "guard": deepcopy(self._repeat_guard), "selected": deepcopy(selected)}
        attempts.append(selected["xyz_m"])
        self._view_proposal_key = None
        self.motion_context = {**deepcopy(pending["motion_context"]), "intent": "observe", "allowed_contact_ids": []}
        self._pending_intent = None
        self._mode = "fragment"
        self._fragment_grants += 1
        self._segment_start = {"step": self.step_index, "pose_wxyz": measured_pose(view.state),
                               "finger_gap_mm": view.gap_mm, "action_type": "acquire_evidence"}
        self._primitive = AdaptiveFragmentSequencePrimitive(view, selected["execution"], self.shared_geometry,
            self.motion_context, step=self.step_index, keep_gripper=self.ctx.memory["user_check_gripper_command"],
            operation_runtime=self.operation_runtime)
        if self._primitive.done:
            self._complete_direct()

    def _relevant_evidence(self, view, context, update=None):
        # Revisions/timestamps, copied reviewed flags and unrelated entity edits
        # cannot make an unchanged blocked attempt eligible again.
        inventory = context.get("scene_inventory") or {}
        ids = set((self.shared_geometry.last_motion or {}).get("evidence_ids") or [])
        ids.update(ident for ident in (context.get("target_id"), context.get("held_object_id")) if ident)
        ids.update(inventory.get("support_ids", []))
        ids.update(inventory.get("nearby_obstacle_ids", []))
        for item in (self.shared_geometry.last_motion or {}).get("missing_evidence_requests") or []:
            if isinstance(item, str) and ":" in item:
                parts = item.split(":")
                ids.add(parts[2] if parts[0] == "sweep_coverage" and len(parts) > 2 else item.partition(":")[2])
        entities = dict(self.shared_geometry.entities)
        update = update or {}
        for entity in update.get("entities", []):
            entities[entity["id"]] = {**entities.get(entity["id"], {}), **entity}
        for ident in update.get("remove_ids", []):
            entities.pop(ident, None)
        shape_keys = ("role", "parent_id", "position_m", "quaternion_wxyz", "boxes", "uncertainty_m",
                      "coverage", "attachment", "anchors")
        relevant = {ident: {key: deepcopy(entities[ident].get(key)) for key in shape_keys}
                    if ident in entities else None for ident in sorted(ids)}
        # A changed calibrated viewpoint of a relevant shape can provide new
        # evidence even if its model OBB remains unchanged. Pixel quantization
        # ignores tiny settled robot jitter; this never verifies the shape.
        viewpoints = {}
        geometry = getattr(getattr(view, "observation", None), "geometry", None) or {}
        for camera, cam in geometry.get("cameras", {}).items():
            projection = {}
            try:
                rotation = np.asarray(cam["rotation_world_from_camera"], float)
                origin = np.asarray(cam["position_world_m"], float)
                fx, fy = cam["focal_length_px"]
                cx, cy = cam["principal_point_px"]
                for ident in sorted(ids):
                    if ident not in entities:
                        continue
                    point = (np.asarray(entities[ident]["position_m"])-origin) @ rotation
                    depth = -point[2]
                    if depth > 1e-6:
                        uv = [cx+fx*point[0]/depth, cy-fy*point[1]/depth]
                        projection[ident] = [int(round(pixel/2)) for pixel in uv]
            except (KeyError, TypeError, ValueError, IndexError):
                projection = {}
            viewpoints[camera] = projection
        return _canonical({"entities": relevant, "viewpoints": viewpoints,
                           "support_ids": sorted(inventory.get("support_ids", [])),
                           "nearby_obstacle_ids": sorted(inventory.get("nearby_obstacle_ids", []))})

    def _stop_for_evidence(self, reason):
        assessment = self._goal_assessment()
        self._evidence_pause = {"reason": reason, "step": self.step_index,
            "missing_evidence_requests": deepcopy(assessment.get("missing_evidence_requests", [])),
            "blocker": assessment, "automatic_retry": False,
            "task_success": False}
        raise EvidenceAcquisitionFailed(_canonical(self._evidence_pause))

    def _reject_repeated(self, action_key, view, context, update, candidate=None):
        guard = self._repeat_guard
        if (not guard or action_key != guard["action"]
                or _blocker_identity(self.shared_geometry.last_motion or {}) != guard["blocker"]
                or self._relevant_evidence(view, context, update) != guard["evidence"]):
            return False
        if candidate is not None and self._patch_changes_path(candidate, view, context):
            return False
        guard["rejections"] += 1
        if guard["rejections"] > 1:
            self._stop_for_evidence("repeated_blocked_action_without_new_relevant_evidence")
        self._record_feedback({"action": "rejected_repeat", "status": "evidence_required",
            "reason": "unchanged_blocker_requires_new_evidence_or_intent",
            "missing_evidence_requests": deepcopy((self.shared_geometry.last_motion or {}).get("missing_evidence_requests", [])),
            "required_route": "acquire_evidence_or_change_intent",
            "continuation": "Provide relevant observed geometry or a supported new viewing action; otherwise escalate. The original move was not executed."})
        self._need_agent = True
        return True

    def _resume_with_local_evidence(self, primitive, saved, kind):
        """Continue an unchanged, fully supported goal on new local evidence.

        Only accepted perception/interaction events can unlock this path. A
        renamed model box, observation timestamp or repeated RGB status cannot
        renew an unknown allowance. Original contact rules and watchdog remain.
        """
        if kind == "program" or primitive.outcome().get("reason") not in (
                "geometry_predicted_block", "geometry_observation_horizon", "geometry_prefix_observed"):
            return False
        target = saved.get("current_target")
        if target is None or saved.get("remaining_watchdog_steps", 0) <= 0:
            return False
        current = self._view(self._expected_observation) if self._expected_observation is not None else self.ctx.view
        relevant = set((self.shared_geometry.last_motion or {}).get("evidence_ids", []))
        relevant.update(value for value in (self.motion_context.get("target_id"),
                                           self.motion_context.get("held_object_id")) if value)
        sources = []
        for ident in relevant:
            entity = self.shared_geometry.entities.get(ident, {})
            evidence = entity.get("local_pose_evidence") or {}
            if evidence.get("accepted") is True and evidence.get("observed_at_step") == self.step_index:
                sources.append(("local_pose", ident, entity.get("local_pose_revision")))
        for event in getattr(self.operation_runtime, "events", [])[-4:]:
            if (event.get("kind") == "interaction_supported" and event.get("step") == self.step_index
                    and event.get("object_id") in relevant):
                sources.append(("interaction", event.get("operation_id"), event.get("interaction"), self.step_index))
        if not sources:
            return False
        token = _canonical([self._active_intent_id, sources])
        if token in self._local_evidence_resumes:
            return False
        assessment = self.shared_geometry.preview_motion(current, target["xyz_m"], target["quaternion_wxyz"],
                                                         self.motion_context, self.step_index)
        if not assessment.get("execution_allowed", assessment.get("status") == "supported"):
            return False
        self._local_evidence_resumes.add(token)
        cls = AdaptiveFragmentSequencePrimitive if kind == "execute" else AdaptiveLocalMotionPrimitive
        resumed = cls.from_continuation(current, deepcopy(saved), self.shared_geometry, self.motion_context,
            step=self.step_index, keep_gripper=self.ctx.memory["user_check_gripper_command"],
            operation_runtime=self.operation_runtime)
        if resumed.done:
            return False
        self._primitive, self._pending_intent, self._need_agent = resumed, None, False
        event = {"intent_id": self._active_intent_id, "step": self.step_index,
                 "action": "local_evidence_resume", "sources": sources,
                 "current_guard_rechecked": True, "execution_allowed": True,
                 "whole_goal_supported": assessment.get("status") == "supported",
                 "original_target_preserved": True, "remaining_watchdog_steps": saved["remaining_watchdog_steps"]}
        self._continuation_events.append(event)
        self._record_feedback(event)
        return True

    def _complete_direct(self):
        if getattr(self, "_evidence_view_session", None) is not None:
            session = self._evidence_view_session
            self._evidence_view_session = None
            outcome = self._primitive.outcome()
            super()._complete_direct()
            self._pending_intent = session["pending"]
            self._intent_action = session["action"]
            self._active_intent_id = session["pending"]["intent_id"]
            self._repeat_guard = session["guard"]
            self.motion_context = deepcopy(session["pending"]["motion_context"])
            self._view_proposal_key = None
            self._record_feedback({"action": "acquire_evidence", "status": "fresh_view_requires_interpretation",
                "motion_outcome": outcome, "viewpoint": session["selected"],
                "original_goal_resumed": False, "preserved_intent_id": self._active_intent_id,
                "continuation": "Inspect the new synchronized images, patch only supported boundary evidence, then explicitly resume the saved goal or escalate."})
            return
        primitive = self._primitive
        if isinstance(primitive, AdaptiveProgramPrimitive):
            self.motion_context = deepcopy(primitive.motion_context)
        tracker = getattr(primitive, "_tracking", None)
        self._last_motion_tracking = tracker.summary() if tracker is not None else {}
        saved = getattr(primitive, "export_continuation", lambda: None)()
        if saved is not None:
            kind = ("program" if isinstance(primitive, AdaptiveProgramPrimitive) else
                    "execute" if isinstance(primitive, AdaptiveFragmentSequencePrimitive) else "motion")
            if self._resume_with_local_evidence(primitive, saved, kind):
                return
            self._pending_intent = {"intent_id": self._active_intent_id or uuid4().hex,
                "kind": kind, "saved": deepcopy(saved), "motion_context": deepcopy(self.motion_context)}
        else:
            self._pending_intent = None
        assessment = self.shared_geometry.last_motion or {}
        reason = primitive.outcome().get("reason", "")
        if reason.startswith(("geometry_", "program_geometry_")) and self._intent_action is not None:
            action = _action_identity(self._intent_action, self.motion_context)
            latest = self._view(self._expected_observation) if self._expected_observation is not None else self.ctx.view
            evidence = self._relevant_evidence(latest, self.motion_context)
            blocker = _blocker_identity(assessment)
            previous = self._repeat_guard
            same = previous and all(previous[key] == value for key, value in
                                    (("action", action), ("blocker", blocker), ("evidence", evidence)))
            self._repeat_guard = {"action": action, "evidence": evidence, "blocker": blocker,
                                  "rejections": previous["rejections"] if same else 0}
        elif saved is None:
            self._repeat_guard = None
        super()._complete_direct()

    def _reject_continuation(self, reason, detail=None):
        self._continuation_rejections += 1
        if self._continuation_rejections > 1:
            self._stop_for_evidence("repeated_invalid_continuation: " + reason)
        self._record_feedback({"action": "continuation", "status": "rejected", "reason": reason,
                               "detail": detail, "executed_steps": 0,
                               "required_route": "acquire_evidence_or_change_intent"})
        self._need_agent = True

    def _continue_intent(self, answer, view):
        pending, continuation = self._pending_intent, answer["continuation"]
        if pending is None or continuation["intent_id"] != pending["intent_id"]:
            return self._reject_continuation("fresh_matching_pending_intent_required")
        if _protected_context(answer["motion_context"]) != _protected_context(pending["motion_context"]):
            return self._reject_continuation("continuation_cannot_change_intent_target_held_or_contacts")
        action_key = _action_identity(self._intent_action, pending["motion_context"])
        candidate = self.shared_geometry.preview_scene_update(answer["scene_update"], view, self.step_index)
        if self._reject_repeated(action_key, view, answer["motion_context"], answer["scene_update"], candidate):
            return
        # Preview uses a frozen visual monitor. Commit the validated hypotheses
        # through the live memory so its real RGB references remain active.
        self.shared_geometry.update_from_model(answer["scene_update"], view, self.step_index)
        self.motion_context = deepcopy(answer["motion_context"])
        self._view_proposal_key = None
        if not continuation["resume"]:
            self._record_feedback({"action": "geometry_patch", "status": "patched_without_motion",
                                   "reason": "explicit_resume_false", "executed_steps": 0})
            self._need_agent = True
            return
        cls = {"program": AdaptiveProgramPrimitive, "execute": AdaptiveFragmentSequencePrimitive,
               "motion": AdaptiveLocalMotionPrimitive}[pending["kind"]]
        try:
            options = {"operation_runtime": self.operation_runtime}
            if pending["kind"] == "program":
                options["interaction_monitor"] = self.interaction_monitor
            primitive = cls.from_continuation(view, deepcopy(pending["saved"]), self.shared_geometry,
                answer["motion_context"], step=self.step_index,
                keep_gripper=self.ctx.memory["user_check_gripper_command"], **options)
        except (ValueError, TypeError) as exc:
            return self._reject_continuation("continuation_not_authorized", str(exc))
        self._continuation_rejections = 0
        self._mode = "fragment" if pending["kind"] in ("program", "execute") else "direct"
        self._primitive = primitive
        self._active_intent_id = pending["intent_id"]
        self._pending_intent = None
        self._continuation_events.append({"intent_id": self._active_intent_id,
            "step": self.step_index, "action": answer["type"], "geometry_revision": self.shared_geometry.revision,
            "semantic_context_preserved": True, "current_guard_rechecked": True})
        if primitive.done:
            self._complete_direct()

    def _sync_framework(self, view=None):
        began = monotonic()
        if (not getattr(self, "_need_agent", True) and getattr(self.plan_lock, "_world_state", None)):
            # The immutable reviewed task contract and full executive snapshot
            # are rebuilt before model calls. Local steps need only current
            # measured state, not repeated deep copies of the entire history.
            state = self.plan_lock._world_state
            state.update(version=POLICY_VERSION, step=self.step_index,
                         local_snapshot=True, geometry_revision=self.shared_geometry.revision)
            if view is not None:
                state["measured_robot"] = {"pose_wxyz": measured_pose(view.state),
                                          "finger_gap_mm": view.gap_mm, "observed_at_step": self.step_index}
            state["last_gripper_command"] = self.ctx.memory["user_check_gripper_command"]
            state["active_motion_context"] = deepcopy(self.motion_context)
            self.plan_lock._last_feedback = self._feedback
            self._framework_light_syncs += 1
            self._framework_sync_s += monotonic()-began
            return
        super()._sync_framework(view)
        self.plan_lock._world_state.update(version=POLICY_VERSION,
            shared_geometry=self.plan_lock._world_state.get("native_estimate", {}).get("shared_geometry"),
            local_snapshot=False)
        self._framework_full_syncs += 1
        self._framework_sync_s += monotonic()-began

    def _apply_agent(self, answer, view):
        if answer["type"] == "acquire_evidence":
            return self._acquire_evidence(answer, view)
        if answer["type"] == "escalate":
            self._stop_for_evidence("model_escalation: " + answer["why"])
        if answer["type"] in ("geometry_patch", "continue_intent"):
            return self._continue_intent(answer, view)
        candidate = self.shared_geometry.preview_scene_update(answer["scene_update"], view, self.step_index)
        if answer["type"] in ("observe", "finish") and self._repeat_guard and self._intent_action is not None:
            pending_context = (self._pending_intent or {}).get("motion_context", self.motion_context)
            original = _action_identity(self._intent_action, pending_context)
            if self._reject_repeated(original, view, pending_context, answer["scene_update"], candidate):
                return
        identity = _action_identity(answer, answer["motion_context"])
        if self._reject_repeated(identity, view, answer["motion_context"], answer["scene_update"], candidate):
            return
        self._continuation_rejections = 0
        self._intent_action = deepcopy({k: answer.get(k) for k in ("type", "motion", "execution", "macro", "program")})
        self._active_intent_id = uuid4().hex
        self._pending_intent = None
        self._repeat_guard = None
        self.shared_geometry.update_from_model(answer["scene_update"], view, self.step_index)
        self._view_proposal_key = None
        self.motion_context = deepcopy(answer["motion_context"])
        action = {k: v for k, v in answer.items() if k not in ("scene_update", "motion_context", "continuation", "program")}
        kind = action["type"]
        if kind == "macro":
            return super()._apply_agent(action, view)
        self._abandon_macro("model took direct control")
        self._segment_start = {"step": self.step_index, "pose_wxyz": measured_pose(view.state),
                               "finger_gap_mm": view.gap_mm, "action_type": kind,
                               "geometry_revision": self.shared_geometry.revision}
        options = {"geometry": self.shared_geometry, "motion_context": self.motion_context,
                   "step": self.step_index, "keep_gripper": self.ctx.memory["user_check_gripper_command"],
                   "operation_runtime": self.operation_runtime}
        if kind == "program":
            self._mode = "fragment"
            self._fragment_grants += 1
            self._primitive = AdaptiveProgramPrimitive(view, answer["program"], self.shared_geometry,
                step=self.step_index, keep_gripper=self.ctx.memory["user_check_gripper_command"],
                interaction_monitor=self.interaction_monitor, operation_runtime=self.operation_runtime)
            self.motion_context = deepcopy(self._primitive.motion_context)
        elif kind == "execute":
            self._mode = "fragment"
            self._fragment_grants += 1
            self._primitive = AdaptiveFragmentSequencePrimitive(view, action["execution"], **options)
        else:
            self._mode = "direct"
            if kind == "motion":
                motion = action["motion"]
            elif kind in ("observe", "finish"):
                motion = {"frame": "world", "position_mode": "relative", "xyz_m": [0., 0., 0.],
                          "quaternion_wxyz": None, "gripper": "keep", "steps": 1}
                options["motion_context"] = {**self.motion_context, "intent": "observe"}
                if kind == "finish":
                    self._model_completion_claims.append({"step": self.step_index, "claim": action["belief"],
                        "source": "model_estimate", "evaluator_success": "not_observed"})
            else:
                raise ValueError(f"Unknown v3 action type {kind!r}")
            self._primitive = AdaptiveLocalMotionPrimitive(view, motion, **options)
        if self._primitive.done:
            self._complete_direct()

    def act(self, observation):
        decision = super().act(observation)
        command = ("close" if decision.action[6] > 0 else "open" if decision.action[6] < 0
                   else self.ctx.memory["user_check_gripper_command"])
        # Local primitives prepare their baselines before evaluating an action.
        # This idempotent hook also covers native macros and fallback holds.
        prepared = self.operation_runtime.before_action(self._view(observation), self.motion_context,
                                                       command, self.step_index-1)
        if (prepared or {}).get("reason") == "active_manipulated_object_binding_conflict":
            self._stop_for_evidence("active_manipulated_object_binding_conflict: " + _canonical(prepared))
        return decision

    def observe_transition(self, before, decision, after):
        # Let the established pending-action protocol validate and acknowledge
        # first. Adaptive primitives update geometry before local reassessment;
        # this idempotent call also covers native/observation-only transitions.
        # FK never refreshes a model observation timestamp.
        result = super().observe_transition(before, decision, after)
        if isinstance(self._primitive, AdaptiveProgramPrimitive):
            self.motion_context = deepcopy(self._primitive.motion_context)
        self.shared_geometry.observe_transition(self._view(before), self._view(after), self.step_index,
                                                self.ctx.memory["user_check_gripper_command"])
        self.operation_runtime.observe_transition(self._view(before), self._view(after), self.motion_context,
                                                  self.ctx.memory["user_check_gripper_command"], self.step_index)
        self._sync_framework(self._view(after))
        result["shared_geometry_revision"] = self.shared_geometry.revision
        result["policy_version"] = POLICY_VERSION
        return result

    def fast_stats(self):
        """Small current UI snapshot; full history stays in terminal stats/audit."""
        primitive = getattr(self, "_primitive", None)
        return {"policy_version": POLICY_VERSION, "step": self.step_index, "phase": self._phase(),
                "motion_context": deepcopy(self.motion_context), "geometry_revision": self.shared_geometry.revision,
                "program_stage": getattr(primitive, "stage_index", None),
                "primitive": getattr(primitive, "name", None), "needs_model": bool(self._need_agent),
                "counts": {"agent_decisions": self._executive_calls, "fragment_actions": self._fragment_actions,
                           "direct_actions": self._direct_actions}, "task_success": "not_available_to_policy"}

    def performance_stats(self):
        geometry_perf = getattr(self.shared_geometry, "performance_stats", lambda: {})()
        return {"framework_sync_s": self._framework_sync_s, "framework_full_syncs": self._framework_full_syncs,
                "framework_light_syncs": self._framework_light_syncs,
                **self.operation_runtime.performance_stats(), **geometry_perf,
                "timing_scope": "subsets of act/observe; do not add to evaluator totals"}

    def agent_stats(self):
        result = super().agent_stats()
        result.update(policy_version=POLICY_VERSION, shared_geometry=self.shared_geometry.snapshot())
        result["current_decision"] = self._decision_core()
        result["continuation_events"] = deepcopy(self._continuation_events)
        result["evidence_pause"] = deepcopy(self._evidence_pause)
        result["evidence_view_attempts"] = deepcopy(self._evidence_view_attempts)
        result["operation_state"] = self.operation_runtime.snapshot()
        result["operation_events"] = deepcopy(self.operation_runtime.events)
        result["versions"]["agent_owned"] = POLICY_VERSION
        result["versions"]["shared_geometry"] = self.shared_geometry.snapshot()["version"]
        return result
