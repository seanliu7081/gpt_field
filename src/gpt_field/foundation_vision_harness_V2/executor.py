"""A committed queue over measured v2 fragments, with one action authority.

The model proposes poses; vision can update anchors but cannot replace missing
contact evidence with a whole-object centre. Every fragment observes the exact
action that was dispatched after the command filter. Evaluator state is absent.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

import numpy as np

from ..agent_speedup_v2_user_check_fragments import FragmentSequencePrimitive, fragment_capabilities
from ..astra import SchemaError
from ..contact_skills import HoldingActionAdapter, StepView
from ..foundation_harness.feedback import grip_pose
from ..foundation_harness.types import jsonable
from ..foundation_vision.bridge import frame_from_step_view
from ..foundation_vision_harness_V1.feedback import PayloadState
from ..policy import Decision
from . import VERSION, REVISION


@dataclass(frozen=True)
class HarnessConfig:
    perception_wait_s: float = 15.
    anchor_update_limit_m: float = .03
    empty_gap_mm: float = 3.
    # Bounds apply to a recovery/fragment, never to the full episode.


class CandidateUnavailable(ValueError):
    def __init__(self, reason, evidence=None):
        super().__init__(reason)
        self.evidence = evidence or {}


def _signature(value):
    return sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _segment_summary(segment):
    if segment is None:
        return None
    return {**{key: segment.get(key) for key in ("id", "object", "anchor", "contact_mode", "checkpoint", "checkpoint_object")},
            "primary": {key: segment["candidates"][0][key] for key in ("skill", "gripper")},
            "candidate_count": len(segment["candidates"])}


def _needs_source_binding(segment):
    return bool(segment.get("anchor") or segment.get("grounding") is not None
                or segment.get("contact_mode", "free") != "free")


def _failure_keys(reason, evidence, view, segment=None, registry=None):
    """Identify blocked material and local robot state, not proposed waypoints.

    Source frame IDs, proposed offsets, overlap magnitudes and target centroids
    change without resolving a blocker. Keep them out of recovery identity.
    Distinct blockers remain separate so adding another rejected alternative
    cannot hide a previously exhausted one inside a different set signature.
    """
    state = np.asarray(view.state, float)
    local = {"position": state[:3].copy(), "rotation": grip_pose(state)[:3, :3],
             "fingers": state[6:8].copy()}
    joints = getattr(view.observation, "joint_position", None)
    if joints is not None:
        local["joints"] = np.asarray(joints, float).copy()
    blockers, leaf_reasons = [], set()

    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get("reason"), str):
                leaf_reasons.add(value["reason"])
            if value.get("part") is not None:
                point = value.get("point_world_m")
                owner = value.get("point_object_id", value.get("owner", ""))
                location = list(point) if point is not None else None
                blockers.append({"part": value["part"], "owner": str(owner), "location": location,
                                 "reason": value.get("reason", reason)})
            for key, item in value.get("conflicts", {}).items():
                if key == "joint_limit" or key.startswith(("self:", "payload_robot:")):
                    blockers.append({"constraint": key})
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(evidence)
    if not blockers:
        reference = (segment or {}).get("grounding") or {}
        pixel = reference.get("pixel")
        blockers = [{"reason": reason, "causes": sorted(leaf_reasons),
                     "object": (segment or {}).get("object"),
                     "contact_region": reference.get("region"), "view": reference.get("view"),
                     "pixel_patch": pixel}]

    def same(first, second):
        a, b = first["local_state"], second["local_state"]
        angle = np.arccos(np.clip((np.trace(a["rotation"].T @ b["rotation"]) - 1) / 2, -1., 1.))
        if (np.linalg.norm(a["position"] - b["position"]) > .01 or angle > .1
                or np.max(np.abs(a["fingers"] - b["fingers"])) > .01):
            return False
        if ("joints" in a) != ("joints" in b):
            return False
        if "joints" in a and np.max(np.abs(a["joints"] - b["joints"])) > .05:
            return False
        a, b = dict(first["blocker"]), dict(second["blocker"])
        for field, tolerance in (("location", .02), ("pixel_patch", 8.)):
            left, right = a.pop(field, None), b.pop(field, None)
            if isinstance(left, (list, tuple, np.ndarray)) and isinstance(right, (list, tuple, np.ndarray)):
                if np.linalg.norm(np.asarray(left) - np.asarray(right)) > tolerance:
                    return False
            elif left != right:
                return False
        return a == b

    keys = set()
    for blocker in blockers:
        record = {"blocker": blocker, "local_state": local}
        prior = next((old for old in registry or () if same(old, record)), None)
        if prior is None:
            record["key"] = _signature(record)
            if registry is not None:
                registry.append(record)
            prior = record
        keys.add(prior["key"])
    return keys


class MeasuredFragment(FragmentSequencePrimitive):
    """Old measured mechanics; the new filter owns environmental constraints.

    The empty legacy map provides only the existing straight-path tracker. It
    is explicitly not evidence of free space. No model-written boxes are used.
    """

    def __init__(self, view, candidate, gripper):
        segment = {**deepcopy(candidate), "observation_after": False, "route": None}
        super().__init__(view, {"segments": [segment]}, keep_gripper=gripper,
            obstacle_map={"avoid_boxes": [], "clearance_m": 0., "revision": 0,
                          "updated_at_step": None, "source": "nominal_tracker_only"})
        self.adapter = HoldingActionAdapter(max_rotation_rad=.1)
        self.adapter._gripper = 1. if gripper == "close" else -1.

    def commit_action(self, actual):
        if self._pending is None:
            raise RuntimeError("no fragment proposal to commit")
        action = tuple(float(x) for x in actual)
        self._pending = (self._pending[0], action)
        self.gripper_command = "close" if action[6] > 0 else "open"
        self.adapter._gripper = action[6]
        return action

    def cancel_proposal(self, actual_gripper):
        self._pending = None
        self.gripper_command = actual_gripper
        self.adapter._gripper = 1. if actual_gripper == "close" else -1.


class FoundationVisionHarnessPolicy:
    def __init__(self, client, *, vision_engine=None, output_dir=None, kinematics=None,
                 config=None, scene_service=None, command_filter=None):
        self.client = client
        self.output_dir = None if output_dir is None else Path(output_dir)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or HarnessConfig()
        if scene_service is None:
            from .scene import SceneService
            scene_service = SceneService(vision_engine, kinematics=kinematics,
                output_dir=None if self.output_dir is None else self.output_dir / "vision")
        if command_filter is None:
            from .motion import CommandFilter
            command_filter = CommandFilter(kinematics=kinematics)
        self.service, self.command_filter = scene_service, command_filter
        self.adapter = HoldingActionAdapter(max_rotation_rad=.1)
        self.reset()

    def reset(self):
        self.step_index = 0
        self.targets, self.scene, self.queue = {}, None, deque()
        self.active = self.primitive = None
        self.candidate_index = 0
        self.entry_state = self.entry_anchor = None
        self.source_anchors = {}
        self.source_scene = None
        self.gripper_command = "open"
        self.payload = PayloadState()
        self.terminal = None
        self.need_model = True
        self.binding_token = None
        self.binding_for_observe = False
        self.checkpoint_pending = None
        self.feedback = {"reason": "initial_observation"}
        self.history, self.completion_conditions = [], []
        self.recoveries, self.failed_segments = set(), set()
        self.failure_evidence = []
        self.last_observe_key = None
        self.model_decisions = self.grasp_attempts = 0
        self.model_seconds = self.vision_wait_seconds = self.filter_seconds = 0.
        self.model_requests, self.protocol_failures = [], set()
        self.grounding_events = []
        self._act_filter_seconds, self._act_filter_calls = 0., 0
        self.started_at = monotonic()
        self.last_view, self.pending_kind, self.last_filter = None, None, {}
        self.path = []

    def close(self):
        self.service.close()

    def _log(self, event, **fields):
        if self.output_dir is not None:
            with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(jsonable({"event": event, "step": self.step_index,
                    "at_s": monotonic(), **fields}), ensure_ascii=False, allow_nan=False)+"\n")

    def _terminal(self, kind, reason):
        self.terminal = {"kind": kind, "reason": reason, "step": self.step_index,
                         "evidence": jsonable(self.feedback), "wall_s": monotonic()-self.started_at}
        self._log("terminal", **self.terminal)

    def _wait(self, token):
        started = monotonic()
        snapshot = self.service.wait(token, timeout_s=self.config.perception_wait_s)
        self.vision_wait_seconds += monotonic()-started
        if snapshot is None:
            errors = getattr(self.service, "errors", [])
            raise RuntimeError("perception_unavailable: "+str(errors[-1] if errors else "publication timeout"))
        self.scene = snapshot
        return snapshot

    def _refresh(self, view, *, force=False, wait=False):
        # A pending source ROI must not be coalesced away by a newer frame.
        # Free-space execution may continue against the current scene while it
        # completes; only an actual dependency or requested refresh waits.
        if self.binding_token is not None:
            self._consume_binding(wait=wait)
            if self.binding_token is not None:
                snapshot = self.service.latest()
                if snapshot is not None:
                    self.scene = snapshot
                return self.scene
            if wait and not force:
                return self.scene
        token = self.service.submit(view, list(self.targets.values()), force=force, frame_id=self.step_index)
        if wait:
            return self._wait(token)
        snapshot = self.service.latest()
        if snapshot is not None:
            self.scene = snapshot
        return self.scene

    def _consume_binding(self, *, wait=False):
        if self.binding_token is None:
            return
        snapshot = self.service.latest()
        ready = snapshot is not None and snapshot.evidence.get("request_token", -1) >= self.binding_token
        if not ready and not wait:
            return
        if not ready:
            snapshot = self._wait(self.binding_token)
        self.scene = snapshot
        self.source_scene = snapshot
        self.source_anchors = self._anchors()
        self._log("source_binding", source_frame_id=snapshot.frame_id, waited=not ready)
        self.binding_token = None
        self.binding_for_observe = False

    def _fresh_object(self, object_id):
        if self.scene is None:
            return None
        obj = self.scene.objects.get(object_id)
        if (obj is None or obj.status in {"missing", "unobserved", "stale", "ambiguous", "conflict"}
                or obj.evidence.get("observed_frame_id", self.scene.frame_id) != self.scene.frame_id
                or not len(obj.points)):
            return None
        return obj

    def _anchors(self):
        if self.scene is None:
            return {}
        return {key: np.median(obj.points, axis=0) for key in self.targets
                if (obj := self._fresh_object(key)) is not None}

    def _scene_context(self):
        if self.scene is None:
            return {"status": "pending"}
        objects = []
        for key, obj in self.scene.objects.items():
            points = np.asarray(obj.points).reshape(-1, 3)
            objects.append({"id": key, "query": obj.query, "status": obj.status,
                "role": obj.evidence.get("role"), "parent_id": obj.evidence.get("parent_id"),
                "material_region_views": sorted(obj.masks),
                "bounds_m": None if not len(points) else [points.min(0), points.max(0)],
                "observed_frame_id": obj.evidence.get("observed_frame_id"),
                "geometry_age_s": obj.evidence.get("geometry_age_s"),
                "position_error_m": obj.evidence.get("position_error_m", obj.evidence.get("sigma_m")),
                "visible_point_count": len(points)})
        evidence = self.scene.evidence
        return jsonable({"frame_id": self.scene.frame_id, "revision": evidence.get("revision"),
            "age_s": max(0., monotonic()-self.scene.captured_at_s), "objects": objects,
            "external_points": len(self.scene.scene_points),
            "unresolved_points": len(evidence.get("unresolved_points", [])),
            "disputed_points": len(evidence.get("disputed_points", [])),
            "self_points": len(evidence.get("self_points", [])),
            "position_error_is_not_calibrated_probability": True})

    def _context(self, view):
        capabilities = fragment_capabilities()
        return jsonable({"task_prompt": view.observation.prompt, "step": self.step_index,
            "frame_id": self.step_index, "scene": self._scene_context(),
            "current_segment": self.active, "queue_remaining": [_segment_summary(s) for s in self.queue],
            "segment_history": [{**h, "segment": _segment_summary(h.get("segment"))} for h in self.history[-6:]],
            "feedback": self.feedback,
            "held": self.payload.summary(), "gripper_command": self.gripper_command,
            "held_review": {"held_object_id": None if self.payload.held is None else self.payload.held["object_id"],
                            "released_object_id": None if self.payload.released is None else self.payload.released["object_id"]},
            "completion_conditions": self.completion_conditions,
            "capabilities": {"skills": capabilities["skills"], "profiles": capabilities["profiles"],
                "minimum_gripper_steps": capabilities["minimum_gripper_steps"],
                "batch_execution": "local measured continuation; a checkpoint is not automatically a model call",
                "candidate_authority": "model proposes; geometry may correct dated anchors; actual command checked locally",
                "contact_grounding": "explicit image material point plus tool offset/standoff is lifted before execution; xyz can be omitted for grounded absolute poses",
                "contact_evidence": "gap and visual comotion; no physical contact sensor",
                "termination": "autonomous finish or unrecoverable blocked"},
            "timing": {"wall_s": monotonic()-self.started_at, "model_s": self.model_seconds}})

    def _ask(self, view, calls):
        started = monotonic()
        self.model_decisions += 1
        answer, failure = None, None
        try:
            answer = self.client.request_harness(view, self._context(view))
        except SchemaError as error:
            failure = type(error).__name__
            self._protocol_repair(error, view)
        except Exception as error:
            failure = type(error).__name__
            raise
        finally:
            # Failed response validation still consumed a complete request.
            elapsed = monotonic()-started
            self.model_seconds += elapsed
            timing = deepcopy(getattr(self.client, "last_harness_timing", {}))
            record = {"kind": "harness", "ordinal": self.model_decisions,
                      "latency_s": elapsed, "ok": answer is not None,
                      "error_type": failure, "request_timing": timing}
            calls.append(record)
            self.model_requests.append(record)
            self._log("model_request", **record)
        if answer is None:
            return
        self._log("semantic_plan", answer=answer, latency_s=elapsed)
        status = answer["status"]
        if status in {"finish", "blocked"}:
            self.feedback = {"reason": answer["why"], "payload": self.payload.summary()}
            if status == "finish" and (self.gripper_command != "open" or self.payload.held is not None):
                self._terminal("blocked", "finish_without_release")
            else:
                self._terminal(status, "policy_completion_claim" if status == "finish" else "semantic_blocked")
            return
        for target in answer["targets"]:
            self.targets[target["id"]] = deepcopy(target)
        self.completion_conditions = list(answer["completion_conditions"])
        # Preserve source ROIs asynchronously. A free first segment need not
        # wait for a contact reference used later in the committed queue.
        self.source_scene, self.source_anchors = None, {}
        needs_source = bool(answer["targets"] or status == "observe"
                            or any(_needs_source_binding(s) for s in answer["segments"]))
        self.binding_token = (self.service.submit(view, list(self.targets.values()), force=True,
                                                  frame_id=self.step_index) if needs_source else None)
        self.binding_for_observe = status == "observe"
        self.active = self.primitive = self.checkpoint_pending = None
        self.queue = deque(deepcopy(answer["segments"]))
        self.need_model = status == "observe"
        if status == "observe":
            key = _signature({"targets": [{k: v for k, v in t.items() if k != "frame_id"}
                                          for t in answer["targets"]], "state": np.round(view.state, 3)})
            if key == self.last_observe_key:
                self._terminal("blocked", "repeated_observation_without_changed_evidence")
            self.last_observe_key = key
        else:
            self.last_observe_key = None

    def _protocol_repair(self, error, view):
        # A local contract error is a semantic repair outcome. No prefix of an
        # invalid batch executes, and an ordinary transport error is not retried.
        evidence = deepcopy(getattr(error, "evidence", {}))
        fingerprint = _signature({"state": np.round(view.state, 2),
                                  "anchors": {k: np.round(v, 2) for k, v in self._anchors().items()}})
        self.feedback = {"reason": "protocol_repair_required", "error": str(error)[:800],
                         "repair": evidence, "payload": self.payload.summary()}
        self._log("protocol_repair", **self.feedback)
        self.active = self.primitive = self.checkpoint_pending = None
        self.queue.clear()
        self.need_model = True
        if fingerprint in self.protocol_failures:
            self._terminal("blocked", "protocol_repair_exhausted_without_changed_state")
        else:
            self.protocol_failures.add(fingerprint)

    def _resolve(self, candidate, view):
        resolved = deepcopy(candidate)
        if candidate["skill"] in {"move_relative", "approach"}:
            # Resolve once from the segment's entry, including when a candidate
            # is retried after a new observation. Never accumulate the delta.
            resolved["xyz_m"] = (self.entry_state[:3]+np.asarray(candidate["xyz_m"])).tolist()
            resolved["skill"] = "move_to_pose"
        if candidate.get("quaternion_wxyz") is None and candidate["skill"] in {
                "move_to_pose", "move_relative", "approach"}:
            from ..astranomy_agentic_motion import measured_pose
            resolved["quaternion_wxyz"] = measured_pose(self.entry_state)[3:7]
        anchor = self.active.get("anchor")
        current = self._fresh_object(anchor)
        original = self.source_anchors.get(anchor)
        if self.active.get("grounding") is not None:
            from .grounding import ground_candidate
            result = ground_candidate(resolved, self.active, self.source_scene, self.scene,
                                      held=self.payload.held)
            event = {"segment_id": self.active["id"], "candidate": self.candidate_index,
                     "status": result["status"], "reason": result["reason"],
                     "evidence": result.get("evidence", {})}
            self.grounding_events.append(jsonable(event))
            self._log("contact_grounding", **event)
            if result["status"] == "unresolved":
                raise CandidateUnavailable(result["reason"], result.get("evidence", {}))
            return result["candidate"]
        # Only correct actual observed displacement since the source image.
        # Do not replace a grasp pose with a body/region centroid.
        if candidate["skill"] == "move_to_pose" and original is not None and current is not None:
            delta = np.median(current.points, axis=0)-original
            if np.linalg.norm(delta) > self.config.anchor_update_limit_m:
                raise ValueError("anchor_changed_beyond_local_correction")
            resolved["xyz_m"] = (np.asarray(resolved["xyz_m"])+delta).tolist()
        return resolved

    def _activate(self, view):
        if self.active is None:
            if not self.queue:
                self.need_model = True
                return False
            self.active = self.queue.popleft()
            self.entry_state = view.state.copy()
            self.candidate_index = 0
            self._log("segment_started", segment=self.active)
        if _needs_source_binding(self.active):
            self._consume_binding(wait=True)
        unavailable = []
        while self.candidate_index < len(self.active["candidates"]):
            try:
                resolved = self._resolve(self.active["candidates"][self.candidate_index], view)
                fragment = MeasuredFragment(view, resolved, self.gripper_command)
                if fragment.done:
                    raise ValueError(fragment.reason)
                self.primitive = fragment
                self._log("candidate_selected", segment_id=self.active["id"],
                          candidate=self.candidate_index, resolved=resolved)
                return True
            except ValueError as error:
                detail = {"reason": str(error), "candidate": self.candidate_index,
                          "evidence": getattr(error, "evidence", {})}
                unavailable.append(detail)
                self._log("candidate_unavailable", **detail)
                self.candidate_index += 1
        self._handoff("candidates_unavailable", {"candidate_count": self.candidate_index,
                      "candidates": unavailable}, view)
        return False

    def _handoff(self, reason, evidence, view):
        segment = deepcopy(self.active or self.checkpoint_pending)
        fingerprints = _failure_keys(reason, evidence, view, segment, self.failure_evidence)
        self.feedback = {"reason": reason, "evidence": jsonable(evidence), "segment": segment,
            "remaining_segments": [_segment_summary(s) for s in self.queue], "payload": self.payload.summary(),
            "blocker_fingerprints": sorted(fingerprints)}
        self._log("recovery_handoff", **self.feedback)
        if fingerprints & self.failed_segments:
            self._terminal("blocked", "recovery_exhausted_without_changed_evidence")
            return
        self.failed_segments.update(fingerprints)
        self.history.append({"segment": segment, "status": "interrupted", "reason": reason,
                             "step": self.step_index})
        self.active = self.primitive = self.checkpoint_pending = None
        self.need_model = True

    def _recover(self, result, view):
        reason = result.get("reason", result.get("status", "motion_unresolved"))
        keys = _failure_keys(reason, result.get("evidence", {}), view, self.active, self.failure_evidence)
        if not keys <= self.recoveries:
            self.recoveries.update(keys)
            self._refresh(view, force=True, wait=True)
            held = self.payload.held
            if (held is not None and held.get("extent_unknown")
                    and self.scene.frame_id == self.step_index):
                body = self._fresh_object(held["object_id"])
                if body is not None and len(body.points) >= 8 and not body.evidence.get("parent_id"):
                    # Re-observing must update the missing shape, not leave an
                    # immutable unknown-at-close flag blocking every retry.
                    # This is a new candidate reference, never proof of a hold.
                    up = held.get("upright_in_gripper")
                    self.payload.attach_candidate(self.scene, held["object_id"], view)
                    if up is not None:
                        self.payload.held["upright_in_gripper"] = up
                    self.payload.held["source"] = "whole_object_reobserved_while_closed"
                    self._log("payload_geometry_reobserved", object_id=held["object_id"])
            self._log("local_reobserve", segment_id=self.active["id"], reason=reason,
                      blocker_fingerprints=sorted(keys))
            return True
        self._log("candidate_unavailable", candidate=self.candidate_index, reason=reason,
                  evidence=result.get("evidence", {}))
        self.candidate_index += 1
        self.primitive = None
        if self.candidate_index >= len(self.active["candidates"]):
            self._handoff(reason, result.get("evidence", {}), view)
            return False
        return self._activate(view)

    def _scene_hold_evidence(self, view):
        held = self.payload.held
        if held is None or self.payload.close_object is None:
            return False
        obj = self._fresh_object(held["object_id"])
        if obj is None:
            return False
        initial = self.payload.close_object[:3, 3]
        predicted = (grip_pose(view.state) @ np.linalg.inv(self.payload.close_grip)
                     @ self.payload.close_object)[:3, 3]
        observed = np.median(obj.points, axis=0)
        moved, follow, stay = (float(np.linalg.norm(predicted-initial)),
            float(np.linalg.norm(observed-predicted)), float(np.linalg.norm(observed-initial)))
        error = float(np.max(np.asarray(obj.evidence.get("position_error_m", obj.evidence.get("sigma_m", .01)))))
        good = moved >= max(.008, error) and follow <= max(.01, 2*error) and stay > follow+.003
        if good and view.gap_mm > self.config.empty_gap_mm:
            held["confirmed"] = True
            self.payload.report = {"state": "held_confirmed", "source": "fresh_body_comotion",
                "object_id": held["object_id"], "follow_error_m": follow, "stationary_error_m": stay,
                "physical_contact_measured": False}
            return True
        return False

    def _release_evidence(self, view):
        if self.payload.report.get("state") == "released_stationary":
            return True
        released = self.payload.released
        if released is None or self.gripper_command != "open":
            return False
        obj = self._fresh_object(released["object_id"])
        if obj is None:
            return False
        hand_delta = view.position-released["released_grip"][:3, 3]
        delta = np.median(obj.points, axis=0)-np.median(released["predicted_world_points"], axis=0)
        error = float(np.max(np.asarray(obj.evidence.get("position_error_m", obj.evidence.get("sigma_m", .01)))))
        stationary, attached = float(np.linalg.norm(delta)), float(np.linalg.norm(delta-hand_delta))
        good = (np.linalg.norm(hand_delta) >= .012 and stationary <= max(.01, 2*error)
                and attached > stationary+.005)
        if good:
            self.payload.report = {"state": "release_observed", "source": "fresh_body_separation",
                "object_id": released["object_id"], "support_contact_measured": False}
        return bool(good)

    def _checkpoint(self, view):
        segment = self.checkpoint_pending
        if segment is None:
            return True
        kind = segment["checkpoint"]
        if kind in {"held", "released"}:
            expected = segment.get("checkpoint_object", segment.get("object"))
            actual = self.payload.held if kind == "held" else self.payload.released
            observed_id = None if actual is None else actual.get("object_id")
            if expected is None or expected != observed_id:
                self._handoff("checkpoint_object_unresolved", {"checkpoint": kind,
                    "expected_object": expected, "observed_object": observed_id}, view)
                return False
        ready = (kind in {"none", "observe"}
                 or kind == "held" and self.payload.held is not None and self.payload.held.get("confirmed")
                 or kind == "released" and self._release_evidence(view))
        if not ready:
            self._refresh(view, force=True, wait=True)
            ready = self._scene_hold_evidence(view) if kind == "held" else self._release_evidence(view)
        if not ready:
            self._handoff(kind+"_evidence_unresolved", self.payload.summary(), view)
            return False
        self.feedback = {"reason": "local_checkpoint_passed", "checkpoint": kind,
                         "payload": self.payload.summary()}
        self._log("checkpoint", segment_id=segment["id"], **self.feedback)
        self.checkpoint_pending = None
        return True

    def _decision(self, action, calls, kind, evidence=None):
        self.pending_kind = kind
        return Decision(tuple(float(x) for x in action), jsonable({"version": VERSION,
            "controller": VERSION, "step": self.step_index,
            "phase": None if self.active is None else self.active["id"],
            "primitive": "measured_v2_fragment" if kind == "fragment" else "observation_hold",
            "api_calls": list(calls), "api_latency_s": sum(c["latency_s"] for c in calls),
            "terminal": self.terminal, "motion_evidence": evidence or {},
            "goal_xyz_m": None if self.primitive is None else self.primitive._target_xyz,
            "goal_quaternion_wxyz": None if self.primitive is None else self.primitive._target_wxyz,
            "control_timing": {"filter_wall_s": self._act_filter_seconds,
                               "filter_calls": self._act_filter_calls},
            "queue_remaining": len(self.queue)}))

    def _hold(self, view, calls=()):
        return self._decision(self.adapter.adapt([0., 0., 0.], view.quaternion_xyzw,
                              self.gripper_command, view.state), calls, "observation_hold")

    def act(self, observation):
        self._act_filter_seconds, self._act_filter_calls = 0., 0
        view = self.last_view if self.last_view is not None and self.last_view.observation is observation else StepView(observation)
        self.last_view = view
        calls = []
        if self.terminal is not None:
            return self._hold(view)
        if getattr(observation, "geometry", None) is None or getattr(observation, "joint_position", None) is None:
            self._terminal("blocked", "public_robot_state_or_calibration_missing")
            return self._hold(view)
        if self.binding_for_observe:
            self._consume_binding(wait=True)
        self._refresh(view, wait=self.scene is None and not self.need_model)
        if self.need_model:
            self._ask(view, calls)
            return self._hold(view, calls)
        if not self._checkpoint(view):
            return self._hold(view, calls)
        # Bounded local alternatives; robot steps only occur in the runner.
        for _ in range(8):
            if self.terminal is not None or self.need_model:
                return self._hold(view, calls)
            if self.primitive is None and not self._activate(view):
                continue
            proposed = self.primitive.next_action(view)
            runtime_segment = {**self.active, "_entry_state": self.entry_state,
                "_goal_xyz_m": self.primitive._target_xyz,
                "_goal_quaternion_wxyz": self.primitive._target_wxyz,
                "_candidate": self.active["candidates"][self.candidate_index]}
            started = monotonic()
            try:
                result = self.command_filter.filter(proposed, view, self.scene, runtime_segment,
                                                    held=self.payload.held)
            except Exception:
                self.primitive.cancel_proposal(self.gripper_command)
                raise
            finally:
                elapsed = monotonic()-started
                self.filter_seconds += elapsed
                self._act_filter_seconds += elapsed
                self._act_filter_calls += 1
            self.last_filter = result
            if result["status"] in {"execute", "scaled"}:
                actual = self.primitive.commit_action(result["action"])
                return self._decision(actual, calls, "fragment", result.get("evidence", {}))
            self.primitive.cancel_proposal(self.gripper_command)
            if not self._recover(result, view):
                return self._hold(view, calls)
        self._handoff("local_recovery_cycle", self.last_filter, view)
        return self._hold(view, calls)

    def observe_transition(self, before, decision, after):
        old = self.last_view if self.last_view is not None and self.last_view.observation is before else StepView(before)
        view = StepView(after)
        previous_gripper = self.gripper_command
        self.gripper_command = "close" if decision.action[6] > 0 else "open"
        if self.gripper_command == "close" and previous_gripper != "close" and self.active:
            self.grasp_attempts += 1
            if self.active.get("object") and self.active["contact_mode"] != "articulated":
                self.payload.attach_candidate(self.scene, self.active["object"], old,
                    keep_upright=self.active.get("keep_upright", False))
        if self.gripper_command == "open" and previous_gripper == "close":
            released_id = None if self.payload.held is None else self.payload.held["object_id"]
            self.payload.released_at(old)
            if released_id:
                self.service.invalidate([released_id], reason="released_reobserve")
        if self.pending_kind == "fragment":
            self.primitive.observe_transition(old, decision.action, view)
        observe = getattr(self.command_filter, "observe_transition", None)
        if callable(observe):
            observe(old, decision.action, view)
        self.step_index += 1
        self.last_view = view
        self.path.append(view.position.tolist())
        self.path = self.path[-1000:]
        frame = frame_from_step_view(view, after.geometry, self.step_index, monotonic())
        self.payload.observe(view, frame, self.gripper_command)
        if self.pending_kind == "fragment" and self.primitive.done:
            outcome = self.primitive.outcome()
            self.history.append({"segment": deepcopy(self.active), "status": outcome["status"],
                                 "reason": outcome["reason"], "step": self.step_index})
            self._log("segment_result", segment_id=self.active["id"], outcome=outcome)
            if outcome["status"] == "succeeded":
                self.checkpoint_pending = self.active
                self.active = self.primitive = None
            else:
                self._handoff(outcome["reason"], outcome, view)
        if (not self.need_model and self.terminal is None
                and self.payload.report.get("state") in {"suspected_slip", "object_absent"}):
            self._handoff(self.payload.report["state"], self.payload.summary(), view)
        self.pending_kind = None
        self.feedback = {**self.feedback, "interaction": deepcopy(self.payload.report)}
        self._log("feedback", state=after.state, interaction=self.payload.report,
                  segment_id=None if self.active is None else self.active["id"], terminal=self.terminal)
        return jsonable(self.feedback)

    def planning_snapshot(self):
        cloud = np.empty((0, 3)) if self.scene is None else np.asarray(self.scene.scene_points)
        if len(cloud) > 1600:
            cloud = cloud[np.linspace(0, len(cloud)-1, 1600, dtype=int)]
        poses = []
        if self.primitive is not None:
            from ..robodojo_direct_sim import quaternion_matrix
            q = self.primitive._target_wxyz
            poses = [{"name": self.active["id"], "position": self.primitive._target_xyz,
                      "rotation": quaternion_matrix(q)}]
        return jsonable({"step": self.step_index, "stage": None if self.active is None else self.active["id"],
            "status": self.terminal or ("awaiting_model" if self.need_model else "executing"),
            "scene_points": cloud, "scene_point_strong": np.zeros(len(cloud), bool),
            "keyposes": poses, "selected_path": [] if not poses or not self.last_view else
                [self.last_view.position, poses[0]["position"]], "measured_path": self.path[-500:],
            "geometry_is_prediction": True})

    def agent_stats(self):
        return jsonable({"version": VERSION, "revision": REVISION, "step": self.step_index, "terminal": self.terminal,
            "model_decisions": self.model_decisions, "grasp_attempts": self.grasp_attempts,
            "queue_remaining": len(self.queue), "segment_history": self.history,
            "feedback": self.feedback, "payload": self.payload.summary(), "config": asdict(self.config),
            "perception": getattr(self.service, "stats", {}),
            "motion": {**getattr(self.command_filter, "stats", {}),
                       "timing": self.command_filter.timing_summary() if hasattr(self.command_filter, "timing_summary") else {}},
            "model_requests": self.model_requests, "contact_grounding": self.grounding_events,
            "timing": {"model_s": self.model_seconds, "perception_wait_s": self.vision_wait_seconds,
                "command_filter_s": self.filter_seconds, "wall_s": monotonic()-self.started_at}})
