"""Local execution advice from uncertain scene hypotheses, not a free-space proof.

The old GeometryMemory remains available for reproducing strict-coverage runs.
This policy retains its identity/attachment/visual evidence, but separates scene
knowledge from execution permission. Missing free-region boxes never certify or
forbid motion. Robot bounds, measured anomalies, unavailable payload shapes and
material changes still require intervention. Obstacle cores are conservative
model-conflict guards, not measured collision truth or calibrated confidence.
"""
from collections import OrderedDict
from copy import deepcopy
import math

import numpy as np

from .action import TRANSLATION_SCALE_M, ROTATION_SCALE_RAD, _rotvec_quaternion, _quaternion_product
from .astranomy_agentic_motion import measured_pose
from .agent_speedup_v3_geometry import (
    GeometryMemory, _box, _quaternion, _rotation, _slerp, _vector, _separations,
)

LOCAL_GEOMETRY_VERSION = "agent-speedup-v3-local-geometry-10"
ROUTE_CHECKPOINT_STEPS = 8
MAX_ROUTE_INTERVALS = 8


class LocalGeometryMemory(GeometryMemory):
    execution_policy = "goal_local"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._local_routes = OrderedDict()
        self.local_compute_counts = {"route_checks": 0, "route_reuses": 0, "command_checks": 0}

    def preview_scene_update(self, update, view, step):
        preview = super().preview_scene_update(update, view, step)
        preview._local_routes = self._local_routes.copy()
        preview.local_compute_counts = self.local_compute_counts.copy()
        return preview

    def _obstacles(self, view, probe_ids):
        obstacles = []
        for entity in self.entities.values():
            if entity["role"] in ("free_region", "destination_region"):
                continue
            pending = entity.get("release_pending_world_envelope") if entity.get("release_pending") else None
            if pending:
                obstacles.append({**_box(pending["center"], np.eye(3), pending["half"], entity["id"],
                    "pending_release_world_alternative", pending["uncertainty"]), "alternative_for": entity["id"]})
            if entity["attachment"] == "held":
                for enabled, prefix in ((entity["id"] in probe_ids, "attachment_probe"),
                                        (entity.get("release_pending"), "release")):
                    if enabled and entity.get(prefix+"_position_estimate_m"):
                        alternative = {**entity, "attachment": "none",
                            "position_m": entity[prefix+"_position_estimate_m"],
                            "quaternion_wxyz": entity.get(prefix+"_quaternion_estimate_wxyz", entity["quaternion_wxyz"])}
                        obstacles.extend({**b, "alternative_for": entity["id"]}
                                         for b in self._entity_boxes(alternative, suppress_stale_padding=True))
                continue
            obstacles.extend(self._entity_boxes(entity, suppress_stale_padding=True))
            for index, excluded in enumerate(entity.get("local_motion_exclusion_boxes", [])):
                obstacles.append(_box(excluded["center"], np.eye(3), excluded["half"], entity["id"],
                                      f"observed_displacement_{index}", excluded["uncertainty"]))
        return obstacles

    def _allowed_pair(self, body, obstacle, context):
        if body["owner"] == obstacle.get("alternative_for") or body["owner"] == obstacle["owner"]:
            return True
        intent = context.get("intent")
        target = self.entities.get(context.get("target_id"), {})
        related = {context.get("target_id"), target.get("parent_id"), *target.get("related_ids", [])}
        allowed = set(context.get("allowed_contact_ids", [])) & related
        if context.get("gripper_action") and context.get("commanded_gripper") == "open":
            allowed |= {context.get("held_object_id")}
        if obstacle["owner"] not in allowed:
            return False
        if body["owner"] == "robot_hand":
            # Pushing intentionally uses the closed tool; transit and approach
            # do not inherit that permission. Grasp contact is finger-only.
            return ((intent == "manipulate" and obstacle["owner"] == context.get("target_id"))
                    or (body["name"].startswith("finger") and intent in ("approach", "grasp", "place", "manipulate"))
                    or (body["name"].startswith("finger") and context.get("gripper_action")
                        and context.get("commanded_gripper") == "open"))
        return intent == "place" and body["owner"] == context.get("held_object_id")

    def _route(self, view, target, q0, q1, context, bodies, obstacles, *, exact, step):
        """Bound the hand/payload over short rigid path intervals; no box unions.

        Translation is enclosed in each body's local axes; the rotation bound
        encloses deviations from the midpoint orientation. The same final
        coupled-command check is always performed, even when route advice is
        cached. A cached route is never a physical free-space certificate.
        """
        key = (self._geometry_epoch, tuple(target), tuple(q1), context.get("intent"),
               context.get("target_id"), context.get("held_object_id"),
               tuple(context.get("allowed_contact_ids", [])), context.get("commanded_gripper"),
               tuple((b["name"], tuple(b["half"]), b["uncertainty"]) for b in bodies))
        cached = self._local_routes.get(key)
        if not exact and cached is not None and 0 <= step-cached[0] < ROUTE_CHECKPOINT_STEPS:
            self.local_compute_counts["route_reuses"] += 1
            return {**deepcopy(cached[1]), "route_advice_reused": True}
        self.local_compute_counts["command_checks" if exact else "route_checks"] += 1
        distance = float(np.linalg.norm(target-view.position))
        angle = 2*math.acos(float(np.clip(abs(np.dot(q0, q1)), 0., 1.)))
        intervals = 1 if exact else min(MAX_ROUTE_INTERVALS, max(1, math.ceil(distance/.02), math.ceil(angle/.1)))
        result = {"conflicts": [], "advisories": [], "evidence_ids": [], "limiting_pair": None,
                  "nominal_clearance_m": None, "route_advice_reused": False, "intervals": intervals}
        if not obstacles:
            return result
        centers = np.asarray([b["center"] for b in obstacles])
        rotations = np.asarray([b["rotation"] for b in obstacles])
        nominal = np.asarray([b["half"] for b in obstacles])
        uncertainty = np.asarray([b["uncertainty"] for b in obstacles])
        # These eroded cores express robustness to the DECLARED model error.
        # They are not statistically calibrated confidence bounds.
        cores = np.maximum(nominal-uncertainty[:, None], 0.)
        has_core = np.all(cores > 1e-9, axis=1)
        evidence, conflicts, advisories = set(), {}, {}
        best = math.inf
        for body in bodies:
            mask = np.array([self._allowed_pair(body, obstacle, context) for obstacle in obstacles]) | ~has_core
            initial = _box(view.position+view.R@body["center"], view.R@body["rotation"],
                           body["half"], body["owner"], body["name"])
            baseline = _separations(initial, centers, rotations, cores)
            for index in range(intervals):
                a, b = index/intervals, (index+1)/intervals
                midpoint = (a+b)/2
                rotation = _rotation(_slerp(q0, q1, midpoint))
                world_rotation = rotation@body["rotation"]
                delta = (target-view.position)/intervals
                radius = float(np.linalg.norm(body["center"])+np.linalg.norm(body["half"]))
                rotation_pad = 2*radius*math.sin(angle/(4*intervals))
                half = body["half"] + np.abs(world_rotation.T@delta)/2 + rotation_pad
                world = _box(view.position+midpoint*(target-view.position)+rotation@body["center"],
                             world_rotation, half, body["owner"], body["name"])
                nominal_gaps = _separations(world, centers, rotations, nominal)
                core_gaps = _separations(world, centers, rotations, cores)
                endpoint_rotation = _rotation(_slerp(q0, q1, b))
                endpoint = _box(view.position+b*(target-view.position)+endpoint_rotation@body["center"],
                                endpoint_rotation@body["rotation"], body["half"], body["owner"], body["name"])
                endpoint_gaps = _separations(endpoint, centers, rotations, cores)
                for oi, obstacle in enumerate(obstacles):
                    if self._allowed_pair(body, obstacle, context):
                        continue
                    gap = float(nominal_gaps[oi])
                    if gap < best:
                        best = gap
                        result["limiting_pair"] = {"body_id": body["owner"], "body_part": body["name"],
                            "obstacle_id": obstacle["owner"], "clearance_estimate_m": gap,
                            "combined_extent_uncertainty_m": float(uncertainty[oi])+body["uncertainty"]}
                    if gap <= .04+uncertainty[oi]+body["uncertainty"]:
                        evidence.add(obstacle["owner"])
                    pair = (body["name"], obstacle["name"])
                    if gap <= uncertainty[oi]+body["uncertainty"]:
                        advisories[pair] = {"reason": "estimated_obstacle_margin", "body_part": body["name"],
                            "entity_id": obstacle["owner"], "nominal_clearance_m": gap,
                            "uncertainty_m": float(uncertainty[oi])+body["uncertainty"]}
                    # Permit an escape from an already overlapping estimate
                    # only if this entire interval never deepens the overlap.
                    escape = (baseline[oi] < 0 and core_gaps[oi] >= baseline[oi]-1e-6
                              and endpoint_gaps[oi] >= baseline[oi]-1e-6)
                    if not mask[oi] and core_gaps[oi] < -1e-6 and not escape:
                        conflicts[pair] = {"reason": "estimated_obstacle_core_conflict", "body_part": body["name"],
                            "entity_id": obstacle["owner"], "path_fraction": b,
                            "core_clearance_m": float(core_gaps[oi]), "measured_collision": False}
        result.update(conflicts=list(conflicts.values())[:8], advisories=list(advisories.values())[:8],
                      evidence_ids=sorted(evidence), nominal_clearance_m=None if not math.isfinite(best) else best)
        if not exact:
            self._local_routes[key] = (int(step), deepcopy(result))
            while len(self._local_routes) > 8:
                self._local_routes.popitem(last=False)
        return result

    def _evaluate(self, view, target_xyz, target_wxyz, context, step, *, exact=False):
        context = context or {}
        target = np.asarray(_vector(target_xyz, 3, "target_xyz"))
        q0 = _quaternion(measured_pose(view.state)[3:7])
        q1 = _quaternion(q0 if target_wxyz is None else target_wxyz)
        distance = float(np.linalg.norm(target-view.position))
        angle = 2*math.acos(float(np.clip(abs(np.dot(q0, q1)), 0., 1.)))
        stationary_gripper = bool(context.get("gripper_action") and distance <= 1e-6 and angle <= .001)
        reference = context.get("gripper_settling_authorization")
        if context.get("gripper_action") and isinstance(reference, dict):
            # Only a runtime-owned, fixed hold anchor can authorize small pose
            # corrections during settling. A moving close/open request cannot.
            anchor = np.asarray(reference.get("hold_position_m", view.position))
            anchor_q = _quaternion(reference.get("hold_quaternion_wxyz", q0))
            tolerance = min(.01, float(reference.get("position_tolerance_m", .004)))
            stationary_gripper |= bool(reference.get("source") == "runtime_validated_stationary_gripper"
                and reference.get("gripper_command") == context.get("commanded_gripper")
                and reference.get("entry_step", step+1) <= step <= reference.get("entry_step", -1)+reference.get("max_steps", 0)
                and np.linalg.norm(view.position-anchor) <= tolerance and np.linalg.norm(target-anchor) <= tolerance
                and abs(float(np.dot(q1, anchor_q))) >= math.cos(min(.1, float(reference.get("orientation_tolerance_rad", .08)))/2))
        settling = self._settling_ids(view, target, q1, context, step)
        probe_ids = self._attachment_probe_ids(context, step)
        bodies, payload_unknown = self._local_bodies(view, context, settling, probe_ids)
        obstacles = self._obstacles(view, probe_ids)
        route = self._route(view, target, q0, q1, context, bodies, obstacles, exact=exact, step=step)
        evidence = set(route["evidence_ids"]) | {i for i in (context.get("target_id"), context.get("held_object_id")) if i in self.entities}
        blocking = ["obstacle_core:"+x["entity_id"]+":"+x["body_part"] for x in route["conflicts"]]
        advisories = [{"reason": "unmeasured_free_space", "coverage_is_execution_precondition": False}, *route["advisories"]]
        tracking = context.get("tracking") or {}
        revoked = tracking.get("status") in ("mismatch", "lost")
        if revoked:
            blocking.append("measured_tracking_mismatch")
        held = context.get("held_object_id")
        if held and not any(body["owner"] == held for body in bodies):
            blocking.append("payload_shape_required:"+str(held))
        uncertain_attachments = [e["id"] for e in self.entities.values() if e.get("attachment_probe_unconfirmed")]
        if uncertain_attachments and not set(uncertain_attachments) <= probe_ids and not stationary_gripper:
            blocking.append("attachment_verification_required")
        changed = [ident for ident in evidence if ident not in (settling | probe_ids) and self.entities[ident].get("stale")
                   and self.entities[ident].get("invalidation_kind") in ("visual_appearance_changed", "possible_slip")]
        blocking.extend("changed_geometry:"+ident for ident in changed)
        target_id = context.get("target_id")
        final_interaction = context.get("intent") in ("grasp", "place", "manipulate")
        if context.get("gripper_action") and context.get("commanded_gripper") == "close" and context.get("intent") == "grasp":
            if target_id not in self.entities or self.entities[target_id]["role"] not in ("object", "graspable_part"):
                blocking.append("grasp_target_shape_required:"+str(target_id))
        obs = getattr(view, "observation", None)
        has_rgb = bool(getattr(obs, "image", None) or getattr(obs, "wrist_image", None))
        moving = distance > 1e-6 or angle > .001
        if moving and not has_rgb:
            blocking.append("current_public_rgb_required")
        allowed = not blocking
        translation_cap = min(.02, max(0., float(context.get("controller_translation_limit_m", .02))))
        rotation_cap = min(.1, max(0., float(context.get("controller_rotation_limit_rad", .1))))
        if exact:
            if np.linalg.norm(target-view.position) > translation_cap+1e-8:
                blocking.append("controller_translation_limit")
            if 2*math.acos(float(np.clip(abs(np.dot(q0, q1)), 0., 1.))) > rotation_cap+1e-8:
                blocking.append("controller_rotation_limit")
            allowed = not blocking
        # Near a nominal surface use the existing contact controller scale;
        # uncertainty is not used to repeatedly divide all commands.
        if route["advisories"] and any(x["nominal_clearance_m"] <= 0 for x in route["advisories"]):
            translation_cap = min(translation_cap, .005)
        reason = blocking[0] if blocking else "bounded_observed_execution_with_geometry_advice"
        prefix = {"status": "admissible" if allowed else "blocked", "fraction": float(allowed),
                  "xyz_m": target.tolist() if allowed else view.position.tolist(),
                  "quaternion_wxyz": q1.tolist() if allowed else q0.tolist(),
                  "reason": reason, "physical_clearance_verified": False}
        result = {"version": LOCAL_GEOMETRY_VERSION, "status": "unknown", "execution_policy": self.execution_policy,
                  "execution_status": prefix["status"], "execution_allowed": allowed,
                  "blocking_reasons": sorted(set(blocking)), "advisories": advisories,
                  "reason": reason, "allowed_fraction": float(allowed), "execution_prefix": prefix,
                  "suggested_steps": None, "translation_step_m": translation_cap, "rotation_step_rad": rotation_cap,
                  "translation_limit_m": translation_cap, "rotation_limit_rad": rotation_cap,
                  "evidence_ids": sorted(evidence), "support_region_ids": [],
                  "payload_evidence_ids": [b["owner"] for b in bodies if b["owner"] != "robot_hand"],
                  "unknown_fraction": 1., "clearance_estimate_m": route["nominal_clearance_m"],
                  "limiting_pair": route["limiting_pair"], "coverage_witnesses": [],
                  "missing_evidence_requests": sorted(set(blocking)), "inventory_missing_evidence_requests": [],
                  "stale_evidence_ids": [i for i in evidence if self.entities[i].get("stale")],
                  "monitoring_unknown_ids": [i for i in evidence if self.entities[i].get("monitoring_unknown")],
                  "tracking_status": tracking.get("status", "unobserved"), "motion_support_revoked": revoked,
                  "payload_geometry_unknown": payload_unknown, "final_interaction_guard": final_interaction,
                  "gripper_settling_expected_ids": sorted(settling), "gripper_settling_reference": None,
                  "allowed_contact_ids": list(context.get("allowed_contact_ids", [])), "at_step": step,
                  "scene_revision": self.revision, "conditional_probe_geometry_supported": False,
                  "conditional_probe_execution_allowed": bool(probe_ids and allowed),
                  "source": "model_geometry_advice_and_measured_robot_feedback",
                  "physical_clearance_verified": False, "clearance_is_physical_certificate": False,
                  "full_arm_checked": False, "geometry_computation": {**route, "counts": dict(self.local_compute_counts)}}
        result["whole_goal_assessment"] = {k: deepcopy(result[k]) for k in
            ("status", "execution_status", "execution_allowed", "reason", "blocking_reasons", "advisories", "allowed_fraction")}
        if allowed and context.get("gripper_action"):
            result["gripper_settling_reference"] = {
                "source": "runtime_validated_stationary_gripper", "physical_clearance_verified": False,
                "execution_policy": self.execution_policy, "target_id": target_id,
                "held_object_id": held, "gripper_command": context.get("commanded_gripper"),
                "allowed_contact_ids": list(context.get("allowed_contact_ids", [])),
                "support_ids": list((context.get("scene_inventory") or {}).get("support_ids", [])),
                "nearby_obstacle_ids": list((context.get("scene_inventory") or {}).get("nearby_obstacle_ids", [])),
                "entity_revisions": {i: self.entities[i]["geometry_revision"] for i in evidence},
                "hold_position_m": target.tolist(), "hold_quaternion_wxyz": q1.tolist(),
                "position_tolerance_m": min(.01, max(1e-6, float(context.get("position_tolerance_m", .004)))),
                "orientation_tolerance_rad": min(.1, max(.001, float(context.get("orientation_tolerance_rad", .08))))}
        return result

    def assess_motion(self, view, target_xyz, target_wxyz, context=None, step=None):
        step = self.step if step is None else int(step)
        result = self._evaluate(view, target_xyz, target_wxyz, context, step)
        epoch = self._geometry_epoch
        self.watch_observation(view, step, result["evidence_ids"])
        if epoch != self._geometry_epoch:
            result = self._evaluate(view, target_xyz, target_wxyz, context, step)
        self.last_motion = deepcopy(result)
        return result

    def verify_command(self, view, normalized_action, context=None, step=None, *, allow_unknown=False):
        action = np.asarray(_vector(normalized_action, 7, "normalized_action"))
        if np.any(np.abs(action) > 1+1e-9):
            raise ValueError("normalized action exceeds controller bounds")
        context = {**(context or {})}
        if abs(action[6]) > 1e-9:
            context["commanded_gripper"] = "close" if action[6] > 0 else "open"
        q = _quaternion_product(_rotvec_quaternion(action[3:6]*ROTATION_SCALE_RAD),
                                _rotvec_quaternion(np.asarray(view.state[3:6])))[[3, 0, 1, 2]]
        result = self._evaluate(view, view.position+action[:3]*TRANSLATION_SCALE_M, q, context,
                                self.step if step is None else int(step), exact=True)
        epoch = self._geometry_epoch
        self.watch_observation(view, self.step if step is None else int(step), result["evidence_ids"])
        if epoch != self._geometry_epoch:
            result = self._evaluate(view, view.position+action[:3]*TRANSLATION_SCALE_M, q, context,
                                    self.step if step is None else int(step), exact=True)
        if not result["execution_allowed"]:
            # Preserve the actual command blocker for the next model decision;
            # an earlier cached route advisory must not conceal this rejection.
            self.last_motion = {**(self.last_motion or {}), **{key: deepcopy(result[key]) for key in
                ("execution_allowed", "execution_status", "reason", "blocking_reasons", "advisories",
                 "missing_evidence_requests", "allowed_fraction", "at_step")}}
        return {"command_allowed": result["execution_allowed"], "status": result["status"],
                "reason": result["reason"], "blocking_reasons": result["blocking_reasons"],
                "advisories": result["advisories"], "physical_clearance_verified": False,
                "source": "exact_coupled_command_local_guard"}

    def summary(self, view=None, context=None):
        result = super().summary(view, context)
        result.update(version=LOCAL_GEOMETRY_VERSION, execution_policy=self.execution_policy,
                      local_geometry_computation_counts=dict(self.local_compute_counts),
                      free_region_coverage_required=False, unknown_space_is_free=False)
        return result
