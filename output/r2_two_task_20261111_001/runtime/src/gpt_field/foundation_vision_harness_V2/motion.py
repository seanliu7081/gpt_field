"""Check the next measured OSC command against typed local scene evidence.

The v2 fragment executor owns target tracking. This filter neither solves a
whole IK route nor requires the entire scene to become conflict-free. Predicted
external surfaces, unknown observations, and the known robot remain distinct.
Only a dispatched command updates the response model via observe_transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from ..action import TRANSLATION_SCALE_M, ROTATION_SCALE_RAD
from ..contact_skills import StepView
from ..foundation_harness.types import KeyPose, jsonable
from ..foundation_vision_harness_V1.geometry import pose_from_state, observed_points, object_sigma, _support_frame
from ..foundation_vision_harness_V1.motion import RollingPlanner, MotionConfig, PlanningError, _payload_shape
from .telemetry import TimingSeries


@dataclass(frozen=True)
class FilterConfig(MotionConfig):
    """Robot/control settings, without an episode or full-route search budget."""
    unknown_command_m: float = .005
    unknown_rotation_rad: float = .03
    unknown_proximity_m: float = .015
    unknown_surface_margin_m: float = .002
    minimum_body_motion_m: float = .0002
    scene_position_tolerance_m: float = .0001


def _array(evidence, name, count, default=0.):
    value = np.asarray(evidence.get(name, default), float)
    if value.ndim == 0:
        value = np.full(count, float(value))
    if value.shape != (count,) or not np.isfinite(value).all() or np.any(value < 0.):
        raise PlanningError("scene_error_components_misaligned", {"field": name})
    return value


class CommandFilter(RollingPlanner):
    """Reuse robot hulls/short OSC prediction, not the V1 route admission gate."""

    def __init__(self, kinematics=None, config=None, *, robot_model=None):
        super().__init__(kinematics, config or FilterConfig(), robot_model=robot_model)
        self._unknown_points = np.empty((0, 3))
        self._unknown_owners = np.empty(0, object)
        self._unknown_reasons = np.empty(0, object)
        self._unknown_radius = np.empty(0)
        self._unknown_position_error = np.empty(0)
        self._unknown_tree = None
        self._pending_command = None
        self._previous_receipt = None
        self._response_needs_slow = False
        self._filter_timing = TimingSeries()
        self.stats.update(filter_calls=0, filter_s=0., dispatched_commands=0,
                          execute=0, scaled=0, reobserve=0, candidate_unavailable=0)

    def _prepare(self, scene, state8, q7):
        self._bound = self.kinematics.bind(state8, q7)
        if scene is None or scene.evidence.get("perception_status") in {"unavailable", "error"}:
            raise PlanningError("scene_unavailable")
        key = (id(scene), scene.evidence.get("revision", scene.frame_id))
        if self._scene_key == key:
            return
        evidence = scene.evidence
        points = np.asarray(scene.scene_points, float).reshape(-1, 3)
        owners = np.asarray(getattr(scene, "scene_point_object_ids", None), object)
        if owners.shape != (len(points),):
            raise PlanningError("scene_ownership_misaligned")
        identities = np.asarray(evidence.get("point_identity", np.full(len(points), "external")), object)
        if identities.shape != (len(points),) or np.any(identities != "external"):
            raise PlanningError("external_map_contains_other_identity")
        # scene_points contains admitted surfaces only. Disputed depth samples
        # remain in scene.evidence for diagnosis; they are not occupied patches
        # and must not be inserted into either collision tree. Credible but
        # unidentified surfaces still use the unresolved tree below.
        # A positional bound and representation radius are separate inputs;
        # identity ambiguity is never multiplied into a larger solid obstacle.
        positional = _array(evidence, "point_position_error_m", len(points), .005)
        representation = _array(evidence, "point_representation_radius_m", len(points))
        valid = np.isfinite(points).all(1)
        self._points, self._owners = points[valid], owners[valid]
        self._sigma = (positional+representation)[valid]
        self._maximum_sigma = float(self._sigma.max(initial=0.))
        self._tree = cKDTree(self._points) if len(self._points) else None
        unknown = np.asarray(evidence.get("unresolved_points", []), float).reshape(-1, 3)
        names = np.asarray(evidence.get("unresolved_owners", np.full(len(unknown), "")), object)
        reasons = np.asarray(evidence.get("unresolved_reason", np.full(len(unknown), "identity_unresolved")), object)
        if names.shape != (len(unknown),) or reasons.shape != (len(unknown),):
            raise PlanningError("unresolved_map_misaligned")
        radius = _array(evidence, "unresolved_representation_radius_m", len(unknown))
        error = _array(evidence, "unresolved_position_error_m", len(unknown), .02)
        valid = np.isfinite(unknown).all(1)
        self._unknown_points, self._unknown_owners = unknown[valid], names[valid]
        self._unknown_reasons = reasons[valid]
        self._unknown_radius, self._unknown_position_error = radius[valid], error[valid]
        self._unknown_tree = cKDTree(self._unknown_points) if len(self._unknown_points) else None
        self._scene_key, self._scene_ref = key, scene
        self.stats["scene_preparations"] += 1

    def _contact_key(self, scene, state, segment, held):
        source, anchor = segment.get("object"), segment.get("anchor")
        mode = segment.get("contact_mode", "free")
        parts = list(segment.get("contact_parts") or [])
        allowed, permissions = [], {}
        if mode != "free" and source:
            source_parts = [part for part in parts if part in {"finger", "pad", "tool"}]
            if source_parts:
                allowed.append(source)
                permissions[source] = source_parts
                # Contact-region points carry a separate identity from body.
                for name, obj in scene.objects.items():
                    if obj.evidence.get("parent_id") == source:
                        allowed.append(name)
                        permissions[name] = source_parts
        metadata = {"contact_parts": parts, "contact_permissions": permissions,
                    "scope": "actual_next_command"}
        if mode == "place" and anchor and "payload" in parts and held is not None:
            support = scene.objects.get(anchor)
            if support is not None and support.status not in {"missing", "unobserved", "stale", "conflict", "ambiguous"}:
                try:
                    points = observed_points(support)
                    if len(points) >= 6:
                        sigma = object_sigma(support)
                        point, normal, _, _ = _support_frame(points, sigma, support.evidence)
                        ids = [anchor]
                        if support.evidence.get("parent_id"):
                            ids.append(support.evidence["parent_id"])
                        for name in ids:
                            allowed.append(name)
                            permissions.setdefault(name, []).append("payload")
                        metadata["payload_contact_plane"] = {"point": point.tolist(), "normal": normal.tolist(),
                            "thickness_m": max(self.config.collision_margin_m, float(sigma@np.abs(normal)))}
                except (ValueError, RuntimeError):
                    metadata["support_contact_permission"] = "unresolved"
        return KeyPose(segment.get("id", "command"), pose_from_state(state), "keep",
                       "contact" if mode != "free" else "transit", tuple(dict.fromkeys(allowed)), metadata)

    def _unknown_check(self, q0, q1, fingers, held, next_fingers=None, key=None):
        """Inspect only unknown material near bodies that this command moves.

        Unknown positional error controls the proximity/slow zone. It is not
        promoted to a thick solid obstacle. Entering its nominal visible patch
        needs another observation; existing non-deepening overlap can leave.
        """
        if self._unknown_tree is None:
            return {"nearby": 0, "entered": [], "existing": 0}
        before = self.robot.shapes(self._bound, q0, fingers)
        after = self.robot.shapes(self._bound, q1, fingers if next_fingers is None else next_fingers)
        if held is not None:
            before.append(_payload_shape(held, self._bound.forward(q0)[0]))
            after.append(_payload_shape(held, self._bound.forward(q1)[0]))
        old = {shape.name: shape for shape in before}
        held_ids = set() if held is None else set(held.get("owner_ids", [held.get("object_id")]))
        nearby, entered, existing = set(), [], 0
        max_radius = float(self._unknown_radius.max(initial=0.))
        max_error = float(self._unknown_position_error.max(initial=0.))
        for shape in after:
            prior = old[shape.name]
            if shape.kind == "arm" and shape.group == 0:
                continue
            motion = float(np.linalg.norm(shape.center-prior.center)
                           + shape.radius*np.linalg.norm(shape.pose[:3, :3]-prior.pose[:3, :3], ord=2))
            if motion < self.config.minimum_body_motion_m:
                continue
            ids = np.asarray(self._unknown_tree.query_ball_point(shape.center,
                shape.radius+max_radius+max_error+self.config.unknown_proximity_m+motion), int)
            if held_ids:
                ids = ids[~np.isin(self._unknown_owners[ids], list(held_ids))]
            # Intended contact remains owner- and part-specific in this map as
            # well. Anonymous/other-object ambiguity is never swept away by a
            # grasp permission. Payload support permissions retain their plane.
            if key is not None and len(ids):
                owners = {owner for owner, parts in key.metadata.get("contact_permissions", {}).items()
                          if owner and shape.kind in parts}
                permitted = np.isin(self._unknown_owners[ids], list(owners))
                plane = key.metadata.get("payload_contact_plane")
                if shape.kind == "payload" and plane:
                    permitted &= np.abs((self._unknown_points[ids]-np.asarray(plane["point"]))
                                        @np.asarray(plane["normal"])) <= float(plane["thickness_m"])
                ids = ids[~permitted]
            if not len(ids):
                continue
            points = self._unknown_points[ids]
            def phi(body):
                local = (points-body.pose[:3, 3])@body.pose[:3, :3]
                return np.max(local@body.equations[:, :3].T+body.equations[:, 3], axis=1)
            start, end = phi(prior), phi(shape)
            band = self._unknown_radius[ids]+self.config.unknown_surface_margin_m
            near = np.minimum(start, end) <= band+self._unknown_position_error[ids]+self.config.unknown_proximity_m
            nearby.update(int(i) for i in ids[near])
            existing += int(np.count_nonzero(start <= band))
            # Persistent ambiguity around the starting robot need not disappear
            # at a far endpoint; only newly entering/deepening material matters.
            conflict = (end <= band) & ((start > band)
                | (end < start-self.config.scene_position_tolerance_m))
            for local_index in np.flatnonzero(conflict)[:3]:
                i = int(ids[local_index])
                entered.append({"part": shape.name, "reason": str(self._unknown_reasons[i]),
                    "owner": str(self._unknown_owners[i]), "point_world_m": self._unknown_points[i].tolist(),
                    "surface_residual_before_m": float(start[local_index]),
                    "surface_residual_after_m": float(end[local_index])})
        return {"nearby": len(nearby), "entered": entered[:8], "existing": existing}

    @staticmethod
    def _can_continue(baseline, future):
        if future["safe"]:
            return True
        # Known self collision and joint limits never become world-map escape
        # permissions. Existing external uncertainty overlap may not deepen.
        if any(key == "joint_limit" or key.startswith(("self:", "payload_robot:"))
               for key in future["conflicts"]):
            return False
        return RollingPlanner._nonworsening(baseline, future)

    def _assess(self, action, state, q, qd, key, held, segment):
        delta = action[:3]*TRANSLATION_SCALE_M
        rotation = action[3:6]*ROTATION_SCALE_RAD
        prediction, velocity = self._prediction(state, q, qd, delta, rotation)
        destination = (np.zeros(2) if action[6] > 0 else np.array([.04, -.04])) if action[6] else state[6:8]
        unknown_nearby = 0
        existing = 0
        for index, (future, band) in enumerate(prediction):
            future_fingers = state[6:8]+np.clip(destination-state[6:8], -.003*(index+1), .003*(index+1))
            baseline = self._check(q, state[6:8], key, held, joint_band=band)
            checked = self._check(future, future_fingers, key, held, joint_band=band)
            existing = max(existing, len(baseline["conflicts"]))
            if not self._can_continue(baseline, checked):
                return False, "external_or_robot_constraint", checked, prediction, velocity
            unknown = self._unknown_check(q, future, state[6:8], held, future_fingers, key=key)
            unknown_nearby = max(unknown_nearby, unknown["nearby"])
            if unknown["entered"]:
                return False, "unresolved_swept_region", unknown, prediction, velocity
            if segment.get("keep_upright") and held is not None and held.get("upright_in_gripper") is not None:
                up = self._bound.forward(future)[0][:3, :3]@np.asarray(held["upright_in_gripper"])
                if np.arccos(np.clip(up@np.array([0., 0., 1.]), -1., 1.)) > .10:
                    return False, "held_orientation_constraint", {}, prediction, velocity
        # This is the next jaw increment, even when the arm is holding still.
        next_fingers = state[6:8]+np.clip(destination-state[6:8], -.003, .003)
        start = self._check(q, state[6:8], key, held)
        for fraction in (.5, 1.):
            intermediate_fingers = state[6:8]+fraction*(next_fingers-state[6:8])
            report = self._check(q, intermediate_fingers, key, held)
            if not self._can_continue(start, report):
                return False, "jaw_sweep_constraint", report, prediction, velocity
            unknown = self._unknown_check(q, q, state[6:8], held, intermediate_fingers, key=key)
            if unknown["entered"]:
                return False, "unresolved_jaw_sweep", unknown, prediction, velocity
        if unknown_nearby:
            length = float(np.linalg.norm(delta))
            angle = float(np.linalg.norm(rotation))
            if length > self.config.unknown_command_m+1e-10 or angle > self.config.unknown_rotation_rad+1e-10:
                return False, "unresolved_nearby_slowdown", {"nearby_points": unknown_nearby}, prediction, velocity
        return True, "checked_local_command", {"unknown_nearby_points": unknown_nearby,
            "existing_external_conflicts": existing}, prediction, velocity

    def filter(self, action, view, scene, segment, held=None):
        started = perf_counter()
        returned = None
        self.stats["filter_calls"] += 1
        self._pending_command = None
        def result(status, reason, actual=None, evidence=None):
            nonlocal returned
            self.stats[status] += 1
            scene_evidence = {} if scene is None else scene.evidence
            returned = {"status": status, "reason": reason, "action": actual,
                    "evidence": jsonable({"scope": "actual_next_OSC_command", "reason": reason,
                        "continuous_collision_certified": False, "unknown_space_is_free": False,
                        "scene_identity_counts": scene_evidence.get("identity_counts", {}),
                        "disputed_measurements": len(scene_evidence.get("disputed_points", [])),
                        "disputed_measurements_are_occupancy": False,
                        **(evidence or {})})}
            return returned
        try:
            candidate = np.asarray(action, float)
            if candidate.shape != (7,) or not np.isfinite(candidate).all() or np.max(np.abs(candidate)) > 1.+1e-12:
                return result("candidate_unavailable", "invalid_normalized_action")
            state = np.asarray(view.state, float)
            q = np.asarray(view.observation.joint_position, float)
            qd = getattr(view.observation, "joint_velocity", None)
            if q.shape != (7,) or not np.isfinite(q).all():
                return result("reobserve", "joint_feedback_unavailable")
            self._prepare(scene, state, q)
            checked_held, release_geometry_unavailable = held, False
            try:
                _payload_shape(held, pose_from_state(state))
            except PlanningError as exc:
                requested = segment.get("_candidate") or {}
                stationary_release = (requested.get("skill") == "set_gripper"
                    and requested.get("gripper") == "open" and candidate[6] < 0.
                    and np.linalg.norm(candidate[:6]) <= 1e-6)
                if exc.reason != "payload_extent_unknown" or not stationary_release:
                    raise
                # A model-requested opening is a recovery action, not a carry
                # trajectory. Check its robot/jaw sweep without fabricating an
                # attached box. The unknown payload record is left untouched.
                checked_held, release_geometry_unavailable = None, True
            age = max(0., perf_counter()-scene.captured_at_s)
            if age > self.config.unavailable_scene_s:
                return result("reobserve", "scene_expired", evidence={"scene_age_s": age})
            key = self._contact_key(scene, state, segment, checked_held)
            limit = 1.
            length = float(np.linalg.norm(candidate[:3])*TRANSLATION_SCALE_M)
            if age > self.config.stale_slow_s or self._response_needs_slow:
                limit = min(limit, self.config.unknown_command_m/max(length, 1e-12),
                    self.config.unknown_rotation_rad/max(float(np.linalg.norm(candidate[3:6])*ROTATION_SCALE_RAD), 1e-12))
            # Bound time spent proposing alternatives, not the episode. Rotation
            # sequencing and two tangential choices can salvage local progress.
            options = []
            for scale in (limit, limit*.5, limit*.25):
                option = candidate.copy(); option[:6] *= scale
                options.append((option, "nominal" if scale == 1. else "scaled", scale))
            if np.linalg.norm(candidate[3:6]) > 1e-6 and np.linalg.norm(candidate[:3]) > 1e-6:
                option = candidate.copy(); option[:3] = 0.; option[3:6] *= limit
                options.append((option, "rotation_first", limit))
                option = candidate.copy(); option[3:6] = 0.; option[:3] *= limit
                options.append((option, "translation_first", limit))
            if length > 1e-6:
                direction = candidate[:3]/np.linalg.norm(candidate[:3])
                side = np.cross(direction, [0., 0., 1.])
                if np.linalg.norm(side) < .1:
                    side = np.cross(direction, [0., 1., 0.])
                side /= max(np.linalg.norm(side), 1e-12)
                for sign in (-1., 1.):
                    option = candidate.copy()
                    option[:3] = (direction+sign*side)*np.linalg.norm(candidate[:3])*limit*.5/np.sqrt(2.)
                    option[3:6] *= limit*.5
                    options.append((option, "local_tangent", limit*.5))
            # Include an exact locally slowed proposal: quartering 100 mm is
            # still too large near unknown material. Both rotations and XYZ
            # scale before prediction and before dispatch.
            slow = min(limit, self.config.unknown_command_m/max(length, 1e-12),
                       self.config.unknown_rotation_rad/max(float(np.linalg.norm(candidate[3:6])*ROTATION_SCALE_RAD), 1e-12))
            if slow < limit*.25:
                option = candidate.copy(); option[:6] *= slow
                options.append((option, "uncertainty_slowdown", slow))
            failures = []
            for option, kind, scale in options:
                accepted, reason, detail, prediction, velocity = self._assess(option, state, q, qd, key, checked_held, segment)
                if not accepted:
                    failures.append({"option": kind, "reason": reason, "detail": detail})
                    continue
                actual = tuple(float(value) for value in option)
                self._pending_command = {"state": state.copy(), "action": actual,
                    "predicted_q": prediction[0][0].copy(), "velocity": velocity.copy()}
                evidence = {"variant": kind, "scale": scale, "scene_age_s": age,
                    "scene_revision": scene.evidence.get("revision", scene.frame_id),
                    "prediction_samples": len(prediction), "joint_error_band_rad": self._error.tolist(),
                    "response_model_slowdown": self._response_needs_slow,
                    "release_geometry_unavailable": release_geometry_unavailable,
                    "candidate_action": candidate.tolist(), "accepted_action": list(actual),
                    "external_surface_points": len(self._points), "unresolved_points": len(self._unknown_points),
                    "rejected_options": failures[-2:], **detail}
                return result("execute" if np.array_equal(option, candidate) else "scaled", reason, actual, evidence)
            ambiguous = any(item["reason"].startswith("unresolved") for item in failures)
            return result("reobserve" if ambiguous else "candidate_unavailable",
                          "local_observation_needed" if ambiguous else "local_candidate_blocked",
                          evidence={"rejected_options": failures[-3:]})
        except (PlanningError, ValueError, np.linalg.LinAlgError) as exc:
            return result("reobserve", getattr(exc, "reason", "local_geometry_unavailable"),
                          evidence=getattr(exc, "evidence", {"message": str(exc)}))
        finally:
            elapsed = perf_counter()-started
            self.stats["filter_s"] += elapsed
            status = "error" if returned is None else returned["status"]
            bucket = "accepted" if status in {"execute", "scaled"} else "rejected" if returned is not None else "error"
            self._filter_timing.add(bucket, elapsed)
            self._filter_timing.add("status:"+status, elapsed)
            if returned is not None:
                returned["evidence"].update(filter_wall_s=elapsed, filter_status=status)

    def timing_summary(self):
        return {"latency": self._filter_timing.summary(),
                "status_buckets_are_subsets_of_accepted_rejected_error": True,
                "scope": "all_filter_calls_including_rejected_candidates"}

    def observe_transition(self, before, actual_action, after):
        """Learn only from a matching, actually acknowledged filtered command."""
        pending, self._pending_command = self._pending_command, None
        if pending is None:
            return {"status": "not_a_filtered_command"}
        before = before if isinstance(before, StepView) else StepView(before)
        after = after if isinstance(after, StepView) else StepView(after)
        if tuple(float(v) for v in actual_action) != pending["action"] or not np.array_equal(before.state, pending["state"]):
            raise ValueError("filter feedback must match the exact dispatched command and source state")
        actual_q = np.asarray(after.observation.joint_position, float)
        residual = np.abs(actual_q-pending["predicted_q"])
        valid = bool(np.isfinite(residual).all() and np.max(residual) <= self.config.joint_prediction_limit_rad)
        if valid:
            self._error = np.maximum(self.config.joint_error_floor_rad, np.maximum(.95*self._error, residual))
            self._last_velocity = pending["velocity"]
            self._response_needs_slow = False
        else:
            self._last_velocity = np.zeros(6)
            self._error = np.maximum(self._error, self.config.joint_error_floor_rad)
            self._response_needs_slow = True
        self.stats["dispatched_commands"] += 1
        self._previous_receipt = {"status": "updated" if valid else "prediction_mismatch",
                                  "joint_residual_rad": residual.tolist()}
        return self._previous_receipt
