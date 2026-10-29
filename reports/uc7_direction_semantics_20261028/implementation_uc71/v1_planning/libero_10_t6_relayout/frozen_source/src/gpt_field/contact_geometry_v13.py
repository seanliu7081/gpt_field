"""Part-specific target contact constraints from RGB model hypotheses.

Known ROBOT collision envelopes constrain every target, independently of the
ordinary obstacle list. Target boxes are model estimates, never measured scene
geometry, collision probabilities, or certificates. This module imports no
simulator, task state, camera calibration, depth or evaluator output.
"""
from __future__ import annotations

import copy
import math
import itertools
from numbers import Real

import numpy as np

from .robot_tool_geometry import world_aabbs

FINGER_CONTACT_PARTS = frozenset(("finger1_collision", "finger2_collision",
                                "finger1_pad_collision", "finger2_pad_collision"))
MODEL_EXTENT_PART = "model_declared_tool_or_payload_extent"
HAND_PART = "hand_collision"


def _vector(value, count, name):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (count,) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw):
        raise ValueError(f"{name} must contain {count} finite real numbers")
    result = np.asarray(raw, float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {count} finite real numbers")
    return result


def _number(value, minimum, maximum, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite in [{minimum}, {maximum}]")
    return float(value)


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _inside(point, lower, upper):
    return bool(np.all(point >= lower-1e-12) and np.all(point <= upper+1e-12))


def _segment_intersection(start, end, lower, upper):
    """Slab intersection of the complete closed segment, including endpoints."""
    delta = end-start
    enter, leave = 0., 1.
    for axis in range(3):
        if abs(delta[axis]) <= 1e-15:
            if start[axis] < lower[axis]-1e-12 or start[axis] > upper[axis]+1e-12:
                return None
            continue
        times = sorted(((lower[axis]-start[axis])/delta[axis],
                        (upper[axis]-start[axis])/delta[axis]))
        enter, leave = max(enter, times[0]), min(leave, times[1])
        if enter > leave+1e-12:
            return None
    return [max(0., enter), min(1., leave)]


def _nearest_face_exit(start, end, lower, upper):
    """Explicit recovery: leave an already-overlapped box through a nearest face.

    This is a separating model-box hypothesis, not proof that any physical
    contact exists or that its friction/contact dynamics will be safe.
    """
    if not _inside(start, lower, upper) or _inside(end, lower, upper):
        return False
    distances = np.r_[start-lower, upper-start]
    nearest = float(np.min(distances))
    return any(distances[i] <= nearest+1e-12 and
               ((end[i] < lower[i]-1e-12) if i < 3 else (end[i-3] > upper[i-3]+1e-12))
               for i in range(6))


class TargetContactGeometry:
    """Anchor an explicitly named target and preserve it across missing updates.

    configure accepts only robot feedback and a model-produced target region.
    Geometry-null is unavailable, not free space: previously named world boxes
    remain exposed with contact exemptions revoked. No local task-success flag
    is ever produced. The caller must put returned boxes into its planner, and
    use the same checker for Direct/gripper actions.
    """
    VERSION = "part-specific-model-target-contact-v13"
    DEFAULT_ORIENTATION_MARGIN_RAD = .04
    HOLD_RESET_RAD = 1e-9

    def __init__(self):
        self.reset()

    def reset(self):
        self._known = {}
        self.target_id = None
        self.available = False
        self.payload_observed = False
        self.tool_radius_m = .045
        self._record = None
        self._last_state = None
        self._grasp_candidates = {}
        self._carried_payload_id = None

    def configure(self, state8, geometry, *, target_id, tool_radius_m=.045, payload_observed=False):
        state = _vector(state8, 8, "state8")
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("target_id must be a nonempty string")
        radius = _number(tool_radius_m, 0., .5, "tool_radius_m")
        payload = _boolean(payload_observed, "payload_observed")
        record = None
        if geometry is not None:
            keys = {"id", "center_offset_m", "half_extents_m", "uncertainty_m", "allowed_contact"}
            if not isinstance(geometry, dict) or set(geometry) != keys:
                raise ValueError("target_geometry has missing or unexpected fields")
            if geometry["id"] != target_id:
                raise ValueError("target_geometry.id must match target_id")
            center = _vector(geometry["center_offset_m"], 3, "center_offset_m")
            half = _vector(geometry["half_extents_m"], 3, "half_extents_m")
            if np.any(abs(center) > .5) or np.any(half < .0001) or np.any(half > .5):
                raise ValueError("target center must be within +/-0.5m and half extents in [0.0001,0.5]m")
            uncertainty = _number(geometry["uncertainty_m"], 0., .1, "uncertainty_m")
            contact = geometry["allowed_contact"]
            if contact not in ("none", "fingers", "payload_support"):
                raise ValueError("allowed_contact must be none, fingers or payload_support")
            world_center = state[:3]+center
            if not np.isfinite(np.r_[world_center-half, world_center+half]).all():
                raise ValueError("target world bounds must remain finite")
            record = {"geometry": copy.deepcopy(geometry), "origin_xyz": state[:3].tolist(),
                      "lower_xyz": (world_center-half).tolist(), "upper_xyz": (world_center+half).tolist(),
                      "uncertainty_m": uncertainty}
        # Commit only after full validation, so rejected input cannot delete a
        # previously known region or replace its current contact permissions.
        if record is not None:
            self._known[target_id] = record
        self.target_id, self.tool_radius_m = target_id, radius
        self.available, self.payload_observed = record is not None, payload
        self._record = copy.deepcopy(self._known.get(target_id))
        self._last_state = state.copy()
        return self.configuration(state)

    def configuration(self, state8, *, orientation_margin_rad=.04, full_finger_range=True, payload_observed=None):
        state = _vector(state8, 8, "state8")
        angle = _number(orientation_margin_rad, 0., math.pi, "orientation_margin_rad")
        fingers = _boolean(full_finger_range, "full_finger_range")
        payload = self.payload_observed if payload_observed is None else _boolean(payload_observed, "payload_observed")
        record = self._record
        declared = None if record is None else record["geometry"]["allowed_contact"]
        contact = declared if self.available else "none"
        exempt = sorted(FINGER_CONTACT_PARTS) if contact == "fingers" else []
        payload_exempt = contact == "payload_support" and payload
        model_extent = contact != "fingers" and not payload_exempt
        result = {"version": self.VERSION, "available": self.available, "retained": record is not None and not self.available,
                  "target_id": self.target_id, "allowed_contact": declared, "effective_allowed_contact": contact,
                  "nominal": None, "core_parts": [], "uncertainty_halo_parts": [],
                  "tool_core_parts": [], "full_halo_parts": [], "sigma_m": None,
                  "exempt_parts": exempt, "model_extent_included": model_extent,
                  "payload_observed": payload, "payload_support_exemption_applied": payload_exempt,
                  "orientation_margin_rad": angle, "full_finger_range": fingers,
                  "world_bounds_fixed_until_explicit_model_revision": True,
                  "geometry_is_model_hypothesis": True, "clearance_certified": False,
                  "task_success_certified": False, "environment_inputs": [],
                  "full_arm_included": False, "union_bounds_are_display_only": True,
                  "uncertainty_semantics": "separate nonnegative model margin, never erased by contact permission"}
        if record is None:
            result["reason"] = "target_geometry_unavailable"
            return result
        low, high = np.asarray(record["lower_xyz"]), np.asarray(record["upper_xyz"])
        sigma = record["uncertainty_m"]
        parts = world_aabbs(state, orientation_margin_rad=angle, full_finger_range=fingers)
        offsets = [(p["name"], np.asarray(p["offset_min_xyz"]), np.asarray(p["offset_max_xyz"]))
                   for p in parts if p["name"] not in exempt]
        if model_extent:
            offsets.append((MODEL_EXTENT_PART, np.full(3, -self.tool_radius_m), np.full(3, self.tool_radius_m)))
        cores, halos = [], []
        for name, part_low, part_high in offsets:
            core_low, core_high = low-part_high, high-part_low
            cores.append({"tool_part": name, "lower_xyz": core_low.tolist(), "upper_xyz": core_high.tolist()})
            halos.append({"tool_part": name, "lower_xyz": (core_low-sigma).tolist(), "upper_xyz": (core_high+sigma).tolist()})
        assert HAND_PART in {p["tool_part"] for p in cores}
        result.update(nominal={"lower_xyz": low.tolist(), "upper_xyz": high.tolist()},
                      core_parts=cores, uncertainty_halo_parts=halos,
                      tool_core_parts=copy.deepcopy(cores), full_halo_parts=copy.deepcopy(halos), sigma_m=sigma,
                      reason="current_model_target_geometry" if self.available else "retained_target_geometry_unavailable")
        if declared == "payload_support" and not payload:
            result["payload_support_constraint"] = "coarse_extent_retained_without_observed_payload"
        return result

    @staticmethod
    def _rotation(rotvec):
        angle = float(np.linalg.norm(rotvec))
        if angle <= 1e-15:
            return np.eye(3)
        x, y, z = np.asarray(rotvec)/angle
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        return np.eye(3)+math.sin(angle)*skew+(1.-math.cos(angle))*(skew@skew)

    def remember_grasp_candidate(self, state8, payload_id):
        """Snapshot a CURRENT model box before the first accepted closure.

        Repeated closing actions cannot move this candidate. No payload starts
        following the robot until independent observed attachment activates it.
        """
        state = _vector(state8, 8, "state8")
        if not isinstance(payload_id, str) or not payload_id.strip():
            raise ValueError("payload_id must be nonempty")
        if payload_id in self._grasp_candidates:
            return {"accepted": True, "reason": "existing_grasp_candidate_retained", "payload_id": payload_id}
        if not self.available or self.target_id != payload_id or self._record is None:
            return {"accepted": False, "reason": "grasp_candidate_requires_current_target_geometry", "payload_id": payload_id}
        record = self._record
        corners = np.array(list(itertools.product(*zip(record["lower_xyz"], record["upper_xyz"]))))
        local = (corners-state[:3])@self._rotation(state[3:6])
        self._grasp_candidates[payload_id] = {
            "payload_id": payload_id, "snapshot_state8": state.tolist(),
            "source_nominal_world": {"lower_xyz": list(record["lower_xyz"]), "upper_xyz": list(record["upper_xyz"])},
            "source_uncertainty_m": record["uncertainty_m"],
            "local_corners_relative_to_eef": local.tolist(),
            "attachment_observed": False, "confirmed_state8": None}
        return {"accepted": True, "reason": "unconfirmed_grasp_candidate_saved", "payload_id": payload_id}

    def confirm_payload_attachment(self, state8, payload_id, *, observed=True):
        state = _vector(state8, 8, "state8")
        evidence = _boolean(observed, "observed")
        if not isinstance(payload_id, str) or not payload_id.strip():
            raise ValueError("payload_id must be nonempty")
        if not evidence:
            return {"accepted": False, "reason": "attachment_not_observed_constraint_retained", "payload_id": payload_id}
        if payload_id not in self._grasp_candidates:
            return {"accepted": False, "reason": "attachment_missing_preclosure_candidate", "payload_id": payload_id}
        if self._carried_payload_id not in (None, payload_id):
            return {"accepted": False, "reason": "different_carried_payload_requires_observed_release", "payload_id": payload_id}
        self._carried_payload_id = payload_id
        self._grasp_candidates[payload_id]["attachment_observed"] = True
        self._grasp_candidates[payload_id]["confirmed_state8"] = state.tolist()
        self._last_state = state.copy()
        return {"accepted": True, "reason": "observed_payload_candidate_activated", "payload_id": payload_id}

    def cancel_unconfirmed_candidate(self, payload_id):
        """Caller requires actual reopening/aperture increase, not just intent."""
        if payload_id == self._carried_payload_id:
            return {"accepted": False, "reason": "active_payload_requires_observed_separation", "payload_id": payload_id}
        removed = self._grasp_candidates.pop(payload_id, None) is not None
        return {"accepted": removed, "reason": "unconfirmed_candidate_cancelled" if removed else "candidate_absent", "payload_id": payload_id}

    def release_payload(self, payload_id, *, observed_separation=False):
        """Clear only with observed separation; an open command is insufficient."""
        if not _boolean(observed_separation, "observed_separation"):
            return {"accepted": False, "reason": "payload_constraint_retained_without_observed_separation", "payload_id": payload_id}
        removed = self._grasp_candidates.pop(payload_id, None) is not None
        if self._carried_payload_id == payload_id:
            self._carried_payload_id = None
        return {"accepted": removed, "reason": "observed_payload_separation" if removed else "candidate_absent", "payload_id": payload_id}

    def carried_summary(self, state8=None):
        state = self._last_state if state8 is None else _vector(state8, 8, "state8")
        identity = self._carried_payload_id
        out = {"active": identity is not None, "payload_id": identity,
               "candidate_ids": sorted(self._grasp_candidates),
               "source": "preclosure_model_geometry_activated_only_by_observed_attachment",
               "static_world_boxes_moved": False, "clearance_certified": False,
               "task_success_certified": False, "attachment_uncertainty_retains_constraint": True}
        if identity is None or state is None:
            return out
        candidate = self._grasp_candidates[identity]
        local = np.asarray(candidate["local_corners_relative_to_eef"])
        world = local@self._rotation(state[3:6]).T+state[:3]
        out.update(snapshot_state8=copy.deepcopy(candidate["snapshot_state8"]),
                   confirmed_state8=copy.deepcopy(candidate["confirmed_state8"]),
                   source_nominal_world=copy.deepcopy(candidate["source_nominal_world"]),
                   nominal_world={"lower_xyz": world.min(axis=0).tolist(), "upper_xyz": world.max(axis=0).tolist()},
                   nominal_hand_frame={"lower_xyz": local.min(axis=0).tolist(), "upper_xyz": local.max(axis=0).tolist()},
                   uncertainty_m=candidate["source_uncertainty_m"],
                   rigid_attachment_is_hypothesis=True,
                   forbidden_robot_parts=[HAND_PART], intended_contact_parts=sorted(FINGER_CONTACT_PARTS))
        return out

    def assess_carried_payload(self, state8, action7):
        """Check carried-object/palm overlap in their shared rigid hand frame.

        Whole-body translation or rotation cannot cure a relative self-overlap.
        The static target planner remains separate; this is not a world obstacle
        following the EEF, nor a certificate for payload/environment clearance.
        """
        state, action = _vector(state8, 8, "state8"), _vector(action7, 7, "action7")
        if np.any(abs(action) > 1.):
            raise ValueError("action7 must be normalized within [-1,1]")
        self._last_state = state.copy()
        carried = self.carried_summary(state)
        delta = action[:3]*.05
        rotation = float(np.linalg.norm(action[3:6])*.5)
        stationary = np.linalg.norm(delta) <= 1e-12 and rotation <= self.HOLD_RESET_RAD+1e-15
        hold, opening = bool(stationary and action[6] == 0.), bool(stationary and action[6] < 0.)
        hits = []
        if carried["active"]:
            # Hand and attached payload share this transform; do not inflate
            # them as if only the robot rotated while the payload stayed still.
            local_state = np.r_[np.zeros(6), state[6:]]
            hand = next(p for p in world_aabbs(local_state, full_finger_range=True) if p["name"] == HAND_PART)
            low, high = np.asarray(carried["nominal_hand_frame"]["lower_xyz"]), np.asarray(carried["nominal_hand_frame"]["upper_xyz"])
            hlow, hhigh = np.asarray(hand["min_xyz"]), np.asarray(hand["max_xyz"])
            if np.all(low <= hhigh+1e-12) and np.all(high >= hlow-1e-12):
                hits.append(HAND_PART)
        accepted = not hits or hold or opening
        reason = ("carried_payload_pose_hold_only" if hits and hold else
                  "carried_payload_opening_only" if hits and opening else
                  "carried_payload_forbidden_palm_intersection" if hits else
                  "carried_payload_relative_geometry_admitted" if carried["active"] else
                  "no_observed_carried_payload")
        return {"accepted": accepted, "reason": reason, "need_refresh": not accepted,
                "action": action.tolist(), "diagnostics": {"carried_payload": carried,
                    "forbidden_part_intersections": hits, "hold_only": hold, "opening_only": opening,
                    "relative_geometry_checked_for_entire_rigid_motion": True,
                    "uncertainty_retained_separately_m": carried.get("uncertainty_m"),
                    "command_modified": False, "clearance_certified": False, "task_success_certified": False}}

    def summary(self):
        if self._last_state is not None:
            return {**self.configuration(self._last_state), "carried_payload": self.carried_summary(self._last_state)}
        return {"version": self.VERSION, "available": False, "retained": False,
                "target_id": None, "allowed_contact": None, "effective_allowed_contact": "none",
                "nominal": None, "core_parts": [], "uncertainty_halo_parts": [],
                "tool_core_parts": [], "full_halo_parts": [], "sigma_m": None,
                "reason": "target_geometry_not_configured", "clearance_certified": False,
                "task_success_certified": False, "geometry_is_model_hypothesis": True,
                "environment_inputs": [], "full_arm_included": False, "carried_payload": self.carried_summary()}

    def assess(self, state8, action7, *, allow_separating=False):
        state = _vector(state8, 8, "state8")
        action = _vector(action7, 7, "action7")
        if np.any(abs(action) > 1.):
            raise ValueError("action7 must be normalized within [-1,1]")
        separating = _boolean(allow_separating, "allow_separating")
        self._last_state = state.copy()
        delta, rotation = action[:3]*.05, float(np.linalg.norm(action[3:6])*.5)
        # Includes the complete commanded SO(3) arc, not merely its endpoint.
        margin = min(math.pi, max(self.DEFAULT_ORIENTATION_MARGIN_RAD, rotation))
        config = self.configuration(state, orientation_margin_rad=margin, full_finger_range=True)
        start, end = state[:3], state[:3]+delta
        intersections = []
        for part in config["core_parts"]:
            low, high = np.asarray(part["lower_xyz"]), np.asarray(part["upper_xyz"])
            interval = _segment_intersection(start, end, low, high)
            if interval is not None:
                exits = separating and rotation <= self.HOLD_RESET_RAD+1e-15 and action[6] == 0. and _nearest_face_exit(start, end, low, high)
                intersections.append({"tool_part": part["tool_part"], "segment_fraction": interval,
                    "start_inside": _inside(start, low, high), "end_inside": _inside(end, low, high),
                    "explicit_nearest_face_separation": bool(exits)})
        # A refusal must retain a way to issue the exact nonmoving zero-grip
        # measured-pose reset; it does not become a clearance assertion.
        hold = bool(np.linalg.norm(delta) <= 1e-12 and rotation <= self.HOLD_RESET_RAD+1e-15 and action[6] == 0.)
        opening = bool(not self.available and np.linalg.norm(delta) <= 1e-12
                       and rotation <= self.HOLD_RESET_RAD+1e-15 and action[6] < 0.)
        if hold:
            accepted, reason = True, "target_geometry_pose_hold_only"
        elif opening:
            accepted, reason = True, "target_geometry_unavailable_opening_only"
        elif not self.available:
            accepted, reason = False, "target_geometry_unavailable"
        elif not intersections:
            accepted, reason = True, "target_nominal_segment_admitted"
        elif separating and all(p["explicit_nearest_face_separation"] for p in intersections):
            accepted, reason = True, "target_nominal_separating_recovery"
        else:
            accepted, reason = False, "target_forbidden_part_intersection"
        return {"accepted": accepted, "reason": reason, "need_refresh": not accepted,
                "action": action.tolist(), "diagnostics": {"configuration": config,
                    "command_delta_m": delta.tolist(), "rotation_bound_rad": margin,
                    "intersections": intersections, "hold_only": hold, "opening_only": opening,
                    "continuous_commanded_segment_checked": True, "command_modified": False,
                    "clearance_certified": False, "task_success_certified": False}}
