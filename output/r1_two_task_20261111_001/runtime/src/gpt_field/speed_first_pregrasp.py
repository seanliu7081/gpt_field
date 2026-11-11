"""Optional uncalibrated pregrasp feedback from independently usable RGB views.

Image features are semantic hypotheses, not metric geometry or contact evidence.
Natural measured robot translations learn per-view response models. Only held-out
validated observable directions can revise an existing Field goal, by at most
6 mm total. This module never chooses actions, probes, closes or certifies success.
"""
from __future__ import annotations

import copy
import math
from collections import deque

import numpy as np

from .action import _rotvec_quaternion
from .observation import PolicyObservation, decode_image

PREGRASP_VERSION = "speed-first-pregrasp-rgb-v2"
_REFERENCE_KEYS = {"reference_id", "target_id", "external", "wrist"}
_VIEW_KEYS = {"tool_contact_uv", "target_contact_uv", "tool_roi_uv", "target_roi_uv", "desired_offset_uv"}
_VIEWS = ("external", "wrist")


def pregrasp_reference_schema():
    point = {"type": "array", "minItems": 2, "maxItems": 2,
             "items": {"type": "number", "minimum": 0., "maximum": 1.}}
    box = {"type": "array", "minItems": 4, "maxItems": 4,
           "items": {"type": "number", "minimum": 0., "maximum": 1.}}
    view = {"type": "object", "additionalProperties": False,
            "required": sorted(_VIEW_KEYS | {"tool_reference_mode"}),
            "properties": {"tool_contact_uv": copy.deepcopy(point), "target_contact_uv": copy.deepcopy(point),
                "tool_roi_uv": copy.deepcopy(box), "target_roi_uv": copy.deepcopy(box),
                "tool_reference_mode": {"type": "string", "enum": ["tracked_material", "fixed_wrist"]},
                "desired_offset_uv": {"type": "array", "minItems": 2, "maxItems": 2,
                    "items": {"type": "number", "minimum": -.25, "maximum": .25}}}}
    external = copy.deepcopy(view)
    external["properties"]["tool_reference_mode"]["enum"] = ["tracked_material"]
    return {"anyOf": [{"type": "null"}, {"type": "object", "additionalProperties": False,
        "required": sorted(_REFERENCE_KEYS), "properties": {
            "reference_id": {"type": "string", "minLength": 1, "maxLength": 64},
            "target_id": {"type": "string", "minLength": 1, "maxLength": 64},
            "external": {"anyOf": [{"type": "null"}, external]},
            "wrist": {"anyOf": [{"type": "null"}, view]}}}]}


def _numbers(value, count, name, low=None, high=None):
    if (not isinstance(value, (list, tuple, np.ndarray)) or len(value) != count
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number))
                   or not math.isfinite(float(v)) for v in value)):
        raise ValueError(f"{name} requires {count} finite numbers")
    result = np.asarray(value, dtype=float)
    if result.shape != (count,) or (low is not None and np.any(result < low)) or (high is not None and np.any(result > high)):
        raise ValueError(f"{name} outside its permitted range")
    return result


def validate_pregrasp_reference(reference):
    if reference is None:
        return None
    if not isinstance(reference, dict) or set(reference) != _REFERENCE_KEYS:
        raise ValueError("pregrasp_reference has missing or unexpected fields")
    result = copy.deepcopy(reference)
    for key in ("reference_id", "target_id"):
        if not isinstance(reference[key], str) or not reference[key].strip() or len(reference[key]) > 64:
            raise ValueError(f"{key} must be nonempty text of at most 64 characters")
    if all(reference[view] is None for view in _VIEWS):
        raise ValueError("at least one pregrasp view must be provided")
    for view in _VIEWS:
        value = reference[view]
        if value is None:
            continue
        if not isinstance(value, dict) or set(value) not in (_VIEW_KEYS, _VIEW_KEYS | {"tool_reference_mode"}):
            raise ValueError(f"{view} has missing or unexpected pregrasp fields")
        mode = value.get("tool_reference_mode", "tracked_material")
        if mode not in ("tracked_material", "fixed_wrist") or (view == "external" and mode != "tracked_material"):
            raise ValueError("fixed_wrist is permitted only in the wrist view")
        for key in _VIEW_KEYS:
            result[view][key] = _numbers(value[key], 4 if "roi" in key else 2, key,
                -.25 if key == "desired_offset_uv" else 0., .25 if key == "desired_offset_uv" else 1.).tolist()
        for role in ("tool", "target"):
            roi = np.asarray(result[view][role + "_roi_uv"])
            point = np.asarray(result[view][role + "_contact_uv"])
            outside = np.maximum(roi[:2] - point, 0.) + np.maximum(point - roi[2:], 0.)
            if np.any(roi[:2] >= roi[2:]) or np.linalg.norm(outside) > .125:
                raise ValueError("visible contact needs an ordered nearby same-body support ROI")
    return result


def pregrasp_instructions():
    return (
        f"Optional pregrasp feedback {PREGRASP_VERSION}. During open-gripper phase=align, "
        "control_context.pregrasp_reference may describe visible tool-to-object alignment "
        "in either or both current RGB views; unavailable external/wrist views are null, "
        "and at least one must be nonnull. Otherwise set the entire reference null. "
        "A nonnull reference requires phase=align, visual_reference=null and gripper_operation=null. "
        "For useful free-space OPEN-gripper pregrasp alignment, coarse_field with phase=align is "
        "appropriate; ordinary Direct remains for uncertain/contact interaction. Supply a "
        "reference when at least one usable visible view exists. Field must use "
        "gripper_command=open; ordinary Direct must keep the gripper negative/open "
        "in all five actions. If unavailable, briefly explain in routing_reason and choose useful "
        "normal motion. The correction is at most 6 mm TOTAL from the immutable nominal goal; "
        "the existing 30-action Field refresh limit is unchanged. Use reference_id,target_id and "
        "external/wrist, each nonnull view with tool_contact_uv,target_contact_uv,tool_roi_uv,"
        "target_roi_uv,desired_offset_uv,tool_reference_mode. Coordinates are normalized [u,v], "
        "top-left origin; ROIs are [u_min,v_min,u_max,v_max]. Initial contact points must be "
        "VISIBLE MATERIAL points: a finger/pad/tool point and the intended physical target "
        "contact location. Never initialize an invisible pinch midpoint, hidden point, empty "
        "gap or arbitrary centroid. Support ROIs contain only the SAME rigid body's visible "
        "texture/edge features. A nearby support can exclude the visible contact point (at most "
        "24 image pixels away); its transported contact is explicitly a local inference with "
        "uncertainty, not a measured contact or a globally fixed pixel offset. A bounded local "
        "2D affine approximation can transport visible nearby supports; it supplies no 3D pose. Do not include "
        "background or the other body. Choose tracked_material for the external view. In the "
        "wrist view, fixed_wrist may retain an initially visible finger reference because the "
        "camera moves with the hand. This is your visible semantic claim, not robot geometry "
        "or a detector certification. It expires after 12 observations without a valid texture "
        "check, with increasing uncertainty; rotation/opening changes invalidate the epoch. "
        "Use tracked_material in the wrist view when material texture is clear. The two views "
        "describe the same intended interaction. desired_offset_uv is desired TOOL-MINUS-TARGET "
        "projection relation, not automatically zero. Choose ordinary orientation first; use "
        "a useful short free-space alignment goal, typically within 3 cm and inside this epoch's "
        "4 cm local domain. Keep orientation and open finger width fixed. A new reference_id "
        "is required after pose, target, desired relation or contact changes, or to reseed lost "
        "views. Prefer retaining the SAME reference_id while the semantic intent is unchanged; "
        "do not restart a valid fit simply because another ordinary model call occurred. "
        "With the same ID retain the objective and report current transported points "
        "and support ROIs (a fixed_wrist tool point and ROI remain at their seed). Views learn "
        "and validate independently: one lost view does not discard a usable other view, but "
        "missing image directions never count as validated. Natural measured XYZ motion learns "
        "a local response; only observable, held-out-validated directions can be corrected. "
        "No calibrated camera, depth or metric object pose is supplied. Visible target drift "
        "in the fixed view invalidates BOTH stationary-target fits. Without that view, target "
        "motion cannot always be separated from robot-induced image motion; wrist-only "
        "feedback never proves the object remained stationary. Insufficient texture, motion "
        "or prediction quality only disables local correction. Keep choosing normal useful "
        "actions instead of probes or waiting for validation. There are no extra model calls, "
        "automatic closure, action vetoes or task-completion claims. Image alignment never "
        "proves 3D contact or a grasp."
    )


def _orientation_distance(first, second):
    qa, qb = _rotvec_quaternion(np.asarray(first)), _rotvec_quaternion(np.asarray(second))
    return 2 * math.acos(float(np.clip(abs(np.dot(qa, qb)), 0., 1.)))


def _visible_seed(image, point):
    """Weak image support for the model's visible claim; not semantic verification."""
    x, y = np.rint(point).astype(int)
    crop = image[max(0, y-3):y+4, max(0, x-3):x+4]
    return crop.size >= 16 and float(np.std(crop)) >= 4. and float(np.ptp(crop)) >= 15.


class _FrozenSupport:
    """Adaptive frozen NCC body supports; off-support contacts are local inferences."""
    radius = 5
    search = 24

    def __init__(self, image, point_uv, roi_uv):
        self.shape = image.shape
        self.scale = np.array([image.shape[1] - 1, image.shape[0] - 1], dtype=float)
        self.seed_point = np.asarray(point_uv) * self.scale
        self.point = self.seed_point.copy()
        self.roi = np.asarray(roi_uv).reshape(2, 2) * self.scale
        self.translation = np.zeros(2)
        outside = np.maximum(self.roi[0] - self.seed_point, 0.) + np.maximum(self.seed_point - self.roi[1], 0.)
        self.support_distance = float(np.linalg.norm(outside))
        if self.support_distance > 24.:
            raise ValueError("same_body_support_too_far_from_contact")
        self.inferred = self.support_distance > .5
        if self.inferred and not _visible_seed(image, self.seed_point):
            raise ValueError("initial_auxiliary_contact_has_no_visible_image_support")
        width = float(np.min(self.roi[1] - self.roi[0]))
        self.radius = 5 if width >= 22 else (3 if width >= 13 else 2)
        self.uncertainty_px = 1.
        self.source = "local_same_body_support_inference" if self.inferred else "material_contact_texture_transport"
        self.patches = []
        r = self.radius
        lo = np.maximum(np.ceil(self.roi[0]).astype(int) + r, r)
        hi = np.minimum(np.floor(self.roi[1]).astype(int) - r, np.array(image.shape[::-1]) - r - 1)
        if np.any(hi - lo < 2):
            raise ValueError("body_roi_too_small_for_texture")
        crop = image[lo[1]-r:hi[1]+r+1, lo[0]-r:hi[0]+r+1]
        candidates = np.lib.stride_tricks.sliding_window_view(crop, (2*r+1, 2*r+1))
        texture = candidates.std(axis=(-2, -1))
        for attempt, flat in enumerate(np.argsort(texture, axis=None)[::-1]):
            if attempt >= 96:
                break
            row, col = np.unravel_index(flat, texture.shape)
            if texture[row, col] < 5. or len(self.patches) == 4:
                break
            center = lo + np.array([col, row])
            if any(np.linalg.norm(center - patch["seed"]) < max(3., 2*r) for patch in self.patches):
                continue
            patch = candidates[row, col].astype(float).copy()
            centered = patch - patch.mean()
            candidate = {"seed": center.astype(float), "position": center.astype(float),
                         "template": centered, "norm": float(np.linalg.norm(centered)), "radius": r}
            # Candidate uniqueness is tested before admitting it. One repeated
            # edge must not discard an otherwise useful body ROI.
            try:
                self._match_one(image, candidate)
            except ValueError:
                continue
            candidate["variants"] = []
            for factor in (.94, 1.06):
                vr = int(math.floor(r*factor))
                grid = np.arange(-vr, vr+1, dtype=float)/factor+r
                # Every sample is inside the original declared-body patch.
                lo_grid = np.floor(grid).astype(int)
                hi_grid = np.minimum(lo_grid+1, 2*r)
                fraction = grid-lo_grid
                horizontal = patch[:, lo_grid]*(1.-fraction)+patch[:, hi_grid]*fraction
                warped = horizontal[lo_grid, :]*(1.-fraction[:, None])+horizontal[hi_grid, :]*fraction[:, None]
                warped -= warped.mean()
                candidate["variants"].append({"template": warped, "norm": float(np.linalg.norm(warped)), "radius": vr})
            self.patches.append(candidate)
        if len(self.patches) < 2:
            raise ValueError("insufficient_frozen_body_texture")
        self.update(image)

    def _match(self, image, patch):
        try:
            return self._match_one(image, patch)
        except ValueError as original:
            for variant in patch.get("variants", []):
                try:
                    return self._match_one(image, {**patch, **variant})
                except ValueError:
                    pass
            raise original

    def _match_one(self, image, patch):
        r, search = patch.get("radius", self.radius), self.search
        x, y = np.rint(patch["position"]).astype(int)
        low = np.maximum([r, r], np.array([x, y]) - search)
        high = np.minimum(np.array(image.shape[::-1]) - r - 1, np.array([x, y]) + search)
        if np.any(high <= low):
            raise ValueError("texture_out_of_view")
        crop = image[low[1]-r:high[1]+r+1, low[0]-r:high[0]+r+1]
        windows = np.lib.stride_tricks.sliding_window_view(crop.astype(float), (2*r+1, 2*r+1))
        means = windows.mean(axis=(-2, -1))
        energy = np.maximum(np.sum(windows*windows, axis=(-2, -1)) - means*means*patch["template"].size, 0.)
        denom = np.sqrt(energy) * patch["norm"]
        numer = np.einsum("ijkl,kl->ij", windows, patch["template"], optimize=False)
        scores = np.divide(numer, denom, out=np.full(means.shape, -1.), where=denom > 1e-10)
        row, col = np.unravel_index(int(np.argmax(scores)), scores.shape)
        score = float(scores[row, col])
        if not math.isfinite(score) or score < .75:
            raise ValueError("texture_match_lost")
        alternatives = scores.copy()
        alternatives[max(0, row-2):row+3, max(0, col-2):col+3] = -1.
        if score - float(alternatives.max()) < .05:
            raise ValueError("ambiguous_texture_match")
        if row in (0, scores.shape[0]-1) or col in (0, scores.shape[1]-1):
            raise ValueError("texture_search_boundary")
        sub = np.zeros(2)
        for axis, minus, plus in ((0, scores[row, col-1], scores[row, col+1]),
                                  (1, scores[row-1, col], scores[row+1, col])):
            curvature = float(minus - 2*score + plus)
            if curvature < -1e-6:
                sub[axis] = np.clip(.5 * float(minus - plus) / curvature, -.5, .5)
        return low + np.array([col, row], float) + sub

    def update(self, image):
        if image.shape != self.shape:
            raise ValueError("image_shape_changed")
        matched = []
        for patch in self.patches:
            try:
                point = self._match(image, patch)
            except ValueError:
                continue
            matched.append((patch, point, point - patch["seed"]))
        if len(matched) < 2 or len(matched) < math.ceil(.75 * len(self.patches)):
            raise ValueError("body_texture_lost_or_ambiguous")
        shifts = np.array([row[2] for row in matched])
        shift = np.median(shifts, axis=0)
        residual = np.linalg.norm(shifts-shift, axis=1)
        seed_points = np.array([row[0]["seed"] for row in matched])
        current_points = np.array([row[1] for row in matched])
        center = seed_points.mean(axis=0)
        design = np.c_[seed_points-center, np.ones(len(matched))]
        spatial = np.linalg.svd(seed_points-center, compute_uv=False)
        affine_ok = len(matched) >= 3 and len(spatial) == 2 and spatial[-1] > 1.5 and spatial[0]/spatial[-1] < 15.
        inferred_extra = 0.
        semantic = self.seed_point+shift
        if affine_ok:
            transform = np.linalg.lstsq(design, current_points, rcond=None)[0]
            linear = transform[:2, :]
            singular = np.linalg.svd(linear, compute_uv=False)
            rotation = math.atan2(float(linear[0, 1]-linear[1, 0]), float(np.trace(linear)))
            residual = np.linalg.norm(design@transform-current_points, axis=1)
            if (np.linalg.det(linear) <= 0. or singular.min() < .85 or singular.max() > 1.15
                    or singular.max()/singular.min() > 1.2 or abs(rotation) > .12
                    or float(residual.max()) > .9):
                raise ValueError("body_support_inconsistent_or_deformed")
            # Leave-one-support-out checks when the remaining geometry permits
            # them prevent an affine model from hiding a single bad feature.
            heldout = []
            if len(matched) >= 4:
                for index in range(len(matched)):
                    mask = np.arange(len(matched)) != index
                    if np.linalg.matrix_rank(design[mask]) == 3:
                        fitted = np.linalg.lstsq(design[mask], current_points[mask], rcond=None)[0]
                        heldout.append(float(np.linalg.norm(design[index]@fitted-current_points[index])))
                if heldout and max(heldout) > 1.75:
                    raise ValueError("body_support_affine_heldout_inconsistent")
            inferred_extra = (.35 if not heldout else .1) + .025*float(np.linalg.norm(self.seed_point-center))*float(np.linalg.norm(linear-np.eye(2)))
            semantic = np.r_[self.seed_point-center, 1.]@transform
        elif float(residual.max()) > 1.5:
            raise ValueError("body_support_inconsistent_or_deformed")
        if self.inferred and not affine_ok and float(residual.max()) > .5:
            raise ValueError("auxiliary_support_perspective_unobservable")
        for patch, point, _ in matched:
            patch["position"] = point
        self.translation = shift
        self.point = semantic
        if np.any(self.point < 0.) or np.any(self.point > self.scale):
            raise ValueError("semantic_contact_out_of_view")
        self.uncertainty_px = .5+float(np.sqrt(np.mean(residual**2)))+inferred_extra
        if self.inferred:
            self.uncertainty_px += .04*self.support_distance+.03*float(np.linalg.norm(shift))
        return self.point.copy()


class _FixedWristReference:
    """Retain a visible seed in the hand camera; never initialize a hidden point."""
    maximum_unchecked_observations = 12

    def __init__(self, image, point_uv, roi_uv):
        self.shape = image.shape
        self.scale = np.array([image.shape[1]-1, image.shape[0]-1], dtype=float)
        self.seed_point = np.asarray(point_uv) * self.scale
        self.point = self.seed_point.copy()
        self.roi = np.asarray(roi_uv).reshape(2, 2) * self.scale
        outside = np.maximum(self.roi[0]-self.point, 0.) + np.maximum(self.point-self.roi[1], 0.)
        if np.linalg.norm(outside) > 24.:
            raise ValueError("same_body_support_too_far_from_contact")
        if not _visible_seed(image, self.point):
            raise ValueError("fixed_wrist_requires_visible_initial_tool_point")
        self.translation = np.zeros(2)
        self.uncertainty_px = .75
        self.unchecked_observations = 0
        self.source = "visible_seed_retained_in_fixed_wrist_frame"
        self.inferred = bool(np.linalg.norm(outside) > .5)
        self._support = None
        try:
            self._support = _FrozenSupport(image, point_uv, roi_uv)
        except ValueError:
            pass
        self.patches = [] if self._support is None else self._support.patches
        self.radius = 2 if self._support is None else self._support.radius

    def update(self, image):
        if image.shape != self.shape:
            raise ValueError("image_shape_changed")
        verified = False
        if self._support is not None:
            try:
                self._support.update(image)
            except ValueError:
                pass
            else:
                if max(np.linalg.norm(self._support.translation), np.linalg.norm(self._support.point-self._support.seed_point)) > 1.5:
                    raise ValueError("wrist_tool_moved_in_fixed_hand_frame")
                verified = True
        self.unchecked_observations = 0 if verified else self.unchecked_observations + 1
        if self.unchecked_observations > self.maximum_unchecked_observations:
            raise ValueError("fixed_wrist_reference_expired_without_texture_check")
        self.uncertainty_px = .75 + .15*self.unchecked_observations
        return self.point.copy()


class _ViewResponse:
    def __init__(self):
        self.samples = deque(maxlen=24)
        self.held_out = deque(maxlen=8)
        self.matrix = self.basis = self.excitation_basis = self.aggregate = None

    def append(self, delta, response):
        if self.basis is not None:
            unsupported = delta - self.excitation_basis @ self.excitation_basis.T @ delta
            if np.linalg.norm(unsupported) <= .0003:
                prediction = delta @ self.basis @ self.matrix
                self.held_out.append((float(np.sum((response-prediction)**2)), float(response@response)))
        self.samples.append((delta, response))
        if len(self.samples) < 3:
            return
        x, y = (np.array(values) for values in zip(*self.samples))
        _, singular, vt = np.linalg.svd(x, full_matrices=False)
        rank = int(np.sum(singular > max(.001, singular[0]/30.)))
        if rank == 0:
            return
        basis = vt[:rank].T
        excitation_basis = basis.copy()
        matrix = np.linalg.lstsq(x@basis, y, rcond=None)[0]
        residual = float(np.sqrt(np.mean(np.sum((x@basis@matrix-y)**2, axis=1))))
        u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
        supported = singular > max(25., singular[0]/30.)
        if not np.any(supported) or residual > 2.5:
            self.matrix = self.basis = self.excitation_basis = None
            self.held_out.clear()
            return
        basis = basis @ u[:, supported]
        matrix = u[:, supported].T @ matrix
        if self.basis is None or self.basis.shape[1] != basis.shape[1] or np.linalg.norm(
                self.basis@self.basis.T-basis@basis.T) > .15 or self.excitation_basis is None or np.linalg.norm(
                self.excitation_basis@self.excitation_basis.T-excitation_basis@excitation_basis.T) > .15:
            self.held_out.clear()
        self.basis, self.matrix, self.excitation_basis = basis, matrix, excitation_basis

    def validated(self):
        if self.basis is None or len(self.held_out) < 2:
            return False
        error2 = sum(row[0] for row in self.held_out)
        zero2 = sum(row[1] for row in self.held_out)
        return bool(zero2 > 1. and error2 < .65*zero2 and error2/len(self.held_out) < 2.5**2)


class PregraspVisualFeedback:
    maximum_orientation_change_rad = .035
    maximum_finger_change_m = .002
    maximum_local_radius_m = .04
    maximum_correction_m = .006
    maximum_nominal_deviation_m = .02
    maximum_local_estimate_m = .03
    image_tolerance_px = 2.5
    maximum_target_drift_px = 1.25
    maximum_wrist_tool_drift_px = 1.5
    minimum_motion_m = .001

    def __init__(self):
        self._reference = None
        self._active = False
        self._epoch = 0
        self._tracks = {}
        self._models = {}
        self._active_views = []
        self._view_status = {}
        self._origin = self._state = self._last_observation = None
        self._samples, self._held_out = deque(maxlen=24), deque(maxlen=8)
        self._matrix = self._basis = self._error = None
        self._last = self._diagnostic("unconfigured")

    def _diagnostic(self, status, **extra):
        return {"version": PREGRASP_VERSION, "status": status, "epoch": self._epoch,
            "reference_id": self._reference["reference_id"] if self._reference else None,
            "target_id": self._reference["target_id"] if self._reference else None,
            "reliable": False, "tracking_valid": False, "response_validated": False,
            "proposal_available": False, "observed_rank": 0, "depth_support": 0.,
            "image_error_px": None, "pixel_error_norm": None, "image_alignment_observed": False,
            "goal_correction_xyz_m": None, "sample_count": 0, "held_out_samples": 0,
            "active_views": list(self._active_views), "validated_views": [], "views": {},
            "grasp_certified": False, "task_success_certified": False, "clearance_certified": False,
            "probe_requested": False, "fallback": "continue_model_selected_field_or_direct",
            "new_reference_id_required": False,
            "semantic_points_source": "model_visible_contact_hypotheses_and_explicit_local_inferences",
            "maximum_orientation_change_rad": self.maximum_orientation_change_rad,
            "maximum_local_radius_m": self.maximum_local_radius_m,
            "limitations": ["No metric object pose, depth or calibrated camera is available.",
                "Image alignment does not certify 3D contact or an engaged grasp.",
                "Wrist-only image motion cannot always distinguish target motion from robot motion.",
                "Visible seeds and same-body membership are model hypotheses, not certified geometry."], **extra}

    def diagnostics(self):
        return copy.deepcopy(self._last)

    def invalidate(self, reason="pregrasp_assumption_changed"):
        if not isinstance(reason, str) or not reason:
            raise ValueError("invalidation reason must be nonempty text")
        self._active = False
        self._matrix = self._basis = None
        self._samples.clear()
        self._held_out.clear()
        for view in self._active_views:
            self._view_status[view] = reason
        self._active_views = []
        self._models.clear()
        self._last = self._diagnostic(reason, new_reference_id_required=self._reference is not None,
            views={view: {"status": status, "tracking_valid": False, "pixel_error_norm": None}
                   for view, status in self._view_status.items()})
        return self.diagnostics()

    @staticmethod
    def _observation(observation):
        if not isinstance(observation, PolicyObservation):
            raise TypeError("pregrasp feedback accepts only PolicyObservation")
        return np.array(observation.state, dtype=float)

    @staticmethod
    def _images(observation):
        return {"external": np.asarray(decode_image(observation.image).convert("L"), dtype=np.uint8),
                "wrist": np.asarray(decode_image(observation.wrist_image).convert("L"), dtype=np.uint8)}

    def _domain_reason(self, state):
        if _orientation_distance(state[3:6], self._origin[3:6]) > self.maximum_orientation_change_rad:
            return "measured_orientation_changed"
        if np.max(abs(state[6:]-self._origin[6:])) > self.maximum_finger_change_m:
            return "measured_finger_width_changed"
        if np.linalg.norm(state[:3]-self._origin[:3]) > self.maximum_local_radius_m:
            return "measured_translation_left_local_domain"
        return None

    def _drop_view(self, view, reason):
        if view in self._active_views:
            self._active_views.remove(view)
        self._models.pop(view, None)
        self._view_status[view] = reason
        self._active = bool(self._active_views)

    def configure(self, observation, reference, *, gripper_open, target_id=None):
        state = self._observation(observation)
        reference = validate_pregrasp_reference(reference)
        if type(gripper_open) is not bool:
            raise TypeError("gripper_open must be the caller's open-command hypothesis")
        if reference is None:
            return self.invalidate("pregrasp_reference_not_requested")
        if not gripper_open:
            return self.invalidate("open_gripper_hypothesis_missing")
        if target_id is not None and target_id != reference["target_id"]:
            return self.invalidate("pregrasp_target_identity_mismatch")
        if self._reference is not None and reference["reference_id"] == self._reference["reference_id"]:
            if not self._active:
                self._last["new_reference_id_required"] = True
                return self.diagnostics()
            self.observe(observation)
            if not self._active:
                return self.diagnostics()
            if reference["target_id"] != self._reference["target_id"]:
                return self.invalidate("same_id_target_changed")
            for view in list(self._active_views):
                if reference[view] is None:
                    self._drop_view(view, "view_withdrawn_by_model")
                    continue
                old, value = self._reference[view], reference[view]
                scale = self._tracks[view+"_tool"].scale
                if value.get("tool_reference_mode", "tracked_material") != old.get("tool_reference_mode", "tracked_material"):
                    return self.invalidate("same_id_tool_reference_mode_changed")
                if np.linalg.norm((np.asarray(value["desired_offset_uv"])-old["desired_offset_uv"])*scale) > 1.:
                    return self.invalidate("same_id_desired_relation_changed")
                for role in ("tool", "target"):
                    track = self._tracks[view+"_"+role]
                    if np.linalg.norm(np.asarray(value[role+"_contact_uv"])*scale-track.point) > 3.:
                        self._drop_view(view, "same_id_semantic_point_changed")
                        break
                    roi = np.asarray(value[role+"_roi_uv"]).reshape(2, 2)*scale
                    if np.max(abs(roi-(track.roi+track.translation))) > 4.:
                        self._drop_view(view, "same_id_body_support_changed")
                        break
            if not self._active:
                return self.invalidate("no_usable_pregrasp_view")
            self._refresh("response_validated" if self._validated() else "tracking_accumulating_natural_motion")
            self._last["mapping_preserved_across_model_call"] = True
            return self.diagnostics()
        self._epoch += 1
        self._reference = reference
        self._origin = self._state = state.copy()
        self._samples.clear()
        self._held_out.clear()
        self._matrix = self._basis = self._error = None
        self._last_observation = observation
        self._tracks, self._models, self._view_status = {}, {}, {}
        self._active_views = []
        images = self._images(observation)
        for view in _VIEWS:
            value = reference[view]
            if value is None:
                self._view_status[view] = "view_not_requested"
                continue
            try:
                # Target remains available as a stationarity guard even if tool seeding fails.
                target = _FrozenSupport(images[view], value["target_contact_uv"], value["target_roi_uv"])
                self._tracks[view+"_target"] = target
                cls = _FixedWristReference if value.get("tool_reference_mode") == "fixed_wrist" else _FrozenSupport
                tool = cls(images[view], value["tool_contact_uv"], value["tool_roi_uv"])
                self._tracks[view+"_tool"] = tool
                if any(np.linalg.norm(a["seed"]-b["seed"]) < tool.radius+target.radius+1
                       for a in tool.patches for b in target.patches):
                    raise ValueError("tool_target_body_support_overlaps")
                # Fixed references without texture still cannot borrow target support.
                if np.linalg.norm(tool.seed_point-target.seed_point) < 2.:
                    raise ValueError("tool_target_body_support_overlaps")
            except ValueError as exc:
                self._view_status[view] = str(exc)
                continue
            self._active_views.append(view)
            self._view_status[view] = "tracking_accumulating_natural_motion"
            self._models[view] = _ViewResponse()
            self._models[view].aggregate = self._snapshot(view)
        self._active = bool(self._active_views)
        if not self._active:
            reasons = [status for status in self._view_status.values() if status != "view_not_requested"]
            return self.invalidate(reasons[0] if reasons else "no_usable_pregrasp_view")
        return self._refresh("tracking_accumulating_natural_motion")

    def _snapshot(self, view):
        tool, target = self._tracks[view+"_tool"], self._tracks[view+"_target"]
        error = tool.point-target.point-np.asarray(self._reference[view]["desired_offset_uv"])*tool.scale
        return {"state": self._state.copy(), "tool": tool.point.copy(), "error": error}

    def _validated(self):
        return any(model.validated() for model in self._models.values())

    def _combined(self, views):
        if not views:
            return None, None
        # An unexcited direction is UNKNOWN, never an assumed zero response.
        # Combine only directions excited in every contributing view. Directions
        # excited with no response may legitimately be null in that view.
        projectors = sum(self._models[v].excitation_basis@self._models[v].excitation_basis.T for v in views)
        eigenvalues, eigenvectors = np.linalg.eigh(projectors)
        common = eigenvectors[:, eigenvalues >= len(views)-1e-6]
        if common.shape[1] == 0:
            return None, None
        response = np.concatenate([self._models[v].basis@self._models[v].matrix for v in views], axis=1)
        u, singular, _ = np.linalg.svd(common.T@response, full_matrices=False)
        supported = singular > max(25., singular[0]/30.)
        if not np.any(supported):
            return None, None
        basis = common@u[:, supported]
        return basis, basis.T@response

    def _compatible_views(self, views):
        if self._combined(views)[0] is not None or not views:
            return views
        # No jointly supported direction: use one validated view explicitly.
        return [max(views, key=lambda v: (len(self._models[v].held_out), len(self._models[v].samples)))]

    def _refresh(self, status):
        views = {}
        errors, uncertainties, fitted = [], [], []
        for view in _VIEWS:
            if view not in self._active_views:
                views[view] = {"status": self._view_status.get(view, "view_not_requested"),
                               "tracking_valid": False, "response_validated": False, "pixel_error_norm": None}
                continue
            model = self._models[view]
            error = self._snapshot(view)["error"]
            noise = math.hypot(self._tracks[view+"_tool"].uncertainty_px, self._tracks[view+"_target"].uncertainty_px)
            errors.extend(error)
            uncertainties.append(noise)
            if model.basis is not None:
                fitted.append(view)
            views[view] = {"status": "response_validated" if model.validated() else "tracking_accumulating_natural_motion",
                "tracking_valid": True, "response_validated": model.validated(),
                "pixel_error_norm": float(np.linalg.norm(error)), "image_error_px": error.tolist(),
                "uncertainty_px": noise, "sample_count": len(model.samples), "held_out_samples": len(model.held_out),
                "held_out_error2": sum(row[0] for row in model.held_out),
                "held_out_zero_motion_error2": sum(row[1] for row in model.held_out),
                "observed_rank": 0 if model.basis is None else model.basis.shape[1]}
        validated = [view for view in self._active_views if self._models[view].validated()]
        fitted = self._compatible_views(fitted)
        basis, matrix = self._combined(fitted)
        correction_views = self._compatible_views(validated)
        self._basis, self._matrix = self._combined(correction_views)
        self._error = np.asarray(errors)
        norms = [views[view]["pixel_error_norm"] for view in self._active_views]
        self._last = self._diagnostic(status, reliable=self._active, tracking_valid=self._active,
            response_validated=bool(validated), active_views=list(self._active_views), validated_views=validated, views=views,
            observed_rank=0 if basis is None else basis.shape[1],
            observable_basis_xyz=None if basis is None else basis.tolist(),
            response_views=fitted, correction_views=correction_views,
            validated_observed_rank=0 if self._basis is None else self._basis.shape[1],
            validated_depth_support=0. if self._basis is None else float((self._basis@self._basis.T)[2, 2]),
            response_px_per_m=None if matrix is None else (basis@matrix).tolist(),
            depth_support=0. if basis is None else float((basis@basis.T)[2, 2]),
            image_error_px=errors, pixel_error_norm=float(np.linalg.norm(errors)),
            per_view_error_norm_px=norms, per_view_uncertainty_px=uncertainties,
            image_alignment_observed=bool(norms) and all(value+math.sqrt(2.)*noise <= self.image_tolerance_px
                for value, noise in zip(norms, uncertainties)),
            current_points_px={key: track.point.tolist() for key, track in self._tracks.items() if key.split("_")[0] in self._active_views},
            point_uncertainty_px={key: track.uncertainty_px for key, track in self._tracks.items() if key.split("_")[0] in self._active_views},
            point_sources={key: {"source": track.source, "local_inference": bool(track.inferred),
                "unchecked_observations": getattr(track, "unchecked_observations", 0)}
                for key, track in self._tracks.items() if key.split("_")[0] in self._active_views},
            sample_count=max((len(model.samples) for model in self._models.values()), default=0),
            held_out_samples=max((len(model.held_out) for model in self._models.values()), default=0),
            held_out_error2=sum(row[0] for model in self._models.values() for row in model.held_out),
            held_out_zero_motion_error2=sum(row[1] for model in self._models.values() for row in model.held_out),
            target_stationarity_observed="external_target" in self._tracks and self._view_status.get("external_target_guard") != "unavailable")
        return self.diagnostics()

    def observe(self, observation):
        state = self._observation(observation)
        if not self._active or observation == self._last_observation:
            return self.diagnostics()
        reason = self._domain_reason(state)
        if reason:
            return self.invalidate(reason)
        images = self._images(observation)
        # Independent external target guard survives a lost external tool view.
        updated = set()
        target = self._tracks.get("external_target")
        if target is not None:
            try:
                target.update(images["external"])
                updated.add("external_target")
                self._view_status["external_target_guard"] = "observed_stationary"
            except ValueError as exc:
                self._view_status["external_target_guard"] = "unavailable"
                if "external" in self._active_views:
                    self._drop_view("external", str(exc))
            else:
                if max(np.linalg.norm(target.translation), np.linalg.norm(target.point-target.seed_point)) > self.maximum_target_drift_px:
                    return self.invalidate("fixed_camera_target_moved_no_response_learning")
        self._state = state
        self._last_observation = observation
        for view in list(self._active_views):
            try:
                for role in ("tool", "target"):
                    key = view+"_"+role
                    if key not in updated:
                        self._tracks[key].update(images[view])
                if view == "wrist" and np.linalg.norm(self._tracks["wrist_tool"].translation) > self.maximum_wrist_tool_drift_px:
                    raise ValueError("wrist_tool_moved_in_fixed_hand_frame")
                current = self._snapshot(view)
                model = self._models[view]
                delta = current["state"][:3]-model.aggregate["state"][:3]
                flow = current["error"]-model.aggregate["error"]
                if np.linalg.norm(delta) < .0003 and np.linalg.norm(flow) > 1.5:
                    raise ValueError("pixel_drift_without_measured_robot_motion")
                if np.linalg.norm(delta) >= self.minimum_motion_m:
                    response = current["tool"]-model.aggregate["tool"] if view == "external" else flow
                    model.append(delta, response)
                    model.aggregate = current
            except ValueError as exc:
                self._drop_view(view, str(exc))
        if not self._active:
            reason = next((self._view_status[view] for view in _VIEWS if self._reference[view] is not None), "no_usable_pregrasp_view")
            return self.invalidate(reason)
        return self._refresh("response_validated" if self._validated() else "tracking_accumulating_natural_motion")

    def corrected_goal(self, observation, nominal_goal):
        """At most 6 mm TOTAL from immutable nominal, only supported image directions."""
        nominal = _numbers(nominal_goal, 3, "nominal_goal")
        diagnostic = self.observe(observation)
        if not self._active or not self._validated():
            return None, diagnostic
        state = np.asarray(observation.state[:3])
        if np.linalg.norm(nominal-self._origin[:3]) > self.maximum_local_radius_m:
            diagnostic.update(status="nominal_goal_outside_local_domain", proposal_available=False)
            return None, diagnostic
        views = diagnostic["correction_views"]
        basis, matrix = self._combined(views)
        if basis is None:
            return None, diagnostic
        error = np.concatenate([self._snapshot(view)["error"] for view in views])
        weights = 1./np.maximum(.75, np.repeat([diagnostic["views"][view]["uncertainty_px"] for view in views], 2))
        weighted_matrix, weighted_error = matrix*weights, error*weights
        ridge = max(1e-8, float(np.linalg.norm(weighted_matrix))**2*1e-5)
        supported_delta = -np.linalg.solve(weighted_matrix@weighted_matrix.T+ridge*np.eye(matrix.shape[0]), weighted_matrix@weighted_error)
        delta = basis@supported_delta
        delta *= min(1., self.maximum_local_estimate_m/max(float(np.linalg.norm(delta)), 1e-12))
        projector = basis@basis.T
        desired = state+delta+(np.eye(3)-projector)@(nominal-state)
        change = desired-nominal
        change *= min(1., self.maximum_correction_m/max(float(np.linalg.norm(change)), 1e-12))
        goal = nominal+change
        if (np.linalg.norm(goal-self._origin[:3]) > self.maximum_local_radius_m
                or np.linalg.norm(goal-nominal) > self.maximum_nominal_deviation_m):
            diagnostic.update(status="correction_outside_local_domain", proposal_available=False)
            return None, diagnostic
        uncorrected = weighted_error+(nominal-state)@basis@weighted_matrix
        corrected = weighted_error+(goal-state)@basis@weighted_matrix
        if np.linalg.norm(corrected) >= np.linalg.norm(uncorrected)-1e-6:
            diagnostic.update(status="no_predicted_alignment_improvement", proposal_available=False)
            return None, diagnostic
        diagnostic.update(status="pregrasp_goal_correction", proposal_available=True,
            goal_correction_xyz_m=change.tolist(), goal_xyz=goal.tolist(),
            correction_norm_m=float(np.linalg.norm(change)), nominal_goal_xyz=nominal.tolist(),
            correction_views=views, predicted_nominal_error_norm=float(np.linalg.norm(uncorrected)),
            predicted_corrected_error_norm=float(np.linalg.norm(corrected)))
        self._last = copy.deepcopy(diagnostic)
        return goal, diagnostic
