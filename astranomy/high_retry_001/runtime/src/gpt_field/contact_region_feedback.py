"""Opt-in local contact alignment from model-selected RGB *regions*.

Region centers are image-coordinate anchors transported by local texture. They
are not inferred physical contacts, object centers, grasp centers or 3D poses.
Natural measured translations train a frozen image-response model; subsequent
samples validate it before a bounded goal correction is offered. No action,
probe, closure, task stop, model call or privileged observation is generated.
"""
from __future__ import annotations

import copy
import math
import time
from collections import deque

import numpy as np

from .observation import PolicyObservation, decode_image
from .operation_visual import _RegionFlow, _backend, _overlap
from .speed_first_pregrasp import _orientation_distance

CONTACT_REGION_VERSION = "contact-region-feedback-v1"
_VIEWS = ("external", "wrist")
_REFERENCE_KEYS = {"reference_id", "target_id", "external", "wrist"}
_VIEW_KEYS = {"tool_region_uv", "target_region_uv", "desired_offset_uv"}


def contact_region_schema():
    box = {"type": "array", "minItems": 4, "maxItems": 4,
           "items": {"type": "number", "minimum": 0., "maximum": 1.}}
    view = {"type": "object", "additionalProperties": False,
            "required": sorted(_VIEW_KEYS), "properties": {
                "tool_region_uv": copy.deepcopy(box), "target_region_uv": copy.deepcopy(box),
                "desired_offset_uv": {"type": "array", "minItems": 2, "maxItems": 2,
                    "items": {"type": "number", "minimum": -1., "maximum": 1.}}}}
    return {"anyOf": [{"type": "null"}, {"type": "object", "additionalProperties": False,
        "required": sorted(_REFERENCE_KEYS), "properties": {
            "reference_id": {"type": "string", "minLength": 1, "maxLength": 64},
            "target_id": {"type": "string", "minLength": 1, "maxLength": 64},
            **{name: {"anyOf": [{"type": "null"}, copy.deepcopy(view)]} for name in _VIEWS}}}]}


contact_region_reference_schema = contact_region_schema


def _numbers(value, size, name, low=None, high=None):
    try:
        valid = isinstance(value, (tuple, list, np.ndarray)) and len(value) == size and all(
            not isinstance(x, (bool, np.bool_)) and isinstance(x, (int, float, np.number))
            and not isinstance(x, (complex, np.complexfloating)) and math.isfinite(float(x)) for x in value)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} requires {size} finite real numbers")
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,) or (low is not None and np.any(arr < low)) or (high is not None and np.any(arr > high)):
        raise ValueError(f"{name} outside permitted range")
    return arr


def validate_contact_region_reference(reference):
    if reference is None:
        return None
    if not isinstance(reference, dict) or set(reference) != _REFERENCE_KEYS:
        raise ValueError("contact region reference requires reference_id,target_id,external,wrist")
    result = copy.deepcopy(reference)
    for key in ("reference_id", "target_id"):
        if not isinstance(reference[key], str) or not reference[key].strip() or len(reference[key]) > 64:
            raise ValueError(f"{key} requires 1..64 text characters")
    if all(reference[view] is None for view in _VIEWS):
        raise ValueError("contact region reference needs at least one view")
    for view in _VIEWS:
        value = reference[view]
        if value is None:
            continue
        if not isinstance(value, dict) or set(value) != _VIEW_KEYS:
            raise ValueError("contact region view has missing or unexpected fields")
        for key in _VIEW_KEYS:
            offset = key == "desired_offset_uv"
            arr = _numbers(value[key], 2 if offset else 4, key, -1. if offset else 0., 1.)
            if not offset and np.any(arr[:2] >= arr[2:]):
                raise ValueError("contact region boxes must have positive width and height")
            result[view][key] = arr.tolist()
    return result


def _span(rows, threshold=.0005):
    if not len(rows):
        return np.empty((3, 0))
    _, s, vt = np.linalg.svd(np.asarray(rows), full_matrices=False)
    rank = int(np.sum(s > max(threshold, float(s[0]) / 30.)))
    return vt[:rank].T


def _intersection(bases):
    if not bases or any(b.shape[1] == 0 for b in bases):
        return np.empty((3, 0))
    values, vectors = np.linalg.eigh(sum(b @ b.T for b in bases))
    return vectors[:, values >= len(bases) - 1e-6]


class _FrozenResponse:
    """Training and held-out samples never overlap for the active frozen fit.

    New input directions start a new training phase, retiring all old validation.
    Even a frozen fit is only authorized along directions actually exercised by
    later validation motions. Unknown directions are never assigned a zero gain.
    """
    def __init__(self):
        self.training = deque(maxlen=24)
        self.validation = deque(maxlen=12)
        self.matrix = None
        self.input_basis = np.empty((3, 0))
        self.observable_basis = np.empty((3, 0))
        self.fit_version = 0
        self.fit_training_ids = []
        self.sample_id = 0
        self.rejected = False
        self.reason = "accumulating_training_motion"

    def append(self, delta, response):
        self.sample_id += 1
        row = (self.sample_id, np.array(delta, float), np.array(response, float))
        if self.rejected:
            return
        if self.matrix is not None:
            unknown = delta - self.input_basis @ self.input_basis.T @ delta
            if np.linalg.norm(unknown) <= .0003:
                predicted = delta @ self.matrix
                residual = float(np.linalg.norm(response - predicted))
                self.validation.append({"sample_id": self.sample_id, "delta": row[1],
                    "supported_delta": self.input_basis @ self.input_basis.T @ row[1],
                    "unsupported_input_norm_m": float(np.linalg.norm(unknown)),
                    "error2": residual ** 2, "zero_error2": float(response @ response),
                    "prediction_px": predicted, "response_px": row[2], "fit_version": self.fit_version})
                if residual > max(2.5, .75 * float(np.linalg.norm(predicted))):
                    self.rejected = True
                    self.reason = "held_out_response_mismatch"
                else:
                    self.reason = "response_validated" if self.validated_basis().shape[1] else "collecting_held_out_motion"
                return
            # A changed excitation subspace requires a freshly frozen fit. Old
            # held-out rows stay excluded rather than being silently recycled.
            self.validation.clear()
        self.training.append(row)
        if len(self.training) < 3:
            return
        x = np.array([r[1] for r in self.training])
        y = np.array([r[2] for r in self.training])
        basis = _span(x, .001)
        if not basis.shape[1]:
            return
        compact = np.linalg.lstsq(x @ basis, y, rcond=None)[0]
        residual = float(np.sqrt(np.mean(np.sum((x @ basis @ compact - y) ** 2, axis=1))))
        u, s, _ = np.linalg.svd(compact, full_matrices=False)
        visible = s > max(25., float(s[0]) / 30.)
        if residual > 2.5 or not np.any(visible):
            self.matrix = None
            self.reason = "training_response_not_observable_or_inconsistent"
            return
        self.matrix = basis @ compact
        self.input_basis = basis
        self.observable_basis = basis @ u[:, visible]
        self.fit_version += 1
        self.fit_training_ids = [r[0] for r in self.training]
        self.reason = "collecting_held_out_motion"

    def validated_basis(self):
        if self.rejected or self.matrix is None or len(self.validation) < 2:
            return np.empty((3, 0))
        error2 = sum(r["error2"] for r in self.validation)
        zero2 = sum(r["zero_error2"] for r in self.validation)
        if zero2 <= 1. or error2 >= .65 * zero2 or error2 / len(self.validation) >= 2.5 ** 2:
            return np.empty((3, 0))
        return _intersection([self.observable_basis, _span([r["supported_delta"] for r in self.validation])])

    def snapshot(self):
        valid = self.validated_basis()
        return {"status": self.reason, "response_validated": bool(valid.shape[1]),
            "training_samples": len(self.training), "held_out_samples": len(self.validation),
            "fit_version": self.fit_version, "fit_training_sample_ids": list(self.fit_training_ids),
            "held_out_sample_ids": [r["sample_id"] for r in self.validation],
            "held_out_max_unsupported_input_m": max((r["unsupported_input_norm_m"] for r in self.validation), default=0.),
            "validation_coverage_coordinates": "accepted_motion_projected_into_frozen_input_domain",
            "held_out_error2": sum(r["error2"] for r in self.validation),
            "held_out_zero_motion_error2": sum(r["zero_error2"] for r in self.validation),
            "observed_rank": int(self.observable_basis.shape[1]), "validated_rank": int(valid.shape[1]),
            "validated_basis_xyz": valid.tolist(), "unknown_input_directions_are_zero": False,
            "response_px_per_m": None if self.matrix is None else self.matrix.tolist(),
            "response_domain_basis_xyz": self.input_basis.tolist(), "response_rejected": self.rejected}


class _FixedWristRegion:
    """Conditionally rigid image anchor, for visible wrist TOOL regions only.

    A finger may have a visible edge but insufficient corners for optical flow.
    Its image anchor is fixed while the measured opening/orientation are fixed;
    every frame must still match the frozen visible edge and intensity pattern.
    This is a model-selected rigid-hand hypothesis, not measured optical motion,
    a semantic object detector, a hidden contact estimate or camera calibration.
    """
    maximum_age = _RegionFlow.maximum_age

    def __init__(self, image, roi_uv, cv):
        self.cv, self.shape = cv, image.shape
        self.scale = np.array([image.shape[1] - 1, image.shape[0] - 1], dtype=float)
        self.roi = np.asarray(roi_uv).reshape(2, 2) * self.scale
        self.center = self.roi.mean(axis=0)
        self.origin = self.center.copy()
        self._lo = np.maximum(np.ceil(self.roi[0]).astype(int), 1)
        self._hi = np.minimum(np.floor(self.roi[1]).astype(int), np.array(image.shape[::-1]) - 2)
        if np.any(self._hi - self._lo < 5):
            raise ValueError("fixed_wrist_region_too_small_for_edge_check")
        crop = self._crop(image)
        self._seed = crop - crop.mean()
        self._seed_norm = float(np.linalg.norm(self._seed))
        self._seed_std = float(crop.std())
        self._edge = self._gradients(crop)
        self._edge_norm = float(np.linalg.norm(self._edge))
        strengths = np.linalg.norm(self._edge, axis=-1)
        self._edge_support = int(np.sum(strengths > max(10., .2 * float(strengths.max()))))
        if (self._seed_std < 5. or float(np.percentile(crop, 95) - np.percentile(crop, 5)) < 20.
                or self._edge_support < max(8, int(.015 * crop.size)) or self._edge_norm < 1.):
            raise ValueError("fixed_wrist_requires_visible_region_edge")
        self.age = 0
        self.valid = True
        self.reason = "visible_edge_seed_conditional_fixed_wrist_anchor"
        self.appearance_similarity = self.edge_similarity = 1.
        self.uncertainty = 1.
        self.step_displacement = np.zeros(2)

    def _crop(self, image):
        return np.asarray(image[self._lo[1]:self._hi[1] + 1,
                                self._lo[0]:self._hi[0] + 1], dtype=np.float32)

    def _gradients(self, crop):
        # Ignore the crop border: gradients there can arise from an arbitrary
        # region boundary rather than a visible material edge inside the region.
        gx = self.cv.Sobel(crop, self.cv.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
        gy = self.cv.Sobel(crop, self.cv.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
        return np.stack((gx, gy), axis=-1)

    def invalidate(self, reason):
        self.valid, self.reason = False, reason

    def update(self, before, after):
        if not self.valid:
            return
        if before.shape != self.shape or after.shape != self.shape:
            return self.invalidate("image_shape_changed")
        if self.age >= self.maximum_age:
            return self.invalidate("local_tracking_epoch_expired")
        crop = self._crop(after)
        centered = crop - crop.mean()
        edge = self._gradients(crop)
        self.appearance_similarity = float(np.sum(centered * self._seed) /
            max(1e-12, float(np.linalg.norm(centered)) * self._seed_norm))
        self.edge_similarity = float(np.sum(edge * self._edge) /
            max(1e-12, float(np.linalg.norm(edge)) * self._edge_norm))
        ratio = float(crop.std()) / self._seed_std
        if (self.appearance_similarity < .90 or self.edge_similarity < .90
                or ratio < .7 or ratio > 1.4):
            return self.invalidate("fixed_wrist_visible_edge_or_appearance_changed")
        self.age += 1
        self.reason = "visible_pattern_verified_conditional_fixed_wrist_anchor"
        self.uncertainty = 1. + .005 * self.age

    def snapshot(self):
        return {"valid": self.valid, "reason": self.reason,
            "center_uv": (self.center / self.scale).tolist(),
            "displacement_px": [0., 0.], "step_displacement_px": [0., 0.],
            "support_count": self._edge_support, "age": self.age,
            "uncertainty_px": self.uncertainty,
            "appearance_similarity": self.appearance_similarity, "edge_similarity": self.edge_similarity,
            "measurement": "conditional_fixed_wrist_region_anchor_not_measured_optical_flow",
            "image_motion_measured": False, "visible_pattern_checked_each_observation": True,
            "semantic_identity_certified": False, "physical_contact_known": False}


class ContactRegionFeedback:
    maximum_orientation_change_rad = .035
    maximum_finger_change_m = .002
    maximum_local_radius_m = .04
    maximum_correction_m = .006
    maximum_target_drift_px = 1.5
    minimum_motion_m = .001

    def __init__(self):
        self._reference = None
        self._retired_ids = set()
        self._epoch = 0
        self._tracks = {}
        self._models = {}
        self._aggregates = {}
        self._view_status = {}
        self._active_views = []
        self._origin = self._state = self._last_observation = self._images = None
        self._nominal = None
        self._nominal_intent_key = None
        self._retired_intent_keys = set()
        self._last_proposal = None
        self._applied = None
        self._restore_nominal = False
        self._post_application = {"status": "not_applied", "measured_checks": 0}
        self._elapsed_s = 0.
        self._frames = 0
        self._proposals = 0
        self._applications = 0
        self._status = "unconfigured"
        self._last = self._diagnostic()

    @staticmethod
    def _observation(observation):
        if not isinstance(observation, PolicyObservation):
            raise TypeError("contact feedback accepts only allowlisted PolicyObservation")
        return np.asarray(observation.state, dtype=float)

    @staticmethod
    def _gray(observation):
        return {"external": np.asarray(decode_image(observation.image).convert("L")),
                "wrist": np.asarray(decode_image(observation.wrist_image).convert("L"))}

    def _diagnostic(self, **extra):
        return {"version": CONTACT_REGION_VERSION, "status": self._status, "epoch": self._epoch,
            "reference_id": self._reference["reference_id"] if self._reference else None,
            "target_id": self._reference["target_id"] if self._reference else None,
            "tracking_valid": bool(self._active_views), "response_validated": False,
            "proposal_available": False, "active_views": list(self._active_views),
            "validated_views": [], "views": {}, "pixel_error_norm": None,
            "image_error_px": None, "sample_count": 0, "held_out_samples": 0,
            "goal_correction_xyz_m": None, "nominal_goal_xyz": None if self._nominal is None else self._nominal.tolist(),
            "restore_nominal_requested": self._restore_nominal,
            "new_reference_id_required": bool(self._reference) and not self._active_views,
            "post_application": copy.deepcopy(self._post_application),
            "proposal_count": self._proposals, "application_count": self._applications,
            "tracking_elapsed_s": self._elapsed_s, "observations": self._frames,
            "anchor_semantics": "transported_model_region_centers_not_physical_contact_points",
            "desired_offset_semantics": "tool_region_anchor_minus_target_region_anchor",
            "grasp_certified": False, "task_success_certified": False, "clearance_certified": False,
            "probe_requested": False, "fallback": "continue_model_selected_field_or_direct",
            "limitations": ["Texture consistency does not verify semantic region membership or intended grasp geometry.",
                "Pixel alignment does not establish depth, physical contact or grasp success.",
                "Wrist-only response cannot always separate target motion from camera/robot motion.",
                "Fixed wrist tool anchors require the model's rigid-hand region hypothesis; a static wrong region cannot be disproved by appearance alone.",
                "Predicted image benefit is not evidence of task benefit or causal correction benefit."], **extra}

    def diagnostics(self):
        result = copy.deepcopy(self._last)
        result["tracking_elapsed_s"] = self._elapsed_s
        return result

    snapshot = diagnostics

    def invalidate(self, reason="contact_region_assumption_changed"):
        if self._applied is not None:
            self._restore_nominal = True
            self._post_application.update(status="correction_revoked", reason=reason)
        self._applied = self._last_proposal = None
        self._active_views = []
        for track in self._tracks.values():
            track.invalidate(reason)
        self._models.clear()
        self._status = reason
        self._last = self._diagnostic()
        return self.diagnostics()

    def configure(self, observation, reference, *, gripper_open, target_id=None, phase=None):
        started = time.perf_counter()
        nested_elapsed = 0.
        try:
            state = self._observation(observation)
            reference = validate_contact_region_reference(reference)
            if type(gripper_open) is not bool:
                raise TypeError("gripper_open must be a boolean open-command hypothesis")
            if reference is None:
                return self.invalidate("contact_region_reference_not_requested")
            if not gripper_open:
                return self.invalidate("open_gripper_hypothesis_missing")
            if phase is not None and phase not in ("align", "pregrasp", "contact_alignment"):
                return self.invalidate("contact_region_phase_not_alignment")
            if target_id is not None and target_id != reference["target_id"]:
                return self.invalidate("contact_region_target_identity_mismatch")
            identifier = reference["reference_id"]
            if self._reference and identifier == self._reference["reference_id"]:
                if reference != self._reference:
                    return self.invalidate("same_id_reference_changed_requires_new_id")
                before_elapsed = self._elapsed_s
                result = self.observe(observation)
                nested_elapsed = self._elapsed_s - before_elapsed
                return result
            if identifier in self._retired_ids:
                return self.invalidate("retired_reference_id_cannot_restart")
            if self._reference:
                self._retired_ids.add(self._reference["reference_id"])
            # A previous revised target must be restored by the caller before a
            # new semantic epoch can apply any new target. This flag is retained.
            had_applied = self._applied is not None
            previous_nominal = self._nominal
            self._reference = reference
            self._epoch += 1
            self._origin = self._state = state.copy()
            self._last_observation = observation
            self._images = self._gray(observation)
            self._tracks, self._models, self._aggregates, self._view_status = {}, {}, {}, {}
            self._active_views = []
            self._nominal = None
            self._applied = self._last_proposal = None
            self._restore_nominal = had_applied
            self._post_application = {"status": "new_epoch_previous_correction_retired" if had_applied else "not_applied",
                                      "measured_checks": 0,
                                      "previous_nominal_goal_xyz": None if previous_nominal is None else previous_nominal.tolist()}
            cv = _backend()
            for view in _VIEWS:
                value = reference[view]
                if value is None:
                    self._view_status[view] = "view_not_requested"
                    continue
                if _overlap(value["tool_region_uv"], value["target_region_uv"]) > .1:
                    self._view_status[view] = "tool_target_regions_overlap"
                    continue
                for role in ("tool", "target"):
                    key = view + "_" + role
                    try:
                        self._tracks[key] = _RegionFlow(self._images[view], value[role + "_region_uv"], cv)
                    except (ValueError, cv.error) as exc:
                        self._view_status[view] = str(exc)
                        if view == "wrist" and role == "tool":
                            try:
                                self._tracks[key] = _FixedWristRegion(self._images[view], value[role + "_region_uv"], cv)
                            except (ValueError, cv.error) as fallback_error:
                                self._view_status[view] = str(fallback_error)
                if all(view + "_" + role in self._tracks for role in ("tool", "target")):
                    self._active_views.append(view)
                    self._models[view] = _FrozenResponse()
                    self._aggregates[view] = self._sample(view)
                    self._view_status[view] = "tracking_natural_motion"
            self._status = "tracking_natural_motion" if self._active_views else "no_usable_contact_regions"
            return self._refresh()
        except ImportError:
            return self.invalidate("optical_flow_backend_unavailable")
        finally:
            self._elapsed_s += time.perf_counter() - started - nested_elapsed

    def _sample(self, view):
        tool, target = (self._tracks[view + "_" + role] for role in ("tool", "target"))
        offset = np.asarray(self._reference[view]["desired_offset_uv"]) * tool.scale
        return {"state": self._state[:3].copy(), "error": tool.center - target.center - offset,
                "tool": tool.center.copy()}

    def _drop_view(self, view, reason):
        if view in self._active_views:
            self._active_views.remove(view)
        self._models.pop(view, None)
        self._view_status[view] = reason

    def _refresh(self, **extra):
        views = {}
        errors = []
        for view in _VIEWS:
            tracks = {role: self._tracks[view + "_" + role].snapshot() for role in ("tool", "target")
                      if view + "_" + role in self._tracks}
            if view not in self._active_views:
                views[view] = {"tracking_valid": False, "status": self._view_status.get(view, "unconfigured"), "regions": tracks}
                continue
            sample = self._sample(view)
            errors.extend(sample["error"].tolist())
            views[view] = {"tracking_valid": True, **self._models[view].snapshot(),
                "image_error_px": sample["error"].tolist(), "pixel_error_norm": float(np.linalg.norm(sample["error"])),
                "regions": tracks}
        valid = [v for v in self._active_views if self._models[v].validated_basis().shape[1]]
        self._last = self._diagnostic(views=views, validated_views=valid, response_validated=bool(valid),
            image_error_px=errors or None, pixel_error_norm=float(np.linalg.norm(errors)) if errors else None,
            sample_count=max((m.sample_id for m in self._models.values()), default=0),
            held_out_samples=max((len(m.validation) for m in self._models.values()), default=0),
            target_stationarity_observed=bool(self._tracks.get("external_target") and self._tracks["external_target"].valid),
            **extra)
        return self.diagnostics()

    def observe(self, observation):
        started = time.perf_counter()
        try:
            state = self._observation(observation)
            if not self._active_views or observation == self._last_observation:
                return self.diagnostics()
            if observation.prompt != self._last_observation.prompt:
                return self.invalidate("task_prompt_changed")
            for condition, reason in (
                (_orientation_distance(state[3:6], self._origin[3:6]) > self.maximum_orientation_change_rad, "measured_orientation_changed"),
                (np.max(abs(state[6:] - self._origin[6:])) > self.maximum_finger_change_m, "measured_finger_width_changed"),
                (np.linalg.norm(state[:3] - self._origin[:3]) > self.maximum_local_radius_m, "measured_translation_left_local_domain")):
                if condition:
                    return self.invalidate(reason)
            images = self._gray(observation)
            for key, track in self._tracks.items():
                if not track.valid:
                    continue
                view = key.split("_")[0]
                try:
                    track.update(self._images[view], images[view])
                except (ValueError, track.cv.error):
                    track.invalidate("optical_flow_update_failed")
            self._images = images
            self._state = state
            self._last_observation = observation
            self._frames += 1
            target = self._tracks.get("external_target")
            if target is not None and target.valid and np.linalg.norm(target.center - target.origin) > self.maximum_target_drift_px:
                return self.invalidate("fixed_camera_target_moved")
            for view in list(self._active_views):
                tool, target = (self._tracks[view + "_" + role] for role in ("tool", "target"))
                if not tool.valid or not target.valid:
                    self._drop_view(view, tool.reason if not tool.valid else target.reason)
                    continue
                if view == "wrist" and np.linalg.norm(tool.center - tool.origin) > 1.5:
                    self._drop_view(view, "wrist_tool_reference_moved")
                    continue
                current, prior = self._sample(view), self._aggregates[view]
                delta = current["state"] - prior["state"]
                response = current["error"] - prior["error"]
                if np.linalg.norm(delta) < .0003 and np.linalg.norm(response) > 1.5:
                    self._drop_view(view, "image_drift_without_robot_motion")
                    continue
                if np.linalg.norm(delta) >= self.minimum_motion_m:
                    # A stationary external target has small tracking noise; do
                    # not treat its noise as a robot response component.
                    learned = current["tool"] - prior["tool"] if view == "external" else response
                    self._models[view].append(delta, learned)
                    self._aggregates[view] = current
            if self._applied is not None:
                reason = self._check_application()
                if reason:
                    if reason == "applied_correction_view_lost" and self._active_views:
                        # Revoke the combined-view goal, retain independent
                        # surviving tracks/fits. Only after nominal restoration
                        # can a new proposal use that remaining view.
                        self._restore_nominal = True
                        self._applied = self._last_proposal = None
                        self._post_application.update(status="correction_revoked", reason=reason)
                        self._status = reason
                        return self._refresh()
                    return self.invalidate(reason)
            if not self._active_views:
                return self.invalidate("no_usable_contact_regions")
            self._status = "response_validated" if any(m.validated_basis().shape[1] for m in self._models.values()) else "tracking_natural_motion"
            return self._refresh()
        finally:
            self._elapsed_s += time.perf_counter() - started

    def _check_application(self):
        check = self._applied
        delta = self._state[:3] - check["state"]
        if any(v not in self._active_views for v in check["views"]):
            return "applied_correction_view_lost"
        if np.linalg.norm(delta) < self.minimum_motion_m:
            return None
        rows = {}
        for view in check["views"]:
            basis = check["bases"][view]
            unknown = delta - basis @ basis.T @ delta
            if np.linalg.norm(unknown) > .0003:
                return "applied_correction_motion_outside_validated_domain"
            before, after = check["errors"][view], self._sample(view)["error"]
            predicted = delta @ check["matrices"][view]
            observed = after - before
            error = float(np.linalg.norm(observed - predicted))
            rows[view] = {"predicted_image_change_px": predicted.tolist(), "observed_image_change_px": observed.tolist(),
                "prediction_error_px": error, "before_error_norm_px": float(np.linalg.norm(before)),
                "after_error_norm_px": float(np.linalg.norm(after)),
                "observed_error_decreased": bool(np.linalg.norm(after) < np.linalg.norm(before))}
            if error > max(1.5, .6 * float(np.linalg.norm(predicted))):
                self._post_application.update(status="prediction_mismatch", views=rows,
                                              measured_translation_m=delta.tolist())
                return "applied_correction_prediction_mismatch"
        self._post_application = {"status": "prediction_supported_not_grasp_evidence", "views": rows,
            "measured_translation_m": delta.tolist(), "measured_checks": self._post_application.get("measured_checks", 0) + 1,
            "physical_grasp_improved": "unknown", "causal_correction_benefit": "unmeasured"}
        check["state"] = self._state[:3].copy()
        check["errors"] = {v: self._sample(v)["error"].copy() for v in check["views"]}
        return None

    def begin_nominal_goal(self, nominal_goal, intent_key):
        """Start an explicitly accepted model intent without resetting RGB fits.

        The caller supplies an original model goal, never its previous corrected
        target. It must call this only for a newly accepted operation. Repeated
        calls with the same intent cannot shift the immutable correction origin.
        """
        nominal = _numbers(nominal_goal, 3, "nominal_goal")
        if not isinstance(intent_key, str) or not intent_key.strip():
            raise ValueError("nominal intent key must be nonempty text")
        if intent_key == self._nominal_intent_key:
            if self._nominal is None or not np.allclose(nominal, self._nominal, atol=1e-10, rtol=0.):
                raise ValueError("same nominal intent cannot change its original goal")
            return self.diagnostics()
        if intent_key in self._retired_intent_keys:
            raise ValueError("retired nominal intent cannot be reused")
        if self._nominal_intent_key is not None:
            self._retired_intent_keys.add(self._nominal_intent_key)
        self._nominal_intent_key = intent_key
        self._nominal = nominal.copy()
        self._applied = self._last_proposal = None
        self._restore_nominal = False
        self._post_application = {"status": "new_nominal_intent", "measured_checks": 0}
        return self._refresh()

    def notify_nominal_restored(self, nominal_goal):
        nominal = _numbers(nominal_goal, 3, "nominal_goal")
        expected = self._nominal
        if expected is None:
            expected = self._post_application.get("previous_nominal_goal_xyz")
        if expected is None or not np.allclose(nominal, expected, atol=1e-10, rtol=0.):
            raise ValueError("restoration must use the immutable original nominal goal")
        self._restore_nominal = False
        self._applied = self._last_proposal = None
        self._post_application["nominal_restored"] = True
        return self._refresh()

    def corrected_goal(self, observation, nominal_goal):
        nominal = _numbers(nominal_goal, 3, "nominal_goal")
        self.observe(observation)
        if self._nominal is None:
            self._nominal = nominal.copy()
        elif not np.allclose(nominal, self._nominal, atol=1e-10, rtol=0.):
            return None, self.invalidate("nominal_changed_requires_new_intent")
        if not self._active_views or self._restore_nominal:
            return None, self._refresh()
        if np.linalg.norm(nominal - self._origin[:3]) > self.maximum_local_radius_m:
            self._status = "nominal_goal_outside_local_domain"
            return None, self._refresh()
        views = [v for v in self._active_views if self._models[v].validated_basis().shape[1]]
        if not views:
            return None, self._refresh()
        basis = _intersection([self._models[v].validated_basis() for v in views])
        if not basis.shape[1]:
            # Independently useful views need not excite the same directions.
            # Use one explicitly, rather than inventing gains in unknown axes.
            views = [max(views, key=lambda v: sum(r["zero_error2"] for r in self._models[v].validation))]
            basis = self._models[views[0]].validated_basis()
        nominal_delta = nominal - self._state[:3]
        # All nominal displacement must lie in this jointly validated subspace.
        # Projecting merely trained (but unvalidated) axes to zero would give a
        # false residual-at-nominal prediction even if the correction is valid.
        if np.linalg.norm(nominal_delta - basis @ basis.T @ nominal_delta) > .0003:
            self._status = "nominal_motion_has_unknown_image_response"
            return None, self._refresh()
        matrix = np.concatenate([basis.T @ self._models[v].matrix for v in views], axis=1)
        error = np.concatenate([self._sample(v)["error"] for v in views])
        state = self._state[:3]
        # Optimize the goal change from immutable nominal, not from the last
        # corrected target. Unknown directions retain the model's nominal goal.
        nominal_prediction = error + (nominal - state) @ basis @ matrix
        ridge = max(1e-8, float(np.linalg.norm(matrix)) ** 2 * 1e-5)
        change = -basis @ np.linalg.solve(matrix @ matrix.T + ridge * np.eye(basis.shape[1]), matrix @ nominal_prediction)
        change *= min(1., self.maximum_correction_m / max(float(np.linalg.norm(change)), 1e-12))
        goal = nominal + change
        corrected_prediction = nominal_prediction + change @ basis @ matrix
        if (np.linalg.norm(goal - self._origin[:3]) > self.maximum_local_radius_m
                or np.linalg.norm(corrected_prediction) >= np.linalg.norm(nominal_prediction) - .05):
            self._status = "no_admissible_predicted_alignment_improvement"
            return None, self._refresh()
        self._proposals += 1
        self._status = "contact_region_goal_proposal"
        self._last_proposal = {"observation": observation, "nominal": nominal.copy(), "goal": goal.copy(), "views": views,
            "bases": {v: self._models[v].validated_basis().copy() for v in views},
            "matrices": {v: self._models[v].matrix.copy() for v in views}}
        return goal, self._refresh(proposal_available=True, goal_correction_xyz_m=change.tolist(), goal_xyz=goal.tolist(),
            correction_norm_m=float(np.linalg.norm(change)), correction_views=views,
            predicted_nominal_error_norm=float(np.linalg.norm(nominal_prediction)),
            predicted_corrected_error_norm=float(np.linalg.norm(corrected_prediction)))

    def notify_correction_applied(self, observation, nominal_goal, corrected_goal):
        """Register an accepted planner target; shadow/off callers never call this.

        This records target application, not executed motion. Subsequent measured
        observations assess the frozen prediction. It cannot establish causality
        against an unexecuted nominal control trajectory or certify a grasp.
        """
        self._observation(observation)
        nominal = _numbers(nominal_goal, 3, "nominal_goal")
        corrected = _numbers(corrected_goal, 3, "corrected_goal")
        proposal = self._last_proposal
        if (proposal is None or observation != proposal["observation"]
                or not np.allclose(nominal, proposal["nominal"], atol=1e-10, rtol=0.)
                or not np.allclose(corrected, proposal["goal"], atol=1e-10, rtol=0.)):
            raise ValueError("application must match the current observation and exact offered proposal")
        if np.linalg.norm(corrected - nominal) > self.maximum_correction_m + 1e-10:
            raise ValueError("correction exceeds immutable-nominal budget")
        self._applications += 1
        # Replanning after tiny motions must not keep resetting the validation
        # baseline before enough measured displacement accumulates to test it.
        if self._applied is None:
            self._applied = {"state": self._state[:3].copy(), "views": list(proposal["views"]),
                "errors": {v: self._sample(v)["error"].copy() for v in proposal["views"]},
                "bases": copy.deepcopy(proposal["bases"]), "matrices": copy.deepcopy(proposal["matrices"])}
        self._post_application.update(status="awaiting_measured_motion")
        return self._refresh()
