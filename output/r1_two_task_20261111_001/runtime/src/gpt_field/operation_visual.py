"""Persistent, model-seeded RGB observations for local manipulation operations.

Sparse optical flow measures image motion, not object pose or contact. Region
identity is a model hypothesis; texture agreement cannot establish that it was
correct. This module neither selects actions nor certifies grasp/task success.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

from .observation import PolicyObservation, decode_image
from .action import _rotvec_quaternion

OPERATION_VISUAL_VERSION = "operation-rgb-flow-v1"
_VIEWS = ("external", "wrist")
_ROLES = ("object", "tool", "target")


def tracking_reference_schema():
    box = {"type": "array", "minItems": 4, "maxItems": 4,
           "items": {"type": "number", "minimum": 0., "maximum": 1.}}
    view = {"type": "object", "additionalProperties": False,
            "required": [role + "_roi_uv" for role in _ROLES],
            "properties": {role + "_roi_uv": {"anyOf": [{"type": "null"}, copy.deepcopy(box)]}
                           for role in _ROLES}}
    return {"anyOf": [{"type": "null"}, {"type": "object", "additionalProperties": False,
        "required": ["reference_id", "external", "wrist"], "properties": {
            "reference_id": {"type": "string", "minLength": 1, "maxLength": 64},
            **{view_name: {"anyOf": [{"type": "null"}, copy.deepcopy(view)]} for view_name in _VIEWS}}}]}


def validate_tracking_reference(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"reference_id", "external", "wrist"}:
        raise ValueError("tracking_reference fields are reference_id,external,wrist")
    identifier = value["reference_id"]
    if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 64:
        raise ValueError("tracking reference_id must contain 1..64 characters")
    if all(value[name] is None for name in _VIEWS):
        raise ValueError("tracking_reference needs at least one view")
    for name in _VIEWS:
        view = value[name]
        if view is None:
            continue
        if not isinstance(view, dict) or set(view) != {role + "_roi_uv" for role in _ROLES}:
            raise ValueError("tracking view needs the three nullable role ROIs")
        if all(box is None for box in view.values()):
            raise ValueError("nonnull tracking view needs a role ROI")
        for box in view.values():
            if box is None:
                continue
            if (not isinstance(box, (tuple, list)) or len(box) != 4
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(float(x)) for x in box)):
                raise ValueError("tracking ROI needs four finite real numbers")
            a = np.asarray(box)
            if np.any(a < 0.) or np.any(a > 1.) or np.any(a[:2] >= a[2:]):
                raise ValueError("tracking ROI must be ordered inside [0,1]")
    return copy.deepcopy(value)


def _backend():
    # Import only when this opt-in observer is configured. Missing dependency
    # must not change the historical/default-off controller behavior.
    import cv2
    return cv2


def _gray(observation):
    return {"external": np.asarray(decode_image(observation.image).convert("L")),
            "wrist": np.asarray(decode_image(observation.wrist_image).convert("L"))}


def _overlap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    intersection = float(np.prod(np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))))
    return intersection / max(1e-12, min(float(np.prod(a[2:] - a[:2])), float(np.prod(b[2:] - b[:2]))))


class _RegionFlow:
    """Short-lived material features, without re-detection or identity guessing.

    Features are seeded inside an eroded model ROI. Forward/backward checks and
    robust similarity transport tolerate modest perspective/lighting changes.
    Pyramid flow can use surrounding pixels: this is not a segmentation mask.
    Lost features are never replaced automatically with a neighboring body.
    """
    maximum_age = 120

    def __init__(self, image, roi_uv, cv):
        self.cv = cv
        self.shape = image.shape
        self.scale = np.array([image.shape[1] - 1, image.shape[0] - 1], dtype=float)
        self.roi_uv = list(roi_uv)
        self.roi = np.asarray(roi_uv).reshape(2, 2) * self.scale
        self.center = self.roi.mean(axis=0)
        self.origin = self.center.copy()
        self.step_displacement = np.zeros(2)
        self.age = 0
        self.valid = True
        self.reason = "seeded_model_region"
        self.uncertainty = 1.
        self.transform = np.eye(3)
        self.fb_error = self.residual = 0.
        self.inlier_fraction = 1.
        width = float(np.min(self.roi[1] - self.roi[0]))
        self.window = 5 if width < 18 else 9
        radius = self.window // 2
        lo = np.maximum(np.ceil(self.roi[0]).astype(int) + radius, radius)
        hi = np.minimum(np.floor(self.roi[1]).astype(int) - radius,
                        np.array(image.shape[::-1]) - radius - 1)
        if np.any(hi - lo < 3):
            raise ValueError("region_too_small_for_flow")
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1] = 255
        points = cv.goodFeaturesToTrack(image, maxCorners=48, qualityLevel=.025,
                                       minDistance=2.5, mask=mask, blockSize=3)
        if points is None or len(points) < 4:
            raise ValueError("insufficient_region_texture")
        self.points = points.astype(np.float32)
        self.seed_count = len(points)

    def invalidate(self, reason):
        self.valid, self.reason = False, reason

    def update(self, before, after):
        if not self.valid:
            return
        if after.shape != self.shape or before.shape != self.shape:
            return self.invalidate("image_shape_changed")
        if self.age >= self.maximum_age:
            return self.invalidate("local_tracking_epoch_expired")
        cv = self.cv
        params = dict(winSize=(self.window, self.window), maxLevel=2,
                      criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 25, .01),
                      minEigThreshold=1e-4)
        moved, status, errors = cv.calcOpticalFlowPyrLK(before, after, self.points, None, **params)
        if moved is None:
            return self.invalidate("forward_flow_unavailable")
        backward, reverse_status, _ = cv.calcOpticalFlowPyrLK(after, before, moved, None, **params)
        if backward is None:
            return self.invalidate("backward_flow_unavailable")
        source, destination = self.points.reshape(-1, 2), moved.reshape(-1, 2)
        fb = np.linalg.norm(backward.reshape(-1, 2) - source, axis=1)
        finite = np.isfinite(destination).all(axis=1) & np.isfinite(fb)
        good = (status.ravel() > 0) & (reverse_status.ravel() > 0) & finite & (fb <= 1.2)
        good &= errors.ravel() <= 28.
        good &= (destination[:, 0] >= 2.) & (destination[:, 0] < self.shape[1] - 2.)
        good &= (destination[:, 1] >= 2.) & (destination[:, 1] < self.shape[0] - 2.)
        if int(good.sum()) < 4 or float(good.mean()) < .35:
            return self.invalidate("flow_support_lost")
        a, b = source[good], destination[good]
        # RANSAC does not establish semantic identity; it only rejects features
        # inconsistent with the dominant local image transform.
        affine, inliers = cv.estimateAffinePartial2D(a, b, method=cv.RANSAC,
            ransacReprojThreshold=1.5, maxIters=300, confidence=.99, refineIters=5)
        if affine is None or inliers is None:
            return self.invalidate("local_transform_unavailable")
        keep = inliers.ravel().astype(bool)
        if int(keep.sum()) < 4 or float(keep.mean()) < .65:
            return self.invalidate("inconsistent_body_motion")
        singular = np.linalg.svd(affine[:, :2], compute_uv=False)
        rotation = math.atan2(affine[1, 0], affine[0, 0])
        if np.min(singular) < .85 or np.max(singular) > 1.18 or abs(rotation) > .32:
            return self.invalidate("local_transform_outside_validity")
        motion = np.eye(3)
        motion[:2] = affine
        cumulative = motion @ self.transform
        scales = np.linalg.svd(cumulative[:2, :2], compute_uv=False)
        if min(scales) < .5 or max(scales) > 2.:
            return self.invalidate("appearance_scale_epoch_expired")
        center = affine[:, :2] @ self.center + affine[:, 2]
        if np.linalg.norm(center - self.center) > 40.:
            return self.invalidate("image_jump_outside_local_domain")
        self.fb_error = float(np.median(fb[good][keep]))
        prediction = a[keep] @ affine[:, :2].T + affine[:, 2]
        self.residual = float(np.sqrt(np.mean(np.sum((prediction - b[keep]) ** 2, axis=1))))
        self.uncertainty += .04 + .25 * self.fb_error + .15 * self.residual
        self.inlier_fraction = float(keep.mean())
        self.step_displacement = center - self.center
        self.center = center
        self.transform = cumulative
        self.points = b[keep].reshape(-1, 1, 2).astype(np.float32)
        self.age += 1
        self.reason = "tracked_image_features"

    def snapshot(self):
        return {"valid": self.valid, "reason": self.reason,
            "center_uv": (self.center / self.scale).tolist(),
            "displacement_px": (self.center - self.origin).tolist(),
            "step_displacement_px": self.step_displacement.tolist(),
            "support_count": int(len(self.points)), "seed_support_count": self.seed_count,
            "age": self.age, "uncertainty_px": self.uncertainty,
            "forward_backward_error_px": self.fb_error, "fit_residual_px": self.residual,
            "inlier_fraction": self.inlier_fraction,
            "measurement": "model_region_feature_transport_not_object_pose"}


class OperationVisualFeedback:
    """Independent camera observations; optional test-lift co-motion evidence.

    Same ID and reference preserves tracks through model calls. Null explicitly
    clears the reference. A changed ROI with the same ID is rejected rather than
    silently transferring evidence to another surface. No observation requests,
    actions, hidden environment data, response-model fitting or success checks.
    """
    def __init__(self):
        self._reference = None
        self._operation_kind = None
        self._target_id = None
        self._tracks = {}
        self._unavailable = {}
        self._last = None
        self._images = None
        self._status = "not_configured"
        self._verification = None
        self._verification_evidence = "unknown"
        self._verification_reason = "verification_not_requested"
        self._measured_displacement = 0.
        self._elapsed_s = 0.
        self._frames = 0
        self._failures = 0
        self._epoch = 0

    def configure(self, observation, operation, visual_intent=None):
        started = time.perf_counter()
        try:
            if not isinstance(observation, PolicyObservation):
                raise ValueError("visual feedback accepts only allowlisted policy observation")
            self._operation_kind = (operation or {}).get("kind", (operation or {}).get("type"))
            target_id = (operation or {}).get("target_id")
            if self._operation_kind not in ("verify_grasp", "transport", "place_align"):
                self.end_verification("operation_changed_verification_ended")
            reference = validate_tracking_reference((visual_intent or {}).get("tracking_reference"))
            if reference is None:
                self._clear("no_model_tracking_reference")
                return self.snapshot()
            if self._reference is not None and reference["reference_id"] == self._reference["reference_id"]:
                if target_id != self._target_id:
                    self._invalidate("operation_target_changed_requires_new_reference")
                elif reference != self._reference:
                    self._invalidate("same_id_reference_changed_requires_new_id")
                elif self._last is not None and observation != self._last:
                    self.observe(self._last, observation)
                return self.snapshot()
            self._clear("initializing")
            self._reference = reference
            self._target_id = target_id
            self._epoch += 1
            cv = _backend()
            images = _gray(observation)
            for view in _VIEWS:
                for role in _ROLES:
                    box = (reference[view] or {}).get(role + "_roi_uv")
                    if box is None:
                        continue
                    key = (view, role)
                    try:
                        self._tracks[key] = _RegionFlow(images[view], box, cv)
                    except (ValueError, cv.error) as exc:
                        self._unavailable[key] = str(exc)
            self._last, self._images = observation, images
            self._status = "tracking" if any(t.valid for t in self._tracks.values()) else "no_usable_regions"
        except ImportError:
            self._clear("optical_flow_backend_unavailable")
            self._failures += 1
        except (ValueError, TypeError) as exc:
            self._clear("invalid_visual_reference:" + str(exc))
            self._failures += 1
        finally:
            self._elapsed_s += time.perf_counter() - started
        return self.snapshot()

    def _clear(self, reason):
        self._reference = None
        self._target_id = None
        self._tracks = {}
        self._unavailable = {}
        self._last = self._images = None
        self._verification = None
        self._verification_evidence = "unknown"
        self._verification_reason = "verification_not_requested"
        self._measured_displacement = 0.
        self._status = reason

    def _invalidate(self, reason):
        for track in self._tracks.values():
            track.invalidate(reason)
        self._status = reason
        self._verification = None
        self._verification_evidence = "unknown"
        self._verification_reason = reason

    def begin_verification(self, observation=None):
        """Called explicitly at the start of an already-authorized test motion."""
        if observation is not None and self._last is not None and observation != self._last:
            self.observe(self._last, observation)
        self._verification_evidence = "unknown"
        self._verification_reason = "insufficient_observed_verification_motion"
        self._measured_displacement = 0.
        if self._last is None:
            self._verification = None
            self._verification_reason = "no_visual_reference"
        else:
            self._verification = {"state": np.asarray(self._last.state), "frames": self._frames,
                                  "centers": {key: t.center.copy() for key, t in self._tracks.items() if t.valid}}
        return self.snapshot()

    def end_verification(self, reason="verification_ended"):
        """Clear attachment evidence without discarding independent visual tracks."""
        self._verification = None
        self._verification_evidence = "unknown"
        self._verification_reason = reason
        self._measured_displacement = 0.
        return self.snapshot()

    def observe(self, before, after):
        started = time.perf_counter()
        try:
            if self._last is None or after == self._last:
                return self.snapshot()
            if before != self._last:
                self._invalidate("observation_discontinuity_requires_new_reference")
                self._last = after
                return self.snapshot()
            images = _gray(after)
            for (view, _role), track in self._tracks.items():
                if track.valid:
                    try:
                        track.update(self._images[view], images[view])
                    except Exception as exc:
                        # Backend/texture failure is diagnostic and never blocks
                        # an otherwise authorized controller action.
                        track.invalidate("optical_flow_error:" + type(exc).__name__)
                        self._failures += 1
            self._last, self._images = after, images
            self._frames += 1
            self._status = "tracking" if any(t.valid for t in self._tracks.values()) else "all_regions_lost"
            self._update_verification()
        except (ValueError, TypeError) as exc:
            self._invalidate("invalid_visual_observation:" + str(exc))
            self._failures += 1
        finally:
            self._elapsed_s += time.perf_counter() - started
        return self.snapshot()

    def _update_verification(self):
        self._verification_evidence = "unknown"
        if self._verification is None:
            return
        baseline = self._verification
        state = np.asarray(self._last.state)
        self._measured_displacement = float(np.linalg.norm(state[:3] - baseline["state"][:3]))
        current_quaternion = _rotvec_quaternion(state[3:6])
        baseline_quaternion = _rotvec_quaternion(baseline["state"][3:6])
        rotation_angle = 2. * math.acos(float(np.clip(
            abs(np.dot(current_quaternion, baseline_quaternion)), 0., 1.)))
        if rotation_angle > .15:
            self._verification_reason = "verification_rotation_changed_relation"
            return
        if np.linalg.norm(state[6:] - baseline["state"][6:]) > .012:
            self._verification_reason = "verification_finger_geometry_changed"
            return
        keys = [("external", role) for role in ("object", "tool")]
        if any(key not in baseline["centers"] or key not in self._tracks or not self._tracks[key].valid for key in keys):
            self._verification_reason = "fixed_view_object_and_tool_unavailable_wrist_is_indeterminate"
            return
        if _overlap(self._reference["external"]["object_roi_uv"], self._reference["external"]["tool_roi_uv"]) > .25:
            self._verification_reason = "object_tool_support_overlap_is_ambiguous"
            return
        if self._frames - baseline["frames"] < 3 or self._measured_displacement < .005:
            self._verification_reason = "insufficient_observed_verification_motion"
            return
        obj, tool = [self._tracks[key].center - baseline["centers"][key] for key in keys]
        obj_norm, tool_norm = float(np.linalg.norm(obj)), float(np.linalg.norm(tool))
        uncertainty = max(self._tracks[key].uncertainty for key in keys)
        if tool_norm < max(4., 2. * uncertainty):
            self._verification_reason = "tool_image_motion_below_uncertainty"
            return
        relative = float(np.linalg.norm(obj - tool))
        ratio = obj_norm / tool_norm
        cosine = float(np.dot(obj, tool) / max(1e-6, obj_norm * tool_norm))
        if (obj_norm >= max(3., 1.5 * uncertainty) and .5 <= ratio <= 1.6 and cosine >= .8
                and relative <= max(2.5, .35 * tool_norm)):
            self._verification_evidence = "supporting"
            self._verification_reason = "fixed_view_object_tool_co_motion_supports_attachment_not_certification"
        elif obj_norm <= 1.5 and tool_norm >= max(5., 3. * uncertainty):
            self._verification_evidence = "contradicted"
            self._verification_reason = "tool_moved_while_selected_object_remained_image_stationary"
        elif relative >= max(5., .8 * tool_norm + uncertainty):
            self._verification_evidence = "contradicted"
            self._verification_reason = "selected_object_motion_diverged_from_tool"
        else:
            self._verification_reason = "image_motion_relation_ambiguous"

    def snapshot(self):
        views = {view: {} for view in _VIEWS}
        for (view, role), track in self._tracks.items():
            views[view][role] = track.snapshot()
        for (view, role), reason in self._unavailable.items():
            views[view][role] = {"valid": False, "reason": reason}
        active = [view for view in _VIEWS if any(item["valid"] for item in views[view].values())]
        target = self._tracks.get(("external", "target"))
        target_motion = None
        if target is not None and target.valid and target.age >= 3:
            target_motion = float(np.linalg.norm(target.center - target.origin)) > max(4., 2. * target.uncertainty)
        return {"version": OPERATION_VISUAL_VERSION, "status": self._status,
            "reference_id": None if self._reference is None else self._reference["reference_id"],
            "operation_kind": self._operation_kind, "epoch": self._epoch,
            "target_id": self._target_id,
            "tracking_valid": bool(active), "active_views": active, "views": views,
            "verification_active": self._verification is not None,
            "verification_evidence": self._verification_evidence,
            "verification_reason": self._verification_reason,
            "measured_robot_displacement_m": self._measured_displacement,
            "external_target_image_motion_observed": target_motion,
            "tracking_elapsed_s": self._elapsed_s, "observed_transitions": self._frames,
            "tracking_errors": self._failures,
            "grasp_certified": False, "task_success_certified": False,
            "limitations": ["Model-selected ROI identity is unverified; optical flow supplies image motion only.",
                "Sparse flow is not segmentation and can follow a wrong surface or background.",
                "External view is assumed fixed; wrist-only co-motion cannot verify attachment.",
                "Supporting co-motion does not establish stable force closure, contact or task success.",
                "Lost regions are not silently reseeded; a new model reference is needed."]}
