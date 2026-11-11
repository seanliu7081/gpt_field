"""Conservative image-based planar guidance from allowlisted observations.

This is a local, uncalibrated visual servo, not a pose estimator or a task
success detector. Two immutable patches are tracked in the FIXED external RGB
camera. Their interpretation is supplied explicitly by the policy: the target
is the desired projection of the same rigid point, at the same working height.
Coincidence of a bowl patch and a plate patch does not establish 3-D placement.

The image Jacobian is learned from measured end-effector XY motion, never from
commands or simulator object coordinates. No correction is returned until the
observed motion excites both planar directions and passes conditioning and
residual checks. Any lost patch or changed control assumption invalidates the
epoch until a new configure() call.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .action import _rotvec_quaternion
from .observation import PolicyObservation, decode_image


LOCAL_VISUAL_VERSION = "paired-rgb-planar-jacobian-v1"
ANCHOR_SEMANTICS = "same_rigid_point_desired_projection"


class LocalVisualFeedback:
    """Track a pair of explicit RGB anchors and estimate a bounded XY correction.

    ``goal_reference`` has ``kind`` (``end_effector`` or ``held_object``) and
    ``visual_anchor`` with normalized ``moving_point_uv`` and ``target_point_uv``.
    Calibration additionally requires ``local_alignment`` with
    ``semantic=ANCHOR_SEMANTICS`` and ``planar_alignment=True``. Without these
    assertions, only tracking is provided. The assertions are model assumptions,
    not facts verified by this module. Held-object use additionally
    assumes a rigid grasp; a changed finger position or inconsistent response
    invalidates it, but undetectable slip remains possible.

    ``reliable`` only means that both patches passed image-tracking checks;
    ``calibration_valid`` also requires the numerical fit. The caller must enable
    calibration only in a planar alignment phase and, for held objects, with a
    closed gripper and a rigid-grasp hypothesis. ``task_success_certified`` is
    always False.
    """

    def __init__(self, *, patch_radius: int = 5, search_radius: int = 24,
                 minimum_score: float = .8, uniqueness_margin: float = .1,
                 minimum_texture_std: float = 8., minimum_motion_m: float = .001,
                 minimum_samples: int = 4, minimum_excitation_m: float = .003,
                 maximum_motion_condition: float = 15., maximum_jacobian_condition: float = 15.,
                 maximum_residual_px: float = 1.25, minimum_sensitivity_px_per_m: float = 20.,
                 maximum_sensitivity_px_per_m: float = 5000., maximum_correction_m: float = .01,
                 maximum_height_change_m: float = .003, maximum_orientation_change_rad: float = .035,
                 maximum_gripper_change_m: float = .002, maximum_local_radius_m: float = .06,
                 stationary_drift_px: float = 1.5, image_tolerance_px: float = 1.5,
                 maximum_samples: int = 24):
        for name, value, low in (("patch_radius", patch_radius, 2),
                                 ("search_radius", search_radius, patch_radius + 1),
                                 ("minimum_samples", minimum_samples, 3),
                                 ("maximum_samples", maximum_samples, minimum_samples)):
            if type(value) is not int or value < low:
                raise ValueError(f"{name} must be an integer >= {low}")
        positive = {
            "minimum_score": minimum_score, "uniqueness_margin": uniqueness_margin,
            "minimum_texture_std": minimum_texture_std, "minimum_motion_m": minimum_motion_m,
            "minimum_excitation_m": minimum_excitation_m, "maximum_motion_condition": maximum_motion_condition,
            "maximum_jacobian_condition": maximum_jacobian_condition,
            "maximum_residual_px": maximum_residual_px,
            "minimum_sensitivity_px_per_m": minimum_sensitivity_px_per_m,
            "maximum_sensitivity_px_per_m": maximum_sensitivity_px_per_m,
            "maximum_correction_m": maximum_correction_m, "maximum_height_change_m": maximum_height_change_m,
            "maximum_orientation_change_rad": maximum_orientation_change_rad,
            "maximum_gripper_change_m": maximum_gripper_change_m,
            "maximum_local_radius_m": maximum_local_radius_m,
            "stationary_drift_px": stationary_drift_px, "image_tolerance_px": image_tolerance_px,
        }
        for name, value in positive.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (minimum_score > 1 or uniqueness_margin > 2 or maximum_motion_condition < 1
                or maximum_jacobian_condition < 1
                or minimum_sensitivity_px_per_m >= maximum_sensitivity_px_per_m
                or maximum_correction_m > maximum_local_radius_m):
            raise ValueError("invalid visual-feedback thresholds")
        self.patch_radius, self.search_radius = patch_radius, search_radius
        self.minimum_samples, self.maximum_samples = minimum_samples, maximum_samples
        for name, value in positive.items():
            setattr(self, name, float(value))
        self._tracks: dict[str, dict] = {}
        self._unavailable_reason = "unconfigured"
        self._reference_kind: str | None = None
        self._motion_samples: list[np.ndarray] = []
        self._pixel_samples: list[np.ndarray] = []
        self._jacobian: np.ndarray | None = None
        self._calibration_requested = False

    @staticmethod
    def _observation(observation: PolicyObservation) -> PolicyObservation:
        if not isinstance(observation, PolicyObservation):
            raise TypeError("local feedback only accepts PolicyObservation")
        return observation

    @staticmethod
    def _gray(observation: PolicyObservation) -> np.ndarray:
        return np.asarray(decode_image(observation.image).convert("L"), dtype=float)

    def _diagnostic(self, status: str, **extra: Any) -> dict:
        return {
            "version": LOCAL_VISUAL_VERSION,
            "status": status,
            "reliable": False,
            "calibration_valid": False,
            "goal_correction_xy_m": None,
            "pixel_error": None,
            "confidence": 0.,
            "task_success_certified": False,
            "reference_kind": self._reference_kind,
            "sample_count": len(self._motion_samples),
            "calibration_domain": ({
                "origin_xyz_m": self._initial_state[:3].tolist(),
                "maximum_xy_radius_m": self.maximum_local_radius_m,
                "maximum_height_change_m": self.maximum_height_change_m,
            } if self._tracks and self._calibration_requested and hasattr(self, "_initial_state") else None),
            "assumptions": ["fixed external camera", "same rigid point at the desired projection",
                            "constant working height and orientation",
                            "rigid grasp" if self._reference_kind == "held_object" else "point fixed to end effector"],
            "limitations": ["RGB projection agreement does not certify 3-D alignment, contact, or task success",
                            "anchor semantics and rigid attachment are assumptions; undetected slip remains possible",
                            "Jacobian is valid only locally; no depth, object pose, or camera calibration is available"],
            **extra,
        }

    def invalidate(self, reason: str = "control_assumption_changed") -> dict:
        """Invalidate after caller-known release/contact/rotation/slip uncertainty.

        This method reads no additional environment data. The caller must disable
        this guidance when its motion phase no longer satisfies the assumptions.
        """
        if not isinstance(reason, str) or not reason:
            raise ValueError("invalidation reason must be a nonempty string")
        self._tracks = {}
        self._jacobian = None
        self._motion_samples = []
        self._pixel_samples = []
        self._calibration_requested = False
        self._unavailable_reason = reason
        return self._diagnostic(reason)

    def validate_metric_goal(self, goal_xyz) -> bool:
        """Check a proposed world-frame goal before adopting an XY correction.

        The displacement bound on ``goal_correction_xy_m`` alone does not keep
        its endpoint inside the region where the mapping was calibrated. The
        caller must also check this domain and its own geometry/workspace
        constraints. This method reads no new observations and changes no state;
        the latest observe() must already have validated the current pose and
        orientation. An unavailable mapping never validates a goal.
        """
        if not self._tracks or not self._calibration_requested or self._jacobian is None:
            return False
        try:
            goal = np.asarray(goal_xyz, dtype=float)
        except (ValueError, TypeError, OverflowError):
            return False
        if goal.shape != (3,) or not np.isfinite(goal).all():
            return False
        return bool(
            np.linalg.norm(goal[:2] - self._initial_state[:2]) <= self.maximum_local_radius_m + 1e-12
            and abs(goal[2] - self._initial_state[2]) <= self.maximum_height_change_m + 1e-12
        )

    def _match(self, pixels: np.ndarray, track: dict) -> tuple[np.ndarray | None, dict]:
        x, y = (int(round(v)) for v in track["position"])
        radius, search = self.patch_radius, self.search_radius
        height, width = pixels.shape
        left, right = max(radius, x - search), min(width - radius - 1, x + search)
        top, bottom = max(radius, y - search), min(height - radius - 1, y + search)
        if right < left or bottom < top:
            return None, {"status": "out_of_view"}
        crop = pixels[top-radius:bottom+radius+1, left-radius:right+radius+1]
        patches = np.lib.stride_tricks.sliding_window_view(crop, (2*radius+1, 2*radius+1))
        means = patches.mean(axis=(-2, -1))
        energies = np.maximum(np.sum(patches*patches, axis=(-2, -1)) - means*means*track["template"].size, 0.)
        denominators = np.sqrt(energies) * track["norm"]
        numerator = np.einsum("ijkl,kl->ij", patches, track["template"], optimize=False)
        scores = np.divide(numerator, denominators, out=np.full(means.shape, -1.), where=denominators > 1e-10)
        row, col = np.unravel_index(int(np.argmax(scores)), scores.shape)
        score = float(scores[row, col])
        if score < self.minimum_score:
            return None, {"status": "match_lost", "score": score}
        alternatives = scores.copy()
        exclusion = max(2, radius // 2)
        alternatives[max(0, row-exclusion):row+exclusion+1,
                     max(0, col-exclusion):col+exclusion+1] = -1.
        second = float(alternatives.max())
        if score - second < self.uniqueness_margin:
            return None, {"status": "ambiguous_match", "score": score, "alternative_score": second}
        position = np.array([left+col, top+row], dtype=float)
        # A peak on the search edge cannot distinguish a successful match from
        # an object that moved beyond the search region.
        if max(abs(position - track["position"])) >= search:
            return None, {"status": "search_boundary", "score": score}
        return position, {"status": "tracked", "score": score, "uniqueness": score-second}

    def configure(self, observation: PolicyObservation, goal_reference: dict | None) -> dict:
        allowed = self._observation(observation)
        self.invalidate("visual_anchor_unavailable")
        self._reference_kind = None
        if goal_reference is None:
            return self._diagnostic(self._unavailable_reason)
        if not isinstance(goal_reference, dict) or goal_reference.get("kind") not in ("end_effector", "held_object"):
            return self.invalidate("unsupported_goal_reference")
        self._reference_kind = goal_reference["kind"]
        anchor = goal_reference.get("visual_anchor")
        if not isinstance(anchor, dict):
            return self._diagnostic(self._unavailable_reason)
        alignment = goal_reference.get("local_alignment")
        self._calibration_requested = (isinstance(alignment, dict)
                                       and alignment.get("semantic") == ANCHOR_SEMANTICS
                                       and alignment.get("planar_alignment") is True)
        pixels = self._gray(allowed)
        self._shape = pixels.shape
        height, width = pixels.shape
        qualities = {}
        for name, key in (("moving", "moving_point_uv"), ("target", "target_point_uv")):
            value = anchor.get(key)
            if (not isinstance(value, (list, tuple)) or len(value) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                           or v < 0 or v > 1 for v in value)):
                return self.invalidate("invalid_visual_anchor")
            point = np.rint(np.array(value) * [width-1, height-1])
            x, y = (int(v) for v in point)
            radius = self.patch_radius
            if min(x, y) < radius or x+radius >= width or y+radius >= height:
                return self.invalidate(f"{name}_anchor_at_image_boundary")
            patch = pixels[y-radius:y+radius+1, x-radius:x+radius+1].copy()
            if float(patch.std()) < self.minimum_texture_std:
                return self.invalidate(f"{name}_textureless_anchor")
            template = patch - patch.mean()
            track = {"position": point, "template": template, "norm": float(np.linalg.norm(template))}
            matched, diagnostic = self._match(pixels, track)
            if matched is None:
                return self.invalidate(f"{name}_{diagnostic['status']}")
            self._tracks[name] = track
            qualities[name] = diagnostic
        # Two overlapping templates may follow one object rather than observe an
        # independent target. Reject this before accumulating calibration data.
        if np.linalg.norm(self._tracks["moving"]["position"] - self._tracks["target"]["position"], ord=np.inf) <= 2*self.patch_radius:
            return self.invalidate("overlapping_anchors")
        self._initial_state = np.array(allowed.state, dtype=float)
        self._sample_state = self._initial_state.copy()
        self._sample_moving = self._tracks["moving"]["position"].copy()
        self._quaternion = _rotvec_quaternion(self._initial_state[3:6])
        error = self._tracks["target"]["position"] - self._tracks["moving"]["position"]
        self._last_diagnostic = self._diagnostic(
            "awaiting_excitation" if self._calibration_requested else "tracking_only",
            reliable=True, pixel_error=error.tolist(), pixel_error_norm=float(np.linalg.norm(error)),
            moving_point_px=self._tracks["moving"]["position"].tolist(),
            target_point_px=self._tracks["target"]["position"].tolist(),
            tracking=qualities, confidence=float(min(value["score"] for value in qualities.values())))
        self._unavailable_reason = "configured"
        return copy.deepcopy(self._last_diagnostic)

    def observe(self, observation: PolicyObservation) -> dict:
        allowed = self._observation(observation)
        if not self._tracks:
            return self._diagnostic(self._unavailable_reason)
        state = np.array(allowed.state, dtype=float)
        if self._calibration_requested and abs(state[2] - self._initial_state[2]) > self.maximum_height_change_m:
            return self.invalidate("height_changed")
        q = _rotvec_quaternion(state[3:6])
        angle = 2*math.acos(float(np.clip(abs(q @ self._quaternion), 0., 1.)))
        if self._calibration_requested and angle > self.maximum_orientation_change_rad:
            return self.invalidate("orientation_changed")
        if (self._calibration_requested and self._reference_kind == "held_object"
                and max(abs(state[6:] - self._initial_state[6:])) > self.maximum_gripper_change_m):
            return self.invalidate("gripper_changed")
        if self._calibration_requested and np.linalg.norm(state[:2] - self._initial_state[:2]) > self.maximum_local_radius_m:
            return self.invalidate("outside_local_calibration_region")
        pixels = self._gray(allowed)
        if pixels.shape != self._shape:
            return self.invalidate("image_shape_changed")
        matches, qualities = {}, {}
        for name, track in self._tracks.items():
            point, quality = self._match(pixels, track)
            if point is None:
                result = self.invalidate(f"{name}_{quality['status']}")
                result["tracking"] = {name: quality}
                return result
            matches[name], qualities[name] = point, quality
        delta_xy = state[:2] - self._sample_state[:2]
        delta_pixel = matches["moving"] - self._sample_moving
        pixel_error = matches["target"] - matches["moving"]
        diagnostic = {
            "pixel_error": pixel_error.tolist(),
            "pixel_error_norm": float(np.linalg.norm(pixel_error)),
            "moving_point_px": matches["moving"].tolist(),
            "target_point_px": matches["target"].tolist(),
            "tracking": qualities,
            "confidence": float(min(q["score"] for q in qualities.values())),
            "reliable": True,
        }
        for name in matches:
            self._tracks[name]["position"] = matches[name]
        if not self._calibration_requested:
            return self._diagnostic("tracking_only", **diagnostic)
        if np.linalg.norm(delta_xy) < self.minimum_motion_m:
            if np.linalg.norm(delta_pixel) > self.stationary_drift_px:
                result = self.invalidate("point_moved_without_robot_motion")
                result.update(diagnostic)
                return result
        else:
            if self._jacobian is not None:
                prediction_error = float(np.linalg.norm(delta_pixel - self._jacobian @ delta_xy))
                if prediction_error > 2*self.maximum_residual_px:
                    result = self.invalidate("jacobian_prediction_inconsistent")
                    result.update(diagnostic, prediction_error_px=prediction_error)
                    return result
            self._motion_samples.append(delta_xy)
            self._pixel_samples.append(delta_pixel)
            self._motion_samples = self._motion_samples[-self.maximum_samples:]
            self._pixel_samples = self._pixel_samples[-self.maximum_samples:]
            self._sample_state = state.copy()
            self._sample_moving = matches["moving"].copy()
        if len(self._motion_samples) < self.minimum_samples:
            return self._diagnostic("awaiting_excitation", **diagnostic)
        movements = np.array(self._motion_samples)
        responses = np.array(self._pixel_samples)
        excitation = np.linalg.svd(movements, compute_uv=False)
        if excitation[-1] < self.minimum_excitation_m:
            self._jacobian = None
            return self._diagnostic("insufficient_planar_excitation", **diagnostic,
                                    excitation_singular_values_m=excitation.tolist())
        if excitation[0] / excitation[-1] > self.maximum_motion_condition:
            self._jacobian = None
            return self._diagnostic("ill_conditioned_motion", **diagnostic)
        mapping = np.linalg.lstsq(movements, responses, rcond=None)[0]
        residual = float(np.sqrt(np.mean(np.sum((movements @ mapping - responses)**2, axis=1))))
        if residual > self.maximum_residual_px:
            result = self.invalidate("inconsistent_planar_response")
            result.update(diagnostic, residual_px=residual)
            return result
        sensitivities = np.linalg.svd(mapping, compute_uv=False)
        if (sensitivities[-1] < self.minimum_sensitivity_px_per_m
                or sensitivities[0] > self.maximum_sensitivity_px_per_m
                or sensitivities[0] / sensitivities[-1] > self.maximum_jacobian_condition):
            self._jacobian = None
            return self._diagnostic("ill_conditioned_image_jacobian", **diagnostic,
                                    jacobian_singular_values_px_per_m=sensitivities.tolist())
        self._jacobian = mapping.T
        correction = np.linalg.solve(self._jacobian, pixel_error)
        norm = float(np.linalg.norm(correction))
        # Clipping applies only to the proposed local displacement, not the goal
        # or calibration data. The caller still performs geometry checks/planning.
        if norm > self.maximum_correction_m:
            correction *= self.maximum_correction_m / norm
        if np.linalg.norm(pixel_error) <= self.image_tolerance_px:
            correction[:] = 0.
        return self._diagnostic("local_guidance_ready", **diagnostic, calibration_valid=True,
                                goal_correction_xy_m=correction.tolist(),
                                correction_was_bounded=norm > self.maximum_correction_m,
                                jacobian_px_per_m=self._jacobian.tolist(), residual_px=residual,
                                excitation_singular_values_m=excitation.tolist())
