"""Optional, local object-relative RGB guidance for Speed-first v3.

The model selects two *visible features*, their desired image displacement,
and an alignment epoch. A local measured-motion fit supplies an XY correction
only while the held feature and fixed target remain trackable at the same
working height/orientation. This is not metric object reconstruction, a
success detector, an action generator, or an admission gate for normal Field
motion. In particular, unavailable guidance never requests a probe or a hold.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from .local_visual_feedback import ANCHOR_SEMANTICS, LocalVisualFeedback
from .observation import PolicyObservation, decode_image


SPEED_FIRST_VISUAL_VERSION = "speed-first-object-relative-rgb-v1"
_REFERENCE_KEYS = frozenset({"reference_id", "moving_point_uv", "target_point_uv",
                             "desired_offset_uv", "planar_alignment", "rigid_grasp"})


def visual_reference_schema() -> dict:
    """Strict JSON schema shared by the opt-in client and local controller."""
    point = {"type": "array", "minItems": 2, "maxItems": 2,
             "items": {"type": "number", "minimum": 0., "maximum": 1.}}
    return {"anyOf": [{"type": "null"}, {
        "type": "object", "additionalProperties": False,
        "required": sorted(_REFERENCE_KEYS),
        "properties": {
            "reference_id": {"type": "string", "minLength": 1, "maxLength": 64},
            "moving_point_uv": copy.deepcopy(point),
            "target_point_uv": copy.deepcopy(point),
            "desired_offset_uv": {"type": "array", "minItems": 2, "maxItems": 2,
                                  "items": {"type": "number", "minimum": -.25, "maximum": .25}},
            "planar_alignment": {"type": "boolean", "enum": [True]},
            "rigid_grasp": {"type": "boolean", "enum": [True]},
        },
    }]}


def validate_visual_reference(reference: dict | None) -> dict | None:
    """Validate model intent; flags are hypotheses, not measured grasp facts."""
    if reference is None:
        return None
    if not isinstance(reference, dict) or set(reference) != _REFERENCE_KEYS:
        raise ValueError("visual_reference must have exactly the object-alignment reference keys")
    identifier = reference["reference_id"]
    if not isinstance(identifier, str) or not identifier.strip() or len(identifier) > 64:
        raise ValueError("visual reference_id must be a nonempty string of at most 64 characters")
    if reference["planar_alignment"] is not True or reference["rigid_grasp"] is not True:
        raise ValueError("visual alignment requires explicit planar_alignment and rigid_grasp hypotheses")
    result = copy.deepcopy(reference)
    for name, low, high in (("moving_point_uv", 0., 1.), ("target_point_uv", 0., 1.),
                            ("desired_offset_uv", -.25, .25)):
        value = reference[name]
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number))
                       or not math.isfinite(float(v)) or not low <= v <= high for v in value)):
            raise ValueError(f"{name} must contain two finite numbers in [{low}, {high}]")
        result[name] = [float(v) for v in value]
    return result


class SpeedFirstVisualFeedback:
    """Bounded guidance from a persistent, model-selected held-object feature.

    Call ``observe`` at every completed transition, including ordinary Direct
    motion, to learn from actual XY displacement. ``configure`` with the same
    ID, desired offset and tracked points preserves the fit across model calls.
    A changed ID starts a new epoch. An invalidated ID cannot silently restart:
    the model must deliberately select a new reference ID from a new image.

    ``corrected_goal`` returns an optional world-frame goal for the existing
    Field planner, never an action. Its Z is the *current measured height*.
    A missing fit returns None; the caller continues its ordinary plan. Nothing
    in this module commands exploratory motion, blocks an ordinary action,
    certifies rigid attachment, or marks a task complete.
    """

    def __init__(self, *, tracker: LocalVisualFeedback | None = None,
                 gain: float = .6, maximum_correction_m: float = .01,
                 reference_match_tolerance_px: float = 4.,
                 maximum_target_drift_px: float = 3.):
        for name, value in (("gain", gain), ("maximum_correction_m", maximum_correction_m),
                            ("reference_match_tolerance_px", reference_match_tolerance_px),
                            ("maximum_target_drift_px", maximum_target_drift_px)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if gain > 1. or maximum_correction_m > .01:
            raise ValueError("gain must be <= 1 and correction must be <= 0.01 m")
        self.tracker = tracker or LocalVisualFeedback(
            patch_radius=7, search_radius=32, minimum_score=.75, uniqueness_margin=.05,
            minimum_texture_std=5., maximum_residual_px=2., stationary_drift_px=2.,
            image_tolerance_px=2., maximum_correction_m=maximum_correction_m)
        self.gain = float(gain)
        self.maximum_correction_m = float(maximum_correction_m)
        self.reference_match_tolerance_px = float(reference_match_tolerance_px)
        self.maximum_target_drift_px = float(maximum_target_drift_px)
        self._reference: dict | None = None
        self._epoch = 0
        self._active = False
        self._offset_px: np.ndarray | None = None
        self._origin_target: np.ndarray | None = None
        self._last_error: float | None = None
        self._last_observation: PolicyObservation | None = None
        self._last = self._diagnostic("unconfigured")

    def _diagnostic(self, status: str, **extra: Any) -> dict:
        return {"version": SPEED_FIRST_VISUAL_VERSION, "status": status,
                "reference_id": self._reference["reference_id"] if self._reference else None,
                "epoch": self._epoch, "reliable": False, "calibration_valid": False,
                "goal_correction_xy_m": None, "task_success_certified": False,
                "fallback": "continue_model_selected_field_or_direct",
                "probe_requested": False, **extra}

    def diagnostics(self) -> dict:
        return copy.deepcopy(self._last)

    def invalidate(self, reason: str = "control_assumption_changed") -> dict:
        self.tracker.invalidate(reason)
        self._active = False
        self._last_observation = None
        self._last_error = None
        self._last = self._diagnostic(reason)
        return self.diagnostics()

    def _translate(self, base: dict, observation: PolicyObservation) -> dict:
        """Apply desired feature displacement after fitting the measured J."""
        result = copy.deepcopy(base)
        result.update(version=SPEED_FIRST_VISUAL_VERSION,
                      tracker_version=base.get("version"), epoch=self._epoch,
                      reference_id=self._reference["reference_id"] if self._reference else None,
                      fallback="continue_model_selected_field_or_direct", probe_requested=False,
                      task_success_certified=False, goal_correction_xy_m=None,
                      desired_offset_px=self._offset_px.tolist() if self._offset_px is not None else None)
        result["assumptions"] = ["fixed external camera", "rigidly held visible moving feature",
                                 "stationary target feature", "constant working height and orientation",
                                 "model-selected desired moving-minus-target image offset"]
        # Existing tracking may attach reliable=True to an invalid numerical
        # fit for diagnostics. Its own calibration_valid controls all proposals.
        if base.get("reliable") and base.get("moving_point_px") is not None:
            target = np.asarray(base["target_point_px"], dtype=float)
            if self._origin_target is not None and np.linalg.norm(target-self._origin_target) > self.maximum_target_drift_px:
                return self.invalidate("stationary_target_moved")
            error = target + self._offset_px - np.asarray(base["moving_point_px"], dtype=float)
            error_norm = float(np.linalg.norm(error))
            result.update(pixel_error=error.tolist(), pixel_error_norm=error_norm,
                          image_error_reduction_px=(None if self._last_error is None else self._last_error-error_norm))
            self._last_error = error_norm
            if base.get("calibration_valid"):
                mapping = np.asarray(base["jacobian_px_per_m"], dtype=float)
                raw = self.gain * np.linalg.solve(mapping, error)
                magnitude = float(np.linalg.norm(raw))
                correction = raw * min(1., self.maximum_correction_m / max(magnitude, 1e-15))
                if error_norm <= self.tracker.image_tolerance_px:
                    correction[:] = 0.
                goal = np.asarray(observation.state[:3], dtype=float)
                goal[:2] += correction
                if not self.tracker.validate_metric_goal(goal):
                    result.update(status="correction_outside_local_domain", calibration_valid=False)
                else:
                    result.update(goal_correction_xy_m=correction.tolist(), correction_gain=self.gain,
                                  correction_was_bounded=magnitude > self.maximum_correction_m)
        self._last = result
        return self.diagnostics()

    def configure(self, observation: PolicyObservation, reference: dict | None, *, gripper_closed: bool) -> dict:
        if not isinstance(observation, PolicyObservation):
            raise TypeError("local feedback only accepts PolicyObservation")
        reference = validate_visual_reference(reference)
        if type(gripper_closed) is not bool:
            raise TypeError("gripper_closed must be the caller's boolean closed-command hypothesis")
        if reference is None:
            return self.invalidate("visual_reference_not_requested")
        if not gripper_closed:
            return self.invalidate("closed_gripper_hypothesis_missing")
        width, height = decode_image(observation.image).size
        scale = np.array([width-1, height-1], dtype=float)
        if self._reference is not None and reference["reference_id"] == self._reference["reference_id"]:
            if not self._active:
                self._last["new_reference_id_required"] = True
                return self.diagnostics()
            # The measured fit must first see any new observation. Repeated
            # configuration at one response boundary does not add a sample.
            current = self.observe(observation)
            if not self._active:
                current["new_reference_id_required"] = True
                self._last = current
                return self.diagnostics()
            for key, tracked in (("moving_point_uv", "moving_point_px"),
                                 ("target_point_uv", "target_point_px")):
                if np.linalg.norm(np.array(reference[key])*scale-np.array(current[tracked])) > self.reference_match_tolerance_px:
                    return self.invalidate("reference_id_feature_changed")
            if np.linalg.norm((np.array(reference["desired_offset_uv"])-np.array(self._reference["desired_offset_uv"]))*scale) > 1.:
                return self.invalidate("reference_id_desired_offset_changed")
            # The epoch's image objective is immutable, just like its patch
            # templates. Tolerate subpixel response rounding without silently
            # moving the objective or accumulating small goal drift.
            reference["desired_offset_uv"] = self._reference["desired_offset_uv"].copy()
            self._reference = reference
            self._last["mapping_preserved_across_model_call"] = True
            return self.diagnostics()
        self._epoch += 1
        self._reference = reference
        self._offset_px = np.array(reference["desired_offset_uv"]) * scale
        self._origin_target = None
        self._last_error = None
        base = self.tracker.configure(observation, {
            "kind": "held_object",
            "visual_anchor": {"moving_point_uv": reference["moving_point_uv"],
                              "target_point_uv": reference["target_point_uv"]},
            "local_alignment": {"semantic": ANCHOR_SEMANTICS, "planar_alignment": True}})
        self._active = bool(base.get("reliable"))
        self._last_observation = observation
        if self._active:
            self._origin_target = np.array(base["target_point_px"], dtype=float)
        return self._translate(base, observation)

    def observe(self, observation: PolicyObservation) -> dict:
        if not isinstance(observation, PolicyObservation):
            raise TypeError("local feedback only accepts PolicyObservation")
        if not self._active or observation == self._last_observation:
            return self.diagnostics()
        self._last_observation = observation
        base = self.tracker.observe(observation)
        # Distinguish unavailable excitation (still active) from a permanently
        # invalidated tracker. No private camera or simulator state is read.
        unavailable = self.tracker._unavailable_reason
        if unavailable != "configured":
            self._active = False
        return self._translate(base, observation)

    def corrected_goal(self, observation: PolicyObservation, nominal_goal_xyz) -> tuple[np.ndarray | None, dict]:
        """Return a <=1 cm XY revision at measured Z, or immediately fall back.

        The caller enables this only for its explicit planar-alignment phase.
        Nominal goals that change height are not silently converted to planar
        goals. The ordinary field planner remains responsible for workspace,
        optional obstacles, orientation, path tracking and command limits.
        """
        diagnostic = self.observe(observation)
        try:
            nominal = np.asarray(nominal_goal_xyz, dtype=float)
        except (ValueError, TypeError, OverflowError):
            nominal = np.array([])
        if nominal.shape != (3,) or not np.isfinite(nominal).all():
            raise ValueError("nominal goal must be three finite coordinates")
        if abs(nominal[2]-observation.state[2]) > self.tracker.maximum_height_change_m:
            diagnostic.update(status="nominal_goal_changes_working_height", proposal_available=False)
            return None, diagnostic
        if not self._active or not diagnostic.get("calibration_valid") or diagnostic.get("goal_correction_xy_m") is None:
            diagnostic["proposal_available"] = False
            return None, diagnostic
        goal = np.array(observation.state[:3], dtype=float)
        goal[:2] += diagnostic["goal_correction_xy_m"]
        if not self.tracker.validate_metric_goal(goal):
            diagnostic.update(status="correction_outside_local_domain", proposal_available=False)
            return None, diagnostic
        diagnostic.update(proposal_available=True, proposed_goal_xyz_m=goal.tolist(),
                          proposal_preserves_measured_height=True)
        return goal, diagnostic
