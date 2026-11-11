"""Observable goal grounding and a conservative fixed-camera consistency check.

Object offsets are model estimates from RGB, not simulator poses. The checker
tracks an immutable image patch; it cannot certify grasp, geometry, or success.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np

from .action import _quaternion_product, _rotvec_quaternion
from .astra import SchemaError, _numbers
from .compact_field import compact_field_schema, validate_compact_field
from .observation import PolicyObservation, decode_image


GROUNDED_FIELD_VERSION = "analytic-goal-rbf-grounded-v2"
_REFERENCE_KEYS = {"kind", "object_center_offset_m", "desired_object_center_offset_m", "visual_anchor"}
_ANCHOR_KEYS = {"moving_point_uv", "target_point_uv"}
FIELD_EVENT_REASONS = frozenset({"visual_divergence", "goal_reached", "field_boundary_reached",
                                 "invalid_direction", "no_progress"})


def _vector_schema(length: int, minimum: float, maximum: float) -> dict:
    return {"type": "array", "minItems": length, "maxItems": length,
            "items": {"type": "number", "minimum": minimum, "maximum": maximum}}


def grounded_compact_field_schema() -> dict:
    schema = compact_field_schema()
    vector = _vector_schema(3, -.5, .5)
    anchor = {"type": "object", "additionalProperties": False, "required": sorted(_ANCHOR_KEYS),
              "properties": {key: _vector_schema(2, 0., 1.) for key in sorted(_ANCHOR_KEYS)}}
    reference = {"type": "object", "additionalProperties": False, "required": sorted(_REFERENCE_KEYS),
                 "properties": {
                     "kind": {"type": "string", "enum": ["end_effector", "held_object"]},
                     "object_center_offset_m": {"anyOf": [vector, {"type": "null"}]},
                     "desired_object_center_offset_m": {"anyOf": [copy.deepcopy(vector), {"type": "null"}]},
                     "visual_anchor": {"anyOf": [anchor, {"type": "null"}]},
                 }}
    schema["properties"]["goal_reference"] = {"anyOf": [reference, {"type": "null"}]}
    schema["required"].append("goal_reference")
    return schema


def _bounded(value: Any, count: int, label: str, low: float, high: float) -> list[float]:
    result = _numbers(value, count, label)
    if any(number < low or number > high for number in result):
        raise ValueError(f"{label} must be within [{low}, {high}]")
    return result


def validate_goal_reference(reference: Any) -> None:
    if reference is None:
        return
    if not isinstance(reference, dict) or set(reference) != _REFERENCE_KEYS:
        raise SchemaError("goal_reference has missing or unexpected fields")
    try:
        if reference["kind"] not in ("end_effector", "held_object"):
            raise ValueError("goal_reference.kind must be end_effector or held_object")
        for key in ("object_center_offset_m", "desired_object_center_offset_m"):
            if reference["kind"] == "held_object":
                _bounded(reference[key], 3, key, -.5, .5)
            elif reference[key] is not None:
                raise ValueError(f"end_effector reference requires {key}=null")
        anchor = reference["visual_anchor"]
        if anchor is not None:
            if not isinstance(anchor, dict) or set(anchor) != _ANCHOR_KEYS:
                raise ValueError("visual_anchor has missing or unexpected fields")
            for key in _ANCHOR_KEYS:
                _bounded(anchor[key], 2, key, 0., 1.)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(str(exc)) from exc


def validate_grounded_compact_field(data: Any) -> dict:
    if not isinstance(data, dict) or "goal_reference" not in data:
        raise SchemaError("grounded compact field requires goal_reference")
    validate_compact_field({key: value for key, value in data.items() if key != "goal_reference"})
    validate_goal_reference(data["goal_reference"])
    return data


def _observation(value: PolicyObservation | Mapping[str, Any]) -> PolicyObservation:
    return value if isinstance(value, PolicyObservation) else PolicyObservation.from_dict(value)


def effective_compact_field(data: Any, observation: PolicyObservation | Mapping[str, Any]) -> tuple[dict, dict]:
    """Return a legacy field and audit record, without mutating model output.

    p_object = p_eef + offset_current. For a rigid held object, the final offset
    is R_target R_current.T offset_current, so eef_goal = object_goal - offset_final.
    Slip and visual estimation errors remain possible and require new observations.
    """
    validate_grounded_compact_field(data)
    allowed = _observation(observation)
    result = copy.deepcopy({key: value for key, value in data.items() if key != "goal_reference"})
    reference = data["goal_reference"]
    diagnostics = {"version": GROUNDED_FIELD_VERSION, "reference_kind":
                   reference["kind"] if reference is not None else "unspecified",
                   "raw_goal_offset_m": list(data["goal_offset_m"]),
                   "effective_goal_offset_m": list(data["goal_offset_m"]),
                   "rotation_compensated": False, "goal_reference": copy.deepcopy(reference)}
    if reference is not None and reference["kind"] == "held_object":
        current = _rotvec_quaternion(np.asarray(allowed.state[3:6], dtype=float))
        target = np.asarray(data["orientation_target_xyzw"], dtype=float)
        target = target / np.linalg.norm(target)
        relative = _quaternion_product(target, np.r_[-current[:3], current[3]])
        offset = np.asarray(reference["object_center_offset_m"], dtype=float)
        rotated = offset + 2 * np.cross(relative[:3],
                                       np.cross(relative[:3], offset) + relative[3] * offset)
        goal = np.asarray(reference["desired_object_center_offset_m"], dtype=float) - rotated
        # Reject rather than clip: clipping would hide inconsistent model estimates.
        if not np.isfinite(goal).all() or np.any(np.abs(goal) > .5):
            raise SchemaError("derived held-object end-effector goal must stay within +/-0.5 m per axis")
        result["goal_offset_m"] = goal.tolist()
        diagnostics.update(effective_goal_offset_m=goal.tolist(), rotation_compensated=True,
                           rotated_object_center_offset_m=rotated.tolist())
    validate_compact_field(result)
    return result, diagnostics


class VisualGoalConsistencyChecker:
    """NCC patch tracking in the permitted fixed external camera only.

    This is an opt-in heuristic, not calibrated perception. A null/occluded,
    textureless, ambiguous, or lost anchor produces no refresh request. Two
    consecutive reliable worsening observations with measured robot motion are
    required; model waiting and repeated identical robot states do not count.
    """

    def __init__(self, *, patch_radius: int = 5, search_radius: int = 28,
                 minimum_score: float = .75, uniqueness_margin: float = .08,
                 minimum_texture_std: float = 8., worsening_px: float = 2.,
                 minimum_motion_m: float = .001, consecutive_checks: int = 2):
        if (type(patch_radius) is not int or patch_radius < 2 or type(search_radius) is not int
                or search_radius <= patch_radius or type(consecutive_checks) is not int
                or consecutive_checks < 2):
            raise ValueError("invalid visual consistency window or check count")
        for name, value in (("minimum_score", minimum_score), ("uniqueness_margin", uniqueness_margin),
                            ("minimum_texture_std", minimum_texture_std), ("worsening_px", worsening_px),
                            ("minimum_motion_m", minimum_motion_m)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if minimum_score > 1 or uniqueness_margin > 2:
            raise ValueError("NCC thresholds exceed their valid range")
        self.patch_radius, self.search_radius = patch_radius, search_radius
        self.minimum_score, self.uniqueness_margin = minimum_score, uniqueness_margin
        self.minimum_texture_std, self.worsening_px = minimum_texture_std, worsening_px
        self.minimum_motion_m, self.consecutive_checks = minimum_motion_m, consecutive_checks
        self._template = None
        self._unavailable_reason = "unconfigured"

    @staticmethod
    def _gray(observation: PolicyObservation) -> np.ndarray:
        return np.asarray(decode_image(observation.image).convert("L"), dtype=np.float64)

    def _diagnostic(self, status: str, **extra) -> dict:
        return {"status": status, "refresh_requested": False, "reason": None, **extra}

    def _match(self, pixels: np.ndarray) -> tuple[np.ndarray | None, dict]:
        x, y = (int(round(v)) for v in self._position)
        radius, search = self.patch_radius, self.search_radius
        height, width = pixels.shape
        left, right = max(radius, x-search), min(width-radius-1, x+search)
        top, bottom = max(radius, y-search), min(height-radius-1, y+search)
        if right < left or bottom < top:
            return None, self._diagnostic("out_of_view")
        crop = pixels[top-radius:bottom+radius+1, left-radius:right+radius+1]
        patches = np.lib.stride_tricks.sliding_window_view(crop, (2*radius+1, 2*radius+1))
        means = patches.mean(axis=(-2, -1))
        energies = np.maximum(np.sum(patches*patches, axis=(-2, -1)) - means*means*self._template.size, 0)
        numerators = np.einsum("ijkl,kl->ij", patches, self._template, optimize=False)
        denominators = np.sqrt(energies) * self._template_norm
        scores = np.divide(numerators, denominators, out=np.full(means.shape, -1.), where=denominators > 1e-10)
        row, col = np.unravel_index(int(np.argmax(scores)), scores.shape)
        score = float(scores[row, col])
        if score < self.minimum_score:
            return None, self._diagnostic("match_lost", match_score=score)
        alternatives = scores.copy()
        # Nearby pixels represent the same peak; distinct strong peaks are ambiguous.
        exclusion = max(2, radius // 2)
        alternatives[max(0, row-exclusion):row+exclusion+1,
                     max(0, col-exclusion):col+exclusion+1] = -1.
        second = float(alternatives.max())
        if score - second < self.uniqueness_margin:
            return None, self._diagnostic("ambiguous_match", match_score=score, alternative_score=second)
        return np.array([left+col, top+row], dtype=float), self._diagnostic("tracked", match_score=score)

    def configure(self, observation: PolicyObservation | Mapping[str, Any], goal_reference: Any) -> dict:
        self._template = None
        self._unavailable_reason = "visual_anchor_unavailable"
        self._worsening_count = 0
        validate_goal_reference(goal_reference)
        if goal_reference is None or goal_reference["visual_anchor"] is None:
            return self._diagnostic(self._unavailable_reason)
        allowed = _observation(observation)
        pixels = self._gray(allowed)
        height, width = pixels.shape
        anchor = goal_reference["visual_anchor"]
        self._shape = pixels.shape
        self._position = np.rint(np.asarray(anchor["moving_point_uv"]) * [width-1, height-1])
        self._target = np.asarray(anchor["target_point_uv"]) * [width-1, height-1]
        x, y = (int(v) for v in self._position)
        radius = self.patch_radius
        if min(x, y) < radius or x+radius >= width or y+radius >= height:
            self._unavailable_reason = "anchor_at_image_boundary"
            return self._diagnostic(self._unavailable_reason)
        template = pixels[y-radius:y+radius+1, x-radius:x+radius+1].copy()
        if float(template.std()) < self.minimum_texture_std:
            self._unavailable_reason = "textureless_anchor"
            return self._diagnostic(self._unavailable_reason)
        self._template = template - template.mean()
        self._template_norm = float(np.linalg.norm(self._template))
        matched, diagnostic = self._match(pixels)
        if matched is None:
            self._template = None
            self._unavailable_reason = diagnostic["status"]
            return diagnostic
        self._baseline_distance = float(np.linalg.norm(self._position-self._target))
        self._previous_eef = np.asarray(allowed.state[:3], dtype=float)
        return self._diagnostic("configured", baseline_distance_px=self._baseline_distance)

    def observe(self, observation: PolicyObservation | Mapping[str, Any]) -> dict:
        if self._template is None:
            return self._diagnostic(self._unavailable_reason)
        allowed = _observation(observation)
        pixels = self._gray(allowed)
        if pixels.shape != self._shape:
            self._worsening_count = 0
            return self._diagnostic("image_shape_changed")
        matched, diagnostic = self._match(pixels)
        if matched is None:
            self._worsening_count = 0
            return diagnostic
        position = np.asarray(allowed.state[:3], dtype=float)
        motion = float(np.linalg.norm(position-self._previous_eef))
        distance = float(np.linalg.norm(matched-self._target))
        self._position = matched
        diagnostic.update(current_distance_px=distance, baseline_distance_px=self._baseline_distance,
                          measured_eef_motion_m=motion, moving_point_uv=
                          (matched / np.array([pixels.shape[1]-1, pixels.shape[0]-1])).tolist())
        if motion < self.minimum_motion_m:
            self._worsening_count = 0
            diagnostic.update(status="insufficient_robot_motion", worsening_checks=0)
            return diagnostic
        self._previous_eef = position
        self._worsening_count = (self._worsening_count + 1
                                if distance > self._baseline_distance + self.worsening_px else 0)
        diagnostic["worsening_checks"] = self._worsening_count
        if self._worsening_count >= self.consecutive_checks:
            diagnostic.update(refresh_requested=True, reason="visual_divergence")
        return diagnostic
