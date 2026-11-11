"""Expand model-predicted analytic parameters into the existing dense field.

All positions here are world-axis offsets from the end-effector position at
field generation. The representation uses no simulator geometry or object pose.
Its values remain arbitrary-scale, lower-is-better potentials, not distances.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .astra import SchemaError, _numbers
from .schema import FieldPrediction, GridSpec


COMPACT_FIELD_VERSION = "analytic-goal-rbf-v1"
_AUXILIARY = ("stage", "orientation_target_xyzw", "gripper_command", "motion_mode")
_KEYS = frozenset((*_AUXILIARY, "goal_offset_m", "attraction_weights", "repulsors",
                   "blocked_boxes", "unknown_boxes"))
_REPULSOR_KEYS = frozenset(("center_offset_m", "radius_m", "weight"))
_BOX_KEYS = frozenset(("lower_offset_m", "upper_offset_m"))
_LENGTH_SCALE_M = 0.08


def _vector_schema(minimum: float | None = None, maximum: float | None = None,
                   length: int = 3) -> dict[str, Any]:
    item = {"type": "number"}
    if minimum is not None:
        item["minimum"] = minimum
    if maximum is not None:
        item["maximum"] = maximum
    return {"type": "array", "items": item, "minItems": length, "maxItems": length}


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def compact_field_schema() -> dict[str, Any]:
    """Strict structured-output schema for an analytic attraction/repulsion field."""
    repulsor = _object_schema({
        "center_offset_m": _vector_schema(-0.5, 0.5),
        "radius_m": {"type": "number", "minimum": 0.005, "maximum": 0.5},
        "weight": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
    })

    def boxes():
        return {"type": "array", "maxItems": 4, "items": _object_schema({
            "lower_offset_m": _vector_schema(-0.5, 0.5),
            "upper_offset_m": _vector_schema(-0.5, 0.5),
        })}

    return _object_schema({
        "stage": {"type": "string", "minLength": 1},
        "orientation_target_xyzw": _vector_schema(length=4),
        "gripper_command": {"type": "string", "enum": ["open", "close", "hold"]},
        "motion_mode": {"type": "string", "enum": ["move", "hold"]},
        "goal_offset_m": _vector_schema(-0.5, 0.5),
        "attraction_weights": _vector_schema(0.1, 10),
        "repulsors": {"type": "array", "maxItems": 4, "items": repulsor},
        "blocked_boxes": boxes(),
        "unknown_boxes": boxes(),
    })


def _bounded_vector(value: Any, label: str, minimum: float, maximum: float) -> list[float]:
    result = _numbers(value, 3, label)
    if any(number < minimum or number > maximum for number in result):
        raise ValueError(f"{label} must be within [{minimum}, {maximum}] on every axis")
    return result


def _objects(value: Any, label: str, keys: frozenset[str]):
    if not isinstance(value, (list, tuple)) or len(value) > 4:
        raise ValueError(f"{label} must be an array containing at most four objects")
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError(f"{label}[{index}] has missing or unexpected fields")
        yield index, item


def validate_compact_field(data: Any) -> dict[str, Any]:
    """Validate without clipping, normalizing, or altering the model's parameters."""
    if not isinstance(data, dict) or set(data) != _KEYS:
        raise SchemaError("compact field has missing or unexpected fields")
    try:
        # Reuse the original auxiliary-head semantics without allocating a grid.
        orientation = _numbers(data["orientation_target_xyzw"], 4, "orientation_target_xyzw")
        FieldPrediction(stage=data["stage"], orientation_target_xyzw=tuple(orientation),
                        gripper_command=data["gripper_command"], motion_mode=data["motion_mode"],
                        values=(), blocked_indices=(), unknown_indices=())
        _bounded_vector(data["goal_offset_m"], "goal_offset_m", -0.5, 0.5)
        _bounded_vector(data["attraction_weights"], "attraction_weights", 0.1, 10)
        for index, repulsor in _objects(data["repulsors"], "repulsors", _REPULSOR_KEYS):
            prefix = f"repulsors[{index}]"
            _bounded_vector(repulsor["center_offset_m"], prefix + ".center_offset_m", -0.5, 0.5)
            radius = _numbers([repulsor["radius_m"]], 1, prefix + ".radius_m")[0]
            weight = _numbers([repulsor["weight"]], 1, prefix + ".weight")[0]
            if not 0.005 <= radius <= 0.5:
                raise ValueError(f"{prefix}.radius_m must be within [0.005, 0.5]")
            if not 0 < weight <= 10:
                raise ValueError(f"{prefix}.weight must be positive and at most 10")
        for label in ("blocked_boxes", "unknown_boxes"):
            for index, box in _objects(data[label], label, _BOX_KEYS):
                prefix = f"{label}[{index}]"
                lower = _bounded_vector(box["lower_offset_m"], prefix + ".lower_offset_m", -0.5, 0.5)
                upper = _bounded_vector(box["upper_offset_m"], prefix + ".upper_offset_m", -0.5, 0.5)
                if any(low > high for low, high in zip(lower, upper)):
                    raise ValueError(f"{prefix} lower_offset_m must not exceed upper_offset_m")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SchemaError(str(exc)) from exc
    return data


def expand_compact_field(data: Any, grid: GridSpec) -> dict[str, Any]:
    """Evaluate the analytic potential and predicted boxes at every grid vertex.

    The offset array uses GridSpec's i*N*N+j*N+k ordering (x changes slowest).
    Blocked boxes take precedence over overlapping unknown boxes. Repulsion
    affects potential only; it never silently adds collision-mask geometry.
    """
    validate_compact_field(data)
    if not isinstance(grid, GridSpec):
        raise ValueError("grid must be a GridSpec")
    offsets = grid.coordinates((0., 0., 0.))
    difference = (offsets - np.asarray(data["goal_offset_m"], dtype=float)) / _LENGTH_SCALE_M
    values = np.sqrt(np.sum(np.asarray(data["attraction_weights"], dtype=float) * difference ** 2, axis=1))
    for repulsor in data["repulsors"]:
        difference = offsets - np.asarray(repulsor["center_offset_m"], dtype=float)
        squared_distance = np.sum(difference ** 2, axis=1)
        values += repulsor["weight"] * np.exp(-0.5 * squared_distance / repulsor["radius_m"] ** 2)

    def mask(boxes):
        result = np.zeros(grid.count, dtype=bool)
        for box in boxes:
            lower, upper = np.asarray(box["lower_offset_m"]), np.asarray(box["upper_offset_m"])
            result |= np.all((offsets >= lower) & (offsets <= upper), axis=1)
        return result

    blocked = mask(data["blocked_boxes"])
    unknown = mask(data["unknown_boxes"]) & ~blocked
    result = {key: data[key] for key in _AUXILIARY}
    result.update(values=values.tolist(), blocked_indices=np.flatnonzero(blocked).tolist(),
                  unknown_indices=np.flatnonzero(unknown).tolist())
    return FieldPrediction.from_dict(result, grid).to_dict()
