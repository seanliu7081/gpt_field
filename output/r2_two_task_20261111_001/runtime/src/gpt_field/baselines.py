"""Deterministic cached-field ablations preserving commands and masks."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any

import numpy as np

from .field import GridField
from .schema import FieldPrediction, GridSpec, vector3


@lru_cache(maxsize=256)
def _affine_values(
    grid: GridSpec,
    values: tuple[float, ...],
    blocked: tuple[int, ...],
    unknown: tuple[int, ...],
) -> tuple[float, ...]:
    valid = np.ones(grid.count, dtype=bool)
    if blocked or unknown:
        valid[np.asarray(blocked + unknown, dtype=int)] = False
    if not valid.any():
        raise ValueError("affine baseline requires at least one valid grid value")
    # Center-relative, spacing-normalized coordinates improve conditioning and
    # make the fit/cache independent of the absolute field center.
    coordinates = grid.coordinates((0.0, 0.0, 0.0)) / grid.spacing_m
    design = np.column_stack((coordinates, np.ones(grid.count)))
    original = np.asarray(values)
    coefficients, _, _, _ = np.linalg.lstsq(design[valid], original[valid], rcond=None)
    fitted = original.copy()
    fitted[valid] = design[valid] @ coefficients
    if not np.isfinite(fitted).all():
        raise ValueError("affine fit produced nonfinite values")
    return tuple(float(value) for value in fitted)


def transform_field(field: GridField, method: str, seed: int = 7) -> GridField:
    """Transform values from one cached prediction; retain all other outputs.

    ``affine`` removes curvature but retains the least-squares linear trend.
    ``shuffled`` permutes valid values only. ``minimum`` replaces the values
    with Euclidean distance to the lowest valid sampled grid point.
    If no node is valid, every supported mode preserves the original field so
    the planner reports the same no-valid-candidate outcome across ablations.
    """
    if method not in ("field", "affine", "shuffled", "minimum"):
        raise ValueError(f"unknown field baseline: {method!r}")
    if method == "field":
        return field
    prediction = field.prediction
    valid = np.asarray(field.valid_indices, dtype=int)
    if len(valid) == 0:
        return field
    if method == "affine":
        values = _affine_values(field.grid, prediction.values, prediction.blocked_indices, prediction.unknown_indices)
    elif method == "shuffled":
        updated = np.asarray(prediction.values).copy()
        updated[valid] = np.random.default_rng(seed).permutation(updated[valid])
        values = tuple(float(value) for value in updated)
    elif method == "minimum":
        updated = np.asarray(prediction.values).copy()
        coordinates = field.grid.coordinates(field.center)
        target_index = int(valid[np.argmin(updated[valid])])
        updated[valid] = np.linalg.norm(coordinates[valid] - coordinates[target_index], axis=1)
        values = tuple(float(value) for value in updated)
    return GridField(field.grid, field.center, replace(prediction, values=values))


def waypoint_field(
    grid: GridSpec,
    center: Any,
    aux: dict[str, Any],
    target_xyz: Any,
) -> GridField:
    """Build an analytic waypoint potential with the supplied auxiliary outputs.

    No target is inferred or read from the simulator here. The caller supplies
    the waypoint (model-predicted for a live baseline; known in synthetic tests).
    """
    center = vector3(center, "center")
    target = np.asarray(vector3(target_xyz, "target_xyz"))
    required = {"stage", "orientation_target_xyzw", "gripper_command", "motion_mode"}
    optional = {"blocked_indices", "unknown_indices"}
    # Accept the auxiliary dictionary alone or either full model-output shape.
    # The primary output is supplied by target_xyz, not copied into the schema.
    primary = {"values", "waypoint_xyz"}
    if not isinstance(aux, dict) or not required.issubset(aux) or set(aux) - required - optional - primary:
        raise ValueError(f"waypoint aux requires {sorted(required)} and optional {sorted(optional | primary)}")
    data = {key: value for key, value in aux.items() if key in required | optional}
    data.setdefault("blocked_indices", [])
    data.setdefault("unknown_indices", [])
    data["values"] = np.linalg.norm(grid.coordinates(center) - target, axis=1).tolist()
    return GridField(grid, center, FieldPrediction.from_dict(data, grid))
