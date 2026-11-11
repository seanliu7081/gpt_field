"""Validated, explicit contracts for model-produced local potential fields.

``values`` are arbitrary-scale, lower-is-better potentials. They are not
asserted to be distances or to have a physical unit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


def vector3(value: Any, name: str) -> tuple[float, float, float]:
    """Copy a finite 3-vector into an immutable Python tuple."""
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 3-vector") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return tuple(float(x) for x in result)


@dataclass(frozen=True)
class GridSpec:
    size: int = 9
    spacing_m: float = 0.02
    frame: str = "world"

    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, Integral) or self.size < 2:
            raise ValueError("grid size must be an integer >= 2")
        if (
            isinstance(self.spacing_m, bool)
            or not isinstance(self.spacing_m, Real)
            or not np.isfinite(self.spacing_m)
            or self.spacing_m <= 0
        ):
            raise ValueError("grid spacing_m must be finite and positive")
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("grid frame must be a nonempty string")
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "spacing_m", float(self.spacing_m))

    @property
    def count(self) -> int:
        return self.size**3

    @property
    def half_width_m(self) -> float:
        return (self.size - 1) * self.spacing_m / 2.0

    def coordinates(self, center: Any) -> np.ndarray:
        """Coordinates in i*N*N + j*N + k order, with x varying slowest."""
        origin = np.asarray(vector3(center, "center"))
        offsets = (np.arange(self.size) - (self.size - 1) / 2.0) * self.spacing_m
        return np.stack(np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 3) + origin

    def to_dict(self) -> dict[str, Any]:
        return {"size": self.size, "spacing_m": self.spacing_m, "frame": self.frame}


@dataclass(frozen=True)
class FieldPrediction:
    stage: str
    orientation_target_xyzw: tuple[float, float, float, float]
    gripper_command: str
    motion_mode: str
    values: tuple[float, ...]
    blocked_indices: tuple[int, ...]
    unknown_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a nonempty string")
        if self.gripper_command not in ("open", "close", "hold"):
            raise ValueError("gripper_command must be open, close, or hold")
        if self.motion_mode not in ("move", "hold"):
            raise ValueError("motion_mode must be move or hold")
        try:
            orientation = tuple(self.orientation_target_xyzw)
        except TypeError as exc:
            raise ValueError("orientation_target_xyzw must contain four finite numbers") from exc
        if len(orientation) != 4 or not all(_finite_number(x) for x in orientation):
            raise ValueError("orientation_target_xyzw must contain four finite numbers")
        orientation = tuple(float(x) for x in orientation)
        if abs(float(np.linalg.norm(orientation)) - 1.0) > 1e-3:
            raise ValueError("orientation_target_xyzw must have unit norm within 0.001")
        object.__setattr__(self, "orientation_target_xyzw", orientation)
        try:
            values = tuple(self.values)
        except TypeError as exc:
            raise ValueError("values must be a sequence of finite numbers") from exc
        if not all(_finite_number(x) for x in values):
            raise ValueError("all values must be finite numbers, including masked placeholders")
        object.__setattr__(self, "values", tuple(float(x) for x in values))
        for name in ("blocked_indices", "unknown_indices"):
            try:
                indices = tuple(getattr(self, name))
            except TypeError as exc:
                raise ValueError(f"{name} must be a sequence of integer indices") from exc
            if not all(isinstance(i, Integral) and not isinstance(i, bool) and i >= 0 for i in indices):
                raise ValueError(f"{name} must contain nonnegative integer indices")
            if len(indices) != len(set(indices)):
                raise ValueError(f"{name} must not contain duplicate indices")
            object.__setattr__(self, name, tuple(int(i) for i in indices))
        if set(self.blocked_indices).intersection(self.unknown_indices):
            raise ValueError("blocked_indices and unknown_indices must be disjoint")

    def validate_grid(self, grid: GridSpec) -> None:
        if len(self.values) != grid.count:
            raise ValueError(f"values must contain exactly {grid.count} entries")
        if any(i >= grid.count for i in self.blocked_indices + self.unknown_indices):
            raise ValueError("masked index is outside the grid")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], grid: GridSpec) -> FieldPrediction:
        if not isinstance(data, Mapping):
            raise ValueError("field prediction must be an object")
        required = {
            "stage", "orientation_target_xyzw", "gripper_command", "motion_mode",
            "values", "blocked_indices", "unknown_indices",
        }
        if set(data) != required:
            missing = sorted(required - set(data))
            extra = sorted(set(data) - required, key=str)
            raise ValueError(f"field prediction fields differ: missing={missing}, extra={extra}")
        result = cls(**dict(data))
        result.validate_grid(grid)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "orientation_target_xyzw": list(self.orientation_target_xyzw),
            "gripper_command": self.gripper_command,
            "motion_mode": self.motion_mode,
            "values": list(self.values),
            "blocked_indices": list(self.blocked_indices),
            "unknown_indices": list(self.unknown_indices),
        }


def _finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and bool(np.isfinite(value))
