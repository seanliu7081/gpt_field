"""Fixed-center trilinear fields and complete interpolation-cell traversal.

Validity refers only to the predicted mask. It does not certify actual scene
geometry, robot-link clearance, contact feasibility, or controller tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from itertools import product
from typing import Any

import numpy as np

from .schema import FieldPrediction, GridSpec, vector3


# Dimensionless lattice-coordinate tolerance for floating-point grid boundaries.
_GRID_EPS = 1e-10


@dataclass(frozen=True)
class GridField:
    grid: GridSpec
    center: tuple[float, float, float]
    prediction: FieldPrediction
    _values: np.ndarray = dataclass_field(init=False, repr=False, compare=False)
    _valid: np.ndarray = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.prediction.validate_grid(self.grid)
        object.__setattr__(self, "center", vector3(self.center, "center"))
        values = np.asarray(self.prediction.values).reshape((self.grid.size,) * 3)
        valid = np.ones(self.grid.count, dtype=bool)
        invalid = self.prediction.blocked_indices + self.prediction.unknown_indices
        if invalid:
            valid[np.asarray(invalid, dtype=int)] = False
        valid = valid.reshape((self.grid.size,) * 3)
        values.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_valid", valid)

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.center) - self.grid.half_width_m

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.center) + self.grid.half_width_m

    @property
    def valid_indices(self) -> tuple[int, ...]:
        return tuple(int(i) for i in np.flatnonzero(self._valid.ravel()))

    def query(self, position: Any) -> float | None:
        """Interpolate only when every nonzero-weight corner is valid.

        An exact valid grid vertex or face does not use off-face corners.
        Invalid/malformed positions and positions outside the cube return None.
        """
        try:
            point = np.asarray(vector3(position, "position"))
        except ValueError:
            return None
        lattice = (point - self.lower) / self.grid.spacing_m
        if np.any(lattice < -_GRID_EPS) or np.any(lattice > self.grid.size - 1 + _GRID_EPS):
            return None
        lattice = np.clip(lattice, 0.0, self.grid.size - 1.0)
        axes: list[list[tuple[int, float]]] = []
        for coordinate in lattice:
            nearest = int(round(float(coordinate)))
            if abs(float(coordinate) - nearest) <= _GRID_EPS:
                axes.append([(nearest, 1.0)])
            else:
                low = int(np.floor(coordinate))
                fraction = float(coordinate - low)
                axes.append([(low, 1.0 - fraction), (low + 1, fraction)])
        result = 0.0
        for corner in product(*axes):
            index = tuple(item[0] for item in corner)
            if not self._valid[index]:
                return None
            weight = corner[0][1] * corner[1][1] * corner[2][1]
            result += weight * float(self._values[index])
        return float(result) if np.isfinite(result) else None

    def query_many(self, positions: Any) -> list[float | None]:
        return [self.query(position) for position in positions]

    def segment_valid(self, start: Any, end: Any) -> bool:
        """Check every interpolation stencil traversed by a line segment.

        Grid-plane crossings partition the segment into intervals with a
        constant active corner set. Endpoints and one midpoint per interval
        therefore check the entire segment, including grid-aligned segments.
        This is stronger than fixed-spacing sampling, which can miss short
        crossings near a cell corner.
        """
        try:
            a = np.asarray(vector3(start, "start"))
            b = np.asarray(vector3(end, "end"))
        except ValueError:
            return False
        if self.query(a) is None or self.query(b) is None:
            return False
        delta = b - a
        crossings = {0.0, 1.0}
        lower = self.lower
        for axis in range(3):
            if delta[axis] == 0.0:
                continue
            planes = lower[axis] + np.arange(self.grid.size) * self.grid.spacing_m
            for plane in planes:
                parameter = float((plane - a[axis]) / delta[axis])
                if 0.0 < parameter < 1.0:
                    crossings.add(parameter)
        ordered = sorted(crossings)
        for parameter in ordered:
            if self.query(a + parameter * delta) is None:
                return False
        for left, right in zip(ordered[:-1], ordered[1:]):
            if self.query(a + ((left + right) / 2.0) * delta) is None:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid.to_dict(),
            "center": list(self.center),
            "prediction": self.prediction.to_dict(),
        }
