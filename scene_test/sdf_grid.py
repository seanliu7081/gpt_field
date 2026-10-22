"""Vertex-sampled Euclidean scene SDF; independent of all robot code.

Arrays use values[x_index, y_index, z_index]. Distances and coordinates are
metres. Queries outside the recorded ROI return NaN, never free space.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class SDFGrid:
    values: np.ndarray
    spacing: float
    lower: np.ndarray
    vertex_quantization_max_m: float | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float32)
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.spacing = float(self.spacing)
        if self.values.ndim != 3 or min(self.values.shape) < 2:
            raise ValueError("values must be a three-dimensional vertex grid")
        if self.lower.shape != (3,) or not np.all(np.isfinite(self.lower)):
            raise ValueError("lower must be a finite XYZ vector")
        if self.spacing <= 0 or not np.isfinite(self.spacing):
            raise ValueError("spacing must be positive and finite")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("grid contains non-finite distances")

    @property
    def upper(self):
        return self.lower + (np.asarray(self.values.shape) - 1) * self.spacing

    @classmethod
    def build(cls, scene, spacing, lower=(-0.12, -0.12, -0.12),
              upper=(0.12, 0.12, 0.12), chunk_size=250_000):
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        intervals = (upper - lower) / float(spacing)
        if np.any(intervals <= 0) or not np.allclose(intervals, np.round(intervals), atol=1e-10, rtol=0):
            raise ValueError("ROI lengths must be positive integer multiples of spacing")
        shape = tuple((np.rint(intervals).astype(int) + 1).tolist())
        values = np.empty(shape, dtype=np.float32)
        flat = values.ravel()
        quantization_max = 0.
        for first in range(0, len(flat), chunk_size):
            ids = np.arange(first, min(first + chunk_size, len(flat)), dtype=np.int64)
            coords = np.column_stack(np.unravel_index(ids, shape)).astype(np.float64)
            points = lower + coords * spacing
            distance = np.asarray(scene.sdf(points), dtype=np.float64)
            if distance.shape != (len(points),) or not np.all(np.isfinite(distance)):
                raise ValueError("reference returned invalid distances")
            flat[first:first + len(ids)] = distance
            quantization_max = max(quantization_max, float(np.max(np.abs(flat[first:first + len(ids)].astype(float) - distance))))
        return cls(values, spacing, lower, quantization_max)

    def query(self, points, chunk_size=250_000):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("query points must have shape (N,3)")
        result = np.full(len(points), np.nan, dtype=np.float64)
        for first in range(0, len(points), chunk_size):
            p = points[first:first + chunk_size]
            inside = np.all(np.isfinite(p) & (p >= self.lower) & (p <= self.upper), axis=1)
            if not inside.any():
                continue
            q = (p[inside] - self.lower) / self.spacing
            index = np.floor(q).astype(np.int64)
            index = np.minimum(index, np.asarray(self.values.shape) - 2)
            # Only roundoff can escape this range after the domain check.
            # Convex weights matter to later conditional interpolation bounds.
            frac = np.clip(q - index, 0.0, 1.0)
            v = np.zeros(len(index), dtype=np.float64)
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        corner = np.array([dx, dy, dz])
                        weight = np.prod(np.where(corner, frac, 1.0 - frac), axis=1)
                        c = index + corner
                        v += weight * self.values[c[:, 0], c[:, 1], c[:, 2]]
            result[first + np.flatnonzero(inside)] = v
        return result

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, values=self.values, spacing_m=self.spacing,
                            lower_m=self.lower, upper_m=self.upper,
                            vertex_quantization_max_m=np.nan if self.vertex_quantization_max_m is None else self.vertex_quantization_max_m,
                            axis_order="xyz", units="m", negative="inside_material")

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data["axis_order"]) != "xyz" or str(data["units"]) != "m":
                raise ValueError("unsupported grid coordinate convention")
            grid = cls(data["values"], float(data["spacing_m"]), data["lower_m"])
            if "vertex_quantization_max_m" in data:
                error = float(data["vertex_quantization_max_m"])
                grid.vertex_quantization_max_m = error if np.isfinite(error) else None
            if not np.allclose(grid.upper, data["upper_m"], atol=1e-14, rtol=0):
                raise ValueError("saved ROI and grid shape disagree")
            return grid

    def surface(self):
        """Marching cubes in XYZ axis order; empty output returns None."""
        if self.values.min() >= 0 or self.values.max() <= 0:
            return None
        import trimesh
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(
            self.values, level=0, spacing=(self.spacing,) * 3,
            gradient_direction="descent", method="lewiner", allow_degenerate=False)
        return trimesh.Trimesh(vertices=verts + self.lower, faces=faces, process=False)


def build(scene, spacing, **kwargs):
    return SDFGrid.build(scene, spacing, **kwargs)
