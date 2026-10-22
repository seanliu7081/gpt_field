"""Scale-aware unsigned distance to a continuous triangle mesh.

The broad phase uses trimesh's rtree candidates. Candidate distances use a
separate, normalized face-plane/edge kernel and a strict minimum: no normal
tie-breaking can replace the nearest distance. No library tolerance is changed.
"""
from pathlib import Path
import sys

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent / "_deps"))


def point_triangle_squared_distances(triangles, points):
    """Corresponding point/triangle distances, including degenerate triangles.

    Each triangle is translated and normalized by its longest edge, so region
    predicates have dimensionless tolerances. Vanishing-area faces fall back to
    their closed edges. The discarded altitude in this fallback is at most
    approximately 64 machine epsilons times the longest edge.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape (N,3,3)")
    if points.shape == (3,):
        points = np.broadcast_to(points, (len(triangles), 3))
    if points.shape != (len(triangles), 3):
        raise ValueError("points must have shape (N,3) or (3,)")
    if not np.all(np.isfinite(triangles)) or not np.all(np.isfinite(points)):
        raise ValueError("triangle and query coordinates must be finite")
    if len(triangles) == 0:
        return np.empty(0, dtype=np.float64)
    a, b, c = np.moveaxis(triangles, 1, 0)
    ab0, ac0, bc0 = b-a, c-a, c-b
    dot = lambda u, v: np.einsum("ij,ij->i", u, v)
    scale2 = np.maximum.reduce([dot(ab0, ab0), dot(ac0, ac0), dot(bc0, bc0)])
    scale = np.sqrt(scale2)
    safe_scale = np.where(scale > 0, scale, 1.)
    ab, ac = ab0/safe_scale[:, None], ac0/safe_scale[:, None]
    ap = (points-a)/safe_scale[:, None]
    result = np.full(len(triangles), np.inf)
    zero = np.zeros_like(ab)
    for start, end in ((zero, ab), (zero, ac), (ab, ac)):
        edge = end-start
        edge2 = dot(edge, edge)
        fraction = np.divide(dot(ap-start, edge), edge2, out=np.zeros(len(edge2)), where=edge2 > 0)
        delta = ap-start-np.clip(fraction, 0., 1.)[:, None]*edge
        np.minimum(result, dot(delta, delta), out=result)
    normal = np.cross(ab, ac)
    area2 = dot(normal, normal)
    relative_epsilon = 64*np.finfo(np.float64).eps
    valid = area2 > relative_epsilon**2
    u = np.divide(dot(np.cross(ap, ac), normal), area2, out=np.zeros(len(area2)), where=valid)
    v = np.divide(dot(np.cross(ab, ap), normal), area2, out=np.zeros(len(area2)), where=valid)
    in_face = valid & (u >= -relative_epsilon) & (v >= -relative_epsilon) & (u+v <= 1+relative_epsilon)
    plane2 = np.divide(dot(ap, normal)**2, area2, out=np.full(len(area2), np.inf), where=in_face)
    np.minimum(result, plane2, out=result)
    # safe_scale=1 handles collapsed point triangles without forcing zero.
    return np.maximum(result, 0.)*safe_scale**2


def exhaustive_mesh_distance(triangles, points):
    """Exhaustive query for validation of broad-phase candidate completeness."""
    triangles = np.asarray(triangles, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if not len(triangles):
        raise ValueError("empty mesh has no nearest surface")
    return np.array([np.sqrt(point_triangle_squared_distances(triangles, p).min()) for p in points])


class MeshDistance:
    """Continuous triangle-surface distance; unsigned, float64, metres."""
    backend = "rtree_candidates_normalized_plane_edge_projection_strict_min_float64"

    def __init__(self, mesh, chunk_size=512):
        if len(mesh.faces) == 0:
            raise ValueError("empty mesh has no nearest surface")
        self.mesh = mesh
        self.triangles = np.asarray(mesh.triangles, dtype=np.float64)
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise ValueError("finite query points must have shape (N,3)")
        result = np.empty(len(points), dtype=np.float64)
        for first in range(0, len(points), self.chunk_size):
            batch = points[first:first+self.chunk_size]
            candidates = trimesh.proximity.nearby_faces(self.mesh, batch)
            counts = np.array([len(ids) for ids in candidates], dtype=np.int64)
            if np.any(counts == 0):
                raise ValueError("broad phase returned no triangles")
            ids = np.concatenate(candidates)
            repeated = np.repeat(batch, counts, axis=0)
            squared = point_triangle_squared_distances(self.triangles[ids], repeated)
            starts = np.r_[0, np.cumsum(counts)[:-1]]
            result[first:first+len(batch)] = np.sqrt(np.minimum.reduceat(squared, starts))
        return result

    def validate(self, points):
        """Check broad phase and an independent scaled Ericson projector.

        Unscaled trimesh naive queries are unsuitable as the sole reference:
        their absolute triangle-region tolerance can bias millimetre meshes.
        The independent implementation runs in millimetres, while this class's
        kernel uses per-triangle dimensionless normalization.
        """
        sample = np.asarray(points, dtype=np.float64)[:12]
        if not len(sample):
            raise ValueError("validation requires at least one point")
        accelerated = self(sample)
        exhaustive = exhaustive_mesh_distance(self.triangles, sample)
        scaled = trimesh.Trimesh(vertices=np.asarray(self.mesh.vertices)*1000.,
                                 faces=np.asarray(self.mesh.faces), process=False)
        independent = np.array([trimesh.proximity.closest_point_naive(scaled, p[None]*1000.)[1][0]/1000.
                                for p in sample])
        broad_error = float(np.max(np.abs(accelerated-exhaustive)))
        independent_error = float(np.max(np.abs(accelerated-independent)))
        tolerance = 2e-7
        if broad_error > tolerance or independent_error > tolerance:
            raise ValueError(f"continuous mesh distance oracle mismatch: broad_phase={broad_error}, independent_scaled={independent_error}")
        return {"queries": len(sample), "reference": "exhaustive_normalized_plane_edge_and_independent_mm_scaled_trimesh_naive",
                "max_difference_m": max(broad_error, independent_error),
                "broad_phase_max_difference_m": broad_error,
                "independent_projector_max_difference_m": independent_error,
                "independent_projector_coordinate_scale": 1000.,
                "tolerance_m": tolerance, "passed": True}
