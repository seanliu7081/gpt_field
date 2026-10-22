"""Continuous SDF of the exposed union of actual LIBERO collision boxes.

Manifold constructs the union boundary in double precision. Unsigned distances
come from a native triangle BVH; signs come from exact box-membership predicates.
Neither min-of-overlapping-SDF nor winding-weighted distance is used as truth.
All coordinates and returned distances use metres.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import time
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_deps"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "_deps"))

import manifold3d
import numpy as np
import point_cloud_utils as pcu
import trimesh
from numba import njit


@njit(cache=False)
def _inside_boxes(points, centers, half, rotation, lower, upper):
    occupied = np.zeros(len(points), dtype=np.bool_)
    for i in range(len(points)):
        x, y, z = points[i, 0], points[i, 1], points[i, 2]
        for j in range(len(centers)):
            if x < lower[j, 0] or x > upper[j, 0] or y < lower[j, 1] or y > upper[j, 1] or z < lower[j, 2] or z > upper[j, 2]:
                continue
            dx, dy, dz = x-centers[j, 0], y-centers[j, 1], z-centers[j, 2]
            inside = True
            for axis in range(3):
                local = dx*rotation[j, 0, axis]+dy*rotation[j, 1, axis]+dz*rotation[j, 2, axis]
                if abs(local) > half[j, axis]:
                    inside = False
                    break
            if inside:
                occupied[i] = True
                break
    return occupied


@njit(cache=False)
def _nearest_primitive_boundary(points, centers, half, rotation):
    indices = np.empty(len(points), dtype=np.int64)
    distances = np.empty(len(points), dtype=np.float64)
    for i in range(len(points)):
        best, selected = np.inf, -1
        for j in range(len(centers)):
            dx, dy, dz = points[i, 0]-centers[j, 0], points[i, 1]-centers[j, 1], points[i, 2]-centers[j, 2]
            outside2, maximum = 0., -np.inf
            for axis in range(3):
                local = dx*rotation[j, 0, axis]+dy*rotation[j, 1, axis]+dz*rotation[j, 2, axis]
                q = abs(local)-half[j, axis]
                maximum = max(maximum, q)
                outside2 += max(q, 0.)**2
            distance = abs(np.sqrt(outside2)+min(maximum, 0.))
            if distance < best:
                best, selected = distance, j
        indices[i], distances[i] = selected, best
    return indices, distances


def _points(points):
    result = np.ascontiguousarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not np.all(np.isfinite(result)):
        raise ValueError("points must be a finite (N,3) array")
    return result


def unsigned_mesh_distance(points, mesh):
    """Exact nearest triangle distances, NOT winding-weighted signed values."""
    points = _points(points)
    if len(points) == 0:
        return np.empty(0, dtype=np.float64)
    count = len(points)
    # PCU 0.34.0's singleton path can return an incorrect distance, not merely
    # a squeezed scalar. Repeating the query uses its verified batched path.
    query = np.repeat(points, 2, axis=0) if count == 1 else points
    return np.asarray(pcu.closest_points_on_mesh(query, np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                                                 np.ascontiguousarray(mesh.faces, dtype=np.int64))[0]).reshape(-1)[:count]


class UnionScene:
    def __init__(self, name, boxes, planes=None):
        if not boxes:
            raise ValueError("finite box geometry is required")
        self.name = str(name)
        self.category = "libero_collision_union"
        self.seed = 0
        self.boxes = [dict(box) for box in boxes]
        self.centers = np.array([b["center_m"] for b in boxes], dtype=np.float64)
        self.half = np.array([b["half_extents_m"] for b in boxes], dtype=np.float64)
        self.rotation = np.array([b["rotation_local_to_world"] for b in boxes], dtype=np.float64)
        if self.centers.shape != (len(boxes), 3) or self.half.shape != self.centers.shape or self.rotation.shape != (len(boxes), 3, 3):
            raise ValueError("invalid collision box array shapes")
        if not all(np.all(np.isfinite(a)) for a in (self.centers, self.half, self.rotation)) or np.any(self.half <= 0):
            raise ValueError("box coordinates must be finite and half extents positive")
        for r in self.rotation:
            if not np.allclose(r.T@r, np.eye(3), atol=2e-12, rtol=0) or not np.isclose(np.linalg.det(r), 1., atol=2e-12):
                raise ValueError("box orientation must be a proper orthogonal matrix")
        extent = np.einsum("nij,nj->ni", np.abs(self.rotation), self.half)
        self.aabb_lower, self.aabb_upper = self.centers-extent, self.centers+extent
        self.planes = []
        if len(planes or []) > 1:
            raise ValueError("only one disjoint floor halfspace is supported")
        signs = np.array(list(product((-1., 1.), repeat=3)))
        for p in planes or []:
            center = np.asarray(p["center_m"], dtype=np.float64)
            normal = np.asarray(p["rotation_local_to_world"], dtype=np.float64)[:, 2]
            if not np.isclose(np.linalg.norm(normal), 1., atol=2e-12):
                raise ValueError("floor plane requires a unit normal")
            corners = np.concatenate([(signs*h)@r.T+c for c, h, r in zip(self.centers, self.half, self.rotation)])
            clearance = float(np.min((corners-center)@normal))
            if clearance <= 1e-9:
                raise ValueError("finite solids overlap/touch floor: disjoint-union plane formula would be invalid")
            self.planes.append({"center_m": center.tolist(), "normal_world": normal.tolist(),
                                "finite_geometry_minimum_plane_clearance_m": clearance,
                                "metadata": dict(p)})
        started = time.perf_counter()
        solids = [manifold3d.Manifold.cube(tuple(2*h), center=True).transform(np.column_stack((r, c)))
                  for c, h, r in zip(self.centers, self.half, self.rotation)]
        solid = manifold3d.Manifold.batch_boolean(solids, manifold3d.OpType.Add)
        status = str(solid.status())
        if status != "Error.NoError":
            raise ValueError(f"boolean union failed: {status}")
        raw = solid.to_mesh64()
        vertices = np.asarray(raw.vert_properties, dtype=np.float64)[:, :3].copy()
        faces = np.asarray(raw.tri_verts, dtype=np.int64).copy()
        self._mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if not self._mesh.is_watertight or not self._mesh.is_winding_consistent or self._mesh.volume <= 0:
            raise ValueError("boolean boundary must be consistently oriented, watertight and positive-volume")
        self.vertices = np.ascontiguousarray(vertices)
        self.faces = np.ascontiguousarray(faces)
        self.boolean_seconds = time.perf_counter()-started
        self.boolean_tolerance_m = float(solid.get_tolerance())
        self.reference_boundary_tolerance_m = max(1e-9, 100*self.boolean_tolerance_m)
        self.minimum_feature_size_m = float(2*np.min(self.half))
        self.object_names = np.array([b.get("object", b.get("name", f"box_{i}")) for i, b in enumerate(boxes)], dtype=object)
        # Compile occupancy once before timed grid generation; caching is disabled.
        _inside_boxes(np.zeros((1, 3)), self.centers, self.half, self.rotation, self.aabb_lower, self.aabb_upper)

    @classmethod
    def from_snapshot(cls, snapshot):
        return cls(snapshot["name"], snapshot["boxes"], snapshot.get("planes", []))

    def contains_finite(self, points):
        return _inside_boxes(_points(points), self.centers, self.half, self.rotation, self.aabb_lower, self.aabb_upper)

    def contains(self, points):
        points = _points(points)
        result = self.contains_finite(points)
        for p in self.planes:
            result |= (points-np.asarray(p["center_m"]))@np.asarray(p["normal_world"]) <= 0
        return result

    def sdf(self, points):
        points = _points(points)
        if not len(points):
            return np.empty(0, dtype=np.float64)
        query = np.repeat(points, 2, axis=0) if len(points) == 1 else points
        distance = np.asarray(pcu.closest_points_on_mesh(query, self.vertices, self.faces)[0]).reshape(-1)[:len(points)]
        distance[self.contains_finite(points)] *= -1
        for p in self.planes:
            np.minimum(distance, (points-np.asarray(p["center_m"]))@np.asarray(p["normal_world"]), out=distance)
        return distance

    def mesh(self):
        """Finite exposed union boundary; infinite floor is separately analytic."""
        return self._mesh.copy()

    def _sample_surface_with_faces(self, n, rng):
        triangles = self._mesh.triangles
        ids = rng.choice(len(triangles), int(n), p=self._mesh.area_faces/self._mesh.area)
        u, v = rng.random((2, int(n)))
        s = np.sqrt(u)
        points = ((1-s)[:, None]*triangles[ids, 0]+(s*(1-v))[:, None]*triangles[ids, 1]
                  +(s*v)[:, None]*triangles[ids, 2])
        return points, ids

    def sample_surface(self, n, rng):
        """Area-uniform finite exposed surface; plane relevance must be gated."""
        return self._sample_surface_with_faces(n, rng)[0]

    def surface_object_labels(self, points):
        """Nearest source-box face label on exposed surface, deterministic ties."""
        ids, _ = _nearest_primitive_boundary(_points(points), self.centers, self.half, self.rotation)
        return self.object_names[ids]

    def prove_planes_irrelevant(self, lower, upper):
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
            raise ValueError("ROI requires ordered XYZ bounds")
        corners = np.array(list(product(*zip(lower, upper))))
        # For any fixed finite boundary vertex, its furthest ROI point is a
        # corner. The closest such worst-case vertex supplies a uniform bound.
        finite_upper = min(float(np.linalg.norm(corners-v, axis=1).max()) for v in self.vertices)
        reports = []
        for p in self.planes:
            plane_lower = float(np.min((corners-np.asarray(p["center_m"]))@np.asarray(p["normal_world"])))
            reports.append({"plane_distance_lower_m": plane_lower,
                            "finite_surface_distance_upper_m": finite_upper,
                            "passed": plane_lower > finite_upper+self.reference_boundary_tolerance_m})
        return {"passed": all(p["passed"] for p in reports), "planes": reports,
                "method": "linear_plane_min_over_roi_corners_exceeds_finite_boundary_vertex_distance_upper_bound"}

    def validate(self, sample_count=3000, seed=441):
        """Cross-check union sign, exposed faces and native distance independently."""
        from mesh_queries import MeshDistance
        rng = np.random.default_rng(seed)
        uniform = rng.uniform(self._mesh.bounds[0]-.02, self._mesh.bounds[1]+.02, (int(sample_count), 3))
        surface, face_ids = self._sample_surface_with_faces(int(sample_count), rng)
        near = surface+rng.normal(scale=.001, size=surface.shape)
        points = np.concatenate((uniform, near))
        unsigned = pcu.closest_points_on_mesh(points, self.vertices, self.faces)[0]
        # Winding is an independent INSIDE classifier only. Its weighted
        # distance magnitude is demonstrably not Euclidean and is not reused.
        winding_values = pcu.signed_distance_to_mesh(points, self.vertices, self.faces)[0]
        analytic_inside = self.contains_finite(points)
        away = unsigned > self.reference_boundary_tolerance_m
        sign_mismatches = int(np.count_nonzero((winding_values[away] < 0) != analytic_inside[away]))
        check = np.concatenate((uniform[:32], near[:32]))
        reference = MeshDistance(self._mesh)(check)
        native = pcu.closest_points_on_mesh(check, self.vertices, self.faces)[0]
        distance_error = float(np.max(np.abs(native-reference)))
        delta = max(1e-8, 1000*self.boolean_tolerance_m)
        normal = self._mesh.face_normals[face_ids]
        inward_misses = int(np.count_nonzero(~self.contains_finite(surface-delta*normal)))
        outward_hits = int(np.count_nonzero(self.contains_finite(surface+delta*normal)))
        _, surface_distance = _nearest_primitive_boundary(surface, self.centers, self.half, self.rotation)
        boundary_error = float(surface_distance.max())
        passed = (sign_mismatches == 0 and distance_error <= self.reference_boundary_tolerance_m
                  and inward_misses == 0 and outward_hits == 0 and boundary_error <= self.reference_boundary_tolerance_m)
        return {"passed": passed, "source": "actual_MuJoCo_box_union_not_rendered_visual_mesh",
                "watertight": bool(self._mesh.is_watertight), "winding_consistent": bool(self._mesh.is_winding_consistent),
                "sample_count": len(points), "occupancy_boundary_excluded": int((~away).sum()),
                "winding_vs_analytic_union_sign_mismatches": sign_mismatches,
                "native_unsigned_vs_independent_normalized_projection_queries": len(check),
                "maximum_unsigned_distance_disagreement_m": distance_error,
                "exposed_face_inward_misses": inward_misses, "exposed_face_outward_inside_hits": outward_hits,
                "normal_probe_offset_m": delta, "maximum_exposed_surface_to_primitive_boundary_m": boundary_error,
                "boolean_tolerance_m": self.boolean_tolerance_m,
                "validation_tolerance_m": self.reference_boundary_tolerance_m,
                "winding_weighted_magnitude_max_difference_m": float(np.max(np.abs(np.abs(winding_values)-unsigned))),
                "winding_magnitudes_used_for_sdf": False,
                "limitations": "finite sampled checks; boolean boundary is double-precision numerical geometry"}

    def to_dict(self):
        return {"name": self.name, "category": self.category, "units": "m", "negative": "inside_material",
                "box_count": len(self.boxes), "boxes": self.boxes, "planes": self.planes,
                "mesh_vertices": len(self.vertices), "mesh_triangles": len(self.faces),
                "mesh_bounds_m": self._mesh.bounds.tolist(), "volume_m3": float(self._mesh.volume),
                "boolean_backend": "manifold3d_batch_union_of_double_precision_transformed_cubes_to_Mesh64",
                "boolean_tolerance_m": self.boolean_tolerance_m, "boolean_seconds": self.boolean_seconds,
                "distance_backend": "point_cloud_utils_unsigned_triangle_BVH_plus_analytic_OBB_union_sign",
                "native_singleton_query_workaround": "repeat_single_point_then_select_first_verified_batched_result",
                "signed_distance_is_min_of_overlapping_primitive_sdfs": False,
                "surface_sampling": "area_uniform_on_finite_exposed_union_only; floor_irrelevance_requires_ROI_certificate",
                "object_labeling": "nearest_source_box_surface_on_exposed_boundary_deterministic_ties",
                "minimum_box_thickness_m": self.minimum_feature_size_m,
                "reference_boundary_tolerance_m": self.reference_boundary_tolerance_m}
