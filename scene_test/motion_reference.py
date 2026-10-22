"""Independent rigid-body collision and interval certificates, in metres.

This module never queries a distance-field grid or Scene.sdf. Sphere/OBB
distances and normalized separating axes provide the geometric reference.
Unknown numerical or temporal cases remain unknown, never collision-free.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

TOL = 1e-10


@dataclass
class Part:
    center: np.ndarray
    half: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.half = np.asarray(self.half, float)
        self.rotation = np.asarray(self.rotation, float)
        if (self.center.shape != (3,) or self.half.shape != (3,) or np.any(self.half <= 0)
                or not np.all(np.isfinite(self.center)) or not np.all(np.isfinite(self.half))):
            raise ValueError("part center and positive half extents must be XYZ")
        if (self.rotation.shape != (3, 3) or not np.all(np.isfinite(self.rotation))
                or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-12, rtol=0)
                or not np.isclose(np.linalg.det(self.rotation), 1., atol=1e-12, rtol=0)):
            raise ValueError("part orientation must be orthogonal")

    def vertices(self):
        return np.array(list(product((-1., 1.), repeat=3))) * self.half @ self.rotation.T + self.center

    def to_dict(self):
        return {"center_m": self.center.tolist(), "half_extents_m": self.half.tolist(),
                "rotation_local_to_body": self.rotation.tolist()}


@dataclass
class Body:
    name: str
    sphere_radius: float | None = None
    parts: list[Part] = field(default_factory=list)

    def __post_init__(self):
        if (self.sphere_radius is None) == (not self.parts):
            raise ValueError("body must be either a sphere or a nonempty union of boxes")
        if self.sphere_radius is not None and (not np.isfinite(self.sphere_radius) or self.sphere_radius <= 0):
            raise ValueError("sphere radius must be positive")

    @property
    def radius(self):
        if self.sphere_radius is not None:
            return self.sphere_radius
        return float(max(np.linalg.norm(p.vertices(), axis=1).max() for p in self.parts))

    def bounds(self, translation, rotation):
        if self.sphere_radius is not None:
            return translation - self.sphere_radius, translation + self.sphere_radius
        vertices = np.concatenate([p.vertices() for p in self.parts]) @ rotation.T + translation
        return vertices.min(axis=0), vertices.max(axis=0)

    def world_parts(self, translation, rotation):
        return [Part(translation + rotation @ p.center, p.half, rotation @ p.rotation) for p in self.parts]

    def sample_lattice(self, maximum_spacing=.002):
        if maximum_spacing <= 0:
            raise ValueError("lattice spacing must be positive")
        if self.sphere_radius is not None:
            return np.zeros((1, 3)), np.zeros((1, 3)), self.sphere_radius
        volume, surface, radii = [], [], []
        for part in self.parts:
            intervals = np.maximum(1, np.ceil(2 * part.half / maximum_spacing).astype(int))
            axes = [np.linspace(-h, h, n + 1) for h, n in zip(part.half, intervals)]
            local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            boundary = np.any(np.isclose(np.abs(local), part.half, atol=1e-14, rtol=0), axis=1)
            world = local @ part.rotation.T + part.center
            volume.append(world)
            surface.append(world[boundary])
            radii.append(.5 * np.linalg.norm(2 * part.half / intervals))
        return np.concatenate(volume), np.concatenate(surface), float(max(radii))

    def to_dict(self):
        return {"name": self.name, "sphere_radius_m": self.sphere_radius,
                "parts": [p.to_dict() for p in self.parts], "bounding_radius_about_motion_origin_m": self.radius}


class Trajectory:
    """Linear origin translation and shortest-quaternion-arc orientation."""
    def __init__(self, start, end, q0=None, q1=None):
        self.start, self.end = np.asarray(start, float), np.asarray(end, float)
        if (self.start.shape != (3,) or self.end.shape != (3,)
                or not np.all(np.isfinite(self.start)) or not np.all(np.isfinite(self.end))):
            raise ValueError("trajectory endpoints must be finite XYZ vectors")
        q0 = [0, 0, 0, 1] if q0 is None else q0
        q1 = q0 if q1 is None else q1
        rotations = Rotation.from_quat([q0, q1])
        self.slerp = Slerp([0., 1.], rotations)
        self.angle = float((rotations[0].inv() * rotations[1]).magnitude())
        self.q0, self.q1 = rotations.as_quat()

    def pose(self, t):
        return self.start + float(t) * (self.end - self.start), self.slerp(float(t)).as_matrix()

    def displacement_bound(self, a, b, body):
        translation = np.linalg.norm(self.end - self.start) * (b - a) / 2
        rotation = 0. if body.sphere_radius is not None else 2 * body.radius * np.sin(self.angle * (b - a) / 4)
        return float(translation + rotation)

    def to_dict(self):
        return {"start_translation_m": self.start.tolist(), "end_translation_m": self.end.tolist(),
                "start_quaternion_xyzw": self.q0.tolist(), "end_quaternion_xyzw": self.q1.tolist(),
                "shortest_rotation_radians": self.angle,
                "interpolation": "linear_translation_and_shortest_arc_SLERP"}


def sphere_box(center, radius, box):
    local = box.rotation.T @ (np.asarray(center) - box.center)
    excess = np.maximum(np.abs(local) - box.half, 0.)
    outside_distance = float(np.sqrt(np.dot(excess, excess)))
    if np.any(np.abs(local) > box.half):
        signed_center = outside_distance
    else:
        signed_center = -float(np.min(box.half - np.abs(local)))
    gap = signed_center - radius
    return {"status": "free" if gap > TOL else "blocked" if gap < -TOL else "unknown",
            "clearance_lower_m": max(0., gap), "signed_separation_m": gap,
            "method": "independent_sphere_oriented_box"}


def _strict_common_interior(a, b):
    """LP proposes a witness; direct face margins must independently accept it."""
    from scipy.optimize import linprog
    rows, bounds = [], []
    for box in (a, b):
        for axis, half in zip(box.rotation.T, box.half):
            for sign in (-1., 1.):
                normal = sign * axis
                rows.append(np.r_[normal, 1.])
                bounds.append(half + np.dot(normal, box.center))
    result = linprog([0., 0., 0., -1.], A_ub=rows, b_ub=bounds,
                     bounds=[(None, None)] * 3 + [(0., None)], method="highs")
    if not result.success:
        return None
    point = result.x[:3]
    margin = min(float(np.min(box.half - np.abs(box.rotation.T @ (point - box.center)))) for box in (a, b))
    return point if margin > TOL else None


def box_box(a, b):
    """15 normalized SAT axes, with explicit handling of nearly parallel axes."""
    axes = list(a.rotation.T) + list(b.rotation.T)
    near_parallel = False
    for av in a.rotation.T:
        for bv in b.rotation.T:
            cross = np.cross(av, bv)
            length = float(np.linalg.norm(cross))
            if length == 0.:
                continue
            if length < 1e-12:
                near_parallel = True
                continue
            axes.append(cross / length)
    axes = np.asarray(axes)
    delta = b.center - a.center
    gaps = (np.abs(axes @ delta) - np.abs(axes @ a.rotation) @ a.half
            - np.abs(axes @ b.rotation) @ b.half)
    best = float(gaps.max())
    witness = None
    if best > TOL:
        status = "free"
    elif best < -TOL and not near_parallel:
        status = "blocked"
    elif best < -TOL:
        witness = _strict_common_interior(a, b)
        status = "blocked" if witness is not None else "unknown"
    else:
        status = "unknown"
    return {"status": status, "clearance_lower_m": max(0., best),
            "signed_separating_axis_gap_m": best, "method": "normalized_15_axis_SAT",
            "near_parallel_cross_axis": near_parallel,
            "strict_interior_witness_m": None if witness is None else witness.tolist()}


class ExactChecker:
    """Independent closed-world geometry oracle; not restricted by grid ROI."""
    def __init__(self, body, obstacles):
        self.body, self.obstacles = body, obstacles
        self.scalar_queries = 0

    def pose(self, translation, rotation):
        if not self.obstacles:
            return {"status": "free", "clearance_lower_m": float("inf"), "roi_margin_m": None,
                    "witness": None, "reason": "declared_empty_scene"}
        parts = [] if self.body.sphere_radius is not None else self.body.world_parts(translation, rotation)
        results = []
        for j, obstacle in enumerate(self.obstacles):
            if self.body.sphere_radius is not None:
                result = sphere_box(translation, self.body.sphere_radius, obstacle)
                result.update(body_part=0, obstacle=j)
                results.append(result)
                self.scalar_queries += 1
            else:
                for i, part in enumerate(parts):
                    result = box_box(part, obstacle)
                    result.update(body_part=i, obstacle=j)
                    results.append(result)
                    self.scalar_queries += 1
        blocked = next((r for r in results if r["status"] == "blocked"), None)
        status = "blocked" if blocked else "free" if all(r["status"] == "free" for r in results) else "unknown"
        return {"status": status, "clearance_lower_m": min(r["clearance_lower_m"] for r in results),
                "roi_margin_m": None, "witness": blocked,
                "reason": "strict_geometric_overlap" if blocked else "separation" if status == "free" else "numerical_or_tangent"}


def continuous_check(checker, trajectory, body, *, max_depth=12, node_budget=257, motion_tolerance=1e-5):
    """Certify the whole path or preserve unresolved intervals.

    A colliding sampled pose is a valid path witness. Sampled free poses alone
    never certify a path. The clearance and ROI margins at an interval midpoint
    must exceed the rigid-body displacement bound over the entire interval.
    """
    if max_depth < 0 or node_budget < 1 or motion_tolerance < 0:
        raise ValueError("invalid adaptive checking limits")
    cache, reasons, certified = {}, set(), []
    before = checker.scalar_queries
    deepest = 0

    def evaluate(t):
        key = float(t)
        if key not in cache:
            cache[key] = checker.pose(*trajectory.pose(key))
        return cache[key]

    def outcome(status, witness_time=None):
        return {"status": status, "witness_time": witness_time,
                "witness": cache[witness_time] if witness_time is not None else None,
                "pose_queries": len(cache), "primitive_or_sdf_queries": checker.scalar_queries - before,
                "max_depth_visited": deepest, "certified_interval_count": len(certified),
                "uncertain_reasons": sorted(reasons), "node_budget": node_budget,
                "max_depth_limit": max_depth, "motion_tolerance_m": motion_tolerance}

    for t in (0., 1.):
        result = evaluate(t)
        if result["status"] == "blocked":
            return outcome("blocked", t)
    stack = [(0., 1., 0)]
    while stack:
        a, b, depth = stack.pop()
        if len(cache) >= node_budget + 2:
            reasons.add("pose_query_budget")
            break
        mid = (a + b) / 2
        result = evaluate(mid)
        deepest = max(deepest, depth)
        if result["status"] == "blocked":
            return outcome("blocked", mid)
        bound = trajectory.displacement_bound(a, b, body)
        roi = result.get("roi_margin_m")
        inside_roi = roi is None or roi > bound + TOL
        if result["status"] == "free" and result["clearance_lower_m"] > bound + TOL and inside_roi:
            certified.append((a, b))
            continue
        if depth >= max_depth or bound <= motion_tolerance:
            reasons.add(result.get("reason", "insufficient_clearance") if result["status"] != "free" else "interval_clearance_or_roi")
            reasons.add("depth_limit" if depth >= max_depth else "motion_tolerance_limit")
            continue
        stack.append((mid, b, depth + 1))
        stack.append((a, mid, depth + 1))
    return outcome("unknown" if reasons else "free")
