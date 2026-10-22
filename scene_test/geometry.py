"""Exact, deterministic reference geometry for the isolated scene experiment.

All lengths are metres. SDFs are negative in material and positive in air.
Container distances use exposed boundary rectangles, not a CSG pseudo-SDF.
Disjoint unions use minima of exact primitive SDFs; their solids never overlap.
The sphere mesh is a visualization approximation; its analytic SDF is reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from itertools import product
from typing import Callable

import numpy as np
import trimesh
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation


def _points(points):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return p


@dataclass
class Sphere:
    center: np.ndarray
    radius: float

    def sdf(self, points):
        return np.linalg.norm(_points(points) - self.center, axis=1) - self.radius

    @property
    def area(self):
        return 4 * np.pi * self.radius**2

    def sample_surface(self, n, rng):
        directions = rng.normal(size=(n, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        return self.center + self.radius * directions

    def mesh(self):
        mesh = trimesh.creation.icosphere(subdivisions=5, radius=self.radius)
        mesh.apply_translation(self.center)
        return mesh

    def to_dict(self):
        return {"type": "sphere", "center_m": self.center.tolist(), "radius_m": self.radius}


@dataclass
class Box:
    center: np.ndarray
    half_extents: np.ndarray
    rotation: np.ndarray

    def world(self, local):
        return np.asarray(local) @ self.rotation.T + self.center

    def sdf(self, points):
        q = np.abs((_points(points) - self.center) @ self.rotation) - self.half_extents
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(np.max(q, axis=1), 0)

    @property
    def area(self):
        h = self.half_extents
        return float(8 * (h[0]*h[1] + h[1]*h[2] + h[0]*h[2]))

    def sample_surface(self, n, rng):
        h = self.half_extents
        weights = np.array([h[1]*h[2], h[0]*h[2], h[0]*h[1]])
        axes = rng.choice(3, size=n, p=weights/weights.sum())
        local = rng.uniform(-1, 1, (n, 3)) * h
        local[np.arange(n), axes] = rng.choice([-1, 1], size=n) * h[axes]
        return self.world(local)

    def mesh(self):
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.center
        return trimesh.creation.box(extents=2*self.half_extents, transform=transform)

    def to_dict(self):
        return {"type": "oriented_box", "center_m": self.center.tolist(),
                "half_extents_m": self.half_extents.tolist(), "rotation_local_to_world": self.rotation.tolist()}


class OpenContainer:
    """Open-top rectangular vessel with exact Euclidean signed distance.

    Four vertical walls and a bottom form one solid. The rim, outer sides,
    cavity walls, cavity floor and underside are the only exposed surfaces.
    """
    def __init__(self, center, half_extents, wall, bottom, rotation):
        self.center = np.asarray(center, dtype=float)
        self.half_extents = np.asarray(half_extents, dtype=float)
        self.wall = float(wall)
        self.bottom = float(bottom)
        self.rotation = np.asarray(rotation, dtype=float)
        x, y, z = self.half_extents
        a, b, f = x-self.wall, y-self.wall, -z+self.bottom
        if min(a, b, 2*z-self.bottom) <= 0:
            raise ValueError("container wall or bottom too thick")
        # (normal axis, coordinate, axis u, interval u, axis v, interval v)
        self.patches = []
        for s in (-1, 1):
            self.patches.extend([(0, s*x, 1, -y, y, 2, -z, z),
                                 (1, s*y, 0, -x, x, 2, -z, z),
                                 (0, s*a, 1, -b, b, 2, f, z),
                                 (1, s*b, 0, -a, a, 2, f, z)])
        self.patches.extend([(2, -z, 0, -x, x, 1, -y, y),
                             (2, f, 0, -a, a, 1, -b, b),
                             (2, z, 0, -x, x, 1, -y, -b),
                             (2, z, 0, -x, x, 1, b, y),
                             (2, z, 0, -x, -a, 1, -b, b),
                             (2, z, 0, a, x, 1, -b, b)])

    def world(self, local):
        return np.asarray(local) @ self.rotation.T + self.center

    def inside_local(self, p):
        x, y, z = self.half_extents
        outer = (np.abs(p) <= self.half_extents).all(axis=1)
        cavity = ((np.abs(p[:, 0]) < x-self.wall) &
                  (np.abs(p[:, 1]) < y-self.wall) &
                  (p[:, 2] > -z+self.bottom))
        return outer & ~cavity

    def sdf(self, points):
        p = (_points(points)-self.center) @ self.rotation
        best = np.full(len(p), np.inf)
        for axis, level, u, lo_u, hi_u, v, lo_v, hi_v in self.patches:
            du = np.maximum(np.maximum(lo_u-p[:, u], p[:, u]-hi_u), 0)
            dv = np.maximum(np.maximum(lo_v-p[:, v], p[:, v]-hi_v), 0)
            squared = (p[:, axis]-level)**2 + du**2 + dv**2
            np.minimum(best, squared, out=best)
        result = np.sqrt(best)
        result[self.inside_local(p)] *= -1
        return result

    @property
    def patch_areas(self):
        return np.asarray([(p[4]-p[3])*(p[7]-p[6]) for p in self.patches])

    @property
    def area(self):
        return float(self.patch_areas.sum())

    def sample_surface(self, n, rng):
        ids = rng.choice(len(self.patches), n, p=self.patch_areas/self.area)
        local = np.empty((n, 3))
        for i, (axis, level, u, lo_u, hi_u, v, lo_v, hi_v) in enumerate(self.patches):
            sel = ids == i
            local[sel, axis] = level
            local[sel, u] = rng.uniform(lo_u, hi_u, int(sel.sum()))
            local[sel, v] = rng.uniform(lo_v, hi_v, int(sel.sum()))
        return self.world(local)

    def mesh(self):
        # Split the solid into elementary cells ONLY to triangulate its exposed
        # faces conformingly: no coincident internal faces or T junctions remain.
        x, y, z = self.half_extents
        knots = [np.array([-x, -x+self.wall, x-self.wall, x]),
                 np.array([-y, -y+self.wall, y-self.wall, y]),
                 np.array([-z, -z+self.bottom, z])]
        cells = set(product(range(3), range(3), range(2)))
        cells.remove((1, 1, 1))
        verts, faces = [], []
        for cell in sorted(cells):
            for axis in range(3):
                u, v = (axis+1) % 3, (axis+2) % 3
                for side in (-1, 1):
                    neighbor = list(cell)
                    neighbor[axis] += side
                    if tuple(neighbor) in cells:
                        continue
                    coord = knots[axis][cell[axis]+(side == 1)]
                    quad = []
                    for a, b in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                        p = np.zeros(3)
                        p[axis] = coord
                        p[u] = knots[u][cell[u]+a]
                        p[v] = knots[v][cell[v]+b]
                        quad.append(p)
                    if side == -1:
                        quad.reverse()
                    k = len(verts)
                    verts.extend(quad)
                    faces.extend([[k, k+1, k+2], [k, k+2, k+3]])
        return trimesh.Trimesh(vertices=self.world(verts), faces=faces, process=True)

    def to_dict(self):
        return {"type": "open_rectangular_container", "center_m": self.center.tolist(),
                "half_extents_m": self.half_extents.tolist(), "wall_m": self.wall,
                "bottom_m": self.bottom, "rotation_local_to_world": self.rotation.tolist(),
                "exact_distance_method": "minimum_distance_to_14_exposed_rectangles_with_material_sign"}


@dataclass
class Scene:
    name: str
    category: str
    seed: int
    objects: list
    probes: list
    cross_sections: list = field(default_factory=list)

    @property
    def minimum_feature_size_m(self):
        sizes = []
        for obj in self.objects:
            if isinstance(obj, Sphere):
                sizes.append(2*obj.radius)
            elif isinstance(obj, Box):
                sizes.append(float(2*obj.half_extents.min()))
            elif isinstance(obj, OpenContainer):
                sizes.extend([obj.wall, obj.bottom])
        if self.category == "thin_walls_and_narrow_gaps":
            sizes.append(float(np.linalg.norm(self.objects[0].center-self.objects[1].center)
                               - self.objects[0].half_extents[0]-self.objects[1].half_extents[0]))
        return min(sizes)

    def sdf(self, points):
        p = _points(points)
        result = np.full(len(p), np.inf)
        for obj in self.objects:
            np.minimum(result, obj.sdf(p), out=result)
        return result

    def sample_surface(self, n, rng):
        areas = np.array([obj.area for obj in self.objects])
        counts = rng.multinomial(n, areas/areas.sum())
        samples = np.concatenate([obj.sample_surface(int(count), rng)
                                  for obj, count in zip(self.objects, counts)], axis=0)
        rng.shuffle(samples)
        return samples

    def mesh(self):
        return trimesh.util.concatenate([obj.mesh() for obj in self.objects])

    def feature_checks(self, query_callable: Callable):
        """Fixed interior line probes; not a proof of global topology preservation."""
        details = []
        for probe in self.probes:
            pts = np.linspace(probe["start_m"], probe["end_m"], probe["sample_count"])
            ref = self.sdf(pts)
            predicted = np.asarray(query_callable(pts), dtype=float)
            if predicted.shape != ref.shape:
                raise ValueError("feature query must return shape (N,)")
            occupied = probe["expected_material"]
            good = predicted < 0 if occupied else predicted > 0
            reference_good = ref < 0 if occupied else ref > 0
            if not np.all(reference_good):
                raise AssertionError("declared feature probe is not in intended material/air")
            details.append({**probe, "passed": bool(np.all(good)),
                            "correct_fraction": float(good.mean()),
                            "reference_min_abs_distance_m": float(np.abs(ref).min()),
                            "predicted_min_m": float(predicted.min()),
                            "predicted_max_m": float(predicted.max()),
                            "max_abs_distance_error_m": float(np.abs(predicted-ref).max())})
        widths = []
        for section in self.cross_sections:
            origin, direction = np.asarray(section["origin_m"]), np.asarray(section["direction"])
            parameters = np.linspace(section["min_m"], section["max_m"], 4097)
            values = np.asarray(query_callable(origin + parameters[:, None]*direction))
            roots = list(parameters[values == 0])
            for i in np.flatnonzero(values[:-1]*values[1:] < 0):
                root = brentq(lambda t: float(query_callable((origin+t*direction)[None])[0]),
                              parameters[i], parameters[i+1], xtol=1e-12)
                roots.append(root)
            roots = np.unique(np.round(roots, 12))
            expected = np.asarray(section["expected_zero_crossings_m"])
            count_matches = len(roots) == len(expected)
            widths.append({**section, "search_samples": len(parameters),
                           "predicted_zero_crossings_m": roots.tolist(),
                           "crossing_count_matches": bool(count_matches),
                           "predicted_inter_crossing_widths_m": np.diff(roots).tolist(),
                           "expected_inter_crossing_widths_m": np.diff(expected).tolist(),
                           "width_errors_m": (np.diff(roots)-np.diff(expected)).tolist() if count_matches else None,
                           "max_surface_crossing_error_m": float(np.max(np.abs(roots-expected))) if count_matches else None})
        return {"method": "predeclared_line_sign_and_zero_crossing_probes_not_global_topology_certificate",
                "probe_count": len(details), "all_passed": all(p["passed"] for p in details),
                "probes": details, "cross_sections": widths}

    def to_dict(self):
        return {"name": self.name, "category": self.category, "seed": self.seed,
                "units": "m", "sign": "negative_inside_material", "objects": [obj.to_dict() for obj in self.objects],
                "union": "minimum_of_exact_SDFs_for_nonoverlapping_solids",
                "feature_probes": self.probes,
                "cross_sections": self.cross_sections,
                "minimum_feature_size_m": self.minimum_feature_size_m,
                "sphere_mesh_note": "icosphere visualization only; distance and sampling use exact analytic sphere"}


def _probe(name, start, end, material, count=101):
    return {"name": name, "start_m": np.asarray(start).tolist(), "end_m": np.asarray(end).tolist(),
            "expected_material": bool(material), "sample_count": int(count)}


def _section(name, origin, direction, extent, expected):
    return {"name": name, "origin_m": np.asarray(origin).tolist(),
            "direction": np.asarray(direction).tolist(), "min_m": -extent, "max_m": extent,
            "expected_zero_crossings_m": list(expected),
            "inter_crossing_interpretation": "alternating_material_air_material_for_four_crossings"}


def make_scenes():
    """Return the frozen 5-category x 3-seed geometry matrix (15 scenes)."""
    scenes = []
    for index, seed in enumerate([11, 23, 37]):
        rng = np.random.default_rng(seed)
        shift = rng.uniform(-.007, .007, 3)
        rotation = Rotation.from_euler("xyz", [.17+.03*index, -.13-.04*index, .23+.11*index]).as_matrix()

        sphere = Sphere(shift.copy(), [.0387, .0433, .0479][index])
        scenes.append(Scene(f"sphere_seed{seed}", "sphere", seed, [sphere],
                            [_probe("solid_diameter_interior", sphere.center+[-.02, 0, 0],
                                    sphere.center+[.02, 0, 0], True)]))

        box = Box(shift.copy(), np.array([.0423, .0317, .0251]), rotation)
        scenes.append(Scene(f"rotated_box_seed{seed}", "rotated_box", seed, [box],
                            [_probe("box_interior", box.world([-.025, 0, 0]), box.world([.025, 0, 0]), True)]))

        container = OpenContainer(shift.copy(), [.0513, .0427, .0341], .0063, .0071, rotation)
        floor = -container.half_extents[2]+container.bottom
        probes = [_probe("open_cavity_to_exterior", container.world([0, 0, floor+.003]),
                         container.world([0, 0, container.half_extents[2]+.008]), False),
                  _probe("bottom_material", container.world([-.03, 0, floor-container.bottom/2]),
                         container.world([.03, 0, floor-container.bottom/2]), True),
                  _probe("side_wall_material", container.world([container.half_extents[0]-container.wall/2, 0, -.02]),
                         container.world([container.half_extents[0]-container.wall/2, 0, .02]), True)]
        x = container.half_extents[0]
        scenes.append(Scene(f"open_hollow_container_seed{seed}", "open_hollow_container", seed, [container], probes,
                            [_section("container_wall_and_cavity_width", shift, rotation[:, 0], .065,
                                      [-x, -x+container.wall, x-container.wall, x])]))

        thickness, gap = [(0.001, .004), (.002, .002), (.004, .001)][index]
        half = np.array([thickness/2, .0467, .0283])
        walls = [Box(shift+rotation @ np.array([s*(gap+thickness)/2, 0, 0]), half.copy(), rotation)
                 for s in (-1, 1)]
        probes = [_probe("wall_left", walls[0].world([0, -.035, 0]), walls[0].world([0, .035, 0]), True),
                  _probe("wall_right", walls[1].world([0, -.035, 0]), walls[1].world([0, .035, 0]), True),
                  _probe("narrow_gap", shift+rotation @ [0, -.035, 0], shift+rotation @ [0, .035, 0], False)]
        scene = Scene(f"thin_walls_and_narrow_gaps_seed{seed}", "thin_walls_and_narrow_gaps", seed, walls, probes)
        scene.cross_sections = [_section("two_wall_thicknesses_and_gap", shift, rotation[:, 0], .02,
                                         [-gap/2-thickness, -gap/2, gap/2, gap/2+thickness])]
        scenes.append(scene)

        ball = Sphere(shift+[-.046, -.025, 0], .0203)
        other = Box(shift+[.039, .023, .004], np.array([.0221, .0187, .0273]), rotation)
        delta = other.center-ball.center
        start, end = ball.center+.40*delta, ball.center+.57*delta
        scenes.append(Scene(f"separated_multi_object_scene_seed{seed}", "separated_multi_object_scene", seed,
                            [ball, other], [_probe("between_objects_no_merge", start, end, False)]))
    category_order = ["sphere", "rotated_box", "open_hollow_container", "thin_walls_and_narrow_gaps", "separated_multi_object_scene"]
    scenes.sort(key=lambda s: (category_order.index(s.category), s.seed))
    return scenes


def _winding_inside(mesh, points):
    """Triangle solid-angle winding number; independent of primitive predicates."""
    triangles = mesh.triangles
    result = []
    for point in points:
        a, b, c = np.moveaxis(triangles-point, 1, 0)
        la, lb, lc = [np.linalg.norm(x, axis=1) for x in (a, b, c)]
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = la*lb*lc + np.einsum("ij,ij->i", a, b)*lc + np.einsum("ij,ij->i", b, c)*la + np.einsum("ij,ij->i", c, a)*lb
        result.append(abs((2*np.arctan2(numerator, denominator)).sum()) > 2*np.pi)
    return np.asarray(result)


def validate_reference(scenes=None):
    """Independent analytic/mesh reference checks before accepting any SDF run.

    Flat geometry: exhaustive nearest-triangle distances and solid-angle signs.
    Spheres: 50-digit Decimal distances (the faceted mesh is not sphere truth).
    These are sampled numerical checks, not formal proofs for arbitrary geometry.
    """
    scenes = make_scenes() if scenes is None else scenes
    reports = []
    tolerance = 2e-12
    for scene in scenes:
        mesh = scene.mesh()
        primitive_reports = []
        rng = np.random.default_rng(scene.seed+701)
        for obj in scene.objects:
            points = np.concatenate([rng.uniform(-.10, .10, (80, 3)),
                                     obj.sample_surface(80, rng)+rng.normal(scale=.001, size=(80, 3))])
            if isinstance(obj, Sphere):
                with localcontext() as ctx:
                    ctx.prec = 50
                    reference = np.array([float(sum((Decimal(float(p[i]))-Decimal(float(obj.center[i])))**2
                                                    for i in range(3)).sqrt()-Decimal(float(obj.radius)))
                                          for p in points])
                method = "50_digit_decimal_analytic_sphere"
            else:
                primitive_mesh = obj.mesh()
                _, reference, _ = trimesh.proximity.closest_point_naive(primitive_mesh, points)
                reference[_winding_inside(primitive_mesh, points)] *= -1
                method = "exhaustive_triangle_distance_and_solid_angle_sign"
            distance = obj.sdf(points)
            error = float(np.max(np.abs(reference-distance)))
            primitive_reports.append({"type": type(obj).__name__, "method": method,
                                      "samples": len(points), "maximum_absolute_error_m": error,
                                      "passed": error < tolerance})
        separated = True
        separation_method = "single_solid"
        if len(scene.objects) > 1:
            a, b = scene.objects
            if isinstance(a, Sphere) and isinstance(b, Box):
                separated = np.linalg.norm(a.center-b.center) > a.radius + np.linalg.norm(b.half_extents)
                separation_method = "disjoint_bounding_spheres"
            elif isinstance(a, Box) and isinstance(b, Box):
                separated = bool(np.allclose(a.rotation, b.rotation) and
                                 np.any(np.abs((a.center-b.center) @ a.rotation) > a.half_extents+b.half_extents))
                separation_method = "common_frame_separating_axis"
            else:
                raise ValueError("unvalidated union combination")
        bounds_ok = bool(np.all(mesh.bounds[0] >= -.10) and np.all(mesh.bounds[1] <= .10))
        probes = scene.feature_checks(scene.sdf)
        widths_ok = all(c["crossing_count_matches"] and c["max_surface_crossing_error_m"] < tolerance
                        for c in probes["cross_sections"])
        passed = (bounds_ok and bool(mesh.is_watertight) and bool(mesh.is_winding_consistent)
                  and bool(mesh.volume > 0) and bool(separated) and probes["all_passed"] and widths_ok
                  and all(p["passed"] for p in primitive_reports))
        reports.append({"scene": scene.name, "passed": bool(passed),
                        "bounds_m": mesh.bounds.tolist(), "minimum_20mm_roi_margin": bounds_ok,
                        "watertight": bool(mesh.is_watertight), "consistent_winding": bool(mesh.is_winding_consistent),
                        "positive_volume": bool(mesh.volume > 0), "union_solids_disjoint": bool(separated),
                        "separation_method": separation_method, "primitive_checks": primitive_reports,
                        "feature_probe_reference_passed": probes["all_passed"], "cross_section_reference_passed": widths_ok})
    return {"passed": all(r["passed"] for r in reports), "scene_count": len(reports),
            "distance_tolerance_m": tolerance, "checks": reports,
            "limitation": "sampled validation; exact sphere uses analytic truth, not faceted mesh"}
