"""Known original-robot material and ray occlusion, with no scene geometry.

SELF is the union of the native MuJoCo collision hulls at the measured initial
joint state. Treating that material as non-environment is an explicit initial
noninterpenetration prior, not a visual free-space measurement. Nothing outside
the exact initial material is exempted. Rays clip original convex hull planes;
no sphere, capsule, bounding box or simplified gripper replaces a collider.
Axis-aligned bounds below are rejection-only acceleration structures.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import ConvexHull, Delaunay

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "field_planning"
sys.path.insert(0, str(PREVIOUS))
from robot_geometry import RobotGeometry


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InitialRobotSelfGeometry:
    """Immutable q0 native hull union in the explicitly calibrated camera frame."""
    def __init__(self, robot_or_package, tolerance_m=1e-10, rig_calibration_path=None):
        self.robot = robot_or_package if isinstance(robot_or_package, RobotGeometry) else RobotGeometry(robot_or_package, sample_spacing_m=None)
        self.tolerance_m = float(tolerance_m)
        if not np.isfinite(self.tolerance_m) or not 0 <= self.tolerance_m <= 1e-9:
            raise ValueError("only a numerical boundary tolerance between0and1nm is allowed")
        self.q_initial = self.robot.q_initial.copy()
        self._consumed = {}
        package = self.robot.package
        for path in [package / "robot.json", package / "robot.xml"] + [package / item["file"] for item in self.robot.metadata["assets"]]:
            self._consumed[str(path.resolve())] = _sha(path)
        for path in (Path(__file__).resolve(), PREVIOUS / "robot_geometry.py"):
            self._consumed[str(path)] = _sha(path)
        self.rig_K, self.image_shape, self.rig_calibration = None, None, None
        if rig_calibration_path is not None:
            path = Path(rig_calibration_path).resolve()
            calibration = json.loads(path.read_text())
            self.rig_K = self._validate_K(calibration["K"])
            self.image_shape = (int(calibration["height"]), int(calibration["width"]))
            if not np.allclose(calibration["T_camera_cv_from_world"], self.robot.camera_from_world, atol=1e-12, rtol=0):
                raise ValueError("rig and original robot camera frames disagree")
            self.rig_calibration = {"path": str(path), "sha256": _sha(path),
                                    "source_calibration_sha256": calibration.get("source_calibration_sha256"),
                                    "use": "known robot-rig intrinsic/extrinsic calibration only; no scene geometry"}
            self._consumed[str(path)] = _sha(path)
        self.hulls = []
        transforms = self.robot.transforms(self.q_initial)
        for gid in self.robot.geom_ids:
            gid = int(gid)
            vertices, faces = self.robot.local_meshes[gid]
            hull = ConvexHull(vertices)
            tetrahedra = Delaunay(vertices[hull.vertices])
            # Trigger Delaunay's lazy initialization before concurrent queries.
            tetrahedra.find_simplex(vertices.mean(axis=0)[None], tol=0.)
            transform = np.eye(4)
            transform[:3, :3] = transforms[gid]["rotation_world_from_local"]
            transform[:3, 3] = transforms[gid]["translation_world_m"]
            camera_transform = self.robot.camera_from_world @ transform
            rotation, translation = camera_transform[:3, :3], camera_transform[:3, 3]
            camera_vertices = vertices @ rotation.T + translation
            self.hulls.append({"geom_id": gid, "name": transforms[gid]["name"],
                               "rotation_camera_from_local": rotation, "translation_camera_m": translation,
                               "lower_camera_m": camera_vertices.min(axis=0), "upper_camera_m": camera_vertices.max(axis=0),
                               "equations": hull.equations.copy(), "vertices_local": vertices.copy(),
                               "faces": faces.copy(), "tetrahedra": tetrahedra})
        self._ray_image_cache = {}

    @staticmethod
    def _validate_K(K):
        K = np.asarray(K, float)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.array_equal(K[2], [0., 0., 1.]):
            raise ValueError("finite pinhole K with positive focal and lastrow[0,0,1] required")
        return K

    @staticmethod
    def _points(points):
        p = np.asarray(points, float)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError("expected Nx3 camera-frame coordinates")
        return p

    def contains_with_owner(self, points_camera):
        """Union membership, returning the first native geom owner or -1.

        Delaunay interior tests are acceleration only. Unresolved/boundary points
        are checked against every original hull plane, with a declared metric
        tolerance; query points outside the known material remain outside.
        """
        points = self._points(points_camera)
        owner = np.full(len(points), -1, dtype=np.int32)
        finite = np.all(np.isfinite(points), axis=1)
        for hull in self.hulls:
            candidate = finite & (owner < 0) & np.all(points >= hull["lower_camera_m"] - self.tolerance_m, axis=1) & np.all(points <= hull["upper_camera_m"] + self.tolerance_m, axis=1)
            ids = np.flatnonzero(candidate)
            if not len(ids):
                continue
            local = (points[ids] - hull["translation_camera_m"]) @ hull["rotation_camera_from_local"]
            inside = hull["tetrahedra"].find_simplex(local, tol=0.) >= 0
            remaining = np.flatnonzero(~inside)
            equations = hull["equations"]
            # Stop plane checks for a point as soon as any separating plane is found.
            for offset in range(0, len(equations), 64):
                if not len(remaining):
                    break
                planes = equations[offset:offset + 64]
                residual = local[remaining] @ planes[:, :3].T + planes[:, 3]
                remaining = remaining[np.all(residual <= self.tolerance_m, axis=1)]
            inside[remaining] = True
            owner[ids[inside]] = hull["geom_id"]
        return owner >= 0, owner

    def contains(self, points_camera):
        return self.contains_with_owner(points_camera)[0]

    def ray_intervals(self, directions_camera, origins_camera=None):
        """First positive intersection with q0 material, using native hulls.

        p(t)=origin+t*direction. For optical-origin rays with direction.z=1,
        returned t is camera-Z depth in metres. Exit is the far boundary of the
        FIRST intersected geom (not a claimed union exit). Misses have inf/-1.
        Any space behind the front hit remains visually occluded/unknown unless
        a separate source supports it; this routine makes no free-space claim.
        """
        direction = self._points(directions_camera)
        n = len(direction)
        origin = np.zeros((n, 3)) if origins_camera is None else np.broadcast_to(np.asarray(origins_camera, float), (n, 3))
        finite = np.all(np.isfinite(direction), axis=1) & np.all(np.isfinite(origin), axis=1) & (np.linalg.norm(direction, axis=1) > 0)
        front, exit_t = np.full(n, np.inf), np.full(n, np.inf)
        owner = np.full(n, -1, np.int32)
        for hull in self.hulls:
            # Exact slab rejection against a containing box; accepted rays are
            # subsequently tested against all original collision-hull planes.
            lo, hi = np.zeros(n), np.full(n, np.inf)
            valid = finite.copy()
            for axis in range(3):
                d, o = direction[:, axis], origin[:, axis]
                parallel = d == 0.
                valid &= ~parallel | ((o >= hull["lower_camera_m"][axis] - self.tolerance_m) & (o <= hull["upper_camera_m"][axis] + self.tolerance_m))
                with np.errstate(divide="ignore", invalid="ignore"):
                    a = (hull["lower_camera_m"][axis] - o) / d
                    b = (hull["upper_camera_m"][axis] - o) / d
                lo = np.maximum(lo, np.where(parallel, -np.inf, np.minimum(a, b)))
                hi = np.minimum(hi, np.where(parallel, np.inf, np.maximum(a, b)))
            ids = np.flatnonzero(valid & (hi >= lo) & (lo < front))
            equations = hull["equations"]
            for first in range(0, len(ids), 256):
                selected = ids[first:first + 256]
                local_o = (origin[selected] - hull["translation_camera_m"]) @ hull["rotation_camera_from_local"]
                local_d = direction[selected] @ hull["rotation_camera_from_local"]
                slope = local_d @ equations[:, :3].T
                intercept = local_o @ equations[:, :3].T + equations[:, 3]
                parallel = np.abs(slope) <= 32 * np.finfo(float).eps * np.linalg.norm(local_d, axis=1)[:, None]
                feasible = ~np.any(parallel & (intercept > self.tolerance_m), axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    roots = -intercept / slope
                entering = np.where((slope < 0) & ~parallel, roots, -np.inf).max(axis=1)
                leaving = np.where((slope > 0) & ~parallel, roots, np.inf).min(axis=1)
                entering = np.maximum(0., entering)
                hit = feasible & (leaving >= entering) & (entering < front[selected])
                accepted = selected[hit]
                front[accepted], exit_t[accepted], owner[accepted] = entering[hit], leaving[hit], hull["geom_id"]
        return {"front_t": front, "exit_t": exit_t, "geom_id": owner}

    def ray_image(self, K, image_shape):
        K = self._validate_K(K)
        height, width = map(int, image_shape)
        if min(height, width) <= 0:
            raise ValueError("positive image shape required")
        key = (K.tobytes(), height, width)
        if key not in self._ray_image_cache:
            y, x = np.mgrid[:height, :width]
            pixels = np.stack([x.ravel(), y.ravel(), np.ones(height * width)], axis=1)
            directions = pixels @ np.linalg.inv(K).T
            result = self.ray_intervals(directions)
            self._ray_image_cache[key] = {"front_depth_m": result["front_t"].reshape(height, width),
                                          "exit_depth_m": result["exit_t"].reshape(height, width),
                                          "geom_id": result["geom_id"].reshape(height, width)}
            for value in self._ray_image_cache[key].values():
                value.setflags(write=False)
        return self._ray_image_cache[key]

    def provenance(self, verify=True):
        intact = all(_sha(path) == digest for path, digest in self._consumed.items()) if verify else None
        if intact is False:
            raise ValueError("known robot source changed during self-geometry queries")
        return {"kind": "known_initial_native_robot_geometry", "consumed_sha256": dict(self._consumed),
                "inputs_unchanged": intact, "q_initial": self.q_initial.tolist(),
                "finger_q_initial": self.robot.finger_q_initial.tolist(),
                "geom_count": len(self.hulls), "geom_names": [h["name"] for h in self.hulls],
                "boundary_tolerance_m": self.tolerance_m, "intentional_dilation_m": 0.,
                "self_prior": "known initial robot material, conditional on valid robot state/calibration and no initial environment interpenetration",
                "outside_material": "no exemption; no free-space inference",
                "ray_exit_semantics": "exit of first-hit geom, not necessarily entire material union",
                "rig_calibration": self.rig_calibration,
                "environment_geometry_inputs": [], "simulator_contacts_used_to_choose_exemption": False}
