"""Cached, obstacle-aware cost-to-go on the model's known local grid.

This module does not interpret the compact attraction potential as a distance.
It solves a 26-neighbour graph with Euclidean edge lengths, optionally weighted
by a nonnegative Gaussian traversal density. Unknown and blocked geometry is
excluded; validity still describes model predictions, not certified collisions.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from itertools import product
from numbers import Real
from typing import Any

import numpy as np

from .compact_field import validate_compact_field
from .field import GridField
from .planner import CandidateRecord, PlanResult, PlannerConfig
from .schema import vector3

_OFFSETS = tuple(delta for delta in product((-1, 0, 1), repeat=3) if any(delta))
_EPS = 1e-12


class DistanceFieldPlanner:
    """Dijkstra cost-to-go with one extracted, bounded local path per call.

    The optimality claim is for the finite 26-neighbour graph, with edge cost
    ``length * mean(endpoint_density)``. It is not an exact continuous Eikonal
    solver. Each diagonal checks every corner of its interpolation stencil, so
    it cannot cut through a masked corner. Compact boxes are additionally
    checked continuously, including boxes thinner than the grid spacing.

    ``distance_grid_m`` exposes the last immutable cached cost grid; unreachable
    and invalid vertices have infinity. Never serialize it as finite potentials.
    Cache entries depend on field contents, center, goal, tolerance and workspace.
    """

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig(horizon=10)
        if self.config.step_m > .01 + _EPS:
            raise ValueError("distance-field path segments must not exceed 0.01 m")
        self.reset_cache()

    def reset_cache(self) -> None:
        self._cache_key: str | None = None
        self._cache: dict[str, Any] | None = None

    @property
    def distance_grid_m(self) -> np.ndarray | None:
        return None if self._cache is None else self._cache["distance"].reshape(self._cache["shape"])

    @staticmethod
    def _density(points: np.ndarray, field: GridField, compact: dict) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        density = np.ones(points.shape[:-1], dtype=float)
        for repulsor in compact["repulsors"]:
            center = np.asarray(field.center) + np.asarray(repulsor["center_offset_m"])
            squared = np.sum((points - center) ** 2, axis=-1)
            density += repulsor["weight"] * np.exp(-.5 * squared / repulsor["radius_m"] ** 2)
        return density

    @staticmethod
    def _boxes(field: GridField, compact: dict) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(np.asarray(field.center) + np.asarray(box["lower_offset_m"]),
                 np.asarray(field.center) + np.asarray(box["upper_offset_m"]))
                for name in ("blocked_boxes", "unknown_boxes") for box in compact[name]]

    @staticmethod
    def _avoids_boxes(starts: np.ndarray, ends: np.ndarray, boxes: list) -> np.ndarray:
        """Closed-segment/closed-AABB slab test, vectorized over segments."""
        starts, ends = np.atleast_2d(starts), np.atleast_2d(ends)
        result = np.ones(len(starts), dtype=bool)
        delta = ends - starts
        for low, high in boxes:
            enter, leave = np.zeros(len(starts)), np.ones(len(starts))
            possible = np.ones(len(starts), dtype=bool)
            for axis in range(3):
                moving = np.abs(delta[:, axis]) > _EPS
                possible &= moving | ((starts[:, axis] >= low[axis] - _EPS)
                                       & (starts[:, axis] <= high[axis] + _EPS))
                a = np.divide(low[axis] - starts[:, axis], delta[:, axis],
                              out=np.full(len(starts), -np.inf), where=moving)
                b = np.divide(high[axis] - starts[:, axis], delta[:, axis],
                              out=np.full(len(starts), np.inf), where=moving)
                enter = np.maximum(enter, np.minimum(a, b))
                leave = np.minimum(leave, np.maximum(a, b))
            result &= ~(possible & (enter <= leave + _EPS))
        return result

    def _valid_segment(self, field: GridField, start: np.ndarray, end: np.ndarray, boxes: list) -> bool:
        points = np.asarray([start, end])
        return bool(np.all(points >= np.asarray(self.config.workspace_min) - _EPS)
                    and np.all(points <= np.asarray(self.config.workspace_max) + _EPS)
                    and field.segment_valid(start, end)
                    and self._avoids_boxes(start, end, boxes)[0])

    @staticmethod
    def _corners(field: GridField, point: np.ndarray) -> list[int]:
        lattice = np.clip((point - field.lower) / field.grid.spacing_m, 0, field.grid.size - 1)
        axes = []
        for coordinate in lattice:
            nearest = int(round(float(coordinate)))
            if abs(coordinate - nearest) <= 1e-10:
                axes.append((nearest,))
            else:
                axes.append((int(np.floor(coordinate)), int(np.ceil(coordinate))))
        return [int(np.ravel_multi_index(index, (field.grid.size,) * 3)) for index in product(*axes)]

    def _key(self, field: GridField, goal: np.ndarray, tolerance: float, compact: dict) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(field.prediction.values, dtype=np.float64).tobytes())
        digest.update(json.dumps({"grid": field.grid.to_dict(), "center": field.center,
                                 "blocked": field.prediction.blocked_indices,
                                 "unknown": field.prediction.unknown_indices,
                                 "goal": goal.tolist(), "tolerance": tolerance,
                                 "workspace_min": self.config.workspace_min,
                                 "workspace_max": self.config.workspace_max,
                                 "repulsors": compact["repulsors"],
                                 "blocked_boxes": compact["blocked_boxes"],
                                 "unknown_boxes": compact["unknown_boxes"]},
                                sort_keys=True, separators=(",", ":")).encode())
        return digest.hexdigest()

    def _build(self, field: GridField, goal: np.ndarray, tolerance: float, compact: dict, boxes: list) -> dict:
        size, count = field.grid.size, field.grid.count
        shape = (size,) * 3
        points = field.grid.coordinates(field.center)
        valid = np.ones(count, dtype=bool)
        invalid = field.prediction.blocked_indices + field.prediction.unknown_indices
        if invalid:
            valid[list(invalid)] = False
        valid &= np.all(points >= np.asarray(self.config.workspace_min) - _EPS, axis=1)
        valid &= np.all(points <= np.asarray(self.config.workspace_max) + _EPS, axis=1)
        valid &= self._avoids_boxes(points, points, boxes)
        stencil_valid = valid.reshape(shape)
        density = self._density(points, field, compact)
        indices = np.arange(count, dtype=np.int32).reshape(shape)
        neighbours = np.full((count, 26), -1, dtype=np.int32)
        weights = np.zeros((count, 26), dtype=float)
        for column, offset in enumerate(_OFFSETS):
            starts = [max(0, -delta) for delta in offset]
            stops = [min(size, size - delta) for delta in offset]
            source_slice = tuple(slice(a, b) for a, b in zip(starts, stops))
            target_slice = tuple(slice(a + delta, b + delta) for a, b, delta in zip(starts, stops, offset))
            source = indices[source_slice].ravel()
            target = indices[target_slice].ravel()
            edge_valid = np.ones(len(source), dtype=bool)
            # Between adjacent vertices, these are exactly all active corners
            # on the segment interior (axis-aligned edges need just 2 corners).
            for corner in product(*(tuple({0, delta}) for delta in offset)):
                corners = tuple(slice(a + delta, b + delta) for a, b, delta in zip(starts, stops, corner))
                edge_valid &= stencil_valid[corners].ravel()
            edge_valid &= self._avoids_boxes(points[source], points[target], boxes)
            source, target = source[edge_valid], target[edge_valid]
            neighbours[source, column] = target
            weights[source, column] = (field.grid.spacing_m * np.linalg.norm(offset)
                                       * (density[source] + density[target]) * .5)

        distance = np.full(count, np.inf)
        successor = np.full(count, -1, dtype=np.int32)
        seed_end: dict[int, np.ndarray] = {}
        goal_distance = np.linalg.norm(points - goal, axis=1)
        seeds = np.flatnonzero(valid & (goal_distance <= tolerance + _EPS))
        for index in seeds:
            distance[index] = 0.
            seed_end[int(index)] = points[index]
        # Virtual terminal edges let small/fractional goal regions work even
        # when they contain no grid vertex. They end on the tolerance sphere.
        for index in self._corners(field, goal):
            if not valid[index] or goal_distance[index] <= tolerance + _EPS:
                continue
            endpoint = goal + (points[index] - goal) * (tolerance / goal_distance[index])
            if not self._valid_segment(field, points[index], endpoint, boxes):
                continue
            cost = float(np.linalg.norm(points[index] - endpoint)
                         * (density[index] + self._density(endpoint, field, compact)) * .5)
            distance[index] = cost
            seed_end[index] = endpoint
        heap = [(float(distance[index]), index) for index in seed_end]
        heapq.heapify(heap)
        settled = 0
        while heap:
            cost, index = heapq.heappop(heap)
            if cost != distance[index]:
                continue
            settled += 1
            for neighbour, weight in zip(neighbours[index], weights[index]):
                if neighbour < 0:
                    continue
                proposed = cost + weight
                if proposed < distance[neighbour]:
                    distance[neighbour] = proposed
                    successor[neighbour] = index
                    heapq.heappush(heap, (float(proposed), int(neighbour)))
        distance.setflags(write=False)
        return {"distance": distance, "successor": successor, "seed_end": seed_end,
                "points": points, "density": density, "shape": shape,
                "settled_vertex_count": settled, "seed_count": len(seed_end),
                "valid_vertex_count": int(valid.sum())}

    def plan_analytic(self, field: GridField, position: Any, *, goal_xyz: Any,
                      compact_prediction: dict[str, Any], goal_tolerance_m: float = .005) -> PlanResult:
        """Compatibility entry point for FieldPolicy; always runs Dijkstra.

        The name matches the existing opt-in policy hook, not the algorithm.
        Diagnostics retain ``method=dijkstra_distance_field``.
        """
        return self.plan(field, position, goal_xyz=goal_xyz,
                         compact_prediction=compact_prediction, goal_tolerance_m=goal_tolerance_m)

    def plan(self, field: GridField, position: Any, *, goal_xyz: Any,
             compact_prediction: dict[str, Any], goal_tolerance_m: float = .005) -> PlanResult:
        validate_compact_field(compact_prediction)
        start, goal = np.asarray(vector3(position, "position")), np.asarray(vector3(goal_xyz, "goal_xyz"))
        if (isinstance(goal_tolerance_m, bool) or not isinstance(goal_tolerance_m, Real)
                or not np.isfinite(goal_tolerance_m) or goal_tolerance_m <= 0):
            raise ValueError("goal_tolerance_m must be finite and positive")
        tolerance = float(goal_tolerance_m)
        with np.errstate(over="ignore", invalid="ignore"):
            goal_distance = float(np.linalg.norm(goal - start))
        if not np.isfinite(goal_distance):
            raise ValueError("distance between goal_xyz and position must be finite")
        boxes = self._boxes(field, compact_prediction)
        search = {"method": "dijkstra_distance_field", "connectivity": 26,
                  "candidate_semantics": "one_extracted_graph_shortest_path",
                  "distance_semantics": "weighted_graph_path_cost_to_goal_region",
                  "edge_cost": "euclidean_length_times_mean_endpoint_density",
                  "cost_unit": "weighted_m" if compact_prediction["repulsors"] else "m",
                  "density_minimum": 1., "uses_attraction_potential": False,
                  "goal_xyz": goal.tolist(), "goal_tolerance_m": tolerance,
                  "goal_distance_m": goal_distance,
                  "goal_reached": False, "endpoint_is_goal": False,
                  "endpoint_in_goal_region": False, "boundary_clipped": False,
                  "direction_invalid": False, "needs_observation": False,
                  "cache_hit": False, "segment_count": 0,
                  "max_path_length_m": self.config.horizon * self.config.step_m}

        def stop(status: str, reason: str) -> PlanResult:
            search["path_termination"] = reason
            search["needs_observation"] = status == "needs_observation"
            search["direction_invalid"] = status == "no_valid_candidates"
            return PlanResult((0., 0., 0.), status, 0, 0, reason=reason, search_metadata=search)

        if field.prediction.motion_mode == "hold":
            return stop("hold", "explicit_motion_hold")
        if not self._valid_segment(field, start, start, boxes):
            return stop("no_valid_candidates", "invalid_start")
        if np.any(goal < field.lower - _EPS) or np.any(goal > field.upper + _EPS):
            return stop("needs_observation", "goal_outside_known_grid")
        if (np.any(goal < np.asarray(self.config.workspace_min) - _EPS)
                or np.any(goal > np.asarray(self.config.workspace_max) + _EPS)):
            return stop("no_valid_candidates", "goal_outside_workspace")
        if not self._valid_segment(field, goal, goal, boxes):
            return stop("needs_observation", "goal_region_not_known_valid")
        if np.linalg.norm(goal - start) <= tolerance + _EPS:
            search.update(goal_reached=True, endpoint_in_goal_region=True)
            return stop("goal_reached", "measured_goal_within_tolerance")
        key = self._key(field, goal, tolerance, compact_prediction)
        search["cache_hit"] = self._cache is not None and key == self._cache_key
        if not search["cache_hit"]:
            self._cache = self._build(field, goal, tolerance, compact_prediction, boxes)
            self._cache_key = key
        cache = self._cache
        assert cache is not None
        search.update({name: cache[name] for name in ("settled_vertex_count", "seed_count", "valid_vertex_count")})
        start_density = float(self._density(start, field, compact_prediction))
        choices = []
        for index in self._corners(field, start):
            if not np.isfinite(cache["distance"][index]):
                continue
            vertex = cache["points"][index]
            if self._valid_segment(field, start, vertex, boxes):
                cost = (float(np.linalg.norm(vertex - start)) * (start_density + cache["density"][index]) * .5
                        + cache["distance"][index])
                choices.append((float(cost), index))
        if not choices:
            return stop("no_valid_candidates", "goal_unreachable_in_known_grid")
        initial_cost, index = min(choices)
        points = [start]
        while True:
            point = cache["points"][index]
            if np.linalg.norm(point - points[-1]) > _EPS:
                points.append(point)
            following = int(cache["successor"][index])
            if following < 0:
                endpoint = cache["seed_end"][index]
                if np.linalg.norm(endpoint - points[-1]) > _EPS:
                    points.append(endpoint)
                break
            index = following
        raw = np.asarray(points)
        lengths = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        densities = self._density(raw, field, compact_prediction)
        costs = lengths * (densities[:-1] + densities[1:]) * .5
        arc = np.r_[0., np.cumsum(lengths)]
        cumulative_cost = np.r_[0., np.cumsum(costs)]
        # Merge only collinear graph edges. No shortcut can alter the route or
        # erase the selected obstacle/repulsor detour.
        turns = [0]
        for i in range(1, len(raw) - 1):
            a, b = raw[i] - raw[i - 1], raw[i + 1] - raw[i]
            if np.dot(a, b) <= 0 or np.linalg.norm(np.cross(a, b)) > 1e-10 * np.linalg.norm(a) * np.linalg.norm(b):
                turns.append(i)
        turns.append(len(raw) - 1)
        selected, selected_arcs = [start], [0.]
        for left, right in zip(turns[:-1], turns[1:]):
            distance = float(arc[right] - arc[left])
            direction = (raw[right] - raw[left]) / distance
            progress = 0.
            while progress < distance - _EPS and len(selected) <= self.config.horizon:
                progress = min(distance, progress + self.config.step_m)
                selected.append(raw[left] + progress * direction)
                selected_arcs.append(float(arc[left] + progress))
            if len(selected) > self.config.horizon:
                break
        valid = all(self._valid_segment(field, a, b, boxes) for a, b in zip(selected[:-1], selected[1:]))
        if not valid:
            return stop("no_valid_candidates", "extracted_path_failed_segment_validation")
        remaining = np.maximum(0., cumulative_cost[-1] - np.interp(selected_arcs, arc, cumulative_cost))
        # Start/terminal fields in the compatibility record carry the new cost,
        # never the legacy arbitrary attraction potential.
        initial_cost = float(cumulative_cost[-1])
        candidate = CandidateRecord(("distance_path",) * (len(selected) - 1),
                                    tuple(tuple(float(v) for v in point) for point in selected),
                                    float(selected_arcs[-1]), True, float(remaining[-1]), float(remaining[1]))
        reached = float(np.linalg.norm(selected[-1] - goal)) <= tolerance + _EPS
        search.update(start_cost_m=initial_cost, terminal_cost_m=float(remaining[-1]),
                      shortest_graph_path_length_m=float(arc[-1]),
                      endpoint_xyz=selected[-1].tolist(), segment_count=len(selected) - 1,
                      endpoint_is_goal=bool(np.linalg.norm(selected[-1] - goal) <= _EPS),
                      endpoint_in_goal_region=reached,
                      path_termination="goal_region" if reached else "horizon")
        return PlanResult(tuple(float(v) for v in selected[1] - start), "move", 1, 1,
                          (candidate,), 0, initial_cost, float(initial_cost - remaining[-1]),
                          search_metadata=search)
