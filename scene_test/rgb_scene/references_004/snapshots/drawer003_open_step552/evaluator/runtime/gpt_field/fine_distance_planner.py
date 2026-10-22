"""Lazy 1 mm compact fields and on-demand shortest-path planning.

The lattice really contains 161**3 vertices over 16 cm. Values and AABB masks
are evaluated when queried; neither a Python list of four million potentials
nor a 26-edge-per-vertex graph is materialized. This is a storage change, not
coarse planning presented as a finer grid.

A valid straight path with uniform traversal density is an analytic Euclidean
shortest path to the goal region. Otherwise exact A* searches the actual
26-neighbour lattice using the existing endpoint-density edge cost. Its
Euclidean lower bound is admissible because traversal density is at least one.
Only explored vertices are retained; no dense solved distance map is claimed.
Predicted geometry remains a model estimate, never a collision certificate.
"""
from __future__ import annotations

import copy
import heapq
import json
import math
from itertools import product
from numbers import Real
from types import SimpleNamespace
from typing import Any

import numpy as np

from .compact_field import validate_compact_field
from .distance_planner import DistanceFieldPlanner
from .planner import CandidateRecord, PlanResult, PlannerConfig
from .schema import GridSpec, vector3

FINE_GRID = GridSpec(size=161, spacing_m=.001)
FINE_FIELD_VERSION = "lazy-compact-trilinear-1mm-v1"
_EPS = 1e-12
_GRID_EPS = 1e-10
_OFFSETS = tuple(d for d in product((-1, 0, 1), repeat=3) if any(d))


class _CompactPrediction(SimpleNamespace):
    """Auxiliary action heads with explicitly compact diagnostic serialization."""

    def to_dict(self):
        return {"representation": FINE_FIELD_VERSION, "grid": self.grid.to_dict(),
                "compact_field": copy.deepcopy(self.compact),
                "dense_values_materialized": False}


class FineCompactField:
    """Trilinear potential on a lazy lattice, with exact compact-box checks.

    Vertex masking and interpolation match ``expand_compact_field + GridField``.
    Closed AABB intersection is checked additionally, so subvoxel obstacles
    cannot disappear merely because they cover no lattice vertex.
    """

    def __init__(self, grid: GridSpec, center: Any, compact: dict):
        validate_compact_field(compact)
        if not isinstance(grid, GridSpec):
            raise ValueError("grid must be a GridSpec")
        self.grid, self.center = grid, vector3(center, "center")
        self.compact = copy.deepcopy(compact)
        self._lower = np.asarray(self.center) - grid.half_width_m
        self._upper = np.asarray(self.center) + grid.half_width_m
        self._boxes = DistanceFieldPlanner._boxes(self, self.compact)
        self.prediction = _CompactPrediction(
            **{key: copy.deepcopy(compact[key]) for key in
               ("stage", "orientation_target_xyzw", "gripper_command", "motion_mode")},
            grid=grid, compact=self.compact, values=(), blocked_indices=(), unknown_indices=())

    @property
    def lower(self):
        return self._lower.copy()

    @property
    def upper(self):
        return self._upper.copy()

    def _axes(self, point):
        lattice = (point - self._lower) / self.grid.spacing_m
        if np.any(lattice < -_GRID_EPS) or np.any(lattice > self.grid.size - 1 + _GRID_EPS):
            return None
        lattice = np.clip(lattice, 0., self.grid.size - 1.)
        axes = []
        for coordinate in lattice:
            nearest = int(round(float(coordinate)))
            if abs(float(coordinate) - nearest) <= _GRID_EPS:
                axes.append(((nearest, 1.),))
            else:
                low = int(math.floor(coordinate))
                fraction = float(coordinate - low)
                axes.append(((low, 1. - fraction), (low + 1, fraction)))
        return axes

    def _vertex_valid(self, point):
        # Use the same closed vertex mask as dense compact expansion, with
        # tolerance only for arithmetic roundoff at physical box boundaries.
        return not any(np.all(point >= low - _EPS) and np.all(point <= high + _EPS)
                       for low, high in self._boxes)

    def _point_valid(self, point):
        axes = self._axes(point)
        if axes is None or not self._vertex_valid(point):
            return False
        return all(self._vertex_valid(self._lower + np.asarray([x[0] for x in corner]) * self.grid.spacing_m)
                   for corner in product(*axes))

    def _potential(self, points):
        offsets = np.asarray(points) - np.asarray(self.center)
        difference = (offsets - np.asarray(self.compact["goal_offset_m"])) / .08
        values = np.sqrt(np.sum(np.asarray(self.compact["attraction_weights"]) * difference ** 2, axis=-1))
        for repulsor in self.compact["repulsors"]:
            squared = np.sum((offsets - np.asarray(repulsor["center_offset_m"])) ** 2, axis=-1)
            values += repulsor["weight"] * np.exp(-.5 * squared / repulsor["radius_m"] ** 2)
        return values

    def query(self, position):
        try:
            point = np.asarray(vector3(position, "position"))
        except ValueError:
            return None
        axes = self._axes(point)
        if axes is None or not self._vertex_valid(point):
            return None
        corners = list(product(*axes))
        points = self._lower + np.asarray([[x[0] for x in c] for c in corners]) * self.grid.spacing_m
        if any(not self._vertex_valid(p) for p in points):
            return None
        weights = np.asarray([math.prod(x[1] for x in c) for c in corners])
        return float(np.dot(weights, self._potential(points)))

    def query_many(self, positions):
        return [self.query(p) for p in positions]

    def segment_valid(self, start, end):
        try:
            a, b = np.asarray(vector3(start, "start")), np.asarray(vector3(end, "end"))
        except ValueError:
            return False
        if not self._point_valid(a) or not self._point_valid(b):
            return False
        if not self._boxes:
            return True  # The known cube is convex and every vertex is valid.
        if not DistanceFieldPlanner._avoids_boxes(a, b, self._boxes)[0]:
            return False
        crossings = {0., 1.}
        delta = b - a
        for axis in range(3):
            if delta[axis] == 0:
                continue
            planes = self._lower[axis] + np.arange(self.grid.size) * self.grid.spacing_m
            crossings.update(float(t) for t in (planes - a[axis]) / delta[axis] if 0 < t < 1)
        ordered = sorted(crossings)
        checks = ordered + [(a + b) * .5 for a, b in zip(ordered[:-1], ordered[1:])]
        return all(self._point_valid(a + t * delta) for t in checks)

    def to_dict(self):
        return {"grid": self.grid.to_dict(), "center": list(self.center),
                "prediction": self.prediction.to_dict()}


class FineDistancePlanner(DistanceFieldPlanner):
    """Analytic clear-space paths, otherwise exact lazy lattice A*.

    ``plan_analytic`` reserves half the caller's goal tolerance for measured
    tracking, matching the Speed-first planner contract. ``plan`` uses the
    supplied tolerance literally, useful for independent geometry tests.
    """

    def reset_cache(self):
        self._cache_key = None
        self._cache = None

    @property
    def distance_grid_m(self):
        # A* solves a query, not every vertex. Never fabricate a dense distance
        # field by assigning finite values to unexpanded nodes.
        return None

    def plan_analytic(self, field, position, *, goal_xyz, compact_prediction, goal_tolerance_m=.005):
        return self.plan(field, position, goal_xyz=goal_xyz, compact_prediction=compact_prediction,
                         goal_tolerance_m=goal_tolerance_m / 2)

    def _key(self, field, goal, tolerance, compact):
        return json.dumps({"grid": field.grid.to_dict(), "center": field.center,
            "goal": goal.tolist(), "tolerance": tolerance, "compact": compact,
            "blocked": field.prediction.blocked_indices, "unknown": field.prediction.unknown_indices,
            "workspace_min": self.config.workspace_min, "workspace_max": self.config.workspace_max},
            sort_keys=True, separators=(",", ":"))

    def _search(self, field, start, goal, tolerance, compact, boxes):
        size, spacing = field.grid.size, field.grid.spacing_m
        lower = tuple(float(x) for x in field.lower)
        work_low, work_high = self.config.workspace_min, self.config.workspace_max
        box_values = tuple((tuple(low), tuple(high)) for low, high in boxes)
        blocked = set(field.prediction.blocked_indices + field.prediction.unknown_indices)
        repulsors = tuple((tuple(np.asarray(field.center) + r["center_offset_m"]),
                          0.5 / r["radius_m"] ** 2, r["weight"]) for r in compact["repulsors"])
        validity, densities = self._cache["validity"], self._cache["densities"]
        width2 = size * size

        def xyz(index):
            i, rem = divmod(index, width2)
            j, k = divmod(rem, size)
            return lower[0] + i * spacing, lower[1] + j * spacing, lower[2] + k * spacing

        def valid(index):
            known = validity.get(index)
            if known is not None:
                return known
            p = xyz(index)
            ok = (index not in blocked and all(work_low[i] - _EPS <= p[i] <= work_high[i] + _EPS for i in range(3))
                  and not any(all(low[i] - _EPS <= p[i] <= high[i] + _EPS for i in range(3))
                              for low, high in box_values))
            validity[index] = ok
            return ok

        def density(index):
            known = densities.get(index)
            if known is None:
                p = xyz(index)
                known = 1. + sum(weight * math.exp(-scale * sum((p[i] - center[i]) ** 2 for i in range(3)))
                                 for center, scale, weight in repulsors)
                densities[index] = known
            return known

        def avoids(a, b):
            for low, high in box_values:
                enter, leave = 0., 1.
                for axis in range(3):
                    d = b[axis] - a[axis]
                    if abs(d) <= _EPS:
                        if a[axis] < low[axis] - _EPS or a[axis] > high[axis] + _EPS:
                            break
                    else:
                        x, y = (low[axis] - a[axis]) / d, (high[axis] - a[axis]) / d
                        enter, leave = max(enter, min(x, y)), min(leave, max(x, y))
                        if enter > leave + _EPS:
                            break
                else:
                    return False
            return True

        goal_coordinate = (goal - np.asarray(lower)) / spacing
        radius = tolerance / spacing
        axes = [range(max(0, math.ceil(c - radius - _GRID_EPS)),
                      min(size - 1, math.floor(c + radius + _GRID_EPS)) + 1) for c in goal_coordinate]
        terminals = {}
        for ijk in product(*axes):
            index = ijk[0] * width2 + ijk[1] * size + ijk[2]
            p = xyz(index)
            if valid(index) and math.dist(p, goal) <= tolerance + _EPS:
                terminals[index] = (0., p)
        for index in self._corners(field, goal):
            if not valid(index) or index in terminals:
                continue
            p = np.asarray(xyz(index))
            distance = np.linalg.norm(p - goal)
            endpoint = goal + (p - goal) * (tolerance / distance)
            if self._valid_segment(field, p, endpoint, boxes):
                cost = float(np.linalg.norm(p - endpoint) *
                             (density(index) + self._density(endpoint, field, compact)) * .5)
                terminals[index] = (cost, tuple(endpoint))
        if not terminals:
            return None, {"settled_vertex_count": 0, "seed_count": 0, "discovered_vertex_count": 0}

        edges = []
        for offset in _OFFSETS:
            delta = offset[0] * width2 + offset[1] * size + offset[2]
            corners = tuple(c[0] * width2 + c[1] * size + c[2]
                            for c in product(*(tuple({0, d}) for d in offset)))
            edges.append((offset, delta, spacing * math.sqrt(sum(d * d for d in offset)), corners))
        costs, parent, queue = {}, {}, []
        start_density = float(self._density(start, field, compact))

        def heuristic(p):
            return max(0., math.dist(p, goal) - tolerance)

        for index in self._corners(field, start):
            if not valid(index):
                continue
            p = np.asarray(xyz(index))
            if self._valid_segment(field, start, p, boxes):
                cost = float(np.linalg.norm(start - p) * (start_density + density(index)) * .5)
                costs[index], parent[index] = cost, None
                heapq.heappush(queue, (cost + heuristic(p), cost, index))
        best_cost, best_index, settled = math.inf, None, 0
        while queue:
            estimate, cost, index = heapq.heappop(queue)
            if estimate >= best_cost - _EPS:
                break
            if cost != costs[index]:
                continue
            settled += 1
            if index in terminals:
                terminal_cost = cost + terminals[index][0]
                if terminal_cost < best_cost:
                    best_cost, best_index = terminal_cost, index
            i, rem = divmod(index, width2)
            j, k = divmod(rem, size)
            p, current_density = xyz(index), density(index)
            for offset, delta, length, corners in edges:
                if not (0 <= i + offset[0] < size and 0 <= j + offset[1] < size and 0 <= k + offset[2] < size):
                    continue
                following = index + delta
                if not all(valid(index + corner) for corner in corners):
                    continue
                q = xyz(following)
                if box_values and not avoids(p, q):
                    continue
                proposed = cost + length * (current_density + density(following)) * .5
                if proposed < costs.get(following, math.inf):
                    costs[following], parent[following] = proposed, index
                    heapq.heappush(queue, (proposed + heuristic(q), proposed, following))
        metadata = {"settled_vertex_count": settled, "seed_count": len(terminals),
                    "discovered_vertex_count": len(costs), "cached_vertex_count": len(validity)}
        if best_index is None:
            return None, metadata
        route, index = [terminals[best_index][1]], best_index
        while index is not None:
            route.append(xyz(index))
            index = parent[index]
        route.append(tuple(start))
        raw = [np.asarray(p) for p in reversed(route)]
        raw = [p for n, p in enumerate(raw) if n == 0 or np.linalg.norm(p - raw[n - 1]) > _EPS]
        metadata["shortest_graph_cost_m"] = best_cost
        return np.asarray(raw), metadata

    def plan(self, field, position, *, goal_xyz, compact_prediction, goal_tolerance_m=.005):
        validate_compact_field(compact_prediction)
        start, goal = np.asarray(vector3(position, "position")), np.asarray(vector3(goal_xyz, "goal_xyz"))
        if (isinstance(goal_tolerance_m, bool) or not isinstance(goal_tolerance_m, Real)
                or not np.isfinite(goal_tolerance_m) or goal_tolerance_m <= 0):
            raise ValueError("goal_tolerance_m must be finite and positive")
        tolerance = float(goal_tolerance_m)
        distance = float(np.linalg.norm(goal - start))
        if not np.isfinite(distance):
            raise ValueError("distance between goal_xyz and position must be finite")
        boxes = self._boxes(field, compact_prediction)
        search = {"method": "lazy_fine_grid_astar", "connectivity": 26,
                  "grid": field.grid.to_dict(), "logical_vertex_count": field.grid.count,
                  "dense_graph_materialized": False, "dense_distance_grid_materialized": False,
                  "candidate_semantics": "one_extracted_graph_shortest_path",
                  "distance_semantics": "weighted_graph_path_cost_to_goal_region",
                  "edge_cost": "euclidean_length_times_mean_endpoint_density",
                  "cost_unit": "weighted_m" if compact_prediction["repulsors"] else "m",
                  "density_minimum": 1., "uses_attraction_potential": False,
                  "goal_xyz": goal.tolist(), "goal_tolerance_m": tolerance,
                  "goal_distance_m": distance, "goal_reached": False, "endpoint_is_goal": False,
                  "endpoint_in_goal_region": False, "boundary_clipped": False,
                  "direction_invalid": False, "needs_observation": False,
                  "cache_hit": False, "segment_count": 0,
                  "max_path_length_m": self.config.horizon * self.config.step_m}

        def stop(status, reason):
            search.update(path_termination=reason, needs_observation=status == "needs_observation",
                          direction_invalid=status == "no_valid_candidates")
            return PlanResult((0., 0., 0.), status, 0, 0, reason=reason, search_metadata=search)

        if field.prediction.motion_mode == "hold":
            return stop("hold", "explicit_motion_hold")
        if not self._valid_segment(field, start, start, boxes):
            return stop("no_valid_candidates", "invalid_start")
        if np.any(goal < field.lower - _EPS) or np.any(goal > field.upper + _EPS):
            return stop("needs_observation", "goal_outside_known_grid")
        if np.any(goal < self.config.workspace_min) or np.any(goal > self.config.workspace_max):
            return stop("no_valid_candidates", "goal_outside_workspace")
        if not self._valid_segment(field, goal, goal, boxes):
            return stop("needs_observation", "goal_region_not_known_valid")
        if distance <= tolerance + _EPS:
            search.update(goal_reached=True, endpoint_in_goal_region=True)
            return stop("goal_reached", "measured_goal_within_tolerance")
        key = self._key(field, goal, tolerance, compact_prediction)
        search["cache_hit"] = key == self._cache_key
        if not search["cache_hit"]:
            self._cache_key = key
            self._cache = {"validity": {}, "densities": {}}
        endpoint = goal + (start - goal) * (tolerance / distance)
        if not compact_prediction["repulsors"] and self._valid_segment(field, start, endpoint, boxes):
            raw = np.asarray([start, endpoint])
            search.update(method="analytic_uniform_goal_distance", connectivity=None,
                          candidate_semantics="one_continuous_euclidean_shortest_path",
                          distance_semantics="euclidean_distance_to_goal_region",
                          settled_vertex_count=0, discovered_vertex_count=0, seed_count=0)
        else:
            raw, metadata = self._search(field, start, goal, tolerance, compact_prediction, boxes)
            search.update(metadata)
            if raw is None:
                return stop("no_valid_candidates", "goal_unreachable_in_known_grid")
        lengths = np.linalg.norm(np.diff(raw, axis=0), axis=1)
        density = self._density(raw, field, compact_prediction)
        costs = lengths * (density[:-1] + density[1:]) * .5
        arc, cost_arc = np.r_[0., np.cumsum(lengths)], np.r_[0., np.cumsum(costs)]
        turns = [0]
        for index in range(1, len(raw) - 1):
            a, b = raw[index] - raw[index - 1], raw[index + 1] - raw[index]
            if np.dot(a, b) <= 0 or np.linalg.norm(np.cross(a, b)) > 1e-10 * np.linalg.norm(a) * np.linalg.norm(b):
                turns.append(index)
        turns.append(len(raw) - 1)
        points, arcs = [start], [0.]
        for left, right in zip(turns[:-1], turns[1:]):
            length, progress = float(arc[right] - arc[left]), 0.
            direction = (raw[right] - raw[left]) / length
            while progress < length - _EPS and len(points) <= self.config.horizon:
                progress = min(length, progress + self.config.step_m)
                points.append(raw[left] + progress * direction)
                arcs.append(float(arc[left] + progress))
            if len(points) > self.config.horizon:
                break
        if not all(self._valid_segment(field, a, b, boxes) for a, b in zip(points[:-1], points[1:])):
            return stop("no_valid_candidates", "extracted_path_failed_segment_validation")
        remaining = np.maximum(0., cost_arc[-1] - np.interp(arcs, arc, cost_arc))
        candidate = CandidateRecord(("distance_path",) * (len(points) - 1),
            tuple(tuple(float(v) for v in point) for point in points), float(arcs[-1]), True,
            float(remaining[-1]), float(remaining[1]))
        reached = bool(np.linalg.norm(points[-1] - goal) <= tolerance + _EPS)
        search.update(start_cost_m=float(cost_arc[-1]), terminal_cost_m=float(remaining[-1]),
                      shortest_graph_path_length_m=float(arc[-1]), endpoint_xyz=points[-1].tolist(),
                      segment_count=len(points) - 1,
                      endpoint_is_goal=bool(np.linalg.norm(points[-1] - goal) <= _EPS),
                      endpoint_in_goal_region=reached, path_termination="goal_region" if reached else "horizon")
        return PlanResult(tuple(float(v) for v in points[1] - start), "move", 1, 1, (candidate,), 0,
                          float(cost_arc[-1]), float(cost_arc[-1] - remaining[-1]), search_metadata=search)
