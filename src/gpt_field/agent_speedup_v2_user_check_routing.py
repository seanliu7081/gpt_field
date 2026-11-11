"""Default transit routes using a reusable model-estimated obstacle map.

The map stays fixed for one fragment. Boxes and clearance are model estimates;
this adapter checks the grip-site path against them, not whole-robot collision.
"""
from __future__ import annotations

from copy import deepcopy
import math
from time import perf_counter

import numpy as np

from .fine_distance_planner import FineCompactField, FineDistancePlanner
from .planner import PlannerConfig
from .schema import GridSpec, vector3
from .tracking import PathTracker


GRID_SPACING_M = .02
DETOUR_MARGIN_M = .10
MAX_PATH_PREFIXES = 20


class TransitObstacleMap:
    """Episode-local avoid set; updates are validated by the executive schema.

    Replacing the whole set also removes stale boxes. Reuse does not refresh its
    observation timestamp or infer object motion from gripper commands.
    """

    def __init__(self):
        self._route = {"avoid_boxes": [], "clearance_m": 0.}
        self.revision = 0
        self.updated_at_step = None

    def update(self, value, step):
        if value is None:
            return
        self._route = deepcopy({"avoid_boxes": value["avoid_boxes"],
                                "clearance_m": value["clearance_m"]})
        self.revision += 1
        self.updated_at_step = int(step)

    def snapshot(self):
        return {**deepcopy(self._route), "revision": self.revision,
                "updated_at_step": self.updated_at_step,
                "source": "model_estimate" if self.revision else "uninitialized"}


class RouteUnavailable(ValueError):
    """The supplied local map did not yield an executable route."""

    def __init__(self, reason, details=None):
        self.reason = str(reason)
        self.details = deepcopy(details or {})
        super().__init__(self.reason)


class TransitRoute:
    """Assemble a geometric route, then follow it from measured robot poses."""

    def __init__(self, start_xyz, goal_xyz, orientation_wxyz, route, *,
                 max_command_m, position_tolerance_m, grid_spacing_m=GRID_SPACING_M):
        started = perf_counter()
        self.start = np.asarray(vector3(start_xyz, "start_xyz"))
        self.goal = np.asarray(vector3(goal_xyz, "goal_xyz"))
        self.position_tolerance_m = float(position_tolerance_m)
        self.max_command_m = float(max_command_m)
        spacing = GridSpec(size=2, spacing_m=grid_spacing_m).spacing_m
        center = (self.start + self.goal) / 2
        half_cells = math.ceil((float(np.max(np.abs(self.goal - self.start))) / 2
                                + DETOUR_MARGIN_M) / spacing)
        grid = GridSpec(size=2 * half_cells + 1, spacing_m=spacing)
        if grid.half_width_m > .5:
            raise RouteUnavailable("route_exceeds_local_map")
        lower, upper = center - grid.half_width_m, center + grid.half_width_m
        clearance = float(route["clearance_m"])
        expanded_boxes, compact_boxes = [], []
        for box in route["avoid_boxes"]:
            low = np.asarray(box["lower_xyz_m"], dtype=float) - clearance
            high = np.asarray(box["upper_xyz_m"], dtype=float) + clearance
            clipped_low, clipped_high = np.maximum(low, lower), np.minimum(high, upper)
            intersects = bool(np.all(clipped_low <= clipped_high))
            expanded_boxes.append({"lower_xyz_m": low.tolist(), "upper_xyz_m": high.tolist(),
                                   "intersects_local_map": intersects})
            if intersects:
                # Preserve all box geometry inside this bounded planning volume.
                # Parts outside it cannot intersect a route confined to the volume.
                compact_boxes.append({"lower_offset_m": (clipped_low - center).tolist(),
                                      "upper_offset_m": (clipped_high - center).tolist()})
        w, x, y, z = orientation_wxyz
        compact = {"stage": "user_check_transit_route", "orientation_target_xyzw": [x, y, z, w],
                   "gripper_command": "hold", "motion_mode": "move",
                   "goal_offset_m": (self.goal - center).tolist(), "attraction_weights": [1., 1., 1.],
                   "repulsors": [], "blocked_boxes": compact_boxes, "unknown_boxes": []}
        self.field = FineCompactField(grid, center.tolist(), compact)
        self.workspace_min, self.workspace_max = tuple(lower), tuple(upper)
        self._diagnostics = {"grid": grid.to_dict(), "center_xyz_m": center.tolist(),
                             "clearance_m": clearance, "expanded_boxes": expanded_boxes,
                             "obstacle_count": len(expanded_boxes),
                             "intersecting_box_count": len(compact_boxes),
                             "max_command_m": self.max_command_m,
                             "scope": "model_supplied_boxes_and_grip_site_path",
                             "planner": {"calls": 0, "methods": [], "settled_vertex_count": 0}}
        if self.field.query(self.start) is None:
            raise RouteUnavailable("route_start_blocked", self._diagnostics)
        if self.field.query(self.goal) is None:
            raise RouteUnavailable("route_goal_blocked", self._diagnostics)

        points = [self.start.copy()]
        straight_valid = self.field.segment_valid(self.start, self.goal)
        self._diagnostics["straight_path_valid"] = bool(straight_valid)
        if straight_valid:
            self._diagnostics["planner"]["methods"] = ["direct_segment"]
        else:
            planner = FineDistancePlanner(PlannerConfig(
                horizon=10, step_m=.01, workspace_min=self.workspace_min,
                workspace_max=self.workspace_max))
            tolerance = min(self.position_tolerance_m / 2, .001)
            # The legacy public API returns at most ten 1 cm segments. Assemble
            # its validated prefixes without modifying that planner's contract.
            for _ in range(MAX_PATH_PREFIXES):
                plan = planner.plan(self.field, points[-1], goal_xyz=self.goal,
                                    compact_prediction=compact, goal_tolerance_m=tolerance)
                metadata = plan.search_metadata or {}
                evidence = self._diagnostics["planner"]
                evidence["calls"] += 1
                method = metadata.get("method")
                if method not in evidence["methods"]:
                    evidence["methods"].append(method)
                evidence["settled_vertex_count"] += int(metadata.get("settled_vertex_count", 0))
                if plan.status == "goal_reached":
                    break
                if plan.status != "move" or plan.selected_index is None:
                    raise RouteUnavailable(plan.reason or "route_not_found", self._diagnostics)
                prefix = np.asarray(plan.candidates[plan.selected_index].positions, dtype=float)
                if len(prefix) < 2 or float(np.linalg.norm(prefix[-1] - points[-1])) < 1e-9:
                    raise RouteUnavailable("route_planner_no_progress", self._diagnostics)
                points.extend(point.copy() for point in prefix[1:])
                if metadata.get("endpoint_in_goal_region"):
                    break
            else:
                raise RouteUnavailable("route_length_limit", self._diagnostics)

        # The graph planner may stop at its goal tolerance. Keep the model's
        # exact endpoint and validate the final connector as well.
        if not self.field.segment_valid(points[-1], self.goal):
            raise RouteUnavailable("route_goal_connector_blocked", self._diagnostics)
        if float(np.linalg.norm(points[-1] - self.goal)) > 1e-12:
            points.append(self.goal.copy())
        self.path = np.asarray(points)
        self.lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.arcs = np.r_[0., np.cumsum(self.lengths)]
        self.tracker = PathTracker(lookahead_m=max(.03, self.max_command_m),
                                   max_command_m=self.max_command_m,
                                   endpoint_tolerance_m=self.position_tolerance_m)
        if len(self.path) > 1:
            self.tracker.reset(self.path)
        self._diagnostics.update(path_xyz_m=self.path.tolist(), path_length_m=float(self.arcs[-1]),
                                 planning_s=perf_counter() - started,
                                 method="local_box_transit_route")

    def command(self, position):
        """Return a checked Cartesian command and measured-path diagnostics."""
        if len(self.path) == 1:
            # An in-place orientation stage still holds its XYZ target when
            # robot dynamics cause translational drift.
            point = np.asarray(vector3(position, "position"))
            if not self.field.segment_valid(point, self.goal):
                raise RouteUnavailable("no_valid_reference", self._diagnostics)
            delta = self.goal - point
            distance = float(np.linalg.norm(delta))
            arrived = distance <= self.position_tolerance_m
            delta = np.zeros(3) if arrived else delta * min(1., self.max_command_m / distance)
            return tuple(float(x) for x in delta), {
                "status": "path_complete" if arrived else "tracking",
                "remaining_path_m": distance, "endpoint_distance_m": distance,
                "command_target_xyz": (point + delta).tolist(),
                "command_norm_m": float(np.linalg.norm(delta)),
            }
        delta, details = self.tracker.command(position, self.field,
                                              self.workspace_min, self.workspace_max)
        if details["status"] in ("invalid_measured_start", "no_valid_reference", "no_selected_path"):
            raise RouteUnavailable(details["status"], details)
        details["remaining_path_m"] = self.progress(position)
        return delta, details

    def progress(self, position):
        """Remaining polyline distance plus measured distance back to the path."""
        point = np.asarray(vector3(position, "position"))
        if len(self.path) == 1:
            return float(np.linalg.norm(point - self.goal))
        best_distance, best_arc = math.inf, self.tracker.progress_m
        for index, length in enumerate(self.lengths):
            if self.arcs[index + 1] < self.tracker.progress_m:
                continue
            direction = self.path[index + 1] - self.path[index]
            low = max(0., (self.tracker.progress_m - self.arcs[index]) / length)
            fraction = float(np.clip(np.dot(point - self.path[index], direction) / length ** 2, low, 1.))
            distance = float(np.linalg.norm(point - self.path[index] - fraction * direction))
            if distance < best_distance - 1e-12:
                best_distance = distance
                best_arc = float(self.arcs[index] + fraction * length)
        return float(best_distance + self.arcs[-1] - best_arc)

    def diagnostics(self):
        return deepcopy(self._diagnostics)
