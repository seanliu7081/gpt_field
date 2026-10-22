"""Measured-pose following of a geometric path, without simulator state access."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from .field import GridField
from .schema import vector3


class PathTracker:
    """Follow a fixed world-frame polyline with a bounded, checked lookahead.

    Progress follows the observed end-effector position, never elapsed actions.
    Segment checks concern the predicted field masks, not physical collision
    guarantees. A stopped or invalid path yields a zero translation command.
    """

    def __init__(self, *, lookahead_m: float = 0.03, max_command_m: float = 0.05,
                 endpoint_tolerance_m: float = 1e-9):
        for name, value in (("lookahead_m", lookahead_m), ("max_command_m", max_command_m),
                            ("endpoint_tolerance_m", endpoint_tolerance_m)):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if max_command_m > 0.05:
            raise ValueError("max_command_m must not exceed the OSC translation norm limit of 0.05 m")
        self.lookahead_m = float(lookahead_m)
        self.max_command_m = float(max_command_m)
        self.endpoint_tolerance_m = float(endpoint_tolerance_m)
        self.reset()

    def reset(self, positions=None):
        self.positions = (None if positions is None else
                          np.asarray([vector3(point, "path position") for point in positions]))
        if self.positions is not None and len(self.positions) < 2:
            raise ValueError("tracking path must contain at least two positions")
        self.lengths = (np.array([], dtype=float) if self.positions is None else
                        np.linalg.norm(np.diff(self.positions, axis=0), axis=1))
        if np.any(self.lengths <= 0):
            raise ValueError("tracking path segments must have positive length")
        self.arcs = np.r_[0., np.cumsum(self.lengths)]
        self.progress_m = 0.

    def endpoint_distance(self, position):
        """Measured distance to the final point; projection progress is not arrival."""
        point = np.asarray(vector3(position, "position"))
        if self.positions is None:
            return None
        return float(np.linalg.norm(point - self.positions[-1]))

    def endpoint_reached(self, position):
        distance = self.endpoint_distance(position)
        return distance is not None and distance <= self.endpoint_tolerance_m

    def _point(self, arc):
        index = min(int(np.searchsorted(self.arcs, arc, side="right")) - 1, len(self.lengths) - 1)
        index = max(index, 0)
        fraction = np.clip((arc - self.arcs[index]) / self.lengths[index], 0., 1.)
        return self.positions[index] + fraction * (self.positions[index + 1] - self.positions[index])

    def command(self, position, field: GridField, workspace_min, workspace_max):
        point = np.asarray(vector3(position, "position"))
        minimum = np.asarray(workspace_min)
        maximum = np.asarray(workspace_max)
        details = {"lookahead_m": self.lookahead_m, "max_command_m": self.max_command_m,
                   "progress_m": self.progress_m, "path_length_m": float(self.arcs[-1]),
                   "endpoint_tolerance_m": self.endpoint_tolerance_m,
                   "endpoint_distance_m": self.endpoint_distance(point),
                   "endpoint_reached": self.endpoint_reached(point),
                   "reference_xyz": None, "reference_progress_m": None,
                   "command_target_xyz": point.tolist(), "command_norm_m": 0.}

        def stop(status):
            details["status"] = status
            return (0., 0., 0.), details

        if self.positions is None:
            return stop("no_selected_path")
        if field.query(point) is None or np.any(point < minimum) or np.any(point > maximum):
            return stop("invalid_measured_start")
        # A practical arrival tolerance is opt-in. Keep the legacy default's
        # projection/lookahead and tiny residual commands unchanged. In the new
        # mode, measured endpoint proximity ends translation without consuming
        # the remainder of an action-count window. The policy decides whether
        # to replan or request a new field; this is not task-completion logic.
        if (self.endpoint_tolerance_m > 1e-9 and details["endpoint_reached"]
                and np.all(self.positions[-1] >= minimum) and np.all(self.positions[-1] <= maximum)
                and field.segment_valid(point, self.positions[-1])):
            details.update(reference_xyz=self.positions[-1].tolist(),
                           reference_progress_m=float(self.arcs[-1]), shortened_for_validity=False)
            return stop("path_complete")

        # Closest projection onto the remaining path. Equal-distance ties use
        # the earliest arc position, avoiding artificial jumps at path loops.
        best_distance, best_progress = float("inf"), self.progress_m
        for index, length in enumerate(self.lengths):
            if self.arcs[index + 1] < self.progress_m:
                continue
            direction = self.positions[index + 1] - self.positions[index]
            lower = max(0., (self.progress_m - self.arcs[index]) / length)
            fraction = np.clip(np.dot(point - self.positions[index], direction) / length ** 2, lower, 1.)
            projection = self.positions[index] + fraction * direction
            distance = float(np.linalg.norm(point - projection))
            arc = float(self.arcs[index] + fraction * length)
            if distance < best_distance - 1e-12:
                best_distance, best_progress = distance, arc
        self.progress_m = max(self.progress_m, best_progress)
        details["progress_m"] = self.progress_m
        details["distance_to_path_m"] = best_distance
        desired_arc = min(float(self.arcs[-1]), self.progress_m + self.lookahead_m)

        # A lookahead chord may cut across a masked corner. Try successively
        # earlier path vertices, then the measured-position projection, rather
        # than issuing an unchecked shortcut across that corner.
        references = [desired_arc]
        references.extend(float(arc) for arc in reversed(self.arcs)
                          if self.progress_m < arc < desired_arc)
        references.append(self.progress_m)
        for arc in references:
            reference = self._point(arc)
            if np.any(reference < minimum) or np.any(reference > maximum):
                continue
            if not field.segment_valid(point, reference):
                continue
            delta = reference - point
            distance = float(np.linalg.norm(delta))
            if distance > self.max_command_m:
                delta *= self.max_command_m / distance
            target = point + delta
            if not field.segment_valid(point, target):
                continue
            details.update(reference_xyz=reference.tolist(), reference_progress_m=float(arc),
                           command_target_xyz=target.tolist(), command_norm_m=float(np.linalg.norm(delta)),
                           shortened_for_validity=arc < desired_arc - 1e-12,
                           status="path_complete" if distance <= 1e-9 and arc >= self.arcs[-1] - 1e-12
                           else "tracking")
            return tuple(float(value) for value in delta), details
        return stop("no_valid_reference")
