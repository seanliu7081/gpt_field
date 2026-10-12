"""Equal-travel-length receding-horizon search over nonzero axis motions.

Terminal potential is the objective; no physical-distance interpretation is
required. Waiting is available only through an explicit model hold command or
the absence of a valid candidate. Masks are predictions, not collision proof.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from numbers import Integral, Real
from typing import Any

import numpy as np

from .field import GridField
from .schema import vector3


_AXES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
_LABELS = ("+x", "-x", "+y", "-y", "+z", "-z")


@dataclass(frozen=True)
class PlannerConfig:
    step_m: float = 0.01
    horizon: int = 3
    workspace_min: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    workspace_max: tuple[float, float, float] = (1.0, 1.0, 1.5)
    # Potential units, not meters. Only classifies progress; never alters rank.
    plateau_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.step_m, bool) or not isinstance(self.step_m, Real) or not np.isfinite(self.step_m) or self.step_m <= 0:
            raise ValueError("step_m must be finite and positive")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, Integral) or not 1 <= self.horizon <= 10:
            raise ValueError("horizon must be an integer between 1 and 10")
        if isinstance(self.plateau_tolerance, bool) or not isinstance(self.plateau_tolerance, Real) or not np.isfinite(self.plateau_tolerance) or self.plateau_tolerance < 0:
            raise ValueError("plateau_tolerance must be finite and nonnegative")
        minimum = vector3(self.workspace_min, "workspace_min")
        maximum = vector3(self.workspace_max, "workspace_max")
        if any(low >= high for low, high in zip(minimum, maximum)):
            raise ValueError("workspace_min must be strictly less than workspace_max")
        object.__setattr__(self, "workspace_min", minimum)
        object.__setattr__(self, "workspace_max", maximum)
        object.__setattr__(self, "step_m", float(self.step_m))
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "plateau_tolerance", float(self.plateau_tolerance))


@dataclass(frozen=True)
class CandidateRecord:
    primitives: tuple[str, ...]
    positions: tuple[tuple[float, float, float], ...]
    path_length_m: float
    valid: bool
    terminal_potential: float | None
    immediate_potential: float | None
    invalid_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitives": list(self.primitives),
            "positions": [list(point) for point in self.positions],
            "path_length_m": self.path_length_m,
            "valid": self.valid,
            "terminal_potential": self.terminal_potential,
            "immediate_potential": self.immediate_potential,
            "invalid_reason": self.invalid_reason,
        }


@dataclass(frozen=True)
class PlanResult:
    delta_xyz: tuple[float, float, float]
    status: str
    candidate_count: int
    valid_count: int
    candidates: tuple[CandidateRecord, ...] = ()
    selected_index: int | None = None
    start_potential: float | None = None
    predicted_improvement: float | None = None
    reason: str | None = None
    numerical_tie_tolerance: float = 0.0
    search_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "delta_xyz": list(self.delta_xyz),
            "status": self.status,
            "candidate_count": self.candidate_count,
            "valid_count": self.valid_count,
            "selected_index": self.selected_index,
            "start_potential": self.start_potential,
            "predicted_improvement": self.predicted_improvement,
            "reason": self.reason,
            "numerical_tie_tolerance": self.numerical_tie_tolerance,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }
        if self.search_metadata is not None:
            result["search"] = dict(self.search_metadata)
        return result


class LocalPlanner:
    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()

    def plan_analytic(self, field: GridField, position: Any, *, goal_xyz: Any,
                      compact_prediction: dict[str, Any],
                      goal_tolerance_m: float = 0.005) -> PlanResult:
        """Opt-in straight geodesic for an explicitly supplied pure compact goal.

        A fixed positive diagonal attraction metric has straight geodesics.
        Paths have at most ``horizon`` segments, with no forced minimum length;
        the last segment can end at the goal or the local validity boundary.
        The caller supplies the goal in world coordinates, rather than choosing
        a minimum sampled grid vertex. Compact repulsors, predicted boxes, or
        dense masks retain the existing lattice/exhaustive planner instead.

        Every segment is checked against the field and the convex workspace.
        These are checks of predicted validity, not robot collision guarantees.
        A planned goal endpoint is not a claim that the measured robot arrived.
        """
        from .compact_field import validate_compact_field

        validate_compact_field(compact_prediction)
        start = np.asarray(vector3(position, "position"))
        goal = np.asarray(vector3(goal_xyz, "goal_xyz"))
        if (isinstance(goal_tolerance_m, bool) or not isinstance(goal_tolerance_m, Real)
                or not np.isfinite(goal_tolerance_m) or goal_tolerance_m <= 0):
            raise ValueError("goal_tolerance_m must be finite and positive")
        goal_tolerance_m = float(goal_tolerance_m)
        distance = float(np.linalg.norm(goal - start))
        if not np.isfinite(distance):
            raise ValueError("distance between goal_xyz and position must be finite")
        search = {"method": "analytic_goal_line", "candidate_semantics": "explicit_goal_path",
                  "goal_xyz": goal.tolist(), "goal_distance_m": distance,
                  "goal_tolerance_m": goal_tolerance_m,
                  "goal_reached": False, "endpoint_is_goal": False,
                  "boundary_clipped": False, "direction_invalid": False,
                  "path_termination": None, "segment_count": 0,
                  "max_path_length_m": self.config.horizon * self.config.step_m}
        fallback = None
        if compact_prediction["repulsors"]:
            fallback = "compact_repulsors_present"
        elif compact_prediction["blocked_boxes"] or compact_prediction["unknown_boxes"]:
            # Even boxes outside this grid disable the pure-attraction shortcut.
            fallback = "compact_boxes_present"
        elif field.prediction.blocked_indices or field.prediction.unknown_indices:
            fallback = "dense_masks_present"
        if fallback is not None:
            result = self.plan(field, start)
            metadata = dict(result.search_metadata or {})
            metadata.update(analytic_fallback_reason=fallback,
                            requested_method="analytic_goal_line",
                            goal_xyz=goal.tolist(), goal_distance_m=distance,
                            goal_tolerance_m=goal_tolerance_m)
            return replace(result, search_metadata=metadata)

        initial = field.query(start)
        if field.prediction.motion_mode == "hold":
            return PlanResult((0., 0., 0.), "hold", 0, 0, start_potential=initial,
                              reason="explicit_motion_hold", search_metadata=search)
        minimum = np.maximum(field.lower, np.asarray(self.config.workspace_min))
        maximum = np.minimum(field.upper, np.asarray(self.config.workspace_max))
        if initial is None or np.any(start < minimum) or np.any(start > maximum):
            search.update(direction_invalid=True, path_termination="invalid_start")
            return PlanResult((0., 0., 0.), "no_valid_candidates", 0, 0,
                              start_potential=initial, reason="invalid_start", search_metadata=search)
        if distance <= goal_tolerance_m:
            search.update(goal_reached=True, path_termination="goal_reached")
            return PlanResult((0., 0., 0.), "goal_reached", 0, 0, start_potential=initial,
                              predicted_improvement=0., reason="measured_goal_within_tolerance",
                              search_metadata=search)

        direction = (goal - start) / distance
        boundary_distance = float("inf")
        for axis in range(3):
            if direction[axis] > 0:
                boundary_distance = min(boundary_distance, (maximum[axis] - start[axis]) / direction[axis])
            elif direction[axis] < 0:
                boundary_distance = min(boundary_distance, (minimum[axis] - start[axis]) / direction[axis])
        path_length = min(distance, self.config.horizon * self.config.step_m, boundary_distance)
        boundary_clipped = (boundary_distance <= path_length + 1e-12
                            and distance > path_length + 1e-12)
        search["boundary_clipped"] = bool(boundary_clipped)
        if path_length <= 1e-12:
            search.update(direction_invalid=True, path_termination="outward_at_boundary")
            return PlanResult((0., 0., 0.), "direction_invalid", 0, 0, start_potential=initial,
                              reason="no_positive_valid_segment", search_metadata=search)

        # Include the shortened final segment without a duplicate endpoint when
        # path_length is (to rounding precision) an exact multiple of step_m.
        arcs = [index * self.config.step_m for index in range(1, self.config.horizon + 1)
                if index * self.config.step_m < path_length - 1e-12]
        arcs.append(path_length)
        points = [start] + [np.clip(start + arc * direction, minimum, maximum) for arc in arcs]
        positions = tuple(tuple(float(value) for value in point) for point in points)
        valid = all(field.segment_valid(left, right) for left, right in zip(points[:-1], points[1:]))
        terminal = field.query(points[-1]) if valid else None
        immediate = field.query(points[1]) if valid else None
        candidate = CandidateRecord(("analytic",) * len(arcs), positions, float(path_length),
                                    valid and terminal is not None and immediate is not None,
                                    terminal, immediate, None if valid else "invalid_field_segment")
        search.update(segment_count=len(arcs), endpoint_xyz=list(positions[-1]),
                      endpoint_is_goal=bool(np.linalg.norm(points[-1] - goal) <= 1e-12),
                      path_termination=("local_boundary" if boundary_clipped else
                                        "goal" if distance <= path_length + 1e-12 else "horizon"))
        if not candidate.valid:
            search.update(direction_invalid=True, path_termination="invalid_field_segment")
            return PlanResult((0., 0., 0.), "direction_invalid", 1, 0, (candidate,),
                              start_potential=initial, reason="invalid_field_segment", search_metadata=search)
        improvement = float(initial - terminal)
        if not np.isfinite(improvement):
            improvement = None
        return PlanResult(tuple(float(value) for value in points[1] - start), "move", 1, 1,
                          (candidate,), 0, initial, improvement,
                          "local_boundary_clipped" if boundary_clipped else None,
                          search_metadata=search)

    def plan(self, field: GridField, position: Any) -> PlanResult:
        if self.config.horizon > 6:
            return self._plan_lattice(field, position)
        start = np.asarray(vector3(position, "position"))
        initial = field.query(start)
        if field.prediction.motion_mode == "hold":
            return PlanResult((0.0, 0.0, 0.0), "hold", 0, 0, start_potential=initial, reason="explicit_motion_hold")
        minimum = np.asarray(self.config.workspace_min)
        maximum = np.asarray(self.config.workspace_max)
        if initial is None or np.any(start < minimum) or np.any(start > maximum):
            return PlanResult((0.0, 0.0, 0.0), "no_valid_candidates", 0, 0, start_potential=initial, reason="invalid_start")
        records: list[CandidateRecord] = []
        choices: list[tuple[float, float, tuple[int, ...], int]] = []
        steps = np.asarray(_AXES, dtype=float) * self.config.step_m
        for sequence in product(range(6), repeat=self.config.horizon):
            points = start + np.cumsum(steps[list(sequence)], axis=0)
            all_points = (tuple(float(x) for x in start),) + tuple(tuple(float(x) for x in point) for point in points)
            invalid_reason = None
            if any((left ^ 1) == right for left, right in zip(sequence[:-1], sequence[1:])):
                invalid_reason = "immediate_reversal"
            elif np.any(points < minimum) or np.any(points > maximum):
                invalid_reason = "outside_workspace"
            else:
                for left, right in zip(all_points[:-1], all_points[1:]):
                    if not field.segment_valid(left, right):
                        invalid_reason = "invalid_field_segment"
                        break
            terminal = field.query(points[-1]) if invalid_reason is None else None
            immediate = field.query(points[0]) if invalid_reason is None else None
            valid = invalid_reason is None and terminal is not None and immediate is not None
            index = len(records)
            records.append(CandidateRecord(
                tuple(_LABELS[i] for i in sequence), all_points,
                self.config.step_m * self.config.horizon, valid,
                terminal, immediate, invalid_reason,
            ))
            if valid:
                # Terminal rank, immediate progress, fixed primitive ordering.
                choices.append((terminal, immediate, sequence, index))
        if not choices:
            return PlanResult(
                (0.0, 0.0, 0.0), "no_valid_candidates", len(records), 0,
                tuple(records), start_potential=initial, reason="all_candidates_rejected",
            )
        # Interpolation sums along equivalent paths can differ by a few ulps.
        # Use the magnitude of VALID samples, without a unit-dependent floor:
        # this preserves behavior under positive rescaling to numerical
        # precision and does not let arbitrary masked placeholders affect ties.
        magnitude = max(abs(field.prediction.values[i]) for i in field.valid_indices)
        numerical_tolerance = (64 * np.finfo(float).eps) * magnitude
        best_terminal = min(choice[0] for choice in choices)
        terminal_ties = [choice for choice in choices if choice[0] - best_terminal <= numerical_tolerance]
        best_immediate = min(choice[1] for choice in terminal_ties)
        immediate_ties = [choice for choice in terminal_ties if choice[1] - best_immediate <= numerical_tolerance]
        terminal, _immediate, sequence, selected = min(immediate_ties, key=lambda choice: choice[2])
        improvement = float(initial - terminal)
        # A flat/no-improvement field is surfaced, not silently converted to wait.
        progress_tolerance = max(self.config.plateau_tolerance, numerical_tolerance)
        status = "move" if improvement > progress_tolerance else "plateau_move"
        reason = None if status == "move" else "no_terminal_improvement"
        if not np.isfinite(improvement):
            # Finite inputs can have a difference larger than float64 permits.
            # Keep serialized records finite and make this visible in the log.
            improvement = None
            reason = "improvement_not_representable"
        return PlanResult(
            tuple(float(x) for x in steps[sequence[0]]), status, len(records), len(choices),
            tuple(records), selected, initial, improvement,
            reason, float(numerical_tolerance),
        )

    def _plan_lattice(self, field: GridField, position: Any) -> PlanResult:
        """Exact dynamic programming with at most six logged terminal-band finalists.

        At a fixed depth, prefixes with the same integer displacement, first
        direction, and last direction have identical possible continuations.
        The endpoint determines future segment validity and terminal potential;
        the last direction determines the immediate-reversal restriction; the
        first direction determines the immediate-potential tie break. Retaining
        the lexicographically first prefix therefore preserves the exact winner.
        Counts aggregate all represented valid sequences, including merged ones.

        Canonical positions are start + integer displacement * step_m. This
        avoids path-order-dependent rounding of mathematically identical nodes.
        Masks still describe predicted field validity, not robot collision proof.
        """
        start = np.asarray(vector3(position, "position"))
        initial = field.query(start)
        horizon = self.config.horizon
        search = {
            "method": "exact_lattice_dp",
            "candidate_semantics": "terminal_band_finalists",
            "nominal_candidate_count": 6 ** horizon,
            "represented_valid_sequence_count": 0,
            "layer_state_counts": [],
            "expanded_state_count": 0,
            "transition_count": 0,
            "cached_edge_count": 0,
            "terminal_minimum": None,
        }
        if field.prediction.motion_mode == "hold":
            return PlanResult((0.0, 0.0, 0.0), "hold", 0, 0,
                              start_potential=initial, reason="explicit_motion_hold", search_metadata=search)
        minimum = np.asarray(self.config.workspace_min)
        maximum = np.asarray(self.config.workspace_max)
        if initial is None or np.any(start < minimum) or np.any(start > maximum):
            return PlanResult((0.0, 0.0, 0.0), "no_valid_candidates", 0, 0,
                              start_potential=initial, reason="invalid_start", search_metadata=search)

        origin = (0, 0, 0)
        positions = {origin: start}
        potentials = {origin: initial}
        edges: dict[tuple[tuple[int, int, int], int], bool] = {}
        fully_valid_grid = not (field.prediction.blocked_indices or field.prediction.unknown_indices)

        def point(offset):
            if offset not in positions:
                positions[offset] = start + np.asarray(offset) * self.config.step_m
            return positions[offset]

        def potential(offset):
            if offset not in potentials:
                xyz = point(offset)
                potentials[offset] = (None if np.any(xyz < minimum) or np.any(xyz > maximum)
                                      else field.query(xyz))
            return potentials[offset]

        # State -> (lexicographically first primitive sequence, represented count).
        states = {(0, 0, 0, -1, -1): ((), 1)}
        search["layer_state_counts"] = [1]
        for depth in range(horizon):
            following = {}
            search["expanded_state_count"] += len(states)
            for (x, y, z, first, last), (sequence, count) in states.items():
                offset = (x, y, z)
                for direction, axis in enumerate(_AXES):
                    if last >= 0 and (last ^ 1) == direction:
                        continue
                    search["transition_count"] += 1
                    target = (x + axis[0], y + axis[1], z + axis[2])
                    edge = (offset, direction)
                    if edge not in edges:
                        # Every current endpoint was validated before its state
                        # entered this layer. With no masked vertices, the grid
                        # cube and workspace are convex: a valid target endpoint
                        # also establishes segment validity. Masked fields still
                        # require the complete interpolation-stencil traversal.
                        edges[edge] = (potential(target) is not None
                                       and (fully_valid_grid
                                            or field.segment_valid(point(offset), point(target))))
                    if not edges[edge]:
                        continue
                    key = (*target, direction if first < 0 else first, direction)
                    extended = sequence + (direction,)
                    previous = following.get(key)
                    if previous is None:
                        following[key] = (extended, count)
                    else:
                        following[key] = (min(previous[0], extended), previous[1] + count)
            states = following
            search["layer_state_counts"].append(len(states))
            if not states:
                search["layer_state_counts"].extend([0] * (horizon - depth - 1))
                break
        search["cached_edge_count"] = len(edges)
        search["represented_valid_sequence_count"] = sum(count for _sequence, count in states.values())
        if not states:
            return PlanResult((0.0, 0.0, 0.0), "no_valid_candidates", 0, 0,
                              start_potential=initial, reason="all_candidates_rejected", search_metadata=search)

        magnitude = max(abs(field.prediction.values[i]) for i in field.valid_indices)
        numerical_tolerance = float((64 * np.finfo(float).eps) * magnitude)
        best_terminal = min(potential(state[:3]) for state in states)
        search["terminal_minimum"] = best_terminal
        finalists = {}
        for state, (sequence, _count) in states.items():
            terminal = potential(state[:3])
            if terminal - best_terminal > numerical_tolerance:
                continue
            first = state[3]
            previous = finalists.get(first)
            if previous is None or sequence < previous[0]:
                finalists[first] = (sequence, terminal, potential(_AXES[first]))

        # All endpoints must be filtered against the GLOBAL minimum before
        # merging per first move. Tolerance bands are not transitive: recomputing
        # a minimum on a subset (or pruning each branch locally) can change rank.
        # Within this global terminal band, each first move has one immediate
        # score, so its lexicographically first full path is sufficient.
        choices = [finalists[first] for first in sorted(finalists)]
        best_immediate = min(choice[2] for choice in choices)
        selected = min((index for index, choice in enumerate(choices)
                        if choice[2] - best_immediate <= numerical_tolerance),
                       key=lambda index: choices[index][0])
        records = []
        for sequence, terminal, immediate in choices:
            offset = origin
            points = [tuple(float(value) for value in start)]
            for direction in sequence:
                axis = _AXES[direction]
                offset = tuple(offset[i] + axis[i] for i in range(3))
                points.append(tuple(float(value) for value in point(offset)))
            records.append(CandidateRecord(tuple(_LABELS[i] for i in sequence), tuple(points),
                                           self.config.step_m * horizon, True, terminal, immediate))
        sequence, terminal, _immediate = choices[selected]
        improvement = float(initial - terminal)
        progress_tolerance = max(self.config.plateau_tolerance, numerical_tolerance)
        status = "move" if improvement > progress_tolerance else "plateau_move"
        reason = None if status == "move" else "no_terminal_improvement"
        if not np.isfinite(improvement):
            improvement = None
            reason = "improvement_not_representable"
        return PlanResult(tuple(float(value * self.config.step_m) for value in _AXES[sequence[0]]),
                          status, len(records), len(records), tuple(records), selected, initial,
                          improvement, reason, numerical_tolerance, search)
