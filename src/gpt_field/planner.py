"""Equal-travel-length receding-horizon search over nonzero axis motions.

Terminal potential is the objective; no physical-distance interpretation is
required. Waiting is available only through an explicit model hold command or
the absence of a valid candidate. Masks are predictions, not collision proof.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, Integral) or not 1 <= self.horizon <= 6:
            raise ValueError("horizon must be an integer between 1 and 6")
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

    def to_dict(self) -> dict[str, Any]:
        return {
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


class LocalPlanner:
    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()

    def plan(self, field: GridField, position: Any) -> PlanResult:
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
