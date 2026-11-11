"""Immutable task contracts for the opt-in distance-field experiments.

These identify the requested benchmark and policy-visible task text. They do
not contain object positions, joint targets, success predicates, or trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .direct_history import TASK


@dataclass(frozen=True)
class DistanceTaskSpec:
    key: str
    suite: str
    task_id: int
    prompt: str
    target_regions: tuple[str, ...]
    placement_target: str
    control_phases: tuple[str, ...]
    allow_upward_reobservation: bool


_COMMON_PHASES = ("approach", "grasp", "transport", "align", "release", "verify", "recover")
_TASKS = MappingProxyType({
    "bowl_plate": DistanceTaskSpec(
        key="bowl_plate", suite="libero_spatial", task_id=0, prompt=TASK,
        target_regions=("black_bowl", "plate"), placement_target="plate",
        control_phases=_COMMON_PHASES, allow_upward_reobservation=True,
    ),
    "drawer_bowl": DistanceTaskSpec(
        key="drawer_bowl", suite="libero_goal", task_id=3,
        prompt="open the top drawer and put the bowl inside",
        target_regions=("drawer_handle", "black_bowl", "drawer_interior"),
        placement_target="drawer_interior",
        control_phases=(*_COMMON_PHASES, "handle_grasp", "drawer_open", "handle_release"),
        allow_upward_reobservation=False,
    ),
})
DISTANCE_TASK_KEYS = tuple(_TASKS)


def resolve_distance_task(task_key: str = "bowl_plate") -> DistanceTaskSpec:
    if not isinstance(task_key, str) or task_key not in _TASKS:
        raise ValueError(f"Unknown distance task; choose one of {', '.join(DISTANCE_TASK_KEYS)}")
    return _TASKS[task_key]
