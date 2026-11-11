"""Task identities for the Astranomy port, without task-specific motion recipes.

Retain the original distance-task contracts and reuse the existing LIBERO
registry for additional suites. Clients expose only the official task sentence.
"""
from __future__ import annotations

from .agent_speedup_v2_tasks import (
    V2_TASK_ALIASES, V2_TASK_KEYS, V2TaskSpec, resolve_v2_task,
)
from .distance_tasks import DISTANCE_TASK_KEYS, DistanceTaskSpec, resolve_distance_task


ASTRANOMY_TASK_KEYS = tuple(dict.fromkeys(
    (*DISTANCE_TASK_KEYS, *V2_TASK_KEYS, *V2_TASK_ALIASES)))


def resolve_astranomy_task(task_key: str = "bowl_plate") -> DistanceTaskSpec | V2TaskSpec:
    """Resolve benchmark identity while preserving the original task contracts."""
    if not isinstance(task_key, str) or task_key not in ASTRANOMY_TASK_KEYS:
        raise ValueError(f"Unknown Astranomy task; choose one of {', '.join(ASTRANOMY_TASK_KEYS)}")
    key = V2_TASK_ALIASES.get(task_key, task_key)
    if key in DISTANCE_TASK_KEYS:
        return resolve_distance_task(key)
    return resolve_v2_task(key)
