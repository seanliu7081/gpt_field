"""Measured execution accounting; no action gates or task-success inference."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..robodojo_direct_sim import quaternion_matrix, rotation_matrix


class TimingSeries:
    """Append cheaply in the loop; calculate quantiles only when reporting."""

    def __init__(self):
        self.samples = defaultdict(list)

    def add(self, name, seconds):
        value = float(seconds)
        if np.isfinite(value) and value >= 0.:
            self.samples[name].append(value)

    def total(self, name):
        return float(sum(self.samples.get(name, ())))

    def summary(self):
        return {name: {"count": len(values), "total_s": float(sum(values)),
            "mean_s": float(np.mean(values)), "p50_s": float(np.median(values)),
            "p95_s": float(np.quantile(values, .95)), "max_s": float(max(values))}
            for name, values in self.samples.items() if values}


def _angle(a, b):
    return float(np.arccos(np.clip((np.trace(a@b.T)-1.)*.5, -1., 1.)))


def measured_progress(before, after, diagnostics):
    """Keep measured arm motion separate from advancement toward a local goal."""
    a, b = np.asarray(before.state, float), np.asarray(after.state, float)
    ra, rb = rotation_matrix(a[3:6]), rotation_matrix(b[3:6])
    evidence = diagnostics.get("motion_evidence") or {}
    result = {"source": "measured_robot_state", "task_success_inferred": False,
        "primitive": diagnostics.get("primitive", "unspecified"),
        "translation_m": float(np.linalg.norm(b[:3]-a[:3])),
        "rotation_rad": _angle(rb, ra),
        "finger_gap_change_m": float((b[6]-b[7])-(a[6]-a[7])),
        "command_variant": evidence.get("variant"), "command_scale": evidence.get("scale"),
        "goal_distance_before_m": None, "goal_distance_after_m": None,
        "goal_distance_reduction_m": None, "goal_rotation_reduction_rad": None}
    xyz = diagnostics.get("goal_xyz_m")
    if xyz is not None:
        goal = np.asarray(xyz, float)
        if goal.shape == (3,) and np.isfinite(goal).all():
            first, last = float(np.linalg.norm(goal-a[:3])), float(np.linalg.norm(goal-b[:3]))
            result.update(goal_distance_before_m=first, goal_distance_after_m=last,
                          goal_distance_reduction_m=first-last)
    quaternion = diagnostics.get("goal_quaternion_wxyz")
    if quaternion is not None:
        q = np.asarray(quaternion, float)
        if q.shape == (4,) and np.isfinite(q).all() and np.linalg.norm(q) > 1e-9:
            goal_rotation = quaternion_matrix(q/np.linalg.norm(q))
            result["goal_rotation_reduction_rad"] = _angle(goal_rotation, ra)-_angle(goal_rotation, rb)
    return result


class ProgressTotals:
    def __init__(self):
        self.by_primitive = {}

    def add(self, item):
        row = self.by_primitive.setdefault(item["primitive"], {
            "steps": 0, "translation_m": 0., "rotation_rad": 0.,
            "absolute_finger_gap_change_m": 0., "near_zero_arm_motion_steps": 0,
            "goal_position_samples": 0, "goal_distance_reduction_m": 0.,
            "goal_position_approaching_steps": 0, "goal_position_receding_steps": 0})
        row["steps"] += 1
        row["translation_m"] += item["translation_m"]
        row["rotation_rad"] += item["rotation_rad"]
        row["absolute_finger_gap_change_m"] += abs(item["finger_gap_change_m"])
        # Reporting resolution only; a stationary arm may be intentionally
        # holding while the gripper closes or while the model is consulted.
        row["near_zero_arm_motion_steps"] += int(item["translation_m"] < .0001 and item["rotation_rad"] < .001)
        reduction = item["goal_distance_reduction_m"]
        if reduction is not None:
            row["goal_position_samples"] += 1
            row["goal_distance_reduction_m"] += reduction
            row["goal_position_approaching_steps"] += int(reduction > .0001)
            row["goal_position_receding_steps"] += int(reduction < -.0001)

    def summary(self):
        return {"by_primitive": self.by_primitive,
                "goal_reductions_are_per_step_local_targets_not_task_progress": True,
                "near_zero_arm_motion_is_not_a_failure_classification": True}
