"""Causal progress of a declared visual objective, separate from metric arrival.

Each interval uses one actual tracked robot feature, stationary anchor, desired
image offset and world goal. A reference change archives, never adds together,
incompatible pixel errors. No calibrated projection or task-success claim.
"""
from __future__ import annotations

import copy
import numpy as np


class VisualTaskProgressV21:
    minimum_path_m = .01
    minimum_metric_improvement_m = .008
    minimum_worsening_px = 1.5

    def __init__(self):
        self.active = None
        self.intervals = []
        self._key = None
        self._serial = 0
        self._reported = set()

    @staticmethod
    def _snapshot(monitor, prediction):
        external = (prediction.get("visual_references") or {}).get("external") or {}
        if (external.get("moving_kind") != "tool"
                or external.get("target_id") != prediction.get("target_id")
                or prediction.get("phase") not in ("approach", "align")
                or not monitor._external_reliable()):
            return None
        moving, target = monitor.tracks.get("moving"), monitor.tracks.get("target")
        if moving is None or target is None or not moving.reliable or not target.reliable:
            return None
        offset = np.asarray(external.get("goal_offset_uv"), float)
        if offset.shape != (2,) or not np.isfinite(offset).all():
            return None
        error = np.asarray(moving.position)-np.asarray(target.position)-offset*moving.scale
        noise = np.sqrt(sum(max(.08, float(t.last_residual_px)/np.sqrt(max(3, t.last_inliers)))**2
                            for t in (moving, target)))
        return {"error_px": float(np.linalg.norm(error)), "error_vector_px": error.tolist(),
                "noise_px": float(noise), "anchor_px": np.asarray(target.position).copy(),
                "identity_key": (id(moving), id(target), tuple(offset))}

    def configure(self, state8, monitor, prediction, goal_xyz, *, step):
        if goal_xyz is None:
            # A temporary Direct/gripper repair cannot erase an unfinished
            # Field objective's measured conflict merely by omitting its goal.
            return self.summary()
        snap = self._snapshot(monitor, prediction)
        goal = None if goal_xyz is None else np.asarray(goal_xyz, float)
        key = None if snap is None or goal is None else (
            prediction.get("target_id"), prediction.get("phase"),
            snap["identity_key"], tuple(np.round(goal, 9)))
        if key == self._key and self.active is not None:
            return self.summary()
        if self.active is not None:
            self.intervals.append(copy.deepcopy(self.active))
            self.intervals = self.intervals[-12:]
        self._key, self.active = key, None
        if key is not None:
            self._serial += 1
            self.active = {"interval_id": self._serial, "target_id": prediction["target_id"],
                "phase": prediction["phase"], "start_step": step, "last_step": step,
                "world_goal_xyz": goal.tolist(), "start_xyz": list(state8[:3]),
                "start_metric_error_m": float(np.linalg.norm(goal-np.asarray(state8[:3]))),
                "metric_improvement_m": 0., "measured_path_m": 0., "samples": 0,
                "initial_image_error_px": snap["error_px"], "current_image_error_px": snap["error_px"],
                "initial_error_vector_px": snap["error_vector_px"], "current_error_vector_px": snap["error_vector_px"],
                "maximum_noise_px": snap["noise_px"], "image_error_change_px": 0.,
                "straight_toward_goal_samples": 0, "last_anchor_px": snap["anchor_px"].tolist(),
                "review_required": False, "reason": "observing_declared_objective",
                "goal_offset_uv": list(key[2][2]), "task_success_certified": False}
        return self.summary()

    def observe(self, before, after, action, *, monitor, prediction, goal_xyz,
                accepted_field, straight_route, step):
        # Refreshes retain compatible intervals. An unexpected local retarget
        # starts a distinct objective, rather than carrying an old error over.
        snap = self._snapshot(monitor, prediction)
        if self.active is None or snap is None:
            return None
        goal = np.asarray(goal_xyz, float) if goal_xyz is not None else None
        key = None if goal is None else (prediction.get("target_id"), prediction.get("phase"),
            snap["identity_key"], tuple(np.round(goal, 9)))
        if key != self._key:
            self.configure(after, monitor, prediction, goal_xyz, step=step)
            return None
        current = self.active
        if np.linalg.norm(snap["anchor_px"]-current["last_anchor_px"]) > 6.:
            current["reason"] = "anchor_shifted_interval_unusable"
            self._key = None
            return None
        if current["reason"] == "anchor_shifted_interval_unusable":
            return None
        current["last_anchor_px"] = snap["anchor_px"].tolist()
        delta = np.asarray(after[:3])-np.asarray(before[:3])
        distance = float(np.linalg.norm(delta))
        command = np.asarray(action[:3], float)*.05
        if not accepted_field or distance < .0001 or np.linalg.norm(command) < 1e-8 or action[6] != 0:
            return None
        direction = goal-np.asarray(before[:3])
        aligned = np.dot(command, direction) >= .9*np.linalg.norm(command)*np.linalg.norm(direction)
        # A detour invalidates a monotonic-image expectation. Begin another
        # interval later instead of mislabeling intended obstacle avoidance.
        if not straight_route or not aligned:
            current["reason"] = "detour_no_monotonic_image_expectation"
            self._key = None
            return None
        current["last_step"] = step
        current["samples"] += 1
        current["straight_toward_goal_samples"] += 1
        current["measured_path_m"] += distance
        current["metric_improvement_m"] = current["start_metric_error_m"]-float(np.linalg.norm(goal-np.asarray(after[:3])))
        current["current_image_error_px"] = snap["error_px"]
        current["current_error_vector_px"] = snap["error_vector_px"]
        current["maximum_noise_px"] = max(current["maximum_noise_px"], snap["noise_px"])
        current["image_error_change_px"] = snap["error_px"]-current["initial_image_error_px"]
        threshold = max(self.minimum_worsening_px, 3*np.sqrt(2)*current["maximum_noise_px"])
        current["worsening_threshold_px"] = float(threshold)
        conflict = (current["samples"] >= 4 and current["measured_path_m"] >= self.minimum_path_m
            and current["metric_improvement_m"] >= self.minimum_metric_improvement_m
            and current["image_error_change_px"] > threshold)
        if conflict:
            current["review_required"] = True
            current["reason"] = "metric_goal_progress_disagrees_with_declared_image_objective"
            if current["interval_id"] not in self._reported:
                self._reported.add(current["interval_id"])
                return current["reason"]
        return None

    def summary(self):
        prior = copy.deepcopy(self.intervals[-6:])
        return {"version": "persistent-objective-progress-v21", "active": copy.deepcopy(self.active),
            "prior_intervals": prior,
            "prior_conflicting_intervals": sum(x["review_required"] for x in self.intervals),
            "pixel_errors_summed_across_references": False,
            "task_success_certified": False,
            "interpretation": "Metric arrival and declared-image-objective progress are separate. Review conflicting hypotheses, anatomical image offsets and route; a new reference cannot erase archived evidence. No calibrated physical task-distance claim."}
