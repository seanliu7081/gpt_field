"""Robot-only short-horizon checks, independent of predicted scene geometry.

Uses known joint limits and collision meshes, never SAM/DA3 points or map ages.
Sampled OSC response predictions are not continuous collision certification.
"""
from __future__ import annotations

from collections import OrderedDict
from time import perf_counter

import numpy as np

from ..action import TRANSLATION_SCALE_M, ROTATION_SCALE_RAD
from ..foundation_harness.robot_geometry import RobotCollisionModel
from ..robodojo_direct_sim import PandaKinematics


class RobotGuard:
    def __init__(self, kinematics=None, *, robot_model=None):
        self.kinematics = kinematics or PandaKinematics.from_xml()
        self.robot = robot_model or RobotCollisionModel(self.kinematics)
        self._cache = OrderedDict()
        self._last_velocity = np.zeros(6)
        self._pending = None
        self._stats = {"calls": 0, "rejected": 0, "wall_s": 0.,
                       "shape_checks": 0, "cache_hits": 0, "prediction_error_rad_max": 0.}

    def _check(self, bound, joints, fingers):
        if np.any(joints <= self.kinematics.low) or np.any(joints >= self.kinematics.high):
            return {"reason": "joint_limit", "joint_position_rad": joints.tolist()}
        # Self intersections are invariant to a common world/base transform.
        # Exact state keys avoid reusing a result for a different configuration.
        key = (np.asarray(joints).tobytes(), np.asarray(fingers).tobytes())
        if key in self._cache:
            self._stats["cache_hits"] += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._stats["shape_checks"] += 1
        hit = self.robot.self_collision(self.robot.shapes(bound, joints, fingers))
        self._cache[key] = hit
        while len(self._cache) > 24:
            self._cache.popitem(last=False)
        return hit

    def filter(self, action, view, segment=None):
        started = perf_counter()
        self._stats["calls"] += 1
        self._pending = None
        evidence = {"scope": "robot_joint_limits_and_sampled_self_collision",
                    "external_geometry_checked": False, "continuous_collision_certified": False}
        try:
            action = np.asarray(action, float)
            q = np.asarray(view.observation.joint_position, float)
            state = np.asarray(view.state, float)
            if (action.shape != (7,) or not np.isfinite(action).all() or
                    np.max(np.abs(action)) > 1. + 1e-12):
                raise ValueError("invalid_normalized_action")
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError("joint_feedback_unavailable")
            bound = self.kinematics.bind(state, q)
            hit = self._check(bound, q, state[6:8])
            if hit:
                return self._reject(hit, evidence)
            qd_raw = getattr(view.observation, "joint_velocity", None)
            qd = np.zeros(7) if qd_raw is None else np.asarray(qd_raw, float)
            if qd.shape != (7,) or not np.isfinite(qd).all():
                raise ValueError("joint_velocity_unavailable")
            _, jacobian = bound.forward(q)
            velocity = jacobian @ qd if qd_raw is not None else self._last_velocity.copy()
            # The public OSC response approximation already used by V2, one
            # control interval only; no environmental search or action variants.
            velocity[:3] = .711 * velocity[:3] + .0732 * action[:3] * TRANSLATION_SCALE_M / .05
            velocity[3:] = .90 * velocity[3:] + .027 * action[3:6] * ROTATION_SCALE_RAD / .05
            inverse = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + .008**2 * np.eye(6), np.eye(6))
            predicted = q + .05 * (inverse @ velocity + (np.eye(7) - inverse @ jacobian) @ qd)
            destination = np.zeros(2) if action[6] > 0 else np.array([.04, -.04])
            fingers = state[6:8] + np.clip(destination - state[6:8], -.003, .003)
            for fraction in (.5, 1.):
                hit = self._check(bound, q + fraction * (predicted-q),
                                  state[6:8] + fraction * (fingers-state[6:8]))
                if hit:
                    return self._reject(hit, {**evidence, "sample_fraction": fraction})
            self._pending = (tuple(action), predicted.copy(), velocity.copy())
            return {"status": "execute", "reason": "robot_checks_passed", "action": tuple(action),
                    "evidence": {**evidence, "prediction_samples": 2}}
        except (ValueError, TypeError, np.linalg.LinAlgError) as error:
            return self._reject({"reason": str(error)}, evidence)
        finally:
            self._stats["wall_s"] += perf_counter()-started

    def _reject(self, detail, evidence):
        self._stats["rejected"] += 1
        return {"status": "blocked", "reason": detail["reason"],
                "evidence": {**evidence, **detail}}

    def observe_transition(self, before, actual, after):
        if self._pending is None:
            return
        proposed, predicted, velocity = self._pending
        self._pending = None
        if tuple(actual) != proposed:
            raise ValueError("robot guard feedback does not match dispatched action")
        q = np.asarray(after.observation.joint_position, float)
        self._stats["prediction_error_rad_max"] = max(
            self._stats["prediction_error_rad_max"], float(np.max(np.abs(q-predicted))))
        self._last_velocity = velocity

    def stats(self):
        return dict(self._stats)
