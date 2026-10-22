"""Deterministic, bounded joint-space RRT-Connect with explicit edge checks.

The callback alone supplies collision/unknown decisions. This module imports no
scene, simulator, field, reference, or historical path data. Edge checks are
finite samples, not continuous collision certificates.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import time
import numpy as np


@dataclass(frozen=True)
class PlannerSettings:
    max_iterations: int = 1200
    extension_rad: float = 0.18
    edge_max_joint_step_rad: float = 0.025
    goal_bias: float = 0.15
    shortcut_attempts: int = 40


def interpolate_edge(a, b, max_joint_step):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('finite matching endpoints required')
    if not np.isfinite(max_joint_step) or max_joint_step <= 0:
        raise ValueError('positive joint step required')
    steps = max(1, int(np.ceil(np.max(np.abs(b-a))/max_joint_step)))
    return a + np.linspace(0., 1., steps+1)[:, None] * (b-a)


class JointPlanner:
    def __init__(self, limits, check_state, settings=None):
        self.limits = np.asarray(limits, float)
        if self.limits.ndim != 2 or self.limits.shape[1] != 2 or not np.isfinite(self.limits).all():
            raise ValueError('limits must be finite Nx2')
        if np.any(self.limits[:, 1] <= self.limits[:, 0]):
            raise ValueError('invalid limits')
        self.check_state = check_state
        self.settings = settings or PlannerSettings()
        s = self.settings
        if s.max_iterations < 1 or s.extension_rad <= 0 or s.edge_max_joint_step_rad <= 0 or not 0 <= s.goal_bias <= 1 or s.shortcut_attempts < 0:
            raise ValueError('invalid planner settings')

    def plan(self, start, goal, seed):
        started = time.perf_counter()
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if start.shape != (len(self.limits),) or goal.shape != start.shape or not np.isfinite(start).all() or not np.isfinite(goal).all():
            raise ValueError('finite joint vectors required')
        rng = np.random.default_rng(seed)
        counts = {'state_checks': 0, 'edge_checks': 0, 'iterations': 0, 'tree_nodes': 0}
        cache = {}

        def state(q):
            key = np.asarray(q, np.float64).tobytes()
            if key not in cache:
                counts['state_checks'] += 1
                if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
                    result = {'valid': False, 'reason': 'joint_limit'}
                else:
                    result = self.check_state(q)
                    if isinstance(result, (bool, np.bool_)):
                        result = {'valid': bool(result), 'reason': 'clear' if result else 'collision'}
                    if not isinstance(result, dict) or not isinstance(result.get('valid'), (bool, np.bool_)):
                        raise ValueError('checker must return boolean or dict with boolean valid')
                cache[key] = {**result, 'valid': bool(result['valid'])}
            return cache[key]

        def edge(a, b):
            counts['edge_checks'] += 1
            return all(state(q)['valid'] for q in interpolate_edge(a, b, self.settings.edge_max_joint_step_rad))

        def finish(status, reason, path=None, **details):
            path = [] if path is None else np.asarray(path, float).tolist()
            return {'status': status, 'reason': reason, 'path_q': path,
                    'planning_seconds': time.perf_counter()-started,
                    'counts': counts, 'seed': int(seed), 'settings': asdict(self.settings),
                    'joint_path_length_rad': float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path)>1 else 0.,
                    **details}

        first = state(start)
        if not first['valid']:
            return finish('invalid_start', first.get('reason', 'invalid'), start_check=first)
        last = state(goal)
        if not last['valid']:
            return finish('invalid_goal', last.get('reason', 'invalid'), start_check=first, goal_check=last)
        if edge(start, goal):
            return finish('planned', 'straight_edge_valid', [start, goal], start_check=first, goal_check=last)

        # a_start tracks which physical endpoint the current first tree owns.
        a, ap, b, bp, a_start = [start], [-1], [goal], [-1], True

        def extend(nodes, parents, target):
            index = int(np.argmin(np.linalg.norm(np.asarray(nodes)-target, axis=1)))
            near = nodes[index]
            delta = target-near
            length = float(np.linalg.norm(delta))
            if length <= 1e-12:
                return 'reached', index
            q = target.copy() if length <= self.settings.extension_rad else near + delta*(self.settings.extension_rad/length)
            if not edge(near, q):
                return 'trapped', index
            nodes.append(q)
            parents.append(index)
            return ('reached' if length <= self.settings.extension_rad else 'advanced'), len(nodes)-1

        def trace(nodes, parents, index):
            out = []
            while index >= 0:
                out.append(nodes[index]); index = parents[index]
            return list(reversed(out))

        for iteration in range(self.settings.max_iterations):
            counts['iterations'] = iteration+1
            target = b[0] if rng.random() < self.settings.goal_bias else rng.uniform(self.limits[:, 0], self.limits[:, 1])
            status, ai = extend(a, ap, target)
            if status != 'trapped':
                while True:
                    other, bi = extend(b, bp, a[ai])
                    if other != 'advanced':
                        break
                if other == 'reached':
                    path = trace(a, ap, ai) + list(reversed(trace(b, bp, bi)))[1:]
                    if not a_start:
                        path.reverse()
                    for _ in range(self.settings.shortcut_attempts):
                        if len(path) <= 2: break
                        i, j = sorted(rng.choice(len(path), 2, replace=False).tolist())
                        if j > i+1 and edge(path[i], path[j]):
                            path = path[:i+1]+path[j:]
                    if not all(edge(x, y) for x, y in zip(path, path[1:])):
                        raise RuntimeError('final path failed its own collision callback')
                    counts['tree_nodes'] = len(a)+len(b)
                    return finish('planned', 'rrt_connect', path, start_check=first, goal_check=last)
            a, b, ap, bp, a_start = b, a, bp, ap, not a_start
        counts['tree_nodes'] = len(a)+len(b)
        return finish('no_path_within_budget', 'search_budget_exhausted_not_infeasibility_proof', start_check=first, goal_check=last)
