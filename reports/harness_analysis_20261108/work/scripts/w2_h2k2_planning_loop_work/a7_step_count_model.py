"""A7: control-step estimate for an H3-like T04 keypose plan under keypose-2 execution.

Keypose positions = the H3 live run's own successful Astra proposals for T04
(requests 000001-000011 of runs/foundation_vision_t04_can_obstacle_20261105_001),
top-down orientation. For each keypose: IK seeded from the previous solution
(motion.py:468), direct joint path densified at 0.025 rad (motion.py:407-412),
FK waypoint poses (executor.py:49). Execution model of JointPathStage
(executor.py:67-113): command = waypoint - p clipped to 20 mm (transit) / 3 mm
(contact); waypoint advances at <=4 mm; arrival = final pose within 3 mm on 2
consecutive steps; gripper stages need >=12 (close) / >=8 (open) settled actions.
OSC response: first-order, realized fraction r of the commanded delta per step
(H1 measured 22-30 %: w1/h1_speed_attempts.md section 3). No collision checks.
"""
import json

import numpy as np

from k2common import WORK, scene_at, kin
from gpt_field.foundation_harness.feedback import grip_pose
from gpt_field.foundation_harness.motion import CandidatePlanner

scene, state, q0, record, err = scene_at(13)
bound = kin.bind(state, q0)
R0 = grip_pose(state)[:3, :3]
P = CandidatePlanner.__new__(CandidatePlanner)  # only for _densify
P.config = None

# (name, xyz, profile, gripper)
plan = [
    ('pregrasp', (-0.10, -0.135, 0.65), 'transit', 'keep'),
    ('pregrasp', (-0.085, -0.10, 0.61), 'transit', 'keep'),
    ('pregrasp', (-0.085, -0.10, 0.565), 'transit', 'keep'),
    ('grasp', (-0.085, -0.10, 0.510), 'contact', 'keep'),
    ('close', (-0.085, -0.10, 0.510), 'contact', 'close'),
    ('lift', (-0.085, -0.10, 0.540), 'contact', 'keep'),
    ('transport', (-0.088, -0.10, 0.65), 'transit', 'keep'),
    ('transport', (-0.025, -0.275, 0.65), 'transit', 'keep'),
    ('preplace', (0.01, -0.265, 0.57), 'transit', 'keep'),
    ('place', (0.01, -0.265, 0.517), 'contact', 'keep'),
    ('release', (0.01, -0.265, 0.517), 'contact', 'open'),
    ('retreat', (0.01, -0.265, 0.592), 'contact', 'keep'),
    ('pregrasp', (0.01, -0.26, 0.65), 'transit', 'keep'),
    ('pregrasp', (-0.005, 0.025, 0.565), 'transit', 'keep'),
    ('grasp', (-0.030, 0.013, 0.498), 'contact', 'keep'),
    ('close', (-0.030, 0.013, 0.498), 'contact', 'close'),
    ('lift', (-0.030, 0.013, 0.528), 'contact', 'keep'),
    ('transport', (-0.031, 0.013, 0.63), 'transit', 'keep'),
    ('preplace', (-0.01, 0.245, 0.58), 'transit', 'keep'),
    ('place', (-0.01, 0.245, 0.505), 'contact', 'keep'),
    ('release', (-0.01, 0.245, 0.505), 'contact', 'open'),
    ('retreat', (-0.01, 0.245, 0.58), 'contact', 'keep'),
]


def densify(path):
    result = [np.asarray(path[0]).tolist()]
    for first, second in zip(path[:-1], path[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(np.asarray(second) - first)) / .025)))
        result.extend(np.linspace(first, second, count + 1)[1:].tolist())
    return result


def simulate(r):
    q = np.asarray(q0, float)
    p = state[:3].copy()
    rows, total = [], 0
    for name, xyz, profile, gripper in plan:
        res = bound.solve(np.asarray(xyz), R0, q, max_iterations=80)
        goal = np.asarray(res.joints)
        path = densify([q.tolist(), goal.tolist()])
        poses = [bound.forward(np.asarray(j))[0][:3, 3] for j in path]
        limit = 0.02 if profile == 'transit' else 0.003
        idx, steps, arrival, settle = 0, 0, 0, 0
        while steps < 2000:
            while idx < len(poses) - 1 and np.linalg.norm(poses[idx] - p) <= 0.004:
                idx += 1
            delta = poses[idx] - p
            n = np.linalg.norm(delta)
            if n > limit:
                delta *= limit / n
            p = p + r * delta
            steps += 1
            at = idx == len(poses) - 1 and np.linalg.norm(poses[-1] - p) <= 0.003
            if gripper == 'keep':
                arrival = arrival + 1 if at else 0
                if arrival >= 2:
                    break
            else:
                settle = settle + 1 if at else 0
                if settle >= (12 if gripper == 'close' else 8):
                    break
        total += steps
        rows.append({'keypose': name, 'profile': profile, 'ik_ok': bool(res.ok), 'waypoints': len(poses),
                     'path_len_m': round(float(np.linalg.norm(np.asarray(xyz) - (rows and rows[-1]['_end'] or state[:3]))), 3) if False else None,
                     'steps': steps})
        q = goal
    return total, rows


out = {}
for r in (0.22, 0.27, 0.30):
    total, rows = simulate(r)
    out[f'r={r}'] = {'total_steps': total, 'per_keypose': rows}
    print(r, total, [(x['keypose'], x['profile'], x['steps'], x['waypoints'], x['ik_ok']) for x in rows])
(WORK / 'data/a7_step_count_model.json').write_text(json.dumps(out, indent=1))
