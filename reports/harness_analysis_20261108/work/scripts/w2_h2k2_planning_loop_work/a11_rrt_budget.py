"""A11: does the keypose-2 RRT fallback find a detour within its 12 s budget?

Transit keypose whose straight joint path crosses the (strong, cross-view
supported at this frame) T04 can at fingertip height: start grip (-0.02,-0.12,0.53),
goal (-0.02,-0.31,0.53), top-down. Run-001 frame 14 scene (robot at home in the
images; self-mask with that pose). Also reports how many configuration checks fit.
"""
import json
from time import perf_counter

import numpy as np

from k2common import WORK, scene_at, kin
from gpt_field.contact_skills import matrix_to_quaternion_xyzw
from gpt_field.foundation_harness.feedback import grip_pose
from gpt_field.foundation_harness.motion import CandidatePlanner, PlanningFailure
from gpt_field.robodojo_direct_sim import rotation_vector

scene, state, q, record, err = scene_at(13)
R0 = grip_pose(state)[:3, :3]
xyzw = matrix_to_quaternion_xyzw(R0); wxyz = [float(xyzw[3]), *map(float, xyzw[:3])]
pl = CandidatePlanner(kin); pl._prepare(scene, state, q)
import sys
Z = float(sys.argv[1]) if len(sys.argv) > 1 else 0.53
start_xyz, goal_xyz = np.array([-0.02, -0.12, Z]), np.array([-0.02, -0.31, Z])
qs = np.asarray(pl.bound.solve(start_xyz, R0, q, max_iterations=80).joints)
pose, _ = pl.bound.forward(qs)
st = np.r_[pose[:3, 3], rotation_vector(pose[:3, :3]), .04, -.04]
calls = {'n': 0}
orig = pl._configuration
def counted(*a, **k):
    calls['n'] += 1
    return orig(*a, **k)
pl._configuration = counted
op = {'kind': 'move', 'object': None, 'target': None, 'grasp_region': None, 'contact_mode': 'free_object',
      'keep_upright': False, 'placement_relation': None, 'goal_pose': None, 'displacement': None,
      'approach_direction': None, '_keypose_index': 0,
      'keyposes': [{'name': 'move', 'pose': {'position': goal_xyz.tolist(), 'quaternion_wxyz': wxyz},
                    'gripper': 'keep', 'profile': 'transit', 'observe_after': False}]}
t0 = perf_counter()
try:
    plan = pl.plan(op, scene, st, qs)
    out = {'planned': True, 'path_points': len(plan.joint_paths[0])}
except PlanningFailure as exc:
    ev = exc.evidence.get('evidence', {})
    out = {'planned': False, 'reason': exc.reason,
           'direct_check': {k: (ev.get('direct_check') or {}).get(k) for k in ('reason', 'part', 'point_world_m', 'obstacle_evidence')},
           'last_blocked_edge': {k: (ev.get('last_blocked_edge') or {}).get(k) for k in ('segment_location', 'iteration', 'reason', 'part')}}
out.update(elapsed_s=round(perf_counter() - t0, 2), configuration_checks=calls['n'])
print(json.dumps(out, indent=1))
(WORK / f'data/a11_rrt_budget_z{Z:.2f}.json').write_text(json.dumps(out, indent=1))
