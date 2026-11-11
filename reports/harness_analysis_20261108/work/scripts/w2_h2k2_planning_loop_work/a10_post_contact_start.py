"""A10: can the robot leave a grasp configuration under keypose-2 rules?

At H3's own white-mug grasp pose (grip (-0.085,-0.10,0.51), top-down, H3 requests
000002-000003), evaluate motion.py _configuration() for the START of the next
keypose under the exemptions keypose-2 assigns by keypose name (motion.py:518-521):
  grasp/close/lift : contacts = object ids, contact parts finger/pad
  retreat/transport: no contacts, no contact parts
with fingers open (0.04) and closed on the handle (0.012), payload absent (as when
planning_surface is empty, motion.py:545-548) or present. Scenes: run-001 frames
14 (white mug planning surface empty) and 20 (coherent). Offline only.
"""
import json

import numpy as np

from k2common import WORK, records, restore, kin, events  # noqa: F401
from gpt_field.contact_skills import matrix_to_quaternion_xyzw
from gpt_field.foundation_harness.feedback import grip_pose
from gpt_field.foundation_harness.motion import CandidatePlanner

rows = []
for arrays in ('frame_000014_revision_000030.npz', 'frame_000020_revision_000044.npz'):
    rec = next(r for r in records if r['arrays'] == arrays)
    scene, err = restore(rec)  # noqa: F821
    step = rec['snapshot']['evidence']['observation_index']
    fb = next(e for e in events if e['event'] == 'feedback' and e['step'] == max(step, 1))
    state = np.asarray(fb['measured_state']); q = np.asarray(fb['joint_position'])
    R0 = grip_pose(state)[:3, :3]
    xyzw = matrix_to_quaternion_xyzw(R0); wxyz = [float(xyzw[3]), *map(float, xyzw[:3])]
    pl = CandidatePlanner(kin); pl._prepare(scene, state, q); pl._deadline = float('inf')
    grip = np.array([-0.085, -0.10, 0.51])
    res = pl.bound.solve(grip, R0, q, max_iterations=80)
    qg = np.asarray(res.joints)
    op = {'kind': 'pick_place', 'object': 'white_mug', 'target': 'seed_1', 'grasp_region': None,
          'contact_mode': 'free_object', 'keep_upright': False, 'placement_relation': 'on',
          'goal_pose': None, 'displacement': None, 'approach_direction': None, '_keypose_index': 0,
          'keyposes': [{'name': 'close', 'pose': {'position': grip.tolist(), 'quaternion_wxyz': wxyz},
                        'gripper': 'close', 'profile': 'contact', 'observe_after': False}]}
    payload = pl._astra_keypose(op, scene, state).metadata['payload_after']
    for fingers_label, fingers in (('open', np.array([0.04, -0.04])), ('closed_on_handle', np.array([0.012, -0.012]))):
        for name, parts, contacts in (('lift', ('finger', 'pad'), ('white_mug',)), ('retreat', (), ())):
            for held_label, held in (('no_payload', None), ('payload', payload)):
                pl._contact_parts = parts
                r = pl._configuration(qg, scene, fingers, held, contacts, free_check=False)
                rows.append({'frame': rec['snapshot']['frame_id'], 'ik_ok': bool(res.ok), 'fingers': fingers_label,
                             'next_keypose': name, 'held': held_label,
                             'payload_points': len(payload['points_in_gripper']),
                             **{k: r.get(k) for k in ('safe', 'reason', 'part', 'point_object_id', 'obstacle_evidence')},
                             'point_world_m': None if r.get('point_world_m') is None else np.round(r['point_world_m'], 3).tolist(),
                             'uncertain_samples': r.get('uncertain_obstacle_samples')})
for r in rows:
    print(json.dumps(r))
(WORK / 'data/a10_post_contact_start.json').write_text(json.dumps(rows, indent=1))
