"""A12: does keypose-2 veto a descent that drives the fingers INTO the table?

Contact keypose from grip z 0.50 down to grip z 0.425 (table top 0.437 m by layout
truth; grip site ~1 cm above the fingertips => fingertips ~2 cm into the table)
at free table locations of the run-001 frame-14 scene. A veto requires a strong
table point inside a finger/hand hull (+6 mm). Offline only.
"""
import json

import numpy as np

from k2common import WORK, scene_at, kin, truth_boxes
from gpt_field.contact_skills import matrix_to_quaternion_xyzw
from gpt_field.foundation_harness.feedback import grip_pose
from gpt_field.foundation_harness.motion import CandidatePlanner, PlanningFailure
from gpt_field.robodojo_direct_sim import rotation_vector

scene, state, q, record, err = scene_at(13)
R0 = grip_pose(state)[:3, :3]
xyzw = matrix_to_quaternion_xyzw(R0); wxyz = [float(xyzw[3]), *map(float, xyzw[:3])]
rows = []
for xy in ((0.05, -0.05), (0.08, 0.15), (-0.15, -0.28), (0.10, -0.12), (-0.20, 0.20), (0.06, 0.25)):
    pl = CandidatePlanner(kin); pl._prepare(scene, state, q)
    above = np.array([xy[0], xy[1], 0.50]); into = np.array([xy[0], xy[1], 0.425])
    qa = np.asarray(pl.bound.solve(above, R0, q, max_iterations=80).joints)
    pose, _ = pl.bound.forward(qa)
    st = np.r_[pose[:3, 3], rotation_vector(pose[:3, :3]), .04, -.04]
    op = {'kind': 'grasp', 'object': 'white_mug', 'target': None, 'grasp_region': None, 'contact_mode': 'free_object',
          'keep_upright': False, 'placement_relation': None, 'goal_pose': None, 'displacement': None,
          'approach_direction': None, '_keypose_index': 0,
          'keyposes': [{'name': 'grasp', 'pose': {'position': into.tolist(), 'quaternion_wxyz': wxyz},
                        'gripper': 'keep', 'profile': 'contact', 'observe_after': False}]}
    try:
        plan = pl.plan(op, scene, st, qa)
        r = {'xy': xy, 'planned': True, 'uncertain_obstacle_samples': plan.evidence['uncertain_obstacle_samples'],
             'uncertain_examples': [e['part'] + '@' + str(np.round(e['point_world_m'], 3).tolist()) for e in plan.evidence['uncertain_obstacle_examples']]}
    except PlanningFailure as exc:
        ev = exc.evidence.get('evidence', {})
        r = {'xy': xy, 'planned': False, 'reason': exc.reason, 'part': ev.get('part'),
             'point': None if ev.get('point_world_m') is None else np.round(ev['point_world_m'], 3).tolist(),
             'endpoint': ev.get('endpoint'), 'evidence': ev.get('obstacle_evidence')}
    rows.append(r)
    print(json.dumps(r))
(WORK / 'data/a12_table_penetration.json').write_text(json.dumps(rows, indent=1))
