"""A13: size of the keypose-2 Astra context vs the 120,000-char hard limit (client.py:410-411).

Measures with keypose-2 code: SceneSnapshot.summary() for each saved run-001
revision (scene part of the context), one structured planning-failure detail
(executor.py:283-289), and a 12-keypose operation record. A ValueError from the
limit is uncaught in act() -> runner 'infrastructure_error' (episode ends).
"""
import json

import numpy as np

from k2common import WORK, records, restore, kin, events, scene_at  # noqa: F401
from gpt_field.contact_skills import matrix_to_quaternion_xyzw
from gpt_field.foundation_harness.feedback import grip_pose
from gpt_field.foundation_harness.motion import CandidatePlanner, PlanningFailure
from gpt_field.foundation_harness.types import jsonable

sizes = []
seen = set()
for rec in records:
    if rec['arrays'] in seen:
        continue
    seen.add(rec['arrays'])
    scene, err = restore(rec)  # noqa: F821
    s = json.dumps(jsonable(scene.summary(0.)))
    sizes.append({'arrays': rec['arrays'], 'objects': len(scene.objects), 'candidates': len(scene.candidates), 'scene_summary_chars': len(s)})
# one planning-failure detail as the executor stores it
scene, state, q, record, err = scene_at(13)
R0 = grip_pose(state)[:3, :3]
xyzw = matrix_to_quaternion_xyzw(R0); wxyz = [float(xyzw[3]), *map(float, xyzw[:3])]
pl = CandidatePlanner(kin); pl._prepare(scene, state, q)
kp = lambda name, xyz, prof='transit': {'name': name, 'pose': {'position': list(xyz), 'quaternion_wxyz': wxyz}, 'gripper': 'keep', 'profile': prof, 'observe_after': False}
op = {'kind': 'pick_place', 'object': 'white_mug', 'target': 'seed_1', 'grasp_region': None, 'contact_mode': 'free_object',
      'keep_upright': True, 'placement_relation': 'on', 'goal_pose': None, 'displacement': None, 'approach_direction': None,
      'keyposes': [kp('pregrasp', (-0.085, -0.10, 0.61))] + [kp('grasp', (-0.085, -0.10, 0.51), 'contact')] + [kp('lift', (-0.085, -0.10, 0.60 + 0.005 * i)) for i in range(10)]}
detail = None
try:
    pl.plan({**op, '_keypose_index': 1}, scene, state, q)
except PlanningFailure as exc:
    detail = {'reason': exc.reason, 'evidence': exc.evidence, 'keypose_index': 1}
feedback = {'reason': 'planning_unavailable', 'detail': detail, 'operation': op, 'interaction': {'attachment': 'unknown'}}
history_entry = {**feedback, 'status': 'incomplete'}
fb_chars = len(json.dumps(jsonable(feedback)))
op_chars = len(json.dumps(jsonable(op)))
worst_scene = max(x['scene_summary_chars'] for x in sizes)
estimate = worst_scene + fb_chars * 13 + 3000  # 12 history entries + current feedback + fixed fields (geometry ~2 KB)
out = {'scene_summaries': sizes, 'failure_feedback_chars': fb_chars, 'twelve_keypose_operation_chars': op_chars,
       'failure_reason': None if detail is None else detail['reason'],
       'worst_scene_summary_chars': worst_scene,
       'estimate_after_12_failures_chars': estimate, 'limit_chars': 120000}
print(json.dumps({k: v for k, v in out.items() if k != 'scene_summaries'}, indent=1))
print(sorted(x['scene_summary_chars'] for x in sizes))
(WORK / 'data/a13_context_size.json').write_text(json.dumps(out, indent=1))
