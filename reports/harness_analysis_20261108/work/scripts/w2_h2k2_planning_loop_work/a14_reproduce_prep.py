"""A14: reproduce prep/saved_input_check.json case 2 with the overlay (sanity of the offline harness)."""
import json
from time import perf_counter
import numpy as np
from k2common import WORK, scene_at, kin, events
from gpt_field.contact_skills import matrix_to_quaternion_xyzw
from gpt_field.foundation_harness.feedback import grip_pose
from gpt_field.foundation_harness.motion import CandidatePlanner
scene, state, q, record, err = scene_at(13)
original = next(e['operation'] for e in events if e['event'] == 'operation_interrupted' and e['step'] == 13)
xyzw = matrix_to_quaternion_xyzw(grip_pose(state)[:3, :3]); wxyz = [xyzw[3], *xyzw[:3]]
astra = {**original, 'displacement': None, 'keyposes': [{'name': 'pregrasp', 'pose': {'position': [-0.1, -0.135, 0.65], 'quaternion_wxyz': wxyz},
         'gripper': 'keep', 'profile': 'transit', 'observe_after': True}]}
out = {}
for name, op in (('previously_rejected_small_viewpoint_move', original), ('previous_success_astra_first_keypose', astra)):
    pl = CandidatePlanner(kin); t0 = perf_counter()
    plan = pl.plan(op, scene, state, q)
    out[name] = {'joint_path_points': len(plan.joint_paths[0]), 'uncertain_obstacle_samples': plan.evidence['uncertain_obstacle_samples'],
                 'unknown_sample_visits': plan.evidence['unknown_sample_visits'], 'self_removed': plan.evidence['self_surface_points_removed'],
                 'elapsed_s': round(perf_counter() - t0, 2)}
prep = json.load(open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/h2_keypose2_snapshot/prep/saved_input_check.json'))
out['prep'] = {p['case']: {'joint_path_points': p['joint_path_points'], 'uncertain_obstacle_samples': p['evidence']['uncertain_obstacle_samples'],
               'unknown_sample_visits': p['evidence']['unknown_sample_visits'], 'self_removed': p['evidence']['self_surface_points_removed'],
               'elapsed_s': round(p['elapsed_s'], 2)} for p in prep['paths']}
print(json.dumps(out, indent=1))
(WORK / 'data/a14_reproduce_prep.json').write_text(json.dumps(out, indent=1))
