"""Collision-avoidance sensitivity of keypose-2 evidence: place the grip site INSIDE true obstacles
(evaluation-only truth) and ask CandidatePlanner._configuration whether the pose is rejected.
Scene evidence = saved run-001 frame (initial robot pose); robot model = current static Panda assets.
"""
import json, sys
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
M = json.load(open('/workspace/gpt_field/output/foundation_harness_preparation_001/inputs/layout_manifest.json'))
poses = M['custom_after_passive_hold_object_poses']
step = int(sys.argv[1]) if len(sys.argv) > 1 else 13
rec = record_for_step(step); scene, _ = restore(rec); state, q = step_state(step)
planner = CandidatePlanner(kin); planner._prepare(scene, state, q); planner._deadline = float('inf')
bound = planner.bound
R_now = bound.forward(q)[0][:3, :3]
tz = truth['table_top_z']
cases = []
def probe(label, xyz, owner_note=''):
    res = bound.solve(np.asarray(xyz, float), R_now, q, max_iterations=200)
    if not res.ok:
        cases.append({'case': label, 'xyz': list(map(float, xyz)), 'ik': False, 'pos_err': res.position_error_m}); return
    out = planner._configuration(np.asarray(res.joints), scene, state[6:8], None, (), free_check=False)
    cases.append({'case': label, 'xyz': [round(float(v), 3) for v in xyz], 'ik': True, 'safe': bool(out['safe']), 'reason': out['reason'],
                  'part': out.get('part'), 'blocking_point_owner': out.get('point_object_id'),
                  'uncertain_samples': out.get('uncertain_obstacle_samples', 0),
                  'uncertain_example_parts': [e['part'] for e in out.get('uncertain_obstacle_examples', [])]})
# grip site positions (fingertip centre) inside each true object at 3 heights, top-down orientation kept from the current pose
for name in ('tomato_sauce_obstacle_1', 'red_coffee_mug_1', 'porcelain_mug_1', 'white_yellow_mug_1', 'plate_1', 'plate_2'):
    c = np.array(poses[name]['body_xyz_m'])
    key = {'tomato_sauce_obstacle_1': 'can'}.get(name, name)
    top = truth[key]['upper'][2]
    for frac in (0.25, 0.5, 0.9):
        z = tz + frac * (top - tz)
        probe(f'{name}_inside_z{frac:.2f}', [c[0], c[1], z])
    probe(f'{name}_2cm_above_top', [c[0], c[1], top + .02])
# free-space controls between objects (should be allowed) and table penetration
probe('free_air_mid_table_z0.60', [-0.05, -0.05, 0.60])
probe('free_air_z0.55_between_mugs', [-0.05, -0.04, 0.55])
probe('table_penetration_z0.41_open_area', [0.10, -0.05, 0.41])
probe('table_touch_z0.437_open_area', [0.10, -0.05, 0.437])
out = {'step': step, 'frame': scene.frame_id, 'targets': {k: o.query for k, o in scene.objects.items()},
       'strong_points': int(planner._strong.sum()), 'cloud': int(len(planner._cloud)), 'cases': cases}
print(json.dumps(out, indent=1))
(W2 / f'k2_intrusion_test_step{step}.json').write_text(json.dumps(out, indent=1))
