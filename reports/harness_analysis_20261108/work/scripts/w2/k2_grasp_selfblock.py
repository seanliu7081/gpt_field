"""Would a plausible grasp of a TARGET be rejected by that target's own unowned-but-strong points?
Top-down grip at mug centre, fingers straddling the wall, with the planner's grasp allowances
(allowed_contacts = object id, contact_parts = finger/pad) exactly as motion.py:_astra_keypose sets them."""
import json, sys
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
M = json.load(open('/workspace/gpt_field/output/foundation_harness_preparation_001/inputs/layout_manifest.json'))
poses = M['custom_after_passive_hold_object_poses']
tz = truth['table_top_z']
out = []
for step, target, body, key in ((4, 'yellow_white_mug', 'white_yellow_mug_1', 'white_yellow_mug_1'),
                                (13, 'white_mug', 'porcelain_mug_1', 'porcelain_mug_1'),
                                (15, 'white_mug', 'porcelain_mug_1', 'porcelain_mug_1'),
                                (19, 'white_mug', 'porcelain_mug_1', 'porcelain_mug_1')):
    rec = record_for_step(step); scene, _ = restore(rec); state, q = step_state(step)
    if target not in scene.objects:
        out.append({'step': step, 'target': target, 'note': 'target absent'}); continue
    planner = CandidatePlanner(kin); planner._prepare(scene, state, q); planner._deadline = float('inf')
    bound = planner.bound; R = bound.forward(q)[0][:3, :3]
    c = np.array(poses[body]['body_xyz_m']); top = truth[key]['upper'][2]
    for depth in (0.0, 0.015, 0.03):
        xyz = np.array([c[0], c[1], top - depth])
        res = bound.solve(xyz, R, q, max_iterations=200)
        if not res.ok:
            out.append({'step': step, 'target': target, 'depth_below_rim_m': depth, 'ik': False}); continue
        for fingers in ((.04, -.04), (0., 0.)):
            planner._contact_parts = ('finger', 'pad')
            r = planner._configuration(np.asarray(res.joints), scene, np.array(fingers), None, (target,), free_check=False)
            out.append({'step': step, 'frame': scene.frame_id, 'target': target, 'depth_below_rim_m': depth, 'fingers': fingers,
                        'safe': bool(r['safe']), 'reason': r['reason'], 'part': r.get('part'), 'point_owner': r.get('point_object_id'),
                        'point': None if r.get('point_world_m') is None else [round(v, 3) for v in r['point_world_m']],
                        'uncertain_samples': r.get('uncertain_obstacle_samples')})
for o in out: print(json.dumps(o))
(W2 / 'k2_grasp_selfblock.json').write_text(json.dumps(out, indent=1))
