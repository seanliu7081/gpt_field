"""Which points block the payload at the lift start, and over which grasp depths does payload_robot_collision persist."""
import json, sys
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
from gpt_field.foundation_harness.motion import _payload_shape
from gpt_field.foundation_harness.robot_geometry import CollisionShape, intersect
M = json.load(open('/workspace/gpt_field/output/foundation_harness_preparation_001/inputs/layout_manifest.json'))
poses = M['custom_after_passive_hold_object_poses']
out = {}
for step, target, body, key in ((4, 'yellow_white_mug', 'white_yellow_mug_1', 'white_yellow_mug_1'),
                                (15, 'white_mug', 'porcelain_mug_1', 'porcelain_mug_1'),
                                (19, 'white_mug', 'porcelain_mug_1', 'porcelain_mug_1')):
    rec = record_for_step(step); scene, _ = restore(rec); state, q = step_state(step)
    planner = CandidatePlanner(kin); planner._prepare(scene, state, q); planner._deadline = float('inf')
    bound = planner.bound; R = bound.forward(q)[0][:3, :3]
    c = np.array(poses[body]['body_xyz_m']); top = truth[key]['upper'][2]
    pts, _ = planning_surface(scene.objects[target])
    rows = []
    for depth in (-0.01, 0.0, 0.005, 0.01, 0.02, 0.03, 0.04):
        xyz = np.array([c[0], c[1], top - depth])
        res = bound.solve(xyz, R, q, max_iterations=200)
        if not res.ok:
            rows.append({'depth_below_collision_rim_m': depth, 'ik': False}); continue
        pose = np.eye(4); pose[:3, 3] = xyz; pose[:3, :3] = R
        local = (pts - xyz) @ R
        shape = _payload_shape(local)
        fk, _ = bound.forward(np.asarray(res.joints))
        payload = CollisionShape(shape.name, shape.group, shape.vertices, shape.equations, fk, 'payload')
        shapes = planner.robot.shapes(bound, np.asarray(res.joints), np.array([0., 0.]))
        hits = [s.name for s in shapes if s.kind not in ('finger', 'pad') and intersect(payload, s, planner.config.collision_margin_m)]
        hits0 = [s.name for s in shapes if s.kind not in ('finger', 'pad') and intersect(payload, s, 0.)]
        # scene points inside the payload hull (+6 mm), not owned by the target
        ids = np.asarray(planner._tree.query_ball_point(payload.center, payload.radius + .006), int)
        inside = ids[payload.contains(planner._cloud[ids], .006)]
        inside = inside[~np.isin(planner._owners[inside], [target])]
        strong_in = inside[planner._strong[inside]]
        P = planner._cloud[strong_in]
        in_true_obj = in_box(P, truth[key]['lower'], truth[key]['upper'], .01) if len(P) else np.zeros(0, bool)
        rows.append({'depth_below_collision_rim_m': depth, 'payload_vs_robot_hits_6mm': hits, 'payload_vs_robot_hits_0mm': hits0,
                     'unowned_strong_points_in_hull': int(len(strong_in)),
                     'of_which_inside_true_target_box_1cm': int(in_true_obj.sum()) if len(P) else 0,
                     'unowned_strong_z_range': [round(float(P[:, 2].min()), 3), round(float(P[:, 2].max()), 3)] if len(P) else None,
                     'payload_hull_top_z': round(float((shape.vertices @ R.T + xyz)[:, 2].max()), 3),
                     'payload_hull_bottom_z': round(float((shape.vertices @ R.T + xyz)[:, 2].min()), 3)})
    out[f'step{step}_{target}'] = rows
    for r in rows: print(step, target, json.dumps(r))
(W2 / 'k2_payload_detail.json').write_text(json.dumps(out, indent=1))
