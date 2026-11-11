"""Are grasp-configuration collision hits the robot's own current-pose depth points?

Offline, frozen source, saved arrays only (see cf_planner.py).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.argv = [sys.argv[0], '/dev/null'] + sys.argv[1:]
HERE = Path(__file__).resolve().parent
src = (HERE / 'cf_planner.py').read_text()
# Reuse helpers without re-running the experiments: execute only definitions.
head = src.split('results = {}')[0]
namespace = {'__file__': str(HERE / 'cf_planner.py')}
exec(compile(head, 'cf_planner_head', 'exec'), namespace)
restore, step_inputs, residual, drop_base_points = (namespace[k] for k in ('restore', 'step_inputs', 'residual', 'drop_base_points'))
CandidatePlanner, MotionConfig, kin = namespace['CandidatePlanner'], namespace['MotionConfig'], namespace['kin']
out = {}
for step, object_id in ((4, 'yellow_white_mug'), (16, 'white_mug')):
    record, state, q = step_inputs(step)
    scene, _ = restore(record)
    planner = CandidatePlanner(kin)
    planner._prepare(scene, state, q)
    drop_base_points(planner)
    planner._deadline = time.perf_counter() + 60
    obj = scene.objects[object_id]
    op = {'kind': 'pick_place', 'object': object_id, 'grasp_region': None, 'approach_direction': None}
    grasps = planner._grasps(obj, op)
    rows = []
    for score, pose, fingers, contacts in grasps:
        ik = planner.bound.solve(pose[:3, 3], pose[:3, :3], q, max_iterations=80)
        if not ik.ok:
            continue
        qg = np.asarray(ik.joints)
        hits_by_shape = {}
        shapes = planner.robot.shapes(planner.bound, qg, np.array([.04, -.04]))
        for shape in shapes:
            ids = np.asarray(planner._tree.query_ball_point(shape.center, shape.radius + .006), int)
            if not len(ids):
                continue
            inside = ids[shape.contains(planner._cloud[ids], .006)]
            if not len(inside):
                continue
            hit = planner._cloud[inside]
            own = planner._owners[inside]
            # distance band of each hit point to the CURRENT robot hulls
            current = np.min(np.stack([residual(s, hit) for s in planner._initial_shapes]), axis=0)
            hits_by_shape[shape.name] = {
                'points': int(len(hit)),
                'owned_by_object': int((own != '').sum()),
                'unowned_within_5cm_of_current_robot_hull': int(((own == '') & (current <= .05)).sum()),
                'unowned_farther_than_5cm': int(((own == '') & (current > .05)).sum()),
                'median_residual_to_current_robot_m': float(np.median(current)),
                'min_z': float(hit[:, 2].min()), 'max_z': float(hit[:, 2].max())}
        rows.append({'approach': pose[:3, 2].round(2).tolist(), 'width_m': round(float(np.linalg.norm(np.subtract(*contacts))), 3),
                     'hits_by_shape': hits_by_shape})
    out[f'step{step}_{object_id}'] = rows
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print('ok')
