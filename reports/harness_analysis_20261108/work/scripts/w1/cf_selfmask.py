"""Would image-space robot silhouette masking remove the robot 'ghost' points?

Projects the frozen collision hulls (current measured joints, public calibration)
into each saved camera, rasterizes convex silhouettes, dilates by a few pixels,
and counts how many raw depth points near the current robot fall inside.
Offline, saved arrays only, CPU only.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.argv = [sys.argv[0], '/dev/null'] + sys.argv[1:]
HERE = Path(__file__).resolve().parent
head = (HERE / 'cf_planner.py').read_text().split('results = {}')[0]
ns = {'__file__': str(HERE / 'cf_planner.py')}
exec(compile(head, 'cf_planner_head', 'exec'), ns)
restore, step_inputs, residual = ns['restore'], ns['step_inputs'], ns['residual']
CandidatePlanner, kin = ns['CandidatePlanner'], ns['kin']

out = {}
for step in (9, 18):
    record, state, q = step_inputs(step)
    scene, data = restore(record)
    planner = CandidatePlanner(kin)
    planner._prepare(scene, state, q)
    shapes = planner._initial_shapes
    per_view = {}
    for name, view in scene.geometry._views.items():
        est = view.estimate
        h, w = view.valid.shape
        K, E = est.intrinsics, est.world_to_camera
        mask = np.zeros((h, w), np.uint8)
        for shape in shapes:
            verts = shape.surface_points()
            cam = verts @ E[:3, :3].T + E[:3, 3]
            if np.any(cam[:, 2] <= 1e-3):
                cam = cam[cam[:, 2] > 1e-3]
                if len(cam) < 3:
                    continue
            uv = cam @ K.T
            uv = uv[:, :2] / uv[:, 2:3]
            hull = cv2.convexHull(np.round(uv).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1)
        pts = view.points_world
        rows, cols = view.rows, view.cols
        current = np.full(len(pts), np.inf)
        tree = cKDTree(pts)
        near_ids = set()
        for shape in shapes:
            near_ids.update(tree.query_ball_point(shape.center, shape.radius + .05))
        near_ids = np.array(sorted(near_ids), int)
        if len(near_ids):
            current[near_ids] = np.min(np.stack([residual(s, pts[near_ids]) for s in shapes]), axis=0)
        near = current <= .05
        result = {'depth_hw': [int(h), int(w)], 'points': int(len(pts)),
                  'points_within_5cm_of_current_robot_hulls': int(near.sum()),
                  'silhouette_pixels': int(mask.sum())}
        for dilate in (0, 3, 6):
            m = cv2.dilate(mask, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)) if dilate else mask
            inside = m[rows, cols].astype(bool)
            result[f'dilate_{dilate}px'] = {
                'near_points_inside_silhouette': int((near & inside).sum()),
                'near_points_outside_silhouette': int((near & ~inside).sum()),
                'all_points_inside_silhouette': int(inside.sum())}
        per_view[name] = result
    out[f'step{step}'] = per_view
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
