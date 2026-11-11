"""DA3 depth error measured on the robot's own (known) body in the saved frames.

For depth pixels inside the projected current-robot silhouette (no dilation),
compare the predicted depth along the camera ray with the first entry into the
robot's convex collision hulls along that ray (same calibration). The convex
hull over-approximates the mesh, so 'in front of hull' errors are lower bounds
and 'behind hull entry' includes hull slack. Offline, saved arrays only.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pkg_v2'))  # v2 robot_geometry has ray_entry_distances
src = (HERE / 'cf_v2.py').read_text().split('out = {}')[0]
ns = {'__file__': str(HERE / 'cf_v2.py')}
exec(compile(src, 'cf_v2_head', 'exec'), ns)
restore, step_inputs, m2, kin = ns['restore'], ns['step_inputs'], ns['m2'], ns['kin']

out = {}
for step in (4, 18):
    rec, state, q = step_inputs(step)
    scene = restore(rec)
    planner = m2.CandidatePlanner(kin)
    bound = kin.bind(state, q)
    shapes = planner.robot.shapes(bound, q, state[6:8])
    for name, view in scene.geometry._views.items():
        est = view.estimate
        h, w = view.valid.shape
        mask = np.zeros((h, w), np.uint8)
        for shape in shapes:
            cam = shape.surface_points() @ est.world_to_camera[:3, :3].T + est.world_to_camera[:3, 3]
            cam = cam[cam[:, 2] > 1e-3]
            if len(cam) >= 3:
                uv = cam @ est.intrinsics.T
                cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv[:, :2] / uv[:, 2:3]).astype(np.int32)), 1)
        sel = mask[view.rows, view.cols].astype(bool)
        pts = view.points_world[sel]
        origin = -est.world_to_camera[:3, :3].T @ est.world_to_camera[:3, 3]
        measured = np.linalg.norm(pts - origin, axis=1)
        entry = np.full(len(pts), np.inf)
        for shape in shapes:
            entry = np.minimum(entry, shape.ray_entry_distances(origin, pts))
        hit = np.isfinite(entry)
        err = measured[hit] - entry[hit]          # >0: DA3 behind hull entry; <0: in front of the robot
        res = {'silhouette_points': int(len(pts)), 'rays_hitting_hull': int(hit.sum()),
               'signed_error_m_percentiles': {p: float(np.percentile(err, p)) for p in (5, 25, 50, 75, 95)},
               'abs_error_m_median': float(np.median(np.abs(err))),
               'fraction_in_front_by_more_than_6mm': float(np.mean(err < -.006)),
               'fraction_in_front_by_more_than_20mm': float(np.mean(err < -.02)),
               'fraction_behind_by_more_than_20mm': float(np.mean(err > .02)),
               'median_range_m': float(np.median(measured))}
        out[f'step{step}_{name}'] = res
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
