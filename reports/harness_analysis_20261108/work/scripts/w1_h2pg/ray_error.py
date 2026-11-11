"""Signed DA3 depth error on robot pixels: DA3 z minus ray entry into the robot convex hulls.

Negative = predicted surface in front of (toward the camera) the hull front face.
Positive and below hull thickness = inside hull; beyond hull exit = behind robot.
Offline, saved artifacts only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import restore, record_by_arrays, feedback_state, kinematics  # noqa: E402
from gpt_field.foundation_harness.robot_geometry import RobotCollisionModel  # noqa: E402

OUT = Path(__file__).parent
CASES = [('frame_000005_revision_000011.npz', 4), ('frame_000019_revision_000041.npz', 18)]
kin = kinematics()
model = RobotCollisionModel(kin)
out = []
for arrays, step in CASES:
    record = record_by_arrays(arrays)
    scene, data, _ = restore(record)
    state, q = feedback_state(step)
    shapes = model.shapes(kin.bind(state, q), q, state[6:8])
    for cam in scene.frame.cameras:
        view = scene.geometry._views[cam.name]
        K, E = cam.intrinsics, cam.world_to_camera
        rows, cols = view.rows, view.cols
        rays = np.column_stack((cols, rows, np.ones(len(rows)))) @ np.linalg.inv(K).T  # z = 1
        z = view.estimate.depth_m[rows, cols]
        t_enter_best = np.full(len(rows), np.inf)
        t_exit_best = np.full(len(rows), np.inf)
        part = np.full(len(rows), '', object)
        for shape in shapes:
            # Planes in camera frame: n_c . x_c + d_c <= 0
            world_from_local = shape.pose
            cam_from_local = E @ world_from_local
            Rcl, tcl = cam_from_local[:3, :3], cam_from_local[:3, 3]
            n_local, d_local = shape.equations[:, :3], shape.equations[:, 3]
            n_cam = n_local @ Rcl.T
            d_cam = d_local - n_cam @ tcl
            denom = rays @ n_cam.T  # (N, F)
            numer = -d_cam[None, :]  # origin at camera centre (0)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = numer / denom
            entering = denom < 0
            exiting = denom > 0
            t_in = np.where(entering, t, -np.inf).max(axis=1)
            t_out = np.where(exiting, t, np.inf).min(axis=1)
            # Origin outside a plane with denom == 0 means no hit.
            parallel_out = np.any((np.abs(denom) < 1e-12) & (d_cam[None, :] > 0), axis=1)
            hit = (t_in <= t_out) & (t_out > 0) & ~parallel_out
            t_in = np.maximum(t_in, 0)
            better = hit & (t_in < t_enter_best)
            t_enter_best[better] = t_in[better]
            t_exit_best[better] = t_out[better]
            part[better] = shape.name
        robot = np.isfinite(t_enter_best)
        err = z[robot] - t_enter_best[robot]
        thick = (t_exit_best - t_enter_best)[robot]
        cats = {
            'in_front_of_hull_gt_6mm': int(np.sum(err < -.006)),
            'in_front_3_to_6mm': int(np.sum((err >= -.006) & (err < -.003))),
            'in_front_0_to_3mm': int(np.sum((err >= -.003) & (err < 0))),
            'inside_hull': int(np.sum((err >= 0) & (err <= thick))),
            'behind_hull_exit': int(np.sum(err > thick)),
        }
        by_part = {}
        for name in np.unique(part[robot]):
            sel = part[robot] == name
            e = err[sel]
            by_part[str(name)] = {'n': int(sel.sum()), 'median_err_m': float(np.median(e)),
                                  'p10_err_m': float(np.percentile(e, 10)), 'p90_err_m': float(np.percentile(e, 90)),
                                  'median_hull_entry_depth_m': float(np.median(t_enter_best[robot][sel]))}
        row = {'arrays': arrays, 'view': cam.name, 'robot_ray_pixels_with_depth': int(robot.sum()),
               'signed_error_percentiles_m': {p: float(np.percentile(err, p)) for p in (5, 10, 25, 50, 75, 90, 95)},
               'abs_error_median_m': float(np.median(np.abs(err))),
               'categories': cats, 'by_part': by_part}
        out.append(row)
        print(json.dumps(row, indent=1), flush=True)
(OUT / 'ray_error_results.json').write_text(json.dumps(out, indent=1))
