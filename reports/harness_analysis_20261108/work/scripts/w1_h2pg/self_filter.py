"""Quantify robot self-surface removal (3 mm) vs collision dilation (6 mm).

Offline, saved artifacts only. Uses the frozen run's own robot model and planner.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from common import restore, record_by_arrays, feedback_state, kinematics  # noqa: E402
from gpt_field.foundation_harness.robot_geometry import RobotCollisionModel  # noqa: E402
from gpt_field.foundation_harness.motion import CandidatePlanner  # noqa: E402

OUT = Path(__file__).parent
CASES = [('frame_000005_revision_000011.npz', 4), ('frame_000010_revision_000022.npz', 9),
         ('frame_000019_revision_000041.npz', 18)]

kin = kinematics()
model = RobotCollisionModel(kin)
results = []
for arrays, step in CASES:
    record = record_by_arrays(arrays)
    scene, data, _ = restore(record)
    state, q = feedback_state(step)
    bound = kin.bind(state, q)
    shapes = model.shapes(bound, q, state[6:8])
    pts = np.asarray(scene.scene_points, float)
    owners = scene.scene_point_object_ids
    slices = record['snapshot']['evidence']['scene_point_view_slices']
    view_of = np.empty(len(pts), object)
    for name, (a, b) in slices.items():
        view_of[a:b] = name
    tree = cKDTree(pts)
    residual = np.full(len(pts), np.inf)
    nearest_shape = np.full(len(pts), '', object)
    for shape in shapes:
        ids = np.asarray(tree.query_ball_point(shape.center, shape.radius + 0.08), int)
        if not len(ids):
            continue
        local = (pts[ids] - shape.pose[:3, 3]) @ shape.pose[:3, :3]
        r = np.max(local @ shape.equations[:, :3].T + shape.equations[:, 3], axis=1)
        better = r < residual[ids]
        residual[ids[better]] = r[better]
        nearest_shape[ids[better]] = shape.name
    bins = [-np.inf, 0., .003, .006, .01, .02, .03, .05, .08]
    hist = {}
    for view in slices:
        sel = view_of == view
        counts, _ = np.histogram(residual[sel & np.isfinite(residual)], bins=bins)
        hist[view] = {f'{bins[i]:.3f}..{bins[i+1]:.3f}': int(c) for i, c in enumerate(counts)}
    # Points in the 3-6 mm shell (survive removal, collide at start) per shape.
    shell = (residual > .003) & (residual <= .006) & (owners == '')
    shell_by_shape = {}
    for name in np.unique(nearest_shape[shell]):
        shell_by_shape[str(name)] = int(np.sum(shell & (nearest_shape == name)))
    # Planner's own preparation, and all colliding shapes at the start configuration.
    planner = CandidatePlanner(kin)
    planner._prepare(scene, state, q)
    colliding = {}
    for shape in shapes:
        if planner._tree is None:
            break
        ids = np.asarray(planner._tree.query_ball_point(shape.center, shape.radius + planner.config.collision_margin_m), int)
        if not len(ids):
            continue
        inside = shape.contains(planner._cloud[ids], planner.config.collision_margin_m)
        if inside.any():
            colliding[shape.name] = int(inside.sum())
    # Image-space silhouette of the robot model (convex hull projections).
    sil_stats = {}
    for cam in scene.frame.cameras:
        h, w = cam.rgb.shape[:2]
        sil = np.zeros((h, w), np.uint8)
        K, E = cam.intrinsics, cam.world_to_camera
        for shape in shapes:
            v = shape.surface_points()
            c = v @ E[:3, :3].T + E[:3, 3]
            if np.any(c[:, 2] <= 1e-3):
                # Partly behind camera: rasterize only vertices in front (approximate).
                c = c[c[:, 2] > 1e-3]
                if len(c) < 3:
                    continue
            uv = (c @ K.T)[:, :2] / c[:, 2:3]
            hull = cv2.convexHull(np.round(uv).astype(np.int32))
            cv2.fillConvexPoly(sil, hull, 1)
        a, b = slices[cam.name]
        view = scene.geometry._views[cam.name]
        rows, cols = view.rows, view.cols
        inside_sil = sil[rows, cols].astype(bool)
        removed_mask = np.ones(b - a, bool)
        removed_mask[:] = False
        res_v = residual[a:b]
        sil_stats[cam.name] = {
            'silhouette_pixels': int(sil.sum()),
            'scene_points_in_silhouette': int(inside_sil.sum()),
            'silhouette_points_within_3mm_of_hull(removed)': int(np.sum(inside_sil & (res_v <= .003))),
            'silhouette_points_3_to_6mm': int(np.sum(inside_sil & (res_v > .003) & (res_v <= .006))),
            'silhouette_points_6mm_to_2cm': int(np.sum(inside_sil & (res_v > .006) & (res_v <= .02))),
            'silhouette_points_over_2cm': int(np.sum(inside_sil & (res_v > .02))),
            'silhouette_residual_percentiles_m': {p: float(np.percentile(res_v[inside_sil], p)) for p in (5, 25, 50, 75, 95)} if inside_sil.any() else None,
        }
        cv2.imwrite(str(OUT / f'robot_silhouette_{cam.name}_{arrays[:12]}.png'), sil * 255)
    link0 = next(s for s in shapes if s.name == 'link0_collision')
    lv = link0.surface_points()
    results.append({
        'arrays': arrays, 'step': step,
        'scene_points': int(len(pts)),
        'planner_self_removed_points': int(planner._self_removed),
        'residual_histogram_by_view': hist,
        'unowned_points_in_3_to_6mm_shell_by_shape': shell_by_shape,
        'start_configuration_colliding_shapes_points': colliding,
        'robot_shape_names': [s.name for s in shapes],
        'robot_shape_kinds': sorted({s.kind for s in shapes}),
        'link0_world_bounds_m': [lv.min(0).tolist(), lv.max(0).tolist()],
        'silhouette': sil_stats,
    })
    print(json.dumps(results[-1], indent=1), flush=True)
(OUT / 'self_filter_results.json').write_text(json.dumps(results, indent=1))
