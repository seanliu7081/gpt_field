"""v2 planner with depth-tested robot self-masking applied upstream (offline).

Self pixels: inside the projected current-robot hull silhouette (dilated 6 px)
AND not clearly in front of the robot surface along the ray (3 cm tolerance;
pixels outside the undilated hulls use a 5 cm 3-D proximity test). Masked
points are set to NaN in the scene cloud before planning, emulating removal in
perception, so the 12 s planning budget measures planning only.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE / 'cf_v2.py').read_text().split('out = {}')[0]
ns = {'__file__': str(HERE / 'cf_v2.py')}
exec(compile(src, 'cf_v2_head', 'exec'), ns)
restore, step_inputs, wxyz, m2, kin, planning_surface = (ns[k] for k in (
    'restore', 'step_inputs', 'wxyz', 'm2', 'kin', 'planning_surface'))
SceneSnapshot = ns['SceneSnapshot']
DEPTH_TOL, DILATE = .03, 6


def residual(shape, points):
    local = (np.asarray(points) - shape.pose[:3, 3]) @ shape.pose[:3, :3]
    return np.max(local @ shape.equations[:, :3].T + shape.equations[:, 3], axis=1)


def masked_scene(scene, state, q):
    shapes = m2.CandidatePlanner(kin).robot.shapes(kin.bind(state, q), q, state[6:8])
    pts = np.array(scene.scene_points, float)
    owners = np.asarray(scene.scene_point_object_ids, object)
    slices = scene.evidence['scene_point_view_slices']
    removed, foreground_kept = 0, 0
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
        mask = cv2.dilate(mask, np.ones((2 * DILATE + 1, 2 * DILATE + 1), np.uint8))
        start, end = slices[name]
        inside = np.flatnonzero(mask[view.rows, view.cols].astype(bool))
        cand = view.points_world[inside]
        origin = -est.world_to_camera[:3, :3].T @ est.world_to_camera[:3, 3]
        measured = np.linalg.norm(cand - origin, axis=1)
        entry = np.full(len(cand), np.inf)
        near = np.full(len(cand), np.inf)
        for shape in shapes:
            entry = np.minimum(entry, shape.ray_entry_distances(origin, cand))
            near = np.minimum(near, residual(shape, cand))
        is_self = np.where(np.isfinite(entry), measured >= entry - DEPTH_TOL, near <= .05)
        idx = start + inside[is_self]
        idx = idx[owners[idx] == '']          # never mask assigned object material
        foreground_kept += int((~is_self).sum())
        pts[idx] = np.nan
        removed += len(idx)
    new = SceneSnapshot(scene.frame_id, scene.captured_at_s, scene.generated_at_s, scene.objects, pts,
                        scene.geometry, scene.frame, scene.candidates, scene.evidence,
                        scene_point_object_ids=owners)
    return new, removed, foreground_kept


def plan(op, scene, state, q):
    planner = m2.CandidatePlanner(kin)
    t = time.perf_counter()
    try:
        p = planner.plan(op, scene, state, q)
        return {'ok': True, 'waypoints': len(p.joint_paths[0]), 's': round(time.perf_counter() - t, 2)}
    except m2.PlanningFailure as exc:
        ev = exc.evidence.get('evidence', {})
        return {'ok': False, 'reason': exc.reason, 'part': ev.get('part'), 'point_object_id': ev.get('point_object_id'),
                'point_world_m': ev.get('point_world_m'), 's': round(time.perf_counter() - t, 2)}


out = {}
for step, disp in ((9, [0, -.15, .05]), (13, [0, -.05, 0])):
    rec, state, q = step_inputs(step)
    scene, removed, fg = masked_scene(restore(rec), state, q)
    op = {'kind': 'move', 'object': None, 'target': None, 'grasp_region': None, 'keep_upright': False,
          'goal_pose': None, 'displacement': disp, 'keyposes': [], 'contact_mode': 'free_object'}
    out[f'move_step{step}'] = {'masked_points': removed, 'foreground_kept_in_silhouette': fg, 'plan': plan(op, scene, state, q)}

rec, state, q = step_inputs(18)
t0 = time.perf_counter()
scene, removed, fg = masked_scene(restore(rec), state, q)
mask_s = time.perf_counter() - t0
mug = scene.objects['white_mug']
pts, _ = planning_surface(mug)
center, top = pts.mean(0), float(pts[:, 2].max())
R_now = m2._pose(state)[:3, :3]
out['mask_offline_python_s'] = round(mask_s, 2)
out['masked_points_step18'] = removed
out['foreground_kept_in_silhouette_step18'] = fg
for label, position, name, profile in (('pregrasp_12cm_above_top', [center[0], center[1], top + .12], 'pregrasp', 'transit'),
                                       ('pregrasp_6cm_above_top', [center[0], center[1], top + .06], 'pregrasp', 'transit'),
                                       ('grasp_rim_level_direct', [center[0], center[1], top - .01], 'grasp', 'contact')):
    op = {'kind': 'grasp', 'object': 'white_mug', 'target': None, 'grasp_region': None, 'keep_upright': True,
          'goal_pose': None, 'displacement': None, 'contact_mode': 'free_object', 'placement_relation': None,
          'keyposes': [{'name': name, 'pose': {'position': [float(x) for x in position], 'quaternion_wxyz': wxyz(R_now)},
                        'gripper': 'open' if name == 'pregrasp' else 'keep', 'profile': profile, 'observe_after': True}]}
    out[label] = plan(op, scene, state, q)
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
