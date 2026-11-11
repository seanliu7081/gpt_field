"""v2 pregrasp blocker attribution + silhouette-masking counterfactual (offline)."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
src = (HERE / 'cf_v2.py').read_text().split('out = {}')[0]
ns = {'__file__': str(HERE / 'cf_v2.py')}
sys_argv = sys.argv
exec(compile(src, 'cf_v2_head', 'exec'), ns)
restore, step_inputs, wxyz, m2, kin, planning_surface = (ns[k] for k in (
    'restore', 'step_inputs', 'wxyz', 'm2', 'kin', 'planning_surface'))


def residual(shape, points):
    local = (np.asarray(points) - shape.pose[:3, 3]) @ shape.pose[:3, :3]
    return np.max(local @ shape.equations[:, :3].T + shape.equations[:, 3], axis=1)


def silhouette_keep(planner, scene, dilate=6):
    """Keep mask over planner._cloud after removing points whose source pixel lies
    inside the projected current-robot silhouette (unowned points only)."""
    # Reconstruct which view each kept cloud point came from via nearest match.
    keep = np.ones(len(planner._cloud), bool)
    for name, view in scene.geometry._views.items():
        est = view.estimate
        h, w = view.valid.shape
        mask = np.zeros((h, w), np.uint8)
        for shape in planner._initial_shapes:
            cam = shape.surface_points() @ est.world_to_camera[:3, :3].T + est.world_to_camera[:3, 3]
            cam = cam[cam[:, 2] > 1e-3]
            if len(cam) < 3:
                continue
            uv = cam @ est.intrinsics.T
            uv = uv[:, :2] / uv[:, 2:3]
            cv2.fillConvexPoly(mask, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
        mask = cv2.dilate(mask, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
        robot_points = view.points_world[mask[view.rows, view.cols].astype(bool)]
        if not len(robot_points):
            continue
        d, _ = cKDTree(robot_points).query(planner._cloud, distance_upper_bound=1e-7)
        keep &= ~(np.isfinite(d) & (planner._owners == ''))
    return keep


rec, state, q = step_inputs(18)
scene = restore(rec)
mug = scene.objects['white_mug']
pts, _ = planning_surface(mug)
center, top = pts.mean(0), float(pts[:, 2].max())
R_now = m2._pose(state)[:3, :3]
out = {}
for label, position in (('pregrasp_12cm_above_top', [center[0], center[1], top + .12]),
                        ('pregrasp_6cm_above_top', [center[0], center[1], top + .06])):
    planner = m2.CandidatePlanner(kin)
    planner._prepare(scene, state, q)
    planner._deadline = time.perf_counter() + 60
    planner._contact_parts = ()
    ik = planner.bound.solve(np.asarray(position), R_now, q, max_iterations=80)
    goal = np.asarray(ik.joints)
    count = max(1, int(np.ceil(np.max(np.abs(goal - q)) / planner.config.max_joint_step_rad)))
    first_fail = None
    for i, qq in enumerate(np.linspace(q, goal, count + 1)):
        r = planner._configuration(qq, scene, state[6:8], None, ())
        if not r['safe']:
            first_fail = {'sample': i, 'of': count + 1, **{k: r.get(k) for k in ('reason', 'part', 'point_object_id', 'point_world_m')}}
            if r.get('point_world_m') is not None:
                p = np.asarray([r['point_world_m']])
                first_fail['hit_residual_to_current_robot_m'] = float(min(residual(s, p)[0] for s in planner._initial_shapes))
            break
    entry = {'ik_ok': bool(ik.ok), 'max_joint_delta_rad': float(np.max(np.abs(goal - q))),
             'straight_edge_first_failure': first_fail}
    # counterfactual: silhouette masking of current-robot pixels, then plan()
    keep = silhouette_keep(planner, scene)
    removed = int((~keep).sum())
    original = planner._prepare

    def patched(scene_, state_, joints_, _orig=original, _p=planner):
        _p._scene_key = None
        _orig(scene_, state_, joints_)
        k = silhouette_keep(_p, scene_)
        _p._cloud, _p._owners = _p._cloud[k], _p._owners[k]
        _p._tree = cKDTree(_p._cloud)
    planner._prepare = patched
    op = {'kind': 'grasp', 'object': 'white_mug', 'target': None, 'grasp_region': None, 'keep_upright': True,
          'goal_pose': None, 'displacement': None, 'contact_mode': 'free_object', 'placement_relation': None,
          'keyposes': [{'name': 'pregrasp', 'pose': {'position': [float(x) for x in position], 'quaternion_wxyz': wxyz(R_now)},
                        'gripper': 'open', 'profile': 'transit', 'observe_after': True}]}
    t = time.perf_counter()
    try:
        p = planner.plan(op, scene, state, q)
        entry['silhouette_masked_plan'] = {'ok': True, 'waypoints': len(p.joint_paths[0]), 's': time.perf_counter() - t,
                                           'unknown_sample_visits': p.evidence.get('unknown_sample_visits')}
    except m2.PlanningFailure as exc:
        ev = exc.evidence.get('evidence', {})
        entry['silhouette_masked_plan'] = {'ok': False, 'reason': exc.reason, 'part': ev.get('part'),
                                           'point_object_id': ev.get('point_object_id'), 's': time.perf_counter() - t}
    entry['silhouette_points_removed_from_cloud'] = removed
    out[label] = entry
Path(sys_argv[1]).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
