"""Offline counterfactual checks of the launch-frozen H2 planner on saved arrays.

Read-only with respect to /workspace/gpt_field: imports the frozen source copy
under the run's source_at_launch/, loads saved npz perception arrays, runs
numpy/scipy geometry only. No simulator, model, network or GPU use.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

RUN = Path('/workspace/gpt_field/runs/foundation_harness_t04_can_obstacle_001')
OUT = Path(sys.argv[1])
sys.path.insert(0, str(RUN / 'source_at_launch/src'))
from gpt_field.foundation_vision.types import CameraFrame, VisionFrame, DepthEstimate  # noqa: E402
from gpt_field.foundation_vision.geometry import build_geometry  # noqa: E402
from gpt_field.foundation_harness.types import SceneSnapshot, ObjectGeometry  # noqa: E402
from gpt_field.foundation_harness.motion import CandidatePlanner, MotionConfig, PlanningFailure, _pose  # noqa: E402
from gpt_field.robodojo_direct_sim import PandaKinematics  # noqa: E402

V = RUN / 'trial/foundation_harness/vision'
records = [json.loads(l) for l in (V / 'snapshots.jsonl').read_text().splitlines()]
events = [json.loads(l) for l in (RUN / 'trial/foundation_harness/events.jsonl').read_text().splitlines()]
robot = json.loads((RUN / 'robot_kinematics.json').read_text())
kin = PandaKinematics.from_xml(robot['robot_xml'], robot['gripper_xml'])


def restore(record):
    """Same reconstruction as the run's analysis/reconstruct_rejections.py."""
    data = np.load(V / record['arrays'])
    cameras, depths = [], {}
    for c in record['array_metadata']['cameras']:
        p, name = c['array_prefix'], c['view']
        cameras.append(CameraFrame(name, data[p + '_rgb'], data[p + '_intrinsics'], data[p + '_world_to_camera'], 0.))
        depths[name] = DepthEstimate(name, data[p + '_depth_m'], data[p + '_depth_intrinsics'], data[p + '_world_to_camera'])
    frame = VisionFrame(record['snapshot']['frame_id'], 0., tuple(cameras))
    geom = build_geometry(frame, depths, {})
    objects = {}
    for i, obj in enumerate(record['snapshot']['objects']):
        objects[obj['id']] = ObjectGeometry(obj['id'], obj['query'], data[f'object_{i}_points'].astype(float),
                                            data[f'object_{i}_normals'].astype(float), obj['status'], evidence=obj['evidence'])
    clouds, slices, offset, trees = [], {}, 0, {}
    for name, view in geom._views.items():
        clouds.append(view.points_world)
        slices[name] = offset
        trees[name] = cKDTree(view.points_world)
        offset += len(view.points_world)
    cloud = np.concatenate(clouds)
    owners = np.full(len(cloud), '', object)
    ambiguous = np.zeros(len(cloud), bool)
    parents = {k: v.evidence.get('parent_id') for k, v in objects.items()}

    def ancestors(k):
        result = set()
        while parents.get(k) is not None:
            k = parents[k]
            result.add(k)
        return result
    ancestry = {k: ancestors(k) for k in objects}
    for key, obj in objects.items():
        for part in obj.evidence.get('point_support_ranges', []):
            points = obj.points[part['start']:part['end']]
            if not len(points):
                continue
            distances, indices = trees[part['view']].query(points)
            positions = slices[part['view']] + indices
            existing = owners[positions].copy()
            assign = (existing == '') & ~ambiguous[positions]
            for owner in np.unique(existing[existing != '']):
                same = existing == owner
                if owner == key or owner in ancestry[key]:
                    continue
                if key in ancestry[owner]:
                    assign |= same
                else:
                    ambiguous[positions[same]] = True
            owners[positions[assign]] = key
    owners[ambiguous] = ''
    scene = SceneSnapshot(frame.frame_id, 0., 0., objects, cloud, geom, frame,
                          evidence=record['snapshot']['evidence'], scene_point_object_ids=owners)
    return scene, data


def step_inputs(step):
    feedback = next(x for x in events if x['event'] == 'feedback' and x['step'] == step)
    record = next(x for x in records if x['snapshot']['evidence']['observation_index'] == step)
    return record, np.array(feedback['measured_state']), np.array(feedback['joint_position'])


def residual(shape, points):
    local = (np.asarray(points) - shape.pose[:3, 3]) @ shape.pose[:3, :3]
    return np.max(local @ shape.equations[:, :3].T + shape.equations[:, 3], axis=1)


def drop_base_points(planner):
    """CF1: emulate 'fixed base excluded from scene obstacle checks'."""
    base = next(s for s in planner._initial_shapes if s.name == 'link0_collision')
    near = residual(base, planner._cloud) <= planner.config.collision_margin_m + 1e-4
    planner._cloud, planner._owners = planner._cloud[~near], planner._owners[~near]
    planner._tree = cKDTree(planner._cloud)
    return int(near.sum())


def probe_breakdown(planner, scene, q, fingers, held=None, allowed=()):
    shapes = planner.robot.shapes(planner.bound, q, fingers)
    probes = np.concatenate([s.surface_points() for s in shapes])
    if len(probes) > planner.config.free_probe_limit:
        probes = probes[np.linspace(0, len(probes) - 1, planner.config.free_probe_limit, dtype=int)]
    known = planner._known_start_volume(probes)
    rows = scene.geometry.query(probes[~known], now_s=0., max_age_s=5.)
    counts = {}
    for row in rows:
        key = row['state'] + ':' + row['reason']
        counts[key] = counts.get(key, 0) + 1
    return {'probes': int(len(probes)), 'known_start_volume': int(known.sum()), 'states': counts}


results = {}

# ---------------------------------------------------------------- E0/E1
# Robot-surface DA3 error scale and the 3 mm / 6 mm margin mismatch.
e1 = []
for step in (9, 13, 18):
    record, state, q = step_inputs(step)
    scene, _ = restore(record)
    planner = CandidatePlanner(kin)
    t0 = time.perf_counter(); planner._prepare(scene, state, q); prep_s = time.perf_counter() - t0
    raw = np.asarray(scene.scene_points); raw = raw[np.isfinite(raw).all(1)]
    shell = {}
    for shape in planner._initial_shapes:
        # raw (pre-filter) points around each robot hull, by residual band
        tree = cKDTree(raw) if 'tree' not in locals() else tree
        ids = np.asarray(tree.query_ball_point(shape.center, shape.radius + .05), int)
        if not len(ids):
            continue
        r = residual(shape, raw[ids])
        r = r[r <= .05]
        shell[shape.name] = {'within_0': int((r <= 0).sum()), 'in_0_3mm': int(((r > 0) & (r <= .003)).sum()),
                             'in_3_6mm': int(((r > .003) & (r <= .006)).sum()),
                             'in_6_20mm': int(((r > .006) & (r <= .02)).sum()),
                             'in_20_50mm': int(((r > .02) & (r <= .05)).sum())}
    del tree
    remaining_in_margin = {}
    for shape in planner._initial_shapes:
        if planner._tree is None:
            break
        ids = np.asarray(planner._tree.query_ball_point(shape.center, shape.radius + .006), int)
        if len(ids):
            n = int(shape.contains(planner._cloud[ids], .006).sum())
            if n:
                remaining_in_margin[shape.name] = n
    start = planner._configuration(q, scene, state[6:8], None)
    e1.append({'step': step, 'frame': scene.frame_id, 'prepare_s': prep_s, 'scene_points': int(len(raw)),
               'self_removed_3mm': planner._self_removed, 'raw_points_by_hull_residual_band': shell,
               'kept_points_inside_6mm_margin_at_start': remaining_in_margin, 'start_configuration': start})
results['E1_start_state_margin'] = e1

# ---------------------------------------------------------------- E2
# Moves: what blocks them once the fixed-base hit is removed (CF1)?
e2 = []
for step, displacement in ((9, [0, -.15, .05]), (13, [0, -.05, 0])):
    record, state, q = step_inputs(step)
    scene, _ = restore(record)
    for allow_unknown in (False, True):
        planner = CandidatePlanner(kin, MotionConfig(allow_unknown_space=allow_unknown))
        planner._prepare(scene, state, q)
        dropped = drop_base_points(planner)
        goal = _pose(state); goal[:3, 3] += displacement
        ik = planner.bound.solve(goal[:3, 3], goal[:3, :3], q, max_iterations=80)
        planner._deadline = time.perf_counter() + 12.
        t0 = time.perf_counter()
        edge = planner._edge(q, np.asarray(ik.joints), scene, state[6:8], None, ())
        edge_s = time.perf_counter() - t0
        # breakdown at the first interior sample of the straight edge
        count = max(1, int(np.ceil(np.max(np.abs(np.asarray(ik.joints) - q)) / planner.config.max_joint_step_rad)))
        q1 = np.linspace(q, np.asarray(ik.joints), count + 1)[1]
        breakdown = probe_breakdown(planner, scene, q1, state[6:8])
        goal_breakdown = probe_breakdown(planner, scene, np.asarray(ik.joints), state[6:8])
        plan_result = None
        t1 = time.perf_counter()
        try:
            op = {'kind': 'move', 'displacement': displacement, 'goal_pose': None, 'keep_upright': False,
                  'object': None}
            # full plan() re-prepares (re-adds base points), so emulate CF1 by
            # monkeypatching _prepare to drop base points after the original.
            original = planner._prepare

            def patched(scene_, state_, joints_, _orig=original, _p=planner):
                _p._scene_key = None
                _orig(scene_, state_, joints_)
                drop_base_points(_p)
            planner._prepare = patched
            plan = planner.plan(op, scene, state, q)
            plan_result = {'ok': True, 'waypoints': len(plan.joint_paths[0]), 'planning_s': plan.evidence['planning_s']}
        except PlanningFailure as exc:
            plan_result = {'ok': False, 'reason': exc.reason,
                           'inner': [r.get('reason') for r in exc.evidence.get('rejections', [])],
                           'inner_evidence': [r.get('evidence', {}).get('reason') for r in exc.evidence.get('rejections', [])]}
        e2.append({'step': step, 'displacement': displacement, 'allow_unknown_space': allow_unknown,
                   'base_points_dropped': dropped, 'ik_ok': bool(ik.ok), 'ik_err_m': ik.position_error_m,
                   'edge_samples': count + 1, 'straight_edge_check': edge, 'edge_check_s': edge_s,
                   'first_interior_sample_probe_states': breakdown, 'goal_probe_states': goal_breakdown,
                   'full_plan_cf1': plan_result, 'full_plan_s': time.perf_counter() - t1})
results['E2_moves_cf_base_removed'] = e2

# ---------------------------------------------------------------- E3
# Workspace free-space coverage for the v1 'estimated_free' rule.
record, state, q = step_inputs(18)
scene, _ = restore(record)
plate = scene.objects['seed_1'].points
table_z = float(np.percentile(plate[:, 2], 5))
xs = np.arange(-0.30, 0.16, 0.02); ys = np.arange(-0.40, 0.41, 0.02); zs = np.arange(table_z + .02, table_z + .36, 0.02)
grid = np.array([[x, y, z] for x in xs for y in ys for z in zs])
rows = scene.geometry.query(grid, now_s=0., max_age_s=5.)
cov = {}
for row in rows:
    key = row['state'] + ':' + row['reason']
    cov[key] = cov.get(key, 0) + 1
by_height = {}
for z in zs:
    sel = [r for r, p in zip(rows, grid) if abs(p[2] - z) < 1e-9]
    by_height[f'{z - table_z:.2f}'] = round(sum(r['state'] == 'estimated_free' for r in sel) / len(sel), 3)
results['E3_free_space_coverage_frame19'] = {'grid_points': int(len(grid)), 'spacing_m': .02,
    'box': {'x': [float(xs[0]), float(xs[-1])], 'y': [float(ys[0]), float(ys[-1])],
            'z_above_table': [.02, .36], 'table_z_est': table_z},
    'states': cov, 'estimated_free_fraction_by_height_above_table': by_height}

# ---------------------------------------------------------------- E4
# Grasp proposals + payload/support geometry of the actual pick_place calls.
e4 = []
for step, object_id, target_id, upright in ((4, 'yellow_white_mug', 'seed_3', True), (16, 'white_mug', 'seed_1', True),
                                            (18, 'white_mug', 'seed_1', False)):
    record, state, q = step_inputs(step)
    scene, _ = restore(record)
    planner = CandidatePlanner(kin)
    planner._prepare(scene, state, q)
    dropped = drop_base_points(planner)
    planner._deadline = time.perf_counter() + 12.
    obj = scene.objects[object_id]; target = scene.objects[target_id]
    pts = obj.points
    supported = target.points[np.abs(target.normals[:, 2]) > .85]
    ext_full = np.ptp(pts, 0)
    ext_p = np.percentile(pts, 95, 0) - np.percentile(pts, 5, 0)
    ext_p2 = np.percentile(pts, 98, 0) - np.percentile(pts, 2, 0)
    support_z = float(np.median(supported[:, 2])) if len(supported) else None
    entry = {'step': step, 'object': object_id, 'object_points': int(len(pts)),
             'object_extent_full_m': ext_full.round(4).tolist(), 'object_extent_p5_p95_m': ext_p.round(4).tolist(),
             'object_extent_p2_p98_m': ext_p2.round(4).tolist(),
             'object_min_z': float(pts[:, 2].min()), 'object_p2_z': float(np.percentile(pts[:, 2], 2)),
             'support_points_upward': int(len(supported)), 'support_extent_m': np.ptp(supported, 0).round(4).tolist() if len(supported) else None,
             'support_median_z': support_z,
             'object_points_below_support_z_minus_1cm': int((pts[:, 2] < (support_z - .01)).sum()) if support_z else None,
             'normals_nonzero': int((np.linalg.norm(obj.normals, axis=1) > .5).sum())}
    op = {'kind': 'pick_place', 'object': object_id, 'target': target_id, 'grasp_region': None,
          'keep_upright': upright, 'approach_direction': None, 'placement_relation': 'on', 'contact_mode': 'free_object'}
    try:
        grasps = planner._grasps(obj, op)
    except PlanningFailure as exc:
        entry['grasps'] = exc.reason
        e4.append(entry)
        continue
    robust_center = np.median(pts, 0)
    glist = []
    for score, pose, fingers, contacts in grasps[:16]:
        c = np.asarray(contacts)
        width = float(np.linalg.norm(c[1] - c[0]))
        approach = pose[:3, 2]
        ik = planner.bound.solve(pose[:3, 3], pose[:3, :3], q, max_iterations=80)
        g = {'alignment': round(float(score), 3), 'width_m': round(width, 4), 'center_m': pose[:3, 3].round(3).tolist(),
             'approach': approach.round(2).tolist(), 'center_minus_object_median_m': (pose[:3, 3] - robust_center).round(3).tolist(),
             'center_height_above_support_m': round(float(pose[2, 3] - support_z), 3) if support_z else None,
             'ik_ok': bool(ik.ok)}
        if ik.ok:
            qg = np.asarray(ik.joints)
            allowed = (object_id,)
            g['grasp_config_fingers_open'] = planner._configuration(qg, scene, np.array([.04, -.04]), None, allowed)
            g['grasp_config_fingers_closed_to_width'] = planner._configuration(qg, scene, np.asarray(fingers), None, allowed)
            g['grasp_config_open_free_check_off'] = planner._configuration(qg, scene, np.array([.04, -.04]), None, allowed, free_check=False)
            g['grasp_probe_states'] = probe_breakdown(planner, scene, qg, np.array([.04, -.04]))
        glist.append(g)
    entry['grasps'] = glist
    e4.append(entry)
results['E4_grasp_and_payload_geometry'] = e4

# ---------------------------------------------------------------- E5
# Cost of one configuration check (as used per action by check_motion).
record, state, q = step_inputs(18)
scene, _ = restore(record)
planner = CandidatePlanner(kin)
t0 = time.perf_counter(); planner._prepare(scene, state, q); prep = time.perf_counter() - t0
drop_base_points(planner)
planner._deadline = time.perf_counter() + 60
rng = np.random.default_rng(1)
times_free, times_nofree = [], []
for _ in range(20):
    qq = q + rng.normal(0, .02, 7)
    t = time.perf_counter(); planner._configuration(qq, scene, state[6:8], None); times_free.append(time.perf_counter() - t)
    t = time.perf_counter(); planner._configuration(qq, scene, state[6:8], None, free_check=False); times_nofree.append(time.perf_counter() - t)
t = time.perf_counter(); shapes = planner.robot.shapes(planner.bound, q, state[6:8]); sh = time.perf_counter() - t
t = time.perf_counter(); planner.robot.self_collision(shapes); sc = time.perf_counter() - t
results['E5_check_costs_s'] = {'prepare_new_frame_s': prep, 'configuration_with_free_check_mean_s': float(np.mean(times_free)),
                               'configuration_without_free_check_mean_s': float(np.mean(times_nofree)),
                               'robot_shapes_s': sh, 'self_collision_s': sc, 'n_shapes': len(shapes)}

OUT.write_text(json.dumps(results, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
print('wrote', OUT)
