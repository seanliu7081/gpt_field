"""In-flight H2 'keypose-2' (snapshot 23:48) on the saved T04 arrays.

Scratch copy of the package only (pkg_v2); no edits under /workspace/gpt_field.
Questions: does the v2 self-filter remove robot ghost points; do the saved
'move' requests and plausible model-style pregrasp/grasp keyposes connect;
what does planning_surface do to the inflated mug clouds; per-check cost.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
PKG = HERE.parent / 'pkg_v2'
RUN = Path('/workspace/gpt_field/runs/foundation_harness_t04_can_obstacle_001')
sys.path.insert(0, str(PKG))
from gpt_field.foundation_vision.types import CameraFrame, VisionFrame, DepthEstimate  # noqa: E402
from gpt_field.foundation_vision.geometry import build_geometry  # noqa: E402
from gpt_field.foundation_harness.types import SceneSnapshot, ObjectGeometry  # noqa: E402
from gpt_field.foundation_harness import motion as m2  # noqa: E402
from gpt_field.foundation_harness.surface import planning_surface  # noqa: E402
from gpt_field.robodojo_direct_sim import PandaKinematics, matrix_quaternion  # noqa: E402

assert 'Astra key poses' in (m2.__doc__ or ''), 'not the v2 motion module'
V = RUN / 'trial/foundation_harness/vision'
records = [json.loads(l) for l in (V / 'snapshots.jsonl').read_text().splitlines()]
events = [json.loads(l) for l in (RUN / 'trial/foundation_harness/events.jsonl').read_text().splitlines()]
robot = json.loads((RUN / 'robot_kinematics.json').read_text())
kin = PandaKinematics.from_xml(robot['robot_xml'], robot['gripper_xml'])

# restore() identical to the run's reconstruction, with view slices kept so
# the v2 ray filter can identify each point's source camera.
def restore(record):
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
        clouds.append(view.points_world); slices[name] = offset
        trees[name] = cKDTree(view.points_world); offset += len(view.points_world)
    cloud = np.concatenate(clouds)
    owners = np.full(len(cloud), '', object)
    for key, obj in objects.items():
        for part in obj.evidence.get('point_support_ranges', []):
            points = obj.points[part['start']:part['end']]
            if len(points):
                _, idx = trees[part['view']].query(points)
                pos = slices[part['view']] + idx
                owners[pos[owners[pos] == '']] = key
    evidence = dict(record['snapshot']['evidence'])
    scene = SceneSnapshot(frame.frame_id, 0., 0., objects, cloud, geom, frame,
                          evidence=evidence, scene_point_object_ids=owners)
    return scene


def step_inputs(step):
    fb = next(x for x in events if x['event'] == 'feedback' and x['step'] == step)
    rec = next(x for x in records if x['snapshot']['evidence']['observation_index'] == step)
    return rec, np.array(fb['measured_state']), np.array(fb['joint_position'])


def wxyz(R):
    q = matrix_quaternion(R)
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def plan(planner, op, scene, state, q):
    t = time.perf_counter()
    try:
        p = planner.plan(op, scene, state, q)
        return {'ok': True, 'waypoints': len(p.joint_paths[0]), 's': time.perf_counter() - t,
                'unknown_sample_visits': p.evidence.get('unknown_sample_visits'),
                'self_filter': p.evidence.get('self_filter', {}).get('removed_points')}
    except m2.PlanningFailure as exc:
        ev = exc.evidence.get('evidence', {}) if isinstance(exc.evidence, dict) else {}
        return {'ok': False, 'reason': exc.reason, 'part': ev.get('part'), 'point_object_id': ev.get('point_object_id'),
                'point_world_m': ev.get('point_world_m'), 's': time.perf_counter() - t}


out = {}
# 1. self filter and the saved move requests
for step, disp in ((9, [0, -.15, .05]), (13, [0, -.05, 0])):
    rec, state, q = step_inputs(step)
    scene = restore(rec)
    planner = m2.CandidatePlanner(kin)
    planner._prepare(scene, state, q)
    start = planner._configuration(q, scene, state[6:8], None)
    op = {'kind': 'move', 'object': None, 'target': None, 'grasp_region': None, 'keep_upright': False,
          'goal_pose': None, 'displacement': disp, 'keyposes': [], 'contact_mode': 'free_object'}
    out[f'move_step{step}'] = {'self_filter': planner._self_filter_evidence, 'start_configuration': start,
                               'plan': plan(m2.CandidatePlanner(kin), op, scene, state, q)}

# 2. plausible model-style keyposes on the white mug (frame of step 18)
rec, state, q = step_inputs(18)
scene = restore(rec)
mug = scene.objects['white_mug']
pts, ev = planning_surface(mug)
raw_ext = np.ptp(mug.points, 0)
surface_report = {'raw_points': int(len(mug.points)), 'raw_extent_m': raw_ext.round(4).tolist(),
                  'planning_points': int(len(pts)), 'status': ev.get('status'), 'reason': ev.get('reason'),
                  'planning_extent_m': np.ptp(pts, 0).round(4).tolist() if len(pts) else None,
                  'planning_centroid_m': pts.mean(0).round(4).tolist() if len(pts) else None,
                  'components': ev.get('component_count'), 'mass_fraction': ev.get('selected_mass_fraction'),
                  'extraction_s': ev.get('extraction_s')}
for other in ('yellow_white_mug',):
    rec4, _, _ = step_inputs(4)
    o = restore(rec4).objects[other]
    p4, e4 = planning_surface(o)
    surface_report[other] = {'raw_extent_m': np.ptp(o.points, 0).round(4).tolist(),
                             'planning_extent_m': np.ptp(p4, 0).round(4).tolist() if len(p4) else None,
                             'status': e4.get('status'), 'reason': e4.get('reason')}
out['planning_surface'] = surface_report
R_now = m2._pose(state)[:3, :3]
center = pts.mean(0) if len(pts) else np.median(mug.points, 0)
top = float(pts[:, 2].max()) if len(pts) else float(np.percentile(mug.points[:, 2], 95))
trials = {}
for label, position in (('pregrasp_12cm_above_top', [center[0], center[1], top + .12]),
                        ('pregrasp_6cm_above_top', [center[0], center[1], top + .06]),
                        ('grasp_rim_level', [center[0], center[1], top - .01])):
    op = {'kind': 'grasp', 'object': 'white_mug', 'target': None, 'grasp_region': None, 'keep_upright': True,
          'goal_pose': None, 'displacement': None, 'contact_mode': 'free_object', 'placement_relation': None,
          'keyposes': [{'name': 'pregrasp' if 'pregrasp' in label else 'grasp',
                        'pose': {'position': [float(x) for x in position], 'quaternion_wxyz': wxyz(R_now)},
                        'gripper': 'open' if 'pregrasp' in label else 'keep',
                        'profile': 'transit' if 'pregrasp' in label else 'contact', 'observe_after': True}]}
    trials[label] = {'position': [round(float(x), 4) for x in position], 'plan': plan(m2.CandidatePlanner(kin), op, scene, state, q)}
out['white_mug_model_style_keyposes_from_start'] = trials

# 3. cost of one v2 configuration check
planner = m2.CandidatePlanner(kin)
t = time.perf_counter(); planner._prepare(scene, state, q); prep = time.perf_counter() - t
planner._deadline = time.perf_counter() + 60
rng = np.random.default_rng(2)
ts = []
for _ in range(20):
    qq = q + rng.normal(0, .02, 7)
    t = time.perf_counter(); planner._configuration(qq, scene, state[6:8], None); ts.append(time.perf_counter() - t)
out['v2_costs_s'] = {'prepare_s': prep, 'configuration_mean_s': float(np.mean(ts)), 'configuration_max_s': float(np.max(ts))}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))
print(json.dumps(out, indent=1, default=str)[:6000])
