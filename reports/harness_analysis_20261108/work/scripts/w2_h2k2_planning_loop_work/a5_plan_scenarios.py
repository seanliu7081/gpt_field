"""A5: keypose-2 planner on saved T04 public inputs with plausible Astra keyposes.

Keypose positions follow the H3 live run's own Astra proposals for T04
(runs/foundation_vision_t04_can_obstacle_20261105_001/trial/requests 000001-000006),
top-down orientation = the measured start orientation. Payload = the keypose-2
close hypothesis (white_mug planning surface in the close-pose gripper frame,
motion.py:532-551). Scenarios:
  S1 pregrasp above white mug (transit)
  S2 grasp descent 5.5 cm (contact) from pregrasp
  S3 transport over the can at H3's height (z 0.65) with payload
  S4 transport over the can 6 cm lower (z 0.59) with payload -- the H1-style metric error case
  S5 = S4 with the wrist view removed (can seen only by external camera, as when the
     wrist view is occluded by the held mug / looks elsewhere)
  S6 grasp keypose 4 cm too low (palm/fingers into mug & table)
Also times _configuration, check_motion (per-step), plan.
"""
import json
from time import perf_counter

import numpy as np

from k2common import WORK, scene_at, kin, truth_boxes, restore, records  # noqa: F401
from gpt_field.contact_skills import matrix_to_quaternion_xyzw
from gpt_field.foundation_harness.feedback import grip_pose
from gpt_field.foundation_harness.motion import CandidatePlanner, PlanningFailure
from gpt_field.foundation_harness.surface import planning_surface
from gpt_field.foundation_harness.types import SceneSnapshot

STEP = 13
scene, state, q, record, err = scene_at(STEP)
R0 = grip_pose(state)[:3, :3]
xyzw = matrix_to_quaternion_xyzw(R0)
wxyz = [float(xyzw[3]), *map(float, xyzw[:3])]
T = truth_boxes()


def kp(name, xyz, gripper='keep', profile='transit', observe=False):
    return {'name': name, 'pose': {'position': list(map(float, xyz)), 'quaternion_wxyz': wxyz},
            'gripper': gripper, 'profile': profile, 'observe_after': observe}


def op(kind, obj, target, keyposes, keep_upright=False):
    return {'kind': kind, 'object': obj, 'target': target, 'grasp_region': None,
            'contact_mode': 'free_object', 'keep_upright': keep_upright,
            'placement_relation': 'on' if kind in ('pick_place', 'place') else None,
            'goal_pose': None, 'displacement': None, 'approach_direction': None, 'keyposes': keyposes}


def wrist_blind(mode):
    """Rebuild the same saved frame with the WRIST depth invalidated
    ('can': only pixels whose wrist ray lands within the can box +2 cm, i.e. the
    wrist does not see the can; 'all': whole wrist depth, i.e. wrist view filled
    by a held object). External view untouched. Mirrors restore() otherwise."""
    from scipy.spatial import cKDTree
    from gpt_field.foundation_vision.types import CameraFrame, VisionFrame, DepthEstimate
    from gpt_field.foundation_vision.geometry import build_geometry
    from gpt_field.foundation_harness.types import ObjectGeometry
    from k2common import V
    data = np.load(V / record['arrays'])
    cameras, depths = [], {}
    lo, hi = T['can']
    for c in record['array_metadata']['cameras']:
        p, name = c['array_prefix'], c['view']
        depth = data[p + '_depth_m'].astype(float).copy()
        if name == 'wrist':
            if mode == 'all':
                depth[:] = np.nan
            else:
                K = data[p + '_depth_intrinsics']; E = data[p + '_world_to_camera']
                rows, cols = np.nonzero(np.isfinite(depth) & (depth > 0))
                pix = np.column_stack((cols, rows, np.ones(len(rows))))
                cam_pts = (pix @ np.linalg.inv(K).T) * depth[rows, cols, None]
                world = (cam_pts - E[:3, 3]) @ E[:3, :3]
                hit = np.all((world >= lo - 0.02) & (world <= hi + 0.02), axis=1)
                depth[rows[hit], cols[hit]] = np.nan
        cameras.append(CameraFrame(name, data[p + '_rgb'], data[p + '_intrinsics'], data[p + '_world_to_camera'], 0.))
        depths[name] = DepthEstimate(name, depth.astype(np.float32), data[p + '_depth_intrinsics'], data[p + '_world_to_camera'])
    frame = VisionFrame(record['snapshot']['frame_id'], 0., tuple(cameras))
    geom = build_geometry(frame, depths, {})
    objects = {}
    for i, obj in enumerate(record['snapshot']['objects']):
        objects[obj['id']] = ObjectGeometry(obj['id'], obj['query'], data[f'object_{i}_points'], data[f'object_{i}_normals'], obj['status'], evidence=obj['evidence'])
    clouds, slices, offset, trees = [], {}, 0, {}
    for name, view in geom._views.items():
        clouds.append(view.points_world); slices[name] = [offset, offset + len(view.points_world)]
        trees[name] = cKDTree(view.points_world) if len(view.points_world) else None
        offset += len(view.points_world)
    cloud = np.concatenate(clouds)
    owners = np.full(len(cloud), '', object)
    for key, obj in objects.items():
        for part in obj.evidence.get('point_support_ranges', []):
            pts = obj.points[part['start']:part['end']]
            if not len(pts) or trees.get(part['view']) is None:
                continue
            dist, idx = trees[part['view']].query(pts)
            ok = dist <= 1e-6
            pos = slices[part['view']][0] + idx[ok]
            owners[pos[owners[pos] == '']] = key
    ev = {**record['snapshot']['evidence'], 'scene_point_view_slices': slices}
    return SceneSnapshot(frame.frame_id, 0., 0., objects, cloud, geom, frame, evidence=ev, scene_point_object_ids=owners)


def payload_from_close(scene_in, close_xyz):
    obj = scene_in.objects.get('white_mug')
    pts, ev = planning_surface(obj)
    pose = np.eye(4); pose[:3, :3] = R0; pose[:3, 3] = close_xyz
    local = (np.asarray(pts) - pose[:3, 3]) @ pose[:3, :3]
    return {'object_id': 'white_mug', 'owner_ids': ['white_mug'], 'points_in_gripper': local.tolist(),
            'confirmed': False, 'source': 'astra_close_visible_surface_hypothesis', 'geometry_unknown': False}, ev


def solve(xyz, seed):
    r = None
    pl = CandidatePlanner(kin)
    pl._prepare(scene, state, q)
    r = pl.bound.solve(np.asarray(xyz), R0, seed, max_iterations=80)
    return r


def run(label, scene_in, operation, joints, st, held=None):
    planner = CandidatePlanner(kin)
    # The saved images were captured with the robot at the measured home pose:
    # self-mask with THAT pose (scene-key cache keeps it), then plan from `joints`.
    planner._prepare(scene_in, state, q)
    t0 = perf_counter()
    try:
        plan = planner.plan(operation, scene_in, st, joints, held=held)
        out = {'case': label, 'planned': True, 'path_points': len(plan.joint_paths[0]),
               'uncertain_obstacle_samples': plan.evidence['uncertain_obstacle_samples'],
               'uncertain_examples': plan.evidence['uncertain_obstacle_examples'],
               'unknown_sample_visits': plan.evidence['unknown_sample_visits'],
               'payload_geometry_unknown': plan.evidence['payload_geometry_unknown']}
        end_q = plan.joint_paths[0][-1]
    except PlanningFailure as exc:
        ev = exc.evidence.get('evidence', exc.evidence)
        out = {'case': label, 'planned': False, 'reason': exc.reason,
               'part': ev.get('part'), 'point_world_m': ev.get('point_world_m'),
               'point_object_id': ev.get('point_object_id'), 'endpoint': ev.get('endpoint'),
               'obstacle_evidence': ev.get('obstacle_evidence')}
        end_q = None
    out['planning_s'] = round(perf_counter() - t0, 3)
    return out, end_q, planner


results = []
mug = T['porcelain_mug_1_origin']
# S1 pregrasp above white mug (H3 call 2: (-0.085,-0.1,0.61))
pre = np.array([-0.085, -0.10, 0.61])
grasp = np.array([-0.085, -0.10, 0.555])
s1, q_pre, pl1 = run('S1_pregrasp_transit', scene, op('pick_place', 'white_mug', 'seed_1', [kp('pregrasp', pre)]), q, state)
results.append(s1)
# synthesize the measured state at pregrasp (FK of planned end)
def state_at(qq):
    b = kin.bind(state, q)
    pose, _ = b.forward(np.asarray(qq))
    from gpt_field.robodojo_direct_sim import rotation_vector
    return np.r_[pose[:3, 3], rotation_vector(pose[:3, :3]), .04, -.04]
st_pre = state_at(q_pre) if q_pre is not None else state
q_pre = np.asarray(q_pre) if q_pre is not None else q
# S2 grasp descent (contact) from pregrasp, keyposes [pregrasp, grasp], index 1
ops2 = op('pick_place', 'white_mug', 'seed_1', [kp('pregrasp', pre), kp('grasp', grasp, profile='contact')])
ops2['_keypose_index'] = 1
s2, q_grasp, _ = run('S2_grasp_descent_contact', scene, ops2, q_pre, st_pre)
results.append(s2)
# payload as keypose-2 would form it at close
held, sev = payload_from_close(scene, grasp)
q_grasp = np.asarray(q_grasp) if q_grasp is not None else q_pre
st_g = state_at(q_grasp); st_g[6:8] = [0.012, -0.012]
lift = grasp + [0, 0, 0.095]   # to z 0.65 (H3: lift +3 cm then move up to 0.65)
over_can_hi = np.array([-0.025, -0.275, 0.65])
over_can_lo = np.array([-0.025, -0.275, 0.59])
mid_hi = np.array([-0.05, -0.19, 0.65]); mid_lo = np.array([-0.05, -0.19, 0.59])
for label, sc in (('both_views', scene), ('wrist_blind_to_can', 'can'), ('wrist_depth_absent', 'all')):
    if isinstance(sc, str):
        sc = wrist_blind(sc)
    for name, target in (('S3_transport_z0.65', over_can_hi), ('S4_transport_z0.59', over_can_lo),
                         ('S3b_transport_mid_z0.65', mid_hi), ('S4b_transport_mid_z0.59', mid_lo)):
        o = op('pick_place', 'white_mug', 'seed_1', [kp('transport', target)])
        r, _, _ = run(f'{name}_{label}', sc, o, q_grasp, st_g, held=held)
        results.append(r)
# S6 grasp keypose 4 cm too low (z 0.515) from pregrasp
ops6 = op('pick_place', 'white_mug', 'seed_1', [kp('pregrasp', pre), kp('grasp', grasp - [0, 0, 0.04], profile='contact')])
ops6['_keypose_index'] = 1
s6, _, _ = run('S6_grasp_4cm_too_low', scene, ops6, q_pre, st_pre)
results.append(s6)
# S7 grasp keypose 8 cm too low (z 0.475, fingertips ~table) from pregrasp
ops7 = op('pick_place', 'white_mug', 'seed_1', [kp('pregrasp', pre), kp('grasp', grasp - [0, 0, 0.08], profile='contact')])
ops7['_keypose_index'] = 1
s7, _, _ = run('S7_grasp_8cm_too_low', scene, ops7, q_pre, st_pre)
results.append(s7)

# timing: _configuration and per-step check_motion
pl = CandidatePlanner(kin); pl._prepare(scene, state, q)
pl._deadline = float('inf')
t0 = perf_counter(); n = 30
for i in range(n):
    pl._configuration(q + 0.001 * i, scene, state[6:8], None, ())
conf_ms = (perf_counter() - t0) / n * 1000
t0 = perf_counter()
for i in range(n):
    pl._configuration(q_grasp, scene, st_g[6:8], held, ('white_mug',))
conf_payload_ms = (perf_counter() - t0) / n * 1000
t0 = perf_counter()
for i in range(10):
    pl.check_motion(q, q + 0.025, scene, state)
check_ms = (perf_counter() - t0) / 10 * 1000
timing = {'configuration_ms_empty_hand': round(conf_ms, 1), 'configuration_ms_with_payload': round(conf_payload_ms, 1),
          'check_motion_ms_0.025rad_edge': round(check_ms, 1)}
out = {'step': STEP, 'payload_surface_points': len(held['points_in_gripper']), 'payload_surface_evidence_status': sev.get('status'),
       'results': results, 'timing': timing}
print(json.dumps(out, indent=1))
(WORK / 'data/a5_plan_scenarios.json').write_text(json.dumps(out, indent=1))
