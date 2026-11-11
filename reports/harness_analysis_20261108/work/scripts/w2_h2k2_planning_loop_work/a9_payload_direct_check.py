"""A9: direct keypose-2 check of a held payload at its close pose (lift/transport start).

Uses the planner's own code path: _astra_keypose() forms the close payload
(motion.py:532-551), then _configuration() evaluates the robot + payload at the
close configuration for the NEXT keypose names 'lift' (contact parts finger/pad)
and 'transport' (no contact parts). Scenes: saved run-001 revisions where the
mug planning surface is coherent. Robot image is at home (offline limitation),
so only the payload part matters here.
"""
import json

import numpy as np

from k2common import WORK, records, restore, kin, events  # noqa: F401
from gpt_field.contact_skills import matrix_to_quaternion_xyzw
from gpt_field.foundation_harness.feedback import grip_pose
from gpt_field.foundation_harness.motion import CandidatePlanner
from gpt_field.foundation_harness.surface import planning_surface

out = []
for arrays, obj_id, xyz in (('frame_000020_revision_000044.npz', 'white_mug', None),
                            ('frame_000005_revision_000011.npz', 'yellow_white_mug', None)):
    rec = next(r for r in records if r['arrays'] == arrays)
    scene, err = restore(rec)  # noqa: F821
    step = rec['snapshot']['evidence']['observation_index']
    fb = next(e for e in events if e['event'] == 'feedback' and e['step'] == max(step, 1))
    state = np.asarray(fb['measured_state']); q = np.asarray(fb['joint_position'])
    obj = scene.objects[obj_id]
    pts, ev = planning_surface(obj)
    top = pts[pts[:, 2] > np.quantile(pts[:, 2], 0.9)]
    # top-down rim grasp: grip site at a rim point, 1 cm below the rim top
    rim = top[np.argmax(np.linalg.norm(top[:, :2] - pts[:, :2].mean(0), axis=1))]
    grip = rim - [0, 0, 0.01]
    R0 = grip_pose(state)[:3, :3]
    xyzw = matrix_to_quaternion_xyzw(R0); wxyz = [float(xyzw[3]), *map(float, xyzw[:3])]
    pl = CandidatePlanner(kin)
    pl._prepare(scene, state, q)
    res = pl.bound.solve(grip, R0, q, max_iterations=80)
    qc = np.asarray(res.joints)
    operation = {'kind': 'pick_place', 'object': obj_id, 'target': 'seed_1', 'grasp_region': None,
                 'contact_mode': 'free_object', 'keep_upright': False, 'placement_relation': 'on',
                 'goal_pose': None, 'displacement': None, 'approach_direction': None,
                 'keyposes': [{'name': 'close', 'pose': {'position': grip.tolist(), 'quaternion_wxyz': wxyz},
                               'gripper': 'close', 'profile': 'contact', 'observe_after': False}], '_keypose_index': 0}
    kp = pl._astra_keypose(operation, scene, state)
    payload = kp.metadata['payload_after']
    pl._deadline = float('inf')
    rows = {'arrays': arrays, 'object': obj_id, 'ik_ok': bool(res.ok), 'payload_points': len(payload['points_in_gripper']),
            'payload_source': payload['source']}
    for name, parts, contacts in (('lift', ('finger', 'pad'), (obj_id,)), ('transport', (), ())):
        pl._contact_parts = parts
        r = pl._configuration(qc, scene, np.array([0.01, -0.01]), payload, contacts, free_check=False)
        rows[name] = {k: r.get(k) for k in ('safe', 'reason', 'part', 'point_world_m', 'point_object_id', 'obstacle_evidence')}
    out.append(rows)
    print(json.dumps(rows))
(WORK / 'data/a9_payload_direct_check.json').write_text(json.dumps(out, indent=1))

# --- geometry detail for the second case (hand vs payload hull) ---
from gpt_field.foundation_harness.robot_geometry import intersect, CollisionShape
from gpt_field.foundation_harness.motion import _payload_shape
detail = []
for arrays, obj_id in (('frame_000005_revision_000011.npz', 'yellow_white_mug'), ('frame_000020_revision_000044.npz', 'white_mug')):
    rec = next(r for r in records if r['arrays'] == arrays)
    scene, err = restore(rec)  # noqa: F821
    step = rec['snapshot']['evidence']['observation_index']
    fb = next(e for e in events if e['event'] == 'feedback' and e['step'] == max(step, 1))
    state = np.asarray(fb['measured_state']); q = np.asarray(fb['joint_position'])
    pts, ev = planning_surface(scene.objects[obj_id])
    top = pts[pts[:, 2] > np.quantile(pts[:, 2], 0.9)]
    rim = top[np.argmax(np.linalg.norm(top[:, :2] - pts[:, :2].mean(0), axis=1))]
    grip = rim - [0, 0, 0.01]
    R0 = grip_pose(state)[:3, :3]
    pl = CandidatePlanner(kin); pl._prepare(scene, state, q)
    qc = np.asarray(pl.bound.solve(grip, R0, q, max_iterations=80).joints)
    shapes = pl.robot.shapes(pl.bound, qc, np.array([0.01, -0.01]))
    hand = next(s for s in shapes if s.name == 'hand_collision')
    pose = np.eye(4); pose[:3, :3] = R0; pose[:3, 3] = grip
    local = (pts - grip) @ R0
    shp = _payload_shape(local)
    pay = CollisionShape(shp.name, shp.group, shp.vertices, shp.equations, pose, 'payload')
    hv = hand.surface_points()
    detail.append({'object': obj_id, 'grip_xyz': grip.round(4).tolist(), 'rim_point': rim.round(4).tolist(),
                   'planning_center_xy': pts[:, :2].mean(0).round(4).tolist(),
                   'payload_top_z': float(pts[:, 2].max()), 'hand_bottom_z': float(hv[:, 2].min()),
                   'hand_payload_intersect_margin6mm': bool(intersect(pay, hand, 0.006)),
                   'hand_payload_intersect_margin0': bool(intersect(pay, hand, 0.0))})
print(json.dumps(detail, indent=1))
(WORK / 'data/a9_payload_hand_detail.json').write_text(json.dumps(detail, indent=1))
