"""Payload hull exactly as motion.py:_astra_keypose builds it on a close keypose (planning_surface -> gripper frame),
then the lift-start configuration check (motion.py:224-333) with held=payload and the lift allowances."""
import json, sys
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
from gpt_field.foundation_harness.motion import _payload_shape
from gpt_field.contact_skills import matrix_to_quaternion_xyzw
M = json.load(open('/workspace/gpt_field/output/foundation_harness_preparation_001/inputs/layout_manifest.json'))
poses = M['custom_after_passive_hold_object_poses']
out = []
for step, target, body, key in ((4, 'yellow_white_mug', 'white_yellow_mug_1', 'white_yellow_mug_1'),
                                (15, 'white_mug', 'porcelain_mug_1', 'porcelain_mug_1')):
    rec = record_for_step(step); scene, _ = restore(rec); state, q = step_state(step)
    planner = CandidatePlanner(kin); planner._prepare(scene, state, q); planner._deadline = float('inf')
    bound = planner.bound; R = bound.forward(q)[0][:3, :3]
    c = np.array(poses[body]['body_xyz_m']); top = truth[key]['upper'][2]
    xyz = np.array([c[0], c[1], top - 0.005])
    res = bound.solve(xyz, R, q, max_iterations=200)
    # payload_after exactly as _astra_keypose builds it
    pose = np.eye(4); pose[:3, 3] = xyz; pose[:3, :3] = R
    obj = scene.objects[target]
    pts, ev = planning_surface(obj)
    local = (np.asarray(pts) - pose[:3, 3]) @ pose[:3, :3]
    payload = {'object_id': target, 'owner_ids': [target], 'points_in_gripper': local.tolist(), 'confirmed': False,
               'source': 'astra_close_visible_surface_hypothesis', 'geometry_unknown': False,
               'upright_in_gripper': (pose[:3, :3].T @ np.array([0., 0., 1.])).tolist()}
    shape = _payload_shape(local)
    hull_extent = np.ptp(shape.vertices, 0).round(3).tolist() if shape is not None else None
    planner._contact_parts = ('finger', 'pad')
    r_close = planner._configuration(np.asarray(res.joints), scene, np.array([0., 0.]), payload, (target,), free_check=False)
    # lift: 5 cm straight up from the close pose
    res2 = bound.solve(xyz + [0, 0, .05], R, np.asarray(res.joints), max_iterations=200)
    r_lift = planner._configuration(np.asarray(res2.joints), scene, np.array([0., 0.]), payload, (target,), free_check=False)
    # carry through the can: grip 3 cm above can top at can xy (payload bottom intersects can)
    can_c = np.array(poses['tomato_sauce_obstacle_1']['body_xyz_m'])
    planner._contact_parts = ()
    res3 = bound.solve(np.array([can_c[0], can_c[1], truth['can']['upper'][2] + (xyz[2] - truth[key]['lower'][2]) - 0.03]), R, np.asarray(res.joints), max_iterations=200)
    r_through_can = planner._configuration(np.asarray(res3.joints), scene, np.array([0., 0.]), payload, (), free_check=False) if res3.ok else {'ik': False}
    # carry through the red mug
    red_c = np.array(poses['red_coffee_mug_1']['body_xyz_m'])
    res4 = bound.solve(np.array([red_c[0], red_c[1], truth['red_coffee_mug_1']['upper'][2] + (xyz[2] - truth[key]['lower'][2]) - 0.03]), R, np.asarray(res.joints), max_iterations=200)
    r_through_red = planner._configuration(np.asarray(res4.joints), scene, np.array([0., 0.]), payload, (), free_check=False) if res4.ok else {'ik': False}
    def brief(r):
        return {k: r.get(k) for k in ('safe', 'reason', 'part', 'point_object_id', 'uncertain_obstacle_samples', 'ik') if k in r}
    rec_ = {'step': step, 'frame': scene.frame_id, 'target': target, 'planning_points': len(pts), 'payload_hull_extent_m': hull_extent,
            'truth_extent_m': truth[key]['size'], 'close_ik': bool(res.ok),
            'close_check_with_payload': brief(r_close), 'lift5cm_check': brief(r_lift),
            'carry_payload_bottom_3cm_into_can': brief(r_through_can), 'carry_payload_bottom_3cm_into_red_mug': brief(r_through_red)}
    out.append(rec_); print(json.dumps(rec_))
(W2 / 'k2_payload_test.json').write_text(json.dumps(out, indent=1))
