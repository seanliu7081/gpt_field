"""Same 'strong' definition (support>=2 & no contradiction, geometry.py) at OTHER robot poses, from H3's saved T04 snapshots
(subsampled 2048 points per frame; same SAM/DA3/build_geometry code as H2). Truth (can cylinder) for scoring only."""
import json, glob, sys
import numpy as np
R = '/workspace/gpt_field/runs/foundation_vision_t04_can_obstacle_20261105_001/trial/foundation_vision'
M = json.load(open('/workspace/gpt_field/output/foundation_harness_preparation_001/inputs/layout_manifest.json'))
can_c = np.array(M['custom_after_passive_hold_object_poses']['tomato_sauce_obstacle_1']['body_xyz_m'][:2])
red_c = np.array(M['custom_after_passive_hold_object_poses']['red_coffee_mug_1']['body_xyz_m'][:2])
tz = 0.43676
rows = []
for p in sorted(glob.glob(R + '/frame_*_snapshot.json')):
    s = json.load(open(p)); g = s['geometry']
    P = np.array(g['surface_points_world']); sup = np.array(g['surface_crossview_support_counts']); con = np.array(g['surface_contradiction_counts'])
    strong = (sup >= 2) & (con == 0)
    views = g['views']
    wrist = views.get('wrist', {})
    E = np.array(wrist.get('world_to_camera')) if wrist.get('world_to_camera') is not None else None
    wc = (-E[:3, :3].T @ E[:3, 3]).round(3).tolist() if E is not None else None
    can = (np.linalg.norm(P[:, :2] - can_c, axis=1) < .035) & (P[:, 2] > tz + .003) & (P[:, 2] < .52)
    red = (np.linalg.norm(P[:, :2] - red_c, axis=1) < .05) & (P[:, 2] > tz + .003) & (P[:, 2] < .575)
    tot_valid = sum(v['valid_depth_pixels'] for v in views.values())
    tot_strong = sum(v['crossview_supported_points'] for v in views.values())
    # can visible in wrist image? project can centre at mid-height
    vis = None
    if E is not None:
        K = np.array(wrist['intrinsics'])
        x = E[:3, :3] @ np.array([can_c[0], can_c[1], .475]) + E[:3, 3]
        uv = (K @ x)[:2] / x[2]
        vis = bool(x[2] > 0 and 0 <= uv[0] < 512 and 0 <= uv[1] < 512)
    rows.append({'frame': s['frame_id'], 'wrist_camera_center': wc, 'can_centre_in_wrist_image': vis,
                 'scene_strong_share_all_valid_pixels': round(tot_strong / max(1, tot_valid), 3),
                 'can_sampled_points': int(can.sum()), 'can_sampled_strong': int((can & strong).sum()),
                 'red_mug_sampled_points': int(red.sum()), 'red_mug_sampled_strong': int((red & strong).sum())})
for r in rows: print(json.dumps(r))
json.dump(rows, open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/k2_h3_pose_strength.json', 'w'), indent=1)
