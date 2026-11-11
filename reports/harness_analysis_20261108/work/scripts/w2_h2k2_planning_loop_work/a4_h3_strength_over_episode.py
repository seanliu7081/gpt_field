"""A4: how often is the T04 can 'strong' (cross-view supported, uncontradicted)?

Uses the H3 live run's saved geometry snapshots (2048-point uniform subsample of
the same SAM/DA3 + build_geometry pipeline that keypose-2 reuses) at 11 decision
frames with the robot at different poses. Strong = support>=2 & contradictions==0,
the keypose-2 motion.py:128 rule for unowned points. Layout truth scores only.
"""
import glob
import json
from pathlib import Path

import numpy as np

H3 = Path('/workspace/gpt_field/runs/foundation_vision_t04_can_obstacle_20261105_001/trial')
LAYOUT = json.loads(Path('/workspace/gpt_field/output/layouts/libero_10_t4_can_obstacle_20261103_001/layout_manifest.json').read_text())
can = LAYOUT['can']['collision_bounds']
lo, hi = np.array(can['lower_xyz_m']), np.array(can['upper_xyz_m'])
table_z = LAYOUT['can']['table_support_geometry']['upper_xyz_m'][2]

# measured EEF per step from the H3 trace (executed events carry next_state)
states = {}
with open(H3 / 'trace.jsonl') as handle:
    for line in handle:
        if '"event": "executed"' not in line and '"event":"executed"' not in line:
            continue
        row = json.loads(line)
        if row.get('event') == 'executed':
            states[row['step'] + 1] = row['next_state']

rows = []
for path in sorted(glob.glob(str(H3 / 'foundation_vision/frame_*_snapshot.json'))):
    snap = json.loads(Path(path).read_text())
    g = snap['geometry']
    pts = np.asarray(g['surface_points_world'], float)
    sup = np.asarray(g['surface_crossview_support_counts'])
    con = np.asarray(g['surface_contradiction_counts'])
    strong = (sup >= 2) & (con == 0)
    def frac(mask):
        n = int(mask.sum())
        return {'n': n, 'strong': int((mask & strong).sum()), 'strong_frac': round(float((mask & strong).sum() / n), 3) if n else None}
    in_can = np.all((pts >= lo - 0.01) & (pts <= hi + 0.01), axis=1)
    table = np.abs(pts[:, 2] - table_z) <= 0.01
    high = pts[:, 2] > 0.60
    frame = int(Path(path).name.split('_')[1])
    st = states.get(frame)
    rows.append({'frame': frame, 'eef_xyz': None if st is None else np.round(st[:3], 3).tolist(),
                 'gripper_gap_m': None if st is None else round(st[6] - st[7], 4),
                 'subsample_points': len(pts), 'all': frac(np.ones(len(pts), bool)),
                 'can_box_+1cm': frac(in_can), 'table_+-1cm': frac(table), 'above_z_0.60': frac(high),
                 'views_full_counts': {k: {'valid': v['valid_depth_pixels'], 'crossview_supported': v['crossview_supported_points'],
                                          'contradicted': v['contradicted_points']} for k, v in g['views'].items()}})
for r in rows:
    print(json.dumps(r))
out = Path('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/h2k2_planning_loop_work/data/a4_h3_strength_over_episode.json')
out.write_text(json.dumps(rows, indent=1))
