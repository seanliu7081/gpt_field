"""Per-frame metric drift of the STATIC external view under the two-view baseline-ratio scale (backends.py:315-354):
table-plane height of external sampled points across H3 frames (robot moving). Truth table z for scoring only."""
import json, glob
import numpy as np
R = '/workspace/gpt_field/runs/foundation_vision_t04_can_obstacle_20261105_001/trial/foundation_vision'
tz = 0.43676
objs = [(-0.0276, 0.0779), (-0.0875, -0.1559), (-0.2024, 0.0183), (-0.0045, -0.3232), (0.0054, 0.3109), (-0.0235, -0.2179)]
rows = []
for p in sorted(glob.glob(R + '/frame_*_snapshot.json')):
    s = json.load(open(p)); g = s['geometry']
    P = np.array(g['surface_points_world'])
    n_total = g['surface_hypothesis_count']
    n_ext = g['views']['external']['valid_depth_pixels']
    idx = np.linspace(0, n_total - 1, len(P)).astype(int) if n_total > len(P) else np.arange(len(P))
    is_ext = idx < n_ext
    far = np.ones(len(P), bool)
    for c in objs:
        far &= np.linalg.norm(P[:, :2] - np.array(c), axis=1) > .12
    table = far & (P[:, 0] > -0.4) & (P[:, 0] < 0.15) & (np.abs(P[:, 1]) < 0.55) & (P[:, 2] > 0.30) & (P[:, 2] < 0.55)
    r = {'frame': s['frame_id']}
    for name, m in (('external', is_ext), ('wrist', ~is_ext)):
        sel = table & m
        r[name] = {'n': int(sel.sum()), 'table_z_minus_truth_median_m': round(float(np.median(P[sel, 2]) - tz), 4) if sel.sum() else None}
    rows.append(r); print(json.dumps(r))
json.dump(rows, open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/k2_h3_scale_drift.json', 'w'), indent=1)
