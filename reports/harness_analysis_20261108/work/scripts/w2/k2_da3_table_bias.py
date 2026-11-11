"""DA3 z-depth relative error on open-table pixels (ray/plane truth at the layout table height), per view, run-001 frames."""
import json, sys
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import V, records, truth, W2, in_box
tz = truth['table_top_z']
out = []
seen = set()
for rec in records:
    if rec['arrays'] in seen or rec['snapshot']['frame_id'] not in (1, 5, 10, 14, 20): continue
    seen.add(rec['arrays'])
    d = np.load(V / rec['arrays'])
    row = {'frame': rec['snapshot']['frame_id']}
    for c in rec['array_metadata']['cameras']:
        p = c['array_prefix']; E = d[p + '_world_to_camera']; K = d[p + '_intrinsics']; D = d[p + '_depth_m'].astype(float)
        h, w = D.shape
        v, u = np.indices((h, w))
        rays_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1).reshape(-1, 3)
        R, t = E[:3, :3], E[:3, 3]; C = -R.T @ t
        dirs = rays_cam @ R               # world direction for unit z-depth
        s = (tz - C[2]) / dirs[:, 2]       # z-depth at which the ray meets the table plane
        hit = C + s[:, None] * dirs
        ok = (s > 0) & (hit[:, 0] > -0.45) & (hit[:, 0] < 0.19) & (hit[:, 1] > -0.6) & (hit[:, 1] < 0.7)
        for k in ('white_yellow_mug_1', 'porcelain_mug_1', 'red_coffee_mug_1', 'plate_1', 'plate_2', 'can'):
            lo = np.array(truth[k]['lower']) - .04; hi = np.array(truth[k]['upper']) + .04
            ok &= ~((hit[:, 0] > lo[0]) & (hit[:, 0] < hi[0]) & (hit[:, 1] > lo[1]) & (hit[:, 1] < hi[1]))
        pred = D.reshape(-1)
        ok &= np.isfinite(pred) & (pred > 0)
        # exclude robot-occluded pixels: predicted point must be within 15 cm of the plane-hit (robot pixels are far off)
        P = C + pred[:, None] * dirs
        near = np.linalg.norm(P - hit, axis=1) < .15
        sel = ok & near
        rel = pred[sel] / s[sel] - 1
        row[c['view']] = {'pixels': int(sel.sum()), 'rel_depth_error_median': float(np.median(rel)), 'p10_p90': np.percentile(rel, [10, 90]).round(4).tolist(),
                          'true_depth_median_m': float(np.median(s[sel])), 'z_error_median_m': float(np.median(P[sel, 2] - tz))}
    out.append(row); print(json.dumps(row))
(W2 / 'k2_da3_table_bias.json').write_text(json.dumps(out, indent=1))
