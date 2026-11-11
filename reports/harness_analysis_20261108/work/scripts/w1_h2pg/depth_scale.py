"""Per-view DA3 depth scale error against the table plane (diagnosis only; the table height
z=0.4368 m comes from the run's physical_truth.json and is never available to the policy)."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import restore, RECORDS
OUT = Path(__file__).parent
TABLE_Z = 0.4368
rows_out = []
for rec in RECORDS:
    if rec['snapshot']['evidence']['timings_s'].get('depth_s', 0) == 0:
        continue  # only frames with a fresh depth inference
    scene, data, _ = restore(rec)
    row = {'arrays': rec['arrays']}
    for cam in scene.frame.cameras:
        view = scene.geometry._views[cam.name]
        K, E = cam.intrinsics, cam.world_to_camera
        center = -E[:3, :3].T @ E[:3, 3]
        region = (slice(400, 505), slice(150, 360)) if cam.name == 'external' else (slice(5, 150), slice(60, 450))
        vv, uu = np.mgrid[region]
        d = view.estimate.depth_m[vv, uu]
        rays_c = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)]) .T @ np.linalg.inv(K).T
        rays_w = rays_c @ E[:3, :3]
        t = (TABLE_Z - center[2]) / rays_w[:, 2]          # z-depth to plane (rays have camera z = 1)
        ratio = d.ravel() / t
        ok = np.isfinite(ratio)
        row[cam.name] = {'median_scale_ratio': round(float(np.median(ratio[ok])), 4),
                         'p10_p90': [round(float(np.percentile(ratio[ok], 10)), 4), round(float(np.percentile(ratio[ok], 90)), 4)],
                         'median_true_depth_m': round(float(np.median(t[ok])), 3),
                         'median_depth_error_m': round(float(np.median((d.ravel() - t)[ok])), 4)}
    rows_out.append(row)
    print(json.dumps(row), flush=True)
(OUT / 'depth_scale_results.json').write_text(json.dumps(rows_out, indent=1))
ext = [r['external']['median_scale_ratio'] for r in rows_out]; wr = [r['wrist']['median_scale_ratio'] for r in rows_out]
print('external ratio range', min(ext), max(ext), 'wrist ratio range', min(wr), max(wr))
