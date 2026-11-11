"""Can a robust per-frame affine alignment of wrist depth to the external view remove the wrist bias?
Diagnosis uses the table plane z=0.4368 m from physical_truth.json (never available to policy). Offline only."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import restore, record_by_arrays
TABLE_Z = .4368
res = []
for arrays in ['frame_000005_revision_000011.npz', 'frame_000010_revision_000022.npz', 'frame_000019_revision_000041.npz']:
    record = record_by_arrays(arrays)
    scene, data, _ = restore(record)
    ext = scene.geometry._views['external']; wr = scene.geometry._views['wrist']
    Kw, Ew = wr.camera.intrinsics, wr.camera.world_to_camera
    P = ext.points_world
    c = P @ Ew[:3, :3].T + Ew[:3, 3]
    ok = c[:, 2] > .05
    uv = (c[ok] @ Kw.T)[:, :2] / c[ok, 2:3]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] <= 511) & (uv[:, 1] >= 0) & (uv[:, 1] <= 511)
    zc = c[ok][inside, 2]; px = np.rint(uv[inside]).astype(int)
    # z-buffer: keep the nearest external point per wrist pixel
    order = np.argsort(zc)
    key = px[order, 1] * 512 + px[order, 0]
    _, first = np.unique(key, return_index=True)
    sel = order[first]
    zr = zc[sel]; u, v = px[sel, 0], px[sel, 1]
    dw = wr.estimate.depth_m[v, u]
    good = np.isfinite(dw) & (v < 380)          # exclude the gripper band at the bottom of the wrist image
    ratio = zr[good] / dw[good]
    keep = np.abs(ratio - np.median(ratio)) < 3 * 1.4826 * np.median(np.abs(ratio - np.median(ratio)))
    A = np.column_stack((dw[good][keep], np.ones(keep.sum())))
    a, b = np.linalg.lstsq(A, zr[good][keep], rcond=None)[0]
    # evaluate on the wrist table region against the true plane (diagnosis)
    K, E = Kw, Ew; center = -E[:3, :3].T @ E[:3, 3]
    vv, uu = np.mgrid[5:150, 60:450]
    d = wr.estimate.depth_m[vv, uu].ravel()
    rays = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)]).T @ np.linalg.inv(K).T @ E[:3, :3]
    t = (TABLE_Z - center[2]) / rays[:, 2]
    before = np.nanmedian(d / t); after = np.nanmedian((a * d + b) / t)
    row = {'arrays': arrays, 'overlap_pixels_used': int(keep.sum()), 'fit_scale': round(float(a), 4), 'fit_offset_m': round(float(b), 4),
           'wrist_table_scale_ratio_before': round(float(before), 4), 'wrist_table_scale_ratio_after': round(float(after), 4),
           'wrist_table_depth_error_after_m': round(float(np.nanmedian((a * d + b) - t)), 4)}
    res.append(row); print(json.dumps(row), flush=True)
(Path(__file__).parent / 'crossview_align_results.json').write_text(json.dumps(res, indent=1))
