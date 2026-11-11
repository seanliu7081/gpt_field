"""Offline DA3 metric-accuracy check on the saved T04 initial frame (evaluation-only truth)."""
import sys, json
sys.path.insert(0, '/workspace/gpt_field/src')
import numpy as np
from gpt_field.foundation_vision.bridge import load_manifest
from gpt_field.foundation_vision.backends import depth_on_original_rays
from gpt_field.foundation_vision.types import DepthEstimate
from gpt_field.foundation_vision.geometry import build_geometry, GeometryConfig

R = '/workspace/gpt_field/runs/'
m = load_manifest(R + 'foundation_vision_preparation_001/input/manifest.json')
frame = m.frames[0]
z = np.load(R + 'foundation_vision_preparation_001/da3_real_003/cold_native_prediction.npz')
cams = {c.name: c for c in frame.cameras}
names = [c.name for c in frame.cameras]
raw = z['raw_depth_model_units']; K = z['native_intrinsics']; conf = z['raw_score_map']
S0 = 0.8780772385418903
def depths(se, sw):
    out = {}
    for i, n in enumerate(names):
        s = se if n == 'external' else sw
        d, sc = depth_on_original_rays(raw[i] * s, K[i], cams[n], conf[i])
        out[n] = DepthEstimate(n, d, cams[n].intrinsics, cams[n].world_to_camera, sc, source='da3')
    return out
# 1) reproduce the report
g = build_geometry(frame, depths(S0, S0), {})
rep = {n: {k: v[k] for k in ('valid_depth_pixels','crossview_supported_points','contradicted_points','occluded_in_other_views_points')} for n, v in g.to_dict()['views'].items()}
print('reproduced', json.dumps(rep))
TABLE_Z = 0.43676000000000015
TABLE_XY = ((-0.69432, 0.19432), (-0.61932, 0.7241))
def table_ratio(est, cam):
    # ray/plane depth for every pixel; compare predicted z-depth to plane z-depth
    h, w = est.depth_m.shape
    v, u = np.indices((h, w), dtype=float)
    rays = np.stack((u.ravel(), v.ravel(), np.ones(h*w))) 
    rc = np.linalg.solve(cam.intrinsics, rays)  # camera-frame ray with z=1
    Rw = cam.world_to_camera[:3, :3]; t = cam.world_to_camera[:3, 3]
    C = -Rw.T @ t
    dw = Rw.T @ rc  # world direction for z-depth 1
    lam = (TABLE_Z - C[2]) / dw[2]
    P = C[:, None] + dw * lam
    on = (lam > 0) & (P[0] > TABLE_XY[0][0]) & (P[0] < TABLE_XY[0][1]) & (P[1] > TABLE_XY[1][0]) & (P[1] < TABLE_XY[1][1])
    d = est.depth_m.ravel()
    ok = on & np.isfinite(d)
    return d[ok] / lam[ok], lam[ok]
def mode(x, lo=0.6, hi=1.4, bins=400):
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    i = hist.argmax(); return 0.5*(edges[i]+edges[i+1]), hist[i] / max(1, len(x))
D = depths(S0, S0)
for n in names:
    r, lam = table_ratio(D[n], cams[n])
    mo, frac = mode(r)
    near = np.abs(r - mo) < 0.03
    print(n, 'table-ray pixels', len(r), 'ratio pred/plane mode %.4f (mode bin frac %.3f) median %.4f; within +-3%% of mode: %.3f; plane depth range %.3f-%.3f m'
          % (mo, frac, np.median(r), near.mean(), lam.min(), lam.max()))
    # metric error implied at table: (ratio-1)*plane_depth for the table-like pixels
    err = (r[near] - 1) * lam[near]
    print('   implied depth error on table-mode pixels: median %.4f m, p10 %.4f, p90 %.4f' % (np.median(err), np.percentile(err, 10), np.percentile(err, 90)))
# 2) can: SAM masks from sam_validation on the same frame
mk = np.load(R + 'foundation_vision_sam_validation_001/sam_inference_002/all_candidate_masks.npz')
masks = {'external': mk['mask_000'], 'wrist': mk['mask_001']}
CAN_LO = np.array([-0.054474949094583634, -0.24536935556741868, 0.43696]); CAN_HI = np.array([0.0077168271458434145, -0.18373144547662518, 0.51315])
axis = (CAN_LO[:2] + CAN_HI[:2]) / 2; radius = float(np.mean((CAN_HI - CAN_LO)[:2]) / 2)
print('true can axis xy', axis.round(4), 'radius %.4f' % radius, 'z', CAN_LO[2], CAN_HI[2])
def can_err(gm, n, mask):
    v = gm._views[n]
    from gpt_field.foundation_vision.geometry import _mask_on_depth
    sel = _mask_on_depth(mask, v)
    P = v.points_world[sel]
    rad = np.linalg.norm(P[:, :2] - axis, axis=1) - radius
    zout = np.maximum(CAN_LO[2] - P[:, 2], 0) + np.maximum(P[:, 2] - CAN_HI[2], 0)
    surf = np.where(rad > 0, np.hypot(np.maximum(rad, 0), zout), np.maximum(zout, 0))  # outside distance (0 if inside)
    cam = cams[n]; C = -cam.world_to_camera[:3, :3].T @ cam.world_to_camera[:3, 3]
    rng = np.linalg.norm(P - C, axis=1)
    return dict(points=int(len(P)), centroid=P.mean(0).round(4).tolist(), range_m=float(np.median(rng)),
                radial_minus_r_median=float(np.median(rad)), radial_minus_r_p10=float(np.percentile(rad, 10)), radial_minus_r_p90=float(np.percentile(rad, 90)),
                z_median=float(np.median(P[:, 2])), outside_dist_median=float(np.median(surf)), outside_dist_p90=float(np.percentile(surf, 90)),
                frac_within_2cm_of_can=float((surf <= 0.02).mean()))
for n in names:
    print('can', n, json.dumps(can_err(g, n, masks[n])))
# 3) per-view scale sweep: cross-view agreement as a function of (se, sw)
def xview(se, sw):
    gg = build_geometry(frame, depths(se, sw), {})
    out = {}
    for n, v in gg._views.items():
        comp = int(((v.support >= 2) | (v.contradictions > 0) | (v.occlusions > 0)).sum())
        out[n] = (int((v.support >= 2).sum()), int((v.contradictions > 0).sum()), int((v.occlusions > 0).sum()), len(v.points_world))
    return out, gg
grid = np.round(np.arange(0.70, 1.0001, 0.02), 3)
best = None; rows = []
for se in grid:
    for sw in grid:
        o, _ = xview(se, sw)
        a = o['external']; b = o['wrist']
        score = (a[0] + b[0]) - (a[1] + b[1])
        rows.append((se, sw, a, b, score))
        if best is None or score > best[-1]: best = (se, sw, a, b, score)
print('baseline-ratio scale', S0, '-> ext (supp, contra, occl, n)', xview(S0, S0)[0])
print('best (se, sw) by support-minus-contradiction:', best)
srt = sorted(rows, key=lambda r: -r[-1])[:8]
for r in srt: print('  top', r)
se, sw = best[0], best[1]
Db = depths(se, sw)
for n in names:
    r, lam = table_ratio(Db[n], cams[n]); mo, frac = mode(r)
    print('at best scales', n, 'table ratio mode %.4f' % mo)
_, gb = xview(se, sw)
for n in names:
    print('can at best scales', n, json.dumps(can_err(gb, n, masks[n])))
# table-calibrated scales (evaluation-only: what would a plane-calibrated per-view scale give?)
tab = {}
for n in names:
    r, lam = table_ratio(D[n], cams[n]); mo, _ = mode(r)
    tab[n] = S0 / mo
print('table-calibrated scales', tab)
o, gt = xview(tab['external'], tab['wrist'])
print('cross-view at table-calibrated scales', o)
for n in names:
    print('can at table-calibrated scales', n, json.dumps(can_err(gt, n, masks[n])))
