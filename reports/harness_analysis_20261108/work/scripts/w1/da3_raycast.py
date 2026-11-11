"""Ray-cast truth depth (evaluation only) for table plane and can cylinder; compare DA3 depth at several scales."""
import sys, json
sys.path.insert(0, '/workspace/gpt_field/src')
import numpy as np
from gpt_field.foundation_vision.bridge import load_manifest
from gpt_field.foundation_vision.backends import depth_on_original_rays
R = '/workspace/gpt_field/runs/'
m = load_manifest(R + 'foundation_vision_preparation_001/input/manifest.json')
frame = m.frames[0]; cams = {c.name: c for c in frame.cameras}; names = [c.name for c in frame.cameras]
z = np.load(R + 'foundation_vision_preparation_001/da3_real_003/cold_native_prediction.npz')
raw = z['raw_depth_model_units']; K = z['native_intrinsics']
mk = np.load(R + 'foundation_vision_sam_validation_001/sam_inference_002/all_candidate_masks.npz')
masks = {'external': mk['mask_000'], 'wrist': mk['mask_001']}
TABLE_Z = 0.43676
CAN_LO = np.array([-0.054474949094583634, -0.24536935556741868, 0.43696]); CAN_HI = np.array([0.0077168271458434145, -0.18373144547662518, 0.51315])
AX = (CAN_LO[:2] + CAN_HI[:2]) / 2; RAD = float(np.mean((CAN_HI - CAN_LO)[:2]) / 2)
def rays(cam):
    h, w = cam.rgb.shape[:2]; v, u = np.indices((h, w), dtype=float)
    rc = np.linalg.solve(cam.intrinsics, np.stack((u.ravel(), v.ravel(), np.ones(h*w))))
    Rw = cam.world_to_camera[:3, :3]; t = cam.world_to_camera[:3, 3]
    return -Rw.T @ t, (Rw.T @ rc)
def can_hit(C, d):
    # side: |(C+lam d)_xy - AX| = RAD
    oc = C[:2, None] - AX[:, None]; dd = d[:2]
    a = (dd**2).sum(0); b = 2*(oc*dd).sum(0); c = (oc**2).sum(0) - RAD**2
    disc = b*b - 4*a*c
    lam_side = np.full(d.shape[1], np.inf)
    ok = disc >= 0
    l1 = (-b - np.sqrt(np.where(ok, disc, 0))) / (2*a)
    zs = C[2] + l1*d[2]
    side_ok = ok & (l1 > 0) & (zs >= CAN_LO[2]) & (zs <= CAN_HI[2])
    lam_side[side_ok] = l1[side_ok]
    lt = (CAN_HI[2] - C[2]) / d[2]
    pt = C[:2, None] + lt*dd
    top_ok = (lt > 0) & (np.linalg.norm(pt - AX[:, None], axis=0) <= RAD)
    lam_top = np.where(top_ok, lt, np.inf)
    return np.minimum(lam_side, lam_top)
truth = {}
for n in names:
    C, d = rays(cams[n])
    lam_plane = (TABLE_Z - C[2]) / d[2]
    lam_can = can_hit(C, d)
    truth[n] = (lam_plane, lam_can)
def report(scales, tag):
    out = {}
    for i, n in enumerate(names):
        dep, _ = depth_on_original_rays(raw[i]*scales[n], K[i], cams[n])
        dep = dep.ravel(); lam_plane, lam_can = truth[n]
        mask = masks[n].ravel()
        hit = mask & np.isfinite(lam_can) & np.isfinite(dep)
        err = dep[hit] - lam_can[hit]
        tol = 0.02 + 0.02*np.maximum(dep[hit], lam_can[hit])
        # table pixels: ray hits plane, not the can, and lies in the table footprint & lower image band (exclude objects crudely via being far from can)
        P = rays(cams[n])[0][:, None] + rays(cams[n])[1]*lam_plane
        tab = (lam_plane > 0) & ~np.isfinite(lam_can) & np.isfinite(dep) & (P[0] > -0.69) & (P[0] < 0.19) & (P[1] > -0.62) & (P[1] < 0.72)
        e2 = dep[tab] - lam_plane[tab]
        # robust: the table is the majority surface along such rays only if not occluded by other objects/robot; use mode of relative error
        rel = e2 / lam_plane[tab]
        hist, edges = np.histogram(rel, bins=300, range=(-0.3, 0.3)); k = hist.argmax(); mode_rel = 0.5*(edges[k]+edges[k+1])
        out[n] = dict(can_pixels_hit=int(hit.sum()), can_mask_pixels=int(mask.sum()),
                      can_depth_err_median_m=round(float(np.median(err)), 4), can_abs_err_p50=round(float(np.median(np.abs(err))), 4),
                      can_abs_err_p90=round(float(np.percentile(np.abs(err), 90)), 4), can_within_engine_tol=round(float((np.abs(err) <= tol).mean()), 3),
                      can_true_depth_median=round(float(np.median(lam_can[hit])), 3),
                      table_rel_err_mode=round(float(mode_rel), 4), table_true_depth_median=round(float(np.median(lam_plane[tab])), 3))
    print(tag, json.dumps(out))
S0 = 0.8780772385418903
report({'external': S0, 'wrist': S0}, 'baseline_ratio_0.878')
report({'external': 0.84, 'wrist': 0.72}, 'selfcal_best_0.84_0.72')
report({'external': 0.8484, 'wrist': 0.7435}, 'table_cal_0.848_0.744')
