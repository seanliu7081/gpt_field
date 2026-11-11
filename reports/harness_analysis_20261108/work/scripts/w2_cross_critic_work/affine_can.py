"""Does h2_perception's per-view affine wrist->external alignment keep OBJECT depth right?
Ray-cast the true can cylinder and the true table plane (diagnostic truth only) on saved H2 T04 frames;
compare DA3 depth before and after the affine fit (same fit as w1/h2pg/crossview_align.py)."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/h2pg")
from common import restore, record_by_arrays
TABLE_Z = .4368
CAN_LO = np.array([-0.0545, -0.2454, 0.4370]); CAN_HI = np.array([0.0077, -0.1837, 0.5132])
AX = (CAN_LO[:2] + CAN_HI[:2]) / 2; RAD = float(np.mean((CAN_HI - CAN_LO)[:2]) / 2)

def rays(view):
    K, E = view.camera.intrinsics, view.camera.world_to_camera
    v, u = np.indices((512, 512), dtype=float)
    rc = np.linalg.solve(K, np.stack((u.ravel(), v.ravel(), np.ones(512*512))))
    R, t = E[:3, :3], E[:3, 3]
    d = R.T @ rc
    return -R.T @ t, d, rc[2]   # camera centre, world dirs (unnormalised, z_cam = 1), depth scale

def can_hit(C, d):
    oc = C[:2, None] - AX[:, None]; dd = d[:2]
    a = (dd**2).sum(0); b = 2*(oc*dd).sum(0); c = (oc**2).sum(0) - RAD**2
    disc = b*b - 4*a*c; ok = disc >= 0
    l1 = (-b - np.sqrt(np.where(ok, disc, 0))) / (2*a)
    zs = C[2] + l1*d[2]
    lam_side = np.where(ok & (l1 > 0) & (zs >= CAN_LO[2]) & (zs <= CAN_HI[2]), l1, np.inf)
    lt = (CAN_HI[2] - C[2]) / d[2]; pt = C[:2, None] + lt*dd
    lam_top = np.where((lt > 0) & (np.linalg.norm(pt - AX[:, None], axis=0) <= RAD), lt, np.inf)
    return np.minimum(lam_side, lam_top)

out = []
for arrays in ['frame_000005_revision_000011.npz', 'frame_000010_revision_000022.npz', 'frame_000019_revision_000041.npz']:
    scene, data, _ = restore(record_by_arrays(arrays))
    ext = scene.geometry._views['external']; wr = scene.geometry._views['wrist']
    # --- affine fit, identical to crossview_align.py ---
    Kw, Ew = wr.camera.intrinsics, wr.camera.world_to_camera
    P = ext.points_world; c = P @ Ew[:3, :3].T + Ew[:3, 3]; ok = c[:, 2] > .05
    uv = (c[ok] @ Kw.T)[:, :2] / c[ok, 2:3]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] <= 511) & (uv[:, 1] >= 0) & (uv[:, 1] <= 511)
    zc = c[ok][inside, 2]; px = np.rint(uv[inside]).astype(int)
    order = np.argsort(zc); key = px[order, 1]*512 + px[order, 0]; _, first = np.unique(key, return_index=True)
    sel = order[first]; zr = zc[sel]; u, v = px[sel, 0], px[sel, 1]
    dw = wr.estimate.depth_m[v, u]; good = np.isfinite(dw) & (v < 380)
    ratio = zr[good] / dw[good]; mad = np.median(np.abs(ratio - np.median(ratio)))
    keep = np.abs(ratio - np.median(ratio)) < 3*1.4826*mad
    a, b = np.linalg.lstsq(np.column_stack((dw[good][keep], np.ones(keep.sum()))), zr[good][keep], rcond=None)[0]
    row = {"arrays": arrays, "fit_scale": round(float(a), 4), "fit_offset_m": round(float(b), 4)}
    for name, view in (("external", ext), ("wrist", wr)):
        C, d, zscale = rays(view)
        lam_can = can_hit(C, d); lam_tab = (TABLE_Z - C[2]) / d[2]
        dep = view.estimate.depth_m.ravel()          # camera z-depth
        # convert ray parameter (with z_cam=1 directions) to camera z-depth: z = lam * 1
        can = np.isfinite(lam_can) & (lam_can < lam_tab) & np.isfinite(dep)
        vv = np.repeat(np.arange(512), 512)
        if name == "wrist":
            can &= vv < 380
        tab = (lam_tab > 0) & ~np.isfinite(lam_can) & np.isfinite(dep)
        P_tab = C[:, None] + d*lam_tab
        tab &= (P_tab[0] > -0.69) & (P_tab[0] < 0.19) & (P_tab[1] > -0.62) & (P_tab[1] < 0.72)
        if name == "wrist":
            tab &= vv < 380
        def stats(depth):
            e_can = depth[can] - lam_can[can]
            rel_tab = (depth[tab] - lam_tab[tab]) / lam_tab[tab]
            h, ed = np.histogram(rel_tab, bins=300, range=(-0.3, 0.3)); k = h.argmax()
            return {"can_px": int(can.sum()), "can_err_median_cm": round(100*float(np.median(e_can)), 2) if can.any() else None,
                    "can_abs_err_p50_cm": round(100*float(np.median(np.abs(e_can))), 2) if can.any() else None,
                    "can_true_depth_m": round(float(np.median(lam_can[can])), 3) if can.any() else None,
                    "table_rel_err_mode": round(float(0.5*(ed[k]+ed[k+1])), 4), "table_true_depth_m": round(float(np.median(lam_tab[tab])), 3)}
        row[name + "_shipped"] = stats(dep)
        if name == "wrist":
            row["wrist_affine"] = stats(a*dep + b)
    out.append(row); print(json.dumps(row))
Path("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/cross_critic_work/data/affine_can.json").write_text(json.dumps(out, indent=1))
