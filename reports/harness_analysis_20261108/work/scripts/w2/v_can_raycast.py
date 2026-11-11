"""Cylinder ray-cast of the T04 can at frame 0: silhouette coverage, and the bidirectional mask-agreement bound with TRUE geometry."""
import sys, json
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
import numpy as np
from v_common import *
run = RUNS["T04"]
lm = json.load(open(run + "/inputs/layout_manifest.json")); b = lm["can_after_hold_bounds"]
lo, hi = np.array(b["lower_xyz_m"]), np.array(b["upper_xyz_m"])
C = (lo + hi) / 2; RAD = float(np.mean((hi - lo)[:2]) / 2); Z0, Z1 = lo[2], hi[2]
f = last_user_fields(json.load(open(requests(run)[0]))); assert f["step"] == 0
cams = {n: cam_from_geometry(f["geometry"], n) for n in ("external", "wrist")}
meta = json.load(open(run + "/trial/foundation_vision/frame_000000_sam_candidates.json"))
arr = np.load(run + "/trial/foundation_vision/frame_000000_sam_candidates.npz")
masks = {r["view"]: arr[r["array_key"]] for r in meta["candidates"] if r["target_key"] == "extra_0"}
def cast(K, E):
    v, u = np.indices((512, 512)); pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    R = E[:3, :3]; o = -R.T @ E[:3, 3]; d = (R.T @ np.linalg.solve(K, pix)).T; d /= np.linalg.norm(d, axis=1, keepdims=True)
    t = np.full(len(d), np.inf)
    # side: (ox+t dx - cx)^2 + (oy + t dy - cy)^2 = r^2
    ox, oy = o[0] - C[0], o[1] - C[1]
    a = d[:, 0] ** 2 + d[:, 1] ** 2; bb = 2 * (ox * d[:, 0] + oy * d[:, 1]); cc = ox ** 2 + oy ** 2 - RAD ** 2
    disc = bb ** 2 - 4 * a * cc; ok = disc >= 0
    t1 = np.where(ok, (-bb - np.sqrt(np.where(ok, disc, 0))) / (2 * a), np.inf)
    z1 = o[2] + t1 * d[:, 2]; side = ok & (t1 > 0) & (z1 >= Z0) & (z1 <= Z1)
    t = np.where(side, t1, t)
    tc = (Z1 - o[2]) / d[:, 2]; px = o[0] + tc * d[:, 0] - C[0]; py = o[1] + tc * d[:, 1] - C[1]
    cap = (tc > 0) & (px ** 2 + py ** 2 <= RAD ** 2)
    t = np.where(cap & (tc < t), tc, t)
    hit = np.isfinite(t)
    P = o + np.where(hit, t, 0)[:, None] * d
    return hit.reshape(512, 512), P.reshape(512, 512, 3), o
sil = {}; pts = {}; origin = {}
for n, (K, E, _) in cams.items():
    sil[n], pts[n], origin[n] = cast(K, E)
    m = masks[n]; inter = (m & sil[n]).sum()
    ys, xs = np.nonzero(sil[n])
    print(f"{n}: true silhouette px={sil[n].sum()} bbox x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]; SAM px={m.sum()}; coverage={inter/sil[n].sum():.3f} precision={inter/m.sum():.3f} IoU={inter/(m|sil[n]).sum():.3f}")
# bidirectional agreement with TRUE geometry: points of view A's SAM mask that hit the can (true surface), projected into B; visible in B if B's first hit agrees.
for a, bname in (("external", "wrist"), ("wrist", "external")):
    K, E, _ = cams[bname]
    sel = masks[a] & sil[a]
    P = pts[a][sel]
    uv, z = project(K, E, P)
    u = np.floor(uv[:, 0] + 0.5).astype(int); v = np.floor(uv[:, 1] + 0.5).astype(int)
    inside = (u >= 0) & (u < 512) & (v >= 0) & (v < 512)
    # visible in B: B's ray at that pixel hits the can at nearly the same point
    vis = np.zeros(len(P), bool)
    vis[inside] = sil[bname][v[inside], u[inside]] & (np.linalg.norm(pts[bname][v[inside], u[inside]] - P[inside], axis=1) < 0.005)
    inm = np.zeros(len(P), bool); inm[inside] = masks[bname][v[inside], u[inside]]
    print(f"{a}->{bname}: true-surface points in {a} mask={len(P)}, inside {bname} raster={inside.sum()}, visible in {bname}={vis.sum()}, of which in {bname} SAM mask={(vis & inm).sum()} -> agreement={(vis & inm).sum()/max(1, vis.sum()):.3f}")
