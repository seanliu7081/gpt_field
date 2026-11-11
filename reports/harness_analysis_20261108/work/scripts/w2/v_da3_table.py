import json, numpy as np
P = "/workspace/gpt_field/runs/foundation_vision_preparation_001/da3_real_003/cold_native_prediction.npz"
z = np.load(P)
lm = json.load(open("/workspace/gpt_field/runs/foundation_vision_t04_can_obstacle_20261105_001/inputs/layout_manifest.json"))
tg = lm["can"]["table_support_geometry"]; print("table support", tg)
TZ = lm["can"]["table_height_m"]
poses = {k: np.array(v["body_xyz_m"]) for k, v in lm["custom_after_passive_hold_object_poses"].items()}
depth = z["aligned_depth_m"].astype(float); K = z["native_intrinsics"].astype(float); E = z["aligned_extrinsics"].astype(float)
raw = z["raw_depth_model_units"].astype(float)
print("scale from aligned/raw:", np.nanmedian(depth / raw))
lo_t, hi_t = np.array(tg["lower_xyz_m"]), np.array(tg["upper_xyz_m"])
res = {}
for i, name in enumerate(("external", "wrist")):
    H, W = depth[i].shape
    v, u = np.indices((H, W)); pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    R = E[i][:, :3]; t = E[i][:, 3]; o = -R.T @ t
    dc = np.linalg.solve(K[i], pix).T          # camera-frame direction with z=1
    dw = dc @ R                                 # world direction (R^T dc)
    s = (TZ - o[2]) / dw[:, 2]                  # since dc z=1, s equals the camera z-depth of the plane hit
    hit = o + s[:, None] * dw
    on_table = (s > 0) & (hit[:, 0] >= lo_t[0]) & (hit[:, 0] <= hi_t[0]) & (hit[:, 1] >= lo_t[1]) & (hit[:, 1] <= hi_t[1])
    # exclude object footprints (generous 9 cm radius around each object body xy) - objects and the plates
    far = np.ones(len(hit), bool)
    for k, X in poses.items():
        r = 0.13 if "plate" in k else 0.09
        far &= np.linalg.norm(hit[:, :2] - X[:2], axis=1) > r
    d = depth[i].ravel()
    sel = on_table & far & np.isfinite(d) & (d > 0)
    rel = d[sel] / s[sel] - 1
    # also robust: restrict to the predicted point within 10 cm of plane in depth to drop robot pixels (report both)
    print(f"{name}: table px={sel.sum()} median rel err={np.median(rel)*100:+.2f}% (abs median err {np.median(d[sel]-s[sel])*100:+.2f} cm at median plane depth {np.median(s[sel]):.3f} m); p25/p75 {np.percentile(rel,25)*100:+.1f}/{np.percentile(rel,75)*100:+.1f}%")
    # per-view scale that zeroes the table bias
    res[name] = 1 / (1 + np.median(rel))
print("per-view multiplicative correction to zero table bias:", res, "=> per-view scale", {k: 0.8780772385418903 * v for k, v in res.items()})
print("---- external & wrist error vs true plane depth (objects excluded)")
for i, name in enumerate(("external", "wrist")):
    H, W = depth[i].shape
    v, u = np.indices((H, W)); pix = np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    R = E[i][:, :3]; t = E[i][:, 3]; o = -R.T @ t
    dw = np.linalg.solve(K[i], pix).T @ R
    s = (TZ - o[2]) / dw[:, 2]; hit = o + s[:, None] * dw
    on = (s > 0) & (hit[:, 0] >= lo_t[0]) & (hit[:, 0] <= hi_t[0]) & (hit[:, 1] >= lo_t[1]) & (hit[:, 1] <= hi_t[1])
    far = np.ones(len(hit), bool)
    for k, X in poses.items(): far &= np.linalg.norm(hit[:, :2] - X[:2], axis=1) > (0.13 if "plate" in k else 0.09)
    d = depth[i].ravel(); sel = on & far & np.isfinite(d) & (d > 0)
    r = d[sel] / s[sel]
    hist, edges = np.histogram(r, bins=400, range=(0.6, 1.4)); j = hist.argmax()
    rall = (d / s)[on & np.isfinite(d) & (d > 0)]
    h2, e2 = np.histogram(rall, bins=400, range=(0.6, 1.4)); j2 = h2.argmax()
    print(f"{name}: mode(objects excl)={(edges[j]+edges[j+1])/2:.4f} median={np.median(r):.4f} | all table-ray px (analyst set): mode={(e2[j2]+e2[j2+1])/2:.4f} median={np.median(rall):.4f} n={len(rall)}")
    qs = np.percentile(s[sel], [0, 20, 40, 60, 80, 100])
    for a, b in zip(qs[:-1], qs[1:]):
        m = (s[sel] >= a) & (s[sel] < b)
        print(f"    plane depth {a:.3f}-{b:.3f} m: n={m.sum():6d} median rel err {np.median(r[m]-1)*100:+.1f}%  ({np.median((r[m]-1)*s[sel][m])*100:+.1f} cm)")
