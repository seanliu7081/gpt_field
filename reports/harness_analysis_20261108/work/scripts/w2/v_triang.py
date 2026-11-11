import sys, json, glob
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
import numpy as np
from v_common import *
truth = {}
lm4 = json.load(open(RUNS["T04"] + "/inputs/layout_manifest.json"))
b = lm4["can_after_hold_bounds"]; truth["T04"] = (np.add(b["lower_xyz_m"], b["upper_xyz_m"]) / 2, b)
lm8 = json.load(open(RUNS["T08"] + "/inputs/layout_manifest.json"))
b8 = lm8.get("can_after_hold_bounds") or lm8["bottle"]["collision_bounds"]
truth["T08"] = (np.add(b8["lower_xyz_m"], b8["upper_xyz_m"]) / 2, b8)
print("truth centres", {k: v[0].round(4).tolist() for k, v in truth.items()})
print("T08 bottle collision bounds (layout)", lm8["bottle"]["collision_bounds"]["lower_xyz_m"], lm8["bottle"]["collision_bounds"]["upper_xyz_m"])
print("T08 after-hold bounds", lm8["can_after_hold_bounds"]["lower_xyz_m"], lm8["can_after_hold_bounds"]["upper_xyz_m"])
for tag, run in RUNS.items():
    centre, bounds = truth[tag]
    reqs = requests(run)
    for rp in reqs:
        req = json.load(open(rp)); f = last_user_fields(req); step = f["step"]
        cand = json.load(open(run + f"/trial/foundation_vision/frame_{step:06d}_sam_candidates.json"))
        arr = np.load(run + f"/trial/foundation_vision/frame_{step:06d}_sam_candidates.npz")
        rows = [r for r in cand["candidates"] if r["target_key"] == "extra_0"]
        Ke, Ee, pe = cam_from_geometry(f["geometry"], "external")
        Kw, Ew, pw = cam_from_geometry(f["geometry"], "wrist")
        tc_e, _ = project(Ke, Ee, centre); tc_w, zw = project(Kw, Ew, centre)
        cents = {}
        for r in rows:
            m = arr[r["array_key"]]; ys, xs = np.nonzero(m)
            cents[r["view"]] = (xs.mean(), ys.mean(), m[int(round(tc_e[0,1])), int(round(tc_e[0,0]))] if r["view"] == "external" and 0 <= tc_e[0,0] < 512 and 0 <= tc_e[0,1] < 512 else None)
        line = f"{tag} step {step:4d} truth->ext ({tc_e[0,0]:.1f},{tc_e[0,1]:.1f}) truth->wrist ({tc_w[0,0]:.1f},{tc_w[0,1]:.1f}, z={zw[0]:.3f})"
        if "external" in cents:
            line += f" | ext mask centroid ({cents['external'][0]:.1f},{cents['external'][1]:.1f}) truth-in-mask={cents['external'][2]}"
            # single-view + support plane at the bbox-centre height (oracle) and at table height
            o, d = ray(Ke, Ee, cents["external"][:2])
        if "external" in cents and "wrist" in cents:
            o1, d1 = ray(Ke, Ee, cents["external"][:2]); o2, d2 = ray(Kw, Ew, cents["wrist"][:2])
            X, gap = closest(o1, d1, o2, d2)
            err = X - centre
            line += f" | TRI {X.round(4).tolist()} err_xy={np.linalg.norm(err[:2])*100:.2f}cm err_z={err[2]*100:.2f}cm gap={gap*100:.2f}cm"
        print(line)
