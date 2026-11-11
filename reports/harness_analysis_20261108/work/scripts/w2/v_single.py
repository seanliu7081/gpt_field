import sys, json
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
import numpy as np
from v_common import *
cfg = {"T04": ("inputs/layout_manifest.json", 0.43676), "T08": ("inputs/layout_manifest.json", 0.90)}
for tag, run in RUNS.items():
    lm = json.load(open(run + "/" + cfg[tag][0])); b = lm["can_after_hold_bounds"]
    lo, hi = np.array(b["lower_xyz_m"]), np.array(b["upper_xyz_m"]); tc = (lo + hi) / 2
    res = []
    for rp in requests(run):
        req = json.load(open(rp)); f = last_user_fields(req); step = f["step"]
        cand = json.load(open(run + f"/trial/foundation_vision/frame_{step:06d}_sam_candidates.json"))
        arr = np.load(run + f"/trial/foundation_vision/frame_{step:06d}_sam_candidates.npz")
        ext = [r for r in cand["candidates"] if r["target_key"] == "extra_0" and r["view"] == "external"]
        if len(ext) != 1: res.append((step, None)); continue
        K, E, p = cam_from_geometry(f["geometry"], "external")
        m = arr[ext[0]["array_key"]]; ys, xs = np.nonzero(m)
        ub = xs[ys == ys.max()].mean(); vb = ys.max() + 0.5
        row = {}
        for dz in (-0.02, -0.01, 0.0, 0.01, 0.02):
            pz = cfg[tag][1] + dz
            o, d = ray(K, E, (ub, vb)); s = (pz - o[2]) / d[2]; front = o + s * d
            rng = np.linalg.norm(front - o)
            vm = int((ys.min() + ys.max()) / 2); r_ = xs[ys == vm]
            width = (r_.max() - r_.min() + 1) * rng / K[0, 0]
            hd = d[:2] / np.linalg.norm(d[:2]); centre = front[:2] + hd * width / 2
            row[dz] = round(float(np.linalg.norm(centre - tc[:2])) * 100, 2)
        res.append((step, row))
    print(tag, "xy error (cm) of single-view+plane centre vs truth bbox centre, by plane-height error dz (m):")
    for step, row in res: print("  step", step, row)
