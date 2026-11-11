import sys, json
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
import numpy as np
from v_common import *
cases = [("T04", "extra_0", 0, "same"), ("T08", "extra_0", 243, "same"), ("T08", "extra_0", 328, "same")] + \
        [("T04", "F", f, "wrong") for f in (17, 201, 252, 349, 350)] + [("T08", "D", f, "wrong") for f in (0, 24)]
for tag, key, fid, lab in cases:
    run = RUNS[tag]
    reqs = {last_user_fields(json.load(open(p)))["step"]: p for p in requests(run)}
    f = last_user_fields(json.load(open(reqs[fid])))
    meta = json.load(open(run + f"/trial/foundation_vision/frame_{fid:06d}_sam_candidates.json"))
    arr = np.load(run + f"/trial/foundation_vision/frame_{fid:06d}_sam_candidates.npz")
    rows = {r["view"]: r for r in meta["candidates"] if r["target_key"] == key}
    rays = {}
    for v in ("external", "wrist"):
        K, E, _ = cam_from_geometry(f["geometry"], v)
        m = arr[rows[v]["array_key"]]; ys, xs = np.nonzero(m)
        rays[v] = ray(K, E, (xs.mean(), ys.mean()))
    X, gap = closest(*rays["external"], *rays["wrist"])
    # also: is the triangulated point in front of both cameras and above the table?
    print(f"{tag} f{fid:3d} {key:7s} {lab:5s}: ray gap {gap*100:5.1f} cm, point {X.round(3).tolist()}")
