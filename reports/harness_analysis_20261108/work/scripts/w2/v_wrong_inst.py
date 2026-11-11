import sys, json
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
import numpy as np
from v_common import *
for tag, key, frames in (("T04", "F", (17, 201, 252, 349, 350)), ("T08", "D", (0, 24, 328, 368))):
    run = RUNS[tag]
    lm = json.load(open(run + "/inputs/layout_manifest.json"))
    poses = {k: np.array(v["body_xyz_m"]) for k, v in lm["custom_after_passive_hold_object_poses"].items()}
    print(tag, {k: v.round(3).tolist() for k, v in poses.items()})
    reqs = {last_user_fields(json.load(open(p)))["step"]: p for p in requests(run)}
    for fid in frames:
        f = last_user_fields(json.load(open(reqs[fid])))
        Kw, Ew, _ = cam_from_geometry(f["geometry"], "wrist")
        meta = json.load(open(run + f"/trial/foundation_vision/frame_{fid:06d}_sam_candidates.json"))
        arr = np.load(run + f"/trial/foundation_vision/frame_{fid:06d}_sam_candidates.npz")
        rows = [r for r in meta["candidates"] if r["target_key"] == key and r["view"] == "wrist"]
        for r in rows:
            m = arr[r["array_key"]]; ys, xs = np.nonzero(m)
            line = f"  f{fid} {key} wrist score {r['score']:.3f} bbox {r['bbox_xyxy_exclusive']} centroid ({xs.mean():.0f},{ys.mean():.0f}) |"
            for name, X in poses.items():
                uv, z = project(Kw, Ew, X)
                u, v = uv[0]
                inside = 0 <= u < 512 and 0 <= v < 512 and z[0] > 0
                hit = bool(m[int(v), int(u)]) if inside else False
                if inside: line += f" {name}:({u:.0f},{v:.0f}){'IN' if hit else ''}"
            print(line)
