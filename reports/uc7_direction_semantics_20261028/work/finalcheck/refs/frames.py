"""Read-only: external right/away from the first executive request of each run dir given."""
import json, sys, glob, os
import numpy as np
for run in sys.argv[1:]:
    fs = sorted(glob.glob(os.path.join(run, "**/requests/000001-*.request.json"), recursive=True))
    fs = [f for f in fs if "/review/" not in f] or fs
    if not fs:
        print(run, "no request"); continue
    f = fs[0]
    d = json.load(open(f))
    try:
        j = json.loads(d["payload"]["input"][0]["content"][0]["text"])
        g = j.get("geometry")
    except Exception as exc:
        g = None
    if not g:
        print(os.path.basename(run), "no geometry in", f.split("runs/")[-1]); continue
    c = g["cameras"]["external"]
    R = np.array(c["rotation_world_from_camera"])
    r = R[:2, 0] / np.linalg.norm(R[:2, 0]); a = np.array([-r[1], r[0]])
    print(os.path.basename(run), c["mujoco_name"], c["pose_source"], "pos", np.round(c["position_world_m"], 4).tolist(),
          "right", np.round(r, 4).tolist(), "away", np.round(a, 4).tolist(), "fwd.away", round(float(a @ (-R[:2, 2])), 3))
