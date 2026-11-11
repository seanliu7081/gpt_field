"""Joint-limit proximity at each executive request (Panda limits from franka docs / robosuite xml)."""
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from calls import load, request_fields
LIM = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698),
       (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]
for run in sys.argv[1:]:
    near = []
    n = 0
    for rp in sorted(glob.glob(os.path.join(run, "trial", "requests", "*.request.json"))):
        r = load(rp)
        if r.get("mode") != "agent":
            continue
        f = request_fields(r["payload"]) or {}
        q = (f.get("context", {}).get("measured_state") or {}).get("joint_position_rad")
        if not q:
            continue
        n += 1
        margins = [min(qi - lo, hi - qi) for qi, (lo, hi) in zip(q, LIM)]
        j = min(range(7), key=lambda i: margins[i])
        if margins[j] < 0.05:
            near.append((os.path.basename(rp)[:6], f.get("step"), j + 1, round(margins[j], 4)))
    print(os.path.basename(run)[22:], f"requests={n}", "within 0.05 rad of a joint limit:", len(near), near[:12])
