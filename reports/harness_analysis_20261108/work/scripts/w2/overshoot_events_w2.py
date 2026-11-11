"""Per transit motion segment: did the error pass a minimum < 10 mm and then rise >= 3 mm (overshoot event)?
Also peak speed and the rise size; counts per run, regardless of how the segment ended."""
import json, sys
import numpy as np
for name in sys.argv[1:]:
    rows = json.load(open(f"data/steps_{name}.json"))
    segs, cur = [], []
    for r in rows:
        if r["profile"] == "transit" and r["skill"] in ("move_to_pose", "move_relative"):
            if cur and r["seg"] == cur[-1]["seg"] and r["seg_actions"] is not None and cur[-1]["seg_actions"] is not None and r["seg_actions"] > cur[-1]["seg_actions"]:
                cur.append(r)
            else:
                if cur: segs.append(cur)
                cur = [r]
        else:
            if cur: segs.append(cur); cur = []
    if cur: segs.append(cur)
    ev, rises, peaks, ends = 0, [], [], {}
    for s in segs:
        e = np.array([x["pos_err"] for x in s]) * 1000
        sp = np.array([np.linalg.norm(x["dxyz"]) * 1000 if x["dxyz"] else 0 for x in s])
        i = int(np.argmin(e)); rise = e[i:].max() - e[i]
        peaks.append(sp.max())
        if e[i] < 10 and rise >= 3:
            ev += 1; rises.append(rise)
            k = s[-1]["reason"] or s[-1]["seg_status"]; ends[k] = ends.get(k, 0) + 1
    print(name, "transit segments", len(segs), "overshoot events", ev, "median rise %.1f mm" % (np.median(rises) if rises else 0),
          "peak speed p50/max %.1f/%.1f mm/step" % (np.median(peaks), max(peaks)), "segment end of overshooting segments:", ends)
