"""Classify every segment_stall from per-step data: contact approach / transit overshoot / transit other.
Overshoot := transit segment whose error minimum (before the stall) is < 10 mm and rose >= 3 mm after it.
Also: pending segments discarded, final speed, steps since minimum."""
import json, sys
import numpy as np
out = {}
for name in sys.argv[1:]:
    rows = json.load(open(f"data/steps_{name}.json"))
    res = []
    for i, r in enumerate(rows):
        if r["reason"] != "segment_stall":
            continue
        # collect this segment's rows backwards
        seg_rows = [r]
        j = i - 1
        while j >= 0 and rows[j]["seg"] == r["seg"] and rows[j]["status"] == "running" and rows[j]["seg_actions"] is not None and rows[j]["seg_actions"] < seg_rows[-1]["seg_actions"]:
            seg_rows.append(rows[j]); j -= 1
        seg_rows.reverse()
        key = "route_rem" if r["route_rem"] is not None else "pos_err"
        err = np.array([(x[key] if x[key] is not None else x["pos_err"]) for x in seg_rows]) * 1000
        imin = int(np.argmin(err))
        rise = float(err[imin:].max() - err[imin]) if imin < len(err) else 0.0
        speeds = [float(np.linalg.norm(x["dxyz"])) * 1000 for x in seg_rows[-4:] if x["dxyz"]]
        cls = ("contact" if r["profile"] == "contact" else
               "transit_overshoot" if err[imin] < 10 and rise >= 3 and imin < len(err) - 1 else "transit_other")
        res.append({"step": r["step"], "skill": r["skill"], "profile": r["profile"], "seg_actions": r["seg_actions"],
                    "err_start": round(float(err[0]), 1), "err_min": round(float(err[imin]), 1), "rise_after_min": round(rise, 1),
                    "err_end": round(float(err[-1]), 1), "steps_since_min": len(err) - 1 - imin,
                    "final_speed_mm": round(float(np.mean(speeds)), 2) if speeds else None,
                    "pending": r["pending"], "n_segs": r["n_segs"], "seg_index": r["seg"], "cls": cls})
    out[name] = res
json.dump(out, open("data/stall_w2.json", "w"), indent=1)
from collections import Counter
tot = Counter(); disc = Counter()
for name, res in out.items():
    c = Counter(x["cls"] for x in res); tot.update(c)
    d = Counter(x["cls"] for x in res if x["pending"]); disc.update(d)
    print(name, len(res), dict(c), "discarded pending:", dict(d))
print("TOTAL", sum(tot.values()), dict(tot), "discarded", dict(disc), sum(disc.values()))
