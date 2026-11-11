"""Per contact-approach segment: steps and path; re-estimate steps with a fast part (7.5 mm/step) and a 20 mm slow tail."""
import json, sys
import numpy as np
for name in sys.argv[1:]:
    rows = json.load(open(f"data/steps_{name}.json"))
    segs, cur, key = [], [], None
    for r in rows:
        k = (r["seg"], r["skill"], r["profile"])
        if r["skill"] == "approach":
            if key == k and cur and r["seg_actions"] is not None and r["seg_actions"] > cur[-1]["seg_actions"]:
                cur.append(r)
            else:
                if cur: segs.append(cur)
                cur, key = [r], k
        else:
            if cur: segs.append(cur); cur, key = [], None
    if cur: segs.append(cur)
    saved = 0; tot = 0
    for s in segs:
        sp = np.array([np.linalg.norm(x["dxyz"]) * 1000 for x in s if x["dxyz"]])
        d = sp.sum(); n = len(s)
        # moving steps at ~contact speed; non-moving (stall/settle) steps kept
        moving = int((sp > 0.3).sum())
        slow_tail = min(d, 20.0) / max(sp[sp > 0.3].mean() if moving else 1.0, 0.3)
        new_moving = max(d - 20.0, 0) / 7.5 + slow_tail
        saved += max(moving - new_moving, 0); tot += n
    print(name, "approach segments", len(segs), "approach actions", tot, "estimated saving %.0f actions" % saved)
