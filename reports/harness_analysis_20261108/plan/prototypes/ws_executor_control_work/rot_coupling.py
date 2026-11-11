"""Per-step realised/predicted translation ratio vs simultaneous rotation command (contact-free)."""
import pickle
import numpy as np
BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
FILES = {"T04_t100": 0.10, "T08_t100": 0.10, "T08_stoveoff": 0.10, "drawer001": 0.10, "drawer003": 0.10,
         "T04_20mm": 0.05, "T04_default": 0.05, "H3_T04": 0.10, "H3_T08": 0.10}
A, B = 0.7114, 0.0732
buckets = {(0, .02): [], (.02, .04): [], (.04, .07): [], (.07, .2): []}
for name, scale in FILES.items():
    d = pickle.load(open(BASE + name + "_steps.pkl", "rb")); ks = sorted(d); rows = []
    for k in ks:
        x = d[k]; b, a = np.asarray(x["before_state"]), np.asarray(x["after_state"]); act = np.asarray(x["norm_action"])
        frag = x.get("frag") or {}
        rows.append(dict(c=act[:3] * scale, cr=act[3:6] * 0.5, s=a[:3] - b[:3], forb=x.get("forb_inc", 0) or 0,
                         allowed=sum((x.get("inc_allowed") or {}).values()), skill=frag.get("skill")))
    for i in range(1, len(rows)):
        r = rows[i]; n = np.linalg.norm(r["c"])
        if n < 1.6e-3 or r["skill"] == "set_gripper" or r["forb"] or r["allowed"] or rows[i-1]["forb"] or rows[i-1]["allowed"]:
            continue
        u = r["c"] / n
        if np.dot(rows[i - 1]["s"], u) < 0: continue
        pred = A * float(np.dot(rows[i - 1]["s"], u)) + B * n
        q = float(np.dot(r["s"], u)) / pred
        cr = float(np.linalg.norm(r["cr"]))
        for lo_hi in buckets:
            if lo_hi[0] <= cr < lo_hi[1]: buckets[lo_hi].append(q)
for k, v in buckets.items():
    v = np.asarray(v)
    if len(v): print(f"rotation cmd [{k[0]:.2f},{k[1]:.2f}) rad: n={len(v)} p0.5={np.percentile(v,0.5):.3f} p1={np.percentile(v,1):.3f} p5={np.percentile(v,5):.3f} frac<0.5={np.mean(v<0.5):.4f}")
