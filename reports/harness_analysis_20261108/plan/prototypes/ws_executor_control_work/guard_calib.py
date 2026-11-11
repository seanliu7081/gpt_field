"""Calibration statistics for the guard: per-step realised/predicted progress on contact-free,
translation-dominant, non-gripper steps with a meaningful prediction; plus the event rate per W."""
import pickle, os
import numpy as np
BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
FILES = {"T04_t100": 0.10, "T08_t100": 0.10, "T08_stoveoff": 0.10, "drawer001": 0.10, "drawer003": 0.10,
         "T04_20mm": 0.05, "T04_default": 0.05, "H3_T04": 0.10, "H3_T08": 0.10}
A, B = 0.7114, 0.0732
ratios = []; evaluable = 0
per_run = {}
for name, scale in FILES.items():
    d = pickle.load(open(BASE + name + "_steps.pkl", "rb"))
    keys = sorted(d); rows = []
    for k in keys:
        x = d[k]; b, a = np.asarray(x["before_state"]), np.asarray(x["after_state"])
        act = np.asarray(x["norm_action"]); frag = x.get("frag") or {}
        rows.append(dict(c=act[:3] * scale, cr=act[3:6] * 0.5, s=a[:3] - b[:3], forb=x.get("forb_inc", 0) or 0,
                         allowed=sum((x.get("inc_allowed") or {}).values()), skill=frag.get("skill"),
                         seg=(frag.get("seg_index"), frag.get("skill"), x.get("primitive"))))
    for i in range(1, len(rows)):
        r = rows[i]; n = np.linalg.norm(r["c"])
        if n < 1e-3 or r["skill"] == "set_gripper" or np.linalg.norm(r["cr"]) > 0.02: continue
        if r["forb"] or r["allowed"] or rows[i - 1]["forb"] or rows[i - 1]["allowed"]: continue
        u = r["c"] / n; pred = A * float(np.dot(rows[i - 1]["s"], u)) + B * n
        if pred < B * n * 0.5: continue           # prediction dominated by reversing momentum: not evaluable
        evaluable += 1
        ratios.append(float(np.dot(r["s"], u)) / pred)
        per_run[name] = per_run.get(name, 0) + 1
q = np.asarray(ratios)
print(f"evaluable contact-free translation steps {evaluable}; per run {per_run}")
for p in (0.1, 0.5, 1, 2, 5, 10, 50):
    print(f"  ratio p{p}: {np.percentile(q, p):.3f}")
print(f"  fraction < 0.5: {np.mean(q < 0.5):.4f}; < 0.3: {np.mean(q < 0.3):.4f}")
