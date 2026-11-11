"""Characterise contact-free fire events of the chosen guard (f=0.5, W=3, low rotation)."""
import pickle, math, os, json
import numpy as np
BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
FILES = {"T04_t100": 0.10, "T08_t100": 0.10, "T08_stoveoff": 0.10, "drawer001": 0.10, "drawer003": 0.10,
         "T04_20mm": 0.05, "T04_default": 0.05, "H3_T04": 0.10, "H3_T08": 0.10}
A, B = 0.7114, 0.0732
f, W, cmin = 0.5, 3, 0.001
for name, scale in FILES.items():
    d = pickle.load(open(BASE + name + "_steps.pkl", "rb"))
    keys = sorted(d)
    rows = []
    for k in keys:
        x = d[k]; b, a = np.asarray(x["before_state"]), np.asarray(x["after_state"])
        act = np.asarray(x["norm_action"]); frag = x.get("frag") or {}
        rows.append(dict(k=k, c=act[:3] * scale, cr=act[3:6] * 0.5, s=a[:3] - b[:3], forb=x.get("forb_inc", 0) or 0,
                         allowed=sum((x.get("inc_allowed") or {}).values()), frag=frag,
                         seg=(frag.get("seg_index"), frag.get("skill"), x.get("primitive")), grip=act[6]))
    prev = False
    for i in range(W, len(rows)):
        good = True
        for j in range(i - W + 1, i + 1):
            c = rows[j]["c"]; n = np.linalg.norm(c)
            if n < cmin or rows[j]["seg"] != rows[i]["seg"] or np.linalg.norm(rows[j]["cr"]) > 0.02: good = False; break
            u = c / n
            pred = A * float(np.dot(rows[j - 1]["s"], u)) + B * n
            if not float(np.dot(rows[j]["s"], u)) < f * max(pred, B * n): good = False; break
        free = all(not rows[j]["forb"] and not rows[j]["allowed"] for j in range(max(0, i - W - 1), i + 1))
        fr = good and free
        if fr and not prev:
            fr_ = rows[i]["frag"]
            later = [rows[j]["frag"].get("seg_status") for j in range(i, min(len(rows), i + 40)) if rows[j]["seg"] == rows[i]["seg"]]
            end = later[-1] if later else None
            print(f"{name:12s} step {rows[i]['k']:5d} skill {fr_.get('skill')} pos_err {fr_.get('pos_err') if fr_.get('pos_err') is not None else fr_.get('position_error_m')} |c|={np.linalg.norm(rows[i]['c'])*1000:.2f}mm real={np.dot(rows[i]['s'], rows[i]['c']/np.linalg.norm(rows[i]['c']))*1000:.2f}mm grip={rows[i]['grip']} seg_end={end}")
        prev = fr
