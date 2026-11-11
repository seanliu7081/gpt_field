"""Shortfall-based guard: D_w = sum(pred - real) along the command over W steps.
Threshold per predicted-speed band = high quantile of contact-free D_w (measured OSC noise)."""
import pickle, math, os
import numpy as np
BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
FILES = {"T04_t100": 0.10, "T08_t100": 0.10, "T08_stoveoff": 0.10, "drawer001": 0.10, "drawer003": 0.10,
         "T04_20mm": 0.05, "T04_default": 0.05, "H3_T04": 0.10, "H3_T08": 0.10}
A, B = 0.7114, 0.0732

def load(name, scale):
    d = pickle.load(open(BASE + name + "_steps.pkl", "rb"))
    rows = []
    for k in sorted(d):
        x = d[k]; b, a = np.asarray(x["before_state"]), np.asarray(x["after_state"])
        act = np.asarray(x["norm_action"]); frag = x.get("frag") or {}
        rows.append(dict(c=act[:3] * scale, s=a[:3] - b[:3], forb=x.get("forb_inc", 0) or 0,
                         allowed=sum((x.get("inc_allowed") or {}).values()),
                         seg=(frag.get("seg_index"), frag.get("skill"), x.get("primitive")), grip=act[6],
                         prim=x.get("primitive")))
    return rows
data = {n: load(n, s) for n, s in FILES.items() if os.path.exists(BASE + n + "_steps.pkl")}

def win(rows, i, W, cmin):
    P = R = 0.0
    for j in range(i - W + 1, i + 1):
        c = rows[j]["c"]; n = np.linalg.norm(c)
        if n < cmin or rows[j]["seg"] != rows[i]["seg"]:
            return None
        u = c / n
        P += A * float(np.dot(rows[j - 1]["s"], u)) + B * n
        R += float(np.dot(rows[j]["s"], u))
    return P, R

W, cmin = 3, 0.001
bands = [(0, 0.002), (0.002, 0.006), (0.006, 0.02), (0.02, 1.0)]   # predicted window progress (m)
free = {bd: [] for bd in bands}
allw = {}
for name, rows in data.items():
    allw[name] = [None] * len(rows)
    for i in range(W, len(rows)):
        r = win(rows, i, W, cmin)
        if r is None: continue
        allw[name][i] = r
        P, R = r
        if all(not rows[j]["forb"] and not rows[j]["allowed"] for j in range(i - W - 1, i + 1)):
            for bd in bands:
                if bd[0] <= P < bd[1]:
                    free[bd].append(P - R)
thr = {}
for bd, v in free.items():
    v = np.asarray(v)
    if len(v) < 20: thr[bd] = None; print(bd, len(v)); continue
    thr[bd] = float(np.percentile(v, 99.5))
    print(f"band P in [{bd[0]*1000:.0f},{bd[1]*1000:.0f}) mm: n={len(v)} D p50={np.median(v)*1000:.2f} p99={np.percentile(v,99)*1000:.2f} p99.5={thr[bd]*1000:.2f} max={v.max()*1000:.2f} mm; P-relative p99.5 of D/P={np.percentile(v/ max(1e-9, (bd[0]+bd[1])/2),99.5):.2f}")

def fires(P, R):
    for bd in bands:
        if bd[0] <= P < bd[1] and thr[bd] is not None:
            return (P - R) > thr[bd] and R < 0.5 * P
    return False

eps = det = 0; delays = []; tot = after = 0; per_run = {}
for name, rows in data.items():
    i = 0
    while i < len(rows):
        if rows[i]["forb"] > 0:
            j = i
            while j + 1 < len(rows) and rows[j + 1]["forb"] > 0: j += 1
            eps += 1; n_s = sum(rows[k]["forb"] for k in range(i, j + 1)); tot += n_s
            f = next((k for k in range(i, j + 1) if allw[name][k] is not None and fires(*allw[name][k])), None)
            moving = sum(1 for k in range(i, j + 1) if np.linalg.norm(rows[k]["c"]) >= cmin)
            per_run.setdefault(name, []).append((i, j, n_s, f, moving, rows[i]["prim"]))
            if f is not None:
                det += 1; delays.append(f - i); after += sum(rows[k]["forb"] for k in range(f + 1, j + 1))
            i = j + 1
        else: i += 1
print(f"forbidden episodes {eps}: detected {det}, delay median {np.median(delays) if delays else None}, samples after fire {after}/{tot}")
for name, v in per_run.items():
    for (i, j, n_s, f, moving, prim) in v:
        print(f"  {name:12s} steps {i}-{j} samples {n_s:5d} cmd>=1mm steps {moving:3d} fire_at {f} prim {prim}")
ev = 0
for name, rows in data.items():
    prev = False
    for i in range(len(rows)):
        r = allw[name][i]
        f = r is not None and fires(*r) and all(not rows[j]["forb"] and not rows[j]["allowed"] for j in range(i - W - 1, i + 1))
        if f and not prev: ev += 1
        prev = f
print("contact-free fire events:", ev)
