"""Offline evaluation of a model-relative translation-progress guard on saved H1/H3 steps.
Contact labels (forb_inc / inc_allowed) are diagnostic truth for evaluation only."""
import pickle, math, os
import numpy as np

BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
FILES = {"T04_t100": 0.10, "T08_t100": 0.10, "T08_stoveoff": 0.10, "drawer001": 0.10, "drawer003": 0.10,
         "T04_20mm": 0.05, "T04_default": 0.05, "H3_T04": 0.10, "H3_T08": 0.10}

def load(name, scale):
    d = pickle.load(open(BASE + name + "_steps.pkl", "rb"))
    rows = []
    for k in sorted(d):
        x = d[k]
        b, a = np.asarray(x["before_state"]), np.asarray(x["after_state"])
        act = np.asarray(x["norm_action"])
        frag = x.get("frag") or {}
        rows.append(dict(c=act[:3] * scale, s=a[:3] - b[:3], forb=x.get("forb_inc", 0) or 0,
                         allowed=sum((x.get("inc_allowed") or {}).values()), prim=x.get("primitive"),
                         skill=frag.get("skill"), seg=(frag.get("seg_index"), frag.get("skill")),
                         grip=act[6]))
    return rows

data = {n: load(n, s) for n, s in FILES.items() if os.path.exists(BASE + n + "_steps.pkl")}

# contact-free transit/contact steps for fitting the plant (exclude any contact and the first step)
X, Y = [], []
for rows in data.values():
    for i in range(1, len(rows)):
        if rows[i]["forb"] or rows[i]["allowed"] or rows[i - 1]["forb"] or rows[i - 1]["allowed"]:
            continue
        for k in range(3):
            X.append([rows[i - 1]["s"][k], rows[i]["c"][k]]); Y.append(rows[i]["s"][k])
X, Y = np.asarray(X), np.asarray(Y)
(A, B), *_ = np.linalg.lstsq(X, Y, rcond=None)
r2 = 1 - np.sum((Y - X @ np.array([A, B])) ** 2) / np.sum((Y - Y.mean()) ** 2)
print(f"contact-free plant: a={A:.4f} b={B:.4f} rho=b/(1-a)={B/(1-A):.4f} R2={r2:.4f} n={len(Y)//3}")
resid = Y - X @ np.array([A, B])
print(f"  per-axis residual std {resid.std()*1000:.3f} mm; p99 |res| {np.percentile(np.abs(resid),99)*1000:.3f} mm")

# rest noise: steps with |c|<1e-5 and |s_prev|<2e-4
rest = [np.linalg.norm(r["s"]) for rows in data.values() for i, r in enumerate(rows)
        if i and np.linalg.norm(r["c"]) < 1e-5 and np.linalg.norm(rows[i-1]["s"]) < 2e-4]
if rest:
    rest = np.asarray(rest); print(f"rest |s|: n={len(rest)} median={np.median(rest)*1000:.4f} mm p99={np.percentile(rest,99)*1000:.4f} mm")

def windowed(rows, W, A, B, cmin):
    """Return per-step windowed ratio Q (None when not evaluable)."""
    out = [None] * len(rows)
    for i in range(W, len(rows)):
        num = den = 0.0; ok = True
        for j in range(i - W + 1, i + 1):
            c = rows[j]["c"]; n = np.linalg.norm(c)
            if n < cmin or rows[j]["seg"] != rows[i]["seg"]:
                ok = False; break
            u = c / n
            num += float(np.dot(rows[j]["s"], u))
            den += A * float(np.dot(rows[j - 1]["s"], u)) + B * n
        if ok and den > 1e-4:
            out[i] = num / den
    return out

for W in (2, 3, 4):
    for cmin in (0.001, 0.002):
        free, fires_free_steps, eval_free = [], 0, 0
        for rows in data.values():
            Q = windowed(rows, W, A, B, cmin)
            for i, q in enumerate(Q):
                if q is None: continue
                if all(not rows[j]["forb"] and not rows[j]["allowed"] for j in range(i - W + 1, i + 1)):
                    free.append(q)
        free = np.asarray(free)
        print(f"W={W} cmin={cmin*1000:.0f}mm free windows n={len(free)} p0.5={np.percentile(free,0.5):.3f} p1={np.percentile(free,1):.3f} p2={np.percentile(free,2):.3f} p5={np.percentile(free,5):.3f}")

# Forbidden-episode detection with W=3, cmin=1mm, thr = free p1
W, cmin = 3, 0.001
free = []
Qs = {}
for name, rows in data.items():
    Q = windowed(rows, W, A, B, cmin); Qs[name] = Q
    for i, q in enumerate(Q):
        if q is not None and all(not rows[j]["forb"] and not rows[j]["allowed"] for j in range(i - W + 1, i + 1)):
            free.append(q)
thr = float(np.percentile(free, 1))
print(f"threshold q_min = free p1 = {thr:.3f}")
eps, detected, delays, samples_total, samples_after = 0, 0, [], 0, 0
for name, rows in data.items():
    Q = Qs[name]; i = 0
    while i < len(rows):
        if rows[i]["forb"] > 0:
            j = i
            while j + 1 < len(rows) and rows[j + 1]["forb"] > 0:
                j += 1
            eps += 1
            tot = sum(rows[k]["forb"] for k in range(i, j + 1)); samples_total += tot
            fire = next((k for k in range(i, j + 1) if Q[k] is not None and Q[k] < thr), None)
            if fire is not None:
                detected += 1; delays.append(fire - i)
                samples_after += sum(rows[k]["forb"] for k in range(fire + 1, j + 1))
            i = j + 1
        else:
            i += 1
print(f"forbidden episodes {eps}: guard fires inside {detected}; delay median {np.median(delays) if delays else None} steps; samples after fire {samples_after}/{samples_total}")
# false fires: contact-free windows below threshold, grouped into events
events = 0
for name, rows in data.items():
    Q = Qs[name]; prev = False
    for i, q in enumerate(Q):
        f = q is not None and q < thr and all(not rows[j]["forb"] and not rows[j]["allowed"] for j in range(i - W + 1, i + 1))
        if f and not prev: events += 1
        prev = f
print(f"contact-free fire events at q_min: {events}")
