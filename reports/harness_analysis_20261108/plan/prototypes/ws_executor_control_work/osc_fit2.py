"""Rotation response fit + translation-progress guard statistics from saved per-step pickles.
Offline, read-only; contact labels are diagnostic truth used only for evaluation."""
import pickle, math, glob, os
import numpy as np

BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
FILES = {"T04_t100": 0.10, "T08_t100": 0.10, "T08_stoveoff": 0.10, "drawer001": 0.10, "drawer003": 0.10,
         "T04_20mm": 0.05, "T04_default": 0.05, "H3_T04": 0.10, "H3_T08": 0.10}

def rotm(rv):
    rv = np.asarray(rv, float); th = np.linalg.norm(rv)
    if th < 1e-12: return np.eye(3)
    k = rv / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def logm(R):
    c = (np.trace(R) - 1) / 2; c = min(1, max(-1, c)); th = math.acos(c)
    if th < 1e-9: return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th

data = {}
for name, scale in FILES.items():
    path = BASE + name + "_steps.pkl"
    if not os.path.exists(path):
        print("missing", name); continue
    d = pickle.load(open(path, "rb"))
    rows = []
    for k in sorted(d):
        x = d[k]
        b, a = np.asarray(x["before_state"]), np.asarray(x["after_state"])
        act = np.asarray(x["norm_action"])
        c_t = act[:3] * scale
        c_r = act[3:6] * 0.5
        s_t = a[:3] - b[:3]
        s_r = logm(rotm(a[3:6]) @ rotm(b[3:6]).T)   # world-frame realised rotation vector
        frag = x.get("frag") or {}
        rows.append(dict(c_t=c_t, c_r=c_r, s_t=s_t, s_r=s_r, forb=x.get("forb_inc", 0) or 0,
                         allowed=sum((x.get("inc_allowed") or {}).values()),
                         prim=x.get("primitive"), skill=frag.get("skill"), grip=act[6],
                         gap=x.get("finger_gap_mm")))
    data[name] = rows

# --- rotation fit: s_r_t = a_r s_r_{t-1} + b_r c_r_t
X, Y = [], []
for name, rows in data.items():
    for i in range(1, len(rows)):
        for k in range(3):
            X.append([rows[i - 1]["s_r"][k], rows[i]["c_r"][k]]); Y.append(rows[i]["s_r"][k])
X, Y = np.asarray(X), np.asarray(Y)
coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
r2 = 1 - np.sum((Y - X @ coef) ** 2) / np.sum((Y - Y.mean()) ** 2)
print(f"rotation fit: a_r={coef[0]:.4f} b_r={coef[1]:.4f} steady={coef[1]/(1-coef[0]):.4f} R2={r2:.4f} n={len(Y)//3}")
# steady rotation rate at saturated 0.1 rad command
rates = []
for rows in data.values():
    for i in range(3, len(rows)):
        cs = [np.linalg.norm(rows[j]["c_r"]) for j in range(i - 3, i + 1)]
        if min(cs) > 0.099:
            rates.append(np.linalg.norm(rows[i]["s_r"]))
rates = np.asarray(rates)
print(f"saturated-rotation realised rate: n={len(rates)} median={np.median(rates):.4f} rad/step p10={np.percentile(rates,10):.4f} p90={np.percentile(rates,90):.4f}")

# --- translation fit per run (per-scale) for completeness
for name, rows in data.items():
    X, Y = [], []
    for i in range(1, len(rows)):
        for k in range(3):
            X.append([rows[i - 1]["s_t"][k], rows[i]["c_t"][k]]); Y.append(rows[i]["s_t"][k])
    X, Y = np.asarray(X), np.asarray(Y)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r2 = 1 - np.sum((Y - X @ coef) ** 2) / np.sum((Y - Y.mean()) ** 2)
    print(f"  translation {name:12s} a={coef[0]:.3f} b={coef[1]:.4f} ratio={coef[1]/(1-coef[0]):.3f} R2={r2:.3f}")

# --- model-relative progress ratio along the command: q_t = <s_t, u_t> / pred_t,
#     pred_t = a <s_{t-1}, u_t> + b |c_t|  with pooled transit/contact constants
A, B = 0.7229, 0.0692
qs_free, qs_contact = [], []
for name, rows in data.items():
    for i in range(1, len(rows)):
        c = rows[i]["c_t"]; n = np.linalg.norm(c)
        if n < 1e-3:       # >= 1 mm commanded
            continue
        u = c / n
        pred = A * float(np.dot(rows[i - 1]["s_t"], u)) + B * n
        if pred < 2e-4:
            continue
        q = float(np.dot(rows[i]["s_t"], u)) / pred
        if rows[i]["forb"] > 0 or rows[i]["allowed"] > 0:
            qs_contact.append(q)
        else:
            qs_free.append(q)
qf, qc = np.asarray(qs_free), np.asarray(qs_contact)
print(f"progress ratio q (free): n={len(qf)} p1={np.percentile(qf,1):.3f} p5={np.percentile(qf,5):.3f} median={np.median(qf):.3f}")
print(f"progress ratio q (any contact): n={len(qc)} p50={np.median(qc):.3f} p75={np.percentile(qc,75):.3f} p90={np.percentile(qc,90):.3f}")
for thr in (0.2, 0.3, 0.4, 0.5):
    print(f"  q<{thr}: free {np.mean(qf<thr):.4f}  contact {np.mean(qc<thr):.4f}")
