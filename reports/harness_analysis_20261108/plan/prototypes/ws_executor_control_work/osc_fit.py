"""Offline fit of the measured OSC translation response from saved compact step data.

Model per step t (world-frame vectors): s_t = a * s_{t-1} + b * c_t
  s_t = realised grip-site displacement over step t (dxyz)
  c_t = commanded translation offset of step t (act[:3] * TRANSLATION_SCALE_M)
Reads only scratch JSON produced by the W2 verifier; no simulator.
"""
import json, sys, math
import numpy as np

BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
RUNS = {"T04": ("steps_T04.json", 0.10), "T08": ("steps_T08.json", 0.10), "stove": ("steps_stove.json", 0.10),
        "d001": ("steps_d001.json", 0.10), "d003": ("steps_d003.json", 0.10), "c20": ("steps_c20.json", 0.05)}

def load(name):
    return json.load(open(BASE + name))

rows_all = []
for run, (name, scale) in RUNS.items():
    steps = load(name)
    for i in range(1, len(steps)):
        prev, cur = steps[i - 1], steps[i]
        if cur.get("act") is None or prev.get("dxyz") is None or cur.get("dxyz") is None:
            continue
        c = np.asarray(cur["act"][:3]) * scale
        s_prev = np.asarray(prev["dxyz"])
        s = np.asarray(cur["dxyz"])
        rows_all.append((run, cur.get("profile"), cur.get("skill"), c, s_prev, s, cur.get("gap"), cur["act"][6]))

def fit(rows):
    X, Y = [], []
    for _, _, _, c, sp, s, _, _ in rows:
        for k in range(3):
            X.append([sp[k], c[k]]); Y.append(s[k])
    X, Y = np.asarray(X), np.asarray(Y)
    coef, res, *_ = np.linalg.lstsq(X, Y, rcond=None)
    pred = X @ coef
    r2 = 1 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2)
    return coef, r2, len(rows)

for label, sel in [("all", lambda r: True),
                   ("transit", lambda r: r[1] == "transit"),
                   ("contact", lambda r: r[1] == "contact"),
                   ("transit_100mm_runs", lambda r: r[1] == "transit" and r[0] != "c20"),
                   ("contact_100mm_runs", lambda r: r[1] == "contact" and r[0] != "c20"),
                   ("c20 transit", lambda r: r[1] == "transit" and r[0] == "c20")]:
    rows = [r for r in rows_all if sel(r)]
    if len(rows) < 10:
        print(label, "n", len(rows)); continue
    (a, b), r2, n = fit(rows)
    ss = b / (1 - a) if a < 1 else float('nan')
    print(f"{label:22s} n={n:5d} a={a:.4f} b={b:.4f} steady_ratio=b/(1-a)={ss:.4f} R2={r2:.4f}")

# Steady-state realisation ratio |s|/|c| on steps with constant command for >=4 steps (direction cosine > 0.99)
ratios = {"transit": [], "contact": []}
for run, (name, scale) in RUNS.items():
    steps = load(name)
    for i in range(4, len(steps)):
        win = steps[i - 4:i + 1]
        if any(w.get("act") is None or w.get("dxyz") is None for w in win):
            continue
        prof = win[-1].get("profile")
        if prof not in ratios or any(w.get("profile") != prof for w in win):
            continue
        cs = [np.asarray(w["act"][:3]) * scale for w in win]
        n0 = np.linalg.norm(cs[-1])
        if n0 < 1e-4:
            continue
        if any(np.linalg.norm(c) < 0.98 * n0 or np.dot(c, cs[-1]) / (np.linalg.norm(c) * n0) < 0.99 for c in cs):
            continue
        s = np.asarray(win[-1]["dxyz"])
        ratios[prof].append(float(np.dot(s, cs[-1]) / n0 ** 2))
for k, v in ratios.items():
    if v:
        v = np.asarray(v)
        print(f"steady-command along-ratio {k}: n={len(v)} median={np.median(v):.3f} p10={np.percentile(v,10):.3f} p90={np.percentile(v,90):.3f}")

# Free decay: steps where command drops to ~0 translation while moving; ratio s_t/s_{t-1}
dec = []
for run, (name, scale) in RUNS.items():
    steps = load(name)
    for i in range(1, len(steps)):
        p, c = steps[i - 1], steps[i]
        if c.get("act") is None or p.get("dxyz") is None or c.get("dxyz") is None:
            continue
        cmd = np.linalg.norm(np.asarray(c["act"][:3]) * scale)
        sp = np.asarray(p["dxyz"]); s = np.asarray(c["dxyz"])
        if cmd < 1e-5 and np.linalg.norm(sp) > 2e-3:
            dec.append(float(np.dot(s, sp) / np.dot(sp, sp)))
if dec:
    dec = np.asarray(dec)
    print(f"zero-command decay ratio s_t/s_(t-1): n={len(dec)} median={np.median(dec):.3f} p10={np.percentile(dec,10):.3f} p90={np.percentile(dec,90):.3f}")
