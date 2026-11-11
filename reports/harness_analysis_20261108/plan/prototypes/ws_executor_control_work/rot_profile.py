"""Rotation-lag and orientation-overshoot statistics from compact steps; plus a plant simulation of
current vs braking-aware translation tracking (fitted first-order OSC model)."""
import json, math
import numpy as np
BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
runs = {"T04": "steps_T04.json", "T08": "steps_T08.json", "stove": "steps_stove.json", "d001": "steps_d001.json", "d003": "steps_d003.json"}
over = 0; segs = 0; lag = []
for r, f in runs.items():
    steps = json.load(open(BASE + f))
    cur, buf = None, []
    def flush(buf):
        global over, segs
        if len(buf) < 5: return
        segs += 1
        oe = [b["ori_err"] for b in buf if b["ori_err"] is not None]
        pe = [b["pos_err"] for b in buf if b["pos_err"] is not None]
        prof = buf[0]["profile"]; tol = .06 if prof == "transit" else .04
        if oe and min(oe) < tol and max(oe[oe.index(min(oe)):]) > tol + 0.02: over += 1
        # rotation lag: fraction of translation done when rotation is 50% done
        if oe and pe and oe[0] > 0.3 and pe[0] > 0.05:
            k_r = next((i for i, v in enumerate(oe) if v < 0.5 * oe[0]), None)
            k_t = next((i for i, v in enumerate(pe) if v < 0.5 * pe[0]), None)
            if k_r is not None and k_t is not None: lag.append((k_r, k_t, oe[0], pe[0]))
    for s in steps:
        key = (s.get("seg"), s.get("skill"), s.get("status"))
        if s.get("prim") != "execute_fragment" or s.get("seg") is None:
            flush(buf); buf = []; cur = None; continue
        if key[:2] != (cur or (None, None))[:2] or (buf and s.get("seg_actions", 0) < buf[-1].get("seg_actions", 0)):
            flush(buf); buf = []
        cur = key; buf.append(s)
    flush(buf)
print(f"segments {segs}; orientation overshoot (dip below tol then +0.02 rad): {over}")
for k_r, k_t, o0, p0 in lag:
    print(f"  combined stage: rot0={o0:.2f} rad pos0={p0*1000:.0f} mm  steps to half-rotation {k_r} vs half-translation {k_t}")

# --- plant simulation (fitted contact-free model a=0.7114 b=0.0732) for a 0.23 m straight transit
a, b, cap, tol = 0.7114, 0.0732, 0.10, 0.004
def run(law, d0=0.23, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x, s, steps, arrived, max_over = 0.0, 0.0, 0, 0, 0.0
    while steps < 200:
        r = d0 - x
        if law == "current":
            c = min(max(r, -cap), cap) if abs(r) > tol else 0.0          # deadband = tolerance (tracking.py:90-95)
        else:
            e = tol / 2
            target_step = (1 - a) * (r - math.copysign(e, r)) if abs(r) > e else 0.0
            c = (target_step - a * s) / b
            c = max(-cap, min(cap, c))
        s = a * s + b * c + noise * rng.normal()
        x += s; steps += 1
        max_over = max(max_over, x - d0)
        arrived = arrived + 1 if abs(d0 - x) <= tol else 0
        if arrived >= 2: break
    return steps, max_over
for law in ("current", "braking"):
    res = [run(law, noise=0.0006, seed=k) for k in range(200)]
    st = np.array([r[0] for r in res]); ov = np.array([r[1] for r in res])
    print(f"{law:8s}: steps to 2-sample arrival median {np.median(st):.0f} p90 {np.percentile(st,90):.0f}; max overshoot median {np.median(ov)*1000:.1f} mm p90 {np.percentile(ov,90)*1000:.1f} mm")
