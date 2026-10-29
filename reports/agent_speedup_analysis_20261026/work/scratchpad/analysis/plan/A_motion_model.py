"""Plan A: motion model for transit/contact command caps (ESTIMATES; read-only; python3 -I).

Physics stand-in: per-channel unit-mass OSC with kp=150, kd=2*sqrt(kp) (simulator.json: kp 150, damping_ratio 1,
control_delta, 20 Hz), goal re-set each 50 ms control step to x_k + c_k (zero-order hold), integrated at 2 ms.
Controller rule copied from fragments.py:_next (clip(target - x, cap)) and the arrival predicate
(2 consecutive samples within tolerance; fragments.py:203-206, 291-293). Rotation channel uses the 0.1 rad
per-step clamp (fragments.py:185) and the same dynamics. Straight-line motion assumed.
Validation: predicted steps at the recorded caps vs recorded steps for arrived, single-channel-dominated segments.
"""
import json, math, statistics as st, collections

P = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/"
S = json.load(open(P + "A_segments.json"))
KP = 150.0
KD = 2 * math.sqrt(KP)
DT = 0.002
SUB = 25
TOL = {"transit": (0.004, 0.06), "contact": (0.002, 0.04)}
CAP = {"transit": 0.02, "contact": 0.005}


def sim_channel(E0, cap, tol, eff=1.0, max_steps=400):
    """Return per-step error list for a single channel starting at rest with error E0."""
    x, v = 0.0, 0.0
    errs = []
    for k in range(max_steps):
        e = E0 - x
        c = max(-cap, min(cap, e))
        g = x + c
        for _ in range(SUB):
            acc = eff * KP * (g - x) - KD * v
            v += acc * DT
            x += v * DT
        errs.append(abs(E0 - x))
    return errs


def steps_to_arrive(E0, R0, cap, rcap, tol, rtol, eff=1.0):
    et = sim_channel(E0, cap, tol, eff)
    er = sim_channel(R0, rcap, rtol, eff) if R0 > 0 else [0.0] * len(et)
    run = 0
    for k, (a, b) in enumerate(zip(et, er)):
        ok = a <= tol and b <= rtol
        run = run + 1 if ok else 0
        if run >= 2:
            return k + 1
    return None


# steady-state check
e = sim_channel(1.0, 0.02, 0.004)
d = [e[k] - e[k+1] for k in range(len(e) - 1)]
print("model steady-state realized per step at 20 mm cap: %.2f mm (ratio %.3f)" % (1000 * d[30], d[30] / 0.02))
e = sim_channel(1.0, 0.05, 0.004)
d = [e[k] - e[k+1] for k in range(len(e) - 1)]
print("model steady-state realized per step at 50 mm cap: %.2f mm (ratio %.3f)" % (1000 * d[20], d[20] / 0.05))

# measured steady state on unobstructed arrived transit segments (steps 6..n-8 at the cap)
ss = [a for s in S if s["skill"] in ("move_to_pose", "move_relative") and s["profile"] == "transit" and s["reason"] == "pose_arrived"
      for i, (a, c) in enumerate(zip(s["along"], s["cmd"])) if i >= 5 and c >= 0.0199]
print("measured transit at cap after 5 steps: n %d mean %.2f mm/step median %.2f" % (len(ss), 1000 * st.mean(ss), 1000 * st.median(ss)))
ssc = [a for s in S if s["skill"] in ("move_to_pose", "move_relative", "approach") and s["profile"] == "contact" and s["reason"] == "pose_arrived"
       for i, (a, c) in enumerate(zip(s["along"], s["cmd"])) if i >= 5 and c >= 0.00499]
print("measured contact at cap after 5 steps (arrived segs): n %d mean %.3f mm/step" % (len(ssc), 1000 * st.mean(ssc)))

# efficiency factor fitted so the model reproduces the measured transit steady state
target = st.median(ss)
best = None
for eff in [x / 100 for x in range(60, 121)]:
    e = sim_channel(1.0, 0.02, 0.004, eff)
    dd = e[30] - e[31]
    if best is None or abs(dd - target) < abs(best[1] - target):
        best = (eff, dd)
EFF = best[0]
print("fitted kp efficiency %.2f -> %.2f mm/step" % (EFF, 1000 * best[1]))

# validation and prediction per arrived motion segment
rows = []
for s in S:
    if s["skill"] not in ("move_to_pose", "move_relative", "approach") or s["reason"] != "pose_arrived":
        continue
    prof = s["profile"]
    tol, rtol = TOL[prof]
    pred = steps_to_arrive(s["E0"], s["R0"], CAP[prof], 0.1, tol, rtol, EFF)
    rows.append(dict(run=s["run"], prof=prof, E0=s["E0"], R0=s["R0"], actual=s["steps"], pred=pred,
                     t35=steps_to_arrive(s["E0"], s["R0"], 0.035, 0.1, tol, rtol, EFF) if prof == "transit" else None,
                     t50=steps_to_arrive(s["E0"], s["R0"], 0.05, 0.1, tol, rtol, EFF) if prof == "transit" else None,
                     t50r=steps_to_arrive(s["E0"], s["R0"], 0.05, 0.25, tol, rtol, EFF) if prof == "transit" else None,
                     c8=steps_to_arrive(s["E0"], s["R0"], 0.008, 0.1, tol, rtol, EFF) if prof == "contact" else None,
                     c10=steps_to_arrive(s["E0"], s["R0"], 0.010, 0.1, tol, rtol, EFF) if prof == "contact" else None))
for prof in ("transit", "contact"):
    g = [r for r in rows if r["prof"] == prof and r["pred"]]
    act = sum(r["actual"] for r in g); pr = sum(r["pred"] for r in g)
    errs = [r["pred"] - r["actual"] for r in g]
    print("%s arrived segs n %d actual steps %d model@recorded cap %d (ratio %.3f) median err %+.1f" % (
        prof, len(g), act, pr, pr / act, st.median(errs)))
    big_r = [r for r in g if r["R0"] > 0.3]
    print("   segments with R0 > 0.3 rad: %d, steps %d" % (len(big_r), sum(r["actual"] for r in big_r)))
    if prof == "transit":
        for col in ("t35", "t50", "t50r"):
            new = sum(r[col] for r in g)
            # scale model prediction by actual/pred ratio (calibration) per segment
            cal = sum(r["actual"] * r[col] / r["pred"] for r in g)
            print("   %s model %d (raw saving %d steps); calibrated %.0f (saving %.0f = %.1f%% of arrived transit steps)" % (
                col, new, pr - new, cal, act - cal, 100 * (act - cal) / act))
    else:
        for col in ("c8", "c10"):
            new = sum(r[col] for r in g)
            cal = sum(r["actual"] * r[col] / r["pred"] for r in g)
            print("   %s model %d; calibrated %.0f (saving %.0f = %.1f%% of arrived contact steps)" % (
                col, new, cal, act - cal, 100 * (act - cal) / act))
json.dump(dict(eff=EFF, rows=rows), open(P + "A_motion_model.json", "w"), indent=0)

# per-run saving (calibrated) for the transit 50 mm option, arrived segments only
per = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    if not r["pred"]:
        continue
    if r["prof"] == "transit":
        per[r["run"]]["t50"] += r["actual"] - r["actual"] * r["t50"] / r["pred"]
        per[r["run"]]["t35"] += r["actual"] - r["actual"] * r["t35"] / r["pred"]
    else:
        per[r["run"]]["c8"] += r["actual"] - r["actual"] * r["c8"] / r["pred"]
        per[r["run"]]["c10"] += r["actual"] - r["actual"] * r["c10"] / r["pred"]
print("per-run calibrated step savings (arrived segments only):")
for run in sorted(per):
    print("  ", run, {k: round(v) for k, v in per[run].items()})
json.dump({k: dict(v) for k, v in per.items()}, open(P + "A_motion_saving.json", "w"), indent=0)
