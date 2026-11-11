"""First-order OSC translation response s_t = a*s_{t-1} + b*c_t (world vectors), per run, plus derived
braking distance d_stop(s) = s*a/(1-a) and rotation realisation. Offline, compact pkl data only."""
import pickle, json, numpy as np
from pathlib import Path
DATA = Path("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data")
RUNS = {"T04_t100": ("T04_t100_steps.pkl", .10), "T08_t100": ("T08_t100_steps.pkl", .10),
        "T08_stoveoff": ("T08_stoveoff_steps.pkl", .10), "drawer001": ("drawer001_steps.pkl", .10),
        "drawer003": ("drawer003_steps.pkl", .10), "H3_T04": ("H3_T04_steps.pkl", .10), "H3_T08": ("H3_T08_steps.pkl", .10)}
out = {}
X, Y = [], []
rot_rates = []
for run, (pkl, scale) in RUNS.items():
    d = pickle.load(open(DATA / pkl, "rb"))
    keys = sorted(d)
    Xr, Yr = [], []
    for k0, k1 in zip(keys[:-1], keys[1:]):
        a, b = d[k0], d[k1]
        if b.get("measured_delta") is None or a.get("measured_delta") is None:
            continue
        c = np.asarray(b["action"][:3]) * scale
        sp, s = np.asarray(a["measured_delta"]), np.asarray(b["measured_delta"])
        for i in range(3):
            Xr.append([sp[i], c[i]]); Yr.append(s[i])
        # rotation realisation when the rotation command is saturated (|rot cmd| ~ 0.1 rad -> action 0.2)
        rc = np.linalg.norm(np.asarray(b["action"][3:6])) * 0.5
        if rc > 0.095:
            R0 = np.asarray(b["state"][3:6]); R1 = np.asarray(b["next_state"][3:6])
            rot_rates.append(float(np.linalg.norm(R1 - R0)))
    Xr, Yr = np.asarray(Xr), np.asarray(Yr)
    coef, *_ = np.linalg.lstsq(Xr, Yr, rcond=None)
    pred = Xr @ coef
    r2 = 1 - ((Yr - pred) ** 2).sum() / ((Yr - Yr.mean()) ** 2).sum()
    out[run] = {"a": round(float(coef[0]), 4), "b": round(float(coef[1]), 4), "R2": round(float(r2), 3),
                "steady_ratio_b_over_1_minus_a": round(float(coef[1] / (1 - coef[0])), 3),
                "brake_steps_a_over_1_minus_a": round(float(coef[0] / (1 - coef[0])), 2)}
    X.append(Xr); Y.append(Yr)
X, Y = np.concatenate(X), np.concatenate(Y)
coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
out["pooled"] = {"a": round(float(coef[0]), 4), "b": round(float(coef[1]), 4),
                 "steady_ratio": round(float(coef[1] / (1 - coef[0])), 3),
                 "brake_steps_a_over_1_minus_a": round(float(coef[0] / (1 - coef[0])), 2)}
rr = np.asarray(rot_rates)
out["rotation_rotvec_change_per_step_when_cmd_saturated"] = {"n": int(len(rr)), "median_rad": round(float(np.median(rr)), 4),
                                                             "p10": round(float(np.percentile(rr, 10)), 4), "p90": round(float(np.percentile(rr, 90)), 4)}
print(json.dumps(out, indent=1))
