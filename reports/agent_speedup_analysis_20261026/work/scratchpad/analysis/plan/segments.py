"""Per-segment motion model over uc7 runs (from the verifier's per-step speed rows).
Outputs, per run: steps by profile, arrived/stalled segments, and step deltas under alternative caps
using a first-order tracking model fitted to the measured ratio (estimate, labelled)."""
import json, math, sys, collections
V4 = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v4/"
RUNS = "t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()
CAP = {"transit": 0.020, "contact": 0.005}
TOL = {"transit": 0.004, "contact": 0.002}

def segments(rows):
    segs, cur, frag, last_fs = [], None, 0, None
    for r in rows:
        if r.get("skill") is None:
            if cur: segs.append(cur); cur = None
            continue
        if last_fs is None or r["fs"] <= last_fs:
            frag += 1
        last_fs = r["fs"]
        key = (frag, r["seg"])
        if cur is None or cur["key"] != key:
            if cur: segs.append(cur)
            cur = {"key": key, "skill": r["skill"], "profile": r["profile"], "rows": []}
        cur["rows"].append(r)
    if cur: segs.append(cur)
    return segs

def n_model(E0, cap, r, tol):
    lam = -math.log(1 - r)
    if E0 <= tol: return 0.0
    if E0 > cap:
        return (E0 - cap) / (r * cap) + math.log(cap / tol) / lam
    return math.log(E0 / tol) / lam

out = {}
for t in RUNS:
    rows = json.load(open(V4 + "speed_rows_%s.json" % t))
    segs = segments(rows)
    stat = collections.Counter()
    deltas = collections.defaultdict(float)
    for s in segs:
        n = len(s["rows"]); prof = s["profile"]; skill = s["skill"]
        reason = s["rows"][-1].get("seg_reason")
        stat[f"steps|{skill}|{prof}"] += n
        if skill not in ("move_to_pose", "move_relative", "approach"):
            continue
        path = sum(x["real_along"] for x in s["rows"])
        capped = [x for x in s["rows"] if x["cmd"] >= CAP[prof] * 0.999]
        stat[f"capped|{prof}"] += len(capped)
        stat[f"path_mm|{prof}"] += 1000 * path
        arrived = (reason == "pose_arrived")
        stat[f"segs|{prof}|{'arrived' if arrived else (reason or 'cut')}"] += 1
        if not arrived or path <= 0:
            continue
        # steady-state ratio of this run/profile measured on capped steps after the first 5 actions
        E0 = path + TOL[prof] / 2
        for label, prof_new, cap_new, rfac in (
                ("transit50_lin", "transit", 0.050, 1.0), ("transit50_cons", "transit", 0.050, 0.8),
                ("transit35_lin", "transit", 0.035, 1.0),
                ("contact8_lin", "contact", 0.008, 1.0), ("contact10_lin", "contact", 0.010, 1.0)):
            if prof != prof_new: continue
            r = 0.24 if prof == "transit" else 0.234
            old = n_model(E0, CAP[prof], r, TOL[prof])
            new = n_model(E0, cap_new, r * rfac, TOL[prof])
            deltas[label] += max(0.0, old - new)
    out[t] = {"segments": len(segs), "stat": dict(stat), "saved_steps": {k: round(v, 1) for k, v in deltas.items()}}
json.dump(out, open(sys.argv[1], "w"), indent=1)
succ = "t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t7_001 t8_001 t9_001".split()
tot = collections.Counter(); sv = collections.Counter()
for t in succ:
    for k, v in out[t]["stat"].items(): tot[k] += v
    for k, v in out[t]["saved_steps"].items(): sv[k] += v
for k in sorted(tot): print(f"{k:40s} {tot[k]:10.1f}")
for k in sorted(sv): print(f"saved {k:30s} total {sv[k]:8.1f}  per-run {sv[k]/11:6.1f}")
for t in RUNS:
    print(t, out[t]["saved_steps"])
