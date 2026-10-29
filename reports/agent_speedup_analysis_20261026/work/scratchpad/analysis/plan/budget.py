"""S300 budget arithmetic over the uc7 runs. Every changed quantity is an ESTIMATE; inputs are measured
per-run components (summary.json, runner.log) plus the per-segment motion model (segments_out.json)."""
import json, statistics as st, sys
P = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/"
R = json.load(open(P + "budget_inputs.json"))
SEG = json.load(open(P + "segments_out.json"))
SWEEP = ["t0_001", "t1_001", "t2_002", "t3_005", "t4_001", "t5_002", "t6_011", "t7_001", "t8_001", "t9_001"]
SUCC = ["t0_001", "t1_001", "t2_002", "t3_003", "t3_005", "t4_001", "t5_001", "t5_002", "t7_001", "t8_001", "t9_001"]
# replacement calls made avoidable by removing the stage-entry envelope veto (index of the replacement call, 0-based)
VETO_REPLACEMENT = {"t4_001": [1], "t8_001": [1], "t9_001": [1], "t6_011": [1, 9]}
PROCESS_START_S = 5.0          # reconstruction under-counts process start-up by ~4.6-5.5 s (supervisord, 2 runs)
VIDEO_COMPACT = 0.0252         # trial_003 measured video_render_s / steps
RUNSTATE_GATED = 0.0001        # trial_003 measured run_state_s / steps
VIS_COEF = 0.0212              # s per visible token (uc7 multivariate fit, model_io.md 1.2)
REAS_COEF = 0.0392

def service(t, *, io=False, sim_cut=0.0, transit=None, contact=None, veto=False, prose_cut=0.0,
            calls_cut=0.0, speed=1.0, reasoning_cut=0.0, first_hidden=False, prep_cut=0.0, startup_cut=0.0):
    r = R[t]
    steps = r["steps"]
    lat = list(r["lat"])
    if veto:
        for i in sorted(VETO_REPLACEMENT.get(t, []), reverse=True):
            del lat[i]
        steps -= len(VETO_REPLACEMENT.get(t, []))     # the hold step
    saved = 0.0
    if transit: saved += SEG[t]["saved_steps"].get(transit, 0.0)
    if contact: saved += SEG[t]["saved_steps"].get(contact, 0.0)
    new_steps = steps - saved
    per = dict(sim=r["sim_s"] / r["steps"], video=r["video_s"] / r["steps"], rs=r["run_state_s"] / r["steps"],
               obs=r["observe_s"] / r["steps"], pol=r["policy_s"] / r["steps"], res=r["residual_s"] / r["steps"])
    if io:
        per["video"], per["rs"] = VIDEO_COMPACT, RUNSTATE_GATED
    per["sim"] = max(0.02, per["sim"] - sim_cut)
    harness = new_steps * sum(per.values())
    vis = [v or 0 for v in r["visible"]]; reas = [x or 0 for x in r["reasoning"]]
    if veto:
        for i in sorted(VETO_REPLACEMENT.get(t, []), reverse=True):
            del vis[i]; del reas[i]
    model = 0.0
    for k, L in enumerate(lat):
        L2 = L - VIS_COEF * prose_cut * vis[k] - REAS_COEF * reasoning_cut * reas[k]
        if first_hidden and k == 0:
            L2 = 0.0
        model += speed * max(3.0, L2)
    # fewer calls: remove calls_cut mean non-first calls
    if calls_cut:
        nonfirst = lat[1:] or [0]
        model -= speed * calls_cut * st.mean(nonfirst)
    prep = r["prep_s"] - prep_cut
    fixed = PROCESS_START_S + (r["approval_to_claim_s"] or 0) + r["startup_final_s"] - startup_cut
    return dict(total=prep + fixed + harness + model, prep=prep, fixed=fixed, harness=harness, model=model,
                steps=new_steps, per_step=sum(per.values()))

SC = {
 "B0 baseline (+5 s start-up)": {},
 "S1 +compact video/state": dict(io=True),
 "S2 +sim slimming (central -40 ms)": dict(io=True, sim_cut=0.040),
 "S2lo +sim slimming (-20 ms)": dict(io=True, sim_cut=0.020),
 "S3 +transit 50mm/veto removal (lin)": dict(io=True, sim_cut=0.040, transit="transit50_lin", veto=True),
 "S3c +transit (conservative 0.8x ratio)": dict(io=True, sim_cut=0.040, transit="transit50_cons", veto=True),
 "S4 +terse prose (-50% prose)": dict(io=True, sim_cut=0.040, transit="transit50_lin", veto=True, prose_cut=0.5*0.52),
 "S4b +1 fewer call": dict(io=True, sim_cut=0.040, transit="transit50_lin", veto=True, prose_cut=0.5*0.52, calls_cut=1.0),
 "S5 +contact 8mm": dict(io=True, sim_cut=0.040, transit="transit50_lin", veto=True, prose_cut=0.5*0.52, contact="contact8_lin"),
 "S5b +contact 8mm +1 fewer call": dict(io=True, sim_cut=0.040, transit="transit50_lin", veto=True, prose_cut=0.5*0.52, contact="contact8_lin", calls_cut=1.0),
 "U1 S4 + effort medium (-50% reasoning)": dict(io=True, sim_cut=0.040, transit="transit50_lin", veto=True, prose_cut=0.5*0.52, reasoning_cut=0.5),
 "U2 S4 + first call during review wait": dict(io=True, sim_cut=0.040, transit="transit50_lin", veto=True, prose_cut=0.5*0.52, first_hidden=True),
}
def table(runs, label, speed=1.0, keyfmt=None):
    print(f"\n== {label} (model speed x{speed}) ==")
    print("scenario".ljust(44), " ".join(t[:5].rjust(6) for t in runs), " median  n<=300")
    res = {}
    for name, kw in SC.items():
        vals = []
        for t in runs:
            v = service(t, speed=speed, **kw)["total"]
            vals.append(v)
        ok = [v for t, v in zip(runs, vals) if R[t]["raw"]]
        med = st.median(vals)
        n300 = sum(1 for t, v in zip(runs, vals) if R[t]["raw"] and v <= 300)
        print(name.ljust(44), " ".join(f"{v:6.0f}" for v in vals), f" {med:6.0f}  {n300}")
        res[name] = dict(zip(runs, vals))
    return res
out = {}
for sp in (1.0, 1.25, 1.5):
    out[f"sweep_x{sp}"] = table(SWEEP, "10-task sweep runs (t6_011 = failure)", sp)
out["succ_x1.0"] = table(SUCC, "11 raw successes", 1.0)
# decomposition for the median task t1_001 and for t4_001
for t in ("t1_001", "t4_001", "t7_001", "t8_001", "t2_002"):
    print("\n--", t)
    for name, kw in SC.items():
        d = service(t, **kw)
        print(f"  {name:44s} total {d['total']:6.1f} prep {d['prep']:5.1f} fixed {d['fixed']:5.1f} "
              f"steps {d['steps']:6.0f} x {d['per_step']*1000:5.1f} ms = {d['harness']:6.1f}  model {d['model']:6.1f}")
json.dump(out, open(P + "budget_out.json", "w"), indent=1)

# ---- cumulative plan steps with pre-warm, pessimistic/optimistic composites ----
def su(t): return R[t]["startup_s"]
PLAN = [
 ("baseline", lambda t: {}),
 ("+H prep/IO (compact video, state gate, pre-warmed sim)", lambda t: dict(io=True, startup_cut=su(t))),
 ("+H sim slimming (-40 ms/step)", lambda t: dict(io=True, startup_cut=su(t), sim_cut=0.040)),
 ("+M transit 50 mm + veto removal", lambda t: dict(io=True, startup_cut=su(t), sim_cut=0.040, transit="transit50_lin", veto=True)),
 ("+C terse prose", lambda t: dict(io=True, startup_cut=su(t), sim_cut=0.040, transit="transit50_lin", veto=True, prose_cut=0.26)),
 ("+M contact 8 mm (gated)", lambda t: dict(io=True, startup_cut=su(t), sim_cut=0.040, transit="transit50_lin", veto=True, prose_cut=0.26, contact="contact8_lin")),
 ("+C chain-through (-1 call)", lambda t: dict(io=True, startup_cut=su(t), sim_cut=0.040, transit="transit50_lin", veto=True, prose_cut=0.26, contact="contact8_lin", calls_cut=1.0)),
]
PESS = lambda t: dict(io=True, startup_cut=su(t), sim_cut=0.020, transit="transit50_cons", veto=True, prose_cut=0.13)
OPT = lambda t: dict(io=True, startup_cut=su(t), sim_cut=0.060, transit="transit50_lin", veto=True, prose_cut=0.26, contact="contact10_lin", calls_cut=1.5)
for sp in (1.0, 1.25, 1.5):
    print(f"\n### cumulative plan, 10-task sweep, model speed x{sp}")
    prev = None
    for name, f in PLAN + [("PESSIMISTIC composite", PESS), ("OPTIMISTIC composite", OPT)]:
        vals = {t: service(t, speed=sp, **f(t))["total"] for t in SWEEP}
        succ_vals = sorted(v for t, v in vals.items() if R[t]["raw"])
        med10 = st.median(list(vals.values()))
        fifth = succ_vals[4]
        n300 = sum(1 for t, v in vals.items() if R[t]["raw"] and v <= 300)
        mean_delta = "" if prev is None or name.endswith("composite") else f"  mean step delta {st.mean(vals[t]-prev[t] for t in SWEEP):6.1f} s"
        print(f"{name:58s} median10 {med10:6.1f}  5th-best-success {fifth:6.1f}  n<=300 {n300}/10  t1_001 {vals['t1_001']:6.1f}{mean_delta}")
        if not name.endswith("composite"): prev = vals
