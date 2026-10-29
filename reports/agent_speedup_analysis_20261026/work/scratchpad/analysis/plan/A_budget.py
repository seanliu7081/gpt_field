"""Plan A: cumulative S300 budget for the uc7 10-task sweep (ESTIMATES; read-only; python3 -I).

Inputs (all measured; see A_runs.py / A_calls.py / A_motion_model.py):
  per run: prep, approval, startup_to_initial_observation, elapsed-control (start-up + finalization),
           per-step sim/video/run_state/observe/policy/residual, steps, per-call latency/tokens.
  trial_003 compact I/O rates: video 25.642 s / 1019 steps, run_state 0.108 s / 1019 steps
           (results/speedup_evidence_20261026T055123Z/evaluations/trial_003/trial/summary.json).
  motion: calibrated per-run step savings of arrived transit segments (A_motion_saving / A_motion_model rows).
Every changed quantity is an estimate with (low, central, high) values; the arithmetic is additive per run.
"""
import json, statistics as st, sys

P = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/"
R = json.load(open(P + "A_runs.json"))
C = json.load(open(P + "A_calls.json"))
MM = json.load(open(P + "A_motion_model.json"))
SWEEP = "t0_001 t1_001 t2_002 t3_005 t4_001 t5_002 t6_011 t7_001 t8_001 t9_001".split()
EXTRA = ["t3_003", "t5_001", "t6_009"]

PROCESS_START = 5.0                       # supervisord wall - reconstruction: 4.56 s (t5_001), 5.52 s (t6_009)
VIDEO_COMPACT = 25.642 / 1019             # trial_003
RUNSTATE_GATED = 0.108 / 1019             # trial_003
VETO_REPLACEMENT = {"t4_001": [2], "t8_001": [2], "t9_001": [2], "t6_011": [2, 10]}   # call k after a 1.2 rad veto
IMG_TOK, INSTR_TOK = 308.5, 2304          # response usage attribution (model_io.md 1.3/3.4)
IN_COEF = 0.0274                          # s per 1k input tokens (uc7 multivariate fit, model_io.md 1.2)
VIS_COEF = 0.0212                         # s per visible output token (same fit)

# per-run calibrated transit savings: steps saved by cap 50 mm + rotation clamp 0.25 rad (t50r) or 50 mm only (t50)
sav = {}
for r in MM["rows"]:
    if r["prof"] != "transit" or not r["pred"]:
        continue
    d = sav.setdefault(r["run"], {"t50": 0.0, "t50r": 0.0, "t35": 0.0})
    for col in ("t50", "t50r", "t35"):
        d[col] += r["actual"] - r["actual"] * r[col] / r["pred"]

LEVELS = {  # (low-effect, central, high-effect) assumptions
    "startup_hidden_frac": (0.5, 0.7, 0.9),     # share of startup_to_initial_observation hidden by pre-starting the worker
    "sim_cut_s": (0.015, 0.030, 0.050),         # s/step removed from simulator_step_s by slimming (after M1)
    "residual_cut_s": (0.0, 0.0, 0.0015),       # opt-in trace rows without per-step PNGs (not counted centrally)
    "transit_col": ("t35", "t50", "t50r"),      # cap choice / realization (low = model too optimistic -> use 35 mm result)
    "prose_s_per_call": (0.6, 1.1, 2.0),        # terse belief/plan/why: s saved per call
    "obs_text_cut": (0.25, 0.40, 0.55),         # fraction of non-image, non-instruction input tokens removed
    "calls_cut": (0.0, 1.0, 2.0),               # fewer non-first calls per task (prompt economy), speculative
}


def calls_of(run):
    return sorted((c for c in C if c["run"] == run), key=lambda c: c["k"])


def service(run, tier, lvl=1, speed=1.0):
    r = R[run]
    steps = r["steps"]
    per = {k: r[k] / steps for k in ("sim", "video", "run_state", "observe", "policy", "residual")}
    cs = calls_of(run)
    lat = {c["k"]: c["latency_s"] for c in cs}
    prep = r["prep"]
    fixed = PROCESS_START + r["approval"] + r["startup_final"]
    if tier >= 1:   # H1: compact video + gated run_state + trace slimming; H2 bounded prep; H3 pre-started worker
        per["video"], per["run_state"] = VIDEO_COMPACT, RUNSTATE_GATED
        per["residual"] = max(0.0, per["residual"] - LEVELS["residual_cut_s"][lvl])
        fixed -= LEVELS["startup_hidden_frac"][lvl] * r["startup"]
    if tier >= 2:   # H4: simulator-step slimming (size known only after measurement M1)
        per["sim"] = max(0.03, per["sim"] - LEVELS["sim_cut_s"][lvl])
    if tier >= 3:   # M: transit cap/rotation clamp + removal of the 1.2 rad stage-entry rotation veto
        steps = steps - sav.get(run, {}).get(LEVELS["transit_col"][lvl], 0.0)
        for k in VETO_REPLACEMENT.get(run, []):
            lat.pop(k, None)
            steps -= 1   # the one-step hold after the veto
    model = 0.0
    for c in cs:
        if c["k"] not in lat:
            continue
        L = lat[c["k"]]
        if tier >= 4:   # C1: terse prose + de-duplicated observation text
            L -= LEVELS["prose_s_per_call"][lvl]
            text_tok = max(0.0, (c["input_tokens"] or 0) - IMG_TOK * 12 - INSTR_TOK)
            L -= IN_COEF * LEVELS["obs_text_cut"][lvl] * text_tok / 1000.0
        model += speed * max(4.0, L)
    if tier >= 5:   # C2: fewer calls (speculative prompt economy)
        nonfirst = [lat[k] for k in lat if k != 1]
        model -= speed * LEVELS["calls_cut"][lvl] * (st.mean(nonfirst) if nonfirst else 0.0)
    harness = steps * sum(per.values())
    return dict(total=prep + fixed + harness + model, prep=prep, fixed=fixed, harness=harness, model=model,
                steps=steps, per_step=sum(per.values()))


TIERS = ["T0 baseline (+5 s process start-up)", "T1 H1-H3 compact I/O, bounded prep, pre-started worker",
         "T2 +H4 simulator-step slimming", "T3 +M transit 50 mm/0.25 rad + rotation-veto removal",
         "T4 +C1 terse prose + de-duplicated observation text", "T5 +C2 one fewer call (speculative)"]
out = {}
for speed in (1.0, 1.5):
    print("\n=== model latency x%.1f (x1.5 ~ 2026-10-26 service speed) ===" % speed)
    for lvl, name in ((1, "central"), (0, "low-effect"), (2, "high-effect")):
        print("-- assumptions:", name)
        for t, tname in enumerate(TIERS):
            vals = {run: service(run, t, lvl, speed)["total"] for run in SWEEP}
            ranked = sorted((v if R[run]["raw"] else float("inf")) for run, v in vals.items())
            n300 = sum(1 for run, v in vals.items() if R[run]["raw"] and v <= 300)
            fifth = ranked[4]
            med10 = (ranked[4] + ranked[5]) / 2
            print("%-58s 5th-best %6.1f  median10 %6.1f  S300 %d/10  t1 %6.1f  t4 %6.1f  t7 %6.1f" % (
                tname, fifth, med10, n300, vals["t1_001"], vals["t4_001"], vals["t7_001"]))
            out[f"x{speed}_{name}_T{t}"] = vals
# decomposition of the 5th-best/median tasks under central assumptions at x1.0
print("\n=== decomposition, central, x1.0 ===")
for run in ("t1_001", "t4_001", "t7_001", "t8_001", "t3_005", "t2_002", "t5_002"):
    print(run)
    for t, tname in enumerate(TIERS):
        d = service(run, t, 1, 1.0)
        print("   %-58s total %6.1f = prep %5.1f + fixed %5.1f + steps %5.0f x %5.1f ms (%5.1f) + model %6.1f" % (
            tname, d["total"], d["prep"], d["fixed"], d["steps"], 1000 * d["per_step"], d["harness"], d["model"]))
# mean over 11 raw successes for reference
SUCC = [k for k in R if R[k]["raw"]]
for t, tname in enumerate(TIERS):
    vals = [service(run, t, 1, 1.0)["total"] for run in SUCC]
    print("11 raw successes mean %-50s %6.1f  median %6.1f  n<=300 %d" % (tname, st.mean(vals), st.median(vals), sum(v <= 300 for v in vals)))
json.dump(out, open(P + "A_budget_out.json", "w"), indent=0)
