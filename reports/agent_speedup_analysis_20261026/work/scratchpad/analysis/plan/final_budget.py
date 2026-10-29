"""Judge synthesis: S300 budget for the phased uc8 -> uc9 -> uc10 plan (ESTIMATES; read-only; python3 -I).

Reuses Plan A's measured per-run inputs (A_runs.json, A_calls.json, A_motion_model.json) and changes only the
tier definitions to the synthesized phase order:
  P0 uc7 measured (+5 s unreconstructed process start-up)
  P1 uc8: compact video (0.0252 s/step) + gated run_state (0.0001 s/step) [C1 trial_003 rates];
          stage-entry envelope vetoes removed (rotation-veto replacement call + 1 hold step dropped);
          frame facts add ~0.03 s per call (input text); bounded/compact prep assumed unchanged when it completes;
          NO pre-started worker (dropped).
  P2 uc9: simulator-step slimming (-15/-30/-50 ms/step); transit cap (t35 / t50 / t50r step savings);
          recorder thread hides a fraction (0.3/0.6/0.9) of the remaining compact video cost.
  P3a uc10 call economy: terse prose (-0.6/-1.1/-2.0 s per call) + de-duplicated observation text
          (-25/-40/-55% of non-image, non-instruction input tokens at 0.0274 s per 1k).
  P3b uc10 + checkpoint economy: 0/1/2 fewer non-first calls (mean non-first latency).
  U1 (user decision, conditional): execution effort high->medium = -50% reasoning tokens at 0.0392 s/token.
Model latency is scaled by a service-speed factor f (1.0 = historical uc7 fit; today's calls ran x1.35-1.51).
"""
import json, statistics as st

P = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/"
R = json.load(open(P + "A_runs.json"))
C = json.load(open(P + "A_calls.json"))
MM = json.load(open(P + "A_motion_model.json"))
SWEEP = "t0_001 t1_001 t2_002 t3_005 t4_001 t5_002 t6_011 t7_001 t8_001 t9_001".split()
PROCESS_START = 5.0
VIDEO_COMPACT = 25.642 / 1019
RUNSTATE_GATED = 0.108 / 1019
VETO_REPLACEMENT = {"t4_001": [2], "t8_001": [2], "t9_001": [2], "t6_011": [2, 10]}
IMG_TOK, INSTR_TOK, IN_COEF, REAS = 308.5, 2304, 0.0274, 0.0392
FACTS_S_PER_CALL = 0.03

sav = {}
for r in MM["rows"]:
    if r["prof"] != "transit" or not r["pred"]:
        continue
    d = sav.setdefault(r["run"], {"t35": 0.0, "t50": 0.0, "t50r": 0.0})
    for col in d:
        d[col] += r["actual"] - r["actual"] * r[col] / r["pred"]

LV = {"sim_cut": (0.015, 0.030, 0.050), "transit": ("t35", "t50", "t50r"), "rec_hide": (0.3, 0.6, 0.9),
      "prose": (0.6, 1.1, 2.0), "text_cut": (0.25, 0.40, 0.55), "fewer": (0, 1, 2)}
TIERS = ["P0 uc7 measured", "P1 uc8 harness+veto+facts", "P2 uc9 +sim slim+transit+rec thread",
         "P3a uc10 +terse/dedup", "P3b uc10 +checkpoint economy", "U1 +effort medium (user)"]


def calls_of(run):
    return sorted((c for c in C if c["run"] == run), key=lambda c: c["k"])


def service(run, tier, lvl=1, f=1.0):
    r = R[run]
    steps = float(r["steps"])
    per = {k: r[k] / r["steps"] for k in ("sim", "video", "run_state", "observe", "policy", "residual")}
    cs = calls_of(run)
    lat = {c["k"]: c["latency_s"] for c in cs}
    prep, fixed = r["prep"], PROCESS_START + r["approval"] + r["startup_final"]
    if tier >= 1:
        per["video"], per["run_state"] = VIDEO_COMPACT, RUNSTATE_GATED
        for k in VETO_REPLACEMENT.get(run, []):
            lat.pop(k, None)
            steps -= 1
    if tier >= 2:
        per["sim"] = max(0.03, per["sim"] - LV["sim_cut"][lvl])
        steps -= sav.get(run, {}).get(LV["transit"][lvl], 0.0)
        per["video"] *= (1 - LV["rec_hide"][lvl])
    model = 0.0
    for c in cs:
        if c["k"] not in lat:
            continue
        L = lat[c["k"]] + (FACTS_S_PER_CALL if tier >= 1 else 0.0)
        if tier >= 3:
            L -= LV["prose"][lvl]
            text_tok = max(0.0, (c["input_tokens"] or 0) - IMG_TOK * 12 - INSTR_TOK)
            L -= IN_COEF * LV["text_cut"][lvl] * text_tok / 1000.0
        if tier >= 5:
            L -= REAS * 0.5 * (c["reasoning_tokens"] or 0)
        model += max(4.0, L)
    if tier >= 4:
        nonfirst = [lat[k] for k in lat if k != 1]
        model -= LV["fewer"][lvl] * (st.mean(nonfirst) if nonfirst else 0.0)
    model *= f
    comp = {"prep": prep, "fixed": fixed, "model": model, "sim": steps * per["sim"], "video": steps * per["video"],
            "run_state": steps * per["run_state"],
            "other_step": steps * (per["observe"] + per["policy"] + per["residual"]), "steps": steps}
    comp["total"] = prep + fixed + model + comp["sim"] + comp["video"] + comp["run_state"] + comp["other_step"]
    return comp


out = {}
for f in (1.0, 1.25, 1.5):
    print("\n=== service-speed factor f = %.2f ===" % f)
    for lvl, name in ((1, "central"), (0, "low-effect"), (2, "high-effect")):
        print("-- assumptions:", name)
        for t, tname in enumerate(TIERS):
            vals = {run: service(run, t, lvl, f)["total"] for run in SWEEP}
            ranked = sorted((v if R[run]["raw"] else float("inf")) for run, v in vals.items())
            n300 = sum(1 for run, v in vals.items() if R[run]["raw"] and v <= 300)
            succ = [service(k, t, lvl, f)["total"] for k in R if R[k]["raw"]]
            print("%-40s 5th-best %6.1f  median10 %6.1f  S300 %d/10 | raw-11 mean %6.1f, <=300 %d/11" % (
                tname, ranked[4], (ranked[4] + ranked[5]) / 2, n300, st.mean(succ), sum(v <= 300 for v in succ)))
            out[f"f{f}_{name}_{t}"] = {"fifth": ranked[4], "median10": (ranked[4] + ranked[5]) / 2, "s300": n300,
                                       "raw11_mean": st.mean(succ), "raw11_le300": sum(v <= 300 for v in succ),
                                       "per_task": vals}

print("\n=== t1_001 (5th-best task of the uc7 sweep) component table, central ===")
rows = {}
for f in (1.0, 1.5):
    for t, tname in enumerate(TIERS):
        d = service("t1_001", t, 1, f)
        rows[f"f{f}_{t}"] = d
        print("f=%.2f %-40s total %6.1f | prep %5.1f fixed %5.1f model %6.1f sim %5.1f video %5.1f run_state %5.1f "
              "other %4.1f steps %5.0f" % (f, tname, d["total"], d["prep"], d["fixed"], d["model"], d["sim"],
                                           d["video"], d["run_state"], d["other_step"], d["steps"]))
print("\n=== raw-11 mean component table, central, f=1.0 ===")
for t, tname in enumerate(TIERS):
    ds = [service(k, t, 1, 1.0) for k in R if R[k]["raw"]]
    m = {key: st.mean(d[key] for d in ds) for key in ds[0]}
    print("%-40s total %6.1f | prep %5.1f fixed %5.1f model %6.1f sim %5.1f video %5.1f run_state %5.1f other %4.1f "
          "steps %5.0f" % (tname, m["total"], m["prep"], m["fixed"], m["model"], m["sim"], m["video"],
                           m["run_state"], m["other_step"], m["steps"]))
json.dump({"tiers": out, "t1_rows": rows}, open(P + "final_budget_out.json", "w"), indent=0)
