#!/usr/bin/env python3
"""Independent recomputation of uc7 service-time claims (V3a/V3b/V3c).

Reads only raw run artifacts under /workspace/gpt_field/runs.
"""
import json
import os
import statistics as st
import sys

ROOT = "/workspace/gpt_field/runs"
PFX = "agent_speedup_v2_user_check_libero_10_"
UC7 = ["t0_001", "t1_001", "t2_002", "t3_003", "t3_005", "t4_001", "t5_001",
       "t5_002", "t6_009", "t6_011", "t7_001", "t8_001", "t9_001"]


def load(key):
    p = os.path.join(ROOT, PFX + key, "summary.json")
    with open(p) as f:
        return json.load(f), p


rows = []
for k in UC7:
    d, p = load(k)
    uc = d.get("user_check") or {}
    tc = d.get("timing_components") or {}
    r = {
        "key": k,
        "path": p,
        "version": uc.get("version"),
        "raw": d.get("libero_success"),
        "v2": d.get("v2_success"),
        "prep": uc.get("preparation_elapsed_s"),
        "prep_model": uc.get("preparation_model_s"),
        "elapsed": d.get("elapsed_s"),
        "control": d.get("control_elapsed_s"),
        "ctrl_novideo": d.get("control_without_video_s"),
        "startup": d.get("startup_to_initial_observation_s"),
        "steps": d.get("steps"),
        "api": d.get("api_calls"),
        "model": tc.get("model_pipeline_s"),
        "decision": tc.get("decision_s"),
        "policy": tc.get("policy_compute_s"),
        "observe": tc.get("observe_transition_s"),
        "run_state": tc.get("run_state_s"),
        "sim": tc.get("simulator_step_s"),
        "video": tc.get("video_render_s"),
        "budget_elapsed": (d.get("budget") or {}).get("elapsed_s"),
        "budget_calls": (d.get("budget") or {}).get("api_calls"),
    }
    r["service"] = r["prep"] + r["elapsed"]
    rows.append(r)

succ = [r for r in rows if r["raw"] is True]
print("n uc7 =", len(rows), " n raw successes =", len(succ))
print("versions:", sorted(set(r["version"] for r in rows)))
print()
hdr = ("key", "prep", "elapsed", "service", "control", "steps", "api", "model",
       "sim", "video", "run_state", "observe", "policy", "startup")
print("%-7s " % hdr[0] + " ".join("%9s" % h for h in hdr[1:]))
for r in rows:
    print("%-7s " % r["key"] + " ".join(
        "%9.2f" % r[h] if isinstance(r[h], float) else "%9s" % r[h]
        for h in hdr[1:]), "" if r["raw"] else "(raw fail)")
print()


def mean(xs):
    return sum(xs) / len(xs)


S = succ
svc = [r["service"] for r in S]
print("service mean %.2f  min %.2f (%s)  max %.2f (%s)  median %.2f" % (
    mean(svc), min(svc), min(S, key=lambda r: r["service"])["key"],
    max(svc), max(S, key=lambda r: r["service"])["key"], st.median(svc)))
print("prep mean %.3f  elapsed mean %.3f  control mean %.3f" % (
    mean([r["prep"] for r in S]), mean([r["elapsed"] for r in S]),
    mean([r["control"] for r in S])))
mm = mean([r["model"] for r in S])
ma = mean([r["api"] for r in S])
print("model_pipeline mean %.3f   api mean %.3f   mean-of-ratios s/call %.3f   "
      "pooled s/call %.3f   mm/ma %.3f" % (
          mm, ma, mean([r["model"] / r["api"] for r in S]),
          sum(r["model"] for r in S) / sum(r["api"] for r in S), mm / ma))
steps_mean = mean([r["steps"] for r in S])
tot_steps = sum(r["steps"] for r in S)
print("steps mean %.2f  total %d" % (steps_mean, tot_steps))
for comp in ("sim", "video", "run_state", "observe", "policy"):
    pooled = sum(r[comp] for r in S) / tot_steps
    mor = mean([r[comp] / r["steps"] for r in S])
    print("  %-9s pooled/step %.4f  mean-of-ratio/step %.4f  mean total %.3f" % (
        comp, pooled, mor, mean([r[comp] for r in S])))
nonmodel4 = sum(r["sim"] + r["video"] + r["run_state"] + r["observe"] for r in S)
print("4-comp non-model pooled/step %.4f ; x mean steps = %.2f ; mean total = %.2f" % (
    nonmodel4 / tot_steps, nonmodel4 / tot_steps * steps_mean, nonmodel4 / len(S)))
# residual: elapsed - (model + sim + video + run_state + observe + policy)
for r in S:
    r["residual"] = r["elapsed"] - (r["model"] + r["sim"] + r["video"] +
                                    r["run_state"] + r["observe"] + r["policy"])
    r["ctrl_resid"] = r["control"] - (r["decision"] + r["sim"] + r["video"] +
                                      r["run_state"] + r["observe"])
print("residual elapsed - (model+sim+video+run_state+observe+policy): mean %.2f"
      % mean([r["residual"] for r in S]),
      [(r["key"], round(r["residual"], 2)) for r in S])
print("control - (decision+sim+video+run_state+observe): ",
      [(r["key"], round(r["ctrl_resid"], 2)) for r in S])
print("elapsed - control: ", [(r["key"], round(r["elapsed"] - r["control"], 2)) for r in S])
print()
# V3b: service <= 300
print("service <= 300:", [(r["key"], round(r["service"], 2)) for r in rows if r["service"] <= 300])
print("service <= 300 among raw succ:", [(r["key"], round(r["service"], 2)) for r in S if r["service"] <= 300])
# model time set to 0
print("service - model (model=0):")
for r in S:
    print("   %s %.2f (service %.2f, model %.2f, prep_model %.2f, service-model-prep_model %.2f)" % (
        r["key"], r["service"] - r["model"], r["service"], r["model"], r["prep_model"],
        r["service"] - r["model"] - r["prep_model"]))
print()
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "v3_uc7_rows.json"), "w"), indent=1)
