#!/usr/bin/env python3
"""Counterfactuals for V3b/V3c from raw artifacts.

Reads uc7 run summaries, runner.log model_call latencies, supervisord logs,
and the trial_003 (B arm, compact I/O) summary from today's campaign.
"""
import datetime as dt
import json
import os
import re

ROOT = "/workspace/gpt_field/runs"
PFX = "agent_speedup_v2_user_check_libero_10_"
UC7 = ["t0_001", "t1_001", "t2_002", "t3_003", "t3_005", "t4_001", "t5_001",
       "t5_002", "t6_009", "t6_011", "t7_001", "t8_001", "t9_001"]
T3 = ("/workspace/gpt_field/results/speedup_evidence_20261026T055123Z/"
      "evaluations/trial_003/trial/summary.json")
T2 = ("/workspace/gpt_field/results/speedup_evidence_20261026T055123Z/"
      "evaluations/trial_002/trial/summary.json")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def runner_latencies(key):
    p = os.path.join(ROOT, PFX + key + "_launch", "runner.log")
    lats = []
    with open(p, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if ev.get("event") == "model_call":
                lats.append(ev.get("latency_s"))
    return lats


def sup_wall(key):
    p = os.path.join(ROOT, PFX + key + "_launch", "supervisord.log")
    if not os.path.exists(p):
        return None
    sp = ex = None
    with open(p) as f:
        for line in f:
            m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) ", line)
            if not m:
                continue
            t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") + \
                dt.timedelta(milliseconds=int(m.group(2)))
            if "spawned:" in line:
                sp = t
            if "exited:" in line:
                ex = t
    if sp and ex:
        return (ex - sp).total_seconds()
    return None


t3 = json.load(open(T3))
t3s = t3["steps"]
t3tc = t3["timing_components"]
vid_ps = t3tc["video_render_s"] / t3s
rs_ps = t3tc["run_state_s"] / t3s
print("trial_003 arm-B: steps %d video %.3f s (%.5f s/step) run_state %.4f s (%.6f s/step)"
      % (t3s, t3tc["video_render_s"], vid_ps, t3tc["run_state_s"], rs_ps))
t2 = json.load(open(T2))
print("trial_002 arm-A: steps %d video %.3f s (%.5f s/step) run_state %.4f s (%.6f s/step)"
      % (t2["steps"], t2["timing_components"]["video_render_s"],
         t2["timing_components"]["video_render_s"] / t2["steps"],
         t2["timing_components"]["run_state_s"],
         t2["timing_components"]["run_state_s"] / t2["steps"]))
print()

rows = []
for k in UC7:
    p = os.path.join(ROOT, PFX + k, "summary.json")
    d = json.load(open(p))
    uc = d["user_check"]
    tc = d["timing_components"]
    lats = runner_latencies(k)
    r = dict(key=k, raw=d["libero_success"], v2=d["v2_success"],
             prep=uc["preparation_elapsed_s"], a2c=uc.get("approval_to_claim_s") or 0.0,
             review_wait=uc.get("review_phase_elapsed_s"),
             elapsed=d["elapsed_s"], steps=d["steps"], api=d["api_calls"],
             model=tc["model_pipeline_s"], video=tc["video_render_s"],
             run_state=tc["run_state_s"], prep_model=uc.get("preparation_model_s"),
             runner_lat_sum=sum(lats), runner_n=len(lats), sup=sup_wall(k))
    r["svc_a"] = r["prep"] + r["elapsed"]                 # claim's stated definition
    r["svc_b"] = r["prep"] + r["a2c"] + r["elapsed"]      # analysts' definition
    # compact I/O counterfactual: video -> vid_ps*steps, run_state -> rs_ps*steps
    delta = (vid_ps * r["steps"] - r["video"]) + (rs_ps * r["steps"] - r["run_state"])
    r["cf_a"] = r["svc_a"] + delta
    r["cf_b"] = r["svc_b"] + delta
    # run_state exactly 0
    delta0 = (vid_ps * r["steps"] - r["video"]) - r["run_state"]
    r["cf0_a"] = r["svc_a"] + delta0
    r["cf0_b"] = r["svc_b"] + delta0
    rows.append(r)

print("%-7s %4s %4s %8s %8s %8s %8s %6s %8s %8s %8s %8s %8s" % (
    "key", "raw", "v2", "svc_a", "svc_b", "cf_a", "cf_b", "api", "model", "lat_sum",
    "sup", "sup-rw", "sup-svcb"))
for r in rows:
    sup = r["sup"]
    print("%-7s %4s %4s %8.2f %8.2f %8.2f %8.2f %6d %8.2f %8.2f %8s %8s %8s" % (
        r["key"], r["raw"], r["v2"], r["svc_a"], r["svc_b"], r["cf_a"], r["cf_b"],
        r["api"], r["model"], r["runner_lat_sum"],
        "%.3f" % sup if sup else "-",
        "%.3f" % (sup - r["review_wait"]) if sup else "-",
        "%.3f" % (sup - r["review_wait"] - r["svc_b"]) if sup else "-"))
print("runner.log n calls == api_calls:", all(r["runner_n"] == r["api"] for r in rows))
print("max |runner lat sum - model_pipeline_s|: %.3f" % max(abs(r["runner_lat_sum"] - r["model"]) for r in rows))
print()
S = [r for r in rows if r["raw"]]
print("n raw successes", len(S))
for lab in ("svc_a", "svc_b", "cf_a", "cf_b", "cf0_a", "cf0_b"):
    xs = [r[lab] for r in S]
    print("%-6s mean %.2f  min %.2f  max %.2f  n<=300: %d %s" % (
        lab, mean(xs), min(xs), max(xs), sum(1 for x in xs if x <= 300),
        [(r["key"], round(r[lab], 2)) for r in S if r[lab] <= 300]))
mm = mean(r["model"] for r in S)
ma = mean(r["api"] for r in S)
for lab in ("cf_a", "cf_b"):
    nonmodel = mean(r[lab] - r["model"] for r in S)
    left = 300 - nonmodel
    print("%s: mean non-model %.2f -> model budget left %.2f s = %.2f calls at %.3f s/call"
          " (= %.2f calls at trial_003's %.2f s/call)" % (
              lab, nonmodel, left, left / (mm / ma), mm / ma,
              left / (t3tc["model_pipeline_s"] / t3["api_calls"]),
              t3tc["model_pipeline_s"] / t3["api_calls"]))
    # per-run model budget left
    per = [(r["key"], round(300 - (r[lab] - r["model"]), 1)) for r in S]
    print("   per-run budget left:", per)
# model=0 for t0_001 under both definitions, w/ and w/o prep model
r0 = [r for r in rows if r["key"] == "t0_001"][0]
print("t0_001 model=0: svc_a-model %.2f  svc_b-model %.2f  svc_b-model-prep_model %.2f" % (
    r0["svc_a"] - r0["model"], r0["svc_b"] - r0["model"],
    r0["svc_b"] - r0["model"] - r0["prep_model"]))
print("all model=0 (svc_b - model):", [(r["key"], round(r["svc_b"] - r["model"], 2)) for r in S])
print("mean model=0 (svc_b - model): %.2f" % mean(r["svc_b"] - r["model"] for r in S))
print("mean a2c %.4f" % mean(r["a2c"] for r in S))
