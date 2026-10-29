"""S300 arithmetic for plan C (read-only; uses verifier rows + runner.log latencies)."""
import json, os
ROOT = "/workspace/gpt_field/runs"
PFX = "agent_speedup_v2_user_check_libero_10_"
rows = json.load(open("/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v3_uc7_rows.json"))
VIDEO_COMPACT = 0.0252   # s/step, trial_003 measured (uc7_runs.md table / V3c)
def lat(key):
    out = []
    with open(os.path.join(ROOT, PFX + key + "_launch", "runner.log"), errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try: ev = json.loads(line)
                except Exception: continue
                if ev.get("event") == "model_call": out.append(ev.get("latency_s"))
    return out
VETO_FIRST = {"t4_001", "t8_001", "t9_001"}   # first call vetoed by the 1.2 rad envelope (V4b); replacement = call #2
res = []
for r in rows:
    if not r["raw"]:
        continue
    k = r["key"]
    approval = 0.0
    s = json.load(open(r["path"]))
    approval = (s.get("user_check") or {}).get("approval_to_claim_s") or 0.0
    base = r["prep"] + approval + r["elapsed"]
    compact = base - r["video"] - r["run_state"] + VIDEO_COMPACT * r["steps"]
    L = lat(k)
    veto = L[1] if k in VETO_FIRST else 0.0
    after_veto = compact - veto
    # unmeasured decode/IPC estimate: 0..24 ms/step (stated as a range)
    lo = after_veto - 0.024 * r["steps"]
    res.append((k, round(base, 1), round(compact, 1), round(veto, 1), round(after_veto, 1), round(lo, 1),
                r["api"], r["steps"], round(r["model"], 1)))
print("key base compactIO vetoReplacement afterVeto withDecodeEst calls steps model")
for x in res: print(*x)
n = len(res)
for i, name in ((1, "base"), (2, "compactIO"), (4, "afterVeto"), (5, "withDecodeEst")):
    vals = [x[i] for x in res]
    print(name, "mean", round(sum(vals)/n, 1), "<=300:", sum(v <= 300 for v in vals), "of", n,
          [x[0] for x in res if x[i] <= 300])
# model budget left under 300 s after all non-model costs (afterVeto case)
left = [300 - (x[4] - x[8]) for x in res]
print("model budget left mean", round(sum(left)/n, 1), "range", round(min(left), 1), round(max(left), 1))
print("calls affordable at 15.1 s/call", round(sum(left)/n/15.1, 1), "; at 21.5 s/call", round(sum(left)/n/21.5, 1))
json.dump(res, open("/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/s300_budget.json", "w"), indent=1)
