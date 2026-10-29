"""Plan A: per-run service decomposition of the 13 uc7 runs (read-only; python3 -I).

Source: R/summary.json (timing_components, user_check, steps, elapsed/control/startup) for
R = /workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_<key>; joined with A_calls.json.
Service (reconstructed) = user_check.preparation_elapsed_s + user_check.approval_to_claim_s + elapsed_s.
"""
import json, statistics as st

ROOT = "/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_"
P = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/"
RUNS = "t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()
SWEEP = "t0_001 t1_001 t2_002 t3_005 t4_001 t5_002 t6_011 t7_001 t8_001 t9_001".split()
calls = json.load(open(P + "A_calls.json"))
out = {}
for key in RUNS:
    s = json.load(open(ROOT + key + "/summary.json"))
    tc, uc = s["timing_components"], s["user_check"]
    comp = sum(tc[k] for k in ("model_pipeline_s", "simulator_step_s", "video_render_s", "run_state_s",
                                "observe_transition_s", "policy_compute_s"))
    cs = sorted((c for c in calls if c["run"] == key), key=lambda c: c["k"])
    lat = [c["latency_s"] for c in cs]
    out[key] = dict(
        raw=s.get("task_success"), v2=s.get("v2_success"), status=s["status"], steps=s["steps"],
        calls=len(cs), model=tc["model_pipeline_s"], sim=tc["simulator_step_s"], video=tc["video_render_s"],
        run_state=tc["run_state_s"], observe=tc["observe_transition_s"], policy=tc["policy_compute_s"],
        residual=s["control_elapsed_s"] - comp, control=s["control_elapsed_s"], elapsed=s["elapsed_s"],
        startup=s.get("startup_to_initial_observation_s"),
        startup_final=s["elapsed_s"] - s["control_elapsed_s"],
        prep=uc["preparation_elapsed_s"], prep_model=uc.get("preparation_model_s"),
        approval=uc.get("approval_to_claim_s") or 0.0,
        service=uc["preparation_elapsed_s"] + (uc.get("approval_to_claim_s") or 0.0) + s["elapsed_s"],
        lat=lat, first=lat[0], nonfirst_mean=st.mean(lat[1:]) if len(lat) > 1 else None,
        vis=[c["visible_tokens"] for c in cs], reas=[c["reasoning_tokens"] for c in cs],
        inp=[c["input_tokens"] for c in cs], prose=[c["prose_chars"] for c in cs],
        outc=[c["out_chars"] for c in cs])
json.dump(out, open(P + "A_runs.json", "w"), indent=0)
hdr = "key raw steps calls prep startup+fin model sim video rs obs pol resid service sim/step first"
print(hdr)
for k, r in out.items():
    print(k, int(bool(r["raw"])), r["steps"], r["calls"], *("%.1f" % r[x] for x in (
        "prep", "startup_final", "model", "sim", "video", "run_state", "observe", "policy", "residual", "service")),
          "%.4f" % (r["sim"] / r["steps"]), "%.1f" % r["first"])
sw = [out[k] for k in SWEEP]
svc = sorted((r["service"] if r["raw"] else float("inf")) for r in sw)
print("sweep service sorted (inf = raw fail):", ["%.1f" % x for x in svc])
print("sweep median (5th/6th):", "%.1f %.1f" % (svc[4], svc[5]))
succ = [out[k] for k in RUNS if out[k]["raw"]]
for name in ("prep", "startup_final", "model", "sim", "video", "run_state", "observe", "policy", "residual", "service", "steps", "calls"):
    vals = [r[name] for r in succ]
    print("succ mean %-14s %.2f  median %.2f" % (name, st.mean(vals), st.median(vals)))
steps = sum(r["steps"] for r in succ)
for name in ("sim", "video", "run_state", "observe", "policy", "residual"):
    print("pooled s/step %-10s %.4f" % (name, sum(r[name] for r in succ) / steps))
print("pooled startup s", st.mean([r["startup"] for r in succ]), "startup_final", st.mean([r["startup_final"] for r in succ]))
