"""Collect per-run inputs for the S300 budget (uc7 runs): timing components, per-call latency/tokens,
prose share of visible output, envelope-vetoed calls."""
import json, sys, os, collections
G = "/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_"
A = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/"
RUNS = "t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()
calls = json.load(open(A + "exec_calls.json"))["uc7"]
bycall = collections.defaultdict(list)
for c in calls:
    bycall[c["run"].rsplit("_", 2)[-2] + "_" + c["run"].rsplit("_", 1)[-1]].append(c)
out = {}
for t in RUNS:
    s = json.load(open(G + t + "/summary.json"))
    tc = s["timing_components"]; uc = s["user_check"]
    steps = s["steps"]; ctrl = s["control_elapsed_s"]; el = s["elapsed_s"]
    comp = sum(tc[k] for k in ("model_pipeline_s", "simulator_step_s", "video_render_s", "run_state_s",
                                "observe_transition_s", "policy_compute_s"))
    lat = []; prose = []; full = []
    log = G + t + "_launch/runner.log"
    for line in open(log):
        if '"event":"model_call"' not in line: continue
        r = json.loads(line); p = r.get("params") or {}
        lat.append(r["latency_s"])
        pr = sum(len(p.get(k) or "") for k in ("belief", "plan", "why"))
        fl = len(json.dumps(p, separators=(",", ":")))
        prose.append(pr); full.append(fl)
    cs = sorted(bycall[t], key=lambda c: c["ordinal"])
    out[t] = dict(
        status=s["status"], raw=s.get("task_success"), v2=s.get("v2_success"), steps=steps,
        prep_s=uc["preparation_elapsed_s"], prep_model_s=(uc.get("preparation") or {}).get("draft_model_s") if isinstance(uc.get("preparation"), dict) else None,
        approval_to_claim_s=uc.get("approval_to_claim_s"), elapsed_s=el, control_s=ctrl,
        startup_final_s=el - ctrl, startup_s=s.get("startup_to_initial_observation_s"),
        model_s=tc["model_pipeline_s"], sim_s=tc["simulator_step_s"], video_s=tc["video_render_s"],
        run_state_s=tc["run_state_s"], observe_s=tc["observe_transition_s"], policy_s=tc["policy_compute_s"],
        residual_s=ctrl - comp, n_calls=len(lat), lat=lat,
        visible=[c.get("visible") for c in cs], reasoning=[c.get("reasoning_tokens") for c in cs],
        output_tokens=[c.get("output_tokens") for c in cs], input_tokens=[c.get("input_tokens") for c in cs],
        prose_chars=prose, json_chars=full,
        service_s=uc["preparation_elapsed_s"] + (uc.get("approval_to_claim_s") or 0) + el)
json.dump(out, open(sys.argv[1], "w"), indent=1)
for t, v in out.items():
    ps = sum(v["prose_chars"]) / max(1, sum(v["json_chars"]))
    print(f"{t} {v['status']:8s} steps {v['steps']:5d} calls {v['n_calls']:2d} model {v['model_s']:6.1f} prep {v['prep_s']:5.1f} "
          f"su+fin {v['startup_final_s']:4.1f} startup {v['startup_s']:4.1f} resid {v['residual_s']:4.1f} service {v['service_s']:6.1f} "
          f"prose_share {ps:.2f} vis_mean {sum(x or 0 for x in v['visible'])/max(1,len(v['visible'])):5.0f} "
          f"reas_mean {sum(x or 0 for x in v['reasoning'])/max(1,len(v['reasoning'])):5.0f} first {v['lat'][0]:.1f}")
