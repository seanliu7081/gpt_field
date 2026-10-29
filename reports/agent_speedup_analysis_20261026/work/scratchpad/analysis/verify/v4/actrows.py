import json
from concurrent.futures import ProcessPoolExecutor
R="/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_"
OUT="/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v4/"
ts="t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()
def proc(t):
    out=[]
    with open(R+t+"/trace.jsonl") as f:
        for line in f:
            if not line.startswith('{"event": "action"'): continue
            r=json.loads(line)
            dg=r["diagnostics"]
            fb=dg.get("feedback") or {}
            out.append(dict(step=r["step"],calls=dg.get("api_calls"),key=dg.get("api_request_key"),log=dg.get("api_request_log"),
                            prim=dg.get("primitive"),kind=dg.get("execution_kind"),
                            dec_type=((dg.get("routing") or {}).get("params") or {}).get("type"),
                            fb_obs=fb.get("observed_at_step"),fb_action=fb.get("action"),fb_reason=fb.get("reason"),fb_steps=fb.get("executed_steps"),
                            fb_evidence_segments=[{k:s.get(k) for k in ("index","skill","status","reason","actions","position_error_m","orientation_error_rad","entry_at_fragment_step")} | {"profile":(s.get("request") or {}).get("profile")} for s in ((fb.get("evidence") or {}).get("segments") or [])] if fb.get("action")=="execute" and fb.get("observed_at_step")==r["step"] else None,
                            decision_latency_s=r.get("decision_latency_s"),api_latency_s=dg.get("api_latency_s")))
    json.dump(out,open(OUT+"act_rows_%s.json"%t,"w"))
    return t,len(out),sum(1 for o in out if o["calls"])
if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=13) as ex:
        for res in ex.map(proc,ts): print(res,flush=True)
