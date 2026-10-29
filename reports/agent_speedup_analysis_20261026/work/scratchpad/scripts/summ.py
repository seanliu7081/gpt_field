import json, sys, os
runs = sys.argv[1:]
for r in runs:
    p = os.path.join(r, "summary.json")
    try:
        s = json.load(open(p))
    except Exception as e:
        print("==", r, "ERR", e); continue
    err = s.get("error")
    perf = {}
    pp = os.path.join(r, "performance.json")
    if os.path.exists(pp):
        perf = json.load(open(pp)).get("completion_target", {})
    tc = s.get("timing_components", {})
    mc = s.get("model_calls", {})
    print("==", r)
    print(" status:", s.get("status"), "| failure_reason:", s.get("failure_reason"), "| libero_success:", s.get("libero_success"), "| task_success:", s.get("task_success"))
    print(" steps:", s.get("steps"), s.get("episode_steps"), s.get("actions"), "| elapsed_s:", s.get("elapsed_s"), "| control_elapsed_s:", s.get("control_elapsed_s"))
    print(" policy_version:", (s.get("user_check") or {}).get("policy_version"), s.get("policy_version"), s.get("version"))
    print(" model_calls runner_calls_by_kind:", mc.get("runner_calls_by_kind"), "| failed:", mc.get("failed_calls_by_kind"))
    print(" timing:", {k: round(v,3) if isinstance(v,(int,float)) else v for k,v in tc.items() if not isinstance(v, dict)})
    print(" perf completion_target:", {k: perf.get(k) for k in ("preparation_elapsed_s","reviewed_execution_elapsed_s","total_service_elapsed_s","achieved")})
    if isinstance(err, dict):
        msg = err.get("message") or err.get("detail") or ""
        print(" error.type:", err.get("type"), "| msg[:1200]:", str(msg)[:1200])
    else:
        print(" error:", str(err)[:1200])
