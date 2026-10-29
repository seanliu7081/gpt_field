import json, sys, os, glob
def short(v, n=500):
    s = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + "...[+%d]" % (len(s)-n)
def user_fields(payload):
    users = [i for i in payload["input"] if i.get("role") == "user"]
    for part in users[-1]["content"]:
        if part.get("type") == "input_text":
            try:
                f = json.loads(part["text"])
                if isinstance(f, dict) and "context" in f:
                    return f
            except Exception:
                pass
    return {}
def output_decision(resp):
    for item in resp.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    try:
                        return json.loads(part["text"])
                    except Exception:
                        return {"raw": part["text"][:300]}
    return None
run = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
reqs = sorted(glob.glob(os.path.join(run, "requests", "*.request.json")))
print("#####", run, "requests:", len(reqs))
for path in reqs[-n:]:
    base = path[:-len(".request.json")]
    req = json.load(open(path))
    f = user_fields(req["payload"])
    core = f.get("current_decision") or {}
    ctx = f.get("context") or {}
    print("---", os.path.basename(base), "| step:", f.get("step"), "| ordinal:", req.get("request_ordinal"), "| prompt_sizes.wire.total:", ((req.get("prompt_sizes") or {}).get("wire") or {}).get("total_prompt_chars"))
    b = core.get("blocker") or {}
    print("  blocker:", short({k: b.get(k) for k in ("status","reason","execution_allowed","blocking_reasons","payload_geometry_unknown","tracking_status","motion_support_revoked","stale_evidence_ids","monitoring_unknown_ids","final_interaction_guard")}, 700))
    print("  missing_evidence_requests:", short(core.get("missing_evidence_requests"), 600))
    pi = core.get("pending_intent") or {}
    print("  pending_intent:", short({k: pi.get(k) for k in ("kind","phase","stage_index","current_operation_id","remaining_watchdog_steps","missing_evidence_requests")}, 500), "| current_target:", None if not pi else (pi.get("current_target") is not None))
    print("  repeated_action_guard:", short(core.get("repeated_action_guard"), 300))
    ea = core.get("evidence_acquisition")
    if ea is not None:
        print("  evidence_acquisition:", short({k: ea.get(k) for k in ("reason","rejected","focus_source","focus_entity_id","payload_geometry_unknown")} | {"n_candidates": len(ea.get("candidates") or [])}, 500))
    lf = ctx.get("last_feedback")
    print("  last_feedback:", short(lf, 900))
    op = core.get("operation_state") or {}
    print("  operation_state:", short({"bound": op.get("bound_object_id"), "last_command": op.get("last_command"), "phase": (op.get("current") or {}).get("phase")}, 300))
    resp_path = base + ".response.json"
    if os.path.exists(resp_path):
        resp = json.load(open(resp_path))
        d = output_decision(resp) or {}
        u = resp.get("usage") or {}
        tr = resp.get("_subscription_transport") or {}
        print("  RESPONSE type:", d.get("type"), "| why:", short(d.get("why"), 600))
        print("   continuation:", d.get("continuation"), "| motion:", short(d.get("motion"),200), "| execution:", short(d.get("execution"), 300), "| program:", short(d.get("program"), 300))
        su = d.get("scene_update") or {}
        print("   scene_update: entities", [e.get("id") for e in su.get("entities", [])], "invalidate", su.get("invalidate_ids"), "remove", su.get("remove_ids"))
        print("   motion_context:", short(d.get("motion_context"), 400))
        print("   usage:", {k: u.get(k) for k in ("input_tokens","output_tokens")}, (u.get("output_tokens_details") or {}), "| v3_bounds.elapsed_s:", (tr.get("v3_bounds") or {}).get("elapsed_s"))
    res_path = base + ".result.json"
    if os.path.exists(res_path):
        res = json.load(open(res_path))
        print("  RESULT status:", res.get("status"), "| error:", short(res.get("error"), 300))
