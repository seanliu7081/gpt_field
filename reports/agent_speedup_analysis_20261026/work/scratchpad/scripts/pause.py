import json, sys, os
for r in sys.argv[1:]:
    p = os.path.join(r, "agent_stats.json")
    try:
        a = json.load(open(p))
    except Exception as e:
        print("==", r, "ERR", e); continue
    ep = a.get("evidence_pause") or {}
    b = ep.get("blocker") or {}
    print("==", r, "| policy_version:", a.get("policy_version"), "| versions:", a.get("versions",{}).get("agent_owned"), a.get("versions",{}).get("shared_geometry"))
    print(" pause.step:", ep.get("step"), "| reason:", (ep.get("reason") or "")[:400])
    print(" pause.missing_evidence_requests:", ep.get("missing_evidence_requests"))
    print(" blocker.status/reason:", b.get("status"), "/", b.get("reason"), "| execution_allowed:", b.get("execution_allowed"), "| blocking_reasons:", b.get("blocking_reasons"))
    print(" blocker.unknown_fraction:", b.get("unknown_fraction"), "| stale:", b.get("stale_evidence_ids"), "| monitoring_unknown:", b.get("monitoring_unknown_ids"), "| tracking:", b.get("tracking_status"), "| payload_unknown:", b.get("payload_geometry_unknown"))
    wg = b.get("whole_goal_assessment") or {}
    print(" whole_goal:", {k: wg.get(k) for k in ("status","reason","execution_allowed","blocking_reasons","missing_evidence_requests")})
    att = a.get("evidence_view_attempts")
    print(" evidence_view_attempts:", {k: len(v) for k,v in (att or {}).items()})
    ce = a.get("continuation_events") or []
    print(" continuation_events:", len(ce), [e.get("action") for e in ce][-8:])
    os_ = a.get("operation_state") or {}
    print(" operation_state.counts:", os_.get("counts"), "| bound:", os_.get("bound_object_id"), "| current.phase:", (os_.get("current") or {}).get("phase"))
