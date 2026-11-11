import pickle, collections, json
OUT = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
res = pickle.load(open(f"{OUT}/stages.pkl", "rb"))
rows = []
for lab, r in res.items():
    for k, f in r["frags"].items():
        for s in f["frag"]["segments"]:
            if s.get("routing") is None: continue
            ro = s["routing"]
            rows.append({"run": lab, "frag_start": k, "seg": s["index"], "entry_frag_step": s.get("entry_at_fragment_step"),
                         "global_entry": k + (s.get("entry_at_fragment_step") or 0) - 1 + 1,
                         "status": ro.get("status", "ok"), "reason": ro.get("reason"),
                         "sv": ro.get("straight_path_valid"), "planner_calls": (ro.get("planner") or {}).get("calls"),
                         "methods": (ro.get("planner") or {}).get("methods"), "seg_reason": s.get("reason"),
                         "skill": s["skill"], "routing": ro, "seg": s})
    for e in r["entry_fail"]:
        for s in e["segments"]:
            if s.get("routing") is None: continue
            ro = s["routing"]
            rows.append({"run": lab, "frag_start": e["observed_at_step"], "seg": s["index"], "entry_frag_step": 0,
                         "global_entry": e["observed_at_step"], "status": ro.get("status","ok"), "reason": ro.get("reason"),
                         "sv": ro.get("straight_path_valid"), "planner_calls": (ro.get("planner") or {}).get("calls"),
                         "methods": (ro.get("planner") or {}).get("methods"), "seg_reason": s.get("reason"),
                         "skill": s["skill"], "routing": ro, "seg": s, "entry_fail": True})
pickle.dump(rows, open(f"{OUT}/route_rows.pkl", "wb"))
for grp, labs in (("H1", ["T04_t100","T04_20mm","T04_default","T08_t100","T08_stoveoff","drawer001","drawer003"]), ("H3", ["H3_T04","H3_T08"])):
    R = [x for x in rows if x["run"] in labs]
    print(grp, "routed stage attempts", len(R))
    print("  status", collections.Counter(x["status"] for x in R), "reasons", collections.Counter(x["reason"] for x in R))
    print("  straight_valid", collections.Counter(x["sv"] for x in R))
    print("  planner calls", collections.Counter(x["planner_calls"] for x in R), "methods", collections.Counter(tuple(x["methods"] or []) for x in R))
    for x in R:
        if x["status"] != "ok":
            print("   refused:", x["run"], "global_entry~", x["global_entry"], x["reason"], "clear", x["routing"].get("clearance_m"))
    # tracking-time failures
    for x in R:
        if x["status"] == "ok" and x["seg_reason"] == "route_unavailable":
            print("   tracking-unavailable:", x["run"], x["global_entry"], (x["seg"].get("route_tracking") or {}).get("reason"), x["seg"].get("detail"))
    print("  seg end reasons (accepted)", collections.Counter(x["seg_reason"] for x in R if x["status"]=="ok"))
