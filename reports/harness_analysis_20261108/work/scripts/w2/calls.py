import json, sys
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
from extract import RUNLIST, RUNS
def calls(label):
    out = []
    with open(f"{RUNS}/{RUNLIST[label]}/trial/trace.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("event") != "action": continue
            dg = d.get("diagnostics") or {}
            ac = dg.get("api_calls") or []
            if not ac: continue
            fb = dg.get("feedback") or {}
            p = (dg.get("routing") or {}).get("params") or {}
            ex = p.get("execution") or {}
            segs = ex.get("segments") if ex else None
            out.append({"step": d["step"], "lat": [round(c.get("latency_s") or 0, 1) for c in ac], "kinds": [c.get("kind") for c in ac],
                        "prev_reason": fb.get("reason"), "prev_action": fb.get("action"), "prev_obs_step": fb.get("observed_at_step"),
                        "type": p.get("type"), "why": (p.get("why") or "")[:140],
                        "segs": [(s["skill"], s["profile"], s.get("xyz_m"), s.get("route") is not None) for s in segs] if segs else None,
                        "map": p.get("obstacle_map")})
    return out
if __name__ == "__main__":
    for c in calls(sys.argv[1]):
        print(c["step"], c["lat"], c["kinds"], "prev:", c["prev_action"], c["prev_reason"], "| new:", c["type"], "| map" if c["map"] else "", "|", c["why"])
