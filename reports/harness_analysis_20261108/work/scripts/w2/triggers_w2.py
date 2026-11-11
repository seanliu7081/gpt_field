"""Independent trigger table: for each execution request, the previous decision's outcome from context.last_feedback."""
import json, glob, sys
from collections import Counter
RUNS = {"T04": "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
        "T08": "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
        "stove": "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001",
        "d001": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001",
        "d003": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003"}
def find_lf(obj):
    if isinstance(obj, dict):
        if "last_feedback" in obj: return obj["last_feedback"], True
        for v in obj.values():
            r, ok = find_lf(v)
            if ok: return r, True
    return None, False
out = {}
for name, run in RUNS.items():
    rows = []
    for f in sorted(glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/*.request.json")):
        d = json.load(open(f))
        last = d["payload"]["input"][-1]
        t = json.loads([c["text"] for c in last["content"] if c.get("type") == "input_text"][0])
        lf, _ = find_lf(t.get("context", t))
        s = lf.get("summary", lf) if isinstance(lf, dict) else None
        ev = (s or {}).get("evidence") or {}
        rows.append({"file": f.split("/")[-1][:6], "step": t.get("step"),
                     "action": (s or {}).get("action"), "status": (s or {}).get("status"), "reason": (s or {}).get("reason"),
                     "executed_steps": (s or {}).get("executed_steps"),
                     "skill": ev.get("current_skill"), "seg_status": ev.get("current_segment_status")})
    out[name] = rows
json.dump(out, open(sys.argv[1], "w"), indent=1)
tot = Counter()
for name, rows in out.items():
    c = Counter((r["action"], r["status"], r["reason"]) for r in rows)
    print(name, len(rows), dict(c)); tot.update(c)
print("TOTAL", sum(tot.values()), dict(tot))
