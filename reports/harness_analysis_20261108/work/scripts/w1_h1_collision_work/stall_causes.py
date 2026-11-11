"""Classify segment_stall handoffs: near-target (<10 mm), forbidden-contact blocked, allowed-contact blocked, other."""
import json
from collections import Counter
W = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/h1_collision_work"
RUNS = [
 "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
 "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001",
 "agent_speedup_v2_uc71_default_routing_t04_20261103_001",
 "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
 "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001",
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001",
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003",
]
FORB = ["robot_scene", "robot_nonfinger_manipulated_object", "robot_object", "manipulated_object_scene"]
ALLOW = ["finger_object", "manipulated_object_support", "finger_articulated_part"]
tot = Counter(); detail = []
for run in RUNS:
    rows = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    t = {r["step"]: r for r in json.load(open(f"{W}/data/{run}_contacts.json"))["table"]}
    seen = set()
    for r in rows:
        if r["event"] != "action": continue
        fb = r["diagnostics"].get("feedback") or {}
        if fb.get("action") != "execute" or fb.get("reason") != "segment_stall": continue
        k = fb.get("observed_at_step")
        if k in seen: continue
        seen.add(k)
        segs = (fb.get("evidence") or {}).get("segment_trace") or []
        seg = segs[-1] if segs else {}
        err = seg.get("position_error_m")
        prof = (seg.get("request") or {}).get("profile")
        window = range(k - 12, k)
        forb = sum(t.get(s, {}).get("inc", {}).get(c, 0) for s in window for c in FORB)
        allow = {c: sum(t.get(s, {}).get("inc", {}).get(c, 0) for s in window) for c in ALLOW}
        # finger_object during an empty-hand approach is "allowed" but usually means pushing an object
        if forb:
            cause = "forbidden_contact_during_stall"
        elif err is not None and err < 0.010:
            cause = "near_target_under_10mm"
        elif any(allow.values()):
            cause = "allowed_contact_during_stall"
        else:
            cause = "no_contact_far_from_target"
        tot[cause] += 1
        detail.append((run[26:60], k, seg.get("skill"), prof, round(err or -1, 4), cause, {c: v for c, v in allow.items() if v}))
for d in detail: print(d)
print(dict(tot), "total", sum(tot.values()))
