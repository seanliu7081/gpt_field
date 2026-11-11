"""Calls that follow failed fragments, by reason, with latency; whether a stall coincided with forbidden contact."""
import json, sys
from collections import Counter, defaultdict
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
grand = defaultdict(lambda: [0, 0.0])
for run in RUNS:
    rows = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    t = {r["step"]: r for r in json.load(open(f"{W}/data/{run}_contacts.json"))["table"]}
    calls = []
    prev_fb = None
    for r in rows:
        if r["event"] != "action": continue
        d = r["diagnostics"]
        if d.get("api_calls_in_act"):
            calls.append((r["step"], d.get("api_latency_s") or 0.0, prev_fb or {}))
        prev_fb = d.get("feedback")
    per = defaultdict(lambda: [0, 0.0])
    stall_contact = Counter()
    for step, lat, fb in calls:
        action = fb.get("action")
        status, reason = fb.get("status"), fb.get("reason")
        if action == "execute" and status in ("failed",) or (action == "model_response" and status == "rejected") or (action == "direct_motion" and status == "rejected"):
            key = f"{action}:{reason}"
        elif action is None:
            key = "initial"
        else:
            key = "after_success"
        per[key][0] += 1; per[key][1] += lat
        grand[key][0] += 1; grand[key][1] += lat
        if key == "execute:segment_stall":
            # contact in the 15 steps before the stall handoff?
            hit = any(any(t.get(s, {}).get("inc", {}).get(c) for c in FORB) for s in range(step - 15, step))
            stall_contact["stall_with_forbidden_contact" if hit else "stall_without_forbidden_contact"] += 1
    print(run[26:], {k: (v[0], round(v[1], 1)) for k, v in sorted(per.items())}, dict(stall_contact))
print("ALL", {k: (v[0], round(v[1], 1)) for k, v in sorted(grand.items())})
