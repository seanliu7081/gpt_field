"""For no-contact detector fires: did the existing stall predicate / segment end fire within 15 steps anyway?"""
import json
import numpy as np
from collections import Counter
W = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/h1_collision_work"
RUNS = {
 "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001": 0.10,
 "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001": 0.05,
 "agent_speedup_v2_uc71_default_routing_t04_20261103_001": 0.05,
 "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001": 0.10,
 "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001": 0.10,
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001": 0.10,
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003": 0.10,
}
K, CMIN, PMAX = 3, 0.002, 0.0003
FORB = ["robot_scene", "robot_nonfinger_manipulated_object", "robot_object", "manipulated_object_scene"]
out = Counter(); per_run = Counter()
for run, scale in RUNS.items():
    rows = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    fb = {r["step"]: r["feedback"] for r in rows if r["event"] == "feedback"}
    t = {r["step"]: r for r in json.load(open(f"{W}/data/{run}_contacts.json"))["table"]}
    # stall / failure handoff steps
    ends = {}
    for r in rows:
        if r["event"] != "action": continue
        f = r["diagnostics"].get("feedback") or {}
        if f.get("action") == "execute" and f.get("observed_at_step") is not None:
            ends[f["observed_at_step"]] = f.get("reason")
    streak = 0
    for step in sorted(fb):
        f = fb[step]; na, md = f.get("normalized_action"), f.get("measured_translation_delta_m")
        row = t.get(step, {})
        if na is None or md is None: streak = 0; continue
        c = np.asarray(na[:3]) * scale; cn = float(np.linalg.norm(c))
        if cn < CMIN: streak = 0; continue
        prog = float(np.dot(np.asarray(md), c / cn))
        streak = streak + 1 if prog < PMAX else 0
        if streak < K: continue
        inc = row.get("inc", {})
        if any(any(t.get(s, {}).get("inc", {}).get(k) for k in FORB) for s in range(step - 3, step + 4)): continue
        if any(inc.get(k) for k in ("finger_articulated_part", "manipulated_object_support", "finger_object")): continue
        nxt = [(s, ends[s]) for s in ends if step < s <= step + 15]
        key = nxt[0][1] if nxt else "no_handoff_within_15"
        out[key] += 1; per_run[run[26:60]] += 1
print(dict(out)); print(dict(per_run))
