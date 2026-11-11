"""Break down guarded-motion detector fires (absolute progress variant) by contact type and stage."""
import json, sys
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
c_all = Counter(); moving = Counter()
for run, scale in RUNS.items():
    rows = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    fb = {r["step"]: r["feedback"] for r in rows if r["event"] == "feedback"}
    t = {r["step"]: r for r in json.load(open(f"{W}/data/{run}_contacts.json"))["table"]}
    streak = 0
    for step in sorted(fb):
        f = fb[step]; na, md = f.get("normalized_action"), f.get("measured_translation_delta_m")
        row = t.get(step, {})
        if na is None or md is None: streak = 0; continue
        c = np.asarray(na[:3]) * scale; m = np.asarray(md); cn = float(np.linalg.norm(c))
        if cn < CMIN: streak = 0; continue
        prog = float(np.dot(m, c / cn))
        streak = streak + 1 if prog < PMAX else 0
        inc = row.get("inc", {})
        window_forb = any(any(t.get(s, {}).get("inc", {}).get(k) for k in FORB) for s in range(step - 3, step + 4))
        if window_forb: kind = "forbidden_contact"
        elif inc.get("finger_articulated_part"): kind = "intended_articulated_contact"
        elif inc.get("manipulated_object_support"): kind = "object_on_support"
        elif inc.get("finger_object"): kind = "finger_on_object"
        else: kind = "no_contact"
        stage = f"{row.get('skill')}/{row.get('profile')}/grip={'closed' if (row.get('gripper_command') or 0) > 0 else 'open'}"
        moving[kind] += 1
        if streak >= K:
            c_all[(kind, stage)] += 1
print("moving steps by contact kind:", dict(moving))
fires = Counter()
for (kind, stage), n in c_all.items(): fires[kind] += n
print("fires by contact kind:", dict(fires))
for k, v in sorted(c_all.items(), key=lambda kv: -kv[1])[:20]: print("  ", k, v)
