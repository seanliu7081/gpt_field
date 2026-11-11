"""Joint-limit margins at each execution request (measured_state.joint_position_rad) vs robosuite Panda ranges."""
import json, glob, sys
LIM = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]
RUNS = {"T04": "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
        "T08": "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
        "stove": "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001",
        "d001": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001",
        "d003": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003"}
def find(o, key):
    if isinstance(o, dict):
        if key in o: return o[key]
        for v in o.values():
            r = find(v, key)
            if r is not None: return r
    return None
for name, run in RUNS.items():
    out = []
    for k, f in enumerate(sorted(glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/*.request.json"))):
        d = json.load(open(f)); last = d["payload"]["input"][-1]
        t = json.loads([c["text"] for c in last["content"] if c.get("type") == "input_text"][0])
        q = find(t.get("context", t), "joint_position_rad")
        if not q: out.append((k + 1, None)); continue
        m = [min(qi - lo, hi - qi) for qi, (lo, hi) in zip(q, LIM)]
        j = min(range(7), key=lambda i: m[i])
        out.append((k + 1, j + 1, round(m[j], 4)))
    near = [x for x in out if x[1] is not None and x[2] < 0.01]
    print(name, "calls with a joint within 0.01 rad of a limit:", near)
