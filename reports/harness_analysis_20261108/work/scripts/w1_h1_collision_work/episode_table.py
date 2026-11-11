"""Label each forbidden contact episode with a manipulation phase from skill/gripper/motion."""
import json, sys, glob, os
from collections import Counter, defaultdict
S = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/h1_collision_work/data"
runs = [
 "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
 "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001",
 "agent_speedup_v2_uc71_default_routing_t04_20261103_001",
 "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
 "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001",
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001",
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003",
]
FORB = ["robot_scene", "robot_nonfinger_manipulated_object", "robot_object", "manipulated_object_scene"]
def phase(row, prev_row):
    sk, prof, g = row["skill"], row["profile"], row["gripper_command"]
    dz = None
    if prev_row is not None and row["eef_after"][2] is not None and prev_row["eef_after"][2] is not None:
        dz = row["eef_after"][2] - prev_row["eef_after"][2]
    closed = g is not None and g > 0
    if sk == "set_gripper":
        return "grasp_close" if closed else "release_open"
    if sk == "local_motion":
        return "direct_motion_closed" if closed else "direct_motion_open"
    if prof == "transit":
        return "carry_transit" if closed else "free_transit"
    # contact-profile approach / move_relative contact
    if dz is None or abs(dz) < 2e-4:
        return ("contact_hold_closed" if closed else "contact_hold_open")
    if closed:
        return "lift_closed" if dz > 0 else "place_descent_closed"
    return "retreat_open" if dz > 0 else "grasp_descent_open"
all_eps = []
stage_actions = {}
for run in runs:
    t = json.load(open(f"{S}/{run}_contacts.json"))["table"]
    by_step = {r["step"]: r for r in t}
    # action counts per stage kind
    sc = Counter()
    for i, r in enumerate(t):
        sc[phase(r, t[i-1] if i else None)] += 1
    stage_actions[run] = dict(sc)
    for cls in FORB:
        cur = None
        for i, r in enumerate(t):
            v = r["inc"].get(cls, 0)
            if v:
                ph = phase(r, t[i-1] if i else None)
                pairs = Counter({k[len(cls)+1:]: n for k, n in r["pair_inc"].items() if k.startswith(cls + "|")})
                if cur and r["step"] == cur["end"] + 1:
                    cur["end"] = r["step"]; cur["samples"] += v; cur["steps"] += 1
                    cur["phases"][ph] += 1; cur["pairs"].update(pairs)
                else:
                    if cur: all_eps.append(cur)
                    cur = {"run": run[26:], "class": cls, "start": r["step"], "end": r["step"], "samples": v, "steps": 1,
                           "phases": Counter({ph: 1}), "pairs": pairs}
        if cur: all_eps.append(cur)
all_eps.sort(key=lambda e: (e["run"], e["start"]))
def short(p):
    out = []
    for part in p.split("|"):
        owner, geom = part.split(":")
        out.append(geom.replace("robot0_gripper", "").replace("_collision", "").replace("gripper0_", ""))
    return "~".join(out)
print("run | class | steps | n_steps | samples | phases | pairs(top)")
for e in all_eps:
    top = ", ".join(f"{short(k)}:{v}" for k, v in e["pairs"].most_common(3))
    print(f"{e['run']} | {e['class']} | {e['start']}-{e['end']} | {e['steps']} | {e['samples']} | {dict(e['phases'])} | {top}")
# aggregate samples by phase across all runs
agg = defaultdict(Counter)
eps_by_phase = Counter()
for e in all_eps:
    dom = e["phases"].most_common(1)[0][0]
    eps_by_phase[(e["class"], dom)] += 1
print()
print("episodes by (class, dominant phase):")
for k, v in sorted(eps_by_phase.items()): print("  ", k, v)
print()
print("stage action counts per run:")
for r, c in stage_actions.items(): print(r[26:], c)
json.dump({"episodes": [{**e, "phases": dict(e["phases"]), "pairs": dict(e["pairs"])} for e in all_eps], "stage_actions": stage_actions}, open(f"{S}/episodes_all.json", "w"), indent=1)
