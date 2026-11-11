"""Absolute-progress variant (RMAX is metres per step). Offline detectability of forbidden contact from proprioception only (commanded vs measured motion).
For each step: c = commanded translation (normalized action * scale), m = measured translation delta.
ratio = (m . c_hat)/|c|. Detector fires when K consecutive steps have |c|>=cmin and ratio<rmax,
excluding set_gripper/wait holds (no commanded motion). Diagnostic only; never fed to a policy here."""
import json, sys
import numpy as np
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
K, CMIN, RMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 3, float(sys.argv[2]) if len(sys.argv) > 2 else 0.002, float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
FORB = ["robot_scene", "robot_nonfinger_manipulated_object", "robot_object", "manipulated_object_scene"]
eps = json.load(open(f"{W}/data/episodes_all.json"))["episodes"]
tot = {"episodes": 0, "detected_before_end": 0, "lead_steps": [], "samples": 0, "samples_after_detect": 0,
       "fp_fires": 0, "free_steps": 0}
for run, scale in RUNS.items():
    rows = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    fb = {r["step"]: r["feedback"] for r in rows if r["event"] == "feedback"}
    t = json.load(open(f"{W}/data/{run}_contacts.json"))["table"]
    forb_steps = {r["step"] for r in t if any(r["inc"].get(c) for c in FORB)}
    flags, ratios = {}, {}
    streak = 0
    for step in sorted(fb):
        f = fb[step]
        na = f.get("normalized_action")
        md = f.get("measured_translation_delta_m")
        if na is None or md is None:
            flags[step] = False; streak = 0; continue
        c = np.asarray(na[:3]) * scale
        m = np.asarray(md)
        cn = float(np.linalg.norm(c))
        if cn < CMIN:
            streak = 0; flags[step] = False; continue
        prog = float(np.dot(m, c / cn)); ratio = prog / cn
        ratios[step] = ratio
        streak = streak + 1 if (prog < RMAX) else 0
        flags[step] = streak >= K
    # false positives: fires on steps with no forbidden contact in a +-3 step window
    for step, fl in flags.items():
        if any(s in forb_steps for s in range(step - 3, step + 4)):
            continue
        tot["free_steps"] += 1
        if fl: tot["fp_fires"] += 1
    for e in eps:
        if not run.endswith(e["run"]): continue
        tot["episodes"] += 1
        fire = next((s for s in range(e["start"], e["end"] + 1) if flags.get(s)), None)
        samples = {r["step"]: sum(r["inc"].get(c, 0) for c in [e["class"]]) for r in t}
        tot["samples"] += e["samples"]
        if fire is not None:
            tot["detected_before_end"] += 1
            tot["lead_steps"].append(fire - e["start"])
            tot["samples_after_detect"] += sum(v for s, v in samples.items() if fire < s <= e["end"])
        print(run[26:66], e["class"][:24], f"{e['start']}-{e['end']}", "fire@", fire,
              "ratios", [round(ratios.get(s, float('nan')), 2) for s in range(e["start"], min(e["end"], e["start"] + 6) + 1)])
print(json.dumps({**tot, "lead_steps": sorted(tot["lead_steps"])}, indent=None))
