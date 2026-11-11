"""Diagnostic only (truth never feeds a policy): with the TRUE added-obstacle AABB + the model's clearance,
which recorded routed transit stages would (a) a grip-site point check and (b) a whole-hand swept check flag?"""
import json, sys
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/h1_collision_work/scripts")
from hand_sweep import R, slerp, HAND, obb_aabb_overlap
CASES = {"agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001": "bottle",
         "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001": "bottle",
         "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001": "can",
         "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001": "can"}
def seg_hits_aabb(a, b, lo, hi):
    for t in np.linspace(0, 1, 201):
        p = a + t*(b-a)
        if np.all(p >= lo) and np.all(p <= hi): return True
    return False
for run, key in CASES.items():
    lay = json.load(open(f"/workspace/gpt_field/runs/{run}/inputs/layout_manifest.json"))[key]["collision_bounds"]
    tlo, thi = np.asarray(lay["lower_xyz_m"]), np.asarray(lay["upper_xyz_m"])
    rows = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    seen = set(); res = []
    for r in rows:
        if r["event"] != "action": continue
        fb = r["diagnostics"].get("feedback") or {}
        if fb.get("action") != "execute": continue
        k = fb.get("observed_at_step")
        if k in seen: continue
        seen.add(k)
        for s in (fb.get("evidence") or {}).get("segment_trace") or []:
            ro = s.get("routing") or {}
            if not ro.get("straight_path_valid") or s.get("actions", 0) == 0: continue
            c = ro["clearance_m"]; lo, hi = tlo - c, thi + c
            a = np.asarray(s["entry_pose_wxyz"][:3]); b = np.asarray(s["target_xyz_m"])
            q0, q1 = np.asarray(s["entry_pose_wxyz"][3:7]), np.asarray(s["target_quaternion_wxyz"])
            point = seg_hits_aabb(a, b, lo, hi)
            hand = False
            for t in np.linspace(0, 1, 41):
                p = a + t*(b-a); Rm = R(slerp(q0, q1, t))
                for hlo, hhi in HAND:
                    if obb_aabb_overlap(p + Rm @ ((hlo+hhi)/2), Rm, (hhi-hlo)/2, lo, hi): hand = True; break
                if hand: break
            if point or hand: res.append((k, s["index"], "point" if point else "-", "hand" if hand else "-"))
    print(run[26:70], key, "stages flagged with true AABB (step, seg, point, hand):", res)
