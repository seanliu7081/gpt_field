"""Transit-stage overshoot vs clearance, and grip-site distance to the true added obstacle (diagnostic truth)."""
import json, os, math
import numpy as np
RUNS = {
  "H1_T04": ("agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001", "T04"),
  "H1_T08": ("agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001", "T08"),
  "H1_stoveoff": ("agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001", "T08"),
  "H1_drawer001": ("agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001", None),
  "H1_drawer003": ("agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003", None),
  "H3_T04": ("foundation_vision_t04_can_obstacle_20261105_001", "T04"),
  "H3_T08": ("foundation_vision_t08_bottle_obstacle_20261105_001", "T08"),
  "T20_T04": ("agent_speedup_v2_uc71_can_obstacle_t04_20261103_001", "T04"),
}
TRUTH = {"T04": (np.array([-0.0545, -0.2454, 0.4370]), np.array([0.0077, -0.1837, 0.5132])),
         "T08": (np.array([-0.0284, -0.0896, 0.9002]), np.array([0.0278, -0.0529, 1.0458]))}
def box_dist(p, lo, hi):
    q = np.maximum(lo - p, 0) + np.maximum(p - hi, 0)
    return float(np.linalg.norm(q))
out = {}
for name, (run, tk) in RUNS.items():
    lines = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    segs = {}  # key -> dict
    frag_id = 0; last_exec = None
    for l in lines:
        if l.get("event") != "feedback": continue
        pf = (l["feedback"].get("primitive_feedback") or {})
        fr = pf.get("fragment")
        if not fr: continue
        es = fr.get("executed_steps")
        if last_exec is not None and es is not None and es < last_exec: frag_id += 1
        last_exec = es
        i = fr.get("current_segment_index")
        if i is None: continue
        seg = next((s for s in fr.get("segments", []) if s.get("index") == i), None)
        if not seg: continue
        req = seg.get("request") or {}
        if req.get("profile") != "transit" or req.get("skill") not in ("move_to_pose", "move_relative"): continue
        key = (frag_id, i)
        d = segs.setdefault(key, {"step0": l["step"], "entry": seg.get("entry_pose_wxyz")[:3], "target": seg.get("target_xyz_m"),
                                   "clearance": (seg.get("routing") or {}).get("clearance_m"), "pos": [], "status": None, "reason": None})
        d["pos"].append(fr["measured_pose_wxyz"][:3]); d["status"] = seg.get("status"); d["reason"] = seg.get("reason")
    rows = []
    for key, d in segs.items():
        if d["target"] is None or len(d["pos"]) < 3: continue
        P = np.array(d["pos"]); T = np.array(d["target"]); E = np.array(d["entry"])
        L = np.linalg.norm(T - E)
        if L < 0.03: continue
        u = (T - E) / L
        dist = np.linalg.norm(P - T, axis=1); kmin = int(np.argmin(dist))
        along = (P - T) @ u
        over = float(max(0.0, along[kmin:].max()))
        rebound = float(dist[kmin:].max() - dist[kmin])
        r = {"step0": d["step0"], "len_m": round(float(L), 3), "dmin_mm": round(1000*float(dist.min()), 1),
             "overshoot_along_mm": round(1000*over, 1), "rebound_mm": round(1000*rebound, 1), "clearance": d["clearance"],
             "status": d["status"], "reason": d["reason"], "n": len(P)}
        if tk:
            lo, hi = TRUTH[tk]
            r["min_dist_true_obstacle_mm"] = round(1000*min(box_dist(p, lo, hi) for p in P), 1)
            # distance of target itself to obstacle
            r["target_dist_true_obstacle_mm"] = round(1000*box_dist(T, lo, hi), 1)
        rows.append(r)
    out[name] = rows
    ov = [r["overshoot_along_mm"] for r in rows]
    print(f"{name:13s} transit stages {len(rows):3d}  overshoot>5mm {sum(1 for x in ov if x>5):3d}  >10mm {sum(1 for x in ov if x>10):3d} max {max(ov) if ov else 0:5.1f}mm  "
          f"stall-ended {sum(1 for r in rows if r['reason']=='segment_stall')}")
    if tk:
        near = [r for r in rows if r["min_dist_true_obstacle_mm"] < 60]
        for r in sorted(near, key=lambda r: r["min_dist_true_obstacle_mm"])[:6]:
            print("     near obstacle:", r)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "data", "overshoot.json"), "w"), indent=1)
# pooled overshoot vs clearance
allr = [r for k, rs in out.items() if not k.startswith("T20") for r in rs]
c = [r for r in allr if r["clearance"] is not None and r["overshoot_along_mm"] > 1000*r["clearance"]]
print("transit100 stages where along-path overshoot exceeded the model clearance in use:", len(c), "of", len(allr))
