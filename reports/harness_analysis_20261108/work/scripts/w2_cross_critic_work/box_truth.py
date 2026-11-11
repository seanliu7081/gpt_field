"""Score every model-written obstacle box for the added obstacle against layout truth (diagnostic only)."""
import json, glob, os
L = "/workspace/gpt_field/output/layouts"
truth = {
  "T04": json.load(open(f"{L}/libero_10_t4_can_obstacle_20261103_001/layout_manifest.json"))["can"],
  "T08": json.load(open(f"{L}/libero_10_t8_bottle_obstacle_20261103_001/layout_manifest.json"))["bottle"],
}
so = json.load(open(f"{L}/libero_10_t8_stove_off_bottle_obstacle_20261103_001/layout_manifest.json"))
truth["T08so"] = so.get("bottle") or next(v for k, v in so.items() if isinstance(v, dict) and "collision_bounds" in v)
RUNS = {
  "T20_T04": ("agent_speedup_v2_uc71_can_obstacle_t04_20261103_001", "T04"),
  "H1_T04": ("agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001", "T04"),
  "H3_T04": ("foundation_vision_t04_can_obstacle_20261105_001", "T04"),
  "H1_T08": ("agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001", "T08"),
  "H1_stoveoff": ("agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001", "T08so"),
  "H3_T08": ("foundation_vision_t08_bottle_obstacle_20261105_001", "T08"),
}
def vol(lo, hi): return max(0, hi[0]-lo[0])*max(0, hi[1]-lo[1])*max(0, hi[2]-lo[2])
def inter(a, b):
    lo = [max(a[0][i], b[0][i]) for i in range(3)]; hi = [min(a[1][i], b[1][i]) for i in range(3)]
    return vol(lo, hi)
out = {}
for name, (run, tk) in RUNS.items():
    t = truth[tk]["collision_bounds"]; T = (t["lower_xyz_m"], t["upper_xyz_m"]); table = truth[tk]["table_height_m"]
    tc = [(T[0][i]+T[1][i])/2 for i in range(3)]
    rows = []
    for p in sorted(glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/*.response.json")):
        r = json.load(open(p)); txt = None
        for o in r.get("output", []):
            if o.get("type") == "message":
                for c in o.get("content", []):
                    if c.get("type") == "output_text": txt = c["text"]
        a = json.loads(txt); om = a.get("obstacle_map")
        if not om: continue
        cl = om["clearance_m"]
        best = None
        for b in om["avoid_boxes"]:
            B = (b["lower_xyz_m"], b["upper_xyz_m"])
            bc = [(B[0][i]+B[1][i])/2 for i in range(3)]
            d = sum((bc[i]-tc[i])**2 for i in range(2))**0.5
            if best is None or d < best[0]: best = (d, B, bc)
        d, B, bc = best
        Be = ([B[0][i]-cl for i in range(3)], [B[1][i]+cl for i in range(3)])
        iou = inter(B, T)/(vol(*B)+vol(*T)-inter(B, T))
        rows.append({"call": int(os.path.basename(p)[:6]), "clearance": cl,
                     "iou": round(iou, 3), "cov_raw": round(inter(B, T)/vol(*T), 3), "cov_expanded": round(inter(Be, T)/vol(*T), 3),
                     "centre_err_xyz_cm": [round(100*(bc[i]-tc[i]), 1) for i in range(3)],
                     "box_bottom_minus_table_cm": round(100*(B[0][2]-table), 1),
                     "box_top_minus_true_top_cm": round(100*(B[1][2]-T[1][2]), 1)})
    out[name] = rows
    print(name, tk, "truth", [round(x, 3) for x in T[0]], [round(x, 3) for x in T[1]], "table", table)
    for x in rows: print("   ", x)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "data", "box_truth.json"), "w"), indent=1)
