"""Diagnostic only: compare model obstacle boxes to the true added-obstacle AABB (layout manifest)."""
import json, sys
import numpy as np
run = sys.argv[1]
key = sys.argv[2]  # 'can' or 'bottle'
lay = json.load(open(f"{run}/inputs/layout_manifest.json"))
tb = lay[key]["collision_bounds"]
tl, tu = np.array(tb["lower_xyz_m"]), np.array(tb["upper_xyz_m"])
rows = [json.loads(l) for l in open(f"{run}/trial/trace.jsonl")]
maps = []
last = None
for r in rows:
    if r["event"] != "action": continue
    d = r["diagnostics"]
    if not d.get("refreshed"): continue
    p = (d.get("routing") or {}).get("params") or {}
    om = p.get("obstacle_map")
    if om is None: continue
    k = json.dumps(om, sort_keys=True)
    if k == last: continue
    last = k
    maps.append((r["step"], om))
def vol(l, u): return float(np.prod(np.clip(u - l, 0, None)))
out = []
for step, om in maps:
    c = om["clearance_m"]
    best = None
    for i, b in enumerate(om["avoid_boxes"]):
        l, u = np.array(b["lower_xyz_m"]), np.array(b["upper_xyz_m"])
        il, iu = np.maximum(l, tl), np.minimum(u, tu)
        inter = vol(il, iu)
        iou = inter / (vol(l, u) + vol(tl, tu) - inter)
        cdist = float(np.linalg.norm((l + u) / 2 - (tl + tu) / 2))
        score = (inter, -cdist)
        # per-axis fraction of true extent covered by the raw box and expanded box
        cov = np.clip(np.minimum(u, tu) - np.maximum(l, tl), 0, None) / (tu - tl)
        el, eu = l - c, u + c
        ecov = np.clip(np.minimum(eu, tu) - np.maximum(el, tl), 0, None) / (tu - tl)
        # inflation needed so expanded box contains the true AABB
        need = float(max(np.max(l - tl), np.max(tu - u), 0.0))
        rec = {"step": step, "box_index": i, "clearance_m": c, "box": [l.round(3).tolist(), u.round(3).tolist()],
               "center_error_xyz_m": ((l + u) / 2 - (tl + tu) / 2).round(3).tolist(), "center_dist_m": round(cdist, 3),
               "iou": round(iou, 3), "true_volume_covered_frac": round(inter / vol(tl, tu), 3),
               "axis_cover_raw": cov.round(2).tolist(), "axis_cover_expanded": ecov.round(2).tolist(),
               "expanded_contains_true": bool(np.all(el <= tl) and np.all(eu >= tu)),
               "inflation_needed_to_contain_m": round(need, 3)}
        if best is None or score > best[0]:
            best = (score, rec)
    out.append(best[1] if best else {"step": step, "no_boxes": True})
print("true AABB", tl.round(3).tolist(), tu.round(3).tolist())
for o in out: print(json.dumps(o))
