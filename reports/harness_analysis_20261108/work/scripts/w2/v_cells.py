import json, glob, numpy as np
R = "/workspace/gpt_field/runs/"
def cell_box(c, W=512, H=512):
    r, col = (c - 1) // 8, (c - 1) % 8
    return col * W // 8, r * H // 8, (col + 1) * W // 8, (r + 1) * H // 8
for run in ["foundation_vision_t04_can_obstacle_20261105_001", "foundation_vision_t08_bottle_obstacle_20261105_001"]:
    plan = json.load(open(R + run + "/trial/approved_plan.json"))
    cells = {p["id"]: (p["x"], p["w"]) for p in plan["parts"]}
    print("=====", run, cells)
    stats = {"amb_resolved": 0, "amb_total": 0, "single_consistent": 0, "single_total": 0}
    for f in sorted(glob.glob(R + run + "/trial/foundation_vision/frame_*_sam_candidates.json")):
        meta = json.load(open(f)); arr = np.load(f.replace(".json", ".npz")); fid = meta["frame_id"]
        by = {}
        for r in meta["candidates"]:
            if r["view"] == "external" and r["target_key"] in cells: by.setdefault(r["target_key"], []).append(r)
        for k, rows in by.items():
            x0, y0, x1, y1 = cell_box(cells[k][0])
            ov = []
            for r in rows:
                m = arr[r["array_key"]]
                ov.append(int(m[y0:y1, x0:x1].sum()))
            if len(rows) > 1:
                stats["amb_total"] += 1; stats["amb_resolved"] += int(sum(o > 0 for o in ov) == 1)
                print(f"  f{fid:4d} {k} AMBIG cell {cells[k][0]} overlaps px per cand {ov} bboxes {[r['bbox_xyxy_exclusive'] for r in rows]}")
            else:
                stats["single_total"] += 1; stats["single_consistent"] += int(ov[0] > 0)
                if ov[0] == 0: print(f"  f{fid:4d} {k} single cand NOT in cell {cells[k][0]} bbox {rows[0]['bbox_xyxy_exclusive']}")
    print("  ", stats)
