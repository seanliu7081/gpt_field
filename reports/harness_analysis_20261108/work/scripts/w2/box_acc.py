import json, sys, numpy as np
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
from extract import RUNLIST, RUNS
from calls import calls
CASES = {"T04_t100": "can", "T04_20mm": "can", "T08_t100": "bottle", "T08_stoveoff": "bottle", "H3_T04": "can", "H3_T08": "bottle"}
def iou(a, b):
    lo = np.maximum(a[0], b[0]); hi = np.minimum(a[1], b[1])
    inter = np.prod(np.clip(hi - lo, 0, None))
    va, vb = np.prod(a[1]-a[0]), np.prod(b[1]-b[0])
    return inter / (va + vb - inter)
for lab, key in CASES.items():
    lay = json.load(open(f"{RUNS}/{RUNLIST[lab]}/inputs/layout_manifest.json"))
    cb = lay[key]["collision_bounds"]; T = (np.asarray(cb["lower_xyz_m"]), np.asarray(cb["upper_xyz_m"]))
    tbl = lay[key].get("table_height_m")
    print(f"== {lab} truth {key} lo={np.round(T[0],3).tolist()} hi={np.round(T[1],3).tolist()} table={tbl}")
    for c in calls(lab):
        m = c["map"]
        if not m: continue
        boxes = [(np.asarray(b["lower_xyz_m"]), np.asarray(b["upper_xyz_m"])) for b in m["avoid_boxes"]]
        if not boxes:
            print(f"   step {c['step']}: empty map"); continue
        ious = [iou(b, T) for b in boxes]
        # best = box with max IoU, else nearest centre
        tc = (T[0]+T[1])/2
        k = int(np.argmax(ious)) if max(ious) > 0 else int(np.argmin([np.linalg.norm((b[0]+b[1])/2 - tc) for b in boxes]))
        b = boxes[k]; bc = (b[0]+b[1])/2
        need = max(0.0, float(np.max(np.r_[b[0] - T[0], T[1] - b[1]])))
        cl = m["clearance_m"]
        contains = bool(np.all(b[0] - cl <= T[0]) and np.all(b[1] + cl >= T[1]))
        bottoms = sorted(set(round(float(x[0][2]),3) for x in boxes))
        print(f"   step {c['step']:4d} nbox={len(boxes)} clr={cl:.3f} IoU={ious[k]:.3f} centre_err={np.round(bc - tc,3).tolist()} extra_infl_needed={need:.3f} expanded_contains={contains} box_bottoms_z={bottoms}")
