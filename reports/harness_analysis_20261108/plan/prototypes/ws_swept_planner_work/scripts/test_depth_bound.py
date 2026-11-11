"""Palm-footprint depth bound vs actual T04 grasps (H1 t100, H1 20mm, H1 default, H3), diagnostic truth only.

At every close onset (gripper command -1 -> +1) take the measured grip pose. Using the exact hand hull
(prototype npz, hull vertices of hand_collision) compute:
  - whether the palm (hand hull) overlaps the truth AABB of each mug at that pose,
  - the hand-hull lowest point relative to each mug's top under the hull footprint,
  - the AABB-based bound: the highest grip z at which the hand hull just touches the mug AABB moving
    down along the approach axis (top-down approximation), and the commanded/actual grip z.
Offline, read-only.
"""
import json, pickle, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from body_runtime import BodyModel  # noqa

DATA = Path("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data")
truth = json.load(open(DATA.parent / "truth_boxes.json"))
bm = BodyModel(HERE.parent / "data/panda_body_a12_g5")
hand_part = next(p for p in bm.parts if p["name"] == "hand_collision")
finger_parts = [p for p in bm.parts if p["kind"] in ("finger", "pad")]
res = {}
for label, pkl in (("H1_T04_t100", "T04_t100_steps.pkl"), ("H1_T04_20mm", "T04_20mm_steps.pkl"),
                   ("H1_T04_default", "T04_default_steps.pkl"), ("H3_T04", "H3_T04_steps.pkl")):
    steps = pickle.load(open(DATA / pkl, "rb"))
    rows = []
    prev = -1.0
    for i, s in sorted(steps.items()):
        g = s.get("gripper_command")
        if g is None:
            g = s["action"][6]
        if prev < 0 and g > 0:  # close onset
            st = np.asarray(s["state"], float)
            hand = bm.hand_from_state(st[None])
            poses = bm.gripper_poses(hand, st[None, 6:8])
            T = poses[hand_part["body"]][0]
            hv = hand_part["hv"] @ T[:3, :3].T + T[:3, 3]  # exact hull vertices in world
            approach = hand[0, :3, 2]
            row = {"step": i, "grip_xyz": st[:3].round(4).tolist(), "approach_z": round(float(approach[2]), 3),
                   "hand_hull_min_z_minus_grip_mm": round(1e3 * float(hv[:, 2].min() - st[2]), 1)}
            for name in ("porcelain_mug_1", "white_yellow_mug_1"):
                lo, hi = np.asarray(truth[name]["lower"]), np.asarray(truth[name]["upper"])
                # xy overlap of the hull's footprint with the mug AABB (top-down approximation)
                xy_in = (hv[:, 0] >= lo[0]) & (hv[:, 0] <= hi[0]) & (hv[:, 1] >= lo[1]) & (hv[:, 1] <= hi[1])
                fx = (hv[:, 0].min() <= hi[0]) and (hv[:, 0].max() >= lo[0]) and (hv[:, 1].min() <= hi[1]) and (hv[:, 1].max() >= lo[1])
                grip_below_top_mm = 1e3 * (hi[2] - st[2])
                row[name] = {"grip_below_mug_top_mm": round(float(grip_below_top_mm), 1),
                             "hull_footprint_overlaps_aabb": bool(fx),
                             "hull_vertices_over_aabb": int(xy_in.sum()),
                             "hull_min_z_minus_top_mm": round(1e3 * float(hv[:, 2].min() - hi[2]), 1)}
            rows.append(row)
        prev = g
    res[label] = rows
print(json.dumps(res, indent=1))
