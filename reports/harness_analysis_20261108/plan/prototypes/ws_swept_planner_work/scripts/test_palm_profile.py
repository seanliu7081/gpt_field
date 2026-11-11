"""Palm profile p(xi) from the built hand hull vs robot_facts_v2.json (offline, numpy only)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from body_runtime import BodyModel
bm = BodyModel(HERE.parent / "data/panda_body_a12_g5")
hand = next(p for p in bm.parts if p["name"] == "hand_collision")
body = bm.bodies[hand["body"]]                 # right_gripper; its T is relative to right_hand
T = np.asarray(body["T"])                      # right_hand <- right_gripper
E = hand["eq"].copy()                          # planes in right_gripper coords: n.x + d <= 0
n = E[:, :3] @ T[:3, :3].T                     # normals in right_hand coords
d = E[:, 3] - n @ T[:3, 3]
facts = json.load(open("/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/src/gpt_field/robot_facts_v2.json"))
prof = np.asarray(facts["asset"]["palm_profile_mm"]["xi_R1_mm_and_p_mm"])
xs = np.linspace(-0.05, 0.05, 2001)
errs = []
for xi_mm, p_mm in prof:
    xi = xi_mm / 1000
    best = -np.inf
    # z upper bound per x from planes with n_z > 0; feasibility from the rest
    up = n[:, 2] > 1e-9; lo = n[:, 2] < -1e-9; fl = ~(up | lo)
    rhs = -d[None, :] - n[None, :, 0] * xs[:, None] - n[None, :, 1] * xi
    zmax = np.min(np.where(up[None], rhs / np.where(up, n[:, 2], 1)[None], np.inf), axis=1)
    zmin = np.max(np.where(lo[None], rhs / np.where(lo, n[:, 2], 1)[None], -np.inf), axis=1)
    ok = (zmax >= zmin) & np.all(np.where(fl[None], rhs >= 0, True), axis=1)
    if ok.any():
        best = zmax[ok].max()
    p_model = (0.097 - best) * 1000
    errs.append((xi_mm, p_mm, p_model, p_model - p_mm))
e = np.asarray(errs)
print(json.dumps({"points": len(e), "max_abs_err_mm": round(float(np.max(np.abs(e[:, 3]))), 3),
                  "median_abs_err_mm": round(float(np.median(np.abs(e[:, 3]))), 3),
                  "p_at_xi0_model_mm": round(float(e[np.argmin(np.abs(e[:, 0])), 2]), 3),
                  "p_at_xi0_facts_mm": float(e[np.argmin(np.abs(e[:, 0])), 1]),
                  "worst": [[round(v, 2) for v in row] for row in e[np.argsort(-np.abs(e[:, 3]))[:3]].tolist()]}, indent=1))
