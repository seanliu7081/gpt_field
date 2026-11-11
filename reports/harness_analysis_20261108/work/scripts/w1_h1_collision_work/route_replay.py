"""Offline: replay TransitRoute on recorded start/goal/map to test hidden lattice inflation (no simulator)."""
import json, sys
import numpy as np
sys.path.insert(0, "/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/src")
from gpt_field.agent_speedup_v2_user_check_routing import TransitRoute, RouteUnavailable

def box_dist(p, lo, hi):
    p, lo, hi = map(np.asarray, (p, lo, hi))
    d = np.maximum(np.maximum(lo - p, 0), p - hi)
    return float(np.linalg.norm(d)), d

def run(start, goal, boxes, clearance, q=(0, -0.7071, -0.7071, 0), label=""):
    route = {"avoid_boxes": [{"lower_xyz_m": b[0], "upper_xyz_m": b[1]} for b in boxes], "clearance_m": clearance}
    dists = []
    for b in boxes:
        lo = np.asarray(b[0]) - clearance; hi = np.asarray(b[1]) + clearance
        dists.append(round(box_dist(start, lo, hi)[0], 4))
    try:
        r = TransitRoute(start, goal, q, route, max_command_m=0.1, position_tolerance_m=0.004)
        res = ("ok", r.diagnostics()["straight_path_valid"], r.diagnostics()["planner"]["calls"])
    except RouteUnavailable as e:
        res = ("unavailable", e.reason)
    print(label, "clearance", clearance, "start->expanded-box distances", dists, "=>", res)
    return res

# T04 transit100 step 162 (recorded route_start_blocked)
B = [([-0.29, -0.04, 0.4], [-0.17, 0.085, 0.57]), ([-0.04, 0.035, 0.4], [0.09, 0.16, 0.53]), ([-0.015, -0.25, 0.4], [0.065, -0.18, 0.5])]
s, g = [-0.08981604955758658, -0.10206545781136724, 0.5272483458135655], [-0.09, -0.102, 0.64]
for c in (0.06, 0.05, 0.045, 0.04, 0.03):
    run(s, g, B, c, label="T04t100@162")
# T08 transit100 step 87 (recorded route_start_blocked)
B8 = [([-0.13, -0.36, 0.89], [0.12, -0.12, 0.95]), ([-0.015, -0.12, 0.89], [0.08, -0.05, 1.055]), ([-0.14, 0.18, 0.89], [0.0, 0.32, 1.08])]
run([-0.007074487542127049, -0.025715402773685593, 1.062958341224202], [0.06, -0.025, 1.105], B8, 0.018, label="T08t100@87")
# T08 transit100 step 597 (recorded route_start_blocked)
B8b = [([-0.14, -0.31, 0.89], [0.08, -0.12, 0.95]), ([-0.02, -0.26, 0.94], [0.075, -0.16, 1.11]), ([-0.015, -0.12, 0.89], [0.08, -0.055, 1.055])]
run([0.002500335847596744, -0.2676234674588176, 1.1413189430737851], [-0.095, 0.18, 1.15], B8b, 0.018, label="T08t100@597")
# Same T04 geometry: does a lowered straight carry trigger A* at all? (counterfactual low carry at z=0.56)
for z in (0.64, 0.60, 0.58, 0.56, 0.54):
    run([-0.09, -0.102, z], [-0.055, -0.272, z], B, 0.04, label=f"T04 carry z={z}")
