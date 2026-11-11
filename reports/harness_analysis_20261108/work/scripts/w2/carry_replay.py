import json, sys, os, numpy as np
os.environ["GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M"] = "0.10"
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/pkg")
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
from gpt_field.agent_speedup_v2_user_check_routing import TransitRoute, RouteUnavailable
from calls import calls
from extract import RUNLIST, RUNS
C = {c["step"]: c for c in calls("T04_t100")}
lay = json.load(open(f"{RUNS}/{RUNLIST['T04_t100']}/inputs/layout_manifest.json"))["can"]["collision_bounds"]
TRUE = {"lower_xyz_m": lay["lower_xyz_m"], "upper_xyz_m": lay["upper_xyz_m"]}
q = [0, -0.7071067811865475, -0.7071067811865475, 0]
for name, m in (("model map@163 (clr .04)", C[163]["map"]), ("model map@347 (clr .03)", C[347]["map"]),
                ("TRUE can only, clr .04", {"avoid_boxes": [TRUE], "clearance_m": .04})):
    acc = []
    for z in np.arange(.50, .665, .01):
        a = np.array([-0.09, -0.102, z]); b = np.array([-0.055, -0.272, z])
        try:
            tr = TransitRoute(a, b, q, m, max_command_m=.1, position_tolerance_m=.004)
            acc.append((round(float(z), 2), "S" if tr.diagnostics()["straight_path_valid"] else "A*"))
        except RouteUnavailable as e:
            acc.append((round(float(z), 2), e.reason.replace("route_", "")[:10]))
    print(name, acc)
print("can top", lay["upper_xyz_m"][2], "| model can box", [b for b in C[163]["map"]["avoid_boxes"]][:1])
