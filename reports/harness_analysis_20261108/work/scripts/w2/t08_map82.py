import json, sys, os, numpy as np, pickle
os.environ["GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M"] = "0.10"
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/pkg")
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
from gpt_field.agent_speedup_v2_user_check_routing import TransitRoute, RouteUnavailable
from calls import calls
from extract import RUNLIST, RUNS
C = {c["step"]: c for c in calls("T08_t100")}
m82 = C[82]["map"]; m0 = C[0]["map"]
start = np.array([-0.022911317485817994, -0.0231726197633934, 1.07655246721481]); goal = np.array([-0.025, -0.023, 1.035])
q = [0.0, 0.7071067811865475, 0.7071067811865475, 0.0]
for name, m in (("map0", m0), ("map82", m82)):
    try:
        tr = TransitRoute(start, goal, q, m, max_command_m=.1, position_tolerance_m=.004)
        print(name, "accepted straight=", tr.diagnostics()["straight_path_valid"], "path", np.round(tr.path,3).tolist()[:4])
    except RouteUnavailable as e:
        print(name, "refused:", e.reason)
    print("   boxes", [[np.round(b["lower_xyz_m"],3).tolist(), np.round(b["upper_xyz_m"],3).tolist()] for b in m["avoid_boxes"]], "clr", m["clearance_m"])
# camera ray direction vs centre-error vector
for lab, key in (("T08_t100","bottle"),("T08_stoveoff","bottle"),("T04_t100","can"),("T04_20mm","can")):
    req = sorted(__import__("glob").glob(f"{RUNS}/{RUNLIST[lab]}/trial/requests/000002-*.request.json"))[0]
    d = json.load(open(req))
    txt = [c["text"] for c in d["payload"]["input"][0]["content"] if c.get("type") == "input_text"][0]
    g = json.loads(txt)["geometry"]["cameras"]["external"]
    cam = np.asarray(g["position_world_m"])
    lay = json.load(open(f"{RUNS}/{RUNLIST[lab]}/inputs/layout_manifest.json"))[key]["collision_bounds"]
    tc = (np.asarray(lay["lower_xyz_m"]) + np.asarray(lay["upper_xyz_m"])) / 2
    ray = (tc - cam) / np.linalg.norm(tc - cam)
    m = calls(lab)[0]["map"]
    boxes = [(np.asarray(b["lower_xyz_m"]), np.asarray(b["upper_xyz_m"])) for b in m["avoid_boxes"]]
    bc = min(((b[0]+b[1])/2 for b in boxes), key=lambda c: np.linalg.norm(c - tc))
    err = bc - tc
    along = float(err @ ray); perp = float(np.linalg.norm(err - along * ray))
    print(f"{lab}: cam={np.round(cam,3).tolist()} ray={np.round(ray,3).tolist()} err={np.round(err,3).tolist()} |err|={np.linalg.norm(err):.3f} along_ray={along:+.3f} perpendicular={perp:.3f}")
tr = TransitRoute(start, goal, q, m82, max_command_m=.1, position_tolerance_m=.004)
print("map82 full A* path:", np.round(tr.path, 4).tolist())
print("max lateral deviation from straight line (m):", float(max(np.linalg.norm(np.cross(p - start, goal - start)) / np.linalg.norm(goal - start) for p in tr.path)))
