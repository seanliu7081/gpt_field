import pickle, sys, numpy as np, os
os.environ["GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M"] = "0.10"
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/pkg")
from gpt_field.agent_speedup_v2_user_check_routing import TransitRoute, RouteUnavailable
from gpt_field.fine_distance_planner import FineCompactField
OUT = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
rows = pickle.load(open(f"{OUT}/route_rows.pkl", "rb"))
def box_dist(p, lo, hi):
    lo, hi, p = map(np.asarray, (lo, hi, p))
    d = np.maximum(np.maximum(lo - p, 0), p - hi)
    inside = np.all((p >= lo) & (p <= hi))
    if inside:
        return -float(np.min(np.minimum(p - lo, hi - p)))
    return float(np.linalg.norm(d))
for x in rows:
    if x["status"] == "ok": continue
    s = x["seg"]; ro = x["routing"]
    start = np.asarray(s["entry_pose_wxyz"][:3]); goal = np.asarray(s["target_xyz_m"])
    dists = [box_dist(start, b["lower_xyz_m"], b["upper_xyz_m"]) for b in ro["expanded_boxes"]]
    gd = [box_dist(goal, b["lower_xyz_m"], b["upper_xyz_m"]) for b in ro["expanded_boxes"]]
    # replay with recorded map: reconstruct raw boxes = expanded - clearance
    c = ro["clearance_m"]
    raw = [{"lower_xyz_m": (np.asarray(b["lower_xyz_m"]) + c).tolist(), "upper_xyz_m": (np.asarray(b["upper_xyz_m"]) - c).tolist()} for b in ro["expanded_boxes"]]
    try:
        TransitRoute(start, goal, s["target_quaternion_wxyz"], {"avoid_boxes": raw, "clearance_m": c}, max_command_m=.1, position_tolerance_m=.004)
        rep = "accepted"
    except RouteUnavailable as e:
        rep = e.reason
    # with exact point test only (no 8-corner): does start lie inside any expanded box?
    print(f"{x['run']:12s} entry~{x['global_entry']:5d} clear={c:.3f} start={np.round(start,4).tolist()} goal={np.round(goal,4).tolist()}")
    print(f"      start signed dist to expanded boxes (neg=inside): {[round(v,4) for v in dists]}  goal: {[round(v,4) for v in gd]}  replay={rep}")
