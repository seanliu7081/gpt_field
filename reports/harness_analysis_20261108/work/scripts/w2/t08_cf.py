"""T08 t100 colliding stage (global steps 35-81): point check vs hand/finger sweep, true bottle box. Diagnostic only."""
import json, math, pickle, sys, os
import numpy as np
os.environ["GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M"] = "0.10"
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/pkg")
from gpt_field.agent_speedup_v2_user_check_routing import TransitRoute, RouteUnavailable
from gpt_field.astranomy_agentic_motion import measured_pose
OUT = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
RUN = "/workspace/gpt_field/runs/agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001"
lay = json.load(open(f"{RUN}/inputs/layout_manifest.json"))["bottle"]["collision_bounds"]
TLO, THI = np.asarray(lay["lower_xyz_m"]), np.asarray(lay["upper_xyz_m"])
S = pickle.load(open(f"{OUT}/T08_t100_steps.pkl", "rb"))
def Rq(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],[2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],[2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
def obb_aabb(c, Rm, half, lo, hi):
    bc, bh = (lo+hi)/2, (hi-lo)/2
    A = [Rm[:, i] for i in range(3)]; B = list(np.eye(3))
    d = bc - c
    for ax in A + B + [np.cross(a, b) for a in A for b in B]:
        n = np.linalg.norm(ax)
        if n < 1e-9: continue
        ax = ax / n
        if abs(d @ ax) > sum(half[i]*abs(A[i] @ ax) for i in range(3)) + sum(bh[i]*abs(B[i] @ ax) for i in range(3)): return False
    return True
# Hand-frame parts relative to grip site (R0, R1, R2). From robot_facts_v2.json (open fingers ~ q 38.7 mm).
PALM = (np.array([-.0316, -.104, -.097]), np.array([.0316, .1004, -.031]))
F1 = (np.array([-.0062, .0382, -.039]), np.array([.0062, .065, .00925]))   # finger1 body+pad, open
F2 = (np.array([-.0062, -.065, -.039]), np.array([.0062, -.0382, .00925]))
PARTS = {"palm": PALM, "finger1": F1, "finger2": F2}
def hits(p, q, lo, hi):
    Rm = Rq(q); out = []
    for name, (a, b) in PARTS.items():
        if obb_aabb(p + Rm @ ((a+b)/2), Rm, (b-a)/2, lo, hi): out.append(name)
    return out
# measured poses for steps 35..82
steps = list(range(33, 84))
print("contact first event pos (0.00999,-0.06302,1.04557)")
for s in steps:
    st = S[s]
    pose = measured_pose(np.asarray(st["after_state"]))
    p, q = np.asarray(pose[:3]), np.asarray(pose[3:7])
    Rm = Rq(q)
    jaw = Rm[:, 1]
    h0 = hits(p, q, TLO, THI); h18 = hits(p, q, TLO - .018, THI + .018)
    f = st.get("frag") or {}
    if s in (33,34,35,36,37,38,40,45,50,55,60,65,70,75,80,81,82,83):
        print(f"step {s:3d} seg={f.get('seg_index')} p={np.round(p,4).tolist()} jaw_xy={np.round(jaw[:2],3).tolist()} forb_inc={st['forb_inc']:4d} sweep_true={h0} sweep_true+0.018={h18}")
# contact point in hand frame at step 37
st = S[37]; pose = measured_pose(np.asarray(st["before_state"])); p, q = np.asarray(pose[:3]), np.asarray(pose[3:7])
cp = np.array([0.009986071078027254, -0.06301504455717793, 1.0455656780842273])
print("contact point in hand frame (R0,R1,R2) relative to grip site at step-37 entry:", np.round(Rq(q).T @ (cp - p), 4).tolist())
# Point-router replay with TRUE box + 0.018 at stage entry (segment 1, global ~35) and at each measured pose (tracking check)
frag_seg = None
for s in range(35, 82):
    f = S[s].get("frag")
    if f and f.get("seg_index") == 1 and f.get("routing"):
        frag_seg = f; break
ro = frag_seg["routing"]
start = np.asarray(ro["path_xyz_m"][0]); goal = np.asarray(ro["path_xyz_m"][-1])
print("stage path", np.round(start,4).tolist(), "->", np.round(goal,4).tolist(), "target q", frag_seg["target_q"], "clearance", ro["clearance_m"])
route = {"avoid_boxes": [{"lower_xyz_m": TLO.tolist(), "upper_xyz_m": THI.tolist()}], "clearance_m": ro["clearance_m"]}
try:
    tr = TransitRoute(start, goal, frag_seg["target_q"], route, max_command_m=.1, position_tolerance_m=.004)
    print("TRUE box + 0.018 point router at entry:", tr.diagnostics()["straight_path_valid"], tr.diagnostics()["planner"])
    bad = []
    for s in range(35, 82):
        pose = measured_pose(np.asarray(S[s]["after_state"]))
        if tr.field.query(pose[:3]) is None: bad.append(s)
    print("  measured grip-site tracking-check failures (8-corner) during stage:", bad)
except RouteUnavailable as e:
    print("TRUE box point router refuses:", e.reason)
# Also the earlier segment 0 (entry at global 1) -> also with the true box
f0 = S[2]["frag"]; ro0 = f0["routing"]
try:
    tr0 = TransitRoute(np.asarray(ro0["path_xyz_m"][0]), np.asarray(ro0["path_xyz_m"][-1]), f0["target_q"], {"avoid_boxes": route["avoid_boxes"], "clearance_m": ro0["clearance_m"]}, max_command_m=.1, position_tolerance_m=.004)
    print("segment 0 with TRUE box:", tr0.diagnostics()["straight_path_valid"])
except RouteUnavailable as e:
    print("segment 0 TRUE box refuses:", e.reason)
# Planned linear+slerp sweep for segment 1
def slerp(a, b, t):
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b); d = float(a @ b)
    if d < 0: b, d = -b, -d
    if d > .9995: q = a + t*(b-a); return q/np.linalg.norm(q)
    th = math.acos(d); return (math.sin((1-t)*th)*a + math.sin(t*th)*b)/math.sin(th)
q0 = np.asarray(measured_pose(np.asarray(S[34]["after_state"]))[3:7]); q1 = np.asarray(frag_seg["target_q"])
print("planned linear+slerp sweep vs TRUE box (+0, +0.018):")
for t in np.linspace(0, 1, 11):
    p = start + t*(goal-start); q = slerp(q0, q1, t)
    print(f"   t={t:.1f} p={np.round(p,4).tolist()} {hits(p, q, TLO, THI)} {hits(p, q, TLO-.018, THI+.018)}")
print("==== A* detour with TRUE box + 0.018")
path = tr.path
print("path vertices", len(path)); print(np.round(path, 4).tolist())
# sweep along detour: assume translation completes first (as measured), orientation slerps over same fraction (two variants)
L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
def at(u):
    s = u * L[-1]; i = min(np.searchsorted(L, s, side="right") - 1, len(path) - 2)
    f = (s - L[i]) / max(L[i+1] - L[i], 1e-12); return path[i] + f * (path[i+1] - path[i])
for mode in ("synchronized", "rotation_lags"):
    flagged = []
    for u in np.linspace(0, 1, 41):
        p = at(u)
        t = u if mode == "synchronized" else min(1., u * 0.15)   # measured: rotation ~15% done when translation done
        q = slerp(q0, q1, t)
        h = hits(p, q, TLO, THI)
        if h: flagged.append((round(float(u),3), h))
    print(mode, "flags vs TRUE box (no clearance):", flagged[:6], "...", len(flagged))
# what about final pose rotating in place at the goal (rotation completes after translation)?
for t in np.linspace(0, 1, 11):
    print(f"  at goal, rotation t={t:.1f}: {hits(goal, slerp(q0, q1, t), TLO, THI)}")
