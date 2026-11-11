"""Offline: would a whole-hand (palm+fingers OBB) swept check against the model's OWN expanded boxes have
flagged the recorded routed transit stages that the point router approved?  Hand envelope from
src/gpt_field/robot_facts_v2.json (R0 +-31.6 mm, R1 -104..+100 mm, palm underside 31 mm behind grip,
fingertips 9.25 mm beyond grip, hand body up to the flange 97 mm behind grip).  Orientation slerped.
Optionally adds a nominal held-object box below the grip when the gripper is closed."""
import json, math, sys
import numpy as np
W = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/h1_collision_work"
RUNS = [
 "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
 "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001",
 "agent_speedup_v2_uc71_default_routing_t04_20261103_001",
 "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
 "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001",
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001",
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003",
]
HELD = len(sys.argv) > 1 and sys.argv[1] == "held"
def R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
def slerp(a, b, t):
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    d = float(np.dot(a, b))
    if d < 0: b, d = -b, -d
    if d > 0.9995: q = a + t*(b-a); return q/np.linalg.norm(q)
    th = math.acos(d); return (math.sin((1-t)*th)*a + math.sin(t*th)*b)/math.sin(th)
# hand-frame boxes relative to grip site: (lo, hi) along (R0, R1, R2=approach)
HAND = [(np.array([-0.0316, -0.104, -0.097]), np.array([0.0316, 0.1004, -0.031])),   # palm/hand body
        (np.array([-0.010, -0.0475, -0.039]), np.array([0.010, 0.0475, 0.00925]))]   # open fingers (outer faces)
HELD_BOX = (np.array([-0.04, -0.04, 0.0]), np.array([0.04, 0.04, 0.07]))           # nominal carried object below grip
def obb_aabb_overlap(c, Rm, half, lo, hi):
    # SAT between OBB (center c, axes columns of Rm, half extents) and AABB(lo,hi)
    bc, bh = (lo+hi)/2, (hi-lo)/2
    A = [Rm[:, i] for i in range(3)]; B = [np.eye(3)[i] for i in range(3)]
    axes = A + B + [np.cross(a, b) for a in A for b in B]
    d = bc - c
    for ax in axes:
        n = np.linalg.norm(ax)
        if n < 1e-9: continue
        ax = ax / n
        ra = sum(half[i]*abs(np.dot(A[i], ax)) for i in range(3))
        rb = sum(bh[i]*abs(np.dot(B[i], ax)) for i in range(3))
        if abs(np.dot(d, ax)) > ra + rb: return False
    return True
tot = {"stages": 0, "flagged": 0, "flagged_endpoints_only": 0}
for run in RUNS:
    rows = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    seen = set(); n = f = fe = 0; flagged_list = []
    for r in rows:
        if r["event"] != "action": continue
        fb = r["diagnostics"].get("feedback") or {}
        if fb.get("action") != "execute": continue
        k = fb.get("observed_at_step")
        if k in seen: continue
        seen.add(k)
        for s in (fb.get("evidence") or {}).get("segment_trace") or []:
            ro = s.get("routing") or {}
            if not ro.get("straight_path_valid") or s.get("actions", 0) == 0: continue
            start = np.asarray(s["entry_pose_wxyz"][:3]); q0 = np.asarray(s["entry_pose_wxyz"][3:7])
            goal = np.asarray(s["target_xyz_m"]); q1 = np.asarray(s["target_quaternion_wxyz"])
            closed = (s.get("entry_gap_mm") or 80) < 60
            boxes = [(np.asarray(b["lower_xyz_m"]), np.asarray(b["upper_xyz_m"])) for b in ro.get("expanded_boxes", [])]
            n += 1; hit_ts = []
            for t in np.linspace(0, 1, 41):
                p = start + t*(goal-start); Rm = R(slerp(q0, q1, t))
                parts = list(HAND) + ([HELD_BOX] if (HELD and closed) else [])
                for lo_h, hi_h in parts:
                    c = p + Rm @ ((lo_h+hi_h)/2); half = (hi_h-lo_h)/2
                    if any(obb_aabb_overlap(c, Rm, half, lo, hi) for lo, hi in boxes):
                        hit_ts.append(round(float(t), 3)); break
            if hit_ts:
                f += 1
                if all(tt <= 0.05 or tt >= 0.95 for tt in hit_ts): fe += 1
                flagged_list.append((k, s["index"], s["skill"], "closed" if closed else "open", hit_ts[0], hit_ts[-1]))
    tot["stages"] += n; tot["flagged"] += f; tot["flagged_endpoints_only"] += fe
    print(run[26:66], "straight-valid routed stages", n, "flagged by hand sweep", f, "(endpoint-only", fe, ")", flagged_list[:8])
print(tot)
