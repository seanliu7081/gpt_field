import pickle, numpy as np, math, collections, sys
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
D = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
src = open("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts/t08_cf.py").read()
ns = {"np": np, "math": math}
exec(src[src.index("def Rq"):src.index("# measured poses")], ns)
Rq, obb_aabb, PARTS = ns["Rq"], ns["obb_aabb"], ns["PARTS"]
ANALYST = {"palm": PARTS["palm"], "fingers_one_box": (np.array([-0.010, -0.0475, -0.039]), np.array([0.010, 0.0475, 0.00925]))}
def slerp(a, b, t):
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b); d = float(a @ b)
    if d < 0: b, d = -b, -d
    if d > .9995: q = a + t*(b-a); return q/np.linalg.norm(q)
    th = math.acos(d); return (math.sin((1-t)*th)*a + math.sin(t*th)*b)/math.sin(th)
rows = pickle.load(open(f"{D}/route_rows.pkl","rb"))
H1 = ["T04_t100","T04_20mm","T04_default","T08_t100","T08_stoveoff","drawer001","drawer003"]
for model_name, parts in (("my 2-finger+palm", PARTS), ("analyst palm+one-box fingers", ANALYST)):
    n = 0; flagged = collections.Counter(); per_run = collections.Counter(); where = collections.Counter()
    for x in rows:
        if x["run"] not in H1 or x["status"] != "ok": continue
        s = x["seg"]
        if not s.get("actions"): continue
        n += 1
        a = np.asarray(s["entry_pose_wxyz"][:3]); b = np.asarray(s["target_xyz_m"])
        q0, q1 = np.asarray(s["entry_pose_wxyz"][3:7]), np.asarray(s["target_quaternion_wxyz"])
        boxes = [(np.asarray(bb["lower_xyz_m"]), np.asarray(bb["upper_xyz_m"])) for bb in x["routing"]["expanded_boxes"]]
        ts = []
        for t in np.linspace(0, 1, 41):
            p = a + t*(b - a); Rm = Rq(slerp(q0, q1, t))
            if any(obb_aabb(p + Rm @ ((lo+hi)/2), Rm, (hi-lo)/2, L, H) for lo, hi in parts.values() for L, H in boxes):
                ts.append(t)
        if ts:
            per_run[x["run"]] += 1
            where["start" if ts[0] == 0 else "goal_only" if ts[0] >= .95 else "mid"] += 1
    print(f"{model_name}: accepted stages with >=1 action {n}; flagged {sum(per_run.values())} by run {dict(per_run)}; first-hit location {dict(where)}")
for x in rows:
    if x["run"] != "T08_t100" or x["status"] != "ok": continue
    s = x["seg"]
    a = np.asarray(s["entry_pose_wxyz"][:3]); b = np.asarray(s["target_xyz_m"])
    q0, q1 = np.asarray(s["entry_pose_wxyz"][3:7]), np.asarray(s["target_quaternion_wxyz"])
    boxes = [(np.asarray(bb["lower_xyz_m"]), np.asarray(bb["upper_xyz_m"])) for bb in x["routing"]["expanded_boxes"]]
    hits = []
    for t in np.linspace(0, 1, 41):
        p = a + t*(b - a); Rm = Rq(slerp(q0, q1, t))
        for nm, (lo, hi) in PARTS.items():
            for bi, (L, H) in enumerate(boxes):
                if obb_aabb(p + Rm @ ((lo+hi)/2), Rm, (hi-lo)/2, L, H): hits.append((round(float(t),3), nm, bi))
    if hits: print("T08_t100 flagged stage: frag_start", x["frag_start"], "seg", x["seg"]["index"], "entry", np.round(a,3).tolist(), "->", np.round(b,3).tolist(), "first/last hits", hits[0], hits[-1], "n", len(hits))
