"""Offline IK pass over logged absolute stage targets (first-segment relative targets resolved from
the request's measured pose). Uses the main-tree robodojo_direct_sim (read-only copy load) and the
pinned robosuite XMLs. Text-only reads of request/response logs; no simulator or model."""
import json, glob, sys, math, importlib.util, time
import numpy as np
SRC = "/workspace/gpt_field/src/gpt_field/robodojo_direct_sim.py"
spec = importlib.util.spec_from_file_location("rdsim_copy", SRC)
mod = importlib.util.module_from_spec(spec); sys.modules["rdsim_copy"] = mod; spec.loader.exec_module(mod)
SIM = "/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/.venv-sim/lib/python3.10/site-packages/robosuite/models/assets/"
kin = mod.PandaKinematics.from_xml(SIM + "robots/panda/robot.xml", SIM + "grippers/panda_gripper.xml")

def rotvec(q):
    w, x, y, z = q
    if w < 0: w, x, y, z = -w, -x, -y, -z
    s = math.sqrt(x*x + y*y + z*z)
    return [0., 0., 0.] if s < 1e-12 else [x / s * 2 * math.atan2(s, w), y / s * 2 * math.atan2(s, w), z / s * 2 * math.atan2(s, w)]

def qmat(q):
    return mod.quaternion_matrix(q)

RUNS = {"T04": "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
        "T08": "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
        "d001": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001",
        "d003": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003"}
times = []
for label, run in RUNS.items():
    reqs = sorted(glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/*.request.json"))
    out = []
    for rp in reqs:
        resp = rp.replace(".request.json", ".response.json")
        try:
            req, res = json.load(open(rp)), json.load(open(resp))
        except Exception:
            continue
        inp = req.get("payload", req).get("input")
        content = inp[-1]["content"]
        text = next((c.get("text") for c in content if c.get("type") == "input_text"), None)
        try:
            fields = json.loads(text[text.index("{"):])
        except Exception:
            continue
        ms = fields.get("context", {}).get("measured_state", {})
        q0, pose = ms.get("joint_position_rad"), ms.get("pose_wxyz")
        if q0 is None: continue
        otext = None
        for item in res.get("output", []):
            for c in item.get("content", []) or []:
                if c.get("type") == "output_text": otext = c.get("text")
        if not otext: continue
        try: dec = json.loads(otext)
        except Exception: continue
        if dec.get("type") != "execute": continue
        b = kin.bind(list(pose[:3]) + rotvec(pose[3:7]) + [0.04, -0.04], q0)
        seed = np.asarray(q0); entry_xyz = np.asarray(pose[:3]); entry_q = pose[3:7]
        for si, seg in enumerate(dec["execution"]["segments"]):
            if seg["skill"] not in ("move_to_pose", "move_relative", "approach"): continue
            if seg["skill"] != "move_to_pose" and si > 0: break   # later relative targets depend on measured entry
            xyz = np.asarray(seg["xyz_m"]) if seg["skill"] == "move_to_pose" else entry_xyz + np.asarray(seg["xyz_m"])
            quat = seg["quaternion_wxyz"] or entry_q
            t0 = time.perf_counter()
            r = b.solve(xyz, qmat(quat), seed)
            times.append(time.perf_counter() - t0)
            qs = np.asarray(r.joints)
            m = np.minimum(qs - kin.low, kin.high - qs)
            out.append((rp.split("/")[-1][:6], si, seg["skill"], r.ok, round(r.position_error_m * 1000, 1), round(r.orientation_error_deg, 2), int(np.argmin(m)) + 1, round(float(m.min()), 3)))
            seed = qs; entry_xyz = xyz; entry_q = quat
            if seg["skill"] != "move_to_pose": break
    bad = [o for o in out if not o[3] or o[7] < 0.05]
    print(label, "targets checked", len(out), "infeasible or margin<0.05:", bad[:20])
print(f"IK solve time median {np.median(times)*1000:.1f} ms p95 {np.percentile(times,95)*1000:.1f} ms (n={len(times)})")

# --- multi-seed re-check of flagged targets (seed set: current q, reset q, current q with J4 flexed, random fixed)
print("multi-seed re-check")
rng = np.random.default_rng(0)
for label in ("d001", "d003"):
    run = RUNS[label]
    reqs = sorted(glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/*.request.json"))
    q_reset = None
    for rp in reqs:
        req = json.load(open(rp)); inp = req.get("payload", req).get("input")
        text = next((c.get("text") for c in inp[-1]["content"] if c.get("type") == "input_text"), None)
        try: fields = json.loads(text[text.index("{"):])
        except Exception: continue
        ms = fields.get("context", {}).get("measured_state", {})
        if ms.get("joint_position_rad") is None: continue
        if q_reset is None: q_reset = np.asarray(ms["joint_position_rad"])
        tag = rp.split("/")[-1][:6]
        if label == "d001" and tag not in ("000016", "000017", "000021", "000022", "000023", "000024"): continue
        if label == "d003" and tag not in ("000004", "000008"): continue
        res = json.load(open(rp.replace(".request.json", ".response.json")))
        otext = [c.get("text") for item in res.get("output", []) for c in (item.get("content") or []) if c.get("type") == "output_text"][-1]
        dec = json.loads(otext); seg = dec["execution"]["segments"][0]
        pose = ms["pose_wxyz"]; q0 = np.asarray(ms["joint_position_rad"])
        b = kin.bind(list(pose[:3]) + rotvec(pose[3:7]) + [0.04, -0.04], q0)
        xyz = np.asarray(seg["xyz_m"]) if seg["skill"] == "move_to_pose" else np.asarray(pose[:3]) + np.asarray(seg["xyz_m"])
        quat = seg["quaternion_wxyz"] or pose[3:7]
        seeds = [q0, q_reset] + [np.clip(q0 + rng.normal(0, 0.4, 7), kin.low + .05, kin.high - .05) for _ in range(6)]
        best = None
        for s in seeds:
            r = b.solve(xyz, qmat(quat), s, max_iterations=300)
            qs = np.asarray(r.joints); m = float(np.min(np.minimum(qs - kin.low, kin.high - qs)))
            key = (not r.ok, r.position_error_m)
            if best is None or key < best[0]: best = (key, r, m)
        _, r, m = best
        reach = float(np.linalg.norm(xyz - b.world_from_base[:3, 3]))
        print(f"  {label} {tag} {seg['skill']} best ok={r.ok} pos_err={r.position_error_m*1000:.1f}mm ori_err={r.orientation_error_deg:.1f}deg min_margin={m:.3f} target_dist_from_base={reach:.3f} m")
