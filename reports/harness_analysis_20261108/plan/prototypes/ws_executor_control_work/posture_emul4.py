"""Variant (i): retract the grip site toward the shoulder (joint-2 origin) with orientation hold, in the
same DLS-as-OSC emulation; report J4 margin and whether the local IK to the logged target then succeeds."""
import json, glob, sys, math, importlib.util
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
def shoulder(b, q):
    T = b.world_from_base.copy()
    for i in range(2):
        T = T @ b.fixed[i]
        if i == 1: return T[:3, 3] + T[:3, :3] @ b.anchors[i]
        R = mod.rotation_matrix(b.axes[i] * q[i]); Jm = np.eye(4); Jm[:3, :3] = R; Jm[:3, 3] = b.anchors[i] - R @ b.anchors[i]; T = T @ Jm
run = "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001"
W = np.r_[np.ones(3), np.full(3, 0.25)]
for tag in ("000016", "000017", "000021", "000022", "000023"):
    rp = glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/{tag}-*.request.json")[0]
    req = json.load(open(rp)); inp = req.get("payload", req).get("input")
    text = next((c.get("text") for c in inp[-1]["content"] if c.get("type") == "input_text"), None)
    ms = json.loads(text[text.index("{"):])["context"]["measured_state"]
    res = json.load(open(rp.replace(".request.json", ".response.json")))
    otext = [c.get("text") for item in res.get("output", []) for c in (item.get("content") or []) if c.get("type") == "output_text"][-1]
    seg = json.loads(otext)["execution"]["segments"][0]
    pose, q = ms["pose_wxyz"], np.asarray(ms["joint_position_rad"])
    b = kin.bind(list(pose[:3]) + rotvec(pose[3:7]) + [0.04, -0.04], q)
    T0, _ = b.forward(q); R_hold = T0[:3, :3].copy()
    target = np.asarray(seg["xyz_m"]); R_t = mod.quaternion_matrix(seg["quaternion_wxyz"] or pose[3:7])
    m0 = np.minimum(q - kin.low, kin.high - q)
    done = None; travel = 0.0
    for step in range(1, 81):
        T, J = b.forward(q)
        d = shoulder(b, q) - T[:3, 3]; d /= np.linalg.norm(d)
        e = np.r_[d * 0.005 * 0.254, mod.rotation_vector(R_hold @ T[:3, :3].T) * 0.27]
        A = W[:, None] * J
        dq = A.T @ np.linalg.solve(A @ A.T + (0.008 ** 2) * np.eye(6), W * e)
        q = np.clip(q + dq, kin.low + 1e-7, kin.high - 1e-7); travel += 0.005 * 0.254
        if step % 4 == 0:
            r = b.solve(target, R_t, q)
            mm = np.minimum(np.asarray(r.joints) - kin.low, kin.high - np.asarray(r.joints)).min()
            if r.ok and mm > 1e-4: done = step; break
    m1 = np.minimum(q - kin.low, kin.high - q)
    print(f"{tag}: J4 margin {m0[3]:.3f} -> {m1[3]:.3f} rad; min margin {m0.min():.3f} -> {m1.min():.3f}; local IK feasible after {done} steps; retract {travel*1000:.0f} mm")
