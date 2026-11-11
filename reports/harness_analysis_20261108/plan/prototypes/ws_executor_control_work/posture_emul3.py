"""Variant: follow the Cartesian image FK(q(l)) of a joint-space interpolation q(l) from the measured
posture to a global IK solution of the target; OSC emulated as per-step DLS on the Cartesian error with
limit clamping. Compare with following the straight Cartesian line (what the OSC stage does today)."""
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
def follow(b, q, poses, steps_per_pose=1, settle=40, q_null=None, k_null=0.0):
    W = np.r_[np.ones(3), np.full(3, 0.25)]
    for T_ref in poses + [poses[-1]] * settle:
        T, J = b.forward(q)
        e = np.r_[T_ref[:3, 3] - T[:3, 3], mod.rotation_vector(T_ref[:3, :3] @ T[:3, :3].T)]
        # OSC-like: per-step realised fraction of the (capped) error
        n = np.linalg.norm(e[:3]); 
        if n > 0.10: e[:3] *= 0.10 / n
        e *= 0.254
        A = W[:, None] * J
        dq = A.T @ np.linalg.solve(A @ A.T + (0.008 ** 2) * np.eye(6), W * e)
        if q_null is not None:
            Jp = np.linalg.pinv(J)
            dq = dq + (np.eye(7) - Jp @ J) @ (k_null * (q_null - q))
        q = np.clip(q + dq, kin.low + 1e-7, kin.high - 1e-7)
    T, _ = b.forward(q)
    return q, float(np.linalg.norm(T[:3, 3] - poses[-1][:3, 3])), math.degrees(np.linalg.norm(mod.rotation_vector(poses[-1][:3, :3] @ T[:3, :3].T)))
run = "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001"
rng = np.random.default_rng(0)
_r0 = sorted(glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/*.request.json"))[0]
_q = json.load(open(_r0)); _i = _q.get("payload", _q).get("input"); _t = next((c.get("text") for c in _i[-1]["content"] if c.get("type") == "input_text"), None)
q_reset = np.asarray(json.loads(_t[_t.index("{"):])["context"]["measured_state"]["joint_position_rad"])
for tag in ("000016", "000017", "000021", "000022", "000023"):
    rp = glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/{tag}-*.request.json")[0]
    req = json.load(open(rp)); inp = req.get("payload", req).get("input")
    text = next((c.get("text") for c in inp[-1]["content"] if c.get("type") == "input_text"), None)
    fields = json.loads(text[text.index("{"):]); ms = fields["context"]["measured_state"]
    res = json.load(open(rp.replace(".request.json", ".response.json")))
    otext = [c.get("text") for item in res.get("output", []) for c in (item.get("content") or []) if c.get("type") == "output_text"][-1]
    seg = json.loads(otext)["execution"]["segments"][0]
    pose, q0 = ms["pose_wxyz"], np.asarray(ms["joint_position_rad"])
    b = kin.bind(list(pose[:3]) + rotvec(pose[3:7]) + [0.04, -0.04], q0)
    T_goal = np.eye(4); T_goal[:3, :3] = mod.quaternion_matrix(seg["quaternion_wxyz"] or pose[3:7]); T_goal[:3, 3] = seg["xyz_m"]
    T0, _ = b.forward(q0)
    # straight Cartesian line (current behaviour, orientation slerp approximated by linear rotvec blend)
    line = []
    dr = mod.rotation_vector(T_goal[:3, :3] @ T0[:3, :3].T)
    for l in np.linspace(0, 1, 40)[1:]:
        T = np.eye(4); T[:3, 3] = (1 - l) * T0[:3, 3] + l * T_goal[:3, 3]; T[:3, :3] = mod.rotation_matrix(dr * l) @ T0[:3, :3]; line.append(T)
    _, e_line, o_line = follow(b, q0.copy(), line)
    _, e_ln, o_ln = follow(b, q0.copy(), line, q_null=q_reset, k_null=0.05)
    # global IK solution (multi-seed), joint interpolation path
    best = None
    for s in [q0] + [np.clip(q0 + rng.normal(0, .4, 7), kin.low + .05, kin.high - .05) for _ in range(8)]:
        r = b.solve(T_goal[:3, 3], T_goal[:3, :3], s, max_iterations=300)
        m = np.minimum(np.asarray(r.joints) - kin.low, kin.high - np.asarray(r.joints)).min()
        if r.ok and (best is None or m > best[1]): best = (np.asarray(r.joints), m)
    qg = best[0]
    jpath = [b.forward((1 - l) * q0 + l * qg)[0] for l in np.linspace(0, 1, 40)[1:]]
    _, e_j, o_j = follow(b, q0.copy(), jpath)
    _, e_jn, o_jn = follow(b, q0.copy(), jpath, q_null=qg, k_null=0.05)
    print(f"{tag}: +nullspace(reset) line {e_ln*1000:.1f} mm/{o_ln:.1f} deg; +nullspace(goal q) joint path {e_jn*1000:.1f} mm/{o_jn:.1f} deg")
    print(f"{tag}: straight-line final error {e_line*1000:.1f} mm/{o_line:.1f} deg; joint-interpolated path final error {e_j*1000:.1f} mm/{o_j:.1f} deg (global solution margin {best[1]:.2f} rad)")
