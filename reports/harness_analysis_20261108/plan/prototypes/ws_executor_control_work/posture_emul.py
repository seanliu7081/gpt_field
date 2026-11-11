"""Kinematic emulation of the proposed posture-recovery phase for drawer-001 J4-limit states.
OSC is emulated as a damped-least-squares joint update with limit clamping (no dynamics).
Command: translation along J_v[:, j] * sign (away from the limit), orientation held, 5 mm/step
commanded offset realised at rho=0.254. Stop when local IK to the logged target succeeds."""
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
run = "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001"
rho = 0.254
for tag in ("000017", "000021", "000022", "000023"):
    rp = glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/{tag}-*.request.json")[0]
    req = json.load(open(rp)); inp = req.get("payload", req).get("input")
    text = next((c.get("text") for c in inp[-1]["content"] if c.get("type") == "input_text"), None)
    fields = json.loads(text[text.index("{"):]); ms = fields["context"]["measured_state"]
    res = json.load(open(rp.replace(".request.json", ".response.json")))
    otext = [c.get("text") for item in res.get("output", []) for c in (item.get("content") or []) if c.get("type") == "output_text"][-1]
    seg = json.loads(otext)["execution"]["segments"][0]
    pose, q = ms["pose_wxyz"], np.asarray(ms["joint_position_rad"])
    b = kin.bind(list(pose[:3]) + rotvec(pose[3:7]) + [0.04, -0.04], q)
    target = np.asarray(seg["xyz_m"]); R_t = mod.quaternion_matrix(seg["quaternion_wxyz"] or pose[3:7])
    T0, _ = b.forward(q); R_hold = T0[:3, :3].copy()
    margins = np.minimum(q - kin.low, kin.high - q); j = int(np.argmin(margins)); sign = 1.0 if (q[j] - kin.low[j]) < (kin.high[j] - q[j]) else -1.0
    ok_step = None; travelled = 0.0
    for step in range(1, 121):
        T, J = b.forward(q)
        v = J[:3, j] * sign
        dx = v / max(np.linalg.norm(v), 1e-9) * 0.005 * rho              # realised Cartesian step
        dr = mod.rotation_vector(R_hold @ T[:3, :3].T) * 0.27              # orientation hold (realised fraction)
        e = np.r_[dx, dr]; W = np.r_[np.ones(3), np.full(3, 0.25)]
        A = W[:, None] * J
        dq = A.T @ np.linalg.solve(A @ A.T + (0.008 ** 2) * np.eye(6), W * e)
        q = np.clip(q + dq, kin.low + 1e-7, kin.high - 1e-7); travelled += np.linalg.norm(dx)
        if step % 5 == 0:
            r = b.solve(target, R_t, q)
            m = np.minimum(np.asarray(r.joints) - kin.low, kin.high - np.asarray(r.joints)).min()
            if r.ok and m > 1e-4:
                ok_step = step; break
    print(f"{tag}: limiting joint J{j+1} margin {margins[j]:.4f} -> {np.minimum(q-kin.low, kin.high-q)[j]:.3f} rad; local IK feasible after {ok_step} steps; grip travel {travelled*1000:.0f} mm")
