"""Offline check: vendored-style numpy FK bound from (state8, joints) gives a constant base pose.
Reads request logs (text only) and the pinned robosuite robot.xml; imports the main-tree
robodojo_direct_sim module by path copy (read-only) without importing gpt_field."""
import json, glob, sys, math, importlib.util, os
import numpy as np
SRC = "/workspace/gpt_field/src/gpt_field/robodojo_direct_sim.py"
spec = importlib.util.spec_from_file_location("rdsim_copy", SRC)
mod = importlib.util.module_from_spec(spec); sys.modules["rdsim_copy"] = mod; spec.loader.exec_module(mod)
SIM = "/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/.venv-sim/lib/python3.10/site-packages/robosuite/models/assets/"
kin = mod.PandaKinematics.from_xml(SIM + "robots/panda/robot.xml", SIM + "grippers/panda_gripper.xml")
print("limits low", np.round(kin.low, 4).tolist()); print("limits high", np.round(kin.high, 4).tolist())

def rotvec_from_wxyz(q):
    w, x, y, z = q
    if w < 0: w, x, y, z = -w, -x, -y, -z
    s = math.sqrt(x*x + y*y + z*z)
    if s < 1e-12: return [0., 0., 0.]
    ang = 2 * math.atan2(s, w)
    return [x / s * ang, y / s * ang, z / s * ang]

for run in ["agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
            "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001"]:
    bases, margins = [], []
    for path in sorted(glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/*.request.json")):
        req = json.load(open(path))
        payload = req.get("payload", req)
        inp = payload.get("input") if isinstance(payload, dict) else None
        if not inp: continue
        content = inp[-1].get("content")
        text = next((c.get("text") for c in content if c.get("type") == "input_text"), None) if isinstance(content, list) else None
        if not text: continue
        try:
            fields = json.loads(text[text.index("{"):])
        except Exception:
            continue
        ms = fields.get("context", {}).get("measured_state", {})
        q = ms.get("joint_position_rad"); pose = ms.get("pose_wxyz")
        if q is None or pose is None: continue
        state8 = list(pose[:3]) + rotvec_from_wxyz(pose[3:7]) + [0.04, -0.04]
        b = kin.bind(state8, q)
        bases.append(b.world_from_base.copy())
        m = np.minimum(np.asarray(q) - kin.low, kin.high - np.asarray(q))
        margins.append((int(np.argmin(m)) + 1, float(m.min())))
    if not bases:
        print(run, "no parsable requests"); continue
    P = np.asarray([B[:3, 3] for B in bases]); R0 = bases[0][:3, :3]
    angs = [math.degrees(math.acos(min(1, max(-1, (np.trace(R0.T @ B[:3, :3]) - 1) / 2)))) for B in bases]
    print(run[:60], "n", len(bases), "base mean", np.round(P.mean(0), 4).tolist(), "pos spread mm", round(float(np.ptp(P, axis=0).max()) * 1000, 3),
          "rot spread deg", round(max(angs), 4))
    print("   min joint margin per request (joint, rad):", [(j, round(v, 3)) for j, v in margins][:45])
