"""How well does a resolved-rate IK lift (OSC-like nullspace pull to the reset posture) reconstruct
arm joints between two request boundaries where the measured joints are recorded?

For each consecutive pair of requests (step a -> step b) in a run: start at the measured q_a, integrate
over the recorded measured Cartesian states a..b, compare to the measured q_b (link-origin errors in m).
Offline, CPU only, read-only on /workspace.
"""
import glob, json, pickle, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, "/workspace/gpt_field/src")
from body_runtime import BodyModel, rotvec_mat  # noqa
from gpt_field.robodojo_direct_sim import PandaKinematics, rotation_vector  # noqa

ASSET = Path("/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/.venv-sim/lib/python3.10/site-packages/robosuite/models/assets")
kin = PandaKinematics.from_xml(ASSET / "robots/panda/robot.xml", ASSET / "grippers/panda_gripper.xml")
bm = BodyModel(HERE.parent / "data/panda_body_a12_g5")
RUNS = {
    "T08_t100": ("agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001", "T08_t100_steps.pkl"),
    "T04_t100": ("agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001", "T04_t100_steps.pkl"),
    "drawer001": ("agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001", "drawer001_steps.pkl"),
}
DATA = Path("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data")


def boundaries(run):
    rows = []
    for f in sorted(glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/*.request.json")):
        req = json.load(open(f))
        inp = req["payload"]["input"][-1]
        for c in inp.get("content", []):
            if c.get("type") == "input_text":
                try:
                    ctx = json.loads(c["text"])
                except Exception:
                    break
                ms = (ctx.get("context") or {}).get("measured_state") or ctx.get("measured_state") or {}
                if ms.get("joint_position_rad") is not None and "step" in ctx:
                    rows.append((int(ctx["step"]), np.asarray(ms["joint_position_rad"], float)))
                break
    uniq = {}
    for s, q in rows:
        uniq[s] = q  # last request at a step wins (same observation)
    return sorted(uniq.items())


def lift(q, states, q_rest, alpha):
    q = q.copy()
    T, J = kin_b.forward(q)
    for k in range(1, len(states)):
        s0, s1 = states[k - 1], states[k]
        dp = np.asarray(s1[:3]) - np.asarray(s0[:3])
        dr = rotation_vector(rotvec_mat(np.asarray(s1[3:6])) @ rotvec_mat(np.asarray(s0[3:6])).T)
        # correct accumulated drift toward the measured pose as well (closed-loop resolved rate)
        T, J = kin_b.forward(q)
        ep = np.asarray(s1[:3]) - T[:3, 3]
        er = rotation_vector(rotvec_mat(np.asarray(s1[3:6])) @ T[:3, :3].T)
        e = np.r_[ep, er]
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-6 * np.eye(6))
        N = np.eye(7) - Jp @ J
        q = q + Jp @ e + N @ (alpha * (q_rest - q))
        q = np.clip(q, kin.low, kin.high)
    return q


link_bodies = [bm.index[f"link{i}"] for i in range(2, 8)]
result = {}
for label, (run, pkl) in RUNS.items():
    steps = pickle.load(open(DATA / pkl, "rb"))
    bnd = boundaries(run)
    s0, q0 = bnd[0]
    kin_b = kin.bind(np.asarray(steps[s0]["state"]), q0)
    W = kin_b.world_from_base
    errs = {a: [] for a in (0.0, 0.02, 0.05)}
    t_lift = []
    for (sa, qa), (sb, qb) in zip(bnd[:-1], bnd[1:]):
        if sb <= sa or sb > len(steps):
            continue
        states = [steps[i]["state"] for i in range(sa, sb)] + [steps[sb - 1]["next_state"]]
        for alpha in errs:
            t0 = time.perf_counter()
            qr = lift(qa, states, q0, alpha)
            t_lift.append((time.perf_counter() - t0) / max(1, len(states)))
            pa = bm.body_poses(np.stack([qr, qb]), np.zeros((2, 2)), W)
            e = max(float(np.linalg.norm(pa[b][0, :3, 3] - pa[b][1, :3, 3])) for b in link_bodies)
            errs[alpha].append((sb - sa, e, float(np.abs(qr - qb).max())))
    result[label] = {}
    for alpha, rows in errs.items():
        arr = np.asarray(rows)
        result[label][f"alpha_{alpha}"] = {
            "intervals": len(rows), "steps_median": float(np.median(arr[:, 0])),
            "link_origin_err_m_median": round(float(np.median(arr[:, 1])), 4),
            "link_origin_err_m_p90": round(float(np.percentile(arr[:, 1], 90)), 4),
            "link_origin_err_m_max": round(float(arr[:, 1].max()), 4),
            "joint_err_rad_max": round(float(arr[:, 2].max()), 3)}
    result[label]["lift_ms_per_step"] = round(1e3 * float(np.median(t_lift)), 3)
print(json.dumps(result, indent=1))
