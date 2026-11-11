"""Offline checks of the numpy-only prototype (controller .venv). Read-only on /workspace.

1. FK parity with robodojo_direct_sim.PandaKinematics (grip pose) for random joints.
2. Gripper-only placement from state8 == full FK placement after bind().
3. T08 t100 colliding stage vs the TRUE bottle AABB (diagnostic truth): measured profile,
   and three candidate executed profiles (coupled slerp, translate-then-rotate, rotate-first).
4. Timing of the vectorised checks.
"""
import json, math, pickle, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/workspace/gpt_field/src")
from body_runtime import BodyModel, sd_spheres_obb, rotvec_mat  # noqa
from gpt_field.robodojo_direct_sim import PandaKinematics, rotation_vector, matrix_quaternion  # noqa

ASSET = Path("/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/.venv-sim/lib/python3.10/site-packages/robosuite/models/assets")
MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "data/panda_body_a12_g5"
DATA = Path("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data")
out = {"model": str(MODEL)}
bm = BodyModel(MODEL)
kin = PandaKinematics.from_xml(ASSET / "robots/panda/robot.xml", ASSET / "grippers/panda_gripper.xml")
rng = np.random.default_rng(1)

# ---------- 1. FK parity ----------
K = 64
q = rng.uniform(kin.low, kin.high, size=(K, 7))
poses = bm.body_poses(q, np.tile([0.04, -0.04], (K, 1)), np.eye(4))
hand = poses[bm.index["right_hand"]]
grip_mine = hand[:, :3, 3] + np.einsum("kij,j->ki", hand[:, :3, :3], bm.grip_offset)
errs = []
for i in range(K):
    T, _ = kin.forward(q[i])
    errs.append(max(np.abs(T[:3, 3] - grip_mine[i]).max(), np.abs(T[:3, :3] - hand[i, :3, :3]).max()))
out["fk_parity_max_abs"] = float(max(errs))

# ---------- 2. state8 placement == bound FK placement ----------
state = np.array([-0.20137, 0.0187, 1.18932, 3.14, 0.0, -0.08, 0.0387, -0.0387])
q0 = np.array([0.01536, -0.15830, 0.00976, -2.41853, 0.00326, 2.20509, 0.80644])
bound = kin.bind(state, q0)
full = bm.body_poses(q0[None], state[None, 6:8], bound.world_from_base)
grip = bm.gripper_poses(bm.hand_from_state(state[None]), state[None, 6:8])
tool_parts = [j for j, p in enumerate(bm.parts) if p["kind"] in ("hand", "finger", "pad")]
a, _, _ = bm.place(full, tool_parts)
b, _, _ = bm.place(grip, tool_parts)
out["state8_vs_bound_fk_sphere_max_abs_m"] = float(np.abs(a - b).max())

# ---------- 3. T08 stage vs true bottle AABB ----------
steps = pickle.load(open(DATA / "T08_t100_steps.pkl", "rb"))
lo = np.array([-0.028435228620234555, -0.08961845611408367, 0.9002])
hi = np.array([0.02778477137976546, -0.05285845611408366, 1.0458100000000001])
box_c, box_h = ((lo + hi) / 2)[None], ((hi - lo) / 2)[None]
box_R = np.eye(3)[None]
names = [p["name"] for p in bm.parts]


def clearance(states):
    states = np.asarray(states, float)
    pz = bm.gripper_poses(bm.hand_from_state(states), states[:, 6:8])
    c, r, pid = bm.place(pz, tool_parts)
    sd = sd_spheres_obb(c, r, box_c, box_R, box_h)[:, :, 0]
    per_part = {names[j]: sd[:, pid == j].min(1) for j in tool_parts}
    return sd.min(1), per_part


window = list(range(26, 84))
S = np.array([steps[i]["state"] for i in window])
forb = np.array([steps[i]["forb_inc"] for i in window])
cmin, per_part = clearance(S)
first_neg = next((window[i] for i in range(len(window)) if cmin[i] < 0), None)
first_contact = next((window[i] for i in range(len(window)) if forb[i] > 0), None)
out["t08_measured_profile"] = {
    "first_contact_step_recorded": first_contact,
    "first_negative_clearance_step_model": first_neg,
    "min_clearance_m": float(cmin.min()),
    "clearance_at_steps": {str(window[i]): round(float(cmin[i]), 4) for i in range(0, len(window), 3)},
    "argmin_part_at_onset": min(per_part, key=lambda k: per_part[k][window.index(first_contact)]) if first_contact else None,
    "contact_steps_detected": int(((cmin < 0) & (forb > 0)).sum()), "contact_steps": int((forb > 0).sum()),
    "neg_without_contact": int(((cmin < 0) & (forb == 0)).sum()),
}

# Candidate executed profiles for the same stage: entry = state at step 35 (descent start),
# target = (-0.025,-0.023,1.035) with target quaternion from the fragment record (jaw ~ world x).
entry_i = 28
entry = np.array(steps[entry_i]["state"])
frag = steps[40]["frag"]
tq = None
import ast
fr = ast.literal_eval(frag) if isinstance(frag, str) else frag
tq = np.asarray(fr["target_q"], float)  # wxyz
tgt = np.asarray(fr["target"], float)
Rt = np.array(rotvec_mat(np.zeros(3)))


def quat_to_mat(wxyz):
    w, x, y, z = wxyz / np.linalg.norm(wxyz)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


R0 = rotvec_mat(entry[3:6])
R1 = quat_to_mat(tq)
rel = rotation_vector(R1 @ R0.T)
p0, p1 = entry[:3], tgt


def profile(tt, tr):
    st = []
    for a_t, a_r in zip(tt, tr):
        R = rotvec_mat(rel * a_r) @ R0
        st.append(np.r_[p0 + a_t * (p1 - p0), rotation_vector(R), entry[6:8]])
    return np.array(st)


n = 60
u = np.linspace(0, 1, n)
cands = {
    "coupled_slerp": profile(u, u),
    "translate_then_rotate_in_place": profile(np.clip(u * 3, 0, 1), np.clip((u * 3 - 1) / 2, 0, 1)),
    "rotate_first_at_entry_then_translate": profile(np.clip((u - 2 / 3) * 3, 0, 1), np.clip(u * 1.5, 0, 1)),
}
out["t08_candidate_profiles"] = {}
for k, st in cands.items():
    c, pp = clearance(st)
    worst = min(pp, key=lambda kk: pp[kk].min())
    out["t08_candidate_profiles"][k] = {"min_clearance_m": round(float(c.min()), 4),
                                         "argmin_fraction": round(float(u[int(np.argmin(c))]), 3),
                                         "worst_part": worst}
out["t08_stage"] = {"entry_step": entry_i, "entry_xyz": entry[:3].round(4).tolist(), "target_xyz": tgt.tolist(),
                    "rotation_rad": round(float(np.linalg.norm(rel)), 3)}

# ---------- 4. timing ----------
arm_parts = [j for j, p in enumerate(bm.parts) if p["kind"] == "arm" and p["body_name"] != "link0"]
all_parts = arm_parts + tool_parts
timing = {}
for K in (1, 60, 200):
    for M in (4, 20):
        qs = q0[None] + rng.normal(0, 0.05, size=(K, 7))
        bc = rng.uniform([-0.3, -0.3, 0.9], [0.1, 0.3, 1.1], size=(M, 3))
        bh = rng.uniform(0.02, 0.08, size=(M, 3))
        bR = np.broadcast_to(np.eye(3), (M, 3, 3))
        reps = 20 if K < 200 else 5
        t0 = time.perf_counter()
        for _ in range(reps):
            pz = bm.body_poses(qs, np.tile(state[6:8], (K, 1)), bound.world_from_base)
        t_fk = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            c, r, pid = bm.place(pz, all_parts)
        t_place = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            sd = sd_spheres_obb(c, r, bc, bR, bh)
        t_full = (time.perf_counter() - t0) / reps
        # broad phase: per part bounding spheres first, narrow only on candidate (part, box) pairs
        t0 = time.perf_counter()
        for _ in range(reps):
            P = len(all_parts)
            bcs = np.stack([np.einsum("kij,j->ki", pz[bm.parts[j]["body"]][:, :3, :3], bm.parts[j]["bc"]) + pz[bm.parts[j]["body"]][:, :3, 3] for j in all_parts], 1)
            brs = np.array([bm.parts[j]["br"] for j in all_parts])
            sdb = sd_spheres_obb(bcs, brs, bc, bR, bh)  # (K,P,M)
            near = np.argwhere((sdb < 0.05).any(0))  # (pair) candidate part/box pairs
            for pi, mi in near:
                sel = pid == all_parts[pi]
                sd_sp = sd_spheres_obb(c[:, sel], r[sel], bc[mi:mi + 1], bR[mi:mi + 1], bh[mi:mi + 1])
        t_broad = (time.perf_counter() - t0) / reps
        timing[f"K{K}_M{M}"] = {"fk_ms": round(1e3 * t_fk, 3), "place_ms": round(1e3 * t_place, 3),
                                "narrow_all_pairs_ms": round(1e3 * t_full, 3),
                                "broad_plus_narrow_ms": round(1e3 * t_broad, 3), "spheres": int(len(r)),
                                "candidate_pairs": int(len(near))}
out["timing"] = timing
# gripper-only from state8 (no IK): K=60
t0 = time.perf_counter()
for _ in range(50):
    st = np.tile(state, (60, 1))
    pz = bm.gripper_poses(bm.hand_from_state(st), st[:, 6:8])
    c, r, pid = bm.place(pz, tool_parts)
    sd = sd_spheres_obb(c, r, box_c, box_R, box_h)
out["timing"]["gripper_only_K60_M1_ms"] = round(1e3 * (time.perf_counter() - t0) / 50, 3)
print(json.dumps(out, indent=1))
