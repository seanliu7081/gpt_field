"""Write robot_facts_v2.json from the simulator's own robot collision geoms (spec §4.4; ROBOT facts only).

Reading the robot's own asset (the compiled MuJoCo model of the Panda arm and PandaGripper) is a ROBOT fact, not
evaluator truth: nothing about objects, fixtures or the scene layout is read, except the robot base and its start
pose (R13/R14), which are the robot's own bodies. Every quantity is expressed in the right_hand body frame
(R0 = x, R1 = y = jaw / finger-separation, R2 = z = approach), relative to the grip site where stated.

Measured:
- grip site offset along R2 (R1 fact), wrist camera mount and fovy -> f at 512 px (R2/R3, under R-2f);
- fingertip reach t_f beyond the grip (R5, asset value; the pinned value is the contact-calibrated 9.3 mm);
- pad span along the approach and pad extents (R4);
- palm underside p_min and the palm profile p(xi) along the jaw (R6), from the convex hull of the hand collision
  mesh (MuJoCo collides with the hull);
- hand footprint (R7); pad inner/outer faces (R8) at the reset finger q AND at the joint limit q = 40 mm (the
  spec's ">= 38.2 mm" is the reset-q value; 39.49 mm is the limit), the pad separation offset gap - separation
  (R9's asset value) and the per-finger q at which the pads touch;
- finger landing footprints keyed by the PER-FINGER joint q in mm (P16; finger gap = 2 q): each finger's lowest
  band (within 1 mm of its lowest point); keys below the pad-contact q are listed as below the hard stop;
- the palm profile includes the hull's exact end points (the footprint's R1 extremes);
- joint-7 range and start q7 (R12); robot base, shoulder and the start pose relative to the base (R13/R14) for
  the requested tasks (one fresh worker process per task).

usage: MUJOCO_EGL_DEVICE_ID=2 /workspace/venvs/libero-plus/bin/python scripts/agent_speedup_v2_robot_facts.py \
           --out src/gpt_field/robot_facts_v2.json [--keys libero_10_t2,bowl_plate,drawer_bowl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FACTS_VERSION = "robot-facts-v2-2"
FINGER_Q_M = (0.0, 0.01, 0.02, 0.03, 0.04)      # per-finger joint q (each finger 0..40 mm; gap = 2 q)
LOW_BAND_M = 0.001
PROFILE_STEP_M = 0.002


def _hull_upper(points_yz, xi):
    """Upper envelope (max z) of the 2-D convex hull of (y, z) points at y = xi; None outside the hull."""
    from gpt_field.agent_speedup_v2_truth import hull2d
    import numpy as np
    hull = hull2d(points_yz)
    best = None
    n = len(hull)
    for i in range(n):
        (y0, z0), (y1, z1) = hull[i], hull[(i + 1) % n]
        if (y0 - xi) * (y1 - xi) > 0:
            continue
        if y0 == y1:
            candidates = [z0, z1] if abs(y0 - xi) < 1e-12 else []
        else:
            t = (xi - y0) / (y1 - y0)
            candidates = [z0 + t * (z1 - z0)] if 0.0 <= t <= 1.0 else []
        for z in candidates:
            best = z if best is None else max(best, z)
    return None if best is None else float(np.round(best, 6))


def single(key):
    import mujoco
    import numpy as np
    from gpt_field.agent_speedup_v2_tasks import build_v2_sim_config, resolve_v2_task
    from gpt_field.libero import _LiberoWorker
    spec = resolve_v2_task(key)
    config, _ = build_v2_sim_config(key)
    from dataclasses import replace
    config = replace(config, collision_check_mode="none", render_size=128, image_size=128)
    worker = _LiberoWorker(config)
    try:
        response = worker.reset(spec.task_id, 0)
        M, D = worker.env.sim.model._model, worker.env.sim.data._data
        name = lambda kind, i: mujoco.mj_id2name(M, kind, i)
        hand = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "robot0_right_hand")
        Rh, ph = D.xmat[hand].reshape(3, 3).copy(), D.xpos[hand].copy()
        to_hand = lambda pts: (np.asarray(pts, float) - ph) @ Rh
        grip = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        grip_h = to_hand([D.site_xpos[grip]])[0]
        geoms = {}
        for g in range(M.ngeom):
            gname = name(mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if not gname.startswith("gripper0_") or not (M.geom_contype[g] or M.geom_conaffinity[g]):
                continue
            R, p = D.geom_xmat[g].reshape(3, 3), D.geom_xpos[g]
            if int(M.geom_type[g]) == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(M.geom_dataid[g])
                v = np.asarray(M.mesh_vert[int(M.mesh_vertadr[mid]):int(M.mesh_vertadr[mid]) + int(M.mesh_vertnum[mid])])
            elif int(M.geom_type[g]) == mujoco.mjtGeom.mjGEOM_BOX:
                v = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * M.geom_size[g]
            else:
                continue
            geoms[gname] = {"hand_frame": to_hand(v @ R.T + p), "body": name(mujoco.mjtObj.mjOBJ_BODY,
                                                                                int(M.geom_bodyid[g]))}
        joints = {}
        for j in range(M.njnt):
            jname = name(mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if jname.startswith(("robot0_joint", "gripper0_finger_joint")):
                joints[jname] = {"range": [float(x) for x in M.jnt_range[j]], "q": float(D.qpos[M.jnt_qposadr[j]]),
                                 "axis_hand": [round(float(x), 6) for x in
                                               (D.xmat[M.jnt_bodyid[j]].reshape(3, 3) @ M.jnt_axis[j]) @ Rh]}
        cam = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_eye_in_hand")
        cam_pos_h = to_hand([D.cam_xpos[cam]])[0]
        cam_axes_h = D.cam_xmat[cam].reshape(3, 3).T @ Rh   # rows: camera x, y, z in hand coordinates
        base = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
        link1 = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "robot0_link1")
        link2 = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, "robot0_link2")
        state = response["observation"]["observation/state"]
        out = {"key": key, "suite": spec.suite, "task_id": spec.task_id,
               "grip_site_hand": grip_h.tolist(), "geoms": {k: {"hand_frame": v["hand_frame"].tolist(),
                                                                 "body": v["body"]} for k, v in geoms.items()},
               "joints": joints, "camera": {"pos_hand": cam_pos_h.tolist(), "axes_hand": cam_axes_h.tolist(),
                                            "fovy_deg": float(M.cam_fovy[cam])},
               "base_xyz": D.xpos[base].tolist(), "link1_xyz": D.xpos[link1].tolist(),
               "link2_xyz": D.xpos[link2].tolist(), "start_state8": list(state),
               "start_grip_minus_base": (np.asarray(state[:3]) - D.xpos[base]).tolist()}
        return out
    finally:
        worker.close()


def combine(rows):
    import numpy as np
    first = rows[0]
    G = np.asarray(first["grip_site_hand"])
    gz = float(G[2])
    geoms = {k: np.asarray(v["hand_frame"]) for k, v in first["geoms"].items()}
    hand = geoms["gripper0_hand_collision"]
    fingers = {k: v for k, v in geoms.items() if "finger" in k and "pad" not in k}
    pads = {k: v for k, v in geoms.items() if "pad" in k}
    rel = lambda pts: np.asarray(pts) - G
    tip_beyond = max(float(rel(v)[:, 2].max()) for v in fingers.values())
    pad_z = [float(min(rel(v)[:, 2].min() for v in pads.values())), float(max(rel(v)[:, 2].max() for v in pads.values()))]
    palm = float(-(rel(hand)[:, 2].max()))
    hand_rel = rel(hand)
    lo_xi, hi_xi = float(hand_rel[:, 1].min()), float(hand_rel[:, 1].max())
    xs = np.arange(round(lo_xi, 3), hi_xi + 1e-9, PROFILE_STEP_M)
    profile = []
    for xi in xs:
        top = _hull_upper(hand_rel[:, 1:3], float(xi))
        if top is not None and lo_xi < xi < hi_xi:
            profile.append([round(float(xi) * 1000, 1), round(-top * 1000, 2)])
    # the hull's exact end points, so the profile spans the whole footprint (hand_footprint_R1_mm)
    ends = [[round(xi * 1000, 2), round(-_hull_upper(hand_rel[:, 1:3], xi) * 1000, 2)] for xi in (lo_xi, hi_xi)]
    profile = [ends[0], *[row for row in profile if ends[0][0] < row[0] < ends[1][0]], ends[1]]
    q_now = {k: v["q"] for k, v in first["joints"].items() if k.startswith("gripper0_finger_joint")}
    footprints = {}
    for finger_q in FINGER_Q_M:
        per = {}
        for fname, pts in list(fingers.items()) + list(pads.items()):
            index = "1" if "finger1" in fname else "2"
            joint = first["joints"][f"gripper0_finger_joint{index}"]
            axis = np.asarray(joint["axis_hand"])
            target = finger_q if joint["range"][1] > 0 else -finger_q
            moved = rel(pts) + axis * (target - q_now[f"gripper0_finger_joint{index}"])
            per[fname] = moved
        for index in ("1", "2"):
            pts = np.vstack([v for k, v in per.items() if f"finger{index}" in k])
            low = pts[pts[:, 2] >= pts[:, 2].max() - LOW_BAND_M]
            footprints.setdefault(f"{finger_q * 1000:.0f}", {})[f"finger{index}"] = {
                "lowest_beyond_grip_mm": round(float(pts[:, 2].max()) * 1000, 2),
                "R0_mm": [round(float(low[:, 0].min()) * 1000, 2), round(float(low[:, 0].max()) * 1000, 2)],
                "R1_mm": [round(float(low[:, 1].min()) * 1000, 2), round(float(low[:, 1].max()) * 1000, 2)],
                "finger_and_pad_R1_mm": [round(float(pts[:, 1].min()) * 1000, 2), round(float(pts[:, 1].max()) * 1000, 2)]}
    pad1 = rel(pads[[k for k in pads if "finger1" in k][0]])
    pad2 = rel(pads[[k for k in pads if "finger2" in k][0]])
    q1, q2 = q_now["gripper0_finger_joint1"], q_now["gripper0_finger_joint2"]
    limit_q = float(first["joints"]["gripper0_finger_joint1"]["range"][1])
    axis1 = np.asarray(first["joints"]["gripper0_finger_joint1"]["axis_hand"])
    axis2 = np.asarray(first["joints"]["gripper0_finger_joint2"]["axis_hand"])

    def faces(per_finger_q):
        """Pad 1 inner / outer face |R1| and the pad separation (both pads) at a per-finger q."""
        a = pad1 + axis1 * (per_finger_q - q1)
        b = pad2 + axis2 * (-per_finger_q - q2)
        separation = float(a[:, 1].min() - b[:, 1].max()) if a[:, 1].mean() > b[:, 1].mean() else float(
            b[:, 1].min() - a[:, 1].max())
        return float(np.abs(a[:, 1]).min()), float(np.abs(a[:, 1]).max()), separation
    reset_q = (q1 - q2) / 2.0
    inner_reset, outer_reset, sep_reset = faces(reset_q)
    inner_limit, outer_limit, sep_limit = faces(limit_q)
    pad_contact_q = reset_q - sep_reset / 2.0          # pads touch: separation 0 (separation falls 2 mm per mm q)
    fovy = first["camera"]["fovy_deg"]
    f512 = 256.0 / np.tan(np.radians(fovy) / 2)
    cam_rel = np.asarray(first["camera"]["pos_hand"])
    starts = []
    for row in rows:
        starts.append({"key": row["key"], "suite": row["suite"], "task_id": row["task_id"],
                       "base_xyz": [round(v, 5) for v in row["base_xyz"]],
                       "shoulder_link2_xyz": [round(v, 5) for v in row["link2_xyz"]],
                       "start_grip_xyz": [round(v, 5) for v in row["start_state8"][:3]],
                       "start_grip_minus_base_m": [round(v, 5) for v in row["start_grip_minus_base"]],
                       "q7_start": round(row["joints"]["robot0_joint7"]["q"], 5)})
    asset = {
        "grip_site_along_R2_m": round(gz, 6),
        "grip_site_R0_R1_m": [round(float(G[0]), 6), round(float(G[1]), 6)],
        "fingertip_reach_beyond_grip_mm": round(tip_beyond * 1000, 2),
        "pad_span_along_approach_mm": [round(pad_z[0] * 1000, 2), round(pad_z[1] * 1000, 2)],
        "pad_extent_R0_mm": round(float(np.ptp(pad1[:, 0])) * 1000, 2),
        "pad_extent_R1_mm": round(float(np.ptp(pad1[:, 1])) * 1000, 2),
        "palm_underside_behind_grip_mm": round(palm * 1000, 2),
        "palm_profile_mm": {"xi_R1_mm_and_p_mm": profile,
                            "definition": "p(xi) = grip minus the hand-collision convex hull's lowest surface "
                                          "(toward the fingertips) at jaw coordinate xi"},
        "hand_footprint_R1_mm": [round(float(hand_rel[:, 1].min()) * 1000, 2), round(float(hand_rel[:, 1].max()) * 1000, 2)],
        "hand_footprint_R0_mm": [round(float(hand_rel[:, 0].min()) * 1000, 2), round(float(hand_rel[:, 0].max()) * 1000, 2)],
        "pad_inner_face_mm": {"at_reset_q": round(inner_reset * 1000, 2), "reset_q_m": round(reset_q, 6),
                              "at_limit_q": round(inner_limit * 1000, 2), "limit_q_m": limit_q},
        "pad_outer_face_mm": {"at_reset_q": round(outer_reset * 1000, 2), "reset_q_m": round(reset_q, 6),
                              "at_limit_q": round(outer_limit * 1000, 2), "limit_q_m": limit_q},
        "pad_separation_offset_mm": {"at_reset_q": round((2 * reset_q - sep_reset) * 1000, 3),
                                     "at_limit_q": round((2 * limit_q - sep_limit) * 1000, 3),
                                     "definition": "finger gap (q1 - q2 = 2 q) minus the pad separation (R9)"},
        "pad_contact_finger_q_mm": round(pad_contact_q * 1000, 3),
        "finger_landing_footprints_by_finger_q_mm": footprints,
        "finger_landing_footprints_key_semantics": "per-finger joint q in mm (each finger 0..40 mm); finger gap = "
                                                   "2 q; the pads touch at pad_contact_finger_q_mm",
        "finger_landing_footprints_below_hard_stop": sorted(
            (key for key in footprints if float(key) < pad_contact_q * 1000), key=float),
        "finger_joint_ranges_m": {k: v["range"] for k, v in first["joints"].items() if "finger" in k},
        "joint7_range_rad": first["joints"]["robot0_joint7"]["range"],
        "wrist_camera": {"pos_hand_m": [round(float(v), 6) for v in cam_rel], "fovy_deg": fovy,
                         "f_px_at_512": round(float(f512), 3),
                         "camera_axes_in_hand": [[round(float(v), 4) for v in row] for row in first["camera"]["axes_hand"]]},
        "low_band_m": LOW_BAND_M,
    }
    return {"version": FACTS_VERSION, "source": "compiled MuJoCo model of the LIBERO Panda + PandaGripper "
            "(robot collision geoms, joints, wrist camera, base); right_hand frame, grip-site relative",
            "frame": "R0 = hand x (wrist image up), R1 = hand y (jaw, wrist image right), R2 = hand z (approach)",
            "asset": asset, "start_poses": starts, "measured_tasks": [r["key"] for r in rows]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--keys", default="libero_10_t2,bowl_plate,drawer_bowl")
    parser.add_argument("--single")
    args = parser.parse_args(argv)
    if args.single:
        print("ROW " + json.dumps(single(args.single)), flush=True)
        return 0
    rows = []
    for key in args.keys.split(","):
        out = subprocess.run([sys.executable, "-B", __file__, "--single", key], capture_output=True, text=True,
                             check=True)
        rows.append(json.loads([l for l in out.stdout.splitlines() if l.startswith("ROW ")][-1][4:]))
    facts = combine(rows)
    text = json.dumps(facts, indent=1, sort_keys=False) + "\n"
    facts_sha = hashlib.sha256(text.encode()).hexdigest()
    args.out.write_text(text)
    print(json.dumps({"out": str(args.out), "sha256": facts_sha, **{k: facts["asset"][k] for k in (
        "fingertip_reach_beyond_grip_mm", "pad_span_along_approach_mm", "palm_underside_behind_grip_mm",
        "hand_footprint_R1_mm", "pad_inner_face_mm", "pad_separation_offset_mm", "pad_contact_finger_q_mm")},
        "f": facts["asset"]["wrist_camera"]["f_px_at_512"],
        "starts": facts["start_poses"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
