"""Strict physical truth check of a finished agent run (DIAGNOSIS ONLY; zero model calls; spec §6.5, AC-2).

usage: MUJOCO_EGL_DEVICE_ID=2 /workspace/venvs/libero-plus/bin/python -B scripts/agent_speedup_v2_truth_check.py \
           <run_trial_dir> <fresh_out_dir> [--release 25] [--retreat auto|up|none] [--retreat-m 0.06] \
           [--settle 60] [--no-images]

Promoted from the Phase A prototype ``survey/grounding/truth_check.py``.

1. Reads the run's manifest.json (suite, task_id, episode_index, simulator configuration) and copies its
   frozen_source to <out>/pkg/gpt_field, so the replay runs the run's own code; nothing is written into the run
   directory. A v1 run replays with its _LiberoWorker (v1 monitor as configured); a v2 run with its
   V2LiberoWorker and the manifest's monitor options.
2. Streams trace.jsonl (reset + executed events; images stripped) and replays every executed action in ONE fresh
   worker (one reset; KITCHEN_SCENE4/6/8 inits depend on the reset count) with stop_on_success=False, checking
   state8, libero_success and the forbidden-contact count against the trace at every step, and the reset
   images byte for byte.
3. Evaluates every BDDL goal atom with LIBERO's own predicate (raw) and the strict physical check of the CURRENT
   ``src/gpt_field/agent_speedup_v2_truth.py`` (loaded by path; its sha256 is reported):
   - end: the recorded end state (windowed over the last 10 recorded steps);
   - A: open the gripper with zero arm motion (--release), retreat the hand (--retreat; reverse of the hand's net
     horizontal motion over the last 20 steps when that exceeds 2 cm and is mostly horizontal, else up), settle;
   - B: restored end state, open + zero motion for release + settle steps (no retreat);
   - C: restored end state, the runner's counterfactual (open 10, lift 8, settle 20; spec §6.5).
   Every tolerance and protocol step is task-independent.
4. Writes <out>/truth.json (+ end / post agentview and wrist PNGs) and prints the verdict.

Verdicts:
- AC-2 (``verdict.ac2``): physical_success / lenient_success / raw_failure / raw_failure_strict_holds, from the end
  state and, when the only failing end sub-checks come from the robot still holding the object, protocol C.
- PA[2] (``verdict.pa2``): PHYSICAL_SUCCESS (raw at end AND strict after A), LENIENT_SUCCESS (raw at end, strict
  fails after A), RAW_FAILURE, RAW_FAILURE_BUT_STRICT_HOLDS.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
TRUTH_SOURCE = ROOT / "src" / "gpt_field" / "agent_speedup_v2_truth.py"
IMG = re.compile(rb'"data:image/png;base64,[A-Za-z0-9+/=]*"')


def load_truth_module():
    spec = importlib.util.spec_from_file_location("agent_speedup_v2_truth_evaluator", TRUTH_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(TRUTH_SOURCE.read_bytes()).hexdigest()


def read_trace(run):
    reset_state, steps, reset_images = None, [], {}
    with open(os.path.join(run, "trace.jsonl"), "rb") as stream:
        for line in stream:
            head = line[:24]
            if head.startswith(b'{"event": "executed"'):
                row = json.loads(IMG.sub(b'"<png>"', line))
                status = row.get("sim_status") or {}
                steps.append({"step": row["step"], "action": row["action"], "next_state": row["next_state"],
                              "libero_success": status.get("libero_success", status.get("success")),
                              "forbidden": ((status.get("collision") or {}).get("counts") or {}).get(
                                  "forbidden_contact_points")})
            elif head.startswith(b'{"event": "reset"'):
                row = json.loads(line)
                reset_state = row["observation"]["observation/state"]
                reset_images = {k: row["observation"].get(k) for k in ("observation/image", "observation/wrist_image")}
    steps.sort(key=lambda s: s["step"])
    if reset_state is None or not steps:
        raise ValueError("trace has no reset/executed events")
    return reset_state, steps, reset_images


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--release", type=int, default=25)
    ap.add_argument("--retreat", choices=("auto", "up", "none"), default="auto")
    ap.add_argument("--retreat-m", type=float, default=0.06)
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args(argv)
    began = time.time()
    run, out = os.path.abspath(args.run_dir), os.path.abspath(args.out_dir)
    if os.path.exists(out):
        ap.error("out_dir exists; choose a fresh directory")
    os.makedirs(out)
    manifest = json.load(open(os.path.join(run, "manifest.json")))["configuration"]
    simc = manifest["simulator"]
    suite, task_id, episode = simc["suite"], int(manifest["task_id"]), int(manifest["episode_index"])
    is_v2 = manifest.get("method") == "agent_speedup_v2"
    pkg = os.path.join(out, "pkg")
    shutil.copytree(os.path.join(run, "frozen_source"), os.path.join(pkg, "gpt_field"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"))
    sys.path.insert(0, pkg)
    import numpy as np
    import mujoco
    from gpt_field.libero import LiberoConfig, _LiberoWorker
    from gpt_field import observation as gobs
    truth, truth_sha = load_truth_module()
    reset_state, steps, reset_images = read_trace(run)

    cfg = LiberoConfig(suite=suite, seed=int(simc["seed"]), max_steps=None,
                       settle_steps=int(simc.get("settle_steps", 10)), libero_root=simc["libero_root"],
                       include_arm_joints=bool(simc.get("include_arm_joints", False)),
                       render_size=int(simc["render_size"]), image_size=int(simc["image_size"]),
                       collision_check_mode=simc.get("collision_check_mode", "none"),
                       stop_on_collision=bool(simc.get("stop_on_collision", False)), stop_on_success=False)
    if is_v2:
        from gpt_field.agent_speedup_v2_sim import V2LiberoWorker
        w = V2LiberoWorker(cfg, manifest.get("simulator_monitor"))
    else:
        w = _LiberoWorker(cfg)
    orig_response = w._response
    last = {}

    def fast_response(obs):
        last["obs"] = obs
        st = (*np.asarray(obs["robot0_eef_pos"], float), *gobs.quaternion_to_axisangle(obs["robot0_eef_quat"]),
              *np.asarray(obs["robot0_gripper_qpos"], float))
        return {"observation": {"observation/state": [float(x) for x in st]}, "status": w.status.copy()}

    w._response = fast_response
    resp = w.reset(task_id, episode)
    env = w.env
    sc = truth.Scene(env)
    D, M = sc.D, sc.M
    gripper = env.robots[0].gripper
    goal = sc.goal()
    grip_site = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    win_objs, win_joints = truth.window_targets(sc)
    window = truth.Window(k=truth.WINDOW_STEPS)

    def raw_atoms():
        return [bool(sc.core._eval_predicate(a)) for a in goal]

    def save_images(tag):
        if args.no_images or "obs" not in last:
            return None
        from PIL import Image
        o = orig_response(last["obs"])["observation"]
        paths = {}
        for key, name in (("observation/image", "agentview"), ("observation/wrist_image", "wrist")):
            img = Image.open(io.BytesIO(base64.b64decode(o[key].split(",", 1)[1])))
            img = img.transpose(Image.FLIP_LEFT_RIGHT)   # the same client-side upright flip as v1
            path = os.path.join(out, f"{tag}_{name}.png")
            img.save(path)
            paths[name] = path
        return paths

    res = {"run_dir": run, "method": manifest.get("method"), "suite": suite, "task_id": task_id, "episode": episode,
           "task": w.prompt, "goal": goal, "tolerances": truth.TOL, "truth_module_sha256": truth_sha,
           "truth_version": truth.TRUTH_VERSION, "diagnosis_only": True, "affects_completion": False,
           "protocol": {"release": args.release, "retreat": args.retreat, "retreat_m": args.retreat_m,
                        "settle": args.settle, "runner_counterfactual": truth.RELEASE_SETTLE}}
    current = orig_response(last["obs"])["observation"]
    res["reset_images_identical"] = {k: (current[k] == v) for k, v in reset_images.items() if v}
    res["reset_state_err"] = float(np.max(np.abs(np.asarray(resp["observation"]["observation/state"])
                                                - np.asarray(reset_state))))

    max_err, err_steps, success_mismatch, forbidden_mismatch = 0.0, [], [], []
    first_true, flips = [None] * len(goal), [0] * len(goal)
    previous = raw_atoms()
    success_steps, grip_hist = [], []
    for idx, s in enumerate(steps):
        rr = w.step(s["action"])
        if idx >= len(steps) - window.k:
            window.update(sc, win_objs, win_joints)
        e = float(np.max(np.abs(np.asarray(rr["observation"]["observation/state"]) - np.asarray(s["next_state"]))))
        max_err = max(max_err, e)
        if e > 0:
            err_steps.append([s["step"], e])
        st = rr["status"]
        ls = bool(st.get("libero_success", st.get("success")))
        if s["libero_success"] is not None and ls != bool(s["libero_success"]):
            success_mismatch.append([s["step"], ls, s["libero_success"]])
        fb = ((st.get("collision") or {}).get("counts") or {}).get("forbidden_contact_points")
        if s["forbidden"] is not None and fb != s["forbidden"]:
            forbidden_mismatch.append([s["step"], fb, s["forbidden"]])
        if ls:
            success_steps.append(s["step"])
        now = raw_atoms()
        for i, (a, b) in enumerate(zip(previous, now)):
            if b and first_true[i] is None:
                first_true[i] = s["step"]
            flips[i] += int(a != b)
        previous = now
        grip_hist.append(np.asarray(D.site_xpos[grip_site], float).copy())
    bitwise = (max_err == 0.0 and res["reset_state_err"] == 0.0 and all(res["reset_images_identical"].values())
               and not success_mismatch and not forbidden_mismatch)
    res["replay"] = {"n_steps": len(steps), "last_step": steps[-1]["step"], "max_state_err": max_err,
                     "bitwise": bitwise, "err_steps": err_steps[:20], "success_flag_mismatches": success_mismatch[:20],
                     "forbidden_count_mismatches": forbidden_mismatch[:20],
                     "first_libero_success_step": success_steps[0] if success_steps else None,
                     "n_libero_success_steps": len(success_steps), "atom_first_true_step": first_true,
                     "atom_raw_flips": flips}
    print("replay", len(steps), "bitwise", bitwise, "max_err", max_err, "first_success",
          res["replay"]["first_libero_success_step"], flush=True)

    def snapshot(tag):
        ev = truth.evaluate_goal(sc, window)
        ev["grip_site"] = truth.r(D.site_xpos[grip_site])
        ev["finger_qpos"] = truth.r([float(sc.sim.data.get_joint_qpos("gripper0_finger_joint1")),
                                     float(sc.sim.data.get_joint_qpos("gripper0_finger_joint2"))], 5)
        ev["images"] = save_images(tag)
        return ev

    def save():
        return {"qpos": D.qpos.copy(), "qvel": D.qvel.copy(), "act": D.act.copy(), "time": float(D.time),
                "warm": D.qacc_warmstart.copy(), "ctrl": D.ctrl.copy(), "grip": gripper.current_action.copy(),
                "qfrc_applied": D.qfrc_applied.copy(), "xfrc_applied": D.xfrc_applied.copy()}

    def restore(saved):
        D.qpos[:] = saved["qpos"]; D.qvel[:] = saved["qvel"]; D.act[:] = saved["act"]; D.time = saved["time"]
        D.ctrl[:] = saved["ctrl"]; D.qfrc_applied[:] = saved["qfrc_applied"]
        D.xfrc_applied[:] = saved["xfrc_applied"]
        gripper.current_action = saved["grip"].copy()
        mujoco.mj_forward(M, D)
        D.qacc_warmstart[:] = saved["warm"]     # after the forward, which rewrites the warm start

    def run_post(actions, tag):
        n_raw, robot_touch = 0, set()
        for a in actions:
            w.step(list(a))
            window.update(sc, win_objs, win_joints)
            n_raw += bool(env.check_success())
            for o in win_objs:
                for c in sc.robot_contacts(sc.geoms_of(sc.object_bodies(o))):
                    robot_touch.add(f"{o}<->{c['other']}")
        ev = snapshot(tag)
        ev.update(post_steps=len(actions), raw_true_fraction_during_post=truth.r(n_raw / max(1, len(actions)), 3),
                  robot_object_contacts_during_post=sorted(robot_touch))
        return ev

    end_state = save()
    res["end"] = snapshot("end")
    gh = np.asarray(grip_hist)
    net = gh[-1] - gh[max(0, len(gh) - 20)]
    if args.retreat == "none":
        rdir = np.zeros(3)
    elif args.retreat == "up" or np.linalg.norm(net[:2]) < 0.02 or abs(net[2]) > np.linalg.norm(net[:2]):
        rdir = np.array([0.0, 0.0, 1.0])
    else:
        rdir = -np.array([net[0], net[1], 0.0]) / np.linalg.norm(net[:2])
    n_ret = int(round(args.retreat_m / 0.01)) if args.retreat != "none" else 0
    opened = [0, 0, 0, 0, 0, 0, -1.0]
    retreat = [float(x) for x in (rdir * 0.2)] + [0, 0, 0, -1.0]   # 0.2 x 0.05 m = 1 cm per step
    res["protocol"].update(retreat_dir=truth.r(rdir, 3), hand_net_motion_last20=truth.r(net))
    res["A1_released"] = run_post([opened] * args.release, "A1_released")
    res["A2_retreated_settled"] = run_post([retreat] * n_ret + [opened] * args.settle, "A2_retreated_settled")
    restore(end_state)
    window.reset()
    res["B_release_settle_no_retreat"] = run_post([opened] * (args.release + args.settle), "B_no_retreat")
    restore(end_state)
    window.reset()
    res["C_runner_counterfactual"] = run_post(truth.release_settle_actions(), "C_runner_counterfactual")

    raw_end = bool(res["end"]["check_success"])
    strict_a = bool(res["A2_retreated_settled"]["strict_conjunction"])
    pa2 = ("PHYSICAL_SUCCESS" if raw_end and strict_a else "LENIENT_SUCCESS" if raw_end else
           ("RAW_FAILURE_BUT_STRICT_HOLDS" if strict_a else "RAW_FAILURE"))
    ac2 = truth.physical_summary(res["end"], res["C_runner_counterfactual"], raw_libero_success=raw_end)
    res["verdict"] = {
        "ac2": ac2["verdict"], "ac2_summary": ac2, "pa2": pa2, "replay_bitwise": bitwise,
        "raw_libero_success_end": raw_end, "strict_end": bool(res["end"]["strict_conjunction"]),
        "raw_after_release_retreat_settle": bool(res["A2_retreated_settled"]["check_success"]),
        "strict_after_release_retreat_settle": strict_a,
        "raw_after_release_no_retreat": bool(res["B_release_settle_no_retreat"]["check_success"]),
        "strict_after_release_no_retreat": bool(res["B_release_settle_no_retreat"]["strict_conjunction"]),
        "raw_after_runner_counterfactual": bool(res["C_runner_counterfactual"]["check_success"]),
        "strict_after_runner_counterfactual": bool(res["C_runner_counterfactual"]["strict_conjunction"]),
        "failing_strict_checks_end": truth.failing_checks(res["end"]),
        "failing_strict_checks_after_A": truth.failing_checks(res["A2_retreated_settled"]),
    }
    res["wall_s"] = round(time.time() - began, 1)
    with open(os.path.join(out, "truth.json"), "w") as stream:
        json.dump(truth.jsonable(res), stream, indent=1, allow_nan=False)
    print("VERDICT ac2", ac2["verdict"], "| pa2", pa2, "| bitwise", bitwise, "| raw_end", raw_end, "strict_end",
          res["verdict"]["strict_end"], "| failing end", res["verdict"]["failing_strict_checks_end"],
          "| wall", res["wall_s"], flush=True)
    w.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
