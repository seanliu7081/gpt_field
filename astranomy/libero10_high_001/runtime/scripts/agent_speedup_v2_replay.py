"""Bitwise replay of a v1 or v2 trace.jsonl to step k (DIAGNOSIS / TEST TOOL; zero model calls; spec §6.6).

usage: MUJOCO_EGL_DEVICE_ID=2 /workspace/venvs/libero-plus/bin/python -B scripts/agent_speedup_v2_replay.py \
           <run_trial_dir> <fresh_out_dir> [--to-step K] [--save-state] [--no-truth]

Promoted from the Phase A prototype ``survey/infra/replay_v1_trace.py``. The run's frozen_source is copied to
<out>/pkg/gpt_field and the replay runs that code in ONE fresh worker (one reset), so the replay is bitwise:

- the reset observation sha256 and every executed observation's sha256 (images included) are asserted against
  the trace, as are state8 (max |deviation| must be 0.0), libero_success and the forbidden-contact count;
- v1 runs: the v1 ``task_manipulation_v1`` monitor runs as recorded, and the CURRENT
  ``agent_speedup_v2_contacts.GenericContactMonitor`` (loaded by path) is chained on top; their per-step forbidden
  flags are compared (the spec §6.4 monitor regression: first events, agreement counts);
- v2 runs: the frozen V2LiberoWorker with the manifest's monitor options;
- at step k: the strict physical truth (current ``agent_speedup_v2_truth.py``, windowed over the last 10 steps)
  and, with --save-state, the MuJoCo state (qpos, qvel, act, ctrl, warm start, gripper action, time) as
  state_k.npz for live-state relation replays.

Running one v2 relation from step k (ORACLE or recorded pointing) arrives with the relations (M1-M4); the replay
is never a scored run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "src" / "gpt_field"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _obs_sha(observation):
    return hashlib.sha256(json.dumps(observation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_trace(run):
    reset, steps = None, []
    with open(os.path.join(run, "trace.jsonl"), "rb") as stream:
        for line in stream:
            head = line[:24]
            if head.startswith(b'{"event": "executed"'):
                row = json.loads(line)
                status = row.get("sim_status") or {}
                steps.append({"step": row["step"], "action": row["action"], "next_state": row["next_state"],
                              "observation_sha256": _obs_sha(row["observation"]) if row.get("observation") else None,
                              "libero_success": status.get("libero_success", status.get("success")),
                              "forbidden": ((status.get("collision") or {}).get("counts") or {}).get(
                                  "forbidden_contact_points")})
            elif head.startswith(b'{"event": "reset"'):
                reset = json.loads(line)["observation"]
    steps.sort(key=lambda s: s["step"])
    if reset is None:
        raise ValueError("trace has no reset event")
    return reset, steps


def image_differences(run, replayed):
    """Pixel-level differences of the mismatching observations (images; the state is compared separately)."""
    import base64
    import io
    import numpy as np
    from PIL import Image

    def decode(url):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"), np.int16)
    out = {}
    with open(os.path.join(run, "trace.jsonl"), "rb") as stream:
        for line in stream:
            if not line.startswith(b'{"event": "executed"'):
                continue
            row = json.loads(line)
            if row["step"] not in replayed:
                continue
            item = {}
            for key in ("observation/image", "observation/wrist_image"):
                a, b = decode(row["observation"][key]), decode(replayed[row["step"]][key])
                diff = np.abs(a - b).max(axis=2)
                item[key] = {"pixels_differing": int((diff > 0).sum()), "max_abs": int(diff.max())}
            others = sorted(k for k in row["observation"] if k not in ("observation/image", "observation/wrist_image")
                            and row["observation"][k] != replayed[row["step"]][k])
            item["non_image_fields_differing"] = others
            out[str(row["step"])] = item
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--to-step", type=int, default=None, help="replay executed steps 0..K-1 (default: all)")
    ap.add_argument("--save-state", action="store_true")
    ap.add_argument("--no-truth", action="store_true")
    args = ap.parse_args(argv)
    began = time.monotonic()
    run, out = os.path.abspath(args.run_dir), os.path.abspath(args.out_dir)
    if os.path.exists(out):
        ap.error("out_dir exists; choose a fresh directory")
    os.makedirs(out)
    manifest = json.load(open(os.path.join(run, "manifest.json")))["configuration"]
    simc = manifest["simulator"]
    is_v2 = manifest.get("method") == "agent_speedup_v2"
    pkg = os.path.join(out, "pkg")
    shutil.copytree(os.path.join(run, "frozen_source"), os.path.join(pkg, "gpt_field"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"))
    sys.path.insert(0, pkg)
    import numpy as np
    from gpt_field.libero import LiberoConfig, _LiberoWorker
    contacts = _load("agent_speedup_v2_contacts_current", CURRENT / "agent_speedup_v2_contacts.py")
    truth = _load("agent_speedup_v2_truth_current", CURRENT / "agent_speedup_v2_truth.py")
    reset_obs, steps = read_trace(run)
    if args.to_step is not None:
        if not 0 <= args.to_step <= len(steps):
            ap.error(f"--to-step must be in [0, {len(steps)}]")
        steps = steps[:args.to_step]
    cfg = LiberoConfig(suite=simc["suite"], seed=int(simc["seed"]), max_steps=None,
                       settle_steps=int(simc.get("settle_steps", 10)), libero_root=simc["libero_root"],
                       include_arm_joints=bool(simc.get("include_arm_joints", False)),
                       render_size=int(simc["render_size"]), image_size=int(simc["image_size"]),
                       collision_check_mode=simc.get("collision_check_mode", "none"),
                       stop_on_collision=bool(simc.get("stop_on_collision", False)), stop_on_success=False)

    if is_v2:
        from gpt_field.agent_speedup_v2_sim import V2LiberoWorker
        worker = V2LiberoWorker(cfg, manifest.get("simulator_monitor"))

        def generic_points():
            return worker.generic.forbidden_points()

        def generic_snapshot():
            return worker.generic.snapshot()
    else:
        class ChainedReplayWorker(_LiberoWorker):
            """The recorded v1 worker plus the current generic monitor chained on top (record-only)."""

            def reset(self, task_id, episode_index):
                response = super().reset(task_id, episode_index)       # attaches the v1 monitor if configured
                self.generic = contacts.GenericContactMonitor(self.env, goal_aware=True)
                if getattr(self, "_collision_monitor", None) is None:  # the v1 monitor already refreshed
                    contacts.forward_keeping_warmstart(self.env.sim)
                self.generic.inspect(phase="initial_state")
                self.generic.attach()
                return response

            def step(self, action):
                self.generic.begin_action(self.status["steps"])
                response = super().step(action)
                chained = getattr(self, "_collision_monitor", None) is not None
                self.generic.finish_action(refresh=not chained)
                return response

        worker = ChainedReplayWorker(cfg)

        def generic_points():
            return worker.generic.forbidden_points()

        def generic_snapshot():
            return worker.generic.snapshot()

    response = worker.reset(int(manifest["task_id"]), int(manifest["episode_index"]))
    reset_sha, recorded_reset_sha = _obs_sha(response["observation"]), _obs_sha(reset_obs)

    def v1_points(status):
        record = status.get("collision_v1_record_only") if is_v2 else status.get("collision")
        return ((record or {}).get("counts") or {}).get("forbidden_contact_points")

    max_dev, obs_mismatch, success_mismatch, forbidden_mismatch = 0.0, [], [], []
    v1_steps, generic_steps, mismatch_obs = [], [], {}
    prev_v1, prev_generic = v1_points(response["status"]), generic_points()
    window = truth.Window(k=truth.WINDOW_STEPS)
    scene = truth.Scene(worker.env) if not args.no_truth else None
    targets = truth.window_targets(scene) if scene is not None else ([], [])
    status = response["status"]
    for index, s in enumerate(steps):
        response = worker.step(s["action"])
        status = response["status"]
        dev = max(abs(a - b) for a, b in zip(response["observation"]["observation/state"], s["next_state"]))
        max_dev = max(max_dev, dev)
        if s["observation_sha256"] is not None and _obs_sha(response["observation"]) != s["observation_sha256"]:
            obs_mismatch.append(s["step"])
            if len(mismatch_obs) < 20:
                mismatch_obs[s["step"]] = response["observation"]
        ls = bool(status.get("libero_success", status.get("success")))
        if s["libero_success"] is not None and ls != bool(s["libero_success"]):
            success_mismatch.append(s["step"])
        recorded_forbidden = s["forbidden"]
        replayed_forbidden = (status.get("collision") or {}).get("counts", {}).get("forbidden_contact_points")
        if recorded_forbidden is not None and replayed_forbidden != recorded_forbidden:
            forbidden_mismatch.append(s["step"])
        current_v1, current_generic = v1_points(status), generic_points()
        if current_v1 is not None and prev_v1 is not None and current_v1 > prev_v1:
            v1_steps.append(s["step"])
        if current_generic > prev_generic:
            generic_steps.append(s["step"])
        prev_v1, prev_generic = current_v1, current_generic
        if scene is not None and index >= len(steps) - window.k:
            window.update(scene, *targets)
    v1s, gs = set(v1_steps), set(generic_steps)
    gen = generic_snapshot()
    pixel_diffs = image_differences(run, mismatch_obs) if mismatch_obs else {}
    v1_record = (status.get("collision_v1_record_only") if is_v2 else status.get("collision")) or {}
    # Physics is bitwise (state8, success flags, contact counts); offscreen EGL renders are not always: a few
    # pixels can differ by 1 grey between replays of the same state (measured in M0), so image identity is
    # reported separately with its pixel statistics.
    bitwise = (reset_sha == recorded_reset_sha and max_dev == 0.0 and not success_mismatch
               and not forbidden_mismatch)
    render_max_abs = max((v["max_abs"] for item in pixel_diffs.values() for k, v in item.items()
                          if k.startswith("observation/")), default=0)
    report = {
        "run_dir": run, "method": manifest.get("method"), "suite": simc["suite"], "task_id": manifest["task_id"],
        "episode_index": manifest["episode_index"], "replayed_steps": len(steps),
        "to_step": args.to_step, "bitwise": bitwise, "bitwise_definition": "physics: reset observation sha256, "
        "state8, libero_success and forbidden-contact counts identical at every step",
        "render_exact": not obs_mismatch, "render_mismatch_max_abs_grey": render_max_abs,
        "reset_observation_sha256": reset_sha, "reset_matches_trace": reset_sha == recorded_reset_sha,
        "max_abs_state8_deviation": max_dev, "observation_sha_mismatch_steps": obs_mismatch[:20],
        "observation_sha_mismatches": len(obs_mismatch), "observation_mismatch_pixels": pixel_diffs,
        "libero_success_mismatch_steps": success_mismatch[:20],
        "forbidden_count_mismatch_steps": forbidden_mismatch[:20],
        "libero_success_at_end": bool(status.get("libero_success", status.get("success"))),
        "monitor_comparison": {
            "v1_monitor": v1_record.get("mode"), "v1_first_event_step": (v1_record.get("first_event") or {}).get(
                "action_index"),
            "generic_first_event_step": (gen.get("first_event") or {}).get("action_index"),
            "v1_forbidden_steps": len(v1s), "generic_forbidden_steps": len(gs), "both": len(v1s & gs),
            "v1_only": sorted(v1s - gs)[:40], "v1_only_n": len(v1s - gs),
            "generic_only": sorted(gs - v1s)[:40], "generic_only_n": len(gs - v1s)},
        "generic_first_by_class": {k: v.get("action_index") for k, v in (gen.get("first_by_class") or {}).items()},
        "generic_counts": gen.get("counts"), "generic_held": gen.get("held_objects"),
        "contacts_module_sha256": hashlib.sha256((CURRENT / "agent_speedup_v2_contacts.py").read_bytes()).hexdigest(),
        "diagnosis_only": True}
    if scene is not None:
        report["truth_at_step"] = truth.evaluate_goal(scene, window)
    if args.save_state:
        D = worker.env.sim.data._data
        np.savez(os.path.join(out, f"state_{len(steps)}.npz"), qpos=D.qpos, qvel=D.qvel, act=D.act, ctrl=D.ctrl,
                 qacc_warmstart=D.qacc_warmstart, time=D.time,
                 gripper_action=worker.env.robots[0].gripper.current_action)
        report["saved_state"] = f"state_{len(steps)}.npz"
    report["wall_s"] = round(time.monotonic() - began, 1)
    with open(os.path.join(out, "replay.json"), "w") as stream:
        json.dump(truth.jsonable(report), stream, indent=1, allow_nan=False)
    print(json.dumps({k: report[k] for k in ("replayed_steps", "bitwise", "max_abs_state8_deviation",
                                             "observation_sha_mismatches", "render_mismatch_max_abs_grey", "libero_success_at_end", "wall_s")}
                     | {"monitors": report["monitor_comparison"]}), flush=True)
    worker.close()
    return 0 if bitwise else 1


if __name__ == "__main__":
    raise SystemExit(main())
