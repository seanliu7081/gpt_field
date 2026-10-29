"""Non-perturbation check of the v2 worker (TEST TOOL; zero model calls; spec §6.4; outside pytest).

The generic contact monitor and the truth window are evaluator-only: a V2LiberoWorker must follow EXACTLY the
physics of the base worker it wraps. This runs the same contact-rich scripted action sequence (descend onto the
table, a smoothed random walk, gripper toggles; seeded, identical in both runs) in two fresh worker processes:

- ``plain``: gpt_field.libero._LiberoWorker with the registry's base collision mode ('none', or v1's own
  task_manipulation_v1 monitor on bowl_plate / drawer_bowl);
- ``v2``: gpt_field.agent_speedup_v2_sim.V2LiberoWorker with the registry's monitor options;

and compares every row (the reset plus every step): state8, the observation sha256 (images included) and the
full MuJoCo state sha256 (qpos, qvel, act, ctrl, qacc_warmstart, time, gripper action). Exit 0 iff all rows are
bitwise identical. It also reports the v2 run's contact counts, so the sequence is known to be contact-rich.

usage: MUJOCO_EGL_DEVICE_ID=2 /workspace/venvs/libero-plus/bin/python -B \
           scripts/agent_speedup_v2_nonperturbation_check.py <fresh_out_dir> [--key libero_10_t2] [--steps 250]
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


def actions(n, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    out, previous, grip = [], np.zeros(6), -1.0
    for t in range(n):
        noise = rng.normal(0, 1, 6) * np.array([0.6, 0.6, 0.4, 0.25, 0.25, 0.4])
        previous = 0.85 * previous + 0.45 * noise
        a = previous.copy()
        a[2] = -0.8 if t < 40 else a[2] - 0.15
        if t % 30 == 0 and t >= 40:
            grip = -grip
        out.append([float(x) for x in np.clip(a, -1, 1)] + [grip])
    return out


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _state_sha(env):
    import numpy as np
    D = env.sim.data._data
    h = hashlib.sha256()
    for array in (D.qpos, D.qvel, D.act, D.ctrl, D.qacc_warmstart):
        h.update(np.ascontiguousarray(array).tobytes())
    h.update(np.float64(D.time).tobytes())
    h.update(np.ascontiguousarray(env.robots[0].gripper.current_action).tobytes())
    return h.hexdigest()


def single(mode, key, steps, seed):
    from dataclasses import replace
    from gpt_field.agent_speedup_v2_tasks import build_v2_sim_config, resolve_v2_task
    spec = resolve_v2_task(key)
    config, monitor = build_v2_sim_config(key)
    config = replace(config, stop_on_success=False)
    if mode == "plain":
        from gpt_field.libero import _LiberoWorker
        worker = _LiberoWorker(config)
    else:
        from gpt_field.agent_speedup_v2_sim import V2LiberoWorker
        worker = V2LiberoWorker(config, monitor)
    rows = []
    try:
        response = worker.reset(spec.task_id, 0)
        rows.append({"i": -1, "state8": response["observation"]["observation/state"],
                     "obs_sha": _sha(response["observation"]), "state_sha": _state_sha(worker.env)})
        for index, action in enumerate(actions(steps, seed)):
            response = worker.step(action)
            rows.append({"i": index, "state8": response["observation"]["observation/state"],
                         "obs_sha": _sha(response["observation"]), "state_sha": _state_sha(worker.env)})
        collision = response["status"].get("collision")
        counts = (collision or {}).get("counts") if mode == "v2" else None
    finally:
        worker.close()
    return {"mode": mode, "key": key, "rows": rows, "v2_counts": counts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", nargs="?", type=Path)
    parser.add_argument("--key", default="libero_10_t2")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--single", choices=("plain", "v2"))
    args = parser.parse_args(argv)
    if args.single:
        print("ROW " + json.dumps(single(args.single, args.key, args.steps, args.seed)), flush=True)
        return 0
    if args.out is None or args.out.exists():
        parser.error("a fresh output directory is required")
    args.out.mkdir(parents=True)
    runs = {}
    for mode in ("plain", "v2"):
        proc = subprocess.run([sys.executable, "-B", __file__, "--single", mode, "--key", args.key, "--steps",
                               str(args.steps), "--seed", str(args.seed)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(proc.stderr[-3000:])
        runs[mode] = json.loads([l for l in proc.stdout.splitlines() if l.startswith("ROW ")][-1][4:])
    plain, v2 = runs["plain"]["rows"], runs["v2"]["rows"]
    differing = {field: [a["i"] for a, b in zip(plain, v2) if a[field] != b[field]]
                 for field in ("state8", "obs_sha", "state_sha")}
    counts = runs["v2"]["v2_counts"] or {}
    contact_points = {k: v for k, v in counts.items() if k.startswith("points_") and v}
    report = {"key": args.key, "steps": args.steps, "seed": args.seed, "rows_compared": min(len(plain), len(v2)),
              "rows": [len(plain), len(v2)],
              "differing_rows": {k: len(v) for k, v in differing.items()},
              "first_differing": {k: v[:5] for k, v in differing.items()},
              "bitwise_identical": len(plain) == len(v2) and not any(differing.values()),
              "v2_contact_points": contact_points, "v2_physics_substeps": counts.get("physics_substeps"),
              "contacts_module_sha256": hashlib.sha256(
                  (ROOT / "src/gpt_field/agent_speedup_v2_contacts.py").read_bytes()).hexdigest(),
              "sim_module_sha256": hashlib.sha256(
                  (ROOT / "src/gpt_field/agent_speedup_v2_sim.py").read_bytes()).hexdigest(),
              "diagnosis_only": True}
    (args.out / "nonperturbation.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report), flush=True)
    return 0 if report["bitwise_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
