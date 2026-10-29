"""Measure frozen reset observation sha256 values for the agent_SpeedUp_v2 task registry (diagnosis only).

Each (key, episode) is reset in a FRESH V2 worker process (one _LiberoWorker per process) through the JSONL
route the runner uses (V2SimProcess), with the runner's simulator configuration (seed 7, settle 10, 512/512,
state8). The hash recipe is v1's: sha256 of json.dumps(observation.to_dict(), sort_keys=True,
separators=(",", ":")). With --repeat 2 every key is measured in two fresh processes. Zero model calls.

usage: MUJOCO_EGL_DEVICE_ID=2 python scripts/agent_speedup_v2_measure_resets.py --out OUT.json [--keys k1,k2]
       [--episodes 0] [--repeat 1]
Run it with the PROJECT venv; the worker uses --sim-python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt_field.agent_speedup_v2_sim import V2SimProcess, initial_observation_sha256  # noqa: E402
from gpt_field.agent_speedup_v2_tasks import V2_TASK_KEYS, build_v2_sim_config, resolve_v2_task  # noqa: E402


def measure(key, episode, sim_python, libero_root):
    spec = resolve_v2_task(key)
    config, monitor = build_v2_sim_config(key, sim_python=sim_python, libero_root=libero_root)
    began = time.monotonic()
    sim = V2SimProcess(config, monitor_options=monitor)
    try:
        sim.start()
        tasks = sim.list_tasks()
        official = [t["prompt"] for t in tasks if t["task_id"] == spec.task_id]
        obs = sim.reset(spec.task_id, episode)
        sha = initial_observation_sha256(obs.to_dict())
        return {"key": key, "suite": spec.suite, "task_id": spec.task_id, "episode": episode, "sha256": sha,
                "official_prompt": official[0] if len(official) == 1 else None,
                "prompt_matches_registry": official == [spec.prompt], "observation_prompt": obs.prompt,
                "initial_state8": list(obs.state), "collision_mode": sim.status["collision"]["mode"],
                "chained_v1": "collision_v1_record_only" in sim.status,
                "frozen": spec.frozen_reset_sha256.get(episode),
                "matches_frozen": spec.frozen_reset_sha256.get(episode) == sha if episode in spec.frozen_reset_sha256
                else None, "wall_s": round(time.monotonic() - began, 1)}
    finally:
        sim.close()


def main(argv=None):
    from gpt_field.agent_speedup_v2_tasks import DEFAULT_LIBERO_ROOT, DEFAULT_SIM_PYTHON
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keys", default=",".join(V2_TASK_KEYS))
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--libero-root", default=DEFAULT_LIBERO_ROOT)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("output exists; choose a fresh path")
    rows = []
    for key in args.keys.split(","):
        for episode in (int(e) for e in args.episodes.split(",")):
            for repeat in range(args.repeat):
                row = {**measure(key, episode, args.sim_python, args.libero_root), "repeat": repeat}
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("key", "episode", "repeat", "sha256", "matches_frozen",
                                                        "prompt_matches_registry", "wall_s")}), flush=True)
    stable = {}
    for row in rows:
        stable.setdefault((row["key"], row["episode"]), set()).add(row["sha256"])
    result = {"rows": rows, "deterministic_across_processes": all(len(v) == 1 for v in stable.values()),
              "all_frozen_match": all(r["matches_frozen"] is not False for r in rows),
              "all_prompts_match": all(r["prompt_matches_registry"] for r in rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
    return 0 if result["all_frozen_match"] and result["all_prompts_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
