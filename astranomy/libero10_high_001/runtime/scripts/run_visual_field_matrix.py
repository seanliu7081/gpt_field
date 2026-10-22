"""Execute the frozen 16-trial matrix sequentially; never impose an episode cap."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from gpt_field.experiment import source_manifest, write_json

ROOT = Path(__file__).resolve().parents[1]


def summarize(matrix, destination):
    trials = []
    for trial in matrix["trials"]:
        path = Path(trial["output"]) / "summary.json"
        summary = json.loads(path.read_text()) if path.exists() else None
        trials.append({"id": trial["id"], "pair_id": trial["pair_id"], "scene": trial["scene"],
            "shot": trial["shot"], "controller": trial["controller"], "output": trial["output"],
            "status": summary["status"] if summary else "not_completed",
            "success": summary["task_success"] if summary else None,
            "time_without_video_s": summary["control_without_video_s"] if summary else None,
            "requests": summary["budget"]["api_calls"] if summary else None,
            "steps": summary["steps"] if summary else None,
            "attempt_episodes": len(summary["attempts"]) if summary else None,
            "recovery_stall_events": len(summary["recovery_events"]) if summary else None,
            "field_actions": summary["field_actions"] if summary else None,
            "completion_time_eligible": bool(summary and summary["completion_time_eligible"])})
    pairs = []
    for pair in matrix["pairs"]:
        arms = {t["controller"]: t for t in trials if t["pair_id"] == pair["id"]}
        eligible = all(t["completion_time_eligible"] for t in arms.values()) and len(arms) == 2
        initial_equal = None
        input_paths = [Path(t["output"]) / "initial_observation.json" for t in arms.values()]
        if all(p.exists() for p in input_paths):
            initial_equal = len({hashlib.sha256(p.read_bytes()).hexdigest() for p in input_paths}) == 1
        eligible = eligible and initial_equal is True
        h, d = arms["hybrid"]["time_without_video_s"], arms["direct"]["time_without_video_s"]
        pairs.append({"id": pair["id"], "scene": pair["scene"], "shot": pair["shot"],
            "initial_observations_equal": initial_equal, "completion_comparison_eligible": eligible,
            "direct_minus_hybrid_s": d - h if eligible else None,
            "hybrid_time_reduction_percent": 100 * (d - h) / d if eligible and d else None})
    result = {"updated_utc": datetime.now(timezone.utc).isoformat(), "trials": trials, "pairs": pairs,
        "notes": ["video overhead excluded; all local visual/planning work included",
                  "failure/incomplete times are attempt durations, never completion speedups",
                  "Direct executes raw five actions and shares local visual feedback",
                  "one paired rollout per condition is exploratory, not a population reliability estimate",
                  "attempt and recovery counters are observable proxies, not oracle physical-event counts"]}
    write_json(destination / "comparison.json", result)
    return result


def run(matrix_path, *, report_only=False):
    matrix_path = Path(matrix_path).resolve()
    matrix = json.loads(matrix_path.read_text())
    output = matrix_path.parent
    if report_only:
        return summarize(matrix, output)
    for entry in matrix["frozen_input_artifacts"]:
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Frozen experiment input has changed")
    software = source_manifest()
    runners = {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() for name in
               ("run_visual_field_matrix.py", "evaluate_visual_field_subscription.py")}
    frozen_manifest = output / "execution_source.json"
    identity = {"software": software, "runners": runners}
    if frozen_manifest.exists():
        if json.loads(frozen_manifest.read_text()) != identity:
            raise ValueError("Execution source differs from the started matrix; preserve it as a separate protocol")
    else:
        write_json(frozen_manifest, identity)
    lookup = {trial["id"]: trial for trial in matrix["trials"]}
    scenes = {scene["id"]: scene for scene in matrix["scenes"]}
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    completed = []
    for pair in matrix["pairs"]:
        expected_initial = None
        for trial_id in pair["execution_order"]:
            trial = lookup[trial_id]
            trial_output = Path(trial["output"])
            summary_file = trial_output / "summary.json"
            if summary_file.exists():
                summary = json.loads(summary_file.read_text())
                if summary["status"] not in ("success", "collision_failure"):
                    raise RuntimeError("Previous incomplete trial retained; do not silently retry or skip it")
                completed.append(trial_id)
                expected_initial = trial_output / "initial_observation.json"
                continue
            if trial_output.exists():
                raise RuntimeError("Unfinished trial already exists; inspect it rather than overwrite")
            if source_manifest() != software:
                raise RuntimeError("Policy code changed during the comparison")
            if any(hashlib.sha256((ROOT / "scripts" / name).read_bytes()).hexdigest() != value
                   for name, value in runners.items()):
                raise RuntimeError("Runner code changed during the comparison")
            for key in ("official_bddl", "official_initial_state"):
                artifact = scenes[trial["scene"]][key]
                if hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
                    raise RuntimeError("Official simulator input changed from the frozen matrix")
            args = [sys.executable, *trial["runner_arguments"]]
            if expected_initial is not None:
                args += ["--expected-initial", str(expected_initial)]
            write_json(output / "run_state.json", {"status": "running", "current_trial": trial_id,
                "completed": completed, "total": len(lookup), "pid": os.getpid()})
            print(json.dumps({"event": "trial_start", "trial": trial_id}), flush=True)
            with (output / (trial_id + ".log")).open("w") as log:
                process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    code = process.wait()
                except KeyboardInterrupt:
                    process.send_signal(2)
                    process.wait()
                    summarize(matrix, output)
                    raise
            result = summarize(matrix, output)
            if code != 0:
                write_json(output / "run_state.json", {"status": "interrupted_or_error", "current_trial": trial_id,
                    "completed": completed, "exit_code": code})
                return result
            completed.append(trial_id)
            expected_initial = trial_output / "initial_observation.json"
            print(json.dumps({"event": "trial_finished", "trial": trial_id, "completed": len(completed)}), flush=True)
    write_json(output / "run_state.json", {"status": "complete", "completed": completed, "total": len(lookup)})
    return summarize(matrix, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    run(args.matrix, report_only=args.report_only)
