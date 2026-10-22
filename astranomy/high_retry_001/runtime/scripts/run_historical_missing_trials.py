"""Fill only four missing historical zero-shot cells; no implicit retries or caps.

--prepare-only freezes commands, source and relayout bytes without credentials,
models or simulator calls. A subsequent invocation runs that prepared queue.
--report-only recomputes adjusted accounting from existing immutable evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

from gpt_field.experiment import source_manifest, write_json

ROOT = Path(__file__).resolve().parents[1]
VERSION = "historical-missing-zero-shot-v1"
RUNNER = "scripts/evaluate_bowl_hybrid_subscription.py"
QUEUE_SCRIPT = "scripts/run_historical_missing_trials.py"
DEFAULT_OUTPUT = "runs/historical_zero_shot_missing_001"
RELAYOUT_SOURCE = "runs/bowl_relayout_distance_medium_one_demo_002/initial_state_override.npy"
RELAYOUT_SHA256 = "65912e2fa0c96a510f84241159c3baa9859e3f68d8595ae03e801e7b94febaf7"
INITIAL_HASHES = {
    "original_bowl": "be2df53ab9e9364892997b0e139692309206335e756502cfa88f788b5b5d8f4a",
    "relayout_bowl": "2ee557d50bf0a2cc33d82b918b25549e5e7329d154d019855807030bbe3d3fc5",
    "original_drawer": "b1e21ae923643f4d51ed44fc0051b1b359f8ac36fc6f65c94e9dad9477a482fd",
}
TRIALS = (
    ("original_bowl_zero_direct", "original_bowl", "bowl_plate", "direct"),
    ("relayout_bowl_zero_hybrid", "relayout_bowl", "bowl_plate", "hybrid"),
    ("relayout_bowl_zero_direct", "relayout_bowl", "bowl_plate", "direct"),
    ("original_drawer_zero_direct", "original_drawer", "drawer_bowl", "direct"),
)
HISTORICAL = (
    ("original_bowl_zero_hybrid", "original_bowl", "bowl_plate", "runs/distance_field_medium_task_001"),
    ("original_drawer_zero_hybrid", "original_drawer", "drawer_bowl", "runs/drawer_distance_medium_task_001"),
)
OBSERVATION_KEYS = {"observation/image", "observation/wrist_image", "observation/state", "prompt"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_observation_hash(observation):
    if not isinstance(observation, dict) or set(observation) != OBSERVATION_KEYS:
        raise ValueError("Historical observation must have exactly RGB, wrist RGB, 8D state and prompt")
    state = observation["observation/state"]
    if not isinstance(state, list) or len(state) != 8 or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in state):
        raise ValueError("Historical observation state must contain eight finite numbers")
    return sha256(json.dumps(observation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_trials(output, python_executable):
    output = Path(output).resolve()
    result = []
    for trial_id, scene, task, controller in TRIALS:
        command = [python_executable, str(ROOT / RUNNER), "--output", str(output / trial_id),
                   "--task", task, "--episode-index", "0", "--reasoning-effort", "medium",
                   "--expected-initial-observation-sha256", INITIAL_HASHES[scene]]
        command += (["--distance-field-control"] if controller == "hybrid" else
                    ["--historical-zero-shot-direct", "--controller-mode", "direct"])
        if scene == "relayout_bowl":
            command += ["--initial-state-override", str(output / "inputs/initial_state_override.npy")]
        result.append({"id": trial_id, "scene": scene, "task_key": task, "controller": controller,
                       "output": str(output / trial_id), "command": command,
                       "expected_initial_observation_sha256": INITIAL_HASHES[scene]})
    return result


def _source_identity():
    return {"software": source_manifest(), "scripts": {
        name: sha256((ROOT / name).read_bytes()) for name in (RUNNER, QUEUE_SCRIPT)}}


def _frozen_file_record(output, relative, data):
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return {"artifact": relative, "sha256": sha256(data), "byte_count": len(data)}


def _historical_inputs():
    records = []
    for trial_id, scene, task, directory in HISTORICAL:
        source = ROOT / directory
        summary_bytes = (source / "summary.json").read_bytes()
        summary = json.loads(summary_bytes)
        with (source / "trace.jsonl").open("rb") as handle:
            reset_bytes = handle.readline()
        reset = json.loads(reset_bytes)
        if reset.get("event") != "reset":
            raise ValueError("Historical trace must begin with reset")
        observation = reset["observation"]
        if canonical_observation_hash(observation) != INITIAL_HASHES[scene]:
            raise ValueError("Historical reset differs from the authorized initial observation")
        records.append(({"id": trial_id, "scene": scene, "task_key": task, "controller": "hybrid",
                         "historical": True, "source_output": str(source),
                         "source_summary_sha256": sha256(summary_bytes),
                         "source_reset_line_sha256": sha256(reset_bytes),
                         "expected_initial_observation_sha256": INITIAL_HASHES[scene]},
                        summary_bytes, observation))
    return records


def prepare(output):
    """Freeze inputs and code only; does not import a simulator or obtain credentials."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preparation requires a fresh directory; never overwrite existing evidence")
    relayout_bytes = (ROOT / RELAYOUT_SOURCE).read_bytes()
    if sha256(relayout_bytes) != RELAYOUT_SHA256:
        raise ValueError("Authorized relayout input hash mismatch")
    historical = _historical_inputs()
    identity = _source_identity()
    source_bytes = {"src/gpt_field/" + name: (ROOT / "src/gpt_field" / name).read_bytes()
                    for name in identity["software"]["source_sha256"]}
    source_bytes.update({name: (ROOT / name).read_bytes() for name in identity["scripts"]})
    for name, data in source_bytes.items():
        expected = (identity["scripts"][name] if name in identity["scripts"] else
                    identity["software"]["source_sha256"][Path(name).name])
        if sha256(data) != expected:
            raise RuntimeError("Source changed during preparation")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = [_frozen_file_record(output, "inputs/initial_state_override.npy", relayout_bytes)]
    for name, data in source_bytes.items():
        artifacts.append(_frozen_file_record(output, "frozen_source/" + name, data))
    baselines = []
    for record, summary_bytes, observation in historical:
        prefix = "historical/" + record["id"]
        artifacts.append(_frozen_file_record(output, prefix + "/summary.json", summary_bytes))
        observation_bytes = (json.dumps(observation, sort_keys=True, separators=(",", ":")) + "\n").encode()
        artifacts.append(_frozen_file_record(output, prefix + "/initial_observation.json", observation_bytes))
        baselines.append({**record, "output": str(output / prefix)})
    manifest = {"version": VERSION, "created_utc": utc_now(), "output": str(output),
        "python_executable": sys.executable, "scope": "Only four missing historical zero-shot cells",
        "settings": {"model": "gpt-6-astra", "reasoning_effort": "medium",
            "billing": "chatgpt_subscription", "api_key_calls_allowed": 0,
            "demonstration": None, "include_arm_joints": False, "collision_check_mode": "none",
            "direct_action_steps": 5, "feedback_history": "block_summary", "seed": 7, "episode_index": 0,
            "max_steps": None, "max_wall_time_s": None, "max_model_calls": None,
            "stop_on_model_complete": False, "completion": "LIBERO success only",
            "time_metric": "control_elapsed_s - timing_components.video_render_s"},
        "relayout_source": {"path": str(ROOT / RELAYOUT_SOURCE), "sha256": RELAYOUT_SHA256},
        "execution_source": identity, "frozen_artifacts": artifacts,
        "trials": build_trials(output, sys.executable), "historical_baselines": baselines,
        "notes": ["Original-scene Hybrid observations were collected earlier; supplemental Direct runs are retrospective comparisons.",
                  "Drawer historical Hybrid ended by user stop; its duration is not a completion time.",
                  "No failed or incomplete attempt is eligible for a completion speed comparison.",
                  "SIGTERM and SIGINT request child SIGINT cleanup; cleanup has no timeout."]}
    write_json(output / "manifest.json", manifest)
    write_json(output / "run_state.json", {"status": "prepared", "completed": [], "total": 4,
                                           "updated_utc": utc_now(), "model_calls_launched": 0})
    summarize(manifest, output)
    return manifest


def _validate_manifest(output, manifest, *, check_current_source):
    if manifest.get("version") != VERSION or manifest.get("output") != str(output):
        raise ValueError("Unexpected queue identity or relocated manifest")
    if manifest.get("trials") != build_trials(output, manifest["python_executable"]):
        raise ValueError("Queue commands differ from the four authorized trials")
    for artifact in manifest["frozen_artifacts"]:
        path = output / artifact["artifact"]
        if not path.resolve().is_relative_to(output):
            raise ValueError("Frozen artifact lies outside the queue")
        if sha256(path.read_bytes()) != artifact["sha256"]:
            raise ValueError("Frozen artifact hash mismatch: " + artifact["artifact"])
    if sha256((output / "inputs/initial_state_override.npy").read_bytes()) != RELAYOUT_SHA256:
        raise ValueError("Frozen relayout differs from authorized input")
    if check_current_source and _source_identity() != manifest["execution_source"]:
        raise RuntimeError("Source changed after preparation; preserve this queue and prepare a fresh one")


def _result(trial):
    output = Path(trial["output"])
    summary_file = output / "summary.json"
    row = {key: trial[key] for key in ("id", "scene", "task_key", "controller", "output")}
    row.update(historical=bool(trial.get("historical")), status="not_started", success=False,
               completion_time_eligible=False, control_elapsed_s=None, video_render_s=None,
               time_without_video_s=None, requests=None, steps=None, initial_observation_matches=None,
               issues=[])
    if not summary_file.exists():
        if output.exists():
            row["status"] = "incomplete_no_summary"
        return row
    data = json.loads(summary_file.read_text())
    row.update(status=data.get("status", "unknown"), requests=data.get("api_calls"), steps=data.get("steps"),
               summary_sha256=sha256(summary_file.read_bytes()))
    raw, video = data.get("control_elapsed_s"), data.get("timing_components", {}).get("video_render_s")
    valid_time = all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                     for x in (raw, video)) and 0 <= video <= raw
    if valid_time:
        row.update(control_elapsed_s=raw, video_render_s=video, time_without_video_s=raw-video)
    else:
        row["issues"].append("Missing or invalid video-excluded timing")
    observation_path = output / "initial_observation.json"
    if observation_path.exists():
        try:
            actual = canonical_observation_hash(json.loads(observation_path.read_text()))
            row["initial_observation_sha256"] = actual
            row["initial_observation_matches"] = actual == trial["expected_initial_observation_sha256"]
        except (ValueError, TypeError):
            row["initial_observation_matches"] = False
    if row["initial_observation_matches"] is not True:
        row["issues"].append("Initial observation not verified against the authorized hash")
    protocol_ok = (data.get("billing") == "chatgpt_subscription" and data.get("api_key_calls") == 0
        and data.get("reasoning_effort") == "medium" and data.get("stop_on_model_complete") is False
        and data.get("seed") == 7 and data.get("episode_index") == 0
        and data.get("controller_mode") == trial["controller"]
        and all(data.get(k) is None for k in ("max_policy_steps", "max_wall_time_s", "max_api_calls")))
    if not protocol_ok:
        row["issues"].append("Recorded core protocol does not match the authorized settings")
    row["success"] = (data.get("status") == "success" and data.get("evaluation_completed") is True
                      and data.get("libero_success") is True)
    row["completion_time_eligible"] = bool(row["success"] and valid_time and protocol_ok
                                             and row["initial_observation_matches"] is True)
    row["time_interpretation"] = "completion_time" if row["completion_time_eligible"] else "attempt_duration_only"
    return row


def summarize(manifest, output):
    rows = [_result(trial) for trial in manifest["trials"] + manifest["historical_baselines"]]
    pairs = []
    for scene in ("original_bowl", "relayout_bowl", "original_drawer"):
        arms = {row["controller"]: row for row in rows if row["scene"] == scene}
        hybrid, direct = arms["hybrid"], arms["direct"]
        initial_equal = (hybrid["initial_observation_matches"] is True
                         and direct["initial_observation_matches"] is True
                         and hybrid.get("initial_observation_sha256") == direct.get("initial_observation_sha256"))
        eligible = all(row["completion_time_eligible"] for row in arms.values()) and initial_equal
        h, d = hybrid["time_without_video_s"], direct["time_without_video_s"]
        pairs.append({"scene": scene, "hybrid_id": hybrid["id"], "direct_id": direct["id"],
            "comparison_type": "retrospective_hybrid_reference" if hybrid["historical"] else "fresh_matched_pair",
            "initial_observations_equal": initial_equal,
            "completion_comparison_eligible": eligible,
            "direct_minus_hybrid_s": d-h if eligible else None,
            "hybrid_time_reduction_percent": 100*(d-h)/d if eligible and d else None,
            "direct_over_hybrid_time_ratio": d/h if eligible and h else None})
    result = {"version": VERSION, "updated_utc": utc_now(), "trials": rows, "pairs": pairs,
        "time_metric": "control_elapsed_s - timing_components.video_render_s",
        "notes": ["Video-excluded values are adjusted accounting, not trials recorded with video disabled.",
                  "Failed/incomplete durations are not successful completion times or speedups.",
                  "Historical original-scene Hybrid runs predate these supplemental Direct trials.",
                  "This summary verifies reset hashes and core settings; full request/rollout audits remain separate.",
                  "One attempt per cell does not estimate population reliability."]}
    write_json(Path(output) / "comparison.json", result)
    return result


def run(output, *, prepare_only=False, report_only=False):
    output = Path(output).resolve()
    if prepare_only and report_only:
        raise ValueError("Choose prepare-only or report-only")
    if prepare_only:
        return prepare(output)
    if not output.exists():
        if report_only:
            raise FileNotFoundError("Report-only requires an existing prepared queue")
        manifest = prepare(output)
    else:
        manifest = json.loads((output / "manifest.json").read_text())
    _validate_manifest(output, manifest, check_current_source=not report_only)
    if report_only:
        return summarize(manifest, output)
    state = json.loads((output / "run_state.json").read_text())
    if state.get("status") != "prepared":
        raise RuntimeError("Only a fresh prepared queue can start; do not silently retry an interrupted trial")
    if any(Path(trial["output"]).exists() for trial in manifest["trials"]):
        raise FileExistsError("A trial output already exists; preserve it instead of overwriting or skipping it")
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    completed = []
    process = None
    current_trial = None
    interrupt_signal = None
    interrupt_forwarded = False

    def interrupt(signum, _frame):
        nonlocal interrupt_signal, interrupt_forwarded
        if interrupt_signal is None:
            interrupt_signal = signum
        # The old evaluator catches KeyboardInterrupt to save its evidence and
        # close the simulator. Sending raw SIGTERM would bypass that cleanup.
        if process is not None and process.poll() is None and not interrupt_forwarded:
            interrupt_forwarded = True
            process.send_signal(signal.SIGINT)

    def save_state(status, **extra):
        write_json(output / "run_state.json", {"status": status, "current_trial": current_trial,
            "completed": list(completed), "total": len(manifest["trials"]), "pid": os.getpid(),
            "updated_utc": utc_now(), "interrupt_signal": interrupt_signal, **extra})

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, interrupt)
    try:
        for trial in manifest["trials"]:
            if interrupt_signal is not None:
                break
            _validate_manifest(output, manifest, check_current_source=True)
            current_trial = trial["id"]
            save_state("running", child_pid=None)
            print(json.dumps({"event": "trial_start", "trial": current_trial}), flush=True)
            log_path = output / (current_trial + ".log")
            with log_path.open("x") as log:
                if interrupt_signal is not None:
                    break
                process = subprocess.Popen(trial["command"], cwd=ROOT, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                save_state("running", child_pid=process.pid)
                if interrupt_signal is not None:
                    interrupt(interrupt_signal, None)
                code = process.wait()  # No episode or cleanup timeout.
                process = None
            result = summarize(manifest, output)
            row = next(item for item in result["trials"] if item["id"] == current_trial)
            if interrupt_signal is not None:
                save_state("interrupted", child_exit_code=code)
                return result
            if code != 0 or not row["completion_time_eligible"]:
                save_state("incomplete_or_error", child_exit_code=code, trial_status=row["status"])
                return result
            completed.append(current_trial)
            save_state("between_trials", child_exit_code=code)
            print(json.dumps({"event": "trial_finished", "trial": current_trial}), flush=True)
        save_state("interrupted" if interrupt_signal is not None else "complete")
        return summarize(manifest, output)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            if not interrupt_forwarded:
                process.send_signal(signal.SIGINT)
            process.wait()  # Allow child evidence/video/simulator cleanup to finish.
        save_state("interrupted" if interrupt_signal is not None else "queue_error",
                   error={"type": type(exc).__name__, "message": str(exc)})
        summarize(manifest, output)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / DEFAULT_OUTPUT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.output, prepare_only=args.prepare_only, report_only=args.report_only)
    print(json.dumps({"output": str(args.output.resolve()), "mode": "prepare" if args.prepare_only else
                      "report" if args.report_only else "run", "version": result["version"]}), flush=True)
    if args.prepare_only or args.report_only:
        return 0
    state = json.loads((args.output / "run_state.json").read_text())
    return 0 if state["status"] == "complete" else 128 + state["interrupt_signal"] if state["interrupt_signal"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
