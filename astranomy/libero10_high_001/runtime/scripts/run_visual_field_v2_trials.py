"""Run only the four prepared zero-shot Hybrid v2 trials, serially and without caps."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

from prepare_visual_field_v2_trials import (
    ROOT, SCENES, VERSION, read_json, runner_arguments, sha256, write_json)

ALLOWED_TERMINAL = {"success", "collision_failure"}
PAUSED_TERMINAL = {"user_stop", "subscription_error", "infrastructure_error"}
POLICY_SETTINGS = {"model": "gpt-6-astra", "reasoning_effort": "medium", "billing": "chatgpt_subscription",
                   "method_version": "v2", "controller": "hybrid", "shots": 0,
                   "api_key_fallback": False, "include_arm_joints": True,
                   "collision_check_mode": "task_manipulation_v1", "model_completion_stopping": False,
                   "max_steps": None, "max_wall_time_s": None, "max_model_calls": None,
                   "automatic_stall_termination": False}


def validate_plan(plan, directory):
    if (plan.get("schema_version") != VERSION or plan.get("trial_count") != 4
            or plan.get("new_direct_trials") != 0 or plan.get("settings") != POLICY_SETTINGS
            or len(plan.get("trials", [])) != 4
            or [t["scene"] for t in plan["trials"]] != list(SCENES)
            or [s["id"] for s in plan.get("scenes", [])] != list(SCENES)):
        raise ValueError("Only the declared four zero-shot Hybrid v2 trials are authorized")
    for trial in plan["trials"]:
        if (trial.get("controller") != "hybrid" or trial.get("method_version") != "v2"
                or trial.get("shot") != "zero" or trial.get("demo") is not None
                or trial["seed"] != 7 or trial["episode_index"] != 0
                or trial["task"] != ("drawer_bowl" if trial["scene"] == "drawer_original" else "bowl_plate")
                or trial["runner_arguments"] != runner_arguments(trial)
                or Path(trial["output"]).resolve() != directory / "trials" / f"{trial['scene']}_zero_hybrid_v2"):
            raise ValueError("Trial command differs from the authorized protocol")
    return plan


def verify_inputs(plan):
    listed = {str(Path(x["path"]).resolve()): x["sha256"] for x in plan["frozen_input_artifacts"]}
    for path, digest in listed.items():
        if sha256(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    for scene, trial in zip(plan["scenes"], plan["trials"]):
        for name in ("official_bddl", "official_initial_state"):
            if sha256(scene[name]["path"]) != scene[name]["sha256"]:
                raise ValueError(f"Official simulator input changed: {scene['id']} / {name}")
        required = [(trial["expected_initial"], trial["expected_initial_sha256"]),
                    (trial["direct_reference"]["snapshot"], trial["direct_reference"]["summary_sha256"])]
        for key in ("initial_state", "bddl"):
            if trial.get(key) != scene.get(key):
                raise ValueError("Trial scene input differs")
            if trial.get(key):
                required.append((trial[key], scene[key + "_sha256"]))
        if trial["expected_initial"] != scene["expected_initial"]:
            raise ValueError("Trial reset reference differs")
        for path, digest in required:
            if listed.get(str(Path(path).resolve())) != digest:
                raise ValueError("Required input is absent from the frozen artifact manifest")


def current_sources(project):
    project = Path(project)
    files = list((project / "src/gpt_field").glob("*.py")) + list((project / "scripts").glob("*.py"))
    if not files:
        raise ValueError("No source files found to freeze")
    return {str(p.relative_to(project)): sha256(p) for p in sorted(files)}


def freeze_or_verify_sources(project, directory, matrix_path):
    """Freeze all policy modules and scripts; never update a frozen snapshot."""
    identity = {"source_sha256": current_sources(project), "matrix_sha256": sha256(matrix_path)}
    record = directory / "execution_source.json"
    if record.exists():
        if read_json(record) != identity:
            raise ValueError("Source or prepared matrix changed after execution freeze")
        for relative, digest in identity["source_sha256"].items():
            if sha256(directory / "frozen_source" / relative) != digest:
                raise ValueError("Frozen source snapshot changed")
        return identity
    if (directory / "frozen_source").exists():
        raise ValueError("Incomplete source freeze requires inspection; no implicit retry")
    for relative, digest in identity["source_sha256"].items():
        data = (Path(project) / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Source changed while freezing")
        target = directory / "frozen_source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    if current_sources(project) != identity["source_sha256"]:
        raise ValueError("Source changed while freezing")
    write_json(record, identity)
    return identity


def qualified_outcome(summary):
    status = summary.get("status")
    collision = summary.get("collision", {})
    if (summary.get("method_version") != "v2" or summary.get("controller") != "hybrid"
            or summary.get("shots") != 0):
        return "invalid_terminal_record"
    if status == "success":
        if not (summary.get("task_success") is True and summary.get("libero_success") is True
                and collision.get("detected") is False and summary.get("completion_time_eligible") is True
                and collision.get("enabled") is True and collision.get("mode") == "task_manipulation_v1"):
            return "invalid_terminal_record"
    elif status == "collision_failure":
        if (collision.get("detected") is not True or summary.get("task_success") is not False
                or collision.get("enabled") is not True or collision.get("mode") != "task_manipulation_v1"):
            return "invalid_terminal_record"
    elif status in PAUSED_TERMINAL:
        if summary.get("task_success") is True or summary.get("completion_time_eligible") is True:
            return "invalid_terminal_record"
    else:
        return "invalid_terminal_record"
    return status


def time_without_video(summary):
    elapsed = summary.get("control_elapsed_s")
    video = summary.get("timing_components", {}).get("video_render_s")
    if elapsed is None or video is None:
        return None
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in (elapsed, video)) or video > elapsed:
        raise ValueError("Invalid measured timing components")
    value = elapsed - video
    if summary.get("control_without_video_s") is not None and not math.isclose(value, summary["control_without_video_s"], abs_tol=1e-6):
        raise ValueError("Reported video-excluded timing is inconsistent")
    return value


def activation_counts(trial_dir, summary):
    """Count only acknowledged actions; planned-but-unexecuted actions are excluded."""
    trace = Path(trial_dir) / "trace.jsonl"
    if not trace.exists():
        return {"source": "summary_only", "local_servo_actions": summary.get("local_servo_actions"),
                "field_rejected_actions": summary.get("field_rejected_actions"),
                "bootstrap_actions": None, "attachment_gate_actions": None,
                "selection_status_counts": summary.get("field_selection_status_counts"),
                "limitation": "No trace available to independently count acknowledged selections."}
    decisions, acknowledged, counts = {}, set(), Counter()
    field = rejected = servo = bootstrap = gated = 0
    with trace.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "action":
                decisions[row["step"]] = row.get("diagnostics", {})
            elif row.get("event") == "executed":
                step = row["step"]
                if step in acknowledged or step not in decisions:
                    raise ValueError("Invalid acknowledged action sequence")
                acknowledged.add(step)
                diagnostics = decisions.pop(step)
                if diagnostics.get("controller") == "visual_field":
                    field += 1
                    selection = diagnostics.get("selection") or {}
                    status = selection.get("status", "missing_selection_status")
                    counts[status] += 1
                    rejected += int(not selection.get("accepted", False))
                    gate = bool(diagnostics.get("attachment_gate"))
                    gated += int(gate)
                    servo += int(status == "visual_distance_descent" and selection.get("accepted") is True and not gate)
                    bootstrap += int(status == "bootstrap_empirical_image_checked" and selection.get("accepted") is True and not gate)
    if len(acknowledged) != summary["steps"] or field != summary["field_actions"]:
        raise ValueError("Summary action counters differ from acknowledged trace")
    for key, actual in (("field_selection_status_counts", dict(counts)), ("field_rejected_actions", rejected), ("local_servo_actions", servo)):
        if key in summary and summary[key] != actual:
            raise ValueError(f"Summary activation counter differs: {key}")
    return {"source": "acknowledged_trace", "acknowledged_actions": len(acknowledged),
            "field_selected_actions": field, "field_rejected_actions": rejected,
            "local_servo_actions": servo, "bootstrap_actions": bootstrap, "attachment_gate_actions": gated,
            "selection_status_counts": dict(counts),
            "definition": "Servo requires accepted visual_distance_descent; bootstrap requires accepted bootstrap_empirical_image_checked. Both exclude attachment-gated holds."}


def summarize(plan, directory):
    rows = []
    for trial in plan["trials"]:
        out = Path(trial["output"])
        path = out / "summary.json"
        row = {"id": trial["id"], "scene": trial["scene"], "status": "not_started",
               "task_success": False, "completion_time_eligible": False,
               "control_without_video_s": None, "speed_ratio": None,
               "speed_ratio_reason": "Archived Direct is nonconcurrent and differs in observations, prompt and potentially contact criteria."}
        if path.exists():
            summary = read_json(path)
            outcome = qualified_outcome(summary)
            row.update({"status": outcome, "summary": str(path), "summary_sha256": sha256(path),
                        "task_success": outcome == "success", "completion_time_eligible": outcome == "success",
                        "scored": outcome in ALLOWED_TERMINAL, "incomplete": outcome in PAUSED_TERMINAL,
                        "control_without_video_s": time_without_video(summary),
                        "time_kind": "completion" if outcome == "success" else "attempt",
                        "actions": summary["steps"], "requests_started": summary.get("budget", {}).get("api_calls"),
                        "field_actions": summary.get("field_actions"), "direct_actions": summary.get("direct_actions"),
                        "attempt_proxies": len(summary.get("attempts", [])), "recovery_event_proxies": len(summary.get("recovery_events", [])),
                        "activation": activation_counts(out, summary), "cleanup_errors": summary.get("cleanup_errors", [])})
            actual = out / "initial_observation.json"
            row["initial_observation_exact_match"] = actual.exists() and read_json(actual) == read_json(trial["expected_initial"])
        elif out.exists():
            state = read_json(out / "run_state.json") if (out / "run_state.json").exists() else {}
            row.update({"status": state.get("status", "started_without_terminal_summary"), "actions": state.get("step"),
                        "progress_is_not_terminal_outcome": True})
        reference = trial["direct_reference"]
        old = read_json(reference["snapshot"])
        row["archived_direct_reference"] = {**reference, "status": old["status"],
                                            "task_success": old.get("task_success", False),
                                            "control_without_video_s": time_without_video(old),
                                            "actions": old.get("steps"), "speed_ratio": None}
        rows.append(row)
    report = {"schema_version": VERSION, "trials": rows, "trial_count": 4, "new_direct_trials": 0,
              "successful": sum(r["status"] == "success" for r in rows),
              "collision_failures": sum(r["status"] == "collision_failure" for r in rows),
              "incomplete": sum(r["status"] in PAUSED_TERMINAL for r in rows),
              "not_started": sum(r["status"] == "not_started" for r in rows),
              "limitations": ["Single v2 trial per scene; no matched new Direct arm.",
                              "Video time is measured subtraction, not a recording-disabled experiment.",
                              "Attempt and recovery counters are observable proxies, not actual failed grasp counts.",
                              "Local servo selection does not certify semantic target correctness or collision freedom."]}
    write_json(Path(directory) / "comparison.json", report)
    return report


class QueueStop:
    def __init__(self):
        self.requested = False
        self.signal_number = None
        self.child = None
        self.forwarded = False

    def request(self, signum=signal.SIGINT, frame=None):
        self.requested = True
        self.signal_number = signum
        self.forward()

    def forward(self):
        if self.requested and self.child is not None and not self.forwarded and self.child.poll() is None:
            try:
                self.child.send_signal(signal.SIGINT)
                self.forwarded = True
            except ProcessLookupError:
                pass


@contextmanager
def forwarding_signals(stop):
    previous = {sig: signal.signal(sig, stop.request) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run(matrix_path, *, report_only=False, popen=subprocess.Popen, stop=None):
    matrix_path = Path(matrix_path).resolve()
    directory = matrix_path.parent
    plan = validate_plan(read_json(matrix_path), directory)
    if report_only:
        return summarize(plan, directory)
    project = Path(plan["project"])
    state_path = directory / "queue_state.json"
    if state_path.exists() and read_json(state_path).get("status") == "paused":
        raise ValueError("Queue is paused; no implicit retry or skip is authorized")
    verify_inputs(plan)
    freeze_or_verify_sources(project, directory, matrix_path)
    stop = stop or QueueStop()
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONPATH"] = str(project / "src")
    state = {"status": "running", "active_trial": None, "child_pid": None, "stop_requested": False}

    def save(**updates):
        state.update(updates)
        write_json(state_path, state)

    with forwarding_signals(stop):
        try:
            for trial in plan["trials"]:
                if stop.requested:
                    save(status="paused", stop_requested=True, reason="queue_signal", signal=stop.signal_number)
                    break
                out = Path(trial["output"])
                if (out / "summary.json").exists():
                    prior = qualified_outcome(read_json(out / "summary.json"))
                    if prior in ALLOWED_TERMINAL:
                        continue
                    save(status="paused", active_trial=trial["id"], reason=prior)
                    break
                if out.exists():
                    save(status="paused", active_trial=trial["id"], reason="existing_unfinished_trial")
                    break
                verify_inputs(plan)
                freeze_or_verify_sources(project, directory, matrix_path)
                if stop.requested:
                    save(status="paused", stop_requested=True, reason="queue_signal", signal=stop.signal_number)
                    break
                save(active_trial=trial["id"], child_pid=None)
                with (directory / f"{trial['id']}.log").open("x") as log:
                    child = popen([sys.executable, *trial["runner_arguments"]], cwd=project, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    stop.child = child
                    stop.forwarded = False
                    stop.forward()
                    save(child_pid=child.pid)
                    while True:
                        try:
                            code = child.wait()
                            break
                        except (KeyboardInterrupt, InterruptedError):
                            stop.request()
                    stop.child = None
                save(child_pid=None, last_return_code=code)
                freeze_or_verify_sources(project, directory, matrix_path)
                terminal = read_json(out / "summary.json") if (out / "summary.json").exists() else {}
                outcome = qualified_outcome(terminal)
                if stop.requested or code != 0 or outcome not in ALLOWED_TERMINAL or terminal.get("cleanup_errors"):
                    save(status="paused", stop_requested=stop.requested, reason="queue_signal" if stop.requested else outcome,
                         signal=stop.signal_number)
                    break
                actual = out / "initial_observation.json"
                if not actual.exists() or read_json(actual) != read_json(trial["expected_initial"]):
                    save(status="paused", reason="initial_observation_mismatch")
                    break
                summarize(plan, directory)
            else:
                save(status="completed", active_trial=None)
        except Exception as exc:
            save(status="paused", reason="orchestration_error", error_type=type(exc).__name__)
            raise
        finally:
            if stop.child is None:
                summarize(plan, directory)
    return {"queue": state, "comparison": read_json(directory / "comparison.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "runs/visual_field_v2_zero_shot_001/matrix.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    result = run(args.matrix, report_only=args.report_only)
    print(json.dumps({"status": result.get("queue", {}).get("status", "report_only")}))
    return 0 if args.report_only or result["queue"]["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
