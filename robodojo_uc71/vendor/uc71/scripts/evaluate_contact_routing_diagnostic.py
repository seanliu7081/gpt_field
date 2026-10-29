"""Finite paired contact diagnostics; never a benchmark full-task evaluation.

The only experimental contrast is contact alignment routing. Checkpoint replay
and privileged engagement evidence belong exclusively to the evaluator. Both
policies start with fresh zero-shot history and receive only RGB/state/task text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time

from gpt_field.astra import AstraConfig
from gpt_field.experiment import write_json
from gpt_field.libero import LiberoConfig
from gpt_field.observation import PolicyObservation

ROOT = Path(__file__).resolve().parents[1]
VERSION = "contact-routing-diagnostic-v1"
LIMITS = {"actions": 100, "model_requests": 8, "completed_close_check_attempts": 2}
ARMS = ("current", "direct5")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration():
    return AstraConfig(model="gpt-6-astra", timeout_s=180,
        max_api_calls=None, max_wall_time_s=None, max_cost_usd=None)


@contextmanager
def simulator_without_credentials():
    names = ("OPENAI_API_KEY", "ACCESS_TOKEN", "OPENAI_ACCESS_TOKEN", "CHATGPT_ACCESS_TOKEN")
    previous = {name: os.environ.pop(name) for name in names if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(previous)


def client_options(checkpoint, arm):
    if arm not in ARMS:
        raise ValueError("Unknown paired diagnostic arm")
    return dict(contact_arm=arm, diagnostic_target=checkpoint["target"],
        task_key=checkpoint["task_key"], reasoning_effort="medium",
        robot_geometry_guidance=True, target_description=checkpoint.get("target_description"),
        planner_mode="field", pregrasp_feedback=True, attempt_memory=True)


def trial_order(checkpoints):
    """Alternate A/B order within each contact type, fixed before outcomes."""
    counts = Counter()
    rows = []
    for checkpoint in checkpoints:
        index = counts[checkpoint["target"]]
        counts[checkpoint["target"]] += 1
        for arm in ARMS if index % 2 == 0 else reversed(ARMS):
            rows.append({"id": f"{checkpoint['id']}__{arm}",
                         "checkpoint_id": checkpoint["id"], "arm": arm})
    return rows


def stop_reason(*, steps, requests, completed_attempts, needs_model_request,
                engagement, limits=LIMITS):
    if engagement:
        return "local_engagement_confirmed"
    if steps >= limits["actions"]:
        return "diagnostic_action_limit"
    if completed_attempts >= limits["completed_close_check_attempts"]:
        return "diagnostic_attempt_limit"
    # Finish already-authorized local actions (including close/check) after the
    # eighth request. Never solicit a ninth decision just to discover this bound.
    if requests >= limits["model_requests"] and needs_model_request:
        return "diagnostic_request_limit"
    return None


def _error(exc):
    # Do not retain arbitrary exception strings from auth/network transports.
    return {"type": type(exc).__name__,
            **({"error_code": exc.error_code} if hasattr(exc, "error_code") else {})}


def run_trial(output, checkpoint, arm, *, limits=LIMITS, client_factory=None,
              policy_factory=None, sim_factory=None, recorder_factory=None,
              token_provider=None, transport=None, clock=time.monotonic):
    from gpt_field.contact_diagnostic_policy import (
        ChatGPTPlanContactDiagnosticClient, ContactDiagnosticPolicy)
    from gpt_field.contact_diagnostic_sim import ContactDiagnosticSimProcess
    from gpt_field.speed_first_v3_video import SpeedFirstV3VideoRecorder
    if not callable(token_provider):
        raise ValueError("A subscription token provider is required; no API-key fallback")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "requests").mkdir()
    client_factory = client_factory or ChatGPTPlanContactDiagnosticClient
    policy_factory = policy_factory or ContactDiagnosticPolicy
    sim_factory = sim_factory or ContactDiagnosticSimProcess
    recorder_factory = recorder_factory or SpeedFirstV3VideoRecorder
    kwargs = client_options(checkpoint, arm)
    kwargs["log_dir"] = output / "requests"
    if token_provider is not None:
        kwargs["token_provider"] = token_provider
    if transport is not None:
        kwargs["transport"] = transport
    client = client_factory(configuration(), **kwargs)
    policy = policy_factory(client, planner_mode="field")
    sim_config = LiberoConfig(**checkpoint["sim_config"])
    sim = None
    recorder = None
    status, error = "preparing", None
    steps = feedbacks = 0
    timing = dict(video_render_s=0., model_s=0., decision_s=0., simulator_step_s=0.,
                  feedback_s=0., planning_s=0.)
    control_start = control_end = None
    start = clock()
    execution = Counter()
    evaluator = {}
    raw_success = False
    write_json(output / "manifest.json", {"version": VERSION, "arm": arm,
        "checkpoint": checkpoint, "limits": limits, "configuration": asdict(configuration()),
        "policy_options": client_options(checkpoint, arm), "created_utc": utc(),
        "full_task": False, "shots": 0, "evaluator_truth_to_policy": False})

    def state(stage):
        write_json(output / "run_state.json", {"updated_utc": utc(), "stage": stage,
            "steps": steps, "feedback_count": feedbacks, "arm": arm,
            "diagnostics": policy.contact_diagnostic_stats(),
            "budget": client.budget_snapshot(), "local_evaluator": evaluator,
            "control_elapsed_s": (control_end if control_end is not None else clock()) - control_start
                if control_start is not None else None})

    try:
        sim = sim_factory(sim_config, target=checkpoint["target"],
                          diagnostic_action_limit=limits["actions"])
        actions = json.loads(Path(checkpoint["prefix_actions_path"]).read_text())
        if isinstance(actions, dict):
            actions = actions["actions"]
        with simulator_without_credentials():
            obs = sim.replay_checkpoint(checkpoint["task_id"], checkpoint["episode_index"],
                actions, expected_fingerprint=checkpoint["expected_fingerprint"],
                initial_fingerprint=checkpoint.get("initial_fingerprint"))
        # Reconstruction through the allowlist prevents evaluator side-channel
        # additions from ever becoming part of a model observation.
        obs = PolicyObservation.from_dict(obs.to_dict())
        write_json(output / "initial_observation.json", obs.to_dict())
        write_json(output / "simulator.json", sim.metadata)
        write_json(output / "checkpoint_check.json", sim.checkpoint_check)
        evaluator = sim.begin_diagnostic(checkpoint["expected_fingerprint"])
        actual_physics = evaluator.get("checkpoint_physics_fingerprint")
        expected_physics = checkpoint.get("paired_physics_fingerprint")
        write_json(output / "paired_state_check.json", {
            "observation_match": sim.checkpoint_check,
            "physics_fingerprint": actual_physics, "expected_physics_fingerprint": expected_physics,
            "physics_matches_first_arm": actual_physics == expected_physics if expected_physics else None})
        if expected_physics is not None and actual_physics != expected_physics:
            raise RuntimeError("Paired checkpoint physics/controller state differs; no inference allowed")
        if not evaluator["criterion_ready"]:
            raise RuntimeError("Independent engagement criterion is unavailable")
        if evaluator.get("engagement_success"):
            raise RuntimeError("Checkpoint is already locally successful")
        policy.reset()
        recorder = recorder_factory(output / "episode.mp4", grid=policy.grid,
            episode_id=f"CONTACT DIAGNOSTIC / {checkpoint['id']} / {arm} / ZERO SHOT",
            offline_smoke=False)
        recorder.observe(obs, step=0)
        setup_s = clock() - start
        control_start = clock()
        with (output / "trace.jsonl").open("w") as trace, (output / "evaluator.jsonl").open("w") as eval_log:
            def append(stream, row):
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
            append(trace, {"event": "reset", "observation": obs.to_dict()})
            append(eval_log, {"step": 0, "evaluator": evaluator})
            while True:
                stats = policy.contact_diagnostic_stats()
                status = stop_reason(steps=steps, requests=client.budget_snapshot()["api_calls"],
                    completed_attempts=stats["completed_close_check_attempts"],
                    needs_model_request=policy.needs_model_request(),
                    engagement=evaluator.get("engagement_success", False), limits=limits)
                if status:
                    break
                if (output.parent.parent / "operator_stop.request.json").exists():
                    status = "user_stop"
                    break
                if sim.status.get("terminated"):
                    status = "simulator_terminated"
                    break
                state("deciding")
                began = clock()
                decision = policy.act(obs)
                latency = clock() - began
                timing["decision_s"] += latency
                details = decision.diagnostics
                timing["model_s"] += details.get("api_latency_s", 0.) or 0.
                timing["planning_s"] += details.get("plan_latency_s", 0.) or 0.
                append(trace, {"event": "action", "step": steps, "state": list(obs.state),
                    "action": list(decision.action), "diagnostics": details,
                    "decision_latency_s": latency})
                began = clock()
                recorder.decision(obs, decision, decision_latency_s=latency)
                timing["video_render_s"] += clock() - began
                before = obs
                state("executing")
                began = clock()
                obs, raw_success = sim.step(decision.action)
                timing["simulator_step_s"] += clock() - began
                obs = PolicyObservation.from_dict(obs.to_dict())
                steps += 1
                evaluator = dict(sim.evaluator_metrics)
                began = clock()
                feedback = policy.observe_transition(before, decision, obs)
                timing["feedback_s"] += clock() - began
                feedbacks += 1
                execution[details.get("execution_kind", "unknown")] += 1
                append(trace, {"event": "executed", "step": steps - 1,
                    "observation": obs.to_dict(), "action": list(decision.action),
                    "feedback": feedback, "operation": policy.operation_stats(),
                    "diagnostic": policy.contact_diagnostic_stats()})
                append(eval_log, {"step": steps, "evaluator": evaluator,
                                  "raw_libero_success": raw_success})
                began = clock()
                recorder.executed(obs, step=steps - 1, success=False,
                    sim_status={"success": False, "terminated": False},
                    feedback=feedback, history_stats=client.history_stats())
                timing["video_render_s"] += clock() - began
                state("observed")
                print(json.dumps({"event": "executed", "trial": output.name, "steps": steps,
                    "requests": client.budget_snapshot()["api_calls"],
                    "local_engagement": evaluator.get("engagement_success", False),
                    "execution_kind": details.get("execution_kind")}), flush=True)
        control_end = clock()
    except KeyboardInterrupt:
        status, error = "user_stop", {"type": "KeyboardInterrupt"}
    except Exception as exc:
        status, error = "diagnostic_error", _error(exc)
    finally:
        if control_start is not None and control_end is None:
            control_end = clock()
        if sim is not None:
            try:
                sim.close()
            except Exception as exc:
                status, error = "cleanup_error", _error(exc)
        if recorder is not None:
            try:
                recorder.close(status=status, error=json.dumps(error) if error else None)
            except Exception as exc:
                status, error = "cleanup_error", _error(exc)
    control_s = control_end - control_start if control_start is not None else None
    summary = {"version": VERSION, "status": status, "error": error,
        "trial_id": output.name, "checkpoint_id": checkpoint["id"], "target": checkpoint["target"],
        "arm": arm, "full_task": False, "local_engagement_success": status == "local_engagement_confirmed",
        "libero_success": bool(raw_success), "task_completion_time_eligible": False,
        "steps": steps, "feedback_count": feedbacks, "execution_counts": dict(execution),
        "model_requests": client.budget_snapshot()["api_calls"], "api_key_calls": 0,
        "model": "gpt-6-astra", "effort": "medium", "billing": "chatgpt_subscription",
        "setup_s": setup_s if control_start is not None else clock() - start,
        "control_elapsed_s": control_s,
        "control_without_video_s": control_s - timing["video_render_s"] if control_s is not None else None,
        "timing_components": timing, "limits": limits,
        "diagnostics": policy.contact_diagnostic_stats(), "operation": policy.operation_stats(),
        "evaluator": evaluator, "ended_utc": utc(), "budget": client.budget_snapshot()}
    write_json(output / "summary.json", summary)
    state(status)
    return summary


def seal_sources():
    files = sorted((ROOT / "src/gpt_field").glob("*.py")) + [Path(__file__)]
    return {str(path.relative_to(ROOT)): sha(path) for path in files}


def verify_manifest(manifest):
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError("Frozen source changed: " + name)
    for name, expected in manifest["input_sha256"].items():
        if sha(name) != expected:
            raise RuntimeError("Frozen input changed: " + name)
    if manifest["limits"] != LIMITS or manifest["full_task"]:
        raise RuntimeError("Diagnostic protocol changed")


def prepare(selection_path, directory):
    from gpt_field.contact_diagnostic_policy import ChatGPTPlanContactDiagnosticClient
    selection_path, directory = Path(selection_path).absolute(), Path(directory).absolute()
    if (directory / "manifest.json").exists():
        raise FileExistsError("Diagnostic manifest already exists")
    directory.mkdir(parents=True, exist_ok=True)
    selection = json.loads(selection_path.read_text())
    checkpoints = selection["checkpoints"]
    if Counter(x["target"] for x in checkpoints) != {"drawer_handle": 2, "bowl": 2}:
        raise ValueError("Expected two distinct checkpoints per contact type")
    if len({x["id"] for x in checkpoints}) != 4:
        raise ValueError("Checkpoint IDs must be unique")
    inputs = {str(selection_path): sha(selection_path)}
    previews = directory / "payload_previews"
    previews.mkdir(exist_ok=False)
    for checkpoint in checkpoints:
        for key in ("prefix_actions_path", "observation_path"):
            inputs[checkpoint[key]] = sha(checkpoint[key])
        cfg = LiberoConfig(**checkpoint["sim_config"])
        if cfg.image_size != 512 or cfg.render_size != 512 or cfg.include_arm_joints:
            raise ValueError("Keep the matched native512 RGB + state8 interface")
        obs = PolicyObservation.from_dict(json.loads(Path(checkpoint["observation_path"]).read_text()))
        for arm in ARMS:
            client = ChatGPTPlanContactDiagnosticClient(configuration(),
                token_provider=lambda: None, **client_options(checkpoint, arm))
            payload = client.build_payload(obs.to_dict())
            assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
            write_json(previews / f"{checkpoint['id']}__{arm}.json", payload)
    manifest = {"version": VERSION, "created_utc": utc(), "full_task": False,
        "approval": "ok do the step 1 and step 2 to identify the problem",
        "checkpoints": checkpoints, "trial_order": trial_order(checkpoints), "limits": LIMITS,
        "source_sha256": seal_sources(), "input_sha256": inputs,
        "shots": 0, "model": "gpt-6-astra", "effort": "medium", "billing": "chatgpt_subscription",
        "independent_evaluator_truth_to_policy": False,
        "selection_note": "Purposive diagnostic checkpoints, not an unbiased task success sample"}
    write_json(directory / "manifest.json", manifest)
    return manifest


def execute(directory, *, token_provider=None):
    directory = Path(directory).absolute()
    manifest = json.loads((directory / "manifest.json").read_text())
    verify_manifest(manifest)
    if token_provider is None:
        from gpt_field.chatgpt_auth import SubscriptionCredentials
        token_provider = SubscriptionCredentials().access_token
    if not callable(token_provider):
        raise ValueError("Execution requires the subscription credential provider")
    trials = directory / "trials"
    trials.mkdir(exist_ok=False)
    checkpoints = {row["id"]: row for row in manifest["checkpoints"]}
    paired_physics = {}
    summaries = []
    for row in manifest["trial_order"]:
        verify_manifest(manifest)
        if (directory / "operator_stop.request.json").exists():
            break
        checkpoint = dict(checkpoints[row["checkpoint_id"]])
        if checkpoint["id"] in paired_physics:
            checkpoint["paired_physics_fingerprint"] = paired_physics[checkpoint["id"]]
        result = run_trial(trials / row["id"], checkpoint, row["arm"], token_provider=token_provider)
        summaries.append(result)
        write_json(directory / "summary.json", {"version": VERSION, "updated_utc": utc(),
            "full_task": False, "trials_completed": len(summaries), "trials_planned": 8,
            "results": summaries})
        if result["status"] in ("user_stop", "diagnostic_error", "cleanup_error", "simulator_terminated"):
            break
        physics = result["evaluator"].get("checkpoint_physics_fingerprint")
        if not isinstance(physics, dict) or not physics.get("combined_sha256"):
            raise RuntimeError("Missing paired physics/controller identity")
        paired_physics[checkpoint["id"]] = physics
    verify_manifest(manifest)
    write_json(directory / "finished.json", {"ended_utc": utc(), "trials_completed": len(summaries),
        "trials_planned": 8, "source_freeze_verified": True,
        "status": "complete" if len(summaries) == 8 else "stopped"})
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_only:
        if args.selection is None:
            parser.error("--prepare-only needs --selection")
        value = prepare(args.selection, args.output)
        print(json.dumps({"status": "prepared", "trials": len(value["trial_order"])}))
    else:
        if args.selection is not None:
            parser.error("Execution uses only the frozen manifest")
        execute(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
