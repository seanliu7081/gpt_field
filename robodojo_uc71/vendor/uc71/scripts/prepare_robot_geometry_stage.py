#!/usr/bin/env python3
"""Prepare the prompt-only stage for review, using archived data exclusively.

This script has no live mode. It does not launch a simulator, authenticate,
submit a model request, start a supervisor, or authorize a task test. Historical
observations/replies are regression fixtures, never demonstration inputs to a
fresh trial. Outputs go to a new directory; archives are read-only.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt_field.astra import AstraConfig
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.observation import PolicyObservation
from gpt_field.robot_geometry_guidance import robot_geometry_instructions, robot_geometry_metadata
from gpt_field.speed_first_v3 import ChatGPTPlanSpeedFirstV3Client, SpeedFirstV3Policy


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def equal(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(left, right, rel_tol=1e-11, abs_tol=1e-12)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def deny_network(*args, **kwargs):
    raise RuntimeError("Network access is prohibited in this offline preparation")


def prepare_task(archive, task, output, runner):
    output.mkdir()
    trial = archive / "trial"
    identity = read(trial / "manifest.json")["configuration"]
    summary = read(trial / "summary.json")
    require(summary["status"] != "running", "Archive must be terminal")
    config = AstraConfig(**identity["api"])
    initial = read(archive / "expected_initial.json")
    prompt_before = (archive / "prompt.txt").read_text().rstrip("\n")
    initial_payload = read(archive / "expected_first_payload.json")
    baseline = ChatGPTPlanSpeedFirstV3Client(config, task_key=task,
        token_provider=deny_network, transport=deny_network)
    changed = ChatGPTPlanSpeedFirstV3Client(config, task_key=task,
        robot_geometry_guidance=True, token_provider=deny_network, transport=deny_network)
    # The real policy supplies its initial local-feedback status before the
    # first request. Reproduce that preparation, without calling policy.act.
    for client in (baseline, changed):
        policy = SpeedFirstV3Policy(client)
        client.latest_local_feedback = copy.deepcopy(policy.local_feedback)
    original_payload = baseline.build_payload(initial)
    new_payload = changed.build_payload(initial)
    require(original_payload == initial_payload, "Default payload differs from archived v3")
    require(original_payload["instructions"] == prompt_before, "Default prompt differs from archive")
    require(new_payload["instructions"] == prompt_before + "\n\n" + robot_geometry_instructions(),
            "Geometry-only prompt must be the original plus one exact suffix")
    comparable = copy.deepcopy(new_payload)
    comparable["instructions"] = original_payload["instructions"]
    require(comparable == original_payload, "Non-prompt payload change detected")
    (output / "prompt_before.txt").write_text(prompt_before + "\n")
    (output / "prompt_after.txt").write_text(new_payload["instructions"] + "\n")
    write(output / "payload_preview.json", new_payload)

    # Use the real CLI preflight, which uses synthetic images and never starts
    # a simulator. Capture its JSON without ever entering the execution branch.
    import contextlib
    import io
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = runner.main(["--speed-first-v3", "--robot-geometry-guidance",
                              "--task", task, "--preflight"])
    require(result == 0, "Preflight failed")
    preflight = json.loads(stream.getvalue())
    require(not preflight["simulator_started"] and not preflight["credentials_checked"],
            "Preflight performed a live operation")
    require(preflight["api_calls"] == preflight["api_key_calls"] == 0, "Preflight contacted model")
    write(output / "preflight.json", preflight)

    used, current_action = [], None
    checked_requests = {}

    def recorded_transport(endpoint, payload, headers, timeout):
        require(headers["Authorization"] == "Bearer OFFLINE_REPLAY_ONLY", "Unexpected credentials")
        require(current_action is not None and current_action["diagnostics"]["refreshed"],
                "Replay requested a response at an unrecorded boundary")
        path = Path(current_action["diagnostics"]["api_request_log"])
        require(path.parent.resolve() == (trial / "requests").resolve(), "Unexpected request source")
        archived = read(path)
        expected_instructions = archived["payload"]["instructions"] + "\n\n" + robot_geometry_instructions()
        require(payload["instructions"] == expected_instructions, "Prompt suffix changed during replay")
        require(payload["text"] == archived["payload"]["text"], "Response schema changed")
        require(payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"},
                "Model or effort changed")
        response = path.with_name(path.name.replace(".request.json", ".response.json"))
        require(str(path) not in checked_requests, "Replayed request twice")
        checked_requests[str(path)] = sha(path)
        checked_requests[str(response)] = sha(response)
        used.append(current_action["step"])
        return read(response)

    client = ChatGPTPlanSpeedFirstV3Client(config, task_key=task, robot_geometry_guidance=True,
        token_provider=lambda: "OFFLINE_REPLAY_ONLY", transport=recorded_transport)
    policy = SpeedFirstV3Policy(client)
    selected_diagnostics = ("controller", "execution_kind", "refreshed", "plan_refreshed",
        "field_center", "plan", "tracking", "control_context", "effective_compact_field",
        "compact_prediction", "local_visual_feedback", "local_goal_updates", "local_field_revision",
        "map_grid", "map_materialization", "local_gripper")
    counts = {name: 0 for name in ("action", "executed", "feedback", "local_visual_feedback")}
    before = decision = actual_feedback = None
    with (trial / "trace.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            event = row["event"]
            if event == "reset":
                require(before is None, "Duplicate reset")
                before = PolicyObservation.from_dict(row["observation"])
            elif event == "action":
                require(row["step"] == counts[event], "Action order mismatch")
                current_action = row
                decision = policy.act(before)
                require(equal(decision.action, row["action"]), f"Action mismatch at {row['step']}")
                for key in selected_diagnostics:
                    require(equal(decision.diagnostics.get(key), row["diagnostics"].get(key)),
                            f"Diagnostic {key} mismatch at {row['step']}")
                counts[event] += 1
            elif event == "executed":
                require(row["step"] == counts[event] and counts["action"] == counts[event] + 1,
                        "Executed outcome without matching action")
                after = PolicyObservation.from_dict(row["observation"])
                actual_feedback = policy.observe_transition(before, decision, after)
                before = after
                counts[event] += 1
            elif event == "feedback":
                require(row["step"] == counts[event] and counts["executed"] == counts[event] + 1,
                        "Feedback order mismatch")
                require(equal(actual_feedback, row["feedback"]), f"Feedback mismatch at {row['step']}")
                counts[event] += 1
            elif event == "local_visual_feedback":
                require(row["step"] == counts[event], "Visual feedback order mismatch")
                require(equal(policy.local_feedback, row["feedback"]),
                        f"Visual feedback mismatch at {row['step']}")
                counts[event] += 1
    require(all(count == summary["steps"] for count in counts.values()), "Incomplete trace replay")
    require(len(used) == summary["api_calls"], "Response boundary count changed")
    require(all(sha(Path(p)) == expected for p, expected in checked_requests.items()),
            "Historical request or response changed")
    result = {
        "task": task, "archive": str(archive), "all_checks_passed": True,
        "default_initial_payload_exact_archive": True, "only_instruction_suffix_added": True,
        "prompt_chars_before": len(prompt_before), "prompt_chars_after": len(new_payload["instructions"]),
        "prompt_version": client.prompt_version, "geometry": robot_geometry_metadata(),
        "recorded_transitions_checked": counts, "recorded_response_boundaries": used,
        "real_model_calls": 0, "simulator_calls": 0, "new_task_tests": 0,
        "numeric_tolerance": {"absolute": 1e-12, "relative": 1e-11},
        "wall_timing_fields_compared": False,
        "interpretation": "Unchanged execution on saved replies/observations only; does not test fresh model behavior, task success or speed.",
    }
    write(output / "offline_validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), "Output must be a new directory")
    archives = [("bowl_plate", ROOT / "runs/speed_first_v3_bowl_zero_shot_001"),
                ("drawer_bowl", ROOT / "runs/speed_first_v3_drawer_zero_shot_001")]
    require(not any(output.is_relative_to(archive) for _, archive in archives), "Do not write inside an archive")
    protected = [archive / relative for _, archive in archives for relative in (
        "analysis.json", "prompt.txt", "expected_initial.json", "expected_first_payload.json",
        "trial/trace.jsonl", "trial/summary.json", "trial/manifest.json")]
    archive_hashes = {str(p): sha(p) for p in protected}
    runtime = [*sorted((ROOT / "src/gpt_field").glob("*.py")),
               ROOT / "scripts/evaluate_bowl_hybrid_subscription.py", Path(__file__).resolve()]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in runtime}
    output.mkdir(parents=True)
    write(output / "source_manifest.json", source_hashes)
    write(output / "archive_integrity_before.json", archive_hashes)
    (output / "added_prompt.txt").write_text(robot_geometry_instructions() + "\n")
    first_task = resolve_distance_task("bowl_plate")
    write(output / "review_plan.json", {
        "stage": 1, "status": "preparing_offline_only", "full_task_approval_received": False,
        "change": "Append audited Panda frame/geometry guidance only; default-off opt-in flag.",
        "future_stages_not_enabled": ["interaction_attempt_memory", "general_local_interaction_feedback"],
        "proposed_first_test": {"task": first_task.key, "suite": first_task.suite, "task_id": first_task.task_id,
            "episode_index": 0, "seed": 7, "shots": 0, "model": "gpt-6-astra", "effort": "medium",
            "transport": "chatgpt_subscription_only", "completion": "raw_LIBERO_success_only",
            "collision_policy": "record_only", "episode_caps": None, "restarts": False,
            "timing": "control_elapsed_s - measured_video_render_s", "scene_queue": [],
            "approval_required_before_start": True},
        "every_subsequent_full_task_requires_separate_approval": True,
        "future_direct_fairness": "Share this exact static guidance if a matched Direct arm is later approved; no Direct protocol or baseline result changed here.",
    })
    # Fail closed even if a future refactor accidentally enters transport code.
    socket.create_connection = deny_network
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    spec = importlib.util.spec_from_file_location("stage1_runner", ROOT / "scripts/evaluate_bowl_hybrid_subscription.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.SimProcess = deny_network
    results = []
    for task, archive in archives:
        results.append(prepare_task(archive, task, output / task, runner))
        print(json.dumps({"task": task, "checked_actions": results[-1]["recorded_transitions_checked"]["action"],
                          "saved_replies": len(results[-1]["recorded_response_boundaries"]),
                          "real_model_calls": 0, "simulator_calls": 0}), flush=True)
    first_preflight = read(output / first_task.key / "preflight.json")
    require(first_preflight["simulator"]["suite"] == first_task.suite
            and first_preflight["task_id"] == first_task.task_id,
            "Proposed test does not match the runner task configuration")
    require(all(sha(Path(p)) == expected for p, expected in archive_hashes.items()), "Archive changed")
    require(all(sha(ROOT / p) == expected for p, expected in source_hashes.items()), "Runtime changed during checks")
    plan = read(output / "review_plan.json")
    plan["status"] = "offline_validated_awaiting_user_approval"
    write(output / "review_plan.json", plan)
    write(output / "validation.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "all_checks_passed": True,
        "source_hashes_unchanged": True, "historical_artifact_hashes_unchanged": True,
        "tasks": results, "real_model_calls": 0, "simulator_calls": 0, "new_task_tests": 0,
        "full_task_approval_received": False,
    })


if __name__ == "__main__":
    main()
