#!/usr/bin/env python3
"""Offline artifact audit of a subscription rollout stopped at first close.

Reads recorded traces, model request/response artifacts and the original MP4.
Never authenticates, calls inference, creates a simulator, or executes actions.
Writes an independent audit directory and an unedited final execution frame.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt_field.astra import AstraClient, ENDPOINT


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def near(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(near(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(near(a, b) for a, b in zip(left, right))
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, abs_tol=1e-12, rel_tol=1e-10)
    return left == right


def norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def expected_feedback(before, action, after, step, progress):
    desired = [0.05 * value for value in action[:3]]
    measured = [after[index] - before[index] for index in range(3)]
    progress.append({"before": before[:3], "after": after[:3], "commanded": norm(desired),
                     "measured": norm(measured), "close": action[6] > 0})
    window = progress[-25:]
    return {"type": "execution_feedback", "step": step, "chunk_index": step // 5, "chunk_step": step % 5,
            "before_state": before, "after_state": after, "normalized_action": action,
            "desired_translation_delta_m": desired,
            "desired_world_rotation_delta_rad": [0.5 * value for value in action[3:6]],
            "measured_translation_delta_m": measured, "measured_translation_norm_m": norm(measured),
            "commanded_translation_norm_m": norm(desired),
            "translation_tracking_error_m": norm([measured[index] - desired[index] for index in range(3)]),
            "gripper_joint_delta": [after[index] - before[index] for index in (6, 7)],
            "gripper_command": action[6], "rolling_progress": {
                "window_size": len(window),
                "translation_command_steps": sum(item["commanded"] >= 0.001 for item in window),
                "low_motion_translation_steps": sum(item["commanded"] >= 0.001 and item["measured"] <= 0.0001 for item in window),
                "commanded_translation_path_m": sum(item["commanded"] for item in window),
                "measured_translation_path_m": sum(item["measured"] for item in window),
                "measured_net_translation_m": norm([window[-1]["after"][index] - window[0]["before"][index] for index in range(3)]),
                "gripper_close_steps": sum(item["close"] for item in window)}}


class Audit:
    def __init__(self):
        self.checks = 0
        self.issues = []

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.issues.append(label)


def audit_run(run: Path, output: Path, frozen: Path | None, *, allow_offline_smoke=False):
    run = run.resolve()
    output = output.resolve()
    if output == run or run in output.parents:
        raise ValueError("Use a separate audit directory, outside the original run.")
    summary = read(run / "summary.json")
    offline = summary.get("billing") == "offline_mock"
    if offline and not allow_offline_smoke:
        raise ValueError("This is a scripted smoke, not live subscription inference; use --allow-offline-smoke explicitly.")
    billing = "offline_mock" if offline else "chatgpt_subscription"
    state = read(run / "run_state.json")
    if state.get("stage") != "finished" or summary.get("status") == "running":
        raise ValueError("Wait for a terminal, finalized run before auditing.")
    output.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    manifest = read(run / "manifest.json")
    config = manifest.get("configuration", manifest)
    trace = rows(run / "trace.jsonl")
    actions = [item for item in trace if item["event"] == "action"]
    executed = [item for item in trace if item["event"] == "executed"]
    feedbacks = [item for item in trace if item["event"] == "feedback"]
    resets = [item for item in trace if item["event"] == "reset"]
    audit.check(len(resets) == 1 and trace[0]["event"] == "reset", "one initial reset precedes actions")
    audit.check(len(actions) == len(executed) == len(feedbacks) == summary["steps"], "all actions have execution and feedback")
    audit.check([item["event"] for item in trace] == ["reset"] + [event for _ in actions for event in ("action", "executed", "feedback")], "trace is strictly reset/action/executed/feedback")
    audit.check(summary.get("pending_action_index") is None, "no pending unexecuted action")
    audit.check(summary.get("scored_episodes") == 0 and summary.get("success_rate") is None, "diagnostic is not reported as a scored evaluation")
    audit.check(summary.get("evaluation_completed") is False, "first-close diagnostic is not task completion")
    audit.check(summary.get("suite") == "libero_spatial" and summary.get("seed") == 7
                and summary.get("task_id") == summary.get("episode_index") == 0,
                "requested LIBERO task and initial state identity")
    audit.check(summary.get("task") == "pick up the black bowl between the plate and the ramekin and place it on the plate",
                "requested black bowl task")
    audit.check(config.get("api_key_fallback") is False and config.get("model_fallback") is False,
                "no API-key or model fallback configured")
    audit.check(config.get("max_policy_steps") is None, "no policy action horizon configured")
    audit.check(config.get("simulator", {}).get("max_steps") is None, "no simulator action horizon configured")
    audit.check(config.get("policy", {}).get("success_feedback_to_policy") is False, "no success input to model")
    audit.check(config.get("policy", {}).get("reward_feedback_to_policy") is False, "no reward input to model")
    close_indexes = [item["step"] for item in executed if item["action"][6] > 0]
    reached_close = bool(close_indexes)
    expected_close_status = summary.get("status") == "first_gripper_close_executed"
    audit.check(expected_close_status == reached_close, "terminal status agrees with first-close execution")
    if reached_close:
        audit.check(close_indexes == [len(executed) - 1], "first positive gripper action is the final and only close action")
        close = summary.get("close_event", {})
        last = executed[-1]
        audit.check(close.get("action_index") == last["step"] and close.get("action_ordinal") == last["step"] + 1,
                    "close event identifies exact final action")
        audit.check(close.get("normalized_gripper_command") == last["action"][6]
                    and close.get("close_execution_acknowledged") is True,
                    "close event records the acknowledged positive command")
        audit.check(close.get("gripper_joints_before") == actions[-1]["state"][6:8]
                    and close.get("gripper_joints_after") == last["next_state"][6:8]
                    and close.get("gripper_joint_delta") == feedbacks[-1]["feedback"]["gripper_joint_delta"],
                    "close event reports exact measured finger joints")
        audit.check(close.get("full_closure_verified") is False and close.get("grasp_verified") is False,
                    "no unsupported full-closure or grasp claim")
    audit.check(summary.get("gripper_close_actions") == len(close_indexes), "close count agrees with trace")
    audit.check(summary.get("gripper_open_actions") == sum(item["action"][6] < 0 for item in executed), "open count agrees with trace")
    previous_state = resets[0]["observation"]["observation/state"] if resets else []
    progress = []
    refreshes = []
    for index, (decision, result, feedback) in enumerate(zip(actions, executed, feedbacks)):
        prefix = f"step {index}: "
        diag = decision["diagnostics"]
        audit.check(decision["step"] == result["step"] == feedback["step"] == index, prefix + "ordered step indexes")
        audit.check(decision["state"] == previous_state, prefix + "decision uses previous measured state")
        audit.check(decision["action"] == result["action"], prefix + "executed exact proposed action")
        audit.check(result["next_state"] == result["observation"]["observation/state"], prefix + "post-observation matches measured state")
        audit.check(diag.get("refreshed") == (index % 5 == 0), prefix + "five-action field refresh cadence")
        audit.check("plan" in diag, prefix + "local plan recorded")
        audit.check(diag.get("field_center") == actions[index // 5 * 5]["state"][:3], prefix + "field retains original observation center")
        audit.check(diag.get("feedback") == (feedbacks[index - 1]["feedback"] if index else None), prefix + "decision contains latest execution feedback")
        audit.check(diag.get("history_stats", {}).get("feedback_records") == index, prefix + "history has every prior execution")
        expected = expected_feedback(decision["state"], decision["action"], result["next_state"], index, progress)
        audit.check(near(feedback["feedback"], expected), prefix + "complete allowlisted feedback math")
        if diag.get("refreshed"):
            refreshes.append(decision)
        previous_state = result["next_state"]
    audit.check(summary.get("local_plan_count") == len(actions), "summary plan count agrees")
    audit.check(summary.get("measured_feedback_count") == len(feedbacks), "summary feedback count agrees")
    audit.check(summary.get("validated_field_refreshes") == len(refreshes), "summary valid field count agrees")
    previous_run = ROOT / "runs" / "astra_field_history_001" / "trace.jsonl"
    same_initial_observation = None
    if previous_run.exists() and resets:
        with previous_run.open() as handle:
            prior_reset = json.loads(next(handle))
        same_initial_observation = resets[0]["observation"] == prior_reset["observation"]
        audit.check(same_initial_observation, "initial robot state and both RGB observations exactly match prior field episode")

    request_dir = run / "requests"
    requests = sorted(request_dir.glob("*.request.json"))
    ledger = summary.get("budget", summary.get("usage", {}))
    audit.check(summary.get("billing") == config.get("billing") == ledger.get("billing") == billing, "consistent subscription/scripted billing labels")
    audit.check(ledger.get("api_key_calls") == 0, "usage ledger has no API-key calls")
    audit.check(ledger.get("mock_response_calls" if offline else "api_calls") == len(requests), "request count agrees with usage ledger")
    if offline:
        audit.check(ledger.get("subscription_inference_calls") == ledger.get("api_calls") == 0, "scripted smoke reports zero actual inference calls")
    audit.check(ledger.get("subscription_cost_usd") is None and ledger.get("cost_usd_upper_bound") is None, "subscription usage has no fabricated dollar price")
    usage_totals = Counter()
    output_history = []
    valid_responses = 0
    for index, path in enumerate(requests):
        request = read(path)
        prefix = f"request {index + 1}: "
        payload = request["payload"]
        audit.check(request.get("billing") == billing, prefix + "subscription/scripted route provenance")
        audit.check(request.get("request_ordinal") == index + 1 and request.get("attempt") == 1, prefix + "ordered requests without hidden retries")
        audit.check(request.get("configuration", {}).get("endpoint") == ENDPOINT, prefix + "official Responses endpoint")
        audit.check(payload.get("stream") is True and payload.get("store") is False, prefix + "streaming private protocol")
        audit.check(payload.get("model") == "gpt-6-astra", prefix + "Astra requested without substitution")
        audit.check("max_output_tokens" not in payload, prefix + "no unsupported output cap transmitted")
        inputs = payload["input"]
        retained_outputs = [item for item in inputs if item.get("type") in {"message", "reasoning"}]
        audit.check(retained_outputs == output_history, prefix + "full prior accepted model output retained")
        contained_feedback = []
        images = 0
        for message in inputs:
            if message.get("role") != "user":
                continue
            for part in message.get("content", []):
                if part.get("type") == "input_image":
                    images += 1
                if part.get("type") == "input_text":
                    try:
                        value = json.loads(part["text"])
                    except (ValueError, TypeError):
                        continue
                    if isinstance(value, dict) and value.get("type") == "execution_feedback":
                        contained_feedback.append(value)
        expected_count = min(index * 5, len(feedbacks))
        audit.check(contained_feedback == [item["feedback"] for item in feedbacks[:expected_count]], prefix + "all prior measured feedback transmitted")
        audit.check(images == min(index + 1, 2) * 2, prefix + "exactly latest two camera pairs retained")
        stem = path.name.removesuffix(".request.json")
        response_path, result_path = request_dir / (stem + ".response.json"), request_dir / (stem + ".result.json")
        result = read(result_path)
        if not response_path.exists():
            audit.check(result.get("status") == "error", prefix + "absent response marked as error")
            usage_totals["unknown_usage_calls"] += 1
            continue
        response = read(response_path)
        usage = response.get("usage")
        if isinstance(usage, dict) and type(usage.get("input_tokens")) is int and type(usage.get("output_tokens")) is int:
            usage_totals["input_tokens"] += usage["input_tokens"]
            usage_totals["output_tokens"] += usage["output_tokens"]
            usage_totals["cached_input_tokens"] += usage.get("input_tokens_details", {}).get("cached_tokens", 0)
            usage_totals["reasoning_tokens"] += usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
        else:
            usage_totals["unknown_usage_calls"] += 1
        audit.check(response.get("billing") == billing, prefix + "response subscription/scripted provenance")
        if result.get("status") == "valid":
            audit.check(response.get("status") == "completed", prefix + "complete response before execution")
            audit.check(response.get("model") in (None, "gpt-6-astra"), prefix + "response model unchanged")
            prediction = AstraClient._extract(response)
            audit.check(valid_responses < len(refreshes), prefix + "valid field has a recorded local decision")
            if valid_responses < len(refreshes):
                decision = refreshes[valid_responses]
                audit.check(prediction == decision["diagnostics"]["field_prediction"], prefix + "executed field is actual accepted model output")
                audit.check(request.get("request_key") == decision["diagnostics"].get("api_request_key"), prefix + "field linked to exact model request")
            output_history.extend(response["output"])
            valid_responses += 1
    audit.check(valid_responses == len(refreshes), "every generated valid field used exactly once as a refresh")
    for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens", "unknown_usage_calls"):
        audit.check(ledger.get(key) == usage_totals[key], f"usage total: {key}")

    video_path = run / (summary.get("video") or "episode.mp4")
    events = rows(video_path.with_suffix(".events.jsonl"))
    expected_sequence = [("observation", 0)]
    for decision in actions:
        if decision["diagnostics"]["refreshed"]:
            expected_sequence.append(("field_refresh", decision["step"]))
        expected_sequence.extend([("local_replan", decision["step"]), ("executed", decision["step"])])
    expected_sequence.append(("terminal", max(0, len(actions) - 1)))
    audit.check([(row["event"], row["step"]) for row in events] == expected_sequence, "video events cover every field/plan/execution in order")
    frame_cursor = 0
    for event in events:
        prefix = f"video {event['event']} {event['step']}: "
        audit.check(event["frame_start"] == frame_cursor and event["frame_count"] > 0, prefix + "contiguous frame coverage")
        audit.check(event.get("offline_mock") is offline and event.get("offline_smoke") is offline, prefix + "real model/scripted run truthfully identified")
        audit.check(math.isclose(event["video_time_s"], frame_cursor / 20), prefix + "frame timestamp agrees")
        frame_cursor += event["frame_count"]
        if event["event"] in {"local_replan", "executed"}:
            index = event["step"]
            audit.check(event["action"] == actions[index]["action"], prefix + "actual action displayed")
            expected_state = actions[index]["state"] if event["event"] == "local_replan" else executed[index]["next_state"]
            audit.check(event["state"] == expected_state, prefix + "actual measured state displayed")
        if event["event"] == "executed":
            audit.check(event.get("feedback") == feedbacks[event["step"]]["feedback"], prefix + "exact execution feedback displayed")
        elif event["event"] == "field_refresh":
            audit.check(event["action"] == [] and event["selected_index"] is None, prefix + "field presented before plan/action")
            audit.check(event.get("field_prediction") == actions[event["step"]]["diagnostics"]["field_prediction"], prefix + "actual field displayed")
    audit.check(events[-1]["status"] == summary["status"], "video terminal status agrees with summary")
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,width,height,nb_read_frames,r_frame_rate,duration", "-of", "json", str(video_path)],
                           check=True, capture_output=True, text=True)
    video = json.loads(probe.stdout)["streams"][0]
    audit.check((video["width"], video["height"]) == (1600, 960), "video dimensions preserve annotated panels")
    audit.check(video["codec_name"] == "h264", "video H.264 codec")
    audit.check(video["r_frame_rate"] == "20/1", "video 20fps matches event timeline")
    audit.check(int(video["nb_read_frames"]) == frame_cursor, "decoded frame count matches all event ranges")
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video_path), "-f", "null", "-"], capture_output=True, text=True)
    audit.check(decoded.returncode == 0 and not decoded.stderr.strip(), "full video decode without errors")
    final_execution = next((event for event in reversed(events) if event["event"] == "executed"), None)
    if final_execution:
        frame_number = final_execution["frame_start"]
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video_path), "-vf",
                        f"select=eq(n\\,{frame_number})", "-frames:v", "1", "-update", "1", "-n",
                        str(output / "final_execution.png")], check=True, capture_output=True)
    else:
        frame_number = None

    frozen_count = 0
    if frozen is not None:
        for name, digest in read(frozen).items():
            audit.check(sha(ROOT / name) == digest, "frozen source unchanged: " + name)
            frozen_count += 1
    report = {"passed": not audit.issues, "checks": audit.checks, "issues": audit.issues,
              "source_run": str(run), "status": summary["status"], "first_close_stop_reached": reached_close,
              "first_close_action_index_zero_based": close_indexes[0] if close_indexes else None,
              "executed_actions": len(executed), "validated_fields": len(refreshes), "billing": billing,
              "subscription_requests": 0 if offline else len(requests), "mock_requests": len(requests) if offline else 0,
              "usage_totals": dict(usage_totals), "video": video, "event_frames": frame_cursor,
              "final_execution_frame": frame_number, "frozen_source_files_checked": frozen_count,
              "grasp_success_inferred": False,
              "same_initial_observation_as_prior_field_run": same_initial_observation,
              "interpretation": "A positive close command is only one controller step. It does not establish full closure, contact, or a successful grasp.",
              "artifact_sha256": {name: sha(run / name) for name in ("summary.json", "manifest.json", "run_state.json", "trace.jsonl", "episode.mp4", "episode.events.jsonl")}}
    if final_execution:
        last = executed[-1]
        report["final_gripper_joints_before"] = actions[-1]["state"][6:8]
        report["final_gripper_joints_after"] = last["next_state"][6:8]
        report["final_gripper_joint_delta"] = feedbacks[-1]["feedback"]["gripper_joint_delta"]
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "checks", "issues", "status", "first_close_stop_reached", "executed_actions", "validated_fields", "subscription_requests", "video")}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-sources", type=Path)
    parser.add_argument("--allow-offline-smoke", action="store_true")
    args = parser.parse_args(argv)
    result = audit_run(args.run, args.output, args.frozen_sources, allow_offline_smoke=args.allow_offline_smoke)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
