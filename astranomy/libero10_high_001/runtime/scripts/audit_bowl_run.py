"""Read-only audit of frozen bowl rollouts, raw API records, traces and videos.

No credentials are read and no network calls are made. Reports are printed, or
written only to --output. A live gzip stream without its footer is partial, not
evidence of corruption. Use --single-episode for a standalone smoke-test run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import zlib

import numpy as np

from gpt_field.action import ActionAdapter
from gpt_field.astra import AstraClient, validate_prediction
from gpt_field.field import GridField
from gpt_field.schema import FieldPrediction, GridSpec


TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"


class Checks:
    def __init__(self):
        self.count = 0
        self.issues = []
        self.issue_count = 0

    def check(self, condition, message):
        self.count += 1
        if not condition:
            self.issue_count += 1
            if len(self.issues) < 30:
                self.issues.append(message)

    def result(self):
        return {"checks": self.count, "issue_count": self.issue_count, "issues": self.issues}


def close(left, right, *, atol=1e-9):
    try:
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        return a.shape == b.shape and bool(np.allclose(a, b, atol=atol, rtol=1e-9))
    except (ValueError, TypeError):
        return False


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def read_lines(path, *, finished=False):
    if not path.exists():
        return [], "missing"
    rows, state = [], "complete"
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt") as stream:
            while True:
                line = stream.readline()
                if not line:
                    break
                if not line.endswith("\n") and not finished:
                    state = "partial"
                    break
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    state = "invalid" if finished else "partial"
                    break
    except (EOFError, OSError, zlib.error, UnicodeError):
        state = "invalid" if finished else "partial"
    return rows, state


def latency(values):
    values = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if not values:
        return {"count": 0}
    return {"count": len(values), "sum_s": sum(values), "min_s": min(values),
            "mean_s": statistics.mean(values), "median_s": statistics.median(values),
            "max_s": max(values)}


def motion_summary(resets, decisions, executions):
    """Observed motion and controller targets only; no inferred object geometry."""
    initial = resets[0].get("observation", {}).get("observation/state") if resets else None
    if initial is None:
        initial = decisions[0].get("state") if decisions else None
    if initial is None:
        return {"executions": 0, "measured_path_length_m": 0.0, "commanded_target_travel_m": 0.0,
                "executed_translation_axes": {}}
    positions = [np.asarray(initial[:3], dtype=float)]
    positions.extend(np.asarray(row["next_state"][:3], dtype=float) for row in executions)
    increments = np.diff(np.asarray(positions), axis=0)
    path_length = float(np.linalg.norm(increments, axis=1).sum())
    displacement = positions[-1] - positions[0]
    decision_by_step = {row["step"]: row for row in decisions}
    axes = Counter()
    target_travel = 0.0
    for row in executions:
        decision = decision_by_step.get(row["step"])
        if decision is None:
            axes["missing_decision"] += 1
            continue
        delta = np.asarray(decision["diagnostics"]["plan"]["delta_xyz"], dtype=float)
        target_travel += float(np.linalg.norm(delta))
        nonzero = np.flatnonzero(np.abs(delta) > 1e-12)
        if not len(nonzero):
            axes["hold"] += 1
        elif len(nonzero) == 1:
            axis = int(nonzero[0])
            axes[("+" if delta[axis] > 0 else "-") + "xyz"[axis]] += 1
        else:
            axes["multiple_axes"] += 1
    last_state = executions[-1]["next_state"] if executions else initial
    return {"executions": len(executions), "initial_xyz_m": positions[0].tolist(),
            "last_xyz_m": positions[-1].tolist(), "endpoint_displacement_xyz_m": displacement.tolist(),
            "endpoint_displacement_norm_m": float(np.linalg.norm(displacement)),
            "vertical_descent_m": float(-displacement[2]), "measured_path_length_m": path_length,
            "commanded_target_travel_m": target_travel, "executed_translation_axes": dict(axes),
            "mean_measured_step_m": path_length / len(executions) if executions else None,
            "realized_path_over_commanded_target_travel": path_length / target_travel if target_travel else None,
            "last_finger_joint_positions": last_state[-2:],
            "interpretation": "Deltas are controller targets, not guaranteed displacement. Stage strings are model output; no target position, clearance, contact or grasp success is inferred here."}


def api_audit(directory, checks, finished):
    requests = {}
    models, requested_models, result_statuses, service_tiers = Counter(), Counter(), Counter(), Counter()
    totals = Counter()
    for request_path in sorted((directory / "api").glob("*.request.json")):
        request = read_json(request_path)
        stem = request_path.name.removesuffix(".request.json")
        response_path = request_path.with_name(stem + ".response.json")
        result_path = request_path.with_name(stem + ".result.json")
        response, result = read_json(response_path), read_json(result_path)
        totals["request_records"] += 1
        record = {"request": request, "response": response, "result": result, "prediction": None}
        requests[request_path.name] = record
        if request is None:
            checks.check(not finished, f"unreadable final request record {request_path.name}")
            totals["partial_or_unreadable_request_records"] += 1
            continue
        payload = request.get("payload", {})
        requested_models[str(payload.get("model"))] += 1
        try:
            prompt = json.loads(payload["input"][0]["content"][0]["text"])["prompt"]
            checks.check(prompt == TASK, f"unexpected API task in {request_path.name}")
        except (ValueError, KeyError, IndexError, TypeError):
            checks.check(False, f"missing API task in {request_path.name}")
        if result:
            result_statuses[str(result.get("status"))] += 1
        error = str((result or {}).get("error", ""))
        blocked = "Batch dispatch stopped" in error
        if blocked:
            totals["locally_blocked_requests_not_dispatched"] += 1
        if response is not None:
            totals["response_records"] += 1
            totals["confirmed_http_calls"] += 1
            models[str(response.get("model"))] += 1
            service_tiers[str(response.get("service_tier"))] += 1
            usage = response.get("usage") or {}
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                if isinstance(usage.get(name), int):
                    totals[name] += usage[name]
            totals["cached_input_tokens"] += (usage.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
            totals["reasoning_tokens_included_in_output"] += (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            if not isinstance(usage.get("input_tokens"), int) or not isinstance(usage.get("output_tokens"), int):
                totals["responses_without_complete_usage"] += 1
            try:
                grid = GridSpec(**request["grid"])
                prediction = validate_prediction(AstraClient._extract(response), grid, request.get("mode", "field"))
                record["prediction"] = prediction
                totals["schema_valid_response_records"] += 1
            except Exception:
                totals["responses_without_valid_prediction"] += 1
        elif "OpenAI HTTP " in error:
            totals["confirmed_http_calls"] += 1
            totals["http_error_records"] += 1
        elif not blocked:
            totals["unresolved_or_transport_failed_requests"] += 1
        if result and result.get("status") == "valid":
            totals["accepted_prediction_records"] += 1
            checks.check(record["prediction"] is not None, f"accepted result lacks valid raw response: {request_path.name}")
    ordinals = sorted(r["request"].get("request_ordinal", -1) for r in requests.values() if r["request"])
    checks.check(ordinals == list(range(1, len(ordinals) + 1)), "API request ordinals are not contiguous from one")
    totals["actual_http_calls_lower_bound"] = totals["confirmed_http_calls"]
    totals["actual_http_calls_upper_bound"] = totals["request_records"] - totals["locally_blocked_requests_not_dispatched"]
    return {"counts": dict(totals), "response_model_ids": dict(models), "requested_models": dict(requested_models),
            "response_service_tiers": dict(service_tiers), "result_statuses": dict(result_statuses),
            "note": "A request log is a reservation; a local shared-gate rejection is not an HTTP call. Unresolved/transport failures may or may not have reached the server. Reasoning tokens are already included in output tokens."}, requests


def plan_audit(entry, prediction, grid, adapter, checks):
    step, details = entry.get("step"), entry.get("diagnostics", {})
    plan = details.get("plan", {})
    state = entry.get("state")
    center = details.get("field_center")
    checks.check(details.get("step") == step, f"step {step}: policy step disagrees with trace")
    try:
        expected_action = adapter.adapt(plan["delta_xyz"], prediction["orientation_target_xyzw"],
                                        prediction["gripper_command"], state)
        checks.check(close(expected_action, entry.get("action")), f"step {step}: action differs from pose/gripper adaptation")
        candidates = plan.get("candidates", [])
        checks.check(plan.get("candidate_count") == len(candidates), f"step {step}: candidate count mismatch")
        checks.check(plan.get("valid_count") == sum(bool(c.get("valid")) for c in candidates),
                     f"step {step}: valid-candidate count mismatch")
        for candidate in candidates:
            if not close(candidate["positions"][0], state[:3]):
                checks.check(False, f"step {step}: candidate starts away from measured end effector")
                break
        selected = plan.get("selected_index")
        if selected is None:
            checks.check(close(plan.get("delta_xyz"), [0, 0, 0]), f"step {step}: unselected plan has nonzero motion")
            return
        checks.check(isinstance(selected, int) and 0 <= selected < len(candidates), f"step {step}: selected candidate index invalid")
        candidate = candidates[selected]
        checks.check(bool(candidate.get("valid")), f"step {step}: selected candidate is invalid")
        checks.check(close(np.asarray(candidate["positions"][1]) - np.asarray(state[:3]), plan["delta_xyz"]),
                     f"step {step}: selected primitive differs from commanded displacement")
        valid = [(i, c) for i, c in enumerate(candidates) if c.get("valid")]
        tolerance = plan.get("numerical_tie_tolerance", 0)
        best_terminal = min(c["terminal_potential"] for _, c in valid)
        ties = [(i, c) for i, c in valid if c["terminal_potential"] - best_terminal <= tolerance]
        best_immediate = min(c["immediate_potential"] for _, c in ties)
        ties = [(i, c) for i, c in ties if c["immediate_potential"] - best_immediate <= tolerance]
        primitive_order = {label: index for index, label in enumerate(("+x", "-x", "+y", "-y", "+z", "-z"))}
        expected = min(ties, key=lambda item: tuple(primitive_order[p] for p in item[1]["primitives"]))[0]
        checks.check(selected == expected, f"step {step}: selected candidate does not follow recorded rank/tie rule")
        field = GridField(grid, center, FieldPrediction.from_dict(prediction, grid))
        direction = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0),
                     "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
        expected_deltas = np.asarray([direction[label] for label in candidate["primitives"]], dtype=float) * 0.01
        checks.check(close(np.diff(np.asarray(candidate["positions"]), axis=0), expected_deltas),
                     f"step {step}: selected path does not follow 1 cm axis primitives")
        checks.check(all(field.segment_valid(a, b) for a, b in zip(candidate["positions"][:-1], candidate["positions"][1:])),
                     f"step {step}: selected path crosses invalid field samples")
        checks.check(close(field.query(state[:3]), plan.get("start_potential")), f"step {step}: start potential does not match measured position")
        checks.check(close(field.query(candidate["positions"][-1]), candidate["terminal_potential"]),
                     f"step {step}: selected terminal potential differs from recorded field")
        checks.check(close(field.query(candidate["positions"][1]), candidate["immediate_potential"]),
                     f"step {step}: selected immediate potential differs from recorded field")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        checks.check(False, f"step {step}: cannot validate plan ({type(exc).__name__})")


def trace_audit(path, checks, finished, requests, grid, *, require_all_executed=True, max_steps=220):
    rows, read_state = read_lines(path, finished=finished)
    checks.check(read_state != "invalid", f"closed trace is invalid: {path.name}")
    decisions, executed, refreshes, resets = [], [], [], []
    prediction = None
    measured_state = None
    adapter = ActionAdapter()
    gripper_predictions, gripper_actions, planner_statuses, stages = Counter(), Counter(), Counter(), Counter()
    motion_modes = Counter()
    executed_grippers = Counter()
    selected_plans = 0
    valid_grid_nodes = []
    for row in rows:
        event = row.get("event")
        if event == "reset":
            resets.append(row)
            measured_state = row.get("observation", {}).get("observation/state")
            checks.check(row.get("observation", {}).get("prompt") == TASK, f"wrong task text at reset in {path.name}")
        elif event == "executed":
            executed.append(row)
            checks.check(row.get("step") == len(executed) - 1, f"execution steps are not contiguous in {path.name}")
            checks.check(bool(row.get("success")) == bool(row.get("sim_status", {}).get("success")),
                         f"step {row.get('step')}: trace success disagrees with simulator predicate")
            if max_steps is None:
                checks.check(row.get("sim_status", {}).get("horizon_reached") is False,
                             f"step {row.get('step')}: unbounded episode reported a reached action horizon")
            checks.check(close(row.get("next_state"), row.get("observation", {}).get("observation/state")),
                         f"step {row.get('step')}: measured observation/next-state mismatch")
            measured_state = row.get("next_state")
            if decisions and decisions[-1].get("step") == row.get("step"):
                executed_grippers[str(decisions[-1]["action"][-1])] += 1
        elif "diagnostics" in row:
            decisions.append(row)
            details, step = row["diagnostics"], row.get("step")
            checks.check(step == len(decisions) - 1, f"decision steps are not contiguous in {path.name}")
            checks.check(len(executed) == step, f"step {step}: decision/execution order mismatch")
            checks.check(close(row.get("state"), measured_state), f"step {step}: policy state differs from latest measured state")
            fresh = bool(details.get("refreshed"))
            checks.check(fresh == (step % 5 == 0), f"step {step}: unexpected field refresh schedule")
            if fresh:
                refreshes.append(row)
                prediction = details.get("field_prediction")
                checks.check(close(details.get("field_center"), row["state"][:3]), f"step {step}: refreshed grid not centered on measured state")
                observation = row.get("observation", {})
                checks.check(close(observation.get("observation/state"), row.get("state")), f"step {step}: refreshed observation state mismatch")
                checks.check(observation.get("prompt") == TASK, f"step {step}: refreshed observation task mismatch")
                request_name = Path(details.get("api_request_log") or "missing").name
                record = requests.get(request_name)
                if requests:
                    checks.check(record is not None, f"step {step}: trace API request reference missing")
                if record and record.get("request"):
                    request = record["request"]
                    checks.check(details.get("api_request_key") == request.get("request_key"), f"step {step}: request hash reference mismatch")
                    checks.check(details.get("prediction") == record.get("prediction"), f"step {step}: trace prediction differs from actual model response")
                    checks.check((record.get("result") or {}).get("status") == "valid", f"step {step}: trace uses a response without accepted result record")
                    try:
                        payload = request["payload"]["input"][0]["content"]
                        sent = json.loads(payload[0]["text"])
                        checks.check(close(sent["observation/state"], row["state"]), f"step {step}: API received different state")
                        checks.check(payload[2]["image_url"] == observation["observation/image"], f"step {step}: API external image differs from trace")
                        checks.check(payload[4]["image_url"] == observation["observation/wrist_image"], f"step {step}: API wrist image differs from trace")
                    except (KeyError, ValueError, TypeError, IndexError):
                        checks.check(False, f"step {step}: cannot verify exact sent observation")
                if prediction:
                    gripper_predictions[prediction.get("gripper_command")] += 1
                    motion_modes[prediction.get("motion_mode")] += 1
                    stages[prediction.get("stage")] += 1
                    valid_grid_nodes.append(grid.count - len(prediction.get("blocked_indices", [])) - len(prediction.get("unknown_indices", [])))
            if prediction:
                plan_audit(row, prediction, grid, adapter, checks)
            else:
                checks.check(False, f"step {step}: no current prediction available")
            selected_plans += details.get("plan", {}).get("selected_index") is not None
            planner_statuses[details.get("status")] += 1
            if row.get("action"):
                gripper_actions[str(row["action"][-1])] += 1
    if finished and require_all_executed:
        checks.check(len(decisions) == len(executed), f"final trace has decision without execution: {path.name}")
    report = {"file": path.name, "read_state": read_state,
              "counts": {"resets": len(resets), "plans": len(decisions), "executions": len(executed),
                         "field_refreshes": len(refreshes), "selected_plans": selected_plans},
              "unexecuted_decisions": len(decisions) - len(executed),
              "refresh_steps": [row["step"] for row in refreshes],
              "gripper_prediction_commands": dict(gripper_predictions), "gripper_action_commands": dict(gripper_actions),
              "executed_gripper_commands": dict(executed_grippers),
              "planner_statuses": dict(planner_statuses), "prediction_stages": dict(stages),
              "prediction_motion_modes": dict(motion_modes),
              "observed_motion": motion_summary(resets, decisions, executed),
              "valid_grid_nodes_per_prediction": valid_grid_nodes,
              "decision_latency": latency([r.get("decision_latency_s") for r in decisions]),
              "api_predict_latency_including_schema_retry": latency([r["diagnostics"].get("api_latency_s") for r in refreshes]),
              "planner_latency": latency([r["diagnostics"].get("plan_latency_s") for r in decisions]),
              "terminal_sim_status": executed[-1].get("sim_status") if executed else (resets[-1].get("sim_status") if resets else None)}
    return report, decisions, executed, refreshes


def probe_video(path):
    try:
        completed = subprocess.run([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_read_frames,nb_frames,r_frame_rate,duration",
            "-of", "json", str(path),
        ], capture_output=True, text=True, timeout=120)
        if completed.returncode:
            return {"error": completed.stderr.strip()[:300]}
        streams = json.loads(completed.stdout).get("streams", [])
        return streams[0] if len(streams) == 1 else {"error": "expected exactly one video stream"}
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def video_audit(trace_path, checks, finished, decisions, executed, refreshes, skip_probe):
    base = trace_path.name.removesuffix(".jsonl.gz")
    event_path = trace_path.with_name(base + ".events.jsonl")
    video_path = trace_path.with_name(base + ".mp4")
    events, read_state = read_lines(event_path, finished=finished)
    checks.check(read_state != "invalid", f"closed video event index invalid: {event_path.name}")
    counts = Counter(e.get("event") for e in events)
    cursor = 0
    maps = {"local_replan": {r["step"]: r for r in decisions}, "executed": {r["step"]: r for r in executed},
            "field_refresh": {r["step"]: r for r in refreshes}}
    for event in events:
        kind, step = event.get("event"), event.get("step")
        checks.check(event.get("frame_start") == cursor, f"video event frame offsets are discontinuous: {base}")
        cursor += event.get("frame_count", 0)
        if kind not in maps:
            continue
        trace = maps[kind].get(step)
        if trace is None:
            checks.check(not finished, f"video {kind} step {step} absent from final trace")
            continue
        state = trace.get("next_state") if kind == "executed" else trace.get("state")
        checks.check(close(event.get("state"), state), f"video {kind} step {step}: measured state mismatch")
        if kind == "local_replan":
            checks.check(close(event.get("action"), trace.get("action")), f"video plan step {step}: action mismatch")
            checks.check(event.get("selected_index") == trace["diagnostics"]["plan"].get("selected_index"),
                         f"video plan step {step}: selected candidate mismatch")
        elif kind == "executed":
            checks.check(bool(event.get("success")) == bool(trace.get("success")), f"video execution step {step}: success mismatch")
        elif kind == "field_refresh" and trace["diagnostics"].get("api_request_key"):
            checks.check(event.get("api_request_key") == trace["diagnostics"].get("api_request_key"),
                         f"video field refresh step {step}: API reference mismatch")
    if finished:
        for kind, expected in (("field_refresh", len(refreshes)), ("local_replan", len(decisions)), ("executed", len(executed))):
            checks.check(counts[kind] == expected, f"video {kind} count {counts[kind]} differs from trace {expected}")
        checks.check(counts["terminal"] == 1, f"final video event index has {counts['terminal']} terminal events")
        checks.check(video_path.exists(), f"final MP4 missing: {video_path.name}")
    probe = None
    if video_path.exists() and not skip_probe:
        probe = probe_video(video_path)
        checks.check("error" not in probe, f"ffprobe failed: {video_path.name}")
        try:
            checks.check(int(probe["nb_read_frames"]) == cursor, f"MP4 decoded frame count differs from event index: {video_path.name}")
        except (KeyError, ValueError, TypeError):
            checks.check(False, f"decoded frame count unavailable: {video_path.name}")
    return {"events_file": event_path.name, "events_read_state": read_state, "event_counts": dict(counts),
            "expected_video_frames": cursor, "final_video_exists": video_path.exists(), "ffprobe": probe}


def audit_episode(directory, index, *, skip_probe=False, batch_mode=True):
    checks = Checks()
    run_state = read_json(directory / "run_state.json") or {}
    coordinator_result = read_json(directory / "coordinator_result.json") or {}
    rows, row_read_state = read_lines(directory / "episodes.jsonl", finished=False)
    terminal_row = rows[0] if len(rows) == 1 else None
    finished = bool(terminal_row)
    checks.check(len(rows) <= 1, f"expected at most one terminal result for initial state {index}")
    if finished:
        checks.check(terminal_row.get("task_id") == 0 and terminal_row.get("episode_index") == index,
                     "terminal result task/initial-state identity mismatch")
        checks.check(terminal_row.get("suite") == "libero_spatial", "terminal result suite mismatch")
    child_manifest = read_json(directory / "manifest.json") or {}
    config = child_manifest.get("configuration", {})
    grid = GridSpec(**{key: value for key, value in config.get("grid", {}).items()
                       if key in ("size", "spacing_m", "frame")})
    max_steps = config.get("simulator", {}).get("max_steps", 220)
    if max_steps is None and terminal_row:
        checks.check(terminal_row.get("status") != "timeout", "unbounded action horizon cannot produce a task timeout")
    if batch_mode and config:
        checks.check(config.get("phase") == "frozen", "child protocol is not frozen")
        checks.check(max_steps == 220, "child horizon differs from 220")
        checks.check(config.get("update_steps") == 5, "child refresh interval differs from five")
        checks.check(config.get("episode_indices") == [index] and config.get("task_ids") == [0], "child requested task/indices mismatch")
    simulator = read_json(directory / "simulator.json") or {}
    if simulator:
        metadata, tasks = simulator.get("metadata", {}), simulator.get("tasks", [])
        checks.check(bool(tasks) and tasks[0].get("prompt") == TASK, "simulator task-zero prompt mismatch")
        checks.check(metadata.get("suite") == "libero_spatial" and metadata.get("task_order_index") == 0,
                     "simulator suite/order mismatch")
        checks.check(metadata.get("initial_states_per_task") == 50, "simulator does not advertise fifty official initial states")
        checks.check(metadata.get("max_policy_steps") == max_steps, "simulator action horizon differs from manifest")
    api, requests = api_audit(directory, checks, finished)
    traces = []
    for path in sorted(directory.glob("*.jsonl.gz")):
        named_final = bool(terminal_row and terminal_row.get("log") == path.name)
        trace, decisions, executed, refreshes = trace_audit(
            path, checks, named_final, requests, grid,
            require_all_executed=bool(terminal_row and terminal_row.get("status") in ("success", "timeout")),
            max_steps=max_steps)
        trace["video"] = video_audit(path, checks, named_final, decisions, executed, refreshes, skip_probe)
        traces.append(trace)
        if named_final:
            checks.check(terminal_row.get("steps") == len(executed), "terminal row step count differs from executed trace")
            status = trace.get("terminal_sim_status") or {}
            if terminal_row["status"] == "success":
                checks.check(status.get("success") is True, "success result has no true simulator success predicate")
                checks.check(status.get("terminated") is True, "success result lacks simulator termination")
            elif terminal_row["status"] == "timeout":
                checks.check(len(executed) == max_steps, "timeout result has not exhausted the policy horizon")
                checks.check(status.get("horizon_reached") is True, "timeout lacks horizon_reached predicate")
                checks.check(status.get("success") is False, "timeout result disagrees with success predicate")
    if finished and terminal_row.get("status") in ("success", "timeout"):
        checks.check(any(trace["file"] == terminal_row.get("log") for trace in traces), "final evaluation trace is missing")
    result_status = terminal_row.get("status") if terminal_row else coordinator_result.get("status", run_state.get("status", "pending"))
    completed_evaluation = bool(terminal_row and terminal_row.get("status") in ("success", "timeout"))
    if terminal_row and terminal_row.get("status") == "timeout" and (max_steps is None or terminal_row.get("steps") != max_steps):
        completed_evaluation = False
    return {"episode_index": index, "directory": directory.name, "status": result_status,
            "finished_record_present": finished, "completed_evaluation": completed_evaluation,
            "coordinator_outcome_present": bool(coordinator_result), "coordinator_status": coordinator_result.get("status"),
            "execution_limits": {"max_policy_steps": max_steps,
                                 **{name: (config.get("api") or {}).get(name) for name in (
                                     "max_api_calls", "max_wall_time_s", "max_cost_usd")}},
            "episode_rows_read_state": row_read_state, "terminal_steps": terminal_row.get("steps") if terminal_row else None,
            "api": api, "traces": traces, **checks.result()}


def audit_run(root, *, skip_probe=False, single_episode=False):
    root = Path(root)
    checks = Checks()
    manifest = read_json(root / "manifest.json") or {}
    configuration = manifest.get("configuration", {})
    if manifest.get("identity_hash"):
        computed = hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()
        checks.check(manifest["identity_hash"] == computed, "manifest configuration hash does not match stored identity")
    if single_episode:
        indices = configuration.get("episode_indices", [0])
        checks.check(len(indices) == 1, "single-episode mode requires one requested index")
        episodes = [audit_episode(root, indices[0], skip_probe=skip_probe, batch_mode=False)]
        intended = 1
    else:
        intended = 10
        indices = list(range(10))
        checks.check(configuration.get("episode_indices") == indices, "batch manifest must request official indices 0–9 exactly")
        checks.check(configuration.get("task_id") == 0 and configuration.get("suite") == "libero_spatial", "batch task/suite mismatch")
        checks.check(configuration.get("task") == TASK, "batch task text mismatch")
        episodes = [audit_episode(root / f"episode_{index:02d}", index, skip_probe=skip_probe) for index in indices]
        actual_directories = {p.name for p in root.glob("episode_*") if p.is_dir()}
        checks.check(actual_directories <= {f"episode_{i:02d}" for i in indices}, "unexpected extra episode directories")
    counts, models, requested_models, grippers, plan_statuses = Counter(), Counter(), Counter(), Counter(), Counter()
    stages, motion_modes, executed_axes = Counter(), Counter(), Counter()
    total_path = total_target_travel = 0.0
    api_totals = Counter()
    for episode in episodes:
        counts["terminal_rows"] += episode["finished_record_present"]
        counts["completed_evaluations"] += episode["completed_evaluation"]
        counts["successes"] += episode["status"] == "success"
        counts["task_timeouts"] += episode["completed_evaluation"] and episode["status"] == "timeout"
        stopped = episode["status"] in ("user_stop", "interrupted")
        budget_limited = episode["status"] in ("budget_exceeded", "budget_limited")
        counts["user_stop_or_interruption_outcomes"] += stopped
        counts["budget_limited_outcomes"] += budget_limited
        other_incomplete = not episode["completed_evaluation"] and not stopped and not budget_limited
        counts["infrastructure_terminal_rows"] += episode["finished_record_present"] and other_incomplete
        counts["infrastructure_outcomes"] += (episode["finished_record_present"] or episode["coordinator_outcome_present"]) and other_incomplete
        api_totals.update(episode["api"]["counts"])
        models.update(episode["api"]["response_model_ids"])
        requested_models.update(episode["api"]["requested_models"])
        for trace in episode["traces"]:
            counts.update(trace["counts"])
            grippers.update(trace["gripper_prediction_commands"])
            plan_statuses.update(trace["planner_statuses"])
            stages.update(trace["prediction_stages"])
            motion_modes.update(trace["prediction_motion_modes"])
            motion = trace["observed_motion"]
            executed_axes.update(motion["executed_translation_axes"])
            total_path += motion["measured_path_length_m"]
            total_target_travel += motion["commanded_target_travel_m"]
    final = counts["completed_evaluations"] == intended
    issues = checks.issue_count + sum(e["issue_count"] for e in episodes)
    complete_artifacts = all(e["finished_record_present"] or e["coordinator_outcome_present"] for e in episodes)
    if complete_artifacts:
        coordinator_summary = read_json(root / "summary.json") or {}
        if not single_episode:
            checks.check(coordinator_summary.get("intended_denominator") == intended, "coordinator denominator mismatch")
            checks.check(coordinator_summary.get("completed_evaluations") == counts["completed_evaluations"], "coordinator completed evaluation count mismatch")
            checks.check(coordinator_summary.get("successes") == counts["successes"], "coordinator success count mismatch")
            issues = checks.issue_count + sum(e["issue_count"] for e in episodes)
    return {"audit_status": "issues_found" if issues else "complete" if complete_artifacts else "partial",
            "evaluation_status": "complete" if final else "provisional", "run": str(root),
            "intended_denominator": intended, "success_rate": counts["successes"] / intended if final else None,
            "counts": dict(counts), "api_counts": dict(api_totals), "response_model_ids": dict(models),
            "requested_models": dict(requested_models), "gripper_prediction_commands": dict(grippers),
            "planner_statuses": dict(plan_statuses), "total_issue_count": issues,
            "prediction_stages": dict(stages), "prediction_motion_modes": dict(motion_modes),
            "observed_motion_all_attempts": {
                "measured_path_length_m": total_path, "commanded_target_travel_m": total_target_travel,
                "executed_translation_axes": dict(executed_axes),
                "mean_measured_step_m": total_path / counts["executions"] if counts["executions"] else None,
                "realized_path_over_commanded_target_travel": total_path / total_target_travel if total_target_travel else None,
            },
            "batch_checks": checks.result(), "episodes": episodes,
            "limitations": "Read-only snapshot; live files may advance between reads. In-progress gzip EOF is partial. Only completed traces are required to match final video event/frame totals. Raw interrupted attempts are retained and counted separately from the requested evaluation set. User stops and budget stops are incomplete outcomes, never task timeouts. This audit makes no API calls."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--output")
    parser.add_argument("--skip-video-probe", action="store_true")
    parser.add_argument("--single-episode", action="store_true")
    args = parser.parse_args(argv)
    report = audit_run(args.run, skip_probe=args.skip_video_probe, single_episode=args.single_episode)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded)
        print(json.dumps({"audit_status": report["audit_status"], "total_issue_count": report["total_issue_count"],
                          "counts": report["counts"], "api_counts": report["api_counts"], "output": str(target)}))
    else:
        print(encoded, end="")
    return 1 if report["total_issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
