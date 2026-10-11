#!/usr/bin/env python3
"""Independent, offline audit of recorded subscription hybrid robot rollouts.

Reconstructs prompts/history and validates actions using the frozen field
planner/adapter or exact direct action slices. It never authenticates, requests
inference, imports a simulator worker, or replays simulator actions. Opaque model
output is compared as data and is never printed or included in audit reports.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt_field.action import ActionAdapter
from gpt_field.astra import AstraClient, ENDPOINT, _canonical, output_schema, validate_prediction
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS
from gpt_field.direct_history import HISTORY_PROTOCOL, TASK
from gpt_field.experiment import digest
from gpt_field.field import GridField
from gpt_field.hybrid_subscription import ChatGPTPlanHybridClient, HYBRID_PROMPT_VERSION, hybrid_output_schema
from gpt_field.observation import PolicyObservation
from gpt_field.planner import LocalPlanner, PlannerConfig
from gpt_field.schema import FieldPrediction, GridSpec

_spec = importlib.util.spec_from_file_location("_close_audit_helpers", ROOT / "scripts/audit_subscription_close.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
read, rows, sha, near = _base.read, _base.rows, _base.sha, _base.near


class Audit(_base.Audit):
    def __init__(self):
        super().__init__()
        self.notes = []


def _wrapper(value, grid, audit, prefix):
    """Independent routing checks plus the frozen nested interface validators."""
    expected = {"controller", "target_region", "near_target", "routing_reason", "field", "direct"}
    audit.check(isinstance(value, dict) and set(value) == expected, prefix + "exact hybrid wrapper keys")
    if not isinstance(value, dict) or set(value) != expected:
        return False
    contact = value["controller"] == "contact_direct"
    audit.check(value["controller"] in ("coarse_field", "contact_direct"), prefix + "known selected controller")
    audit.check(value["target_region"] in ("black_bowl", "plate"), prefix + "active target region allowlist")
    audit.check(type(value["near_target"]) is bool and value["near_target"] is contact,
                prefix + "near-target label agrees with selected branch")
    reason = value["routing_reason"]
    audit.check(isinstance(reason, str) and bool(reason.strip()) and len(reason) <= 500,
                prefix + "short nonempty routing explanation")
    active, unused = ("direct", "field") if contact else ("field", "direct")
    audit.check(value[unused] is None, prefix + "unused branch is null")
    try:
        validate_prediction(value[active], grid, active)
        valid = True
    except Exception:
        valid = False
    audit.check(valid, prefix + "selected branch passes frozen field or direct validator")
    return valid


def _observation_message(observation, index):
    """Construct the allowlisted user message independently of the hybrid client."""
    obs = PolicyObservation.from_dict(observation)
    return {"role": "user", "content": [
        {"type": "input_text", "text": _canonical({"type": "observation", "observation_index": index,
            "prompt": obs.prompt, "observation/state": list(obs.state)})},
        {"type": "input_text", "text": "External camera:"},
        {"type": "input_image", "image_url": obs.image, "detail": "high"},
        {"type": "input_text", "text": "Wrist camera:"},
        {"type": "input_image", "image_url": obs.wrist_image, "detail": "high"},
    ]}


def _retained_input(history, observation, index):
    messages = copy.deepcopy(history) + [_observation_message(observation, index)]
    image_messages = [message for message in messages if message.get("role") == "user"
        and isinstance(message.get("content"), list)
        and any(part.get("type") == "input_image" for part in message["content"])]
    for message in image_messages[:-2]:
        message["content"] = [{"type": "input_text", "text": "[Older camera image omitted by fixed image-retention rule.]"}
            if part.get("type") == "input_image" else part for part in message["content"]]
    return messages


def _history_stats(history, observations, feedback_count):
    return {"protocol": HISTORY_PROTOCOL, "observations": observations, "feedback_records": feedback_count,
        "messages": len(history), "retained_image_pairs": sum(message.get("role") == "user"
            and isinstance(message.get("content"), list)
            and any(part.get("type") == "input_image" for part in message["content"]) for message in history),
        "image_history_limit": 2, "text_history": "all_observations_validated_responses_output_and_state_feedback",
        "reasoning_replay": "opaque_encrypted_output_items"}


def _execution_evidence(summary, state, actions, executed, audit):
    """Distinguish a proposal, an execution-intent marker, and an actual reply."""
    # The runner logs proposals before rendering decision video and setting the
    # execution marker. The marker itself precedes sim.step, so even it does not
    # prove delivery. Only the simulator's returned execution record proves that.
    unacknowledged = len(executed) if len(actions) > len(executed) else None
    pending = summary.get("pending_action_index")
    state_pending = state.get("pending_action_index")
    audit.check(pending == state_pending and type(pending) is type(state_pending),
                "summary and state execution markers agree")
    audit.check(pending is None or (type(pending) is int and pending == unacknowledged),
                "execution marker refers only to a proposed unacknowledged action")
    error_stage = summary.get("error", {}).get("stage") if isinstance(summary.get("error"), dict) else None
    evidence = {"all_proposals_acknowledged": unacknowledged is None,
        "unacknowledged_action_index": unacknowledged, "execution_marker_recorded": pending is not None,
        "dispatch_state": "no_unacknowledged_proposal" if unacknowledged is None else "indeterminate",
        "terminal_error_stage": error_stage}
    if unacknowledged is not None:
        if pending is None and error_stage == "deciding":
            evidence["proposal_stage"] = "before_execution_marker"
            audit.notes.append("Final action was proposed while deciding, without a durable execution marker; "
                               "dispatch or execution is not established by the proposal record.")
        elif pending is None:
            evidence["proposal_stage"] = "unmarked"
            audit.notes.append("Final action has no acknowledgement or execution marker; dispatch is indeterminate.")
        else:
            evidence["proposal_stage"] = "execution_marked"
            audit.notes.append("Final action has an execution-intent marker but no simulator acknowledgement; "
                               "delivery and physical execution remain indeterminate.")
    return evidence


def _video_audit(run, summary, actions, executed, feedback, audit):
    video_path = run / (summary.get("video") or "episode.mp4")
    event_path = video_path.with_suffix(".events.jsonl")
    if not video_path.is_file() or not event_path.is_file():
        audit.check(False, "finalized video and event timeline exist")
        return None, 0
    events = rows(event_path)
    expected = [("observation", 0)]
    previous = None
    for step, action in actions.items():
        diag = action["diagnostics"]
        control = diag["controller"]
        if diag["refreshed"]:
            expected.append(("routing_decision", step))
            if previous is not None and previous != control:
                expected.append(("controller_switch", step))
            expected.append(("field_refresh" if control == "coarse_field" else "action_chunk", step))
        expected.append(("local_replan" if control == "coarse_field" else "direct_action", step))
        if step in executed and step in feedback:
            expected.append(("executed", step))
        previous = control
    expected.append(("terminal", max(actions, default=0)))
    audit.check([(event["event"], event["step"]) for event in events] == expected,
                "video shows every routing, switch, field/chunk, plan/action, execution in order")
    cursor = 0
    offline = summary["billing"] == "offline_mock"
    for event in events:
        prefix = f"video {event.get('event')} step {event.get('step')}: "
        audit.check(event.get("frame_start") == cursor and type(event.get("frame_count")) is int
                    and event["frame_count"] > 0, prefix + "contiguous frame interval")
        audit.check(near(event.get("video_time_s"), cursor / 20), prefix + "20 Hz frame timestamp")
        cursor += event.get("frame_count", 0)
        audit.check(event.get("offline_mock") is offline and event.get("offline_smoke") is offline,
                    prefix + "live or scripted provenance")
        audit.check(event.get("policy") == "hybrid_field_direct", prefix + "hybrid video policy")
        step, kind = event.get("step"), event.get("event")
        if step not in actions or kind in ("observation", "terminal"):
            continue
        action, diag = actions[step], actions[step]["diagnostics"]
        audit.check(event.get("controller") == diag["controller"] and event.get("routing") == diag["routing"],
                    prefix + "actual selected controller and routing displayed")
        if diag["controller"] == "contact_direct":
            audit.check(event.get("field_center") is None and event.get("field_step") is None
                        and event.get("selected_index") is None and event.get("candidate_count") is None,
                        prefix + "direct mode displays no stale field or local plan")
        if kind in ("local_replan", "direct_action", "executed"):
            audit.check(event.get("action") == action["action"], prefix + "exact actual action displayed")
            expected_state = executed[step]["next_state"] if kind == "executed" and step in executed else action["state"]
            audit.check(event.get("state") == expected_state, prefix + "actual robot state displayed")
        if kind == "executed" and step in feedback:
            audit.check(event.get("feedback") == feedback[step]["feedback"], prefix + "exact measured feedback displayed")
        if kind in ("routing_decision", "controller_switch", "field_refresh", "action_chunk"):
            audit.check(event.get("action") == [] and event.get("selected_index") is None,
                        prefix + "generation shown before action and plan selection")
        if kind == "routing_decision":
            audit.check(event.get("hybrid_prediction") == diag.get("hybrid_prediction"), prefix + "complete accepted wrapper displayed")
        if kind == "field_refresh":
            audit.check(event.get("field_prediction") == diag.get("field_prediction"), prefix + "actual generated field displayed")
        if kind == "action_chunk":
            audit.check(event.get("actions") == diag["hybrid_prediction"]["direct"]["actions"], prefix + "actual direct chunk displayed")
    audit.check(bool(events) and events[-1].get("status") == summary["status"], "video terminal outcome matches summary")
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_read_frames,r_frame_rate,duration", "-of", "json", str(video_path)],
            check=True, capture_output=True, text=True)
        video = json.loads(probe.stdout)["streams"][0]
        audit.check((video["width"], video["height"]) == (1600, 960), "video dimensions preserve all panels")
        audit.check(video["codec_name"] == "h264" and video["r_frame_rate"] == "20/1", "video codec and frame rate")
        audit.check(int(video["nb_read_frames"]) == cursor, "decoded frame count matches event ranges")
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video_path), "-f", "null", "-"],
                                 capture_output=True, text=True)
        audit.check(decoded.returncode == 0 and not decoded.stderr.strip(), "complete original MP4 decodes without errors")
        return video, cursor
    except Exception:
        audit.check(False, "ffprobe and full video decode completed")
        return None, cursor


def audit_run(run: Path, output: Path, frozen: Path | None = None, *, allow_offline_smoke=False):
    run, output = run.resolve(), output.resolve()
    if output == run or run in output.parents:
        raise ValueError("Use a fresh separate audit directory outside the original run")
    if output.exists():
        raise FileExistsError("Audit output already exists")
    summary, state = read(run / "summary.json"), read(run / "run_state.json")
    if state.get("stage") != "finished" or summary.get("status") == "running":
        raise ValueError("Audit only a terminal, finalized run")
    offline = summary.get("billing") == "offline_mock"
    if offline and not allow_offline_smoke:
        raise ValueError("Scripted smoke requires explicit --allow-offline-smoke")
    output.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    manifest = read(run / "manifest.json")
    config = manifest["configuration"]
    billing = "offline_mock" if offline else "chatgpt_subscription"
    audit.check(manifest.get("identity_hash") == digest(config), "manifest identity digest")
    audit.check(config.get("kind") == "subscription_hybrid_full_task", "hybrid full-task experiment identity")
    audit.check(config.get("task") == summary.get("task") == TASK, "exact requested black bowl task")
    audit.check(config.get("task_id") == config.get("episode_index") == summary.get("task_id") == summary.get("episode_index") == 0,
                "task zero and initial state zero")
    simconfig = config.get("simulator", {})
    audit.check(simconfig.get("suite") == "libero_spatial" and simconfig.get("seed") == 7,
                "LIBERO-Spatial seed-seven setup")
    audit.check(simconfig.get("max_steps") is None and config.get("max_policy_steps") is None,
                "no robot action horizon")
    for key in ("max_api_calls", "max_wall_time_s", "max_cost_usd", "input_price_per_million",
                "cached_input_price_per_million", "cache_write_price_per_million", "output_price_per_million"):
        audit.check(config.get("api", {}).get(key) is None, "no API-key pricing or episode cap: " + key)
    audit.check(config.get("api_key_fallback") is False and config.get("model_fallback") is False,
                "no API-key or model fallback")
    policy_config = config.get("policy", {})
    for key in ("success_feedback_to_policy", "reward_feedback_to_policy", "object_state_feedback_to_policy"):
        audit.check(policy_config.get(key) is False, "no oracle policy input: " + key)
    audit.check(policy_config.get("grid") == GridSpec().to_dict(), "original field grid")
    audit.check(policy_config.get("planner") == json.loads(json.dumps(asdict(PlannerConfig()))),
                "original local planner configuration")
    audit.check(policy_config.get("adapter_max_rotation_rad") == .15, "original field rotation adapter cap")
    audit.check(policy_config.get("update_steps") == 5, "five-action hybrid routing interval")
    audit.check(config.get("billing") == summary.get("billing") == billing, "billing provenance")
    audit.check(summary.get("api_key_calls") == 0 and summary.get("api_key_fallback") is False, "no API-key calls reported")

    trace = rows(run / "trace.jsonl") if (run / "trace.jsonl").is_file() else []
    resets = [row for row in trace if row.get("event") == "reset"]
    audit.check(len(resets) == 1 and trace[0].get("event") == "reset", "one reset before all actions")
    actions_list = [row for row in trace if row.get("event") == "action"]
    executed_list = [row for row in trace if row.get("event") == "executed"]
    feedback_list = [row for row in trace if row.get("event") == "feedback"]
    groups = {}
    for label, items in (("action", actions_list), ("executed", executed_list), ("feedback", feedback_list)):
        audit.check([row.get("step") for row in items] == list(range(len(items))), label + " indexes contiguous from zero")
        groups[label] = {row["step"]: row for row in items}
    actions, executed, feedback = groups["action"], groups["executed"], groups["feedback"]
    clean = len(actions) == len(executed) == len(feedback)
    expected_events = ["reset"]
    route_rows = {row["step"]: row for row in trace if row.get("event") == "routing_decision"}
    switch_rows = {row["step"]: row for row in trace if row.get("event") == "controller_switch"}
    trace_controller = None
    for step in actions:
        control = actions[step]["diagnostics"]["controller"]
        if step % 5 == 0:
            expected_events.append("routing_decision")
            if trace_controller is not None and control != trace_controller:
                expected_events.append("controller_switch")
        expected_events.append("action")
        if step in executed:
            expected_events.append("executed")
        if step in feedback:
            expected_events.append("feedback")
        trace_controller = control
    audit.check([row["event"] for row in trace] == expected_events, "trace action/execution/feedback ordering")
    audit.check(len(executed) <= len(actions) <= len(executed) + 1, "at most one terminal unacknowledged action")
    audit.check(len(feedback) <= len(executed) <= len(feedback) + 1, "at most one terminal missing-feedback transition")
    execution_evidence = _execution_evidence(summary, state, actions, executed, audit)
    if not clean:
        audit.check(summary.get("status") not in ("success", "diagnostic_complete"), "partial terminal trace cannot claim completion")
        audit.notes.append("Terminal failure leaves an unexecuted action or missing final feedback; only recorded transitions are verified.")
    audit.check(summary.get("steps") == state.get("step") == len(executed), "acknowledged action count")
    audit.check(summary.get("measured_feedback_count") == len(feedback), "measured feedback count")
    audit.check(state.get("status") == summary.get("status"), "state and summary terminal outcome")
    initial = resets[0]["observation"] if resets else None
    observations = {0: initial} if initial else {}
    for step, result in executed.items():
        observations[step + 1] = result["observation"]

    request_paths = sorted((run / "requests").glob("*.request.json"))
    ledger = summary.get("budget", {})
    audit.check(ledger.get("billing") == billing and ledger.get("api_key_calls") == 0, "subscription-only shared ledger")
    audit.check(ledger.get("mock_response_calls" if offline else "api_calls") == len(request_paths), "single ledger request count")
    audit.check(summary.get("mock_response_calls" if offline else "api_calls") == len(request_paths), "summary request count")
    audit.check(ledger.get("subscription_cost_usd") is None and ledger.get("cost_usd_upper_bound") is None,
                "no invented subscription dollar price")
    if offline:
        audit.check(summary.get("api_calls") == ledger.get("api_calls") == ledger.get("subscription_inference_calls") == 0,
                    "scripted fixture has zero actual inference calls")
    history, history_feedback, accepted = [], 0, []
    usage = Counter()
    grid = GridSpec()
    expected_instructions = ChatGPTPlanHybridClient._instructions(grid, "hybrid") + (
        " Retain only the most recent 2 observation image pairs; all earlier textual observations, accepted complete "
        "hybrid output items (including opaque reasoning), and execution feedback remain in the shared conversation.")
    for index, path in enumerate(request_paths):
        record = read(path)
        prefix = f"request {index + 1}: "
        step = len(accepted) * 5
        audit.check(step in observations, prefix + "model request has actual preceding observation")
        while history_feedback < step and history_feedback in feedback:
            history.append({"role": "user", "content": [{"type": "input_text",
                "text": _canonical(feedback[history_feedback]["feedback"])}]})
            history_feedback += 1
        expected_input = _retained_input(history, observations[step], len(accepted)) if step in observations else None
        payload = record.get("payload", {})
        audit.check(record.get("request_ordinal") == index + 1 and record.get("attempt") == 1,
                    prefix + "one ordered attempt per routing request")
        audit.check(index == len(accepted), prefix + "no continuation or retry after rejected request")
        audit.check(record.get("prompt_version") == HYBRID_PROMPT_VERSION and record.get("mode") == "hybrid",
                    prefix + "new hybrid protocol provenance")
        audit.check(record.get("billing") == billing and record.get("configuration") == config.get("api"),
                    prefix + "shared subscription configuration")
        audit.check(record.get("configuration", {}).get("endpoint") == ENDPOINT, prefix + "public Responses endpoint")
        audit.check(payload.get("model") == "gpt-6-astra" and payload.get("stream") is True and payload.get("store") is False,
                    prefix + "exact Astra streaming subscription request")
        audit.check(not UNSUPPORTED_FIELDS.intersection(payload), prefix + "unsupported subscription fields omitted")
        audit.check(payload.get("instructions") == expected_instructions, prefix + "declared hybrid prompt exact match")
        audit.check(payload.get("text", {}).get("format") == {"type": "json_schema", "name": "astra_hybrid_history_v1",
            "strict": True, "schema": hybrid_output_schema()}, prefix + "strict hybrid schema with original nested interfaces")
        audit.check(payload.get("include") == ["reasoning.encrypted_content"], prefix + "opaque reasoning replay enabled")
        audit.check(payload.get("input") == expected_input,
                    prefix + "entire shared history, opaque output, observations, feedback and image retention exact")
        audit.check(record.get("history_stats_before_request") == _history_stats(history, len(accepted), history_feedback),
                    prefix + "history statistics agree with independent transcript")
        material = {"payload": payload, "grid": asdict(grid), "mode": "hybrid", "prompt_version": HYBRID_PROMPT_VERSION,
            "endpoint": ENDPOINT, "history_protocol": HISTORY_PROTOCOL, "image_history": 2}
        expected_key = hashlib.sha256(_canonical(material).encode()).hexdigest()
        audit.check(record.get("request_key") == expected_key, prefix + "request fingerprint recomputed")
        stem = path.name.removesuffix(".request.json")
        response_path, result_path = path.with_name(stem + ".response.json"), path.with_name(stem + ".result.json")
        result = read(result_path) if result_path.exists() else {}
        audit.check(result.get("status") in ("valid", "error"), prefix + "request has terminal validity result")
        if not response_path.exists():
            usage["unknown_usage_calls"] += 1
            audit.check(result.get("status") == "error", prefix + "missing response never accepted")
            continue
        response = read(response_path)
        counters = response.get("usage", {})
        if (isinstance(counters, dict) and type(counters.get("input_tokens")) is int
                and type(counters.get("output_tokens")) is int and counters["input_tokens"] >= 0 and counters["output_tokens"] >= 0):
            usage["input_tokens"] += counters["input_tokens"]
            usage["output_tokens"] += counters["output_tokens"]
            for section, name, total in (("input_tokens_details", "cached_tokens", "cached_input_tokens"),
                                         ("output_tokens_details", "reasoning_tokens", "reasoning_tokens")):
                value = counters.get(section, {}).get(name) if isinstance(counters.get(section), dict) else None
                if type(value) is int and value >= 0:
                    usage[total] += value
        else:
            usage["unknown_usage_calls"] += 1
        audit.check(response.get("billing") == billing, prefix + "response billing provenance")
        if result.get("status") != "valid":
            continue
        audit.check(response.get("status") == "completed" and not response.get("error"), prefix + "response completed before acceptance")
        audit.check(response.get("model") == "gpt-6-astra", prefix + "response model remains exactly Astra")
        try:
            value = AstraClient._extract(response)
        except Exception:
            audit.check(False, prefix + "one complete structured output can be extracted")
            continue
        _wrapper(value, grid, audit, prefix)
        accepted.append({"step": step, "prediction": value, "request_key": expected_key,
                         "request_path": path, "output": response["output"]})
        history = expected_input + copy.deepcopy(response["output"]) if expected_input is not None else history
    for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens", "unknown_usage_calls"):
        audit.check(ledger.get(key) == usage[key], "subscription usage reconciliation: " + key)

    selected = {record["step"]: record for record in accepted}
    planner, adapter = LocalPlanner(PlannerConfig()), ActionAdapter()
    active_field, active_wrapper, previous_controller, last_feedback = None, None, None, None
    progress, counts, switch_steps = [], Counter(), []
    for step, record in actions.items():
        prefix = f"action {step}: "
        diag = record["diagnostics"]
        observation = PolicyObservation.from_dict(observations[step])
        audit.check(record.get("state") == list(observation.state), prefix + "decision uses preceding measured state")
        refreshed = step % 5 == 0
        audit.check(diag.get("refreshed") is refreshed and diag.get("routing_decision_step") == step - step % 5,
                    prefix + "five-action routing commitment")
        if refreshed:
            counts["hybrid"] += 1
            audit.check(step in selected, prefix + "routing decision linked to an accepted model wrapper")
            if step not in selected:
                continue
            selection = selected[step]
            active_wrapper = selection["prediction"]
            audit.check(diag.get("hybrid_prediction") == active_wrapper, prefix + "complete accepted wrapper unchanged")
            audit.check(diag.get("api_request_key") == selection["request_key"], prefix + "correct model request fingerprint")
            audit.check(Path(diag.get("api_request_log", "")).name == selection["request_path"].name,
                        prefix + "correct model request artifact")
            if active_wrapper["controller"] == "coarse_field":
                counts["field"] += 1
                active_field = GridField(grid, observation.state[:3], FieldPrediction.from_dict(active_wrapper["field"], grid))
            else:
                counts["direct_chunk"] += 1
                active_field = None
        if active_wrapper is None:
            continue
        controller = active_wrapper["controller"]
        routing = {key: active_wrapper[key] for key in ("controller", "target_region", "near_target", "routing_reason")}
        audit.check(diag.get("controller") == controller,
                    prefix + "selected controller held through chunk")
        audit.check(diag.get("routing") == routing and diag.get("routing_is_model_prediction") is True,
                    prefix + "routing is model output, not simulator contact")
        changed = previous_controller is not None and previous_controller != controller
        audit.check(diag.get("mode_changed") is changed and diag.get("previous_controller") == previous_controller,
                    prefix + "controller switch metadata")
        if changed:
            switch_steps.append(step)
            audit.check(refreshed, prefix + "controller only switches at boundary")
            switch = switch_rows.get(step, {})
            audit.check(switch.get("from_controller") == previous_controller
                        and switch.get("to_controller") == controller and switch.get("routing") == routing
                        and switch.get("source") == "model_prediction", prefix + "controller switch trace matches routing")
        if refreshed:
            route = route_rows.get(step, {})
            audit.check(route.get("controller") == controller and route.get("routing") == routing
                        and route.get("previous_controller") == previous_controller
                        and route.get("source") == "model_prediction", prefix + "explicit routing trace matches accepted wrapper")
        audit.check(diag.get("field_refreshed") is (refreshed and controller == "coarse_field")
                    and diag.get("direct_chunk_refreshed") is (refreshed and controller == "contact_direct"),
                    prefix + "branch refresh flags")
        audit.check(diag.get("chunk_index") == step // 5 and diag.get("chunk_step") == step % 5,
                    prefix + "shared global chunk indexes")
        audit.check(diag.get("feedback") == last_feedback, prefix + "latest actual feedback before decision")
        audit.check(diag.get("history_stats", {}).get("feedback_records") == step,
                    prefix + "shared history includes every previous action")
        if controller == "coarse_field":
            counts["plan"] += 1
            plan = planner.plan(active_field, observation.state[:3])
            expected_action = adapter.adapt(plan.delta_xyz, active_wrapper["field"]["orientation_target_xyzw"],
                active_wrapper["field"]["gripper_command"], observation.state)
            audit.check(near(diag.get("plan"), plan.to_dict()), prefix + "full original local plan independently recomputed")
            audit.check(diag.get("status") == plan.status, prefix + "original local planner status")
            audit.check(diag.get("field_center") == list(active_field.center), prefix + "field retains its own fresh observation center")
            audit.check(near(record["action"], list(expected_action)), prefix + "original adapter and cross-controller grip intent")
            if refreshed:
                audit.check(diag.get("field_prediction") == active_wrapper["field"], prefix + "field contains exact model values and masks")
        else:
            audit.check(record["action"] == active_wrapper["direct"]["actions"][step % 5], prefix + "exact direct slice without planner or clipping")
            audit.check(not any(key in diag for key in ("plan", "field_center", "field_prediction")),
                        prefix + "direct mode has no field plan or stale field")
        if step in executed:
            result = executed[step]
            counts[controller] += 1
            audit.check(result.get("action") == record["action"] and result.get("controller") == controller,
                        prefix + "simulator acknowledged exactly the selected action")
            audit.check(result.get("next_state") == result["observation"]["observation/state"],
                        prefix + "returned observation matches measured state")
            audit.check(result.get("success") == result.get("sim_status", {}).get("success"),
                        prefix + "success remains evaluator metadata")
            expected = _base.expected_feedback(record["state"], record["action"], result["next_state"], step, progress)
            if step in feedback:
                audit.check(near(feedback[step]["feedback"], expected), prefix + "all feedback values independently recomputed")
                last_feedback = feedback[step]["feedback"]
            if record["action"][6] != 0:
                adapter._gripper = 1.0 if record["action"][6] > 0 else -1.0
        previous_controller = controller
    for name, expected in (("validated_hybrid_refreshes", counts["hybrid"]), ("validated_field_refreshes", counts["field"]),
        ("validated_direct_chunks", counts["direct_chunk"]), ("local_plan_count", counts["plan"]),
        ("coarse_field_actions", counts["coarse_field"]), ("contact_direct_actions", counts["contact_direct"]),
        ("controller_switches", len(switch_steps))):
        audit.check(summary.get(name) == expected, "summary count: " + name)
    unused_valid = len(accepted) - counts["hybrid"]
    audit.check(unused_valid == 0 or (unused_valid == 1 and summary.get("status") in ("infrastructure_error", "user_stop")),
                "every accepted wrapper used once, except documented terminal pre-action failure")
    if unused_valid:
        audit.notes.append("Final accepted model wrapper was not used in a logged action before termination.")
    for name, predicate in (("gripper_close_actions", lambda grip: grip > 0),
                            ("gripper_open_actions", lambda grip: grip < 0), ("gripper_neutral_actions", lambda grip: grip == 0)):
        audit.check(summary.get(name) == sum(predicate(row["action"][6]) for row in executed.values()), "summary count: " + name)
    path_length = sum(_base.norm([result["next_state"][axis] - actions[step]["state"][axis] for axis in range(3)])
                      for step, result in executed.items())
    audit.check(near(summary.get("path_length_m"), path_length), "measured end-effector path length")
    success_steps = [step for step, row in executed.items() if row.get("success")]
    reset_success = bool(resets and resets[0].get("sim_status", {}).get("success"))
    observed_success = bool(success_steps or reset_success)
    audit.check(summary.get("libero_success_observed") is observed_success, "summary success observation agrees with simulator replies")
    if success_steps:
        audit.check(success_steps == [max(executed)], "no action executed after evaluator success")
    completed = summary.get("status") == "success" and not offline
    audit.check(not completed or observed_success, "successful evaluation requires actual LIBERO success")
    audit.check(summary.get("evaluation_completed") is completed and summary.get("scored_episodes") == int(completed)
                and summary.get("success_rate") == (1.0 if completed else None)
                and summary.get("libero_success") == (True if completed else None), "completion and score semantics")
    first_close = next((step for step, row in executed.items() if row["action"][6] > 0), None)
    close = summary.get("first_close_event")
    audit.check((close is not None) == (first_close is not None), "first close metadata presence")
    if first_close is not None:
        audit.check(close.get("action_index") == first_close and close.get("action_ordinal") == first_close + 1
                    and close.get("controller") == executed[first_close]["controller"], "first close exact action identity")
        audit.check(close.get("full_closure_verified") is False and close.get("grasp_verified") is False,
                    "close command not falsely declared a grasp")
    audit.check(summary.get("status") != "first_gripper_close_executed", "hybrid full task is not stopped merely at first close")

    same_initial = None
    prior = ROOT / "runs/astra_field_history_001/trace.jsonl"
    if prior.is_file() and initial is not None:
        with prior.open() as handle:
            prior_reset = json.loads(next(handle))
        same_initial = initial == prior_reset["observation"]
        audit.check(same_initial, "initial state and both RGB views exactly match prior same-task field run")
    source_count = 0
    for filename, expected_hash in config.get("software", {}).get("source_sha256", {}).items():
        source = ROOT / "src/gpt_field" / filename
        audit.check(source.is_file() and sha(source) == expected_hash, "recorded source unchanged: " + filename)
        source_count += 1
    runner = config.get("runner", {})
    if runner.get("path"):
        audit.check(sha(ROOT / runner["path"]) == runner.get("sha256"), "recorded runner source unchanged")
    frozen_count = 0
    if frozen:
        for filename, expected_hash in read(frozen).items():
            audit.check(sha(ROOT / filename) == expected_hash, "frozen source unchanged: " + filename)
            frozen_count += 1
    video, frames = _video_audit(run, summary, actions, executed, feedback, audit)
    artifact_names = ["summary.json", "manifest.json", "run_state.json", "trace.jsonl", "episode.mp4", "episode.events.jsonl"]
    report = {"passed": not audit.issues, "checks": audit.checks, "issues": audit.issues, "notes": audit.notes,
        "source_run": str(run), "status": summary["status"], "billing": billing,
        "executed_actions": len(executed), "feedback_records": len(feedback), "trace_complete": clean,
        "execution_evidence": execution_evidence,
        "validated_hybrid_responses": len(accepted), "coarse_fields": counts["field"], "direct_chunks": counts["direct_chunk"],
        "independently_recomputed_local_plans": counts["plan"], "controller_switch_steps": switch_steps,
        "subscription_requests": 0 if offline else len(request_paths), "mock_requests": len(request_paths) if offline else 0,
        "usage_totals": dict(usage), "libero_success_observed": observed_success, "evaluation_completed": completed,
        "first_close_action_index": first_close, "same_initial_observation_as_prior_field_run": same_initial,
        "video": video, "event_frames": frames, "recorded_source_files_checked": source_count,
        "frozen_source_files_checked": frozen_count,
        "opaque_model_output_printed": False, "live_inference_calls_by_audit": 0, "simulator_actions_by_audit": 0,
        "limitations": "Checks recorded simulator success and data/control consistency; does not independently reconstruct physical object state or prove grasp.",
        "artifact_sha256": {name: sha(run / name) for name in artifact_names if (run / name).is_file()}}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "AUDIT.md").write_text(
        f"Hybrid subscription audit: {'PASS' if report['passed'] else 'FAIL'}\n\n"
        f"{audit.checks} checks; {len(audit.issues)} issues. {len(executed)} acknowledged actions, "
        f"{len(accepted)} validated hybrid responses, {counts['plan']} independently recomputed local plans.\n\n"
        f"Routing switched at zero-based actions {switch_steps}. Terminal status: `{summary['status']}`. "
        f"LIBERO success observed: {observed_success}; scored evaluation completed: {completed}.\n\n"
        "Full shared histories, opaque output equality, direct action slices, original field planning, feedback, "
        "subscription token accounting and the complete MP4 were checked offline. No model or simulator calls were made.\n\n"
        + ("Issues:\n\n" + "\n".join("- " + issue for issue in audit.issues) + "\n" if audit.issues else "")
        + ("Notes:\n\n" + "\n".join("- " + note for note in audit.notes) + "\n" if audit.notes else ""))
    print(json.dumps({key: report[key] for key in ("passed", "checks", "issues", "status", "executed_actions",
        "validated_hybrid_responses", "independently_recomputed_local_plans", "controller_switch_steps", "video")}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen-sources", type=Path)
    parser.add_argument("--allow-offline-smoke", action="store_true")
    args = parser.parse_args(argv)
    report = audit_run(args.run, args.output, args.frozen_sources, allow_offline_smoke=args.allow_offline_smoke)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
