"""Read-only, offset-aware audit of a bowl continuation segment.

Reads original and child evidence only. Never creates an API client, reads a
credential, launches a simulator, retries a request, or releases an unresolved
request's reservation. Live gzip/footer and in-flight API records remain partial.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from gpt_field.action import ActionAdapter
from gpt_field.astra import AstraClient, validate_prediction
from gpt_field.observation import decode_image
from gpt_field.schema import GridSpec
import gpt_field


spec = importlib.util.spec_from_file_location("bowl_audit_helpers", Path(__file__).with_name("audit_bowl_run.py"))
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def safe_error_code(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_]+", value) else None


def numeric_usage(response):
    usage = (response or {}).get("usage") or {}
    if not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens")):
        return None
    def detail(section, key):
        value = (usage.get(section) or {}).get(key, 0)
        return value if type(value) is int and value >= 0 else 0
    return {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
            "cached_input_tokens": detail("input_tokens_details", "cached_tokens"),
            "reasoning_tokens": detail("output_tokens_details", "reasoning_tokens")}


def cost_upper(config, input_tokens, output_tokens):
    rate = max(config["input_price_per_million"], config["cache_write_price_per_million"],
               config.get("cached_input_price_per_million") or 0)
    long = config["model"] == "gpt-6-astra" and input_tokens > 272000
    return (input_tokens * rate * (2 if long else 1)
            + output_tokens * config["output_price_per_million"] * (1.5 if long else 1)) / 1e6


def ledger_equal(left, right):
    """Adjacent snapshots include different wall-clock readings, not spend."""
    return ({key: value for key, value in left.items() if key != "elapsed_s"}
            == {key: value for key, value in right.items() if key != "elapsed_s"})


def parent_endpoint(path):
    """Stream the parent so its many candidate arrays need not remain in RAM."""
    observation = status = stage = None
    executions = 0
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") in ("reset", "continuation_start"):
                observation, status = row["observation"], row["sim_status"]
            elif row.get("event") == "executed":
                executions += 1
                observation, status = row["observation"], row["sim_status"]
            elif "diagnostics" in row:
                stage = row["diagnostics"].get("stage", stage)
    return observation, status, stage, executions


def images_equal(left, right):
    for key in ("observation/image", "observation/wrist_image"):
        a, b = decode_image(left[key]), decode_image(right[key])
        if a.size != b.size or a.tobytes() != b.tobytes():
            return False
    return True


def audit_history(config, parent, checks):
    """Validate immutable ordered history; never treat reconstruction as new work."""
    if config["kind"] == "same_episode_budget_continuation":
        trace = config["source_trace"]
        checks.check(sha(parent / trace["trace"]) == trace["snapshot_sha256"], "Parent trace changed since handoff")
        endpoint = parent_endpoint(parent / trace["trace"])
        checks.check(endpoint[3] == config["continuation_offset"], "Parent action history differs from continuation offset")
        return endpoint, []
    chain = config.get("chain") or []
    bundle = config["source_trace"]
    checks.check(bundle.get("kind") == "ordered_chain" and bundle.get("segments") == chain,
                 "Recovery source bundle differs from its ordered history")
    checks.check(bundle.get("available_completed_actions") == config["continuation_offset"], "Recovery history bundle action count mismatch")
    checks.check(len(chain) >= 2, "Chained continuation must preserve all historical segments")
    cumulative, previous_observation, previous_status, previous_identity = 0, None, None, None
    previous_ledger = {"api_calls": 0, "unknown_usage_calls": 0, "cost_usd_upper_bound": 0.0}
    summaries = []
    endpoint = None
    for index, segment in enumerate(chain):
        run = Path(segment["run"])
        manifest = helpers.read_json(run / "manifest.json") or {}
        identity = manifest.get("configuration", {})
        checks.check(manifest.get("identity_hash") == segment["identity_hash"] == digest(identity), f"History segment {index}: identity changed")
        checks.check(segment["action_start"] == cumulative, f"History segment {index}: global actions are discontinuous")
        trace = segment["trace"]
        path = run / trace["trace"]
        checks.check(sha(path) == trace["snapshot_sha256"], f"History segment {index}: trace changed")
        with gzip.open(path, "rt") as stream:
            first = json.loads(next(stream))
        endpoint = parent_endpoint(path)
        if index == 0:
            checks.check(str(run) == config["root_run"] and segment["identity_hash"] == config["root_identity_hash"], "Recovery root identity mismatch")
            checks.check(first.get("event") == "reset" and first.get("sim_status", {}).get("steps") == 0, "Root history lacks initial reset")
        else:
            checks.check(first.get("event") == "continuation_start" and first.get("continuation_offset") == cumulative, f"History segment {index}: start offset mismatch")
            checks.check(identity.get("parent_identity_hash") == previous_identity, f"History segment {index}: parent link mismatch")
            checks.check(Path(identity.get("parent_run", ".")) == Path(chain[index - 1]["run"]), f"History segment {index}: parent path mismatch")
            checks.check(identity.get("continuation_offset") == cumulative, f"History segment {index}: manifest offset mismatch")
            checks.check(identity.get("original_budget_snapshot") == previous_ledger, f"History segment {index}: parent ledger omitted or changed")
            expected_handoff = copy.deepcopy(previous_ledger)
            expected_handoff["configuration"]["max_cost_usd"] = identity["api"]["max_cost_usd"]
            expected_handoff["max_cost_usd"] = identity["api"]["max_cost_usd"]
            checks.check(identity.get("handoff_budget_snapshot") == expected_handoff, f"History segment {index}: accounting/configuration changed beyond total cap")
            checks.check(identity.get("frozen_policy") == config["frozen_policy"], f"History segment {index}: frozen controller protocol changed")
            if index > 1:
                checks.check(identity.get("source_trace", {}).get("segments") == chain[:index], f"History segment {index}: historical chain omitted or reordered")
            checks.check(first.get("sim_status") == previous_status, f"History segment {index}: simulator discontinuity")
            checks.check(first["observation"]["observation/state"] == previous_observation["observation/state"]
                         and images_equal(first["observation"], previous_observation), f"History segment {index}: observation discontinuity")
        cumulative += segment["action_count"]
        checks.check(endpoint[3] == segment["action_count"] and endpoint[1].get("steps") == cumulative, f"History segment {index}: action count differs from trace")
        raw_api, _ = read_api(run, identity["api"], previous_ledger, True, checks)
        actual_ledger = (helpers.read_json(run / "run_state.json") or {}).get("budget") or {}
        for key, value in raw_api["expected_cumulative_ledger"].items():
            checks.check(helpers.close(actual_ledger.get(key), value), f"History segment {index}: raw API ledger mismatch for {key}")
        previous_ledger = actual_ledger
        previous_observation, previous_status, previous_identity = endpoint[0], endpoint[1], segment["identity_hash"]
        summaries.append({"run": str(run), "action_start": segment["action_start"], "action_count": endpoint[3],
                          "api_counts": raw_api["counts"], "api_ordinals": raw_api["ordinals"],
                          "cumulative_cost_usd_upper_bound": actual_ledger.get("cost_usd_upper_bound"),
                          "cumulative_unknown_usage_calls": actual_ledger.get("unknown_usage_calls")})
    checks.check(cumulative == config["continuation_offset"], "Recovery omitted historical actions")
    checks.check(bool(chain) and Path(chain[-1]["run"]) == parent, "Recovery immediate parent is not its final historical segment")
    checks.check(previous_ledger == config["original_budget_snapshot"], "Recovery omitted historical API accounting")
    return endpoint, summaries


def read_api(directory, config, original, finished, checks):
    records, counts, usage_totals, models = {}, Counter(), Counter(), Counter()
    errors, ordinals = [], []
    observed_cost = unresolved_cost = 0.0
    reservation = cost_upper(config, config["max_input_tokens"], config["max_output_tokens"])
    for path in sorted((directory / "api").glob("*.request.json")):
        request = helpers.read_json(path)
        if request is None:
            checks.check(not finished, "Unparseable final API request record")
            counts["partial_request_records"] += 1
            continue
        stem = path.name.removesuffix(".request.json")
        response = helpers.read_json(path.with_name(stem + ".response.json"))
        result = helpers.read_json(path.with_name(stem + ".result.json"))
        if response is None and result and result.get("status") == "valid":
            # A live request may settle between the two reads above. Responses
            # are persisted before accepted result records.
            response = helpers.read_json(path.with_name(stem + ".response.json"))
        counts["new_request_records"] += 1
        ordinal = request.get("request_ordinal")
        ordinals.append(ordinal)
        checks.check(request.get("configuration") == config, f"Request {ordinal}: configuration differs from continuation manifest")
        checks.check(helpers.close(request.get("reservation_usd"), reservation), f"Request {ordinal}: reservation mismatch")
        prediction = None
        if response is not None:
            counts["confirmed_http_attempts"] += 1
            counts["response_records"] += 1
            models[str(response.get("model"))] += 1
            try:
                prediction = validate_prediction(AstraClient._extract(response), GridSpec(**request["grid"]), request["mode"])
                counts["schema_valid_predictions"] += 1
            except Exception:
                counts["responses_without_valid_prediction"] += 1
        else:
            message = str((result or {}).get("error", ""))
            match = re.search(r"OpenAI HTTP (\d+)", message)
            if match:
                counts["confirmed_http_attempts"] += 1
                counts["http_error_records"] += 1
                try:
                    body = json.loads((result or {}).get("raw_error_body") or "{}")
                except (ValueError, TypeError):
                    body = {}
                error = body.get("error", {}) if isinstance(body, dict) else {}
                errors.append({"request_ordinal": ordinal, "http_status": int(match.group(1)),
                               "code": safe_error_code(error.get("code")), "type": safe_error_code(error.get("type")),
                               "usage_evidence_present": isinstance(body.get("usage"), dict) if isinstance(body, dict) else False})
        usage = numeric_usage(response)
        if usage is None:
            counts["unknown_or_inflight_usage_attempts"] += 1
            unresolved_cost += reservation
        else:
            usage_totals.update(usage)
            observed_cost += cost_upper(config, usage["input_tokens"], usage["output_tokens"])
        if finished:
            checks.check(result is not None, f"Request {ordinal}: final result ledger missing")
        if result and result.get("status") == "valid":
            counts["accepted_predictions"] += 1
            checks.check(prediction is not None, f"Request {ordinal}: accepted result lacks valid response")
        if result:
            budget = result.get("budget") or {}
            checks.check(budget.get("configuration") == config, f"Request {ordinal}: result ledger configuration mismatch")
            for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens"):
                checks.check(budget.get(key) == original.get(key, 0) + usage_totals[key], f"Request {ordinal}: result ledger {key} mismatch")
            checks.check(budget.get("api_calls") == ordinal, f"Request {ordinal}: result ledger ordinal mismatch")
            checks.check(budget.get("unknown_usage_calls") == original["unknown_usage_calls"] + counts["unknown_or_inflight_usage_attempts"], f"Request {ordinal}: unresolved usage count mismatch")
            checks.check(helpers.close(budget.get("cost_usd_upper_bound"), original["cost_usd_upper_bound"] + observed_cost + unresolved_cost), f"Request {ordinal}: cumulative result cost mismatch")
        records[path.name] = {"request": request, "response": response, "result": result, "prediction": prediction}
    expected = list(range(original["api_calls"] + 1, original["api_calls"] + 1 + len(ordinals)))
    checks.check(sorted(ordinals) == expected, "Child API ordinals do not continue the carried ledger without gaps")
    cumulative = {key: original.get(key, 0) + usage_totals[key] for key in (
        "input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens")}
    cumulative.update(api_calls=original["api_calls"] + counts["new_request_records"],
                      unknown_usage_calls=original["unknown_usage_calls"] + counts["unknown_or_inflight_usage_attempts"],
                      cost_usd_upper_bound=original["cost_usd_upper_bound"] + observed_cost + unresolved_cost)
    return {"counts": dict(counts), "new_reported_usage": dict(usage_totals), "response_model_ids": dict(models),
            "ordinals": ordinals, "structured_http_errors": errors,
            "new_reported_usage_cost_upper_usd": observed_cost, "new_unresolved_reservations_usd": unresolved_cost,
            "reservation_per_call_usd": reservation, "expected_cumulative_ledger": cumulative,
            "note": "Unresolved or in-flight attempts retain full reservations. Live state ledgers may precede request settlement; strict cumulative reconciliation is terminal-only. Reasoning tokens are included in output tokens."}, records


def audit(directory, *, skip_video_probe=False):
    directory = Path(directory)
    checks = helpers.Checks()
    manifest = helpers.read_json(directory / "manifest.json") or {}
    config = manifest.get("configuration", {})
    if config.get("kind") not in ("same_episode_budget_continuation", "same_episode_infrastructure_recovery", "same_episode_chain_budget_continuation"):
        raise ValueError("Expected a same-episode continuation or recovery manifest")
    checks.check(manifest.get("identity_hash") == digest(config), "Child manifest identity hash mismatch")
    checks.check(config.get("task") == helpers.TASK and config.get("suite") == "libero_spatial"
                 and config.get("task_id") == 0 and config.get("episode_index") == 0 and config.get("seed") == 7,
                 "Continuation task/initial-state/seed identity mismatch")
    policy = config["frozen_policy"]
    checks.check(policy.get("method") == "field" and policy.get("seed") == 7 and policy.get("update_steps") == 5
                 and policy.get("grid") == GridSpec(size=9, spacing_m=0.02).to_dict()
                 and policy.get("planner", {}).get("horizon") == 3 and policy.get("planner", {}).get("step_m") == 0.01,
                 "Continuation field/controller protocol mismatch")
    package = Path(gpt_field.__file__).parent
    source_hashes = config["software"]["source_sha256"]
    checks.check(set(source_hashes) == {path.name for path in package.glob("*.py")}, "Frozen package source inventory changed")
    source_agreement = {}
    for name, expected in source_hashes.items():
        source_agreement[name] = (package / name).is_file() and sha(package / name) == expected
        checks.check(source_agreement[name], f"Frozen package source changed: {name}")
    script_agreement = {}
    for name, expected in config.get("script_sha256", {}).items():
        script_agreement[name] = Path(__file__).with_name(name).is_file() and sha(Path(__file__).with_name(name)) == expected
        checks.check(script_agreement[name], f"Frozen runner source changed: {name}")
    parent = Path(config["parent_run"])
    for name, expected in config["parent_artifact_sha256"].items():
        checks.check(sha(parent / name) == expected, f"Parent artifact changed: {name}")
    parent_endpoint_result, history = audit_history(config, parent, checks)
    parent_manifest = helpers.read_json(parent / "manifest.json") or {}
    checks.check(parent_manifest.get("identity_hash") == config["parent_identity_hash"], "Parent identity mismatch")
    parent_state = helpers.read_json(parent / "run_state.json") or {}
    original, handoff = config["original_budget_snapshot"], config["handoff_budget_snapshot"]
    checks.check(parent_state.get("budget") == original, "Carried budget differs from immutable parent ledger")
    expected_handoff = copy.deepcopy(original)
    expected_handoff["configuration"]["max_cost_usd"] = config["api"]["max_cost_usd"]
    expected_handoff["max_cost_usd"] = config["api"]["max_cost_usd"]
    checks.check(handoff == expected_handoff, "Handoff altered accounting values beyond the approved total cap")
    checks.check(handoff["configuration"] == config["api"], "Handoff API configuration mismatch")
    if config["kind"] == "same_episode_chain_budget_continuation":
        checks.check(config["api"]["max_cost_usd"] > original["configuration"]["max_cost_usd"], "Budget continuation did not increase total cap")
        checks.check(parent_state.get("status") == "budget_exceeded", "Budget continuation parent was not terminal at its prior cap")
        change = config.get("cost_cap_change") or {}
        checks.check(change.get("old_total_usd") == original["configuration"]["max_cost_usd"]
                     and change.get("new_total_usd") == config["api"]["max_cost_usd"]
                     and change.get("only_configuration_change") == "max_cost_usd", "Budget cap-change provenance mismatch")
        authorization = config.get("authorization") or {}
        approval_source = config.get("authorization_source") or {}
        checks.check(authorization.get("approved_total_api_budget_usd") == config["api"]["max_cost_usd"]
                     and authorization.get("supersedes_approved_total_api_budget_usd") == original["configuration"]["max_cost_usd"]
                     and authorization.get("includes_prior_episode_usage") is True
                     and authorization.get("includes_unresolved_request_reservations") is True,
                     "Budget authorization omitted carried usage or unresolved reservations")
        approval_path = Path(approval_source.get("path", "."))
        checks.check(approval_path.is_file() and sha(approval_path) == approval_source.get("sha256")
                     and helpers.read_json(approval_path) == authorization, "Budget authorization source changed")
    if config["kind"] == "same_episode_infrastructure_recovery":
        checks.check(handoff == original, "Infrastructure recovery changed the approved cap or carried ledger")
        checks.check(config.get("cost_cap_preserved") == config["api"]["max_cost_usd"], "Recovery total cap changed")
        reason = config.get("recovery_reason") or {}
        checks.check(reason.get("failed_request_ordinal") == original["api_calls"] and reason.get("next_request_ordinal") == original["api_calls"] + 1,
                     "Recovery request ordinal handoff mismatch")
        checks.check(original["unknown_usage_calls"] > 0 and helpers.close(reason.get("retained_unknown_reservation_usd"), original["reservation_per_call_usd"]),
                     "Recovery dropped unresolved request reservation")
    checks.check(config["simulator"]["max_steps"] is None, "Continuation unexpectedly has an action horizon")
    checks.check(config["api"]["max_api_calls"] is None and config["api"]["max_wall_time_s"] is None,
                 "Continuation unexpectedly has a call or wall-time horizon")
    offset = config["continuation_offset"]
    checks.check(type(offset) is int and offset > 0 and offset % 5 == 0, "Invalid refresh-boundary action offset")
    rows, rows_state = helpers.read_lines(directory / "episodes.jsonl")
    checks.check(len(rows) <= 1, "Continuation has multiple terminal outcomes")
    terminal = rows[0] if len(rows) == 1 else None
    finished = terminal is not None
    state = helpers.read_json(directory / "run_state.json") or {}
    trace_path = directory / "continuation.jsonl.gz"
    trace, trace_state = helpers.read_lines(trace_path, finished=finished)
    # Proof and API records precede trace writes. Reading them after the trace
    # avoids falsely rejecting a field that arrived during a live snapshot.
    proof = helpers.read_json(directory / "replay_validation.json") or {}
    api, requests = read_api(directory, config["api"], original, finished, checks)
    checks.check(trace_state != "invalid", "Closed continuation trace is invalid")
    starts = [r for r in trace if r.get("event") == "continuation_start"]
    decisions, executed, refreshes = [], [], []
    grid = GridSpec(**config["frozen_policy"]["grid"])
    adapter = ActionAdapter()
    prediction, measured = None, None
    stage = config["restored_policy_state"]["last_stage"]
    grippers, modes, stages, planner_statuses = Counter(), Counter(), Counter(), Counter()
    parent_observation, parent_sim_status, parent_stage, parent_actions = parent_endpoint_result
    checks.check(parent_sim_status.get("steps") == offset, "Parent endpoint differs from continuation offset")
    checks.check(parent_stage == stage, "Restored policy stage differs from last historical stage")
    if trace:
        checks.check(len(starts) == 1 and trace[0] is starts[0], "Trace must begin with exactly one continuation_start")
        start = starts[0] if starts else trace[0]
        measured = start["observation"]["observation/state"]
        checks.check(start.get("continuation_offset") == offset, "Trace start offset mismatch")
        checks.check(start.get("sim_status") == parent_sim_status, "Starting simulator status differs from parent endpoint")
        checks.check(measured == parent_observation["observation/state"], "Starting state differs from parent endpoint")
        checks.check(images_equal(start["observation"], parent_observation), "Starting images differ from parent endpoint")
    if requests or trace:
        checks.check(proof.get("status") == "validated", "Inference/trace exists without validated reconstruction")
        checks.check(proof.get("completed_actions") == offset, "Reconstruction did not replay the complete historical prefix")
        checks.check(proof.get("state_checks") == offset + 1 and proof.get("image_pairs_checked") == offset + 1, "Replay did not check reset and every historical observation")
        checks.check(proof.get("max_state_error") == 0 and proof.get("images_match") is True, "Replay was not exact")
        checks.check(proof.get("astra_requests_during_reconstruction") == 0, "Replay claimed inference during reconstruction")
        checks.check(proof.get("source_trace") == config["source_trace"], "Replay proof refers to different source history")
        checks.check(proof.get("same_simulator_retained_for_continuation") is True, "Continuation did not retain the validated simulator")
    for row in trace[1:]:
        if "diagnostics" in row:
            step = offset + len(decisions)
            details = row["diagnostics"]
            checks.check(row.get("step") == details.get("step") == step, "Global decision indices are not contiguous")
            checks.check(row.get("new_action_index") == len(decisions), f"Step {step}: local new-action index mismatch")
            checks.check(len(executed) == len(decisions), f"Step {step}: decision precedes previous execution")
            checks.check(row.get("state") == measured, f"Step {step}: policy did not receive latest measured state")
            fresh = step % 5 == 0
            checks.check(details.get("refreshed") is fresh, f"Step {step}: refresh cadence mismatch")
            decisions.append(row)
            if fresh:
                refreshes.append(row)
                prediction = details.get("field_prediction")
                checks.check(prediction == details.get("prediction"), f"Step {step}: transformed/missing field prediction")
                checks.check(details.get("field_center") == row["state"][:3], f"Step {step}: field center differs from measured state")
                record = requests.get(Path(details.get("api_request_log") or "missing").name)
                checks.check(record is not None, f"Step {step}: API provenance record missing")
                if record:
                    checks.check(record["prediction"] == details.get("prediction"), f"Step {step}: prediction differs from actual response")
                    checks.check((record.get("result") or {}).get("status") == "valid", f"Step {step}: response was not accepted")
                    checks.check(record["request"].get("request_key") == details.get("api_request_key"), f"Step {step}: request reference differs")
                    content = record["request"]["payload"]["input"][0]["content"]
                    sent = json.loads(content[0]["text"])
                    obs = row["observation"]
                    checks.check(sent.get("prompt") == obs.get("prompt") == helpers.TASK, f"Step {step}: task prompt mismatch")
                    checks.check(sent.get("observation/state") == obs.get("observation/state") == measured, f"Step {step}: API state differs from measured state")
                    checks.check(content[2]["image_url"] == obs["observation/image"] and content[4]["image_url"] == obs["observation/wrist_image"], f"Step {step}: sent images differ from observations")
                if prediction:
                    checks.check(details.get("stage_changed") == (stage is not None and prediction["stage"] != stage), f"Step {step}: restored stage transition mismatch")
                    stage = prediction["stage"]
                    grippers[prediction["gripper_command"]] += 1
                    modes[prediction["motion_mode"]] += 1
                    stages[stage] += 1
            if prediction:
                helpers.plan_audit(row, prediction, grid, adapter, checks)
            else:
                checks.check(False, f"Step {step}: no active prediction")
            planner_statuses[details.get("status")] += 1
        elif row.get("event") == "executed":
            step = offset + len(executed)
            checks.check(len(decisions) == len(executed) + 1, f"Step {step}: execution lacks paired decision")
            checks.check(row.get("step") == step, "Global executed indices are not contiguous")
            checks.check(row.get("new_evaluated_actions") == len(executed) + 1, f"Step {step}: executed local counter mismatch")
            checks.check(row.get("cumulative_actions") == row.get("sim_status", {}).get("steps") == step + 1, f"Step {step}: cumulative simulator counter mismatch")
            checks.check(row.get("next_state") == row.get("observation", {}).get("observation/state"), f"Step {step}: measured next-state mismatch")
            checks.check(row.get("success") == row.get("sim_status", {}).get("success"), f"Step {step}: success predicate mismatch")
            checks.check(row.get("sim_status", {}).get("horizon_reached") is False, f"Step {step}: unexpected finite horizon")
            executed.append(row)
            measured = row["next_state"]
        else:
            checks.check(False, "Unexpected continuation trace event")
    checks.check(len(decisions) - len(executed) in (0, 1), "More than one unexecuted decision")
    last_status = executed[-1]["sim_status"] if executed else parent_sim_status
    claimed_status = terminal.get("status") if terminal else state.get("status", "pending")
    completed = bool(terminal and terminal.get("status") == "success" and last_status.get("success") is True)
    if finished:
        checks.check(terminal.get("task_prompt") == helpers.TASK and terminal.get("suite") == "libero_spatial"
                     and terminal.get("task_id") == 0 and terminal.get("episode_index") == 0 and terminal.get("seed") == 7,
                     "Terminal episode identity differs from approved task")
        checks.check(terminal.get("parent_identity_hash") == config["parent_identity_hash"]
                     and Path(terminal.get("parent_run", ".")) == parent, "Terminal parent provenance differs from manifest")
        checks.check(state.get("status") == terminal.get("status") and state.get("stage") == "terminal", "Final state outcome differs from terminal row")
        checks.check(terminal.get("new_evaluated_actions") == len(executed), "Terminal new-action count differs from trace")
        checks.check(terminal.get("cumulative_actions") == offset + len(executed), "Historical replay was double-counted as new execution")
        checks.check(terminal.get("reconstruction_actions") == offset or not requests, "Terminal reconstruction count mismatch")
        checks.check(terminal.get("evaluation_completed") is completed, "Terminal evaluation-completed flag mismatch")
        checks.check(terminal.get("libero_success") is (True if completed else None), "Incomplete outcome incorrectly assigned success/failure")
        checks.check(terminal.get("additional_api_calls") == api["counts"].get("new_request_records", 0), "Additional API count differs from raw attempts")
        ledger = terminal.get("budget_after") or {}
        checks.check(ledger.get("configuration") == config["api"], "Final ledger configuration differs from approved continuation")
        for key, expected in api["expected_cumulative_ledger"].items():
            checks.check(helpers.close(ledger.get(key), expected), f"Cumulative accounting mismatch: {key}")
        checks.check(ledger_equal(state.get("budget") or {}, ledger), "Final run state and episode ledgers disagree")
        checks.check((state.get("budget") or {}).get("elapsed_s", -1) >= ledger.get("elapsed_s", 0), "Final run state elapsed time precedes episode snapshot")
        checks.check(helpers.close(terminal.get("additional_cost_usd_upper_bound"), ledger.get("cost_usd_upper_bound", 0) - original["cost_usd_upper_bound"]), "Additional cost omitted carried or unresolved spend")
        checks.check(ledger.get("cost_usd_upper_bound", 0) <= config["api"]["max_cost_usd"], "Cumulative accounting exceeded approved total cap")
        if completed:
            checks.check(last_status.get("terminated") is True, "Successful simulator did not terminate")
        else:
            checks.check(terminal.get("status") != "timeout", "Action-unbounded continuation incorrectly recorded a task timeout")
    else:
        ledger = state.get("budget") or {}
    video = helpers.video_audit(trace_path, checks, finished and bool(trace), decisions, executed, refreshes, skip_video_probe)
    if trace:
        video_rows, _ = helpers.read_lines(directory / "continuation.events.jsonl", finished=finished)
        if video_rows:
            checks.check(video_rows[0].get("event") == "observation" and video_rows[0].get("step") == offset, "Video starts at wrong historical offset")
    resets = [{"observation": starts[0]["observation"]}] if starts else []
    motion = helpers.motion_summary(resets, decisions, executed)
    return {"audit_status": "issues_found" if checks.issue_count else "complete" if finished else "partial",
            "evaluation_status": "complete" if completed else "incomplete" if finished else "running",
            "outcome_status": claimed_status, "success_rate": 1.0 if completed else None,
            "directory": str(directory), "parent_directory": str(parent), "action_offset": offset,
            "historical_segments": history,
            "historical_reconstruction_actions": proof.get("completed_actions"), "new_plans": len(decisions),
            "new_executions": len(executed), "cumulative_actions": offset + len(executed),
            "new_field_refreshes": len(refreshes), "refresh_global_steps": [r["step"] for r in refreshes],
            "trace_read_state": trace_state, "terminal_row_read_state": rows_state,
            "replay_validation": {key: proof.get(key) for key in ("status", "state_checks", "image_pairs_checked", "max_state_error", "images_match")},
            "frozen_source_agreement": {"package_files": source_agreement, "runner_files": script_agreement},
            "api": api, "recorded_cumulative_budget": ledger, "gripper_prediction_commands": dict(grippers),
            "executed_gripper_commands": dict(Counter(str(decisions[i]["action"][-1]) for i in range(min(len(executed), len(decisions))))),
            "prediction_motion_modes": dict(modes), "prediction_stages": dict(stages), "planner_statuses": dict(planner_statuses),
            "api_latency": helpers.latency([r["diagnostics"].get("api_latency_s") for r in refreshes]),
            "new_observed_motion": motion, "terminal_simulator_status": last_status, "video": video,
            **checks.result(),
            "scope": "One continuation segment, not a second evaluated episode. The original action prefix is reconstructed but excluded from new-action counts. Unknown API usage retains full reservation; no invoice adjustment is inferred from HTTP error codes. No network or simulator activity occurs in this audit."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--output")
    parser.add_argument("--skip-video-probe", action="store_true")
    args = parser.parse_args(argv)
    report = audit(args.run, skip_video_probe=args.skip_video_probe)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({key: report[key] for key in ("audit_status", "evaluation_status", "outcome_status", "issue_count", "new_executions", "cumulative_actions")}
                         | {"output": str(path), "api_counts": report["api"]["counts"]}))
    else:
        print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if report["issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
