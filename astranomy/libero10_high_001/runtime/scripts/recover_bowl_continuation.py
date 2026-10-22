"""Recover one HTTP-503-stopped first continuation under its unchanged approved cap.

--preflight is read-only and never reads credentials or starts a simulator.
Execution reconstructs the original and first-child history exactly, retains
unknown request charges, then uses the frozen continuation executor for new work.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import getpass
import importlib.util
import json
import math
import os
from pathlib import Path

from gpt_field.astra import AstraClient, AstraConfig


_spec = importlib.util.spec_from_file_location("_isolated_frozen_continuation", Path(__file__).with_name("continue_bowl_episode.py"))
engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(engine)
original_identity = engine.continuation_identity


def read(path):
    return json.loads(Path(path).read_text())


def equal(actual, expected, label):
    if engine.recovery.canonical_hash(actual) != engine.recovery.canonical_hash(expected):
        raise ValueError(label)


def active_coordinators(paths):
    """Reject competing CLI coordinators for this chain without exposing argv."""
    names = {"evaluate_bowl_until_success.py", "continue_bowl_episode.py", "recover_bowl_continuation.py"}
    wanted = {str(Path(path).resolve()) for path in paths}
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = [value.decode() for value in (entry / "cmdline").read_bytes().split(b"\0") if value]
            scripts = [Path(value).name for value in args if Path(value).name in names]
            if not scripts:
                continue
            cwd = (entry / "cwd").resolve()
            for flag in ("--source", "--output"):
                if flag in args and args.index(flag) + 1 < len(args):
                    candidate = Path(args[args.index(flag) + 1])
                    candidate = candidate if candidate.is_absolute() else cwd / candidate
                    if str(candidate.resolve()) in wanted:
                        active.append({"pid": int(entry.name), "script": scripts[0]})
                        break
        except (OSError, UnicodeError, ValueError):
            continue
    return active


def compare_ledger(actual, expected, label):
    for key in engine.LEDGER_VALUES:
        if key == "cost_usd_upper_bound":
            if not math.isclose(actual.get(key, -1), expected[key], rel_tol=0, abs_tol=1e-9):
                raise ValueError(label + ": cost mismatch")
        elif actual.get(key) != expected[key]:
            raise ValueError(label + ": " + key + " mismatch")


def reject_unaccounted_descendants(source, root):
    """All recovery outputs stay beside their source so prior spend is discoverable."""
    source, root = Path(source).resolve(), Path(root).resolve()
    if source.parent != root.parent:
        raise ValueError("Recovery supports adjacent original/child run directories only")
    for manifest_path in source.parent.glob("*/manifest.json"):
        if manifest_path.parent.resolve() in (source, root):
            continue
        try:
            config = read(manifest_path).get("configuration", {})
            parent = config.get("parent_run") if isinstance(config, dict) else None
            if isinstance(parent, str) and Path(parent).resolve() in (source, root):
                raise ValueError("An existing descendant/sibling ledger must be reconciled before another recovery: "
                                 + str(manifest_path.parent))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot exclude prior continuation spend from an unreadable adjacent manifest") from exc


def audit_child_ledger(source, identity, terminal, fresh_decisions):
    """Recompute this segment's deltas from saved response usage, without requests."""
    api = AstraConfig(**identity["api"])
    checker = AstraClient(api, api_key="offline-ledger-validation")
    checker.load_budget_snapshot(terminal)
    if terminal["budget_bound_violated"]:
        raise ValueError("Cannot recover a violated budget envelope")
    evolving = copy.deepcopy(identity["handoff_budget_snapshot"])
    reservation = checker.budget_snapshot()["reservation_per_call_usd"]
    requests = sorted((source / "api").glob("*.request.json"), key=lambda p: read(p)["request_ordinal"])
    if len(requests) != len(fresh_decisions) + 1:
        raise ValueError("Recovery requires valid child fields followed by exactly one failed HTTP attempt")
    first = evolving["api_calls"] + 1
    for index, request_path in enumerate(requests):
        request = read(request_path)
        if (request.get("request_ordinal") != first + index or request.get("configuration") != asdict(api)
                or request.get("accounting_version") != evolving["snapshot_version"]
                or request.get("reservation_usd") != reservation):
            raise ValueError("Child request ordinal/configuration/reservation provenance mismatch")
        if evolving["cost_usd_upper_bound"] + reservation > api.max_cost_usd + 1e-9:
            raise ValueError("Historical child request exceeded the approved budget reservation")
        prefix = request_path.name.removesuffix(".request.json")
        result = read(source / "api" / (prefix + ".result.json"))
        response_path = source / "api" / (prefix + ".response.json")
        evolving["api_calls"] += 1
        if index < len(fresh_decisions):
            response = read(response_path)
            if result.get("status") != "valid" or response.get("status") != "completed":
                raise ValueError("Only accepted child fields may precede the failed request")
            details = fresh_decisions[index]["diagnostics"]
            if details.get("api_request_log") != str(request_path) or details.get("api_request_key") != request.get("request_key"):
                raise ValueError("Accepted child field does not reference its exact request log")
            equal(checker._extract(response), details["prediction"], "Child field and response prediction differ")
            usage = response.get("usage", {})
            inp, out = usage.get("input_tokens"), usage.get("output_tokens")
            if (type(inp) is not int or type(out) is not int or not 0 <= inp <= api.max_input_tokens
                    or not 0 <= out <= api.max_output_tokens):
                raise ValueError("Accepted child field has unresolved or invalid usage")
            evolving["input_tokens"] += inp
            evolving["output_tokens"] += out
            for section, field, counter in (("input_tokens_details", "cached_tokens", "cached_input_tokens"),
                                             ("output_tokens_details", "reasoning_tokens", "reasoning_tokens")):
                value = usage.get(section, {}).get(field)
                if type(value) is int and value >= 0:
                    evolving[counter] += value
            evolving["cost_usd_upper_bound"] += checker._token_cost_upper(inp, out)
        else:
            try:
                body = json.loads(result.get("raw_error_body") or "{}")
            except ValueError as exc:
                raise ValueError("Failed request lacks its HTTP-503 server error evidence") from exc
            if (response_path.exists() or result.get("status") != "error" or result.get("error_type") != "APIError"
                    or "HTTP 503" not in result.get("error", "")
                    or body.get("error", {}).get("type") != "service_unavailable_error"
                    or body.get("error", {}).get("code") != "server_is_overloaded"):
                raise ValueError("Recovery is restricted to one unresolved HTTP 503 server overload")
            evolving["unknown_usage_calls"] += 1
            evolving["cost_usd_upper_bound"] += reservation
        compare_ledger(result.get("budget", {}), evolving, "Child raw-result ledger")
    compare_ledger(terminal, evolving, "Child terminal ledger")
    if terminal["cost_usd_upper_bound"] + reservation > api.max_cost_usd + 1e-9:
        raise ValueError("Approved total cap cannot reserve another request without dropping unknown usage")
    return {"http_status": 503, "failed_request_ordinal": terminal["api_calls"],
            "next_request_ordinal": terminal["api_calls"] + 1,
            "retained_unknown_reservation_usd": reservation,
            "next_reserved_total_usd": terminal["cost_usd_upper_bound"] + reservation}


def prepare_recovery(source):
    source = Path(source).resolve()
    manifest = read(source / "manifest.json")
    identity = manifest["configuration"]
    if (identity.get("kind") != "same_episode_budget_continuation"
            or engine.recovery.canonical_hash(identity) != manifest.get("identity_hash")):
        raise ValueError("Recovery requires an intact first-continuation manifest")
    if active_coordinators([source, identity["parent_run"]]):
        raise ValueError("A competing coordinator is still live for this episode chain")
    reject_unaccounted_descendants(source, identity["parent_run"])
    root = engine.prepare_source(identity["parent_run"], identity["api"]["max_cost_usd"])
    equal(identity, original_identity(root), "First child no longer matches frozen parent provenance")
    authorization = read(source / "authorization.json")
    if (authorization.get("approved_total_api_budget_usd") != root["api_config"].max_cost_usd
            or authorization.get("includes_prior_episode_usage") is not True):
        raise ValueError("Child lacks approval evidence for its unchanged total budget")
    records = [json.loads(line) for line in (source / "episodes.jsonl").read_text().splitlines() if line.strip()]
    state = read(source / "run_state.json")
    if len(records) != 1:
        raise ValueError("Expected one terminal first-child result")
    row = records[0]
    equal(row, read(source / "summary.json"), "Child summary and terminal row differ")
    if (row.get("status") != "infrastructure_error" or state.get("status") != "infrastructure_error"
            or state.get("stage") != "terminal" or "HTTP 503" not in row.get("error", "")
            or row.get("evaluation_completed") is not False or row.get("libero_success") is not None):
        raise ValueError("Child is not a terminal, incomplete HTTP-503 failure")
    if (row.get("parent_identity_hash") != root["source_identity_hash"]
            or Path(row.get("parent_run", "")).resolve() != root["source"]
            or row.get("task_id") != 0 or row.get("episode_index") != 0 or row.get("seed") != 7
            or row.get("method") != "field" or row.get("suite") != "libero_spatial"
            or row.get("task_prompt") != engine.base.TASK):
        raise ValueError("Child episode/parent identity mismatch")
    additional = row.get("new_evaluated_actions")
    offset = root["offset"]
    if type(additional) is not int or additional <= 0 or additional % 5:
        raise ValueError("Child must stop on a positive complete field-refresh boundary")
    total = offset + additional
    for value in (row, state):
        if (value.get("continuation_offset") != offset or value.get("reconstruction_actions") != offset
                or value.get("new_evaluated_actions") != additional or value.get("cumulative_actions") != total):
            raise ValueError("Child segment/cumulative action counts disagree")
    if state.get("step") != total or row.get("log") != "continuation.jsonl.gz":
        raise ValueError("Child final step or trace identity mismatch")
    proof = read(source / "replay_validation.json")
    if (proof.get("status") != "validated" or proof.get("completed_actions") != offset
            or proof.get("max_state_error") != 0 or proof.get("images_match") is not True
            or proof.get("state_checks") != offset + 1 or proof.get("image_pairs_checked") != offset + 1
            or proof.get("astra_requests_during_reconstruction") != 0):
        raise ValueError("First child lacks complete exact replay validation")
    child_rows, child_snapshot = engine.recovery.snapshot_records(source / "continuation.jsonl.gz")
    if not child_snapshot["gzip_complete"] or child_snapshot["partial_final_line_ignored"] or not child_rows:
        raise ValueError("Child trace must be a closed complete gzip stream")
    start = child_rows[0]
    if (start.get("event") != "continuation_start" or start.get("continuation_offset") != offset
            or start.get("parent_identity_hash") != root["source_identity_hash"]):
        raise ValueError("Child trace start identity mismatch")
    equal(start.get("observation"), root["pairs"][-1][1]["observation"], "Child starts from a different parent observation")
    equal(start.get("sim_status"), root["pairs"][-1][1]["sim_status"], "Child starts from a different parent simulator status")
    combined = [root["reset"]] + [entry for pair in root["pairs"] for entry in pair] + child_rows[1:]
    reset, pairs, selection = engine.recovery.select_prefix(combined, total)
    if selection["available_completed_actions"] != total or selection["trailing_unexecuted_decision_ignored"]:
        raise ValueError("Combined trace has extra or unexecuted actions")
    fresh, last_stage = [], root["last_stage"]
    for index, (decision, executed) in enumerate(pairs[offset:]):
        details = decision.get("diagnostics", {})
        step = offset + index
        if (decision.get("new_action_index") != index or details.get("step") != step
                or details.get("method") != "field" or details.get("refreshed") is not (step % 5 == 0)
                or decision["action"][-1] != -1
                or executed.get("new_evaluated_actions") != index + 1
                or executed.get("cumulative_actions") != step + 1):
            raise ValueError("Child cadence/gripper/global-action history mismatch")
        if details["refreshed"]:
            prediction = details.get("prediction", {})
            if prediction.get("gripper_command") not in ("open", "hold") or prediction != details.get("field_prediction"):
                raise ValueError("Child gripper or field prediction cannot be safely restored")
            last_stage = prediction.get("stage")
            fresh.append(decision)
        if not isinstance(last_stage, str) or not last_stage or details.get("stage") != last_stage:
            raise ValueError("Child stage history mismatch")
    if pairs[-1][1]["success"] or pairs[-1][1]["sim_status"].get("terminated"):
        raise ValueError("Physical episode already terminated")
    ledger = copy.deepcopy(state["budget"])
    compare_ledger(row["budget_after"], ledger, "Child row/state budget")
    reason = audit_child_ledger(source, identity, ledger, fresh)
    if row.get("additional_api_calls") != ledger["api_calls"] - root["original_budget"]["api_calls"]:
        raise ValueError("Child additional API-call count mismatch")
    child_snapshot["available_completed_actions"] = additional
    chain = [{"role": "root", "run": str(root["source"]), "identity_hash": root["source_identity_hash"],
              "action_start": 0, "action_count": offset, "trace": root["source_snapshot"]},
             {"role": "parent_segment", "run": str(source), "identity_hash": manifest["identity_hash"],
              "action_start": offset, "action_count": additional, "trace": child_snapshot}]
    return {**root, "source": source, "source_identity_hash": manifest["identity_hash"],
            "episode": row, "offset": total, "last_stage": last_stage, "reset": reset, "pairs": pairs,
            "original_budget": ledger, "handoff_budget": copy.deepcopy(ledger),
            "source_snapshot": {"kind": "ordered_chain", "segments": chain, "available_completed_actions": total},
            "parent_artifact_sha256": {name: engine.recovery.sha256((source / name).read_bytes()) for name in (
                "manifest.json", "simulator.json", "episodes.jsonl", "summary.json", "run_state.json",
                "authorization.json", "replay_validation.json")},
            "chain": chain, "root_run": str(root["source"]), "root_identity_hash": root["source_identity_hash"],
            "recovery_reason": reason, "authorization": authorization}


def recovery_identity(prepared):
    identity = original_identity(prepared)
    identity.pop("cost_cap_change")
    identity.update(kind="same_episode_infrastructure_recovery", chain=prepared["chain"],
                    root_run=prepared["root_run"], root_identity_hash=prepared["root_identity_hash"],
                    recovery_reason=prepared["recovery_reason"], cost_cap_preserved=prepared["api_config"].max_cost_usd,
                    authorization=prepared["authorization"],
                    execution_adapter="Frozen continuation executor; this isolated module changes only its provenance factory.")
    identity["script_sha256"]["recover_bowl_continuation.py"] = engine.recovery.sha256(Path(__file__).read_bytes())
    return identity


def execute_recovery(prepared, output, **kwargs):
    if Path(output).resolve().parent != prepared["source"].parent:
        raise ValueError("Recovery output must stay beside source segments for descendant-ledger discovery")
    reject_unaccounted_descendants(prepared["source"], prepared["root_run"])
    if active_coordinators([prepared["source"], prepared["root_run"]]):
        raise ValueError("A competing coordinator is live; recovery dispatch refused")
    # The engine was loaded into a private module object. Only its metadata
    # factory is substituted; policy, simulation, accounting, and executor code
    # remain byte-for-byte identical to the frozen first continuation.
    engine.continuation_identity = recovery_identity
    try:
        return engine.execute_continuation(prepared, output, **kwargs)
    finally:
        engine.continuation_identity = original_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if not args.preflight and not args.output:
        parser.error("--output is required for execution")
    prepared = prepare_recovery(args.source)
    if args.preflight:
        print(json.dumps({"status": "ready", "continuation_offset": prepared["offset"],
            "api_calls": prepared["original_budget"]["api_calls"],
            "cost_usd_upper_bound": prepared["original_budget"]["cost_usd_upper_bound"],
            "unknown_usage_calls": prepared["original_budget"]["unknown_usage_calls"],
            "max_total_cost_usd": prepared["api_config"].max_cost_usd,
            "last_stage": prepared["last_stage"], "recovery_reason": prepared["recovery_reason"]}), flush=True)
        return 0
    key = os.environ.pop("OPENAI_API_KEY", None) or getpass.getpass("API key (hidden, memory only): ").strip()
    result = execute_recovery(prepared, args.output, api_key=key)
    return (0 if result["status"] == "success" else 130 if result["status"] == "user_stop"
            else 3 if result["status"] == "budget_exceeded" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
