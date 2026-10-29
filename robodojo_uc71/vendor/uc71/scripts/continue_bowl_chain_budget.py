"""Continue the three-segment bowl episode under its explicitly approved $150 total cap.

The current $100 segment must finish at budget_exceeded. --preflight performs
read-only provenance/accounting checks without credentials, simulation, or API
calls. Execution reconstructs all recorded actions before new inference.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import getpass
import importlib.util
import json
import math
import os
from pathlib import Path

from gpt_field.astra import AstraClient, AstraConfig

_spec = importlib.util.spec_from_file_location("_isolated_recovery_for_budget", Path(__file__).with_name("recover_bowl_continuation.py"))
recovery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(recovery)
engine = recovery.engine


def reject_competitors(paths):
    paths = {Path(path).resolve() for path in paths}
    if len({path.parent for path in paths}) != 1:
        raise ValueError("All history segments and new output must share the runs directory")
    if recovery.active_coordinators(paths):
        raise ValueError("A competing paid coordinator is still running")
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            args = [part.decode() for part in (process / "cmdline").read_bytes().split(b"\0") if part]
            if not any(Path(arg).name == Path(__file__).name for arg in args):
                continue
            cwd = (process / "cwd").resolve()
            for flag in ("--source", "--output"):
                if flag in args and args.index(flag) + 1 < len(args):
                    value = Path(args[args.index(flag) + 1])
                    if (value if value.is_absolute() else cwd / value).resolve() in paths:
                        raise ValueError("A competing budget-handoff coordinator is still running")
        except (OSError, UnicodeError):
            continue
    for manifest in next(iter(paths)).parent.glob("*/manifest.json"):
        if manifest.parent.resolve() in paths:
            continue
        config = recovery.read(manifest).get("configuration", {})
        parent = config.get("parent_run") if isinstance(config, dict) else None
        if isinstance(parent, str) and Path(parent).resolve() in paths:
            raise ValueError("An unaccounted sibling/descendant ledger already exists: " + str(manifest.parent))


def audit_completed_requests(source, identity, terminal, fresh):
    config = AstraConfig(**identity["api"])
    checker = AstraClient(config, api_key="offline-accounting-only")
    checker.load_budget_snapshot(terminal)
    if terminal["budget_bound_violated"]:
        raise ValueError("Terminal ledger violated its reservation envelope")
    expected = copy.deepcopy(identity["handoff_budget_snapshot"])
    reservation = checker.budget_snapshot()["reservation_per_call_usd"]
    requests = sorted((source / "api").glob("*.request.json"), key=lambda p: recovery.read(p)["request_ordinal"])
    if len(requests) != len(fresh):
        raise ValueError("Budget handoff requires one settled valid request per new field")
    for request_path, decision in zip(requests, fresh):
        request = recovery.read(request_path)
        expected["api_calls"] += 1
        if (request.get("request_ordinal") != expected["api_calls"]
                or request.get("configuration") != asdict(config)
                or request.get("accounting_version") != expected["snapshot_version"]
                or request.get("reservation_usd") != reservation
                or expected["cost_usd_upper_bound"] + reservation > config.max_cost_usd + 1e-9):
            raise ValueError("Current-segment request configuration/ordinal/reservation mismatch")
        prefix = request_path.name.removesuffix(".request.json")
        result = recovery.read(source / "api" / (prefix + ".result.json"))
        response = recovery.read(source / "api" / (prefix + ".response.json"))
        details = decision["diagnostics"]
        if (result.get("status") != "valid" or response.get("status") != "completed"
                or details.get("api_request_log") != str(request_path)
                or details.get("api_request_key") != request.get("request_key")):
            raise ValueError("Current-segment field/request/result linkage differs")
        recovery.equal(checker._extract(response), details["prediction"], "Recorded field differs from response")
        usage = response.get("usage", {})
        inp, out = usage.get("input_tokens"), usage.get("output_tokens")
        if (type(inp) is not int or type(out) is not int or not 0 <= inp <= config.max_input_tokens
                or not 0 <= out <= config.max_output_tokens):
            raise ValueError("Current-segment request has unverified usage")
        expected["input_tokens"] += inp
        expected["output_tokens"] += out
        for section, field, counter in (("input_tokens_details", "cached_tokens", "cached_input_tokens"),
                                         ("output_tokens_details", "reasoning_tokens", "reasoning_tokens")):
            value = usage.get(section, {}).get(field)
            if type(value) is int and value >= 0:
                expected[counter] += value
        expected["cost_usd_upper_bound"] += checker._token_cost_upper(inp, out)
        recovery.compare_ledger(result.get("budget", {}), expected, "Current-segment result ledger")
    recovery.compare_ledger(terminal, expected, "Current-segment terminal ledger")
    if terminal["cost_usd_upper_bound"] + reservation <= config.max_cost_usd + 1e-12:
        raise ValueError("Source did not exhaust its conservative cost reservation headroom")
    return reservation


def prepare_budget_handoff(source, max_total_cost_usd, authorization_path):
    if isinstance(max_total_cost_usd, bool) or max_total_cost_usd != 150:
        raise ValueError("This handoff requires explicit max_total_cost_usd=150")
    source = Path(source).resolve()
    state = recovery.read(source / "run_state.json")
    if state.get("status") != "budget_exceeded" or state.get("stage") != "terminal":
        raise ValueError("Current $100 segment must finish at terminal budget_exceeded before handoff")
    manifest = recovery.read(source / "manifest.json")
    identity = manifest["configuration"]
    if (identity.get("kind") != "same_episode_infrastructure_recovery"
            or identity.get("api", {}).get("max_cost_usd") != 100
            or engine.recovery.canonical_hash(identity) != manifest.get("identity_hash")):
        raise ValueError("Expected the intact $100 first-recovery manifest")
    paths = [Path(identity["root_run"]).resolve(), Path(identity["parent_run"]).resolve(), source]
    if len(set(paths)) != 3:
        raise ValueError("Expected exactly three distinct historical segments")
    reject_competitors(paths)
    authorization_path = Path(authorization_path).resolve()
    authorization = recovery.read(authorization_path)
    if (authorization.get("approved_total_api_budget_usd") != 150
            or authorization.get("supersedes_approved_total_api_budget_usd") != 100
            or authorization.get("includes_prior_episode_usage") is not True
            or authorization.get("includes_unresolved_request_reservations") is not True
            or authorization.get("max_actions", "missing") is not None
            or authorization.get("root_run") != paths[0].name or authorization.get("current_run") != source.name):
        raise ValueError("Authorization does not approve this episode's $150 total with prior unknown usage")
    # A private module's ancestry guard admits only the declared current segment;
    # the stronger complete-history guard above still rejects every other child.
    original_guard = recovery.reject_unaccounted_descendants
    recovery.reject_unaccounted_descendants = lambda *_args: reject_competitors(paths)
    try:
        ancestor = recovery.prepare_recovery(paths[1])
    finally:
        recovery.reject_unaccounted_descendants = original_guard
    recovery.equal(identity, recovery.recovery_identity(ancestor), "Current manifest differs from its verified ancestry")
    rows = [json.loads(line) for line in (source / "episodes.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("Expected one current-segment terminal result")
    row = rows[0]
    recovery.equal(row, recovery.read(source / "summary.json"), "Current summary/episode result differ")
    if (row.get("status") != "budget_exceeded" or row.get("evaluation_completed") is not False
            or row.get("libero_success") is not None or row.get("parent_identity_hash") != ancestor["source_identity_hash"]
            or Path(row.get("parent_run", "")).resolve() != ancestor["source"]
            or row.get("task_prompt") != engine.base.TASK or row.get("suite") != "libero_spatial"
            or row.get("task_id") != 0 or row.get("episode_index") != 0 or row.get("seed") != 7
            or row.get("method") != "field" or row.get("log") != "continuation.jsonl.gz"):
        raise ValueError("Current-segment terminal identity or outcome mismatch")
    start_offset, additional = ancestor["offset"], row.get("new_evaluated_actions")
    if type(additional) is not int or additional <= 0 or additional % 5:
        raise ValueError("Current segment must stop at a complete field-refresh boundary")
    total = start_offset + additional
    for record in (row, state):
        if (record.get("continuation_offset") != start_offset or record.get("reconstruction_actions") != start_offset
                or record.get("new_evaluated_actions") != additional or record.get("cumulative_actions") != total):
            raise ValueError("Current segment/cumulative action counts disagree")
    if state.get("step") != total:
        raise ValueError("Current terminal step differs from cumulative action count")
    proof = recovery.read(source / "replay_validation.json")
    if (proof.get("status") != "validated" or proof.get("completed_actions") != start_offset
            or proof.get("max_state_error") != 0 or proof.get("images_match") is not True
            or proof.get("state_checks") != start_offset + 1 or proof.get("image_pairs_checked") != start_offset + 1
            or proof.get("astra_requests_during_reconstruction") != 0):
        raise ValueError("Current segment lacks complete exact ancestry replay evidence")
    recovery.equal(proof.get("source_trace"), identity["source_trace"], "Current replay used different ancestry")
    trace, snapshot = engine.recovery.snapshot_records(source / "continuation.jsonl.gz")
    if not trace or not snapshot["gzip_complete"] or snapshot["partial_final_line_ignored"]:
        raise ValueError("Current trace is not a closed complete stream")
    start = trace[0]
    if (start.get("event") != "continuation_start" or start.get("continuation_offset") != start_offset
            or start.get("parent_identity_hash") != ancestor["source_identity_hash"]):
        raise ValueError("Current trace start identity differs")
    recovery.equal(start.get("observation"), ancestor["pairs"][-1][1]["observation"], "Current trace starts at a different observation")
    recovery.equal(start.get("sim_status"), ancestor["pairs"][-1][1]["sim_status"], "Current trace starts at a different simulator state")
    combined = [ancestor["reset"]] + [entry for pair in ancestor["pairs"] for entry in pair] + trace[1:]
    reset, pairs, selection = engine.recovery.select_prefix(combined, total)
    if selection["available_completed_actions"] != total or selection["trailing_unexecuted_decision_ignored"]:
        raise ValueError("Current trace has extra or unexecuted actions")
    fresh, stage = [], ancestor["last_stage"]
    for index, (decision, executed) in enumerate(pairs[start_offset:]):
        step, details = start_offset + index, decision.get("diagnostics", {})
        if (decision.get("new_action_index") != index or details.get("step") != step or details.get("method") != "field"
                or details.get("refreshed") is not (step % 5 == 0) or decision["action"][-1] != -1
                or executed.get("new_evaluated_actions") != index + 1 or executed.get("cumulative_actions") != step + 1):
            raise ValueError("Current policy cadence/gripper/action history differs")
        if details["refreshed"]:
            prediction = details.get("prediction", {})
            if prediction.get("gripper_command") not in ("open", "hold") or prediction != details.get("field_prediction"):
                raise ValueError("Current field or gripper state cannot be safely restored")
            stage = prediction.get("stage")
            fresh.append(decision)
        if not isinstance(stage, str) or not stage or details.get("stage") != stage:
            raise ValueError("Current stage history is incomplete")
    if pairs[-1][1]["success"] or pairs[-1][1]["sim_status"].get("terminated"):
        raise ValueError("Physical task already terminated")
    ledger = copy.deepcopy(state["budget"])
    recovery.compare_ledger(row["budget_after"], ledger, "Current row/run-state ledger")
    reservation = audit_completed_requests(source, identity, ledger, fresh)
    if row.get("additional_api_calls") != ledger["api_calls"] - ancestor["original_budget"]["api_calls"]:
        raise ValueError("Current additional request count differs")
    config = replace(AstraConfig(**identity["api"]), max_cost_usd=max_total_cost_usd)
    handoff = copy.deepcopy(ledger)
    handoff["configuration"] = asdict(config)
    handoff["max_cost_usd"] = max_total_cost_usd
    AstraClient(config, api_key="offline-accounting-only").load_budget_snapshot(handoff)
    if ledger["cost_usd_upper_bound"] + reservation > max_total_cost_usd:
        raise ValueError("New total budget lacks conservative request headroom")
    snapshot["available_completed_actions"] = additional
    chain = copy.deepcopy(ancestor["chain"]) + [{"role": "parent_segment", "run": str(source),
        "identity_hash": manifest["identity_hash"], "action_start": start_offset, "action_count": additional, "trace": snapshot}]
    names = ["manifest.json", "simulator.json", "episodes.jsonl", "summary.json", "run_state.json", "replay_validation.json"]
    if (source / "authorization.json").is_file():
        names.append("authorization.json")
    return {**ancestor, "source": source, "source_identity_hash": manifest["identity_hash"], "episode": row,
            "offset": total, "last_stage": stage, "reset": reset, "pairs": pairs, "api_config": config,
            "original_budget": ledger, "handoff_budget": handoff, "chain": chain,
            "source_snapshot": {"kind": "ordered_chain", "segments": chain, "available_completed_actions": total},
            "parent_artifact_sha256": {name: engine.recovery.sha256((source / name).read_bytes()) for name in names},
            "authorization": authorization, "authorization_source": {"path": str(authorization_path),
                "sha256": engine.recovery.sha256(authorization_path.read_bytes())}, "known_paths": paths}


def budget_identity(prepared):
    identity = recovery.original_identity(prepared)
    identity.update(kind="same_episode_chain_budget_continuation", chain=prepared["chain"],
                    root_run=prepared["root_run"], root_identity_hash=prepared["root_identity_hash"],
                    authorization=prepared["authorization"], authorization_source=prepared["authorization_source"],
                    execution_adapter="Frozen continuation executor; isolated provenance factory and verified three-segment history.")
    for name in ("recover_bowl_continuation.py", "continue_bowl_chain_budget.py"):
        identity["script_sha256"][name] = engine.recovery.sha256(Path(__file__).with_name(name).read_bytes())
    return identity


def execute_budget_handoff(prepared, output, **kwargs):
    if Path(output).resolve().parent != prepared["source"].parent:
        raise ValueError("New segment output must stay beside all historical runs")
    reject_competitors(prepared["known_paths"])
    engine.continuation_identity = budget_identity
    try:
        return engine.execute_continuation(prepared, output, **kwargs)
    finally:
        engine.continuation_identity = recovery.original_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    parser.add_argument("--max-total-cost-usd", type=float, required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if not args.preflight and not args.output:
        parser.error("--output is required for execution")
    prepared = prepare_budget_handoff(args.source, args.max_total_cost_usd, args.authorization)
    if args.preflight:
        ledger = prepared["original_budget"]
        print(json.dumps({"status": "ready", "continuation_offset": prepared["offset"],
            "api_calls": ledger["api_calls"], "next_request_ordinal": ledger["api_calls"] + 1,
            "unknown_usage_calls": ledger["unknown_usage_calls"], "cost_usd_upper_bound": ledger["cost_usd_upper_bound"],
            "max_total_cost_usd": 150, "last_stage": prepared["last_stage"], "history_segments": len(prepared["chain"])}), flush=True)
        return 0
    key = os.environ.pop("OPENAI_API_KEY", None) or getpass.getpass("API key (hidden, memory only): ").strip()
    result = execute_budget_handoff(prepared, args.output, api_key=key)
    return (0 if result["status"] == "success" else 130 if result["status"] == "user_stop"
            else 3 if result["status"] == "budget_exceeded" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
