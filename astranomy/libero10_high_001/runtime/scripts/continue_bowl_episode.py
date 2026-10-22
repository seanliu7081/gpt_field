"""Continue the same budget-stopped bowl episode after explicit budget approval.

The required --max-total-cost-usd is a finite TOTAL budget including prior spend.
The caller must obtain user approval before launching this paid continuation.
All historical actions are reconstructed and verified before any new inference.
Generic --resume is deliberately unsupported: this segment has cumulative API
ordinals and a carried ledger, not an independent zero-cost episode.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import getpass
import gzip
import importlib.util
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from gpt_field.astra import APIError, AstraClient, AstraConfig, BudgetExceeded, SchemaError
from gpt_field.experiment import write_json
from gpt_field.libero import SimProcess, SimulatorError
from gpt_field.planner import PlannerConfig
from gpt_field.schema import GridSpec
from gpt_field.video import EpisodeVideoRecorder


def _script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _script("evaluate_bowl_until_success")
recovery = _script("replay_bowl_state")
LEDGER_VALUES = ("api_calls", "input_tokens", "output_tokens", "cached_input_tokens",
                 "reasoning_tokens", "unknown_usage_calls", "cost_usd_upper_bound", "budget_bound_violated",
                 "configuration", "snapshot_version")


def prepare_source(source, max_total_cost_usd):
    """Read-only preflight; neither credentials nor an inference transport are used."""
    if (isinstance(max_total_cost_usd, bool) or not isinstance(max_total_cost_usd, (int, float))
            or not math.isfinite(max_total_cost_usd) or max_total_cost_usd <= 0):
        raise ValueError("An explicit finite positive max_total_cost_usd is required")
    source = Path(source).resolve()
    config, simulator_identity, identity_hash = recovery.load_identity(source)
    manifest = json.loads((source / "manifest.json").read_text())
    identity = manifest["configuration"]
    state = json.loads((source / "run_state.json").read_text())
    records = [json.loads(line) for line in (source / "episodes.jsonl").read_text().splitlines() if line.strip()]
    if state.get("status") != "budget_exceeded" or len(records) != 1 or records[0].get("status") != "budget_exceeded":
        raise ValueError("Source must have one terminal budget_exceeded episode and terminal run state")
    episode = records[0]
    if (episode.get("suite") != "libero_spatial" or episode.get("task_id") != 0
            or episode.get("episode_index") != 0 or episode.get("seed") != 7
            or episode.get("task_prompt") != base.TASK or episode.get("method") != "field"):
        raise ValueError("Source terminal episode identity does not match the requested bowl episode")
    offset = episode.get("steps")
    if type(offset) is not int or offset <= 0 or offset % 5:
        raise ValueError("Source must stop at a positive five-action field-refresh boundary")
    old_api = AstraConfig(**identity["api"])
    base.validate_config(old_api)
    if max_total_cost_usd <= old_api.max_cost_usd:
        raise ValueError("New total cost budget must be higher than the source total cost budget")
    expected_policy = base.reference_metadata(old_api)
    if (recovery.canonical_hash(identity.get("reference")) != recovery.canonical_hash(expected_policy) or identity.get("method") != "field"
            or identity.get("grid") != GridSpec(size=9, spacing_m=0.02).to_dict()
            or recovery.canonical_hash(identity.get("planner")) != recovery.canonical_hash(asdict(PlannerConfig(step_m=0.01, horizon=3)))
            or identity.get("update_steps") != 5 or identity.get("policy_seed") != 7):
        raise ValueError("Frozen runner/policy/grid/planner/cadence provenance changed")
    package = Path(recovery.libero_module.__file__).parent
    source_hashes = identity["software"]["source_sha256"]
    if set(source_hashes) != {path.name for path in package.glob("*.py")}:
        raise ValueError("Frozen package source inventory changed")
    for name, expected_hash in source_hashes.items():
        if recovery.sha256((package / name).read_bytes()) != expected_hash:
            raise ValueError(f"Frozen package source changed: {name}")
    if episode.get("log") != recovery.TRACE_NAME:
        raise ValueError("Continuation only supports the original single attempt trace")
    rows, snapshot = recovery.snapshot_records(source / recovery.TRACE_NAME)
    reset, pairs, selection = recovery.select_prefix(rows, offset)
    if (not snapshot["gzip_complete"] or snapshot["partial_final_line_ignored"]
            or selection["available_completed_actions"] != offset
            or selection["trailing_unexecuted_decision_ignored"]):
        raise ValueError("Terminal source trace must be closed and contain exactly the executed prefix")
    last_stage = None
    for decision, executed in pairs:
        index = decision["step"]
        details = decision.get("diagnostics", {})
        if (details.get("method") != "field" or details.get("step") != index
                or details.get("refreshed") is not (index % 5 == 0)):
            raise ValueError("Recorded policy cadence/step/method does not match the frozen policy")
        if decision["action"][-1] != -1:
            raise ValueError("Continuation requires every historical executed gripper action to be open")
        if details["refreshed"]:
            prediction = details.get("prediction", {})
            if prediction.get("gripper_command") not in ("open", "hold"):
                raise ValueError("Continuation requires only open/hold historical gripper predictions")
            if prediction != details.get("field_prediction"):
                raise ValueError("Source field prediction was transformed or is missing")
            last_stage = prediction.get("stage")
        if details.get("stage") != last_stage or not isinstance(last_stage, str) or not last_stage:
            raise ValueError("Source stage history is incomplete")
    if pairs[-1][1]["sim_status"].get("terminated") or pairs[-1][1].get("success"):
        raise ValueError("Source physical episode already terminated")
    ledger = state.get("budget")
    if not isinstance(ledger, dict) or ledger.get("budget_bound_violated") is not False:
        raise ValueError("Source has a missing or invalid cumulative API ledger")
    after = episode.get("budget_after", {})
    if any(ledger.get(key) != after.get(key) for key in LEDGER_VALUES):
        raise ValueError("Terminal episode and run-state ledgers disagree")
    # Reconcile only the original configuration/log directory. New segment logs
    # start at K+1 and must never be passed to the generic 1..N reconciliation.
    auditor = AstraClient(old_api, api_key="offline-ledger-validation", log_dir=source / "api")
    auditor.load_budget_snapshot(ledger)
    if auditor.reconcile_request_logs() != 0:
        raise ValueError("Source terminal ledger has unreconciled API attempts")
    requests = sorted((source / "api").glob("*.request.json"),
                      key=lambda path: json.loads(path.read_text())["request_ordinal"])
    if not requests:
        raise ValueError("Source has no auditable prior API requests")
    results = []
    for request in requests:
        result_path = request.with_name(request.name.removesuffix(".request.json") + ".result.json")
        if not result_path.is_file():
            raise ValueError("Source terminal API attempt lacks a result ledger")
        results.append(json.loads(result_path.read_text()))
    if any(results[-1].get("budget", {}).get(key) != ledger.get(key) for key in LEDGER_VALUES):
        raise ValueError("Latest raw API result and terminal cumulative ledger disagree")
    reservation = auditor.budget_snapshot()["reservation_per_call_usd"]
    if ledger["cost_usd_upper_bound"] + reservation > max_total_cost_usd:
        raise ValueError("New total cap has insufficient headroom for one conservative request reservation")
    new_api = replace(old_api, max_cost_usd=max_total_cost_usd)
    handoff = copy.deepcopy(ledger)
    handoff["configuration"] = asdict(new_api)
    handoff["max_cost_usd"] = max_total_cost_usd
    # Validate the cap-adjusted snapshot with the public loader, without a call.
    AstraClient(new_api, api_key="offline-ledger-validation").load_budget_snapshot(handoff)
    return {"source": source, "identity": identity, "source_identity_hash": identity_hash,
            "sim_config": config, "simulator_identity": simulator_identity, "episode": episode,
            "offset": offset, "last_stage": last_stage, "reset": reset, "pairs": pairs,
            "source_snapshot": {**snapshot, **selection}, "original_budget": copy.deepcopy(ledger),
            "handoff_budget": handoff, "api_config": new_api,
            "parent_artifact_sha256": {name: recovery.sha256((source / name).read_bytes())
                for name in ("manifest.json", "simulator.json", "episodes.jsonl", "run_state.json")}}


def continuation_identity(prepared):
    return {"kind": "same_episode_budget_continuation", "task": base.TASK, "suite": "libero_spatial",
            "task_id": 0, "episode_index": 0, "seed": 7,
            "parent_run": str(prepared["source"]), "parent_identity_hash": prepared["source_identity_hash"],
            "parent_artifact_sha256": prepared["parent_artifact_sha256"],
            "source_trace": prepared["source_snapshot"], "continuation_offset": prepared["offset"],
            "original_budget_snapshot": prepared["original_budget"],
            "handoff_budget_snapshot": prepared["handoff_budget"],
            "cost_cap_change": {"old_total_usd": prepared["original_budget"]["configuration"]["max_cost_usd"],
                                "new_total_usd": prepared["api_config"].max_cost_usd,
                                "only_configuration_change": "max_cost_usd",
                                "authorization_requirement": "Explicit user approval is required before paid launch."},
            "api": asdict(prepared["api_config"]), "simulator": asdict(prepared["sim_config"]),
            "frozen_policy": prepared["identity"]["reference"]["policy"],
            "restored_policy_state": {"step_index": prepared["offset"], "last_stage": prepared["last_stage"],
                                      "adapter_gripper": -1.0, "next_action_requires_fresh_field": True},
            "software": prepared["identity"]["software"],
            "script_sha256": {name: recovery.sha256(Path(__file__).with_name(name).read_bytes()) for name in (
                "continue_bowl_episode.py", "replay_bowl_state.py", "evaluate_bowl_until_success.py")},
            "paid_inference_gate": "Exact full-prefix state, image, task, and simulator provenance agreement in the same still-open simulator.",
            "api_ledger_policy": "Carry all prior calls/tokens/cost/unknown usage; log only new requests starting at cumulative ordinal K+1; generic resume unsupported.",
            "stopping_rule": base.STOPPING_RULE,
            "scope": "Continuation of one physical episode. Reconstruction actions are historical and are never new evaluated actions or additional model calls."}


def reconstruct(sim, prepared, persist):
    """Return the verified current observation while keeping this simulator open."""
    proof = {"status": "validating", "requested_actions": prepared["offset"], "completed_actions": 0,
             "max_state_error": 0.0, "state_checks": 0, "image_pairs_checked": 0,
             "images_match": True, "state_atol": 0.0, "astra_requests_during_reconstruction": 0,
             "source_trace": prepared["source_snapshot"], "same_simulator_retained_for_continuation": True}

    def check_observation(obs, expected):
        error = np.abs(np.asarray(obs.state) - np.asarray(expected["observation/state"]))
        proof["state_checks"] += 1
        proof["max_state_error"] = max(proof["max_state_error"], float(error.max()))
        if error.shape != (8,) or not np.isfinite(error).all() or np.any(error != 0) or obs.prompt != base.TASK:
            raise ValueError("Reconstruction state or task differs from the recorded episode")
        comparisons = recovery.image_comparison(expected, obs.to_dict())
        proof["image_pairs_checked"] += 1
        if any(not item["available"] or item["match"] is not True for item in comparisons.values()):
            proof["images_match"] = False
            raise ValueError("Reconstruction image differs from the recorded episode")
        return comparisons

    try:
        if {"metadata": sim.metadata, "tasks": sim.list_tasks()} != prepared["simulator_identity"]:
            raise ValueError("Reconstruction simulator provenance differs from the original")
        obs = sim.reset(0, 0)
        proof["reset_images"] = check_observation(obs, prepared["reset"]["observation"])
        if sim.status != prepared["reset"]["sim_status"]:
            raise ValueError("Reconstruction initialization/status identity differs")
        for decision, executed in prepared["pairs"]:
            if list(obs.state) != decision["state"]:
                raise ValueError("Reconstruction pre-action state differs")
            obs, success = sim.step(decision["action"])
            proof["completed_actions"] += 1
            proof["final_images"] = check_observation(obs, executed["observation"])
            if sim.status != executed["sim_status"] or success != executed["success"]:
                raise ValueError("Reconstruction simulator outcome differs")
            persist(proof["completed_actions"])
        proof["status"] = "validated"
        return obs, proof
    except BaseException as exc:
        proof.update(status="validation_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(prepared["output"] / "replay_validation.json", proof)


class ContinuationProgress(base.ProgressMonitor):
    def emit(self, event):
        try:
            state = json.loads((self.output / "run_state.json").read_text())
        except (OSError, ValueError):
            state = {}
        print(json.dumps({"event": event, **base.read_progress(self.output),
                          **{key: state.get(key) for key in ("continuation_offset", "reconstruction_actions",
                              "new_evaluated_actions", "cumulative_actions")},
                          "elapsed_s": round(time.monotonic() - self.started, 1)}), flush=True)


def execute_continuation(prepared, output, *, api_key, simulator_factory=SimProcess,
                         client_factory=AstraClient, recorder_factory=EpisodeVideoRecorder):
    """Paid execution entry point; call only after explicit approval of the total cap."""
    output = Path(output).resolve()
    if output == prepared["source"] or output.is_relative_to(prepared["source"]):
        raise ValueError("Continuation output must be separate from the immutable parent run")
    if output.exists():
        raise FileExistsError("Continuation requires a fresh output directory; generic resume is unsupported")
    os.environ.pop("OPENAI_API_KEY", None)
    client = client_factory(prepared["api_config"], api_key=api_key,
                             transport=base.redacting_transport(api_key), log_dir=output / "api")
    client.load_budget_snapshot(prepared["handoff_budget"])
    client.preflight()
    policy = base.BowlFieldPolicy(client, method="field", grid=GridSpec(size=9, spacing_m=0.02),
                                  planner_config=PlannerConfig(step_m=0.01, horizon=3), update_steps=5, seed=7)
    policy.step_index, policy.last_stage = prepared["offset"], prepared["last_stage"]
    output.mkdir(parents=True, exist_ok=False)
    prepared = {**prepared, "output": output}
    identity = continuation_identity(prepared)
    write_json(output / "manifest.json", {"identity_hash": recovery.canonical_hash(identity),
        "configuration": identity, "created_utc": datetime.now(timezone.utc).isoformat()})
    row = {"status": "running", "suite": "libero_spatial", "method": "field", "task_id": 0,
           "episode_index": 0, "seed": 7, "task_prompt": base.TASK, "parent_run": str(prepared["source"]),
           "parent_identity_hash": prepared["source_identity_hash"], "continuation_offset": prepared["offset"],
           "reconstruction_actions": 0, "new_evaluated_actions": 0, "cumulative_actions": prepared["offset"],
           "new_path_length_m": 0.0, "log": "continuation.jsonl.gz", "video": None}

    def persist(stage, reconstructed=None):
        if reconstructed is not None:
            row["reconstruction_actions"] = reconstructed
        write_json(output / "run_state.json", {**row, "stage": stage, "step": row["cumulative_actions"],
                                               "budget": client.budget_snapshot()})

    recorder = None
    monitor = ContinuationProgress(output)
    persist("reconstructing")
    monitor.start()
    try:
        with simulator_factory(prepared["sim_config"]) as sim:
            obs, proof = reconstruct(sim, prepared, lambda count: persist("reconstructing", count))
            if proof["status"] != "validated" or proof["completed_actions"] != prepared["offset"]:
                raise ValueError("Paid inference gate requires complete validated reconstruction")
            write_json(output / "simulator.json", prepared["simulator_identity"])
            recorder = recorder_factory(output / "continuation.mp4", grid=policy.grid,
                                         episode_id=f"libero_spatial_t00_i00_field_continued_from_{prepared['offset']}")
            recorder.observe(obs, step=prepared["offset"])
            with gzip.open(output / row["log"], "wt") as log:
                log.write(json.dumps({"event": "continuation_start", "continuation_offset": prepared["offset"],
                    "observation": obs.to_dict(), "sim_status": dict(sim.status),
                    "parent_identity_hash": prepared["source_identity_hash"]}, allow_nan=False) + "\n")
                log.flush()
                while True:
                    if sim.status.get("success"):
                        row["status"] = "success"
                        break
                    if sim.status.get("terminated"):
                        raise SimulatorError("Verified simulator terminated without task success")
                    step = row["cumulative_actions"]
                    persist("deciding")
                    started = time.monotonic()
                    decision = policy.act(obs)
                    latency = time.monotonic() - started
                    entry = {"step": step, "new_action_index": row["new_evaluated_actions"],
                             "state": list(obs.state), "action": list(decision.action),
                             "diagnostics": decision.diagnostics, "decision_latency_s": latency}
                    if decision.diagnostics.get("refreshed"):
                        entry["observation"] = obs.to_dict()
                    log.write(json.dumps(entry, allow_nan=False) + "\n")
                    log.flush()
                    recorder.decision(obs, decision, decision_latency_s=latency)
                    persist("executing")
                    previous = np.asarray(obs.state[:3])
                    obs, success = sim.step(decision.action)
                    row["new_evaluated_actions"] += 1
                    row["cumulative_actions"] = prepared["offset"] + row["new_evaluated_actions"]
                    row["new_path_length_m"] += float(np.linalg.norm(np.asarray(obs.state[:3]) - previous))
                    log.write(json.dumps({"event": "executed", "step": step, "next_state": list(obs.state),
                        "success": success, "observation": obs.to_dict(), "sim_status": dict(sim.status),
                        "new_evaluated_actions": row["new_evaluated_actions"],
                        "cumulative_actions": row["cumulative_actions"]}, allow_nan=False) + "\n")
                    log.flush()
                    recorder.executed(obs, step=step, success=success, sim_status=dict(sim.status))
                    if success:
                        row["status"] = "success"
                        break
    except KeyboardInterrupt:
        row.update(status="user_stop", error="Continuation interrupted by user")
    except BudgetExceeded as exc:
        row.update(status="budget_exceeded", error=base.redact(str(exc), api_key))
    except Exception as exc:
        row.update(status="infrastructure_error", error=base.redact(f"{type(exc).__name__}: {exc}", api_key))
    finally:
        if recorder is not None:
            try:
                recorder.close(status=row["status"], error=row.get("error"))
                row["video"] = "continuation.mp4"
            except Exception as exc:
                row["video_error"] = base.redact(f"{type(exc).__name__}: {exc}", api_key)
        monitor.close()
        budget = client.budget_snapshot()
        row.update(budget_after=budget, evaluation_completed=row["status"] == "success",
                   libero_success=True if row["status"] == "success" else None,
                   additional_api_calls=budget["api_calls"] - prepared["original_budget"]["api_calls"],
                   additional_cost_usd_upper_bound=budget["cost_usd_upper_bound"] - prepared["original_budget"]["cost_usd_upper_bound"],
                   max_total_cost_usd=prepared["api_config"].max_cost_usd,
                   scope="One physical episode continued after exact historical reconstruction. Replay actions are excluded from new evaluated actions; incomplete outcomes are not scored as failures.")
        with (output / "episodes.jsonl").open("x") as results:
            results.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(output / "summary.json", row)
        persist("terminal")
        print(json.dumps({"event": "continuation_finished", **{key: row.get(key) for key in (
            "status", "continuation_offset", "reconstruction_actions", "new_evaluated_actions",
            "cumulative_actions", "additional_api_calls", "max_total_cost_usd", "error")},
            "output": str(output)}), flush=True)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-total-cost-usd", type=float, required=True)
    args = parser.parse_args(argv)
    prepared = prepare_source(args.source, args.max_total_cost_usd)
    key = os.environ.pop("OPENAI_API_KEY", None) or getpass.getpass("API key (hidden, memory only): ").strip()
    result = execute_continuation(prepared, args.output, api_key=key)
    return (0 if result["status"] == "success" else 130 if result["status"] == "user_stop"
            else 3 if result["status"] == "budget_exceeded" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
