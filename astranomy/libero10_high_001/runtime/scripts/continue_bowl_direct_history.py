"""Continue one budget-stopped direct-history episode after a total-budget raise.

The cap includes all parent API usage. Historical actions are replayed in the
same simulator that will continue, with exact state, pixel, and status checks
before restoring history and allowing a new API request. Replay is unscored and
never included in the continuation video. Fresh sibling output only; no resume.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

from gpt_field.astra import BudgetExceeded
from gpt_field.direct_history_resume import load_resume, restore_policy
from gpt_field.direct_history_video import DirectHistoryVideoRecorder
from gpt_field.experiment import digest, source_manifest, write_json
from gpt_field.libero import LiberoConfig, SimProcess
from gpt_field.observation import decode_image


def _base_module():
    path = Path(__file__).with_name("evaluate_bowl_direct_history.py")
    spec = importlib.util.spec_from_file_location("_direct_history_frozen_base", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _base_module()
KIND = "same_episode_direct_history_budget_continuation"
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_output(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or output.parent != source.parent:
        raise ValueError("Continuation output must be a separate sibling of the parent episode")
    if output.exists():
        raise FileExistsError("Continuation output already exists; never overwrite or resume")
    # Never fork the same parent ledger twice after another child may have spent.
    for path in source.parent.glob("*/manifest.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        config = record.get("configuration", {})
        if (config.get("kind") != KIND or
                Path(config.get("parent_run", "/")).resolve() != source):
            continue
        try:
            summary = json.loads(path.with_name("runner_summary.json").read_text())
        except (OSError, ValueError):
            summary = {}
        # A finalized failure before the replay gate cannot have incurred new use.
        unused = (summary.get("status") in ("infrastructure_error", "user_stop")
                  and summary.get("replay_verified") is False
                  and summary.get("new_steps") == 0 and summary.get("new_api_calls") == 0)
        if not unused:
            raise ValueError("A continuation already references this parent; its ledger may be newer")
    return output


def verify_parent_unchanged(resume):
    for name, expected in resume.source_hashes.items():
        if sha(resume.source / name) != expected:
            raise ValueError(f"Parent evidence changed during continuation preparation: {name}")
    config = resume.manifest["configuration"]
    for name, expected in config["software"]["source_sha256"].items():
        if sha(ROOT / "src/gpt_field" / name) != expected:
            raise ValueError(f"Frozen package source changed during replay: {name}")
    if sha(ROOT / config["runner"]["path"]) != config["runner"]["sha256"]:
        raise ValueError("Frozen fresh-run evaluator changed during replay")


def authorization_record(path, max_cost_usd, resume=None):
    if path is None:
        return None
    path = Path(path).resolve()
    record = json.loads(path.read_text())
    if record.get("max_cost_usd_cumulative") != max_cost_usd:
        raise ValueError("Authorization cumulative cap differs from the requested cap")
    if resume is not None:
        parent = Path(record.get("parent_run", ""))
        parent = parent if parent.is_absolute() else ROOT / parent
        if (parent.resolve() != resume.source or record.get("parent_steps") != resume.steps
                or record.get("parent_api_calls") != resume.api_calls
                or record.get("parent_unknown_usage_calls") != resume.budget["unknown_usage_calls"]
                or abs(record.get("parent_cost_usd_upper_bound", -1)
                       - resume.budget["cost_usd_upper_bound"]) > 1e-9):
            raise ValueError("Authorization does not identify this exact parent episode/ledger")
        for name, expected in record.get("parent_artifact_sha256", {}).items():
            if resume.source_hashes.get(name) != expected:
                raise ValueError(f"Authorization parent artifact differs: {name}")
    return {"path": str(path), "sha256": sha(path), "record": record,
            "effective_total_cap_usd": max_cost_usd}


def preflight(parent, output, *, max_cost_usd, authorization=None):
    """Validate frozen history/accounting offline; neither API nor sim is used."""
    resume = load_resume(parent, max_cost_usd=max_cost_usd)
    validate_output(resume.source, output)
    return {"status": "ready_for_exact_simulator_replay", "kind": KIND,
            "parent": str(resume.source), "output": str(Path(output).resolve()),
            "base_steps": resume.steps, "prior_api_calls": resume.api_calls,
            "next_request_ordinal": resume.api_calls + 1,
            "cost_usd_upper_bound": resume.budget["cost_usd_upper_bound"],
            "max_cost_usd": max_cost_usd, "unknown_usage_calls": resume.budget["unknown_usage_calls"],
            "history_digest": resume.history_digest, "next_request_key": resume.next_request_key,
            "verification": resume.verification,
            "authorization": authorization_record(authorization, max_cost_usd, resume),
            "new_api_calls": 0, "simulator_started": False, "files_written": False,
            "scope": "One existing episode; cap includes all parent spending. Replay gate remains required."}


def reconstruct(sim, resume, persist):
    """Rebuild and compare every recorded observation in this still-open sim."""
    proof = {"status": "validating", "requested_actions": resume.steps,
             "completed_actions": 0, "state_checks": 0, "image_pairs_checked": 0,
             "status_checks": 0, "max_state_error": 0.0, "max_pixel_error": 0,
             "state_atol": 0.0, "astra_requests_during_reconstruction": 0,
             "same_simulator_retained_for_continuation": True,
             "hidden_physics_state_saved": False,
             "scope": "Exact recorded robot observations, camera pixels, simulator statuses and provenance; "
                      "historical actions are not new evaluated actions."}

    def check(actual, index):
        expected = resume.observations[index]
        delta = np.abs(np.asarray(actual.state) - np.asarray(expected.state))
        proof["state_checks"] += 1
        if delta.shape != (8,) or not np.isfinite(delta).all():
            raise ValueError(f"Invalid reconstructed state at step {index}")
        proof["max_state_error"] = max(proof["max_state_error"], float(delta.max()))
        if actual.prompt != expected.prompt or actual.prompt != base.TASK or np.any(delta != 0):
            raise ValueError(f"Reconstruction robot state or task mismatch at step {index}")
        for name in ("image", "wrist_image"):
            first = np.asarray(decode_image(getattr(actual, name)))
            second = np.asarray(decode_image(getattr(expected, name)))
            if first.shape != second.shape:
                raise ValueError(f"Reconstruction camera shape mismatch at step {index}")
            difference = int(np.abs(first.astype(np.int16) - second.astype(np.int16)).max())
            proof["max_pixel_error"] = max(proof["max_pixel_error"], difference)
            if difference:
                raise ValueError(f"Reconstruction camera pixel mismatch at step {index}: {name}")
        proof["image_pairs_checked"] += 1
        proof["status_checks"] += 1
        if sim.status != resume.sim_statuses[index]:
            raise ValueError(f"Reconstruction simulator status mismatch at step {index}")

    try:
        expected_sim = json.loads((resume.source / "simulator.json").read_text())
        if {"metadata": sim.metadata, "tasks": sim.list_tasks()} != expected_sim:
            raise ValueError("Reconstruction simulator provenance differs from the parent")
        obs = sim.reset(0, 0)
        check(obs, 0)
        persist(proof)
        for index, action in enumerate(resume.actions):
            obs, success = sim.step(action)
            proof["completed_actions"] += 1
            check(obs, index + 1)
            if success != resume.sim_statuses[index + 1]["success"] or success:
                raise ValueError("Reconstruction success differs or parent episode already succeeded")
            if (index + 1) % 25 == 0:
                persist(proof)
        if obs != resume.final_observation or sim.status.get("terminated"):
            raise ValueError("Reconstruction final observation is not a resumable parent endpoint")
        proof["status"] = "validated"
        persist(proof)
        return obs, proof
    except BaseException as exc:
        proof.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        persist(proof)
        raise


def execute_episode(parent, output, *, max_cost_usd, api_key, authorization=None,
                    sim_factory=None, recorder_factory=None, transport=None):
    resume = load_resume(parent, max_cost_usd=max_cost_usd)
    output = validate_output(resume.source, output)
    approval = authorization_record(authorization, max_cost_usd, resume)
    sim_config = LiberoConfig(**resume.manifest["configuration"]["simulator"])
    base.validate_config(resume.config, sim_config)
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("A live continuation requires a configured API credential")
    os.environ.pop("OPENAI_API_KEY", None)
    sim_factory, recorder_factory = sim_factory or SimProcess, recorder_factory or DirectHistoryVideoRecorder
    output.mkdir(parents=True, exist_ok=False)
    original_budget = resume.summary["budget"]
    identity = {"kind": KIND, "mode": "live_astra_direct_history", "task": base.TASK,
        "suite": "libero_spatial", "task_id": 0, "episode_index": 0, "seed": 7,
        "parent_run": str(resume.source), "parent_identity_hash": resume.manifest["identity_hash"],
        "parent_artifact_sha256": resume.source_hashes, "continuation_offset": resume.steps,
        "original_budget_snapshot": original_budget, "handoff_budget_snapshot": resume.budget,
        "cost_cap_change": {"old_total_usd": original_budget["max_cost_usd"],
                            "new_total_usd": max_cost_usd, "only_configuration_change": "max_cost_usd"},
        "authorization": approval, "api": asdict(resume.config), "simulator": asdict(sim_config),
        "software": source_manifest(), "parent_software": resume.manifest["configuration"]["software"],
        "runner": {"path": "scripts/continue_bowl_direct_history.py", "sha256": sha(__file__)},
        "parent_runner": resume.manifest["configuration"]["runner"],
        "policy": resume.manifest["configuration"]["policy"],
        "history_digest_at_handoff": resume.history_digest,
        "next_request_key_at_handoff": resume.next_request_key,
        "offline_history_verification": resume.verification,
        "stopping_rule": base.STOPPING_RULE, "max_policy_steps": None,
        "paid_inference_gate": "Exact full-prefix robot states, camera pixels, statuses and simulator provenance.",
        "scope": "One continued episode; replay actions and already billed parent requests are counted once.",
        "video_scope": "New segment only; steps and action-chunk numbers are cumulative. Measured path begins at handoff."}
    write_json(output / "manifest.json", {"configuration": identity, "identity_hash": digest(identity),
        "created_utc": datetime.now(timezone.utc).isoformat(), "simulation_pauses_for_api": True})
    status, error, new_steps, pending_action = "running", None, 0, None
    new_close, new_open, new_neutral, new_path, new_chunks, feedback_count = 0, 0, 0, 0.0, 0, 0
    sim, recorder, policy, client = None, None, None, None
    replay_verified, success_seen = False, False
    started = time.monotonic()
    video_name = "episode.mp4"
    cleanup_errors = []

    def budget():
        return client.budget_snapshot() if client is not None else resume.budget

    def state(stage):
        write_json(output / "run_state.json", base.redact({"status": status, "stage": stage,
            "mode": "live_astra_direct_history", "kind": KIND, "base_steps": resume.steps,
            "new_steps": new_steps, "cumulative_steps": resume.steps + new_steps,
            "step": resume.steps + new_steps, "pending_action_index": pending_action,
            "replay_verified": replay_verified, "budget": budget(), "error": error}, api_key))

    def replay_progress(proof):
        write_json(output / "replay.json", base.redact(proof, api_key))
        state("reconstructing")
        print(json.dumps({"event": "reconstruction", "status": proof["status"],
                          "completed_actions": proof["completed_actions"],
                          "requested_actions": resume.steps, "new_api_calls": 0}), flush=True)

    state("starting")
    with base.termination_as_interrupt():
        try:
            sim = sim_factory(sim_config)
            sim.start()
            obs, proof = reconstruct(sim, resume, replay_progress)
            verify_parent_unchanged(resume)
            # History/ledger restore happens only after the exact simulator gate.
            # No policy.reset() may follow this restoration.
            policy = restore_policy(resume, api_key=api_key,
                                    transport=base.redacting_transport(api_key, transport),
                                    log_dir=output / "api")
            client = policy.client
            client.preflight()
            if policy.step_index != resume.steps or policy.chunk:
                raise ValueError("Restored policy is not at the exact five-action boundary")
            if client.request_key(client.build_payload(obs.to_dict(), policy.grid, mode="direct"),
                                  policy.grid, "direct") != resume.next_request_key:
                raise ValueError("Restored next request differs from the verified parent history")
            write_json(output / "history_restore.json", {
                "status": "verified", "history_sha256": resume.history_digest,
                "next_request_key": resume.next_request_key,
                "policy_step_index": policy.step_index, "history_stats": client.history_stats(),
                "budget": client.budget_snapshot(), "source_verification": resume.verification,
                "new_api_calls": 0, "policy_reset_after_restore": False})
            replay_verified = True
            write_json(output / "simulator.json", {"metadata": sim.metadata, "tasks": sim.list_tasks()})
            recorder = recorder_factory(output / video_name,
                episode_id=f"SAME EPISODE CONTINUATION from step {resume.steps} | path shown from handoff",
                offline_smoke=False)
            recorder.refresh_count = resume.steps // 5
            recorder.observe(obs, step=resume.steps)
            with (output / "trace.jsonl").open("w") as log:
                base._append(log, {"event": "resume", "step": resume.steps,
                    "observation": obs.to_dict(), "sim_status": dict(sim.status),
                    "parent_run": str(resume.source), "parent_identity_hash": resume.manifest["identity_hash"],
                    "replay": "replay.json", "history_digest": resume.history_digest,
                    "mode": "live_astra_direct_history"})
                while True:
                    global_step = resume.steps + new_steps
                    if sim.status.get("success"):
                        success_seen, status = True, "success"
                        break
                    if sim.status.get("terminated"):
                        raise RuntimeError("Simulator terminated without LIBERO success")
                    if obs.prompt != base.TASK:
                        raise ValueError("Policy observation task changed during continuation")
                    state("deciding")
                    decision_started = time.monotonic()
                    decision = policy.act(obs)
                    latency = time.monotonic() - decision_started
                    new_chunks += int(bool(decision.diagnostics.get("refreshed")))
                    base._append(log, {"event": "action", "step": global_step,
                        "state": list(obs.state), "action": list(decision.action),
                        "diagnostics": decision.diagnostics, "decision_latency_s": latency,
                        "mode": "live_astra_direct_history"})
                    recorder.decision(obs, decision, decision_latency_s=latency)
                    pending_action = global_step
                    state("executing")
                    before = obs
                    obs, success = sim.step(decision.action)
                    new_steps += 1
                    pending_action = None
                    success_seen = success_seen or success
                    new_close += int(decision.action[6] > 0)
                    new_open += int(decision.action[6] < 0)
                    new_neutral += int(decision.action[6] == 0)
                    new_path += float(np.linalg.norm(np.asarray(obs.state[:3]) - np.asarray(before.state[:3])))
                    base._append(log, {"event": "executed", "step": global_step,
                        "next_state": list(obs.state), "observation": obs.to_dict(),
                        "action": list(decision.action), "success": success,
                        "sim_status": dict(sim.status), "mode": "live_astra_direct_history"})
                    feedback = policy.observe_transition(before, decision, obs)
                    feedback_count += 1
                    base._append(log, {"event": "feedback", "step": global_step, "feedback": feedback})
                    recorder.executed(obs, step=global_step, success=success,
                                      sim_status=dict(sim.status), feedback=feedback)
                    state("observed")
                    print(json.dumps({"event": "executed", "base_steps": resume.steps,
                        "new_steps": new_steps, "cumulative_steps": resume.steps + new_steps,
                        "new_close_actions": new_close, "success_observed": success,
                        "api_attempts": budget()["api_calls"],
                        "cost_usd_upper_bound": budget()["cost_usd_upper_bound"]}), flush=True)
        except KeyboardInterrupt:
            status, error = "user_stop", "Continuation interrupted before successful completion"
        except BudgetExceeded as exc:
            status, error = "budget_exceeded", base.redact(str(exc), api_key)
        except Exception as exc:
            status, error = "infrastructure_error", base.redact(f"{type(exc).__name__}: {exc}", api_key)
        finally:
            if recorder is not None:
                try:
                    recorder.close(status=status, error=error)
                except Exception as exc:
                    cleanup_errors.append(base.redact(f"Video finalization {type(exc).__name__}: {exc}", api_key))
            if sim is not None:
                try:
                    sim.close()
                except Exception as exc:
                    cleanup_errors.append(base.redact(f"Simulator close {type(exc).__name__}: {exc}", api_key))
            if cleanup_errors:
                error = "; ".join(filter(None, [error, *cleanup_errors]))
                if status == "success":
                    status = "infrastructure_error"
            completed = status == "success"
            ledger = budget()
            summary = base.redact({"kind": KIND, "status": status, "mode": "live_astra_direct_history",
                "task": base.TASK, "suite": "libero_spatial", "task_id": 0, "episode_index": 0,
                "seed": 7, "requested_episodes": 1, "parent_run": str(resume.source),
                "base_steps": resume.steps, "new_steps": new_steps,
                "cumulative_steps": resume.steps + new_steps, "steps": resume.steps + new_steps,
                "replay_verified": replay_verified, "reconstruction_actions": resume.steps if replay_verified else None,
                "new_validated_action_chunks": new_chunks,
                "validated_action_chunks": resume.summary["validated_action_chunks"] + new_chunks,
                "new_measured_feedback_count": feedback_count,
                "measured_feedback_count": resume.steps + feedback_count,
                "base_gripper_close_actions": resume.summary["gripper_close_actions"],
                "new_gripper_close_actions": new_close,
                "gripper_close_actions": resume.summary["gripper_close_actions"] + new_close,
                "base_gripper_open_actions": resume.summary["gripper_open_actions"],
                "new_gripper_open_actions": new_open,
                "gripper_open_actions": resume.summary["gripper_open_actions"] + new_open,
                "base_gripper_neutral_actions": resume.summary["gripper_neutral_actions"],
                "new_gripper_neutral_actions": new_neutral,
                "gripper_neutral_actions": resume.summary["gripper_neutral_actions"] + new_neutral,
                "base_path_length_m": resume.summary["path_length_m"], "new_path_length_m": new_path,
                "path_length_m": resume.summary["path_length_m"] + new_path,
                "pending_action_index": pending_action, "libero_success_observed": success_seen,
                "libero_success": True if completed else None, "evaluation_completed": completed,
                "success_rate": 1.0 if completed else None, "scored_episodes": int(completed),
                "incomplete_episodes": int(not completed), "base_api_calls": resume.api_calls,
                "new_api_calls": ledger["api_calls"] - resume.api_calls, "api_calls": ledger["api_calls"],
                "budget": ledger, "max_cost_usd": max_cost_usd,
                "budget_scope": "Cumulative total for the same parent episode, including all prior API usage",
                "elapsed_this_process_s": time.monotonic() - started,
                "video": video_name if (output / video_name).is_file() else None,
                "trace": "trace.jsonl" if (output / "trace.jsonl").exists() else None,
                "error": error, "output": str(output),
                "interpretation": "Continuation of one unbounded episode. Replay is not a second evaluation; "
                                  "budget, interruption and infrastructure stops remain incomplete/unscored."}, api_key)
            write_json(output / "runner_summary.json", summary)
            write_json(output / "summary.json", summary)
            with (output / "episodes.jsonl").open("w") as results:
                base._append(results, summary)
            state("finished")
    print(json.dumps({"event": "finished", **summary}), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cost-usd", required=True, type=float,
                        help="Approved cumulative total including parent spending")
    parser.add_argument("--authorization", help="Existing user authorization JSON to embed and hash")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        print(json.dumps(preflight(args.parent, args.output, max_cost_usd=args.max_cost_usd,
                                   authorization=args.authorization), indent=2), flush=True)
        return 0
    validate_output(args.parent, args.output)
    key = os.environ.pop("OPENAI_API_KEY", None) or getpass.getpass("API key (hidden, memory only): ").strip()
    result = execute_episode(args.parent, args.output, max_cost_usd=args.max_cost_usd,
                             api_key=key, authorization=args.authorization)
    return (0 if result["status"] == "success" else 130 if result["status"] == "user_stop"
            else 3 if result["status"] == "budget_exceeded" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
