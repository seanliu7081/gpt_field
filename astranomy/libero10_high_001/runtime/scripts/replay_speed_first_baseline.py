#!/usr/bin/env python3
"""Replay the speed-first policy against saved successful DistanceHybrid actions.

This is an offline regression, not a fresh rollout or a timing benchmark. Saved
model replies are supplied by an in-memory transport. A saved next observation
is consumed only after the newly generated action matches its archived action.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt_field.observation import PolicyObservation
from gpt_field.speed_first_hybrid import ChatGPTPlanSpeedFirstClient, SpeedFirstHybridPolicy


class ReplayMismatch(RuntimeError):
    """Stop before consuming the outcome of a different action."""


def require(condition, message):
    if not condition:
        raise ReplayMismatch(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest():
    sources = {Path(__file__).resolve()}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("gpt_field") and path and Path(path).is_file():
            sources.add(Path(path).resolve())
    return {str(path): sha256(path) for path in sorted(sources)}


def output_envelope(prediction, ordinal):
    # Original task-conditioned responses, never a newly generated policy or demo.
    return {
        "id": f"offline_saved_response_{ordinal}", "status": "completed",
        "model": "gpt-6-astra", "reasoning": {"effort": "medium"},
        "output": [{"type": "message", "id": f"offline_message_{ordinal}",
                    "role": "assistant", "phase": "final_answer", "content": [
                        {"type": "output_text", "text": json.dumps(prediction)}]}],
        "usage": {"input_tokens": 0, "output_tokens": 0,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    }


class SavedReplyTransport:
    def __init__(self, replies):
        self.replies = replies
        self.calls = 0
        self.current_step = None
        self.current_observation = None
        self.verified_observation_payloads = 0

    def __call__(self, endpoint, payload, headers, timeout_s):
        require(self.calls < len(self.replies), "New policy requested an extra model reply")
        expected_step, reply = self.replies[self.calls]
        require(expected_step == self.current_step,
                f"Model refresh moved from archived step {expected_step} to {self.current_step}")
        require(payload.get("reasoning") == {"effort": "medium"}, "Effort changed")
        require(payload.get("model") == "gpt-6-astra", "Model changed")
        observations = []
        for message in payload["input"]:
            if message.get("role") != "user" or not isinstance(message.get("content"), list):
                continue
            for part in message["content"]:
                if part.get("type") != "input_text":
                    continue
                try:
                    record = json.loads(part["text"])
                except (TypeError, ValueError):
                    continue
                if isinstance(record, dict) and record.get("type") == "observation":
                    observations.append(record)
        require(bool(observations), "No public observation was included in the request")
        newest = observations[-1]
        allowed = {"type", "observation_index", "prompt", "observation/state",
                   "observation/joint_position", "observation/joint_velocity"}
        require(set(newest) <= allowed, "Observation payload included non-allowlisted fields")
        require(newest.get("observation/state") == list(self.current_observation.state),
                "Request does not contain the current saved robot state")
        require(newest.get("prompt") == self.current_observation.prompt,
                "Request task differs from the archived public observation")
        require("observation/joint_position" not in newest and "observation/joint_velocity" not in newest,
                "Historical replay must not invent arm joints absent from its saved interface")
        self.verified_observation_payloads += 1
        self.calls += 1
        return output_envelope(copy.deepcopy(reply), self.calls)


def replay(archive: Path, result: dict):
    trace_path = archive / "trace.jsonl"
    summary_path = archive / "summary.json"
    with trace_path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    archived_summary = json.loads(summary_path.read_text())
    require(archived_summary.get("libero_success") is True,
            "Source archive must be an actual LIBERO-success episode")
    actions = [row for row in rows if row.get("event") == "action"]
    executed = [row for row in rows if row.get("event") == "executed"]
    feedback = [row for row in rows if row.get("event") == "feedback"]
    resets = [row for row in rows if row.get("event") == "reset"]
    require(len(resets) == 1, "Archive must contain exactly one reset")
    require(len(actions) == len(executed) == len(feedback) == archived_summary["steps"],
            "Saved action/executed/feedback transitions are incomplete")
    for label, records in (("action", actions), ("executed", executed), ("feedback", feedback)):
        require([row.get("step") for row in records] == list(range(len(actions))),
                f"Saved {label} indexes are not contiguous")
    replies = [(row["step"], row["diagnostics"]["compact_prediction"])
               for row in actions if row["diagnostics"].get("refreshed")]
    require(all(reply["control_context"]["visual_reference"] is None for _, reply in replies),
            "This baseline replay requires original null visual references")
    require(all(reply["field"]["goal_reference"] is None for _, reply in replies
                if reply["controller"] == "coarse_field"),
            "This baseline replay requires original null held-object goal references")
    result.update(source_archive=str(archive), source_trace_sha256=sha256(trace_path),
                  source_summary_sha256=sha256(summary_path), archived_action_count=len(actions),
                  archived_refresh_steps=[step for step, _ in replies],
                  archived_libero_success=True, policy_received_simulator_status=False,
                  policy_received_evaluator_success=False, public_observation_interface="paired_RGB_plus_state8")

    transport = SavedReplyTransport(replies)
    client = ChatGPTPlanSpeedFirstClient(token_provider=lambda: "offline-replay-no-credentials",
                                        transport=transport)
    policy = SpeedFirstHybridPolicy(client)
    before = PolicyObservation.from_dict(resets[0]["observation"])
    initial_hash = hashlib.sha256(json.dumps(before.to_dict(), sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    result["initial_public_observation_sha256"] = initial_hash
    counts, refreshes = Counter(), []
    max_error = 0.0
    for index, (saved_action, saved_executed, saved_feedback) in enumerate(zip(actions, executed, feedback)):
        result["current_step"] = index
        transport.current_step, transport.current_observation = index, before
        decision = policy.act(before)
        error = float(np.max(np.abs(np.asarray(decision.action) - np.asarray(saved_action["action"]))))
        max_error = max(max_error, error)
        result["maximum_action_absolute_error"] = max_error
        result["all_actions_exact_equal"] &= list(decision.action) == saved_action["action"]
        # A numerical tolerance is permitted only for floating point evaluation;
        # the report separately identifies whether every action is bitwise equal.
        require(np.allclose(decision.action, saved_action["action"], rtol=0.0, atol=1e-12),
                f"Action differs at step {index}: max error {error}")
        require(decision.diagnostics["controller"] == saved_action["diagnostics"]["controller"],
                f"Controller differs at step {index}")
        require(decision.diagnostics["refreshed"] == saved_action["diagnostics"]["refreshed"],
                f"Refresh status differs at step {index}")
        if decision.diagnostics["refreshed"]:
            refreshes.append(index)
        if decision.diagnostics["controller"] == "coarse_field":
            require(decision.diagnostics["plan"]["search"]["method"] == "dijkstra_distance_field",
                    f"Field action {index} did not use the graph distance planner")
        # The saved outcome becomes admissible only after action/controller match.
        after = PolicyObservation.from_dict(saved_executed["observation"])
        actual_feedback = policy.observe_transition(before, decision, after)
        require(actual_feedback == saved_feedback["feedback"],
                f"Measured robot feedback differs at step {index}")
        before = after
        counts[decision.diagnostics["controller"]] += 1
        result["matched_complete_transitions"] = index + 1
    require(transport.calls == len(replies), "Not every archived model reply was consumed")
    require(refreshes == [step for step, _ in replies], "Model refresh boundaries changed")
    require(counts["coarse_field"] == archived_summary["coarse_field_actions"] and
            counts["contact_direct"] == archived_summary["contact_direct_actions"],
            "Controller action counts changed")
    result.update(status="passed", model_refresh_count=transport.calls,
                  verified_public_observation_payloads=transport.verified_observation_payloads,
                  refresh_steps=refreshes, controller_action_counts=dict(counts),
                  remaining_direct_actions=len(policy.chunk),
                  terminal_note="The archive ends at LIBERO success; its final five-action Direct reply executed only three actions. Replay does not execute its remaining two.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "runs/distance_field_medium_task_001")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, archive = args.output.resolve(), args.archive.resolve()
    if output == archive or archive in output.parents:
        parser.error("--output must be outside the immutable source archive")
    if output.exists():
        parser.error("--output already exists; preserve prior replay results")
    result = {"kind": "offline_saved_observation_action_regression", "status": "failed",
              "live_inference_calls": 0, "simulator_starts": 0, "simulator_actions": 0,
              "credential_access": False, "matched_complete_transitions": 0,
              "all_actions_exact_equal": True, "new_task_success": None,
              "task_speedup_established": False,
              "interpretation": "Saved model outputs are injected into the new policy. Matching commands on saved observations establish execution compatibility only; changed prompts, fresh model behavior, task success and speed require a fresh authorized evaluation."}
    sources = source_manifest()
    started = time.monotonic()
    saved_create_connection = socket.create_connection

    def deny_network(*_args, **_kwargs):
        raise ReplayMismatch("Network access is prohibited in offline replay")

    socket.create_connection = deny_network
    try:
        replay(archive, result)
        require(sources == source_manifest(), "Loaded runtime sources changed during replay")
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        socket.create_connection = saved_create_connection
        result["runtime_sources_sha256"] = sources
        result["offline_replay_wall_s_not_task_latency"] = time.monotonic() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result.get(key) for key in (
        "status", "matched_complete_transitions", "model_refresh_count", "maximum_action_absolute_error",
        "all_actions_exact_equal", "error")}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
