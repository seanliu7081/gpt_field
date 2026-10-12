#!/usr/bin/env python3
"""Offline distance-control replay using recorded RGB/state and recorded responses.

No authentication, model request, simulator worker, or simulator action is used.
Policy, client validation, distance solving and visual feedback are shared-code
reconstructions, not independent proofs of object pose or physical success.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import asdict
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt_field.astra import APIError, AstraClient, AstraConfig, ENDPOINT
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS
from gpt_field.distance_policy import DistanceHybridPolicy
from gpt_field.distance_subscription import (
    ChatGPTPlanDistanceClient, distance_field_version, distance_prompt_version, distance_schema_name,
    distance_instructions, distance_output_schema, validate_control_event,
)
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.experiment import digest
from gpt_field.hybrid_subscription import HYBRID_GRID
from gpt_field.observation import PolicyObservation
from gpt_field.planner import PlannerConfig

_spec = importlib.util.spec_from_file_location("_distance_legacy_audit", ROOT / "scripts/audit_hybrid_subscription.py")
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)
Audit, read, rows, sha, near = _legacy.Audit, _legacy.read, _legacy.rows, _legacy.sha, _legacy.near

# These measure this invocation's execution rather than controller semantics.
# Request payloads, feedback, solver/cache metadata and poses are never filtered.
REPLAY_EXCLUSIONS = frozenset({"api_latency_s", "plan_latency_s", "field_expansion_latency_s",
    "local_update_latency_s", "local_visual_latency_s", "api_request_log"})


def stable(value):
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items() if key not in REPLAY_EXCLUSIONS}
    if isinstance(value, (list, tuple)):
        return [stable(item) for item in value]
    return value


def _group(trace, kind, audit):
    selected = [row for row in trace if row.get("event") == kind]
    audit.check([row.get("step") for row in selected] == list(range(len(selected))),
                kind + " indexes are contiguous from zero")
    return {row["step"]: row for row in selected if type(row.get("step")) is int}


def _sources(config, audit, frozen):
    hashes = config.get("software", {}).get("source_sha256", {})
    required = {"distance_policy.py", "distance_planner.py", "distance_subscription.py",
                "local_visual_feedback.py", "hybrid_policy.py", "policy.py", "tracking.py"}
    if config.get("distance_task", "bowl_plate") != "bowl_plate":
        required.add("distance_tasks.py")
    audit.check(required <= set(hashes), "manifest includes every new distance/visual replay module")
    audit.check(bool(hashes), "recorded source manifest is nonempty")
    changed = []
    for filename, expected in hashes.items():
        current = ROOT / "src/gpt_field" / filename
        archived = frozen / filename if frozen is not None else current
        audit.check(archived.is_file() and sha(archived) == expected,
                    "recorded frozen source hash: " + filename)
        matches = current.is_file() and sha(current) == expected
        audit.check(matches, "current shared replay source matches recorded version: " + filename)
        if not matches:
            changed.append(filename)
    runner = config.get("runner", {})
    path = runner.get("path")
    source = (frozen / Path(path).name if frozen is not None else ROOT / path) if path else None
    audit.check(source is not None and source.is_file() and sha(source) == runner.get("sha256"),
                "recorded runner source hash")
    return changed


def _configuration(config, summary, audit):
    task = resolve_distance_task(config.get("distance_task", "bowl_plate"))
    policy = config.get("policy", {})
    required = {"class": "gpt_field.distance_policy.DistanceHybridPolicy", "distance_field_control": True,
        "field_representation": distance_field_version(task.key), "prompt_version": distance_prompt_version(task.key),
        "controller_mode": "hybrid", "execution_mode": "tracking", "feedback_history": "block_summary",
        "reasoning_effort": "medium", "field_improvements": True, "centered_placement_guidance": True,
        "grasp_settle_guidance": True, "field_reuse_limit": 120,
        "field_refresh_policy": "visual_progress_guarded_30_to_120", "local_path_strategy": "dijkstra_distance_field",
        "update_steps": 30, "field_update_steps": 30, "field_execution_steps": 10,
        "direct_action_steps": 5, "direct_action_chunks": 5, "recent_image_observations": 2,
        "goal_tolerance_m": .005, "goal_orientation_tolerance_rad": .05,
        "tracking_lookahead_m": .03, "tracking_max_command_m": .05,
        "adapter_max_rotation_rad": .15,
        "success_feedback_to_policy": False, "reward_feedback_to_policy": False,
        "object_state_feedback_to_policy": False}
    for key, expected in required.items():
        audit.check(policy.get(key) == expected and type(policy.get(key)) is type(expected),
                    "distance protocol setting: " + key)
        if key in summary:
            audit.check(summary[key] == expected and type(summary[key]) is type(expected),
                        "summary agrees with protocol: " + key)
    audit.check(policy.get("grid") == asdict(HYBRID_GRID), "33-cubed field at five-millimeter spacing")
    expected_planner = asdict(PlannerConfig(horizon=10))
    audit.check(policy.get("planner") == json.loads(json.dumps(expected_planner)), "fixed planner workspace and ten-segment budget")
    audit.check(config.get("kind") == "subscription_hybrid_full_task", "full-task experiment identity")
    audit.check(config.get("task") == summary.get("task") == task.prompt
                and config.get("task_id") == summary.get("task_id") == task.task_id
                and config.get("episode_index") == summary.get("episode_index") == 0,
                "requested task language, task index and initial-state zero")
    if task.key != "bowl_plate":
        audit.check(config.get("task_spec") == json.loads(json.dumps(asdict(task))),
                    "recorded task registry and recovery restrictions match the selected task")
        audit.check(summary.get("distance_task") == task.key, "summary preserves explicit task selection")
    simulator = config.get("simulator", {})
    audit.check(simulator.get("suite") == summary.get("suite") == task.suite and simulator.get("seed") == 7,
                "recorded task suite and seed-seven configuration")
    audit.check(simulator.get("stop_on_success") is True and config.get("stop_on_model_complete") is False,
                "only actual evaluator success completes the task")
    audit.check("LIBERO success" in config.get("stopping_rule", [])
                and not any("model" in reason.lower() for reason in config.get("stopping_rule", [])),
                "stopping rule excludes model-declared completion")
    for key in ("max_policy_steps", "max_api_calls", "max_wall_time_s"):
        audit.check(config.get(key) is None, "no experiment cap: " + key)
    audit.check(simulator.get("max_steps") is None, "no simulator episode action cap")
    api = config.get("api", {})
    audit.check(api.get("model") == "gpt-6-astra" and api.get("endpoint") == ENDPOINT,
                "Astra subscription transport endpoint")
    for key in ("max_api_calls", "max_wall_time_s", "max_cost_usd", "input_price_per_million",
                "output_price_per_million", "cached_input_price_per_million", "cache_write_price_per_million"):
        audit.check(api.get(key) is None, "no API-key price or episode cap: " + key)
    audit.check(config.get("api_key_fallback") is False and config.get("model_fallback") is False,
                "no authentication or model fallback")
    audit.check(config.get("billing") == summary.get("billing") == "chatgpt_subscription"
                and summary.get("api_key_calls") == 0 and summary.get("api_key_fallback") is False,
                "recorded subscription-only billing route")


class RecordedTransport:
    """Serve only already-recorded responses; never dispatch an HTTP request."""

    def __init__(self, request_paths, config, audit):
        self.paths, self.config, self.audit = request_paths, config, audit
        self.task = resolve_distance_task(config.get("distance_task", "bowl_plate"))
        self.calls, self.client = 0, None
        self.accepted_raw = []

    def __call__(self, endpoint, payload, headers, timeout):
        index = self.calls
        if index >= len(self.paths):
            self.audit.check(False, "replay must not need an unrecorded model response")
            raise APIError("No recorded response available")
        self.calls += 1
        path = self.paths[index]
        record = read(path)
        prefix = f"request {index + 1}: "
        self.audit.check(endpoint == ENDPOINT, prefix + "offline reconstruction uses the declared endpoint only")
        self.audit.check(record.get("payload") == payload, prefix + "exact full RGB/state/history/schema payload reconstruction")
        self.audit.check(record.get("request_ordinal") == index + 1 and record.get("attempt") == 1,
                         prefix + "one ordered recorded attempt")
        self.audit.check(record.get("mode") == "hybrid" and record.get("grid") == asdict(HYBRID_GRID)
                         and record.get("prompt_version") == distance_prompt_version(self.task.key),
                         prefix + "versioned distance protocol identity")
        self.audit.check(record.get("billing") == "chatgpt_subscription"
                         and record.get("configuration") == self.config.get("api"),
                         prefix + "subscription configuration matches manifest")
        self.audit.check(record.get("request_key") == self.client.last_request_key,
                         prefix + "request fingerprint reconstructed")
        self.audit.check(record.get("history_stats_before_request") == self.client.history_stats(),
                         prefix + "history counts and summaries reconstructed")
        self.audit.check(payload.get("model") == "gpt-6-astra" and payload.get("reasoning") == {"effort": "medium"}
                         and payload.get("stream") is True and payload.get("store") is False
                         and not UNSUPPORTED_FIELDS.intersection(payload), prefix + "exact model effort and subscription controls")
        self.audit.check(payload.get("instructions") == distance_instructions(2, task_key=self.task.key)
                         and payload.get("text", {}).get("format") == {"type": "json_schema",
                             "name": distance_schema_name(self.task.key), "strict": True,
                             "schema": distance_output_schema(task_key=self.task.key)},
                         prefix + "strict selected-task prompt and schema")
        stem = path.name.removesuffix(".request.json")
        result_path = path.with_name(stem + ".result.json")
        response_path = path.with_name(stem + ".response.json")
        result = read(result_path) if result_path.is_file() else {}
        self.audit.check(result.get("status") in ("valid", "error"), prefix + "request has a recorded terminal outcome")
        if not response_path.is_file():
            self.audit.check(result.get("status") == "error", prefix + "missing response was never accepted")
            raise APIError("Recorded request ended without a response")
        response = read(response_path)
        self.audit.check(response.get("billing") == "chatgpt_subscription", prefix + "response billing provenance")
        if result.get("status") == "valid":
            self.audit.check(response.get("status") == "completed" and not response.get("error")
                             and response.get("model") == "gpt-6-astra", prefix + "accepted response completed under requested model")
            effort = (response.get("reasoning") or {}).get("effort")
            self.audit.check(effort in (None, "medium"), prefix + "reported effort agrees when present")
            try:
                self.accepted_raw.append(AstraClient._extract(response))
            except Exception:
                self.audit.check(False, prefix + "accepted raw structured response is extractable")
        return copy.deepcopy(response)


def _usage(paths, summary, audit):
    totals, valid = Counter(), 0
    for path in paths:
        stem = path.name.removesuffix(".request.json")
        response_path, result_path = path.with_name(stem + ".response.json"), path.with_name(stem + ".result.json")
        if result_path.is_file():
            valid += int(read(result_path).get("status") == "valid")
        response = read(response_path) if response_path.is_file() else {}
        usage = response.get("usage") or {}
        if (isinstance(usage, dict) and type(usage.get("input_tokens")) is int
                and type(usage.get("output_tokens")) is int and usage["input_tokens"] >= 0 and usage["output_tokens"] >= 0):
            totals["input_tokens"] += usage["input_tokens"]
            totals["output_tokens"] += usage["output_tokens"]
            for section, name, total in (("input_tokens_details", "cached_tokens", "cached_input_tokens"),
                                         ("output_tokens_details", "reasoning_tokens", "reasoning_tokens")):
                value = (usage.get(section) or {}).get(name)
                if type(value) is int and value >= 0:
                    totals[total] += value
        else:
            totals["unknown_usage_calls"] += 1
    ledger = summary.get("budget", {})
    audit.check(summary.get("api_calls") == ledger.get("api_calls") == ledger.get("subscription_inference_calls") == len(paths),
                "all recorded subscription request counts reconcile")
    audit.check(ledger.get("api_key_calls") == 0 and ledger.get("billing") == "chatgpt_subscription"
                and ledger.get("subscription_cost_usd") is None, "subscription accounting invents no API-key charge")
    for name in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens", "unknown_usage_calls"):
        audit.check(ledger.get(name) == totals[name], "recorded usage reconciliation: " + name)
    return totals, valid


def _distance_video_metadata(run, actions, audit):
    """Check the new display metadata in addition to the legacy MP4 audit."""
    event_path = run / "episode.events.jsonl"
    if not event_path.is_file():
        return  # The full video audit already reports the missing artifact.
    display_by_step = {}
    current_display = None
    for step, row in actions.items():
        diag = row["diagnostics"]
        search = diag.get("plan", {}).get("search") or {}
        if diag["controller"] != "coarse_field":
            current_display = None
        elif diag.get("distance_field_display") is not None:
            current_display = (diag.get("field_center"), search.get("goal_xyz"), search.get("cost_unit", "m"))
        elif diag.get("refreshed") or diag.get("local_field_updated") or diag.get("plan_refreshed"):
            if not (current_display is not None and search.get("cache_hit") is True
                    and search.get("method") == "dijkstra_distance_field"
                    and diag.get("plan", {}).get("status") == "move"
                    and current_display[:2] == (diag.get("field_center"), search.get("goal_xyz"))):
                current_display = None
        display_by_step[step] = current_display
    for event in rows(event_path):
        step, kind = event.get("step"), event.get("event")
        if step not in actions or kind in ("observation", "terminal"):
            continue
        prefix, diag = f"distance video {kind} step {step}: ", actions[step]["diagnostics"]
        display = display_by_step[step]
        audit.check(event.get("field_display_source") == ("local_dijkstra_cost_to_go" if display else "unavailable")
                    and event.get("field_display_cost_unit") == (display[2] if display else None),
                    prefix + "current local map availability and cost units")
        for key in ("local_field_revision", "local_goal_updates", "field_reuse_extensions", "local_visual_feedback"):
            audit.check(near(event.get(key), diag.get(key)), prefix + "decision-time " + key)
        audit.check(event.get("local_metadata_scope") == "pre_action_decision",
                    prefix + "visual metadata explicitly predates the executed image")
        audit.check(event.get("task_success_certified") is False, prefix + "display never certifies task completion")
        if kind in ("field_refresh", "local_replan"):
            audit.check(near(event.get("distance_field_display"), diag.get("distance_field_display")),
                        prefix + "actual distance solve remains separate from raw guidance")


def audit_run(run: Path, output: Path, frozen: Path | None = None, *, allow_offline_smoke=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents:
        raise ValueError("Audit output must be a fresh separate directory outside the experiment")
    if output.exists():
        raise FileExistsError("Audit output already exists")
    summary, state = read(run / "summary.json"), read(run / "run_state.json")
    if state.get("stage") != "finished" or summary.get("status") == "running":
        raise ValueError("Wait for a terminal finalized experiment")
    config = read(run / "manifest.json")["configuration"]
    if config.get("policy", {}).get("distance_field_control") is not True:
        raise ValueError("This auditor only supports explicit v7 distance-field control")
    task = resolve_distance_task(config.get("distance_task", "bowl_plate"))
    output.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    audit.notes.append("This is deterministic shared-code reconstruction from recorded RGB/state and raw responses. "
                       "It makes no network, model, authentication or simulator calls; it does not independently "
                       "prove object poses, collision safety, grasp, support or physical task success.")
    simulator_record = read(run / "simulator.json") if (run / "simulator.json").is_file() else {}
    fixture = simulator_record.get("metadata", {}).get("unit_test_mock_simulator") is True
    if fixture:
        audit.notes.append("These artifacts explicitly declare a synthetic unit-test simulator. "
                           "A passing audit is fixture validation, not evidence of a real task result or its speed.")
    audit.check(read(run / "manifest.json").get("identity_hash") == digest(config), "manifest digest")
    _configuration(config, summary, audit)
    archive = Path(frozen).resolve() if frozen is not None else run / "frozen_source"
    audit.check(archive.is_dir(), "frozen source archive exists")
    changed = _sources(config, audit, archive if archive.is_dir() else None)
    trace = rows(run / "trace.jsonl")
    resets = [row for row in trace if row.get("event") == "reset"]
    audit.check(len(resets) == 1 and bool(trace) and trace[0].get("event") == "reset", "one reset precedes all recorded actions")
    actions, executed, feedback = (_group(trace, kind, audit) for kind in ("action", "executed", "feedback"))
    local_rows = [row for row in trace if row.get("event") == "local_visual_feedback"]
    locals_by_step = {row.get("step"): row for row in local_rows}
    audit.check(len(local_rows) == len(locals_by_step), "one local visual-feedback record per action")
    event_rows = [row for row in trace if row.get("event") == "distance_control_event"]
    events = {}
    for row in event_rows:
        try:
            value = validate_control_event(row.get("distance_control_event"))
            audit.check(row.get("step") == value["step"] and value["step"] not in events,
                        "one allowlisted distance boundary event at its end-exclusive step")
            events[value["step"]] = value
        except Exception:
            audit.check(False, "distance boundary event has the exact observable schema")
    audit.check(len(feedback) <= len(executed) <= len(actions) <= len(executed) + 1
                and len(executed) <= len(feedback) + 1, "at most one incomplete final transition")
    audit.check(summary.get("steps") == state.get("step") == len(executed), "acknowledged action count")
    audit.check(summary.get("measured_feedback_count") == len(feedback), "acknowledged feedback count")
    audit.check(state.get("status") == summary.get("status"), "state and summary terminal status agree")
    audit.check(summary.get("status") in ("success", "user_stop", "subscription_error", "infrastructure_error"),
                "terminal outcome belongs to the uncapped LIBERO-only protocol")
    execution_evidence = _legacy._execution_evidence(summary, state, actions, executed, audit)

    paths = sorted((run / "requests").glob("*.request.json"))
    recorded = RecordedTransport(paths, config, audit)
    client = ChatGPTPlanDistanceClient(AstraConfig(**config["api"]), task_key=task.key,
        token_provider=lambda: "offline-recorded-audit-no-credentials", transport=recorded)
    recorded.client = client
    policy = DistanceHybridPolicy(client)
    observation = PolicyObservation.from_dict(resets[0]["observation"]) if resets else None
    if observation is not None:
        audit.check(observation.prompt == task.prompt, "initial observation contains the exact selected task prompt")
    expected_events = ["reset"]
    counts, switch_steps, reasons = Counter(), [], Counter()
    route_rows = {row.get("step"): row for row in trace if row.get("event") == "routing_decision"}
    switch_rows = {row.get("step"): row for row in trace if row.get("event") == "controller_switch"}
    replay_failed = False
    for step, row in actions.items():
        prefix = f"action {step}: "
        if observation is None:
            audit.check(False, prefix + "requires the actual preceding observation")
            replay_failed = True
            break
        audit.check(row.get("state") == list(observation.state), prefix + "decision starts at preceding measured pose")
        audit.check(observation.prompt == task.prompt, prefix + "decision observes the exact selected task prompt")
        try:
            before_calls = recorded.calls
            decision = policy.act(observation)
        except Exception as exc:
            audit.check(False, prefix + "shared replay raised " + type(exc).__name__)
            replay_failed = True
            break
        diag, expected = row.get("diagnostics", {}), decision.diagnostics
        audit.check(near(row.get("action"), list(decision.action)), prefix + "exact action replay from recorded response and observation")
        audit.check(near(stable(diag), stable(expected)), prefix + "all non-timing decision diagnostics replay exactly")
        for key in ("plan", "effective_compact_field", "control_context", "local_visual_feedback",
                    "local_field_revision", "local_goal_updates", "local_recovery_active", "field_reuse_extensions",
                    "distance_field_display"):
            audit.check(near(stable(diag.get(key)), stable(expected.get(key))), prefix + "reconstructed " + key)
        for key in REPLAY_EXCLUSIONS - {"api_request_log"}:
            if key in diag:
                audit.check(type(diag[key]) in (int, float) and math.isfinite(diag[key]) and diag[key] >= 0,
                            prefix + "nonnegative finite timing: " + key)
        controller = expected["controller"]
        if expected["refreshed"]:
            counts["refresh"] += 1
            counts["field" if controller == "coarse_field" else "direct"] += 1
            expected_events.append("routing_decision")
            route = route_rows.get(step, {})
            audit.check(route.get("controller") == controller and route.get("routing") == expected["routing"]
                        and route.get("previous_controller") == expected["previous_controller"]
                        and route.get("source") == "model_prediction", prefix + "routing trace matches accepted model wrapper")
            audit.check(recorded.calls == before_calls + 1, prefix + "exactly one recorded model response per refresh")
            request = paths[recorded.calls - 1]
            audit.check(Path(diag.get("api_request_log") or "").name == request.name,
                        prefix + "request-log artifact links to the replayed response")
            audit.check(diag.get("compact_prediction") == client.last_compact_prediction,
                        prefix + "raw model wrapper and context preserved independently of local changes")
        else:
            audit.check(recorded.calls == before_calls, prefix + "local control and field reuse make no model request")
        if expected["mode_changed"]:
            switch_steps.append(step)
            expected_events.append("controller_switch")
            switch = switch_rows.get(step, {})
            audit.check(switch.get("from_controller") == expected["previous_controller"]
                        and switch.get("to_controller") == controller
                        and switch.get("routing") == expected["routing"], prefix + "controller switch trace")
        counts["plan"] += int(expected.get("plan_refreshed", False))
        expected_events.append("action")
        audit.check(diag.get("task_success_certified") is False, prefix + "local diagnostics never certify task success")
        if not task.allow_upward_reobservation:
            audit.check(diag.get("local_recovery_active") is False,
                        prefix + "selected task forbids automatic upward view recovery")
        if step not in executed:
            observation = None
            continue
        result = executed[step]
        expected_events.append("executed")
        counts[controller] += 1
        audit.check(result.get("action") == row.get("action") and result.get("controller") == controller,
                    prefix + "simulator acknowledged the exact proposed action")
        try:
            after = PolicyObservation.from_dict(result["observation"])
        except Exception:
            audit.check(False, prefix + "returned observation contains only the four allowed fields")
            replay_failed = True
            break
        audit.check(result.get("next_state") == list(after.state), prefix + "returned observation agrees with measured state")
        audit.check(after.prompt == task.prompt, prefix + "returned observation preserves selected task language")
        audit.check(result.get("success") == result.get("sim_status", {}).get("success"), prefix + "success remains evaluator metadata")
        if step in feedback:
            try:
                reconstructed = policy.observe_transition(observation, decision, after)
            except Exception as exc:
                audit.check(False, prefix + "measured-feedback replay raised " + type(exc).__name__)
                replay_failed = True
                break
            expected_events.append("feedback")
            audit.check(near(feedback[step].get("feedback"), reconstructed), prefix + "all measured execution feedback reconstructed")
            control_event = policy.pending_field_event
            audit.check(near(events.get(step + 1), control_event), prefix + "boundary cause and feedback replay after actual transition")
            if control_event is not None:
                expected_events.append("distance_control_event")
                reasons[control_event["reason"]] += 1
            expected_events.append("local_visual_feedback")
            local = locals_by_step.get(step, {})
            audit.check(near(local.get("feedback"), policy.local_feedback)
                        and local.get("local_field_revision") == policy.local_revision
                        and local.get("local_recovery_active") == policy._recovery_active,
                        prefix + "post-action visual calibration, local revision and recovery replay")
            if not task.allow_upward_reobservation:
                audit.check(local.get("local_recovery_active") is False,
                            prefix + "post-action state preserves task recovery restriction")
        observation = after
    audit.check([row.get("event") for row in trace] == expected_events, "exact trace routing/action/feedback/boundary/visual ordering")
    # A failed or interrupted next request can occur before another action is
    # proposed. Validate its payload and available response without inventing an
    # acknowledged action or treating an accepted response as task completion.
    if not replay_failed and recorded.calls < len(paths) and observation is not None:
        audit.check(len(paths) - recorded.calls == 1 and summary.get("status") in
                    ("user_stop", "subscription_error", "infrastructure_error"),
                    "only one terminal unused or failed request may follow the last action")
        try:
            policy.act(observation)
        except Exception:
            pass
        else:
            audit.notes.append("A recorded terminal response was accepted before an action was logged; no execution is inferred from it.")
    audit.check(recorded.calls == len(paths), "every recorded request payload reconstructed")
    usage, accepted_count = _usage(paths, summary, audit)
    for key, expected in (("validated_hybrid_refreshes", counts["refresh"]),
                          ("validated_field_refreshes", counts["field"]), ("validated_direct_chunks", counts["direct"]),
                          ("local_plan_count", counts["plan"]), ("coarse_field_actions", counts["coarse_field"]),
                          ("contact_direct_actions", counts["contact_direct"]), ("controller_switches", len(switch_steps))):
        audit.check(summary.get(key) == expected, "summary count: " + key)
    success_steps = [step for step, row in executed.items() if row.get("success") is True]
    initial_success = bool(resets and resets[0].get("sim_status", {}).get("success"))
    observed_success = bool(success_steps or initial_success)
    audit.check(summary.get("libero_success_observed") is observed_success, "summary evaluator-success evidence")
    audit.check(not success_steps or success_steps == [max(executed)], "no robot action after first actual evaluator success")
    completed = summary.get("status") == "success" and observed_success
    audit.check(summary.get("status") != "success" or observed_success, "success status requires actual evaluator evidence")
    audit.check(summary.get("model_declared_complete") is False and summary.get("model_completion") is None
                and summary.get("stop_on_model_complete") is False
                and not any(row.get("event") == "model_completion" for row in trace),
                "model or visual placement judgment never ends or scores the experiment")
    audit.check(summary.get("evaluation_completed") is completed and summary.get("benchmark_completed") is completed
                and summary.get("scored_episodes") == int(completed)
                and summary.get("incomplete_episodes") == int(not completed)
                and summary.get("libero_success") == (True if completed else None)
                and summary.get("success_rate") == (1.0 if completed else None), "completion and incomplete-score semantics")
    if not completed:
        audit.notes.append("This experiment is incomplete. Its elapsed wall time is not a successful completion time; no success speed ratio is inferred.")
    video, frames = _legacy._video_audit(run, summary, actions, executed, feedback, audit)
    _distance_video_metadata(run, actions, audit)
    report = {"kind": "distance_control_recorded_shared_code_audit", "passed": not audit.issues,
        "checks": audit.checks, "issues": audit.issues, "notes": audit.notes, "source_run": str(run),
        "status": summary.get("status"), "billing": summary.get("billing"), "requested_reasoning_effort": "medium",
        "distance_task": task.key, "task": task.prompt, "suite": task.suite, "task_id": task.task_id,
        "distance_field_control": True, "executed_actions": len(executed), "feedback_records": len(feedback),
        "trace_complete": len(actions) == len(executed) == len(feedback), "execution_evidence": execution_evidence,
        "replayed_recorded_requests": recorded.calls, "accepted_recorded_responses": accepted_count,
        "reconstructed_local_plans": counts["plan"], "controller_switch_steps": switch_steps,
        "distance_control_events": len(events), "distance_event_reasons": dict(reasons),
        "libero_success_observed": observed_success, "evaluation_completed": completed,
        "synthetic_test_fixture": fixture, "real_task_completion_verified": False,
        "recorded_control_elapsed_s": summary.get("control_elapsed_s"), "usage_totals": dict(usage),
        "video": video, "event_frames": frames, "frozen_sources": str(archive),
        "current_source_mismatches": changed, "shared_code_reconstruction": True,
        "excluded_replay_fields": sorted(REPLAY_EXCLUSIONS), "live_model_calls_by_audit": 0,
        "network_requests_by_audit": 0, "simulator_actions_by_audit": 0, "credentials_loaded_by_audit": False,
        "opaque_model_output_printed": False, "physical_object_state_independently_verified": False,
        "auditor_sha256": sha(Path(__file__)),
        "artifact_sha256": {name: sha(run / name) for name in
            ("manifest.json", "summary.json", "run_state.json", "trace.jsonl", "episode.mp4", "episode.events.jsonl")
            if (run / name).is_file()}}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "checks", "issues", "status", "executed_actions",
        "replayed_recorded_requests", "reconstructed_local_plans", "distance_event_reasons")}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen-sources", type=Path)
    args = parser.parse_args(argv)
    return 0 if audit_run(args.run, args.output, args.frozen_sources)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
