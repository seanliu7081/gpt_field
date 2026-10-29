#!/usr/bin/env python3
"""Offline reconstruction of zero-shot collision-free, five-action Direct comparison arms.

Recorded responses are replayed through shared client/policy code. No model,
authentication, simulator or physical object-state verification is performed.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt_field.astra import APIError, AstraClient, AstraConfig, ENDPOINT
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS
from gpt_field.zero_shot_direct_subscription import (
    ChatGPTPlanZeroShotDirectClient, ZERO_SHOT_DIRECT_PROMPT_VERSION,
    ZERO_SHOT_DIRECT_REPRESENTATION, ZERO_SHOT_DIRECT_SCHEMA_NAME,
    zero_shot_direct_instructions, zero_shot_direct_output_schema,
)
from gpt_field.distance_subscription import shared_zero_shot_task_guidance
from gpt_field.experiment import digest
from gpt_field.hybrid_policy import HybridFieldDirectPolicy
from gpt_field.hybrid_subscription import HYBRID_GRID
from gpt_field.observation import PolicyObservation

_spec = importlib.util.spec_from_file_location("_zero_shot_direct_distance_audit", ROOT / "scripts/audit_distance_control.py")
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_legacy = _helpers._legacy
Audit, read, rows, sha, near = _helpers.Audit, _helpers.read, _helpers.rows, _helpers.sha, _helpers.near
stable = _helpers.stable


def _configuration(config, summary, audit):
    task = _helpers._task_identity(config, summary, audit)
    required = {
        "class": "gpt_field.hybrid_policy.HybridFieldDirectPolicy", "method": "zero_shot_direct",
        "zero_shot_direct_control": True, "zero_shot_collision_free": True, "distance_field_control": False, "field_improvements": False,
        "field_representation": ZERO_SHOT_DIRECT_REPRESENTATION, "prompt_version": ZERO_SHOT_DIRECT_PROMPT_VERSION,
        "controller_mode": "direct", "execution_mode": "fixed", "feedback_history": "block_summary",
        "reasoning_effort": "medium", "centered_placement_guidance": True, "grasp_settle_guidance": True,
        "update_steps": 5, "direct_action_steps": 5, "direct_action_chunks": 5, "recent_image_observations": 2,
        "controllers": ["contact_direct"], "field_generation": False, "local_field_planning": False,
        "local_plans_per_direct_action": 0, "field_refresh_policy": None, "local_path_strategy": None,
        "tracking_lookahead_m": None, "tracking_max_command_m": None, "field_reuse_limit": None,
        "goal_tolerance_m": None, "goal_orientation_tolerance_rad": None,
        "success_feedback_to_policy": False, "reward_feedback_to_policy": False,
        "object_state_feedback_to_policy": False,
    }
    for name, expected in required.items():
        actual = config.get("policy", {}).get(name)
        audit.check(actual == expected and type(actual) is type(expected), "Direct protocol setting: " + name)
        if name in summary:
            audit.check(summary[name] == expected and type(summary[name]) is type(expected), "summary protocol: " + name)
    audit.check(config.get("distance_task", task.key) == summary.get("distance_task", task.key) == task.key
                and config.get("task_spec") == json.loads(json.dumps(asdict(task))), "explicit selected task registry")
    audit.check(config.get("task") == summary.get("task") == task.prompt
                and config.get("task_id") == summary.get("task_id") == task.task_id,
                "selected task language and task index")
    _helpers._episode_identity(config, summary, audit)
    simulator = config.get("simulator", {})
    audit.check(simulator.get("suite") == summary.get("suite") == task.suite
                and simulator.get("seed") == summary.get("seed") == 7, "task suite and seed seven")
    audit.check(config.get("kind") == "subscription_hybrid_full_task"
                and config.get("mode") == summary.get("mode") == "live_chatgpt_subscription_zero_shot_direct",
                "explicit pure Direct experiment identity")
    audit.check(simulator.get("stop_on_success") is True and config.get("stop_on_model_complete") is False
                and "LIBERO success with no forbidden collision" in config.get("stopping_rule", [])
                and not any("model" in reason.lower() for reason in config.get("stopping_rule", [])),
                "only evaluator success completes the uncapped episode")
    for name in ("max_policy_steps", "max_api_calls", "max_wall_time_s"):
        audit.check(config.get(name) is None and summary.get(name) is None, "no episode cap: " + name)
    audit.check(simulator.get("max_steps") is None, "no simulator action cap")
    api = config.get("api", {})
    audit.check(api.get("model") == "gpt-6-astra" and api.get("endpoint") == ENDPOINT,
                "Astra subscription endpoint and model")
    for name in ("max_api_calls", "max_wall_time_s", "max_cost_usd", "input_price_per_million",
                 "output_price_per_million", "cached_input_price_per_million", "cache_write_price_per_million"):
        audit.check(api.get(name) is None, "no API-key price or request cap: " + name)
    audit.check(config.get("billing") == summary.get("billing") == "chatgpt_subscription"
                and summary.get("api_key_calls") == 0 and summary.get("api_key_fallback") is False
                and config.get("api_key_fallback") is False and config.get("model_fallback") is False,
                "subscription-only route without model or credential fallback")
    _helpers._zero_shot_configuration(config, summary, audit)


class RecordedTransport:
    """Return recorded envelopes only; exact input matching includes all images."""

    def __init__(self, paths, config, audit):
        self.paths, self.config, self.audit = paths, config, audit
        self.task = _helpers._task(config)
        self.calls, self.client = 0, None

    def __call__(self, endpoint, payload, headers, timeout):
        index = self.calls
        if index >= len(self.paths):
            self.audit.check(False, "reconstruction cannot invent an unrecorded model response")
            raise APIError("No recorded response")
        self.calls += 1
        path, prefix = self.paths[index], f"request {index + 1}: "
        record = read(path)
        self.audit.check(endpoint == ENDPOINT, prefix + "declared subscription endpoint")
        self.audit.check(record.get("payload") == payload, prefix + "exact full RGB/state/history/schema payload reconstruction")
        self.audit.check(record.get("request_ordinal") == index + 1 and record.get("attempt") == 1,
                         prefix + "one ordered recorded attempt")
        self.audit.check(record.get("mode") == "hybrid" and record.get("grid") == asdict(HYBRID_GRID)
                         and record.get("prompt_version") == ZERO_SHOT_DIRECT_PROMPT_VERSION, prefix + "versioned Direct client identity")
        self.audit.check(record.get("billing") == "chatgpt_subscription"
                         and record.get("configuration") == self.config.get("api"), prefix + "recorded subscription configuration")
        self.audit.check(record.get("request_key") == self.client.last_request_key, prefix + "request fingerprint")
        self.audit.check(record.get("history_stats_before_request") == self.client.history_stats(), prefix + "live history statistics")
        self.audit.check(payload.get("model") == "gpt-6-astra" and payload.get("reasoning") == {"effort": "medium"}
                         and payload.get("stream") is True and payload.get("store") is False
                         and not UNSUPPORTED_FIELDS.intersection(payload), prefix + "model effort and subscription controls")
        self.audit.check(payload.get("instructions") == zero_shot_direct_instructions(task_key=self.task.key)
                         and payload.get("text", {}).get("format") == {"type": "json_schema", "name": ZERO_SHOT_DIRECT_SCHEMA_NAME,
                             "strict": True, "schema": zero_shot_direct_output_schema(task_key=self.task.key)}, prefix + "strict five-action prompt and schema")
        _helpers._zero_shot_request(payload, self.audit, prefix)
        stats = self.client.history_stats()
        self.audit.check(stats.get("observations") == index and stats.get("retained_image_pairs") == min(index, 2),
                         prefix + "only live observation pairs contribute to retained images")
        stem = path.name.removesuffix(".request.json")
        result_path, response_path = path.with_name(stem + ".result.json"), path.with_name(stem + ".response.json")
        result = read(result_path) if result_path.is_file() else {}
        self.audit.check(result.get("status") in ("valid", "error"), prefix + "recorded request outcome")
        if not response_path.is_file():
            self.audit.check(result.get("status") == "error", prefix + "missing response was never accepted")
            raise APIError("Recorded request ended without a response")
        response = read(response_path)
        self.audit.check(response.get("billing") == "chatgpt_subscription", prefix + "response billing provenance")
        if result.get("status") == "valid":
            self.audit.check(response.get("status") == "completed" and not response.get("error")
                             and response.get("model") == "gpt-6-astra", prefix + "accepted response model and completion")
            self.audit.check((response.get("reasoning") or {}).get("effort") in (None, "medium"), prefix + "reported effort when present")
            try:
                raw = AstraClient._extract(response)
                self.audit.check(set(raw) == {"actions", "target_region", "routing_reason", "control_context"}
                                 and len(raw["actions"]) == 5, prefix + "raw model supplies only five Direct actions and observable context")
            except Exception:
                self.audit.check(False, prefix + "accepted structured raw response extractable")
        return copy.deepcopy(response)


def _reference_comparison(reference_run, run, config, summary, resets, paths, audit):
    if reference_run is None:
        audit.notes.append("No reference run was supplied; matched-condition comparison was not checked.")
        return None
    reference = Path(reference_run).resolve()
    before = len(audit.issues)
    manifest = read(reference / "manifest.json")
    other, result = manifest["configuration"], read(reference / "summary.json")
    task = _helpers._task(config)
    audit.check(_helpers._task(other).key == task.key, "reference selects the same live task")
    audit.check(manifest.get("identity_hash") == digest(other), "reference manifest digest")
    audit.check(other.get("policy", {}).get("distance_field_control") is True
                and other.get("policy", {}).get("controller_mode") == "hybrid", "comparison reference is the Hybrid arm")
    reference_trace = rows(reference / "trace.jsonl")
    reference_resets = [row for row in reference_trace if row.get("event") == "reset"]
    audit.check(len(reference_resets) == len(resets) == 1
                and reference_resets[0].get("observation") == resets[0].get("observation"),
                "matched actual initial paired RGB, measured state and task prompt")
    if len(reference_resets) == len(resets) == 1:
        for name in ("task_id", "episode_index", "initial_state_sha256", "bddl_sha256", "initial_state_source", "initial_state_override_sha256",
                     "original_bddl_sha256", "task_bddl_override_sha256", "custom_task_contract"):
            left, right = resets[0].get("sim_status", {}), reference_resets[0].get("sim_status", {})
            if name in left or name in right:
                audit.check(left.get(name) == right.get(name), "matched recorded simulator reset: " + name)
    for name in ("task", "task_id", "episode_index", "distance_task", "task_spec", "api", "billing",
                 "demonstration", "benchmark_variant", "initial_state_override", "task_bddl_override", "stop_on_model_complete", "stopping_rule", "max_policy_steps", "max_api_calls", "max_wall_time_s"):
        audit.check(config.get(name) == other.get(name), "matched reference condition: " + name)
    left_sim, right_sim = copy.deepcopy(config.get("simulator", {})), copy.deepcopy(other.get("simulator", {}))
    for simulator in (left_sim, right_sim):
        for name in ("initial_state_override", "task_bddl_override"):
            if simulator.get(name) is not None:
                simulator[name] = "<run-local-frozen-artifact-checked-by-hash>"
            else:
                simulator.pop(name, None)
    audit.check(left_sim == right_sim, "matched simulator settings, allowing distinct frozen-artifact paths")
    if config.get("initial_state_override") is not None or other.get("initial_state_override") is not None:
        left, right = run / "initial_state_override.npy", reference / "initial_state_override.npy"
        audit.check(left.is_file() and right.is_file() and sha(left) == sha(right), "matched exact custom initial-state bytes")
    if config.get("task_bddl_override") is not None or other.get("task_bddl_override") is not None:
        for name in ("task_override.bddl", "original_task.bddl"):
            left, right = run / name, reference / name
            audit.check(left.is_file() and right.is_file() and sha(left) == sha(right), "matched exact scene artifact: " + name)
    for name in ("reasoning_effort", "feedback_history", "centered_placement_guidance", "grasp_settle_guidance",
                 "direct_action_steps", "recent_image_observations", "zero_shot_collision_free"):
        audit.check(config["policy"].get(name) == other["policy"].get(name), "matched policy condition: " + name)
    audit.check("demonstration" not in config and "demonstration" not in other
                and not (run / "demonstration.json").exists() and not (reference / "demonstration.json").exists(),
                "both comparison arms have zero demonstrations")
    audit.check(other.get("policy", {}).get("zero_shot_collision_free") is True,
                "reference uses the same collision-free zero-shot criterion")
    source_hashes = other.get("software", {}).get("source_sha256", {})
    audit.check(bool(source_hashes), "reference has recorded frozen source hashes")
    for filename, expected in source_hashes.items():
        archived = reference / "frozen_source" / filename
        audit.check(archived.is_file() and sha(archived) == expected, "reference historical frozen source: " + filename)
    audit.check(config.get("software") == other.get("software"), "matched frozen protocol sources in this zero-shot pair")
    reference_actions = {row["step"]: row for row in reference_trace if row.get("event") == "action"}
    reference_executed = {row["step"]: row for row in reference_trace if row.get("event") == "executed"}
    reference_outcome = _helpers._collision_outcome(reference, other, result, read(reference / "run_state.json"),
                                                   reference_resets, reference_actions, reference_executed, audit)
    reference_paths = sorted((reference / "requests").glob("*.request.json"))
    audit.check(bool(paths) and bool(reference_paths), "both arms contain actual request evidence")
    for label, request_paths in (("Direct", paths), ("Hybrid", reference_paths)):
        for index, path in enumerate(request_paths):
            payload = read(path).get("payload", {})
            instructions = payload.get("instructions", "")
            _helpers._zero_shot_request(payload, audit, f"{label} request {index + 1}: ")
            for name, text in shared_zero_shot_task_guidance(task_key=task.key).items():
                audit.check(text in instructions, f"{label} request {index + 1} shares verbatim task guidance: {name}")
    audit.notes.append("Reference comparison checks recorded conditions and common task guidance. The controller-specific "
                       "prompt/schema differ intentionally. These single episodes are not repeated statistical trials, "
                       "and historical frozen sources are checked against their own manifest, not current source hashes.")
    both_completed = summary.get("evaluation_completed") is True and result.get("evaluation_completed") is True
    timing_eligible = (summary.get("status") == result.get("status") == "success"
                       and summary.get("task_success") is True and reference_outcome["completion_time_eligible"])
    return {"reference_run": str(reference), "matched_checks_passed": len(audit.issues) == before,
            "reference_status": result.get("status"), "reference_control_elapsed_s": result.get("control_elapsed_s"),
            "reference_control_excluding_video_s": _helpers._adjusted_control_s(result),
            "both_recorded_episodes_completed": both_completed,
            "both_completion_times_eligible": timing_eligible,
            "source_protocols_identical": config.get("software") == other.get("software"),
            "reference_artifact_sha256": {name: sha(reference / name) for name in
                ("manifest.json", "summary.json", "trace.jsonl", "demonstration.json", "initial_state_override.npy",
                 "task_override.bddl", "original_task.bddl", "collision.json", "task_evaluation.json") if (reference / name).is_file()}}


def audit_run(run: Path, output: Path, frozen: Path | None = None, *, reference_run=None, allow_offline_smoke=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or run in output.parents:
        raise ValueError("Audit output must be a fresh separate directory outside the experiment")
    if output.exists():
        raise FileExistsError("Audit output already exists")
    summary, state, manifest = read(run / "summary.json"), read(run / "run_state.json"), read(run / "manifest.json")
    if state.get("stage") != "finished" or summary.get("status") == "running":
        raise ValueError("Wait for a finalized terminal experiment")
    config = manifest["configuration"]
    if config.get("policy", {}).get("zero_shot_direct_control") is not True:
        raise ValueError("This auditor supports only explicit zero-shot collision-free pure Direct control")
    output.mkdir(parents=True, exist_ok=False)
    audit = Audit()
    audit.notes.append("Deterministic shared-code reconstruction uses recorded RGB/state and raw model responses only. "
                       "It makes no model, network, credential or simulator calls and is not independent physical-state verification.")
    audit.check(manifest.get("identity_hash") == digest(config), "manifest digest")
    _configuration(config, summary, audit)
    _helpers._demonstration(run, config, summary, audit)
    archive = Path(frozen).resolve() if frozen is not None else run / "frozen_source"
    audit.check(archive.is_dir(), "frozen source archive exists")
    audit.check("zero_shot_direct_subscription.py" in config.get("software", {}).get("source_sha256", {}), "new Direct client included in frozen manifest")
    changed = _helpers._sources(config, audit, archive if archive.is_dir() else None)
    simulator_record = read(run / "simulator.json") if (run / "simulator.json").is_file() else {}
    fixture = simulator_record.get("metadata", {}).get("unit_test_mock_simulator") is True
    if fixture:
        audit.notes.append("These artifacts explicitly declare a synthetic simulator fixture; passing is not a real task or timing result.")
    trace = rows(run / "trace.jsonl")
    resets = [row for row in trace if row.get("event") == "reset"]
    audit.check(len(resets) == 1 and bool(trace) and trace[0].get("event") == "reset", "one initial reset precedes actions")
    actions, executed, feedback = (_helpers._group(trace, kind, audit) for kind in ("action", "executed", "feedback"))
    _helpers._reset_identity(config, state, simulator_record, resets, executed, audit)
    override = _helpers._initial_state_override(run, config, summary, simulator_record, resets, executed, audit)
    scene = _helpers._task_bddl_override(run, config, summary, simulator_record, resets, executed, audit)
    audit.check(len(feedback) <= len(executed) <= len(actions) <= len(executed) + 1
                and len(executed) <= len(feedback) + 1, "at most one incomplete final transition")
    audit.check(summary.get("steps") == state.get("step") == len(executed), "acknowledged action count")
    audit.check(summary.get("measured_feedback_count") == len(feedback), "acknowledged feedback count")
    audit.check(state.get("status") == summary.get("status") and summary.get("status") in
                ("success", "collision_failure", "user_stop", "subscription_error", "infrastructure_error"), "terminal uncapped collision-free protocol status")
    evidence = _legacy._execution_evidence(summary, state, actions, executed, audit)
    paths = sorted((run / "requests").glob("*.request.json"))
    recorded = RecordedTransport(paths, config, audit)
    client = ChatGPTPlanZeroShotDirectClient(AstraConfig(**config["api"]), task_key=_helpers._task(config).key,
        token_provider=lambda: "offline-recorded-audit-no-credentials", transport=recorded)
    recorded.client = client
    policy = HybridFieldDirectPolicy(client)
    observation = PolicyObservation.from_dict(resets[0]["observation"]) if resets else None
    task = _helpers._task(config)
    expected_events, refreshes, replay_failed = ["reset"], 0, policy is None
    routes = {row.get("step"): row for row in trace if row.get("event") == "routing_decision"}
    for step, row in actions.items():
        if replay_failed:
            break
        prefix = f"action {step}: "
        audit.check(observation is not None and row.get("state") == list(observation.state)
                    and observation.prompt == task.prompt, prefix + "preceding allowed measured observation")
        try:
            before_calls = recorded.calls
            decision = policy.act(observation)
        except Exception as exc:
            audit.check(False, prefix + "shared replay raised " + type(exc).__name__)
            replay_failed = True
            break
        diag, expected = row.get("diagnostics", {}), decision.diagnostics
        audit.check(near(row.get("action"), list(decision.action)), prefix + "exact unchanged model action replay")
        audit.check(near(stable(diag), stable(expected)), prefix + "all non-timing diagnostics reconstructed")
        audit.check(expected["controller"] == "contact_direct" and expected["refreshed"] is (step % 5 == 0)
                    and expected["chunk_step"] == step % 5, prefix + "fixed five-action cadence")
        audit.check(not any(name in diag for name in ("plan", "tracking", "local_visual_feedback", "distance_field_display",
                    "effective_compact_field", "goal_grounding")) and not diag.get("field_refreshed"),
                    prefix + "no field, motion planning or local visual correction")
        if expected["refreshed"]:
            refreshes += 1
            expected_events.append("routing_decision")
            route = routes.get(step, {})
            audit.check(route.get("controller") == "contact_direct" and route.get("routing") == expected["routing"]
                        and route.get("previous_controller") == expected["previous_controller"]
                        and route.get("source") == "forced_direct", prefix + "forced Direct routing trace")
            audit.check(recorded.calls == before_calls + 1, prefix + "one recorded response per five-action block")
            audit.check(Path(diag.get("api_request_log") or "").name == paths[recorded.calls - 1].name,
                        prefix + "recorded request artifact link")
            audit.check(diag.get("compact_prediction") == client.last_compact_prediction,
                        prefix + "original model actions and context retained")
        else:
            audit.check(recorded.calls == before_calls, prefix + "queued actions make no extra model request")
        raw = client.last_compact_prediction
        audit.check(raw is not None and near(row.get("action"), raw["actions"][step % 5]),
                    prefix + "command equals the selected raw model action without adaptation")
        expected_events.append("action")
        if step not in executed:
            observation = None
            continue
        result = executed[step]
        expected_events.append("executed")
        if result.get("sim_status", {}).get("collision", {}).get("detected") is True:
            expected_events.append("forbidden_collision")
        audit.check(result.get("action") == row.get("action") and result.get("controller") == "contact_direct",
                    prefix + "simulator acknowledged the proposed Direct action")
        try:
            after = PolicyObservation.from_dict(result["observation"])
        except Exception:
            audit.check(False, prefix + "returned observation must contain exactly four allowed fields")
            replay_failed = True
            break
        audit.check(result.get("next_state") == list(after.state) and after.prompt == task.prompt,
                    prefix + "returned task and measured state")
        audit.check(result.get("success") == result.get("sim_status", {}).get("success"), prefix + "evaluator-only success metadata")
        if step in feedback:
            try:
                reconstructed = policy.observe_transition(observation, decision, after)
            except Exception as exc:
                audit.check(False, prefix + "feedback replay raised " + type(exc).__name__)
                replay_failed = True
                break
            expected_events.append("feedback")
            audit.check(near(feedback[step].get("feedback"), reconstructed), prefix + "measured action feedback replay")
        observation = after
    audit.check([row.get("event") for row in trace] == expected_events,
                "exact Direct routing/action/feedback order with zero field, switch or local-control events")
    if not replay_failed and recorded.calls < len(paths) and observation is not None:
        audit.check(len(paths) - recorded.calls == 1 and summary.get("status") in
                    ("user_stop", "subscription_error", "infrastructure_error"), "only one terminal unused request permitted")
        try:
            policy.act(observation)
        except Exception:
            pass
        else:
            audit.notes.append("A terminal accepted response had no logged action; no execution is inferred from it.")
    audit.check(recorded.calls == len(paths), "every recorded request reconstructed")
    usage, accepted = _helpers._usage(paths, summary, audit)
    for name, value in (("validated_hybrid_refreshes", refreshes), ("validated_direct_chunks", refreshes),
                        ("validated_field_refreshes", 0), ("local_plan_count", 0), ("coarse_field_actions", 0),
                        ("contact_direct_actions", len(executed)), ("controller_switches", 0)):
        audit.check(summary.get(name) == value, "summary count: " + name)
    for name in ("field_expansion_s", "local_planning_s"):
        audit.check(summary.get("timing_components", {}).get(name) == 0, "no inactive controller timing: " + name)
    collision_result = _helpers._collision_outcome(run, config, summary, state, resets, actions, executed, audit)
    observed_success, completed = collision_result["libero_success_observed"], collision_result["evaluation_completed"]
    audit.check(summary.get("model_declared_complete") is False and summary.get("model_completion") is None
                and summary.get("stop_on_model_complete") is False, "model completion cannot score the episode")
    video, frames = _legacy._video_audit(run, summary, actions, executed, feedback, audit)
    comparison = _reference_comparison(reference_run, run, config, summary, resets, paths, audit)
    report = {"kind": "zero_shot_direct_recorded_shared_code_audit", "passed": not audit.issues, "checks": audit.checks,
        "issues": audit.issues, "notes": audit.notes, "source_run": str(run), "status": summary.get("status"),
        "billing": summary.get("billing"), "requested_reasoning_effort": "medium", "distance_task": task.key,
        "episode_index": config.get("episode_index"),
        "initial_state_override": override, "benchmark_variant": config.get("benchmark_variant", "official_layout"),
        "task_bddl_override": scene,
        "zero_shot_direct_control": True, "zero_shot_collision_free": True, "distance_field_control": False, "demonstration_conditioned": False,
        "executed_actions": len(executed), "feedback_records": len(feedback),
        "trace_complete": len(actions) == len(executed) == len(feedback), "execution_evidence": evidence,
        "replayed_recorded_requests": recorded.calls, "accepted_recorded_responses": accepted,
        "reconstructed_local_plans": 0, "libero_success_observed": observed_success, "evaluation_completed": completed,
        "synthetic_test_fixture": fixture, "real_task_completion_verified": False,
        "recorded_control_elapsed_s": summary.get("control_elapsed_s"), "usage_totals": dict(usage),
        "recorded_control_excluding_video_s": _helpers._adjusted_control_s(summary),
        "adjusted_timing_note": "Recorded control time minus recorded video_render_s; retrospective accounting, not a no-video rerun.",
        "reference_comparison": comparison, "video": video, "event_frames": frames, "frozen_sources": str(archive),
        "current_source_mismatches": changed, "shared_code_reconstruction": True,
        "excluded_replay_fields": sorted(_helpers.REPLAY_EXCLUSIONS), "live_model_calls_by_audit": 0,
        "network_requests_by_audit": 0, "simulator_actions_by_audit": 0, "credentials_loaded_by_audit": False,
        "opaque_model_output_printed": False, "physical_object_state_independently_verified": False,
        "auditor_sha256": sha(Path(__file__)), "artifact_sha256": {name: sha(run / name) for name in
            ("manifest.json", "summary.json", "run_state.json", "trace.jsonl", "episode.mp4", "episode.events.jsonl", "demonstration.json", "initial_state_override.npy", "task_override.bddl", "original_task.bddl", "collision.json", "task_evaluation.json")
            if (run / name).is_file()}}
    report.update(collision_result)
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "checks", "issues", "status", "executed_actions",
        "replayed_recorded_requests", "reconstructed_local_plans")}, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen-sources", type=Path)
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args(argv)
    return 0 if audit_run(args.run, args.output, args.frozen_sources, reference_run=args.reference_run)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
