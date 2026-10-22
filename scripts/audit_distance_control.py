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
from gpt_field.demonstration import load_demonstration
from gpt_field.experiment import digest
from gpt_field.hybrid_subscription import HYBRID_GRID
from gpt_field.libero import (
    validate_initial_state_override, validate_task_bddl_override, validate_task_bddl_contract,
)
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


def _adjusted_control_s(summary):
    control = summary.get("control_elapsed_s")
    video = summary.get("timing_components", {}).get("video_render_s")
    if all(type(value) in (int, float) and math.isfinite(value) for value in (control, video)):
        return control - video
    return None


def _group(trace, kind, audit):
    selected = [row for row in trace if row.get("event") == kind]
    audit.check([row.get("step") for row in selected] == list(range(len(selected))),
                kind + " indexes are contiguous from zero")
    return {row["step"]: row for row in selected if type(row.get("step")) is int}


def _task(config):
    """Resolve new explicit identities while retaining historical bowl defaults."""
    demonstration = config.get("demonstration") or {}
    return resolve_distance_task(config.get("distance_task", config.get("task_key",
        demonstration.get("task_key", "bowl_plate"))))


def _task_identity(config, summary, audit):
    task = _task(config)
    for source, record in (("manifest", config), ("summary", summary)):
        for name in ("distance_task", "task_key"):
            if name in record:
                audit.check(record[name] == task.key, source + " task identity: " + name)
        if isinstance(record.get("demonstration"), dict):
            audit.check(record["demonstration"].get("task_key") == task.key,
                        source + " demonstration task matches the live task")
    if "task_spec" in config:
        audit.check(config["task_spec"] == json.loads(json.dumps(asdict(task))),
                    "recorded selected-task registry and recovery restrictions")
    return task


def _episode_identity(config, summary, audit):
    index = config.get("episode_index")
    audit.check(type(index) is int and 0 <= index < 50,
                "requested official initial-state index is an integer from zero through 49")
    audit.check(type(summary.get("episode_index")) is int and summary["episode_index"] == index,
                "summary preserves requested initial-state index")
    return index


def _reset_identity(config, state, simulator_record, resets, executed, audit):
    """Check recorded reset bookkeeping, never privileged physical object state."""
    index, task_id = config.get("episode_index"), config.get("task_id")
    for name, expected in (("episode_index", index), ("task_id", task_id)):
        if name in state:
            audit.check(type(state[name]) is int and state[name] == expected,
                        "terminal progress preserves selected " + name)
    count = simulator_record.get("metadata", {}).get("initial_states_per_task")
    if count is not None:
        audit.check(type(count) is int and type(index) is int and 0 <= index < count,
                    "requested initial state exists in recorded simulator inventory")
    for row in resets:
        status = row.get("sim_status", {})
        for name, expected in (("episode_index", index), ("task_id", task_id)):
            values = [("reset trace", row[name])] if name in row else []
            if name in status:
                values.append(("simulator reset", status[name]))
            if name == "episode_index" and index != 0:
                audit.check(bool(values), "nonzero initial state has explicit recorded reset evidence")
            for source, value in values:
                audit.check(type(value) is int and value == expected, source + " agrees with requested " + name)
    for step, row in executed.items():
        for name, expected in (("episode_index", index), ("task_id", task_id)):
            if name in row.get("sim_status", {}):
                value = row["sim_status"][name]
                audit.check(type(value) is int and value == expected,
                            f"executed {step} preserves simulator {name}")


def _initial_state_override(run, config, summary, simulator_record, resets, executed, audit):
    """Validate immutable reset provenance, never pass the full state to policy."""
    artifact = run / "initial_state_override.npy"
    metadata = config.get("initial_state_override")
    configured = config.get("simulator", {}).get("initial_state_override")
    if metadata is None:
        audit.check(configured is None and "initial_state_override" not in summary and not artifact.exists(),
                    "official-layout run contains no unregistered custom initial state")
        return None
    audit.check(isinstance(metadata, dict) and set(metadata) == {"artifact", "sha256", "shape", "dtype"}
                and metadata.get("artifact") == artifact.name, "custom initial state uses the registered frozen artifact")
    custom_scene = config.get("task_bddl_override") is not None
    expected_variant = "custom_scene" if custom_scene else "custom_layout"
    audit.check(summary.get("initial_state_override") == metadata
                and config.get("benchmark_variant") == summary.get("benchmark_variant") == expected_variant,
                "summary explicitly preserves custom-layout provenance")
    audit.check(configured == str(artifact.resolve()), "simulator uses the run's frozen custom state")
    try:
        actual = validate_initial_state_override(artifact)
    except Exception as exc:
        audit.check(False, "frozen custom initial state passes numeric integrity validation: " + type(exc).__name__)
        return None
    identity = {name: actual[name] for name in ("sha256", "shape", "dtype")}
    audit.check(metadata == {"artifact": artifact.name, **identity}, "custom initial-state bytes, shape and dtype match manifest")
    source = config.get("initial_state_override_source")
    audit.check(isinstance(source, dict) and set(source) == {"path", "sha256", "shape", "dtype"}
                and isinstance(source.get("path"), str) and all(source.get(k) == v for k, v in identity.items())
                and summary.get("initial_state_override_source") == source,
                "custom state retains matching original-source identity")
    worker = simulator_record.get("metadata", {})
    worker_variant = ("LIBERO task with custom scene and initial layout (explicit checkout)" if custom_scene
                      else "LIBERO task with custom initial layout (explicit checkout)")
    audit.check(worker.get("initial_state_override") == actual
                and worker.get("benchmark_variant") == worker_variant,
                "simulator metadata identifies the same frozen custom layout")
    for label, row in [("reset", item) for item in resets] + [(f"executed {step}", item) for step, item in executed.items()]:
        status = row.get("sim_status", {})
        audit.check(status.get("initial_state_source") == "custom_override"
                    and status.get("initial_state_override_file") == actual["path"]
                    and status.get("initial_state_override_sha256") == actual["sha256"],
                    label + " confirms selected frozen custom state")
    audit.notes.append("This is a custom-layout evaluation using a frozen full simulator reset state. The audit verifies "
                       "its bytes and recorded reset provenance, not the physical meaning of its coordinates. "
                       "The override and simulator metadata are not policy input; exact request reconstruction "
                       "uses only the demonstration and allowed live RGB/robot-state observations.")
    return metadata


def _task_bddl_override(run, config, summary, simulator_record, resets, executed, audit):
    """Check archived scene identity and task invariants without loading a simulator."""
    artifact, original = run / "task_override.bddl", run / "original_task.bddl"
    metadata = config.get("task_bddl_override")
    configured = config.get("simulator", {}).get("task_bddl_override")
    if metadata is None:
        audit.check(configured is None and "task_bddl_override" not in summary and not artifact.exists(),
                    "unmodified task contains no unregistered BDDL override")
        return None
    audit.check(isinstance(metadata, dict) and set(metadata) == {"artifact", "sha256", "byte_count"}
                and metadata.get("artifact") == artifact.name, "custom scene uses the registered frozen BDDL artifact")
    audit.check(summary.get("task_bddl_override") == metadata
                and config.get("benchmark_variant") == summary.get("benchmark_variant") == "custom_scene",
                "summary explicitly preserves custom-scene provenance")
    audit.check(configured == str(artifact.resolve()), "simulator uses the run's frozen scene BDDL")
    try:
        actual = validate_task_bddl_override(artifact)
    except Exception as exc:
        audit.check(False, "frozen custom scene passes BDDL file integrity validation: " + type(exc).__name__)
        return None
    identity = {name: actual[name] for name in ("sha256", "byte_count")}
    audit.check(metadata == {"artifact": artifact.name, **identity}, "custom BDDL bytes match manifest")
    source = config.get("task_bddl_override_source")
    audit.check(isinstance(source, dict) and set(source) == {"path", "sha256", "byte_count"}
                and isinstance(source.get("path"), str) and all(source.get(k) == v for k, v in identity.items())
                and summary.get("task_bddl_override_source") == source,
                "custom BDDL retains matching original-source identity")
    worker = simulator_record.get("metadata", {})
    audit.check(worker.get("task_bddl_override") == actual
                and worker.get("benchmark_variant") == "LIBERO task with custom scene and initial layout (explicit checkout)",
                "simulator metadata identifies the same frozen custom scene")
    audit.check(original.is_file(), "official source BDDL is archived for offline task-preservation validation")
    contract = None
    if original.is_file():
        try:
            contract = validate_task_bddl_contract(original, artifact)
        except Exception as exc:
            audit.check(False, "custom scene preserves original task language, goal, problem/domain and target asset classes: " + type(exc).__name__)
        else:
            audit.check(contract.get("language_preserved") is True and contract.get("goal_preserved") is True
                        and contract.get("target_assets_preserved") is True and bool(contract.get("target_assets")),
                        "custom scene preserves original task language, goal, problem/domain and target asset classes")
    for label, row in [("reset", item) for item in resets] + [(f"executed {step}", item) for step, item in executed.items()]:
        status = row.get("sim_status", {})
        audit.check(status.get("task_bddl_override_file") == actual["path"]
                    and status.get("task_bddl_override_sha256") == status.get("bddl_sha256") == actual["sha256"],
                    label + " confirms selected frozen custom scene")
        audit.check(original.is_file() and status.get("original_bddl_sha256") == sha(original)
                    and isinstance(status.get("original_bddl_file"), str),
                    label + " anchors archived official BDDL identity")
        audit.check(contract is not None and status.get("custom_task_contract") == contract,
                    label + " preserves the validated custom-scene task contract")
    audit.notes.append("This custom-scene evaluation changes scene objects/layout while retaining the archived task "
                       "language, goal and goal-target asset identities/classes. BDDL validation shares the pure "
                       "production validator and does not prove obstacle difficulty or physical feasibility. "
                       "Scene/reset artifacts and privileged simulator metadata are excluded from reconstructed model input.")
    return {**metadata, "original_bddl_sha256": sha(original) if original.is_file() else None,
            "task_contract": contract}


def _sources(config, audit, frozen):
    hashes = config.get("software", {}).get("source_sha256", {})
    required = {"distance_policy.py", "distance_planner.py", "distance_subscription.py",
                "local_visual_feedback.py", "hybrid_policy.py", "policy.py", "tracking.py"}
    if _task(config).key != "bowl_plate" or "demonstration" in config:
        required.add("distance_tasks.py")
    if "demonstration" in config:
        required.add("demonstration.py")
    if config.get("initial_state_override") is not None or config.get("task_bddl_override") is not None:
        required.add("libero.py")
    if config.get("policy", {}).get("zero_shot_collision_free") is True:
        required.add("libero.py")
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
    task = _task_identity(config, summary, audit)
    demonstration = "demonstration" in config
    policy = config.get("policy", {})
    zero_shot = policy.get("zero_shot_collision_free") is True
    required = {"class": "gpt_field.distance_policy.DistanceHybridPolicy", "distance_field_control": True,
        "field_representation": distance_field_version(task.key),
        "prompt_version": distance_prompt_version(task.key, demonstration=demonstration,
                                                   **({"zero_shot_collision_free": True} if zero_shot else {})),
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
                and config.get("task_id") == summary.get("task_id") == task.task_id,
                "requested task language, task index")
    _episode_identity(config, summary, audit)
    if task.key != "bowl_plate":
        audit.check(config.get("task_spec") == json.loads(json.dumps(asdict(task))),
                    "recorded task registry and recovery restrictions match the selected task")
        audit.check(summary.get("distance_task") == task.key, "summary preserves explicit task selection")
    simulator = config.get("simulator", {})
    audit.check(simulator.get("suite") == summary.get("suite") == task.suite and simulator.get("seed") == 7,
                "recorded task suite and seed-seven configuration")
    audit.check(simulator.get("stop_on_success") is True and config.get("stop_on_model_complete") is False,
                "only actual evaluator success completes the task")
    success_rule = "LIBERO success with no forbidden collision" if zero_shot else "LIBERO success"
    audit.check(success_rule in config.get("stopping_rule", [])
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
    if zero_shot:
        _zero_shot_configuration(config, summary, audit)


def _zero_shot_configuration(config, summary, audit):
    audit.check(config.get("policy", {}).get("zero_shot_collision_free") is True
                and config.get("zero_shot_collision_free") is True
                and summary.get("zero_shot_collision_free") is True,
                "explicit zero-shot collision-free protocol in manifest and summary")
    audit.check(_task(config).key == "bowl_plate", "collision-free protocol selects the bowl task")
    audit.check("demonstration" not in config and "demonstration" not in summary,
                "zero-shot protocol has no external demonstration metadata")
    audit.check(config.get("simulator", {}).get("collision_check_mode") == "bowl_obstacles_v1",
                "substep collision evaluator explicitly enabled")
    audit.check(config.get("initial_state_override") is not None and config.get("task_bddl_override") is not None,
                "zero-shot obstacle test freezes both custom scene and initial state")
    audit.check("forbidden collision (task failure)" in config.get("stopping_rule", []),
                "forbidden collision is an explicit terminal failure condition")
    audit.check(config.get("stop_on_model_complete") is False and summary.get("stop_on_model_complete") is False,
                "collision and goal judgments remain evaluator-only")


def _zero_shot_request(payload, audit, prefix):
    """The reconstructed live transcript has no reference episode or oracle text."""
    from gpt_field.distance_subscription import shared_zero_shot_task_guidance
    for name, value in shared_zero_shot_task_guidance().items():
        audit.check(value in payload.get("instructions", ""), prefix + "shared zero-shot guidance: " + name)
    inputs = payload.get("input", [])
    reference = False
    for item in inputs:
        if not isinstance(item, dict):
            continue
        for part in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if not isinstance(part, dict) or part.get("type") != "input_text":
                continue
            try:
                parsed = json.loads(part.get("text", ""))
            except (ValueError, TypeError):
                continue
            if isinstance(parsed, dict) and (parsed.get("type") == "external_demonstration"
                                            or parsed.get("not_current_episode") is True):
                reference = True
    audit.check(not reference, prefix + "no demonstration prefix in zero-shot live input")


def _collision_outcome(run, config, summary, state, resets, actions, executed, audit):
    """Validate the recorded evaluator ledger, without querying physical state."""
    from gpt_field.libero import _BowlObstacleCollisionMonitor
    scope = {key: list(getattr(_BowlObstacleCollisionMonitor, key))
             for key in ("monitored_scope", "allowances", "excluded_scope")}
    worker = read(run / "simulator.json").get("metadata", {})
    audit.check(worker.get("collision_check_mode") == "bowl_obstacles_v1"
                and worker.get("collision_monitoring") == {
                    "sampling": "every actual physics substep and final control endpoint", **scope},
                "simulator metadata declares the exact finite collision criterion")
    count_keys = {"checks", "physics_substeps", "forbidden_contact_points", "allowed_contact_points",
                  "positive_distance_contacts_ignored"}
    required = {"enabled", "mode", "detected", "first_event", "counts", "monitored_scope", "allowances", "excluded_scope"}
    previous_counts, first, first_step = None, None, None
    raw_observed, qualified_steps = False, []
    records = [("reset", None, row.get("sim_status", {})) for row in resets]
    records += [(f"executed {step}", step, row.get("sim_status", {})) for step, row in executed.items()]
    final = None
    initial_collision = False
    for label, step, status in records:
        action_substeps = None
        collision = status.get("collision")
        audit.check(isinstance(collision, dict) and set(collision) == required,
                    label + " exact evaluator collision ledger schema")
        if not isinstance(collision, dict):
            continue
        final = collision
        audit.check(collision.get("enabled") is True and collision.get("mode") == "bowl_obstacles_v1"
                    and type(collision.get("detected")) is bool, label + " collision checking remains enabled")
        for key in ("monitored_scope", "allowances", "excluded_scope"):
            audit.check(isinstance(collision.get(key), list) and bool(collision[key])
                        and all(isinstance(item, str) and item for item in collision[key]),
                        label + " explicit finite collision scope: " + key)
            audit.check(collision.get(key) == scope[key], label + " collision scope matches frozen evaluator: " + key)
            if records and records[0][2].get("collision") is not None:
                audit.check(collision.get(key) == records[0][2]["collision"].get(key),
                            label + " collision scope preserved from reset: " + key)
        counts = collision.get("counts", {})
        valid_counts = (isinstance(counts, dict) and set(counts) == count_keys
                        and all(type(value) is int and value >= 0 for value in counts.values()))
        audit.check(valid_counts, label + " nonnegative integer cumulative contact counts")
        if valid_counts and previous_counts is not None:
            audit.check(all(counts[key] >= previous_counts[key] for key in count_keys),
                        label + " collision counters never decrease")
            if step is not None:
                action_substeps = counts["physics_substeps"] - previous_counts["physics_substeps"]
                delta_checks = counts["checks"] - previous_counts["checks"]
                audit.check(counts["checks"] > previous_counts["checks"]
                            and counts["physics_substeps"] > previous_counts["physics_substeps"],
                            label + " evaluator inspected physical substeps for this action")
                audit.check(delta_checks in (action_substeps, action_substeps + 1)
                            and (collision.get("detected") is True or delta_checks == action_substeps + 1),
                            label + " contact checks cover each substep and the final endpoint unless collision aborted")
        if valid_counts:
            audit.check(counts["checks"] >= counts["physics_substeps"] + 1,
                        label + " contact checks include reset and each recorded physics substep")
            if step is None:
                audit.check(counts["checks"] == 1 and counts["physics_substeps"] == 0,
                            label + " settling is excluded and initial configuration checked once")
            previous_counts = counts
        event = collision.get("first_event")
        detected = collision.get("detected") is True
        audit.check(detected is (event is not None), label + " collision detection and first event agree")
        if valid_counts:
            audit.check(detected is (counts["forbidden_contact_points"] > 0),
                        label + " forbidden contact count and collision flag agree")
        if first is not None:
            audit.check(detected and event == first, label + " first collision remains latched")
        if detected and first is None:
            first, first_step = event, step
            initial_collision = step is None
            event_keys = {"phase", "action_index", "physics_substep", "contact_distance_m", "geom1", "geom2", "reason"}
            audit.check(isinstance(event, dict) and set(event) == event_keys,
                        label + " first collision has exact contact provenance")
            if isinstance(event, dict):
                audit.check(event.get("phase") == ("initial_state" if step is None else "action")
                            and event.get("action_index") == step
                            and (event.get("action_index") is None if step is None else type(event.get("action_index")) is int),
                            label + " collision belongs to the recorded reset or action")
                substep = event.get("physics_substep")
                audit.check(substep is None if step is None else type(substep) is int and substep >= 0,
                            label + " collision identifies a physical substep")
                if step is not None:
                    audit.check(action_substeps is not None and substep == action_substeps - 1,
                                label + " physics stops at the first offending substep or synchronized endpoint")
                distance = event.get("contact_distance_m")
                audit.check(type(distance) in (int, float) and math.isfinite(distance) and distance <= 0,
                            label + " forbidden contact is actual touching or penetration")
                audit.check(isinstance(event.get("reason"), str) and bool(event["reason"]),
                            label + " forbidden-contact reason is recorded")
                for key in ("geom1", "geom2"):
                    geom = event.get(key)
                    audit.check(isinstance(geom, dict) and set(geom) == {"id", "name", "body", "object", "role"}
                                and type(geom.get("id")) is int and geom["id"] >= 0
                                and all(isinstance(geom.get(k), str) for k in ("name", "body", "object", "role")),
                                label + " exact evaluator geometry identity: " + key)
                if all(isinstance(event.get(k), dict) and "role" in event[k] for k in ("geom1", "geom2")):
                    reason = _BowlObstacleCollisionMonitor._forbidden(event["geom1"], event["geom2"])
                    audit.check(reason is not None and event.get("reason") == reason,
                                label + " recorded contact roles satisfy the documented forbidden-pair rule")
        raw = status.get("libero_success")
        qualified = status.get("success")
        audit.check(type(raw) is bool and type(qualified) is bool and qualified == (raw and not detected),
                    label + " qualified success requires raw LIBERO goal and no forbidden contact")
        raw_observed = raw_observed or raw is True
        if qualified is True:
            qualified_steps.append(step)
        if detected or qualified:
            audit.check(status.get("terminated") is True, label + " collision or qualified success terminates evaluator")
        if step is not None:
            audit.check(executed[step].get("success") is qualified, label + " trace uses qualified success only")
    audit.check(bool(records) and final is not None, "collision-free run includes evaluator reset evidence")
    if first is not None:
        audit.check(not actions if initial_collision else max(actions, default=-1) == first_step,
                    "no proposed or executed robot action follows first forbidden collision")
        audit.check(not executed if initial_collision else max(executed, default=-1) == first_step,
                    "first forbidden collision is the terminal acknowledged action")
    if qualified_steps:
        audit.check(qualified_steps == ([max(executed)] if executed else [None]),
                    "no action follows first collision-free goal success")
    audit.check(summary.get("collision") == final and state.get("collision") == final,
                "summary and terminal progress retain exact final collision ledger")
    audit.check(state.get("zero_shot_collision_free") is True, "terminal progress preserves zero-shot protocol")
    collision_path = run / "collision.json"
    audit.check(collision_path.is_file() and read(collision_path) == final,
                "separate collision artifact retains the complete final evaluator ledger")
    evaluation_path = run / "task_evaluation.json"
    evaluation_keys = ("status", "task_success", "libero_success", "collision", "failure_reason",
                       "evaluation_completed", "success_rate", "scored_episodes", "incomplete_episodes", "completion_time_eligible")
    audit.check(evaluation_path.is_file() and read(evaluation_path) == {key: summary.get(key) for key in evaluation_keys},
                "separate task evaluation artifact exactly preserves the outcome and failure reason")
    success = bool(qualified_steps) and first is None
    failed = first is not None and not initial_collision
    completed = success or failed
    audit.check(summary.get("failure_reason") == ("forbidden_collision" if failed else None),
                "failure reason explicitly distinguishes forbidden collision")
    collision_events = [row for row in rows(run / "trace.jsonl") if row.get("event") == "forbidden_collision"]
    expected_collision_events = ([{"event": "forbidden_collision", "step": first_step, "collision": final,
                                  "libero_success": records[-1][2].get("libero_success"), "task_success": False}]
                                 if failed else [])
    audit.check(collision_events == expected_collision_events,
                "one exact evaluator-only forbidden-collision event accompanies the final failed action")
    expected_status = "success" if success else "collision_failure" if failed else None
    audit.check(expected_status is None or summary.get("status") == expected_status,
                "collision failure cannot be relabeled as successful completion")
    audit.check(summary.get("status") not in ("success", "collision_failure") or completed,
                "scored terminal outcomes require qualified goal or forbidden-contact evidence")
    if initial_collision:
        audit.check(summary.get("status") == "infrastructure_error", "invalid initial collision is unscored infrastructure failure")
    audit.check(summary.get("libero_success_observed") is raw_observed,
                "raw LIBERO observation remains distinct from collision-free task success")
    final_status = records[-1][2] if records else {}
    audit.check(summary.get("libero_success") is final_status.get("libero_success")
                and summary.get("task_success") is success,
                "summary preserves raw goal signal and qualified task outcome separately")
    audit.check(summary.get("completion_time_eligible") is success,
                "only collision-free task success is eligible for completion time comparison")
    audit.check(summary.get("evaluation_completed") is completed and summary.get("benchmark_completed") is completed
                and summary.get("scored_episodes") == int(completed)
                and summary.get("incomplete_episodes") == int(not completed)
                and summary.get("success_rate") == (float(success) if completed else None)
                and summary.get("requested_episodes") == 1,
                "collision failure is one scored failure, not success or an incomplete episode")
    audit.notes.append("Collision verification checks recorded substep counts, contact identities and the latched evaluator "
                       "outcome. It does not independently replay physics or establish absence of all possible contacts "
                       "outside the documented monitored scope. Contacts and scene state are excluded from model input.")
    if failed:
        audit.notes.append("This trial ended in a scored collision failure. Its elapsed time is not a successful completion "
                           "time and is ineligible for a completion-speed ratio.")
    elif not completed:
        audit.notes.append("This trial is incomplete; elapsed time is ineligible for a completion-speed ratio.")
    return {"collision": final, "task_success": success, "collision_failure": failed,
            "libero_success_observed": raw_observed, "evaluation_completed": completed,
            "completion_time_eligible": success}


def _demonstration(run, config, summary, audit):
    """Load only the run's immutable observable reference, never a simulator."""
    if "demonstration" not in config:
        audit.check("demonstration" not in summary and not (run / "demonstration.json").exists(),
                    "unconditioned protocol contains no unregistered demonstration")
        return None
    task = _task(config)
    metadata = config["demonstration"]
    audit.check(task.key in ("drawer_bowl", "bowl_plate"), "demonstration conditioning uses a supported task")
    audit.check(summary.get("demonstration") == metadata, "summary preserves exact demonstration identity")
    audit.check(isinstance(metadata, dict) and metadata.get("artifact") == "demonstration.json",
                "demonstration uses a fixed local run artifact, not an external evaluation path")
    if not isinstance(metadata, dict) or metadata.get("artifact") != "demonstration.json":
        return None
    try:
        reference = load_demonstration(run / "demonstration.json", task_key=task.key)
    except Exception as exc:
        audit.check(False, "frozen demonstration passes strict observable-only schema and integrity validation: " + type(exc).__name__)
        return None
    audit.check(metadata == {"artifact": "demonstration.json", **reference.metadata()},
                "frozen demonstration hashes, source, frame indexes and identity match the manifest")
    audit.notes.append("This is a one-external-demonstration-conditioned evaluation. The frozen external reference "
                       "is model input only, not a live observation, executed episode, or current-task success signal.")
    audit.notes.append("The demo bundle contains recorded normalized actions and their observable after-states plus "
                       "sampled RGB keyframes. Its HDF5 identity is exporter provenance; this audit validates the "
                       "frozen bundle and exact request inclusion without independently reopening the source HDF5.")
    return reference


class RecordedTransport:
    """Serve only already-recorded responses; never dispatch an HTTP request."""

    def __init__(self, request_paths, config, audit, demonstration=None):
        self.paths, self.config, self.audit = request_paths, config, audit
        self.task = _task(config)
        self.demonstration = demonstration
        self.demonstration_claimed = "demonstration" in config
        self.zero_shot = config.get("policy", {}).get("zero_shot_collision_free") is True
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
                         and record.get("prompt_version") == distance_prompt_version(
                             self.task.key, demonstration=self.demonstration_claimed,
                             **({"zero_shot_collision_free": True} if self.zero_shot else {})),
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
        self.audit.check(payload.get("instructions") == distance_instructions(
                             2, task_key=self.task.key, demonstration=self.demonstration_claimed,
                             **({"zero_shot_collision_free": True} if self.zero_shot else {}))
                         and payload.get("text", {}).get("format") == {"type": "json_schema",
                             "name": distance_schema_name(self.task.key,
                                 **({"zero_shot_collision_free": True} if self.zero_shot else {})), "strict": True,
                             "schema": distance_output_schema(task_key=self.task.key)},
                         prefix + "strict selected-task prompt and schema")
        if self.zero_shot:
            _zero_shot_request(payload, self.audit, prefix)
        if self.demonstration is not None:
            message = self.demonstration.message()
            inputs = record.get("payload", {}).get("input", [])
            header = message.get("content", [])[:1]
            def is_demo_message(item):
                return (isinstance(item, dict) and item.get("role") == message.get("role")
                        and isinstance(item.get("content"), list) and item["content"][:1] == header)
            self.audit.check(bool(inputs) and inputs[0] == message
                             and sum(is_demo_message(item) for item in inputs) == 1,
                             prefix + "one exact immutable external demonstration precedes live observations")
            self.audit.check(not any(is_demo_message(item) for item in self.client.history_snapshot()),
                             prefix + "external demonstration stays outside live transcript and image retention")
            stats = self.client.history_stats()
            self.audit.check(stats.get("observations") == index
                             and stats.get("retained_image_pairs") == min(index, 2),
                             prefix + "demonstration frames do not count as live observations or retained live image pairs")
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
    task = _task(config)
    zero_shot = config.get("policy", {}).get("zero_shot_collision_free") is True
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
    demonstration = _demonstration(run, config, summary, audit)
    archive = Path(frozen).resolve() if frozen is not None else run / "frozen_source"
    audit.check(archive.is_dir(), "frozen source archive exists")
    changed = _sources(config, audit, archive if archive.is_dir() else None)
    trace = rows(run / "trace.jsonl")
    resets = [row for row in trace if row.get("event") == "reset"]
    audit.check(len(resets) == 1 and bool(trace) and trace[0].get("event") == "reset", "one reset precedes all recorded actions")
    actions, executed, feedback = (_group(trace, kind, audit) for kind in ("action", "executed", "feedback"))
    _reset_identity(config, state, simulator_record, resets, executed, audit)
    override = _initial_state_override(run, config, summary, simulator_record, resets, executed, audit)
    scene = _task_bddl_override(run, config, summary, simulator_record, resets, executed, audit)
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
    audit.check(summary.get("status") in (("success", "user_stop", "subscription_error", "infrastructure_error")
                + (("collision_failure",) if zero_shot else ())),
                "terminal outcome belongs to the uncapped LIBERO-only protocol")
    execution_evidence = _legacy._execution_evidence(summary, state, actions, executed, audit)

    paths = sorted((run / "requests").glob("*.request.json"))
    recorded = RecordedTransport(paths, config, audit, demonstration)
    client = ChatGPTPlanDistanceClient(AstraConfig(**config["api"]), task_key=task.key,
        demonstration=demonstration,
        **({"zero_shot_collision_free": True} if zero_shot else {}),
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
        if zero_shot and result.get("sim_status", {}).get("collision", {}).get("detected") is True:
            expected_events.append("forbidden_collision")
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
    completed = summary.get("status") == "success" and observed_success
    collision_result = None
    if zero_shot:
        collision_result = _collision_outcome(run, config, summary, state, resets, actions, executed, audit)
        observed_success = collision_result["libero_success_observed"]
        completed = collision_result["evaluation_completed"]
    else:
        audit.check(summary.get("libero_success_observed") is observed_success, "summary evaluator-success evidence")
        audit.check(not success_steps or success_steps == [max(executed)], "no robot action after first actual evaluator success")
        audit.check(summary.get("status") != "success" or observed_success, "success status requires actual evaluator evidence")
    audit.check(summary.get("model_declared_complete") is False and summary.get("model_completion") is None
                and summary.get("stop_on_model_complete") is False
                and not any(row.get("event") == "model_completion" for row in trace),
                "model or visual placement judgment never ends or scores the experiment")
    if not zero_shot:
        audit.check(summary.get("evaluation_completed") is completed and summary.get("benchmark_completed") is completed
                    and summary.get("scored_episodes") == int(completed)
                    and summary.get("incomplete_episodes") == int(not completed)
                    and summary.get("libero_success") == (True if completed else None)
                    and summary.get("success_rate") == (1.0 if completed else None), "completion and incomplete-score semantics")
    if demonstration is not None:
        audit.check(summary.get("requested_episodes") == 1,
                    "external demonstration is not counted as an executed or scored evaluation episode")
    if not completed and not zero_shot:
        audit.notes.append("This experiment is incomplete. Its elapsed wall time is not a successful completion time; no success speed ratio is inferred.")
    video, frames = _legacy._video_audit(run, summary, actions, executed, feedback, audit)
    _distance_video_metadata(run, actions, audit)
    report = {"kind": "distance_control_recorded_shared_code_audit", "passed": not audit.issues,
        "checks": audit.checks, "issues": audit.issues, "notes": audit.notes, "source_run": str(run),
        "status": summary.get("status"), "billing": summary.get("billing"), "requested_reasoning_effort": "medium",
        "distance_task": task.key, "task": task.prompt, "suite": task.suite, "task_id": task.task_id,
        "episode_index": config.get("episode_index"),
        "initial_state_override": override, "benchmark_variant": config.get("benchmark_variant", "official_layout"),
        "task_bddl_override": scene,
        "demonstration_conditioned": "demonstration" in config,
        "demonstration_validated": demonstration is not None,
        "demonstration": demonstration.metadata() if demonstration is not None else None,
        "distance_field_control": True, "executed_actions": len(executed), "feedback_records": len(feedback),
        "trace_complete": len(actions) == len(executed) == len(feedback), "execution_evidence": execution_evidence,
        "replayed_recorded_requests": recorded.calls, "accepted_recorded_responses": accepted_count,
        "reconstructed_local_plans": counts["plan"], "controller_switch_steps": switch_steps,
        "distance_control_events": len(events), "distance_event_reasons": dict(reasons),
        "libero_success_observed": observed_success, "evaluation_completed": completed,
        "synthetic_test_fixture": fixture, "real_task_completion_verified": False,
        "recorded_control_elapsed_s": summary.get("control_elapsed_s"), "usage_totals": dict(usage),
        "recorded_control_excluding_video_s": _adjusted_control_s(summary),
        "adjusted_timing_note": "Recorded control time minus recorded video_render_s; retrospective accounting, not a no-video rerun.",
        "video": video, "event_frames": frames, "frozen_sources": str(archive),
        "current_source_mismatches": changed, "shared_code_reconstruction": True,
        "excluded_replay_fields": sorted(REPLAY_EXCLUSIONS), "live_model_calls_by_audit": 0,
        "network_requests_by_audit": 0, "simulator_actions_by_audit": 0, "credentials_loaded_by_audit": False,
        "opaque_model_output_printed": False, "physical_object_state_independently_verified": False,
        "auditor_sha256": sha(Path(__file__)),
        "artifact_sha256": {name: sha(run / name) for name in
            ("manifest.json", "summary.json", "run_state.json", "trace.jsonl", "episode.mp4", "episode.events.jsonl", "demonstration.json", "initial_state_override.npy", "task_override.bddl", "original_task.bddl", "collision.json", "task_evaluation.json")
            if (run / name).is_file()}}
    if zero_shot:
        report.update(zero_shot_collision_free=True, **collision_result)
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
