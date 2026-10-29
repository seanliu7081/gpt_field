"""Independent v2 user-check runner: prepare -> selector review -> one test.

Existing v2 modules and its evaluator are imported read-only. ``prepare`` resets
the simulator and requests a model draft without sending any policy actions.
``start`` performs preparation, serves the local review UI, then executes only
after the user checks every step. ``run`` consumes an already approved review.
``preflight`` performs no network calls, credential reads, or simulator work.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = "scripts/evaluate_agent_speedup_v2_user_check.py"
HTML_PATH = "src/gpt_field/agent_speedup_v2_user_check.html"
METHOD = "agent_speedup_v2_user_check"
PLAN_CONFLICT_RULE = "plan_conflict: initial grounding changed the user-reviewed plan"

_spec = importlib.util.spec_from_file_location(
    "_agent_speedup_v2_user_check_base_readonly", ROOT / "scripts/evaluate_agent_speedup_v2.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

from gpt_field.agent_speedup_v2_user_check import (  # noqa: E402
    DISPLAY_NAME, VERSION, ReviewError, ReviewStore, canonical_sha256,
    initial_observation_sha256, validate_review_plan,
)
from gpt_field.agent_speedup_v2_user_check_client import (  # noqa: E402
    PlanLock, UserCheckV2Client,
    USER_CHECK_REASONING_EFFORT,
)
from gpt_field.agent_speedup_v2_user_check_agent_client import AgentFirstV2Client  # noqa: E402
from gpt_field.agent_speedup_v2_user_check_agent import AgentFirstV2Policy  # noqa: E402
from gpt_field.agent_speedup_v2_user_check_sim import UserCheckSimProcess  # noqa: E402
from gpt_field.agent_speedup_v2_user_check_fragments import fragment_capabilities  # noqa: E402
from gpt_field.astranomy_agentic_motion import ShortTargetConfig  # noqa: E402
from gpt_field.agent_speedup_v2_user_check_control import control_settings  # noqa: E402
from gpt_field.agent_speedup_v2_user_check_framework import (  # noqa: E402
    FEEDBACK_STATUSES, FRAMEWORK_VERSION,
)


def framework_settings():
    return {
        "version": "user-check-agent-fragments-framework-2",
        "legacy_contract_version": FRAMEWORK_VERSION,
        "interfaces": ["task_contract", "shared_state", "skill_contracts", "execution_feedback"],
        "feedback_statuses": list(FEEDBACK_STATUSES),
        "state_file": "user_check_framework.json",
        "goal_source": "official_task_prompt_and_reviewed_initial_plan",
        "task_specific_branches": False,
        "model_reasoning_effort": USER_CHECK_REASONING_EFFORT,
        "control_owner": "continuous_model_agent",
        "macro_role": "independent_monitored_fragments_and_optional_native_macros; yields_back_to_model",
        "execution_fragments": fragment_capabilities(),
        "constraint_scope": "reviewed_task_constraints; model_selects_execution_method",
        "contact_semantics": "task_manipulation_constraints; evaluator_records_contacts_independently",
        "feedback": "persistent_visual_action_history_and_measured_motion; full_audit_on_disk",
        "uncertainty": "evidence_or_unavailable_observer; never_automatic_proof_of_failure",
        "observation": "two_upright_rgb_state8_seven_measured_arm_joints_synchronized_camera_and_fixed_gripper_geometry",
        "spatial_plan_fields": ["regions", "placement_goals"],
        "placement_verification": "model_visual_evidence; independent_evaluator_decides_success",
        "recovery_handoff": "model_can_finish_subtask_directly_without_return_to_macro",
        "local_motion": {
            "interface": "Astranomy_short_target_direct_control",
            "frame": "world",
            "position_modes": ["absolute", "relative"],
            "position_units": "meters",
            "orientation": "quaternion_wxyz_or_keep",
            "gripper_commands": ["keep", "open", "close"],
            "fresh_observation_between_targets": True,
            "phase_checkpoints": "model_selected_short_horizon_with_fresh_feedback",
            "controller": ShortTargetConfig().to_dict(),
        },
    }


def user_check_sim_factory(monitor_options):
    def create(config):
        return UserCheckSimProcess(config, monitor_options=monitor_options)
    return create


def execution_options(config, sim_config, *, monitor_options=None, video=True,
                      stop_after_steps=None, smoke=False, smoke_steps=None):
    """Nonsecret settings bound to approval; output/credential paths are excluded."""
    return {
        "api": asdict(config), "simulator": asdict(sim_config),
        "user_check": {
            "version": VERSION, "approval_scope": "initial_plan_only",
            "automatic_model_replanning": True, "lowering_control": control_settings(),
            "framework": framework_settings(),
        },
        "monitor": dict(monitor_options or {"goal_aware": True, "truth_window": True}),
        "video": video, "stop_after_steps": stop_after_steps, "smoke": smoke,
        "smoke_steps": smoke_steps if smoke else None,
        "python_executable": sys.executable,
    }


def runtime_inventory(config, sim_config, *, monitor_options=None, video=True,
                      stop_after_steps=None, smoke=False, smoke_steps=None):
    """Current code/data + configuration hashes, checked again before consumption."""
    names = {str(path.relative_to(ROOT)) for path in (ROOT / "src/gpt_field").rglob("*.py")}
    names.update((RUNNER_PATH, HTML_PATH, base.RUNNER_PATH, base.V1_RUNNER_PATH))
    names.update(base.V1_HELPER_SCRIPTS)
    names.update(base.HELPER_CHAIN)
    names.update(base.DATA_FILES)
    inventory = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in sorted(names)
    }
    inventory["@configuration"] = canonical_sha256(execution_options(
        config, sim_config, monitor_options=monitor_options, video=video,
        stop_after_steps=stop_after_steps, smoke=smoke, smoke_steps=smoke_steps))
    return inventory


def prepare_review(review_dir, *, config, sim_config, task_key, episode_index,
                   monitor_options=None, token_provider=None, client_factory=None,
                   sim_factory=None, transport=None, expected_initial_observation_sha256=None,
                   video=True, stop_after_steps=None, smoke=False, smoke_steps=None,
                   auto_approve=False):
    """Make one model draft before any policy action. Injected factories are test seams."""
    from gpt_field.agent_speedup_v2_client import plan_images
    from gpt_field.agent_speedup_v2_geometry import frame_alignment_turns
    from gpt_field.agent_speedup_v2_types import FINGERTIP_EDGES
    from gpt_field.contact_skills import StepView

    directory = Path(review_dir).expanduser().resolve()
    if directory.exists():
        raise FileExistsError("Preparation requires a fresh review directory")
    task = base.resolve_v2_task(task_key)
    episode_index = base._validate_episode_index(episode_index)
    base.validate_configuration(config, sim_config, task)
    base._validate_stop_after_steps(stop_after_steps)
    expected, expected_source = base.resolve_expected_sha(
        task.key, episode_index, expected_initial_observation_sha256,
        injected_sim=sim_factory is not None)
    monitor = dict(monitor_options or base.build_sim_config(task.key)[1])
    if monitor.get("oracle", False) is not False:
        raise ReviewError("User-check preparation does not use the simulator oracle")
    runtime = runtime_inventory(
        config, sim_config, monitor_options=monitor, video=video,
        stop_after_steps=stop_after_steps, smoke=smoke, smoke_steps=smoke_steps)
    directory.mkdir(parents=True, exist_ok=False)
    sim, client = None, None
    started = time.monotonic()
    try:
        with base._simulator_without_credentials():
            sim = (sim_factory or user_check_sim_factory(monitor))(sim_config)
            sim.start()
        if (getattr(sim, "metadata", None) or {}).get("oracle_enabled") is True:
            raise ReviewError("Oracle-enabled simulator is not allowed")
        tasks = [item for item in sim.list_tasks() if item.get("task_id") == task.task_id]
        if len(tasks) != 1 or tasks[0].get("prompt") != task.prompt:
            raise ReviewError("Simulator task does not match the official task registry")
        observation = sim.reset(task.task_id, episode_index)
        if observation.prompt != task.prompt:
            raise ReviewError("Reset prompt does not match the reviewed task")
        initial = observation.to_dict()
        actual = initial_observation_sha256(initial)
        if expected is not None and actual != expected:
            raise ReviewError("Reset observation differs from the frozen reset; no model call performed")
        view = StepView(observation)
        turns = int(frame_alignment_turns(view.R, None))
        images = [
            {"label": label, "data_url": url}
            for label, url in plan_images(view.external_rgb_upright, view.wrist_rgb_upright, turns)
        ]
        client = (client_factory or UserCheckV2Client)(
            config, token_provider=token_provider, task_prompt=observation.prompt,
            transport=transport, log_dir=directory / "requests")
        client.check_observation(observation)
        model_started = time.monotonic()
        draft = client.request_plan(
            view.external_rgb_upright, view.wrist_rgb_upright,
            quarter_turns=turns, fingertip_edge=FINGERTIP_EDGES[turns])
        model_seconds = time.monotonic() - model_started
        preparation = {
            "kind": "model_draft_before_policy_actions", "policy_actions": 0,
            "approval_mode": "user_authorized_automatic" if auto_approve else "manual_browser_review",
            "manual_review": not auto_approve,
            "draft_model_s": model_seconds,
            "expected_reset_sha256": expected, "expected_reset_source": expected_source,
            "quarter_turns": turns, "fingertip_edge": FINGERTIP_EDGES[turns],
            "review_images": images,
            "model_budget": client.budget_snapshot(),
            "model_history": client.history_stats(),
            "execution": execution_options(
                config, sim_config, monitor_options=monitor, video=video,
                stop_after_steps=stop_after_steps, smoke=smoke, smoke_steps=smoke_steps),
        }
    finally:
        if sim is not None:
            sim.close()
    preparation["elapsed_s"] = time.monotonic() - started
    if runtime != runtime_inventory(
            config, sim_config, monitor_options=monitor, video=video,
            stop_after_steps=stop_after_steps, smoke=smoke, smoke_steps=smoke_steps):
        raise ReviewError("Runtime changed during preparation; prepare a new review")
    store = ReviewStore.create(
        directory, task_key=task.key, episode_index=episode_index,
        task_prompt=observation.prompt, initial_observation=initial,
        plan=draft, runtime_sha256=runtime, preparation=preparation)
    frozen = directory / "frozen_source"
    frozen.mkdir()
    for name in runtime:
        if not name.startswith("@"):
            destination = frozen / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
    return store


def _seconds(start, end):
    return max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())


def stopping_rules(inherited):
    """The variant asks the model again instead of exhausting a forced-skip ladder."""
    return [rule for rule in inherited if not rule.startswith("recovery_exhausted")] + [PLAN_CONFLICT_RULE]


def reviewed_recorder_factory(factory, lock, *, output_dir=None):
    """Adapt labels/status and publish live framework state without editing base v2."""
    class ReviewedRecorder:
        def __init__(self, recorder):
            self._recorder = recorder
            self._last_framework_json = None
            self._next_framework_write = 0.0

        def __getattr__(self, name):
            return getattr(self._recorder, name)

        def _publish_framework(self, *, force=False):
            if output_dir is None:
                return
            now = time.monotonic()
            # Keep a readable live snapshot without serializing growing histories
            # on every low-level robot action. Close always writes final state.
            if not force and now < self._next_framework_write:
                return
            self._next_framework_write = now + 1.0
            value = lock.framework_snapshot()
            encoded = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
            if encoded == self._last_framework_json:
                return
            path = Path(output_dir) / "user_check_framework.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)
            self._last_framework_json = encoded

        def observe(self, *args, **kwargs):
            result = self._recorder.observe(*args, **kwargs)
            self._publish_framework(force=True)
            return result

        def decision(self, *args, **kwargs):
            result = self._recorder.decision(*args, **kwargs)
            self._publish_framework()
            return result

        def executed(self, *args, **kwargs):
            result = self._recorder.executed(*args, **kwargs)
            self._publish_framework()
            return result

        def close(self, *, status, error=None):
            if lock.conflict is not None:
                status = "plan_conflict"
                error = json.dumps({"type": "PlanConflict", "details": lock.conflict})
            try:
                return self._recorder.close(status=status, error=error)
            finally:
                self._publish_framework(force=True)

    def create(*args, **kwargs):
        if isinstance(kwargs.get("episode_id"), str):
            kwargs["episode_id"] = kwargs["episode_id"].replace(
                "AGENT SPEEDUP V2", "SPEED UP V2 USER CHECK")
        return ReviewedRecorder(factory(*args, **kwargs))

    return create


def _normalize_artifacts(output, summary, review, lock, *, variant_sources, runtime,
                         client_metadata):
    """Publish variant outcomes while retaining the unchanged evaluator's raw records."""
    output = Path(output)
    raw = output / "base_v2_artifacts"
    raw.mkdir(exist_ok=False)
    names = (
        "summary.json", "runner_summary.json", "run_state.json", "manifest.json",
        "task_evaluation.json", "episodes.jsonl", "agent_stats.json",
    )
    for name in names:
        if (output / name).exists():
            shutil.copyfile(output / name, raw / name)
    now = datetime.now(timezone.utc).isoformat()
    approval = review["approved"]
    preparation = review.get("preparation") or {}
    manual_review = preparation.get("approval_mode", "manual_browser_review") != "user_authorized_automatic"
    mode = ("offline_scripted_" if summary.get("mode", "").startswith("offline") else
            "live_chatgpt_subscription_") + METHOD
    plan_history = lock.snapshot()
    meta = {
        "version": VERSION, "display_name": DISPLAY_NAME,
        "review_id": review["id"], "revision": review["revision"],
        "approved": approval, "claim": review["claim"],
        "approval_mode": preparation.get("approval_mode", "manual_browser_review"),
        "manual_review": manual_review,
        "plan_sha256": canonical_sha256(review["plan"]),
        "context_sha256": review["context_sha256"], "runtime_sha256": runtime,
        "fixed_semantics": not lock.allow_auto_replan, "user_changes_during_test": False,
        "approval_scope": "initial_plan_only" if lock.allow_auto_replan else "episode_plan",
        "automatic_model_replanning": lock.allow_auto_replan,
        "plan_revision_count": plan_history["plan_revision_count"],
        "plan_history_file": "user_check_plan_history.json",
        "framework": framework_settings(),
        "lowering_control": control_settings(),
        "execution_client": client_metadata,
        "plan_conflict": lock.conflict,
        "preparation_elapsed_s": preparation.get("elapsed_s"),
        "preparation_model_s": preparation.get("draft_model_s"),
        "preparation_model_budget": preparation.get("model_budget"),
        "review_phase_elapsed_s": _seconds(review["created_at"], approval["at"]),
        "approval_to_claim_s": _seconds(approval["at"], review["claim"]["at"]),
        "timing_note": "Preparation and pre-test review time are separate from base control timing. "
                       "Review phase elapsed time is wall-clock waiting, not measured active human work.",
        "normalized_at": now, "base_records": "base_v2_artifacts",
    }
    summary = dict(summary, method=METHOD, mode=mode, display_name=DISPLAY_NAME,
                   zero_shot=False, human_reviewed_plan=manual_review, user_check=meta,
                   interpretation="One user-check episode with " +
                                  ("a manually reviewed" if manual_review else "a user-authorized automatically approved") +
                                  " initial plan, continuous model decisions and monitored execution fragments. "
                                  "Approval provenance and native/fragment/direct control counts are separate.")
    if lock.conflict is not None:
        summary.update(
            status="plan_conflict", failure_reason="plan_conflict",
            error={"type": "PlanConflict", "message": lock.conflict["reason"],
                   "details": lock.conflict},
            task_success=False, v2_success=False, benchmark_completed=False,
            evaluation_completed=False, completion_time_eligible=False,
            success_rate=None, scored_episodes=0, incomplete_episodes=1,
            diagnostic_complete=False,
        )
    for name in ("summary.json", "runner_summary.json"):
        base.write_json(output / name, summary)
    (output / "episodes.jsonl").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    for name in ("run_state.json", "task_evaluation.json"):
        path = output / name
        if path.exists():
            value = json.loads(path.read_text())
            value.update(method=METHOD, mode=mode, user_check=meta)
            for key in ("status", "task_success", "failure_reason", "evaluation_completed",
                        "completion_time_eligible", "success_rate", "scored_episodes",
                        "incomplete_episodes", "error"):
                if key in value or key == "status":
                    value[key] = summary[key]
            base.write_json(path, value)
    agent_path = output / "agent_stats.json"
    if agent_path.exists():
        agent = json.loads(agent_path.read_text())
        if isinstance(agent, dict):
            agent["user_check"] = meta
            base.write_json(agent_path, agent)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    identity = manifest["configuration"]
    identity.update(
        method=METHOD, mode=mode, kind=METHOD + "_full_task", zero_shot=False,
        human_reviewed_plan=manual_review, user_check=meta,
        base_runner=identity["runner"],
        runner={"path": RUNNER_PATH, "sha256": runtime[RUNNER_PATH]},
    )
    identity["client"].update(client_metadata)
    identity["client"]["instructions_note"] = (
        "Base v2 instruction identity remains under agent_speedup_v2; the active "
        "continuous agent instruction hashes are recorded here.")
    identity["agent_speedup_v2"]["scope"] = "unchanged_evaluator_and_optional_macro_reference"
    identity["active_agent"] = {
        "method": METHOD, "version": VERSION, "framework": framework_settings(),
        "model_role": "continuous_visual_task_control_and_optional_macro_selection",
        "model_declares_completion": "belief_only; independent_evaluator_owns_success",
        "camera_geometry_to_policy": True, "hidden_scene_truth_to_policy": False,
    }
    identity["stopping_rule"] = stopping_rules(identity.get("stopping_rule", []))
    manifest["identity_hash"] = base.digest(identity)
    base.write_json(manifest_path, manifest)
    base.write_json(output / "user_check_review.json", review)
    base.write_json(output / "approved_plan.json", review["plan"])
    base.write_json(output / "user_check_plan_history.json", plan_history)
    base.write_json(output / "user_check_framework.json", plan_history.get("framework", {}))
    for name, blob in variant_sources.items():
        (output / "frozen_source" / Path(name).name).write_bytes(blob)
    with (output / "trace.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "event": "user_check_terminal", "method": METHOD,
            "status": summary["status"], "step": summary.get("steps"),
            "review_id": review["id"], "plan_conflict": lock.conflict,
            "automatic_model_replanning": lock.allow_auto_replan,
            "plan_revision_count": plan_history["plan_revision_count"],
            "plan_history_file": "user_check_plan_history.json",
            "framework_version": framework_settings()["version"],
            "framework_file": "user_check_framework.json",
            "base_terminal_status": json.loads((raw / "summary.json").read_text())["status"],
        }) + "\n")
    return summary


def execute_reviewed(output, *, review_dir, config, sim_config, task_key, episode_index,
                     monitor_options=None, token_provider=None, client_factory=None,
                     policy_factory=None, sim_factory=None, recorder_factory=None,
                     video=True, stop_after_steps=None, smoke=False,
                     smoke_steps=base.DEFAULT_SMOKE_STEPS):
    """Consume approval once, then run the continuous agent with isolated adapters."""
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Every reviewed test requires a fresh output directory")
    store = ReviewStore(review_dir)
    task = base.resolve_v2_task(task_key)
    monitor = dict(monitor_options or base.build_sim_config(task.key)[1])
    base.validate_configuration(config, sim_config, task)
    base._validate_episode_index(episode_index)
    base._validate_stop_after_steps(stop_after_steps)
    if monitor.get("oracle", False) is not False:
        raise ReviewError("Reviewed execution does not use the simulator oracle")
    if not smoke and not callable(token_provider):
        raise base.MissingCredentials("Reviewed execution requires a subscription token provider")
    review = store.load()
    if review["task_prompt"] != task.prompt:
        raise ReviewError("Review prompt differs from the official task prompt")
    runtime = runtime_inventory(
        config, sim_config, monitor_options=monitor, video=video,
        stop_after_steps=stop_after_steps, smoke=smoke,
        smoke_steps=smoke_steps if smoke else None)
    variant_names = (RUNNER_PATH, HTML_PATH, *(
        name for name in runtime if name.startswith("src/gpt_field/agent_speedup_v2_user_check")
    ))
    variant_sources = {name: (ROOT / name).read_bytes() for name in variant_names}
    # A reset mismatch below deliberately consumes this attempt but sends zero
    # policy actions: it cannot silently reuse approval for another reset.
    review = store.claim(
        output=output, task_key=task.key, episode_index=episode_index,
        initial_observation_sha256=review["initial_observation_sha256"],
        runtime_sha256=runtime)
    lock = PlanLock(review["plan"], task_prompt=task.prompt)
    client_metadata = {}

    def make_client(*args, **kwargs):
        client = (client_factory or AgentFirstV2Client)(
            *args, approved_plan=review["plan"], lock=lock, **kwargs)
        instructions = getattr(client, "_instruction_text", {})
        client_metadata.update(
            class_factory=type(client).__qualname__,
            prompt_version=getattr(client, "prompt_version", None),
            reasoning_effort=getattr(client, "reasoning_effort", None),
            effective_instructions_sha256={
                kind: hashlib.sha256(instructions[kind].encode("utf-8")).hexdigest()
                for kind in instructions
            })
        return client

    def make_policy(client):
        return (policy_factory or AgentFirstV2Policy)(client, lock=lock)

    make_recorder = reviewed_recorder_factory(
        recorder_factory or (base._default_recorder_factory() if video else base.NullRecorder),
        lock, output_dir=output)
    summary = base.execute(
        output, config=config, sim_config=sim_config, task_key=task.key,
        episode_index=episode_index, monitor_options=monitor, token_provider=token_provider,
        client_factory=make_client, policy_factory=make_policy,
        sim_factory=sim_factory or user_check_sim_factory(monitor),
        recorder_factory=make_recorder,
        expected_initial_observation_sha256=review["initial_observation_sha256"],
        video=video, smoke=smoke, smoke_steps=smoke_steps,
        stop_after_steps=stop_after_steps,
    )
    return _normalize_artifacts(
        output, summary, review, lock, variant_sources=variant_sources, runtime=runtime,
        client_metadata=client_metadata)


def _live_gate(config, sim_config, task_key, episode_index, expected):
    task = base.resolve_v2_task(task_key)
    base.validate_configuration(config, sim_config, task)
    base.resolve_expected_sha(task.key, episode_index, expected)
    base.check_render_environment(warn=lambda *a, **k: print(*a, file=sys.stderr, **k))
    lint = base.lint_check_summary(sim_config.libero_root, True)
    if not lint["passed"]:
        raise ReviewError(f"T0 lint fails (require_all): {lint}")
    base.live_build_check(task.prompt)


def _start_command(task_key, episode_index, *, video=True, stop_after_steps=None):
    words = [
        sys.executable, RUNNER_PATH, "start", "--task", task_key,
        "--episode-index", str(episode_index),
        "--review-dir", "runs/speed_up_v2_user_check_review_001",
        "--output", "runs/speed_up_v2_user_check_test_001",
    ]
    if not video:
        words.append("--no-video")
    if stop_after_steps is not None:
        words.extend(("--stop-after-steps", str(stop_after_steps)))
    return f"cd {shlex.quote(str(ROOT))} && PYTHONPATH=src " + shlex.join(words)


def preflight(config, sim_config, *, task_key, episode_index, monitor_options=None,
              video=True, stop_after_steps=None):
    """Describe active agent protocol locally; never instantiate a simulator or call a model."""
    from gpt_field.agent_speedup_v2_user_check_agent_client import agent_schema
    result = base.preflight(config, sim_config, task_key=task_key, episode_index=episode_index)
    result["base_v2_reference_payloads"] = result.pop("payloads")
    result["base_v2_reference_payload_sha256"] = result.pop("payload_sha256")
    result["base_v2_reference_client"] = result.pop("client")
    result.update(
        method=METHOD, display_name=DISPLAY_NAME, version=VERSION,
        user_check_required=True, implemented_workflow=["prepare", "review", "run", "start"],
        zero_shot=False, human_reviewed_plan=True,
        automatic_model_replanning=True, approval_scope="initial_plan_only",
        framework=framework_settings(), active_response_schema=agent_schema(),
        preparation_client={"class": UserCheckV2Client.__qualname__,
                            "reasoning_effort": USER_CHECK_REASONING_EFFORT},
        client={"class": AgentFirstV2Client.__qualname__,
                "prompt_version": AgentFirstV2Client.prompt_version,
                "reasoning_effort": AgentFirstV2Client.reasoning_effort,
                "history_protocol": AgentFirstV2Client.history_protocol},
        policy={"class": AgentFirstV2Policy.__qualname__, "continuous_model_control": True},
        payload_note="Base payloads are reference-only. The active executive uses continuous "
                     "visual history, high effort, direct short targets and optional bounded macros. "
                     "No approved plan or task-specific action sequence is synthesized here.",
        lint_scope_note="Inherited lint/build metadata describes the unchanged v2 baseline; "
                        "it is not verification of the active agent's task performance.",
        live_command=_start_command(task_key, episode_index, video=video,
                                    stop_after_steps=stop_after_steps),
        live_command_options="start opens mandatory user review before execution. "
                             "Use fresh review and output directories.",
        stopping_rule=stopping_rules(result["stopping_rule"]),
        runtime_sha256=runtime_inventory(
            config, sim_config, monitor_options=monitor_options, video=video,
            stop_after_steps=stop_after_steps),
    )
    return result


def _show_review(store, *, host, port, until_approved):
    from gpt_field.agent_speedup_v2_user_check_server import ReviewServer
    with ReviewServer(store, host=host, port=port) as server:
        print(json.dumps({
            "event": "user_review_required", "review_url": server.url,
            "review_dir": str(store.path), "task": store.load()["task_prompt"],
            "note": "Select values and check every step, then confirm in the browser. "
                    "For remote access, forward the same local port through SSH.",
        }), flush=True)
        pause = threading.Event()
        while not pause.wait(0.25):
            if until_approved and store.load()["status"] == "approved":
                return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "start", "preflight"):
        command = sub.add_parser(name)
        if name != "preflight":
            command.add_argument("--review-dir", type=Path, required=True)
        if name in ("run", "start"):
            command.add_argument("--output", type=Path, required=True)
        command.add_argument("--task", choices=(*base.V2_TASK_KEYS, *base.V2_TASK_ALIASES))
        command.add_argument("--episode-index", type=int, choices=range(50))
        command.add_argument("--sim-python")
        command.add_argument("--libero-root")
        command.add_argument("--expected-initial-observation-sha256")
        command.add_argument("--no-video", action="store_true", default=None)
        command.add_argument("--stop-after-steps", type=int)
        if name != "preflight":
            command.add_argument("--store-dir", type=Path)
            command.add_argument("--profile")
        if name == "start":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8765)
            command.add_argument("--auto-approve", action="store_true",
                                 help="Explicitly delegate approval of this task's validated model plan; recorded separately from manual review")
    review_parser = sub.add_parser("review")
    review_parser.add_argument("--review-dir", type=Path, required=True)
    review_parser.add_argument("--host", default="127.0.0.1")
    review_parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == "review":
            _show_review(ReviewStore(args.review_dir), host=args.host, port=args.port,
                         until_approved=False)
            return 0
        previous = ReviewStore(args.review_dir).load() if args.command == "run" else None
        if args.command in ("start", "run") and args.output.exists():
            raise FileExistsError("Every reviewed test requires a fresh output directory")
        if args.command == "start" and (
                args.host not in ("127.0.0.1", "localhost") or not 0 <= args.port <= 65535):
            raise ReviewError("Review UI requires a loopback host and a port from 0 to 65535")
        saved = ((previous or {}).get("preparation") or {}).get("execution") or {}
        task_key = args.task or (previous or {}).get("task_key") or base.PRIMARY_TASK
        episode_index = (args.episode_index if args.episode_index is not None
                         else (previous or {}).get("episode_index", 0))
        sim_config, monitor = base.build_sim_config(
            task_key,
            sim_python=args.sim_python or saved.get("simulator", {}).get("python_executable")
            or base.DEFAULT_SIM_PYTHON,
            libero_root=args.libero_root or saved.get("simulator", {}).get("libero_root")
            or base.DEFAULT_LIBERO_ROOT)
        config = base.configuration()
        video = not args.no_video if args.no_video is not None else saved.get("video", True)
        cap = args.stop_after_steps if args.stop_after_steps is not None else saved.get("stop_after_steps")
        base._validate_stop_after_steps(cap)
        expected = (args.expected_initial_observation_sha256
                    or (previous or {}).get("initial_observation_sha256"))
        if args.command == "preflight":
            result = preflight(
                config, sim_config, task_key=task_key, episode_index=episode_index,
                monitor_options=monitor, video=video, stop_after_steps=cap)
            print(json.dumps(result, indent=2), flush=True)
            return 0 if result["lint_check"]["passed"] else 2
        _live_gate(config, sim_config, task_key, episode_index, expected)
        from gpt_field.chatgpt_auth import SubscriptionCredentials
        token_provider = SubscriptionCredentials(
            store_dir=args.store_dir, profile=args.profile).access_token
        if args.command in ("prepare", "start"):
            store = prepare_review(
                args.review_dir, config=config, sim_config=sim_config, task_key=task_key,
                episode_index=episode_index, monitor_options=monitor, token_provider=token_provider,
                expected_initial_observation_sha256=expected, video=video, stop_after_steps=cap,
                auto_approve=getattr(args, "auto_approve", False))
            if args.command == "prepare":
                print(json.dumps({
                    "status": "review_required", "method": METHOD,
                    "review_dir": str(store.path), "validation": store.load()["validation"],
                }), flush=True)
                return 0
            if args.auto_approve:
                draft = store.load()
                approval = store.approve(expected_revision=draft["revision"],
                                         reviewed_steps=list(range(len(draft["plan"]["steps"]))))
                print(json.dumps({"event": "plan_automatically_approved",
                                  "authorization": "explicit_auto_approve_argument",
                                  "manual_review": False, "review_id": approval["id"],
                                  "plan_sha256": approval["approved"]["plan_sha256"]}), flush=True)
            else:
                _show_review(store, host=args.host, port=args.port, until_approved=True)
        result = execute_reviewed(
            args.output, review_dir=args.review_dir, config=config, sim_config=sim_config,
            task_key=task_key, episode_index=episode_index, monitor_options=monitor,
            token_provider=token_provider, video=video, stop_after_steps=cap)
        print(json.dumps({
            "event": "user_check_finished", "method": METHOD, "status": result["status"],
            "output": result["output"], "review_id": result["user_check"]["review_id"],
        }), flush=True)
        return 0 if result["evaluation_completed"] else 3
    except KeyboardInterrupt:
        print("Stopped before test execution or while serving review.", file=sys.stderr)
        return 130
    except (ReviewError, ValueError, FileExistsError, base.MissingCredentials) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
