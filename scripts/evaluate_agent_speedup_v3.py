"""Agent SpeedUp v3: shared geometry and adaptive motion, with manual plan review.

prepare resets the official task and requests its initial draft, but executes no
policy actions. review serves the existing selection controls. start prepares,
waits for human approval, then runs once; run consumes an already approved plan.
An explicit prepare-only delegation note permits agent review before a separate
approve call. No auto-approval, smoke, episode-cap or retry option is exposed. The read-only
v2 evaluator retains raw independent LIBERO outcomes and full request/action logs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import time

from gpt_field.agent_speedup_v3_review import (
    DISPLAY_NAME, VERSION, ReviewError, ReviewStore, canonical_sha256,
    MANUAL_REVIEW, approval_metadata, delegated_review_preparation,
)
from gpt_field.agent_speedup_v3_server import ReviewServer
from gpt_field.agent_speedup_v3_performance import (
    adapt_private_evaluator, completion_target, performance_settings,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = "scripts/evaluate_agent_speedup_v3.py"
HTML_PATH = "src/gpt_field/agent_speedup_v3.html"
REFERENCE_RUNNER_PATH = "scripts/evaluate_agent_speedup_v2_user_check.py"
REFERENCE_HTML_PATH = "src/gpt_field/agent_speedup_v2_user_check.html"
METHOD = "agent_speedup_v3"
EVALUATION_TASKS = ("libero_10_t1", "libero_10_t2", "libero_10_t3")


def _reference_module():
    spec = importlib.util.spec_from_file_location(
        "_agent_speedup_v3_runner_reference", ROOT / REFERENCE_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Public evaluator/configuration seam, with no simulator or credential access.
_reference = _reference_module()
base = _reference.base


def framework_settings(approval_preparation=None):
    from gpt_field.agent_speedup_v3_fragments import adaptive_fragment_capabilities
    from gpt_field.agent_speedup_v3_client import subscription_recovery_settings
    settings = _reference.framework_settings()
    settings.update(
        version="agent-speedup-v3-shared-geometry-framework-9",
        control_owner="continuous_model_agent_with_shared_geometry_local_planner",
        execution_fragments=adaptive_fragment_capabilities(),
        shared_geometry={
            "persistent": True,
            "source": "public_robot_measurements_model_hypotheses_and_calibrated_local_RGB_evidence",
            "scope": "task_objects_parts_supports_regions_tool_and_payload",
            "uncertainty": "source_validity_age_and_dependencies_retained",
            "planning": "candidate_path_sweep_and_adaptive_local_horizon",
            "validity": "directed_pose_dependencies_and_local_visual_disturbance; tracking_revokes_motion_not_scene; age_is_diagnostic_only",
            "visual_monitoring": "local_RGB_consistency_and_credible_two_view_displacement; not_a_free_space_certificate",
            "interaction_inventory": "model_reviewed_support_and_neighbor_ids_required_near_interaction",
            "automatic_global_route_search": False,
            "unknown_is_free": False,
            "simulator_scene_truth": False,
        },
        approval_mode=approval_metadata(approval_preparation)["approval_mode"],
        model_reasoning_effort="high",
        subscription_recovery=subscription_recovery_settings(),
        performance=performance_settings(),
        operation_state={"persistent_across_routes": True, "verification_target": "explicit_manipulated_object",
                         "grasp_release": "two_view_evidence_and_accepted_geometry_confirmation",
                         "local_recovery": "bounded_geometry_supported_candidates; cumulative_budget"},
    )
    return settings



class TerminalActTimingMixin:
    """Retain call timings when policy application raises before act returns."""
    def reset(self):
        self._v3_terminal_act_timings = []
        self._v3_current_act_calls = []
        return super().reset()

    def _ask_agent(self, view, calls):
        self._v3_current_act_calls = calls
        return super()._ask_agent(view, calls)

    def _next_macro(self, view, calls, outcomes, pointing):
        self._v3_current_act_calls = calls
        return super()._next_macro(view, calls, outcomes, pointing)

    def act(self, observation):
        self._v3_current_act_calls = []
        started = self._clock()
        try:
            return super().act(observation)
        except BaseException as exc:
            duration = max(0., self._clock()-started)
            calls = [{"kind": call.kind, "latency_s": float(call.latency_s),
                      "ok": bool(getattr(call, "ok", True)),
                      "error": getattr(call, "error", None),
                      "request_key": getattr(call, "request_key", None),
                      "request_log": getattr(call, "request_log", None)}
                     for call in self._v3_current_act_calls]
            self._v3_terminal_act_timings.append({
                "step": self.step_index, "decision_s": duration,
                "model_pipeline_s": sum(call["latency_s"] for call in calls),
                "calls": calls, "exception_type": type(exc).__name__,
                "action_returned": False, "counted_by_base_evaluator": False})
            raise

    def agent_stats(self):
        result = super().agent_stats()
        result["terminal_act_timings"] = deepcopy(self._v3_terminal_act_timings)
        profile = getattr(self, "performance_stats", None)
        if callable(profile):
            result["performance_timings"] = profile()
        return result


def _timed_policy_class(policy_class):
    return type("Timed" + policy_class.__name__, (TerminalActTimingMixin, policy_class),
                {"__module__": __name__})


def _restore_terminal_act_accounting(output_dir, summary):
    """Supplement normalized records only, preserving the evaluator's raw files."""
    if "terminal_policy_act_accounting" in summary:
        return summary
    path = Path(output_dir) / "agent_stats.json"
    if not path.exists():
        return summary
    rows = json.loads(path.read_text()).get("terminal_act_timings") or []
    if not rows:
        return summary
    added_decision = sum(float(row["decision_s"]) for row in rows)
    added_model = sum(float(row["model_pipeline_s"]) for row in rows)
    timing = summary.setdefault("timing_components", {})
    timing["decision_s"] = float(timing.get("decision_s") or 0) + added_decision
    timing["model_pipeline_s"] = float(timing.get("model_pipeline_s") or 0) + added_model
    timing["policy_compute_s"] = max(0., timing["decision_s"]-timing["model_pipeline_s"])
    accounting = summary.setdefault("model_calls", {})
    kinds = accounting.setdefault("runner_calls_by_kind", {})
    failed = accounting.setdefault("failed_calls_by_kind", {})
    latencies = accounting.setdefault("latency_s_by_kind", {})
    pipelines = timing.setdefault("model_pipeline_s_by_kind", {})
    for row in rows:
        for call in row["calls"]:
            kind, duration = call["kind"], float(call["latency_s"])
            kinds[kind] = kinds.get(kind, 0) + 1
            if not call["ok"]:
                failed[kind] = failed.get(kind, 0) + 1
            latencies[kind] = float(latencies.get(kind) or 0) + duration
            pipelines[kind] = float(pipelines.get(kind) or 0) + duration
    correction = {"version": "v3-terminal-act-accounting-1", "failed_act_count": len(rows),
                  "decision_s_added": added_decision, "model_pipeline_s_added": added_model,
                  "runner_calls_added": sum(len(row["calls"]) for row in rows),
                  "source": "measured policy act and ExecutiveCall timers; no artifact-time approximation",
                  "client_acceptance_counts_unchanged": True,
                  "raw_evaluator_records": "base_v2_artifacts", "details": rows}
    summary["terminal_policy_act_accounting"] = correction
    summary["timing_components_note"] = (summary.get("timing_components_note", "") +
        " V3 includes measured terminal policy.act time and its model calls even when act raises before returning.")
    base.write_json(Path(output_dir) / "terminal_policy_act_timing.json", correction)
    return summary

def _runtime(approval_preparation=None):
    """Bind factories in a private module namespace, never patch a shared v2 module."""
    from gpt_field.agent_speedup_v3 import AgentSpeedUpV3Policy
    from gpt_field.agent_speedup_v3_client import AgentSpeedUpV3Client, AgentSpeedUpV3PlanClient
    approval = approval_metadata(approval_preparation)
    module = _reference_module()
    adapt_private_evaluator(module.base)
    from gpt_field.agent_speedup_v3_video import AgentSpeedUpV3VideoRecorder
    module.base._default_recorder_factory = lambda: AgentSpeedUpV3VideoRecorder
    module.RUNNER_PATH, module.HTML_PATH, module.METHOD = RUNNER_PATH, HTML_PATH, METHOD
    module.VERSION, module.DISPLAY_NAME = VERSION, DISPLAY_NAME
    module.ReviewStore, module.ReviewError = ReviewStore, ReviewError
    module.AgentFirstV2Client = AgentSpeedUpV3Client
    module.AgentFirstV2Policy = _timed_policy_class(AgentSpeedUpV3Policy)
    module.UserCheckV2Client = AgentSpeedUpV3PlanClient
    module.framework_settings = lambda: framework_settings(approval_preparation)
    inherited_recorder = module.reviewed_recorder_factory

    def reviewed_recorder(factory, *args, **kwargs):
        def labelled(*values, **options):
            directory = kwargs.get("output_dir")
            if directory is not None:
                base.write_json(Path(directory) / "active_agent.json", {
                    "method": METHOD, "version": VERSION, "display_name": DISPLAY_NAME,
                    "framework": module.framework_settings(),
                    "human_reviewed_initial_plan": approval["manual_review"],
                    "approval_mode": approval["approval_mode"],
                    "approved_by": approval["reviewer"], "approval_provenance": approval,
                    "model": "gpt-6-astra", "reasoning_effort": "high",
                    "billing": "chatgpt_subscription", "api_key_fallback": False,
                    "success_source": "independent_raw_LIBERO_evaluator",
                    "base_manifest_note": "The base evaluator manifest is preserved; v3 normalization follows finalization.",
                })
            if isinstance(options.get("episode_id"), str):
                options["episode_id"] = options["episode_id"].replace(
                    "AGENT SPEEDUP V2", "AGENT SPEEDUP V3").replace(
                    "SPEED UP V2 USER CHECK", "AGENT SPEEDUP V3")
            return factory(*values, **options)
        return inherited_recorder(labelled, *args, **kwargs)

    module.reviewed_recorder_factory = reviewed_recorder
    inherited_inventory = module.runtime_inventory

    def inventory(*args, **kwargs):
        result = inherited_inventory(*args, **kwargs)
        for name in (REFERENCE_RUNNER_PATH, REFERENCE_HTML_PATH):
            result[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        return dict(sorted(result.items()))

    module.runtime_inventory = inventory
    return module


def _uncapped(*, stop_after_steps=None, smoke=False, auto_approve=False, **_kwargs):
    if stop_after_steps is not None or smoke:
        raise ReviewError("agent speedup v3 evaluates real uncapped tasks; smoke and episode caps are disabled")
    if auto_approve:
        raise ReviewError("agent speedup v3 requires manual initial-plan approval")


def runtime_inventory(*args, approval_preparation=None, **kwargs):
    _uncapped(**kwargs)
    return _runtime(approval_preparation).runtime_inventory(*args, **kwargs)


def execution_options(*args, approval_preparation=None, **kwargs):
    _uncapped(**kwargs)
    return _runtime(approval_preparation).execution_options(*args, **kwargs)


def prepare_review(review_dir, *, delegated_review_note=None, **kwargs):
    _uncapped(**kwargs)
    delegation = (delegated_review_preparation(delegated_review_note)
                  if delegated_review_note is not None else None)
    module = _runtime(delegation)
    if delegation is not None:
        class DelegatedReviewStore(ReviewStore):
            @classmethod
            def create(cls, path, **options):
                preparation = dict(options.get("preparation") or {})
                preparation.update(delegation)
                options["preparation"] = preparation
                return super().create(path, **options)
        module.ReviewStore = DelegatedReviewStore
    # Preparation always returns a draft. The authorized reviewer must inspect
    # initial images and plan, then explicitly call ReviewStore.approve.
    return module.prepare_review(review_dir, **kwargs)


def _publish_approval_provenance(output_dir, summary, approval):
    """Correct every normalized artifact while keeping raw evaluator records intact."""
    output_dir = Path(output_dir)
    history = summary.get("history_stats") or {}
    recovery = {key: history.get(key, {} if key.endswith("codes") else 0) for key in (
        "transient_retry_requests", "transient_retry_wait_s", "transient_retry_codes",
        "overload_retry_requests", "overload_retry_wait_s")}
    recovery["total_backoff_wait_s"] = sum(float(recovery.get(key) or 0) for key in (
        "transient_retry_wait_s", "overload_retry_wait_s"))
    recovery["timing_note"] = "Backoff and failed attempts remain in wall/control time; a successful model pipeline includes its recovery time."
    summary["subscription_recovery"] = recovery
    summary.setdefault("model_calls", {}).update({key: recovery[key] for key in (
        "transient_retry_requests", "transient_retry_wait_s", "transient_retry_codes")})
    if (summary.get("error") or {}).get("type") == "EvidenceAcquisitionFailed":
        summary.update(status="evidence_acquisition_failed", failure_reason="evidence_acquisition_failed",
                       task_success=False, v2_success=False, v3_success=False,
                       benchmark_completed=False, evaluation_completed=False,
                       completion_time_eligible=False, success_rate=None,
                       scored_episodes=0, incomplete_episodes=1, diagnostic_complete=False)
    meta = summary["user_check"]
    meta.update(approval_mode=approval["approval_mode"], manual_review=approval["manual_review"],
                approved_by=approval["reviewer"], approval_provenance=approval)
    summary.update(human_reviewed_plan=approval["manual_review"],
                   approved_by=approval["reviewer"], approval_provenance=approval)
    if not approval["manual_review"]:
        summary["interpretation"] = (
            "One episode with an initial plan reviewed and approved by Codex under explicit user delegation, "
            "continuous model decisions and monitored execution fragments. The user did not manually review "
            "this draft; delegation did not automatically approve it. Independent LIBERO owns success.")
    for name in ("summary.json", "runner_summary.json"):
        base.write_json(output_dir / name, summary)
    (output_dir / "episodes.jsonl").write_text(json.dumps(summary) + "\n", encoding="utf-8")
    for name in ("run_state.json", "task_evaluation.json", "agent_stats.json"):
        path = output_dir / name
        if path.exists():
            data = json.loads(path.read_text())
            if summary.get("terminal_policy_act_accounting"):
                data["terminal_policy_act_accounting"] = summary["terminal_policy_act_accounting"]
                data["timing_components"] = summary["timing_components"]
                if name == "run_state.json":
                    data["model_calls"] = sum(summary["model_calls"]["runner_calls_by_kind"].values())
                elif isinstance(data.get("model_calls"), dict):
                    data["model_calls"] = summary["model_calls"]
            data.update(user_check=meta, human_reviewed_plan=approval["manual_review"],
                        approved_by=approval["reviewer"], approval_provenance=approval,
                        subscription_recovery=recovery)
            if summary.get("status") == "evidence_acquisition_failed":
                for key in ("status", "failure_reason", "task_success", "v3_success",
                            "evaluation_completed", "completion_time_eligible", "success_rate",
                            "scored_episodes", "incomplete_episodes", "error"):
                    data[key] = summary[key]
            base.write_json(path, data)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        identity = manifest["configuration"]
        identity.update(user_check=meta, human_reviewed_plan=approval["manual_review"],
                        approved_by=approval["reviewer"], approval_provenance=approval)
        identity["active_agent"].update(approval_mode=approval["approval_mode"],
                                      human_reviewed_plan=approval["manual_review"],
                                      approved_by=approval["reviewer"])
        manifest["identity_hash"] = base.digest(identity)
        base.write_json(manifest_path, manifest)
    return summary


def _freeze_reviewed_runtime(output, runtime, sources):
    """Preserve the bytes captured before execution, including the v3 entry point."""
    output = Path(output)
    if not output.is_dir():
        return
    frozen = output / "frozen_runtime"
    frozen.mkdir(exist_ok=True)
    for name, blob in sources.items():
        path = frozen / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    changed = [name for name, blob in sources.items()
               if not (ROOT / name).is_file() or (ROOT / name).read_bytes() != blob]
    base.write_json(output / "v3_runtime_integrity.json", {
        "version": VERSION, "method": METHOD, "runtime_sha256": runtime,
        "frozen_runtime": "frozen_runtime", "changed_since_start": changed,
        "unchanged": not changed, "checked_at": datetime.now(timezone.utc).isoformat(),
    })


def execute_reviewed(output, **kwargs):
    execution_started = time.monotonic()
    _uncapped(**kwargs)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Every v3 test requires a fresh output directory")
    # The store validates v3 identity, approval provenance, state/context hashes and
    # reset integrity before any credential/model/simulator activity.
    review = ReviewStore(kwargs["review_dir"]).load()
    if review["status"] != "approved":
        raise ReviewError("A fresh approved v3 plan is required before execution")
    approval = approval_metadata(review.get("preparation"))
    module = _runtime(review.get("preparation"))
    runtime = review["runtime_sha256"]
    sources = {name: (ROOT / name).read_bytes() for name in runtime if not name.startswith("@")}
    if any(hashlib.sha256(blob).hexdigest() != runtime[name] for name, blob in sources.items()):
        raise ReviewError("Runtime code changed; prepare and review the v3 plan again")
    inherited_normalize = module._normalize_artifacts

    def normalize(*args, **options):
        summary = inherited_normalize(*args, **options)
        output_dir = Path(args[0])
        summary["user_check"]["preparation_model_history"] = review.get("preparation", {}).get("model_history", {})
        summary["v3_success"] = bool(summary.get("task_success") and summary.get("libero_success") is True)
        summary["v3_success_rule"] = "independent raw LIBERO success; model belief and geometry checks are not success"
        summary["shared_geometry"] = module.framework_settings()["shared_geometry"]
        summary["performance_settings"] = performance_settings()
        summary["performance_report"] = "performance.json"
        summary = _restore_terminal_act_accounting(output_dir, summary)
        return _publish_approval_provenance(output_dir, summary, approval)

    module._normalize_artifacts = normalize
    try:
        summary = module.execute_reviewed(output, **kwargs)
    finally:
        _freeze_reviewed_runtime(output, runtime, sources)
    # All large evaluator artifacts, normalization, video close and runtime
    # copies are complete before measuring the target. Only this small report
    # remains; no background encoder work can escape the measured wall time.
    stats_path, video_path = output / "agent_stats.json", output / "episode.timing.json"
    policy_timings = (json.loads(stats_path.read_text()).get("performance_timings", {})
                      if stats_path.exists() else {})
    video_timings = json.loads(video_path.read_text()) if video_path.exists() else None
    gate = module.base._v3_live_snapshot_gate
    report = {
        "settings": performance_settings(),
        "components": summary.get("timing_components", {}),
        "policy_subcomponents": policy_timings,
        "recording": module.base._v3_recording_timings.snapshot(),
        "video": video_timings,
        "live_snapshots": {"published": gate.published, "skipped": gate.skipped},
        "accounting_note": "Policy and recording subcomponents are nested diagnostics, not additional wall time. "
                           "Video drain is included in reviewed execution elapsed time. Full action/contact/model audit is retained.",
        "completion_target": completion_target(summary, time.monotonic()-execution_started),
    }
    base.write_json(output / "performance.json", report)
    summary["performance"] = report
    return summary


def _start_command(task_key, episode_index, *, video=True):
    words = [sys.executable, RUNNER_PATH, "start", "--task", task_key,
             "--episode-index", str(episode_index),
             "--review-dir", f"runs/agent_speedup_v3_{task_key}_review_001",
             "--output", f"runs/agent_speedup_v3_{task_key}_001"]
    if not video:
        words.append("--no-video")
    return f"cd {shlex.quote(str(ROOT))} && PYTHONPATH=src " + shlex.join(words)


def preflight(config, sim_config, *, task_key, episode_index, monitor_options=None, video=True):
    from gpt_field.agent_speedup_v3_client import v3_agent_schema
    result = _runtime().preflight(
        config, sim_config, task_key=task_key, episode_index=episode_index,
        monitor_options=monitor_options, video=video)
    result.update(
        active_response_schema=v3_agent_schema(),
        live_command=_start_command(task_key, episode_index, video=video),
        evaluation_tasks=list(EVALUATION_TASKS), manual_review_required=True,
        automatic_approval_supported=False,
        payload_note="The v3 executive receives persistent shared geometry and chooses a single action "
                     "plus explicit scene updates. Local swept-geometry checks adapt execution length. "
                     "Initial task plans require manual approval; independent LIBERO owns task success.",
    )
    return result


def _show_review(store, *, host="127.0.0.1", port=8765, until_approved=False):
    # Reuse all original CSRF, origin, token, selection and per-step checks.
    with ReviewServer(store, host=host, port=port) as server:
        print(json.dumps({
            "event": "user_review_required", "method": METHOD, "review_url": server.url,
            "review_dir": str(store.path), "task": store.load()["task_prompt"],
            "note": "Review every step and approve in the browser. This approval is single-use for "
                    "this v3 runtime and task reset. Use SSH forwarding for remote access.",
        }), flush=True)
        pause = threading.Event()
        while not pause.wait(0.25):
            if until_approved and store.load()["status"] == "approved":
                return


@contextmanager
def _exclusive_execution(path):
    """Serialize real episodes after approval; waiting never consumes approval."""
    if path is None:
        yield
        return
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        print(json.dumps({"event": "execution_waiting_for_lock", "method": METHOD,
                          "lock": str(path)}), flush=True)
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            print(json.dumps({"event": "execution_lock_acquired", "method": METHOD,
                              "lock": str(path), "pid": os.getpid()}), flush=True)
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "start", "preflight"):
        command = sub.add_parser(name)
        if name != "preflight":
            command.add_argument("--review-dir", type=Path, required=True)
        if name in ("run", "start"):
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--execution-lock", type=Path,
                                 help="Shared advisory lock for serial task execution after manual approval")
        command.add_argument("--task", choices=(*base.V2_TASK_KEYS, *base.V2_TASK_ALIASES))
        command.add_argument("--episode-index", type=int, choices=range(50))
        command.add_argument("--sim-python")
        command.add_argument("--libero-root")
        command.add_argument("--expected-initial-observation-sha256")
        command.add_argument("--no-video", action="store_true", default=None)
        if name == "prepare":
            command.add_argument("--delegated-review-note",
                                 help="Explicit user authorization for Codex to review this one draft; does not approve it")
        if name != "preflight":
            command.add_argument("--store-dir", type=Path)
            command.add_argument("--profile")
        if name in ("start", "run"):
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=8765)
        if name == "run":
            command.add_argument("--wait-for-approval", action="store_true",
                                 help="Serve an existing draft and wait for manual approval before running")
    review_parser = sub.add_parser("review")
    review_parser.add_argument("--review-dir", type=Path, required=True)
    review_parser.add_argument("--host", default="127.0.0.1")
    review_parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == "review":
            store = ReviewStore(args.review_dir)
            store.load()
            _show_review(store, host=args.host, port=args.port)
            return 0
        previous = ReviewStore(args.review_dir).load() if args.command == "run" else None
        if args.command == "run" and previous["status"] != "approved":
            if previous["status"] != "draft" or not args.wait_for_approval:
                raise ReviewError("A fresh approved v3 plan is required before execution")
        if args.command in ("start", "run") and args.output.exists():
            raise FileExistsError("Every v3 test requires a fresh output directory")
        if args.command in ("start", "run") and (
                args.host not in ("127.0.0.1", "localhost") or not 0 <= args.port <= 65535):
            raise ReviewError("Review UI requires a loopback host and a port from 0 to 65535")
        saved = ((previous or {}).get("preparation") or {}).get("execution") or {}
        task_key = args.task or (previous or {}).get("task_key") or base.PRIMARY_TASK
        episode_index = (args.episode_index if args.episode_index is not None
                         else (previous or {}).get("episode_index", 0))
        sim_config, monitor = base.build_sim_config(
            task_key, sim_python=args.sim_python or saved.get("simulator", {}).get("python_executable")
            or base.DEFAULT_SIM_PYTHON,
            libero_root=args.libero_root or saved.get("simulator", {}).get("libero_root")
            or base.DEFAULT_LIBERO_ROOT)
        config = base.configuration()
        video = not args.no_video if args.no_video is not None else saved.get("video", True)
        expected = (args.expected_initial_observation_sha256
                    or (previous or {}).get("initial_observation_sha256"))
        if args.command == "preflight":
            result = preflight(config, sim_config, task_key=task_key, episode_index=episode_index,
                               monitor_options=monitor, video=video)
            print(json.dumps(result, indent=2), flush=True)
            return 0 if result["lint_check"]["passed"] else 2
        if args.command == "run" and args.wait_for_approval and previous["status"] == "draft":
            _show_review(ReviewStore(args.review_dir), host=args.host, port=args.port, until_approved=True)
        _reference._live_gate(config, sim_config, task_key, episode_index, expected)
        from gpt_field.chatgpt_auth import SubscriptionCredentials
        token_provider = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile).access_token
        if args.command in ("prepare", "start"):
            store = prepare_review(
                args.review_dir, config=config, sim_config=sim_config, task_key=task_key,
                episode_index=episode_index, monitor_options=monitor, token_provider=token_provider,
                expected_initial_observation_sha256=expected, video=video,
                delegated_review_note=getattr(args, "delegated_review_note", None))
            if args.command == "prepare":
                print(json.dumps({"status": "review_required", "method": METHOD,
                                  "review_dir": str(store.path), "validation": store.load()["validation"]}), flush=True)
                return 0
            _show_review(store, host=args.host, port=args.port, until_approved=True)
        with _exclusive_execution(args.execution_lock):
            result = execute_reviewed(
                args.output, review_dir=args.review_dir, config=config, sim_config=sim_config,
                task_key=task_key, episode_index=episode_index, monitor_options=monitor,
                token_provider=token_provider, video=video)
        print(json.dumps({"event": "agent_speedup_v3_finished", "method": METHOD,
                          "status": result["status"], "v3_success": result["v3_success"],
                          "output": result["output"], "review_id": result["user_check"]["review_id"]}), flush=True)
        return 0 if result["evaluation_completed"] else 3
    except KeyboardInterrupt:
        print("Stopped before test execution or while serving review.", file=sys.stderr)
        return 130
    except (ValueError, FileExistsError, base.MissingCredentials) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
