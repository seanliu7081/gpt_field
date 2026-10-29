"""Matched full-task timing: compact tracking Hybrid versus five-action Direct.

Both use Astra medium, ChatGPT subscription and block feedback summaries. The
opt-in matched-guidance protocol supplies identical grasp/placement guidance
and requires actual LIBERO success in both arms. Legacy comparisons retain
their first-LIBERO-or-model completion rule. No API-key/model fallback exists.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import evaluate_bowl_hybrid_subscription as runner
from gpt_field.experiment import digest, source_manifest, write_json
from gpt_field.astra import _canonical
from gpt_field.libero import LiberoConfig
from gpt_field.compact_subscription import compact_operational_guidance


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matched_guidance():
    return compact_operational_guidance(centered_placement_guidance=True, grasp_settle_guidance=True)


def arm_metrics(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())["configuration"]
    first, executed = None, []
    with (directory / "trace.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if first is None and row.get("event") == "reset":
                first = row
            if row.get("event") == "executed":
                executed.append({"success": row.get("success"), "step": row.get("step")})
    requests = sorted((directory / "requests").glob("*.request.json"))
    responses = sorted((directory / "requests").glob("*.response.json"))
    request_checks, response_checks, inputs = [], [], []
    payload_bytes, operational_guidance_counts = [], []
    shared_guidance = _matched_guidance()
    simulator_record = directory / "simulator.json"
    simulator_metadata = json.loads(simulator_record.read_text()).get("metadata") if simulator_record.exists() else None
    for path in requests:
        row = json.loads(path.read_text())
        payload = row["payload"]
        request_checks.append(row.get("billing") == "chatgpt_subscription"
            and payload.get("model") == "gpt-6-astra" and payload.get("reasoning") == {"effort": "medium"})
        payload_bytes.append(len(_canonical(payload).encode()))
        operational_guidance_counts.append(payload.get("instructions", "").count(shared_guidance))
    for path in responses:
        row = json.loads(path.read_text())
        response_checks.append(row.get("status") == "completed" and row.get("billing") == "chatgpt_subscription"
            and row.get("model") == "gpt-6-astra" and row.get("reasoning", {}).get("effort") == "medium")
        inputs.append(row.get("usage", {}).get("input_tokens"))
    return {
        "directory": str(directory), "status": summary["status"],
        "error": summary.get("error"), "accepted_response_count": len(responses),
        "completed": summary["benchmark_completed"], "libero_success": summary["libero_success_observed"],
        "model_declared_complete": summary["model_declared_complete"],
        "control_elapsed_s": summary["control_elapsed_s"], "total_elapsed_s": summary["elapsed_s"],
        "startup_s": summary["startup_to_initial_observation_s"],
        "timing_components": summary["timing_components"], "steps": summary["steps"],
        "field_actions": summary["coarse_field_actions"], "direct_actions": summary["contact_direct_actions"],
        "model_calls": summary["api_calls"], "local_plans": summary["local_plan_count"],
        "billing": summary["billing"], "api_key_calls": summary["api_key_calls"],
        "input_tokens": summary["budget"]["input_tokens"], "output_tokens": summary["budget"]["output_tokens"],
        "first_request_input_tokens": inputs[0] if inputs else None,
        "last_request_input_tokens": inputs[-1] if inputs else None,
        "request_payload_bytes": payload_bytes,
        "all_requests_and_responses_match_model_effort_route": bool(requests) and
            len(requests) == len(responses) == summary["api_calls"] and all(request_checks) and all(response_checks),
        "initial_observation_sha256": digest(first["observation"]) if first else None,
        "simulator_configuration": manifest["simulator"], "stopping_rule": manifest["stopping_rule"],
        "source_sha256": manifest["software"]["source_sha256"],
        "runner_sha256": manifest["runner"]["sha256"],
        "feedback_history": manifest["policy"]["feedback_history"],
        "controller_mode": manifest["policy"]["controller_mode"],
        "execution_mode": manifest["policy"]["execution_mode"],
        "direct_action_steps": manifest["policy"]["direct_action_steps"],
        "field_update_steps": manifest["policy"]["field_update_steps"],
        "field_representation": manifest["policy"]["field_representation"],
        "centered_placement_guidance": manifest["policy"].get("centered_placement_guidance", False),
        "grasp_settle_guidance": manifest["policy"].get("grasp_settle_guidance", False),
        "stop_on_model_complete": manifest.get("stop_on_model_complete"),
        "operational_guidance_occurrences_per_request": operational_guidance_counts,
        "all_requests_have_exact_shared_guidance_once": bool(requests) and all(
            count == 1 for count in operational_guidance_counts),
        "operational_guidance_sha256": hashlib.sha256(shared_guidance.encode()).hexdigest()
            if requests and all(count == 1 for count in operational_guidance_counts) else None,
        "actual_libero_success": (summary.get("status") == "success"
            and summary.get("libero_success") is True and summary.get("evaluation_completed") is True
            and summary.get("libero_success_observed") is True and bool(executed)
            and executed[-1]["success"] is True and len(executed) == summary["steps"]),
        "task_identity": {key: manifest.get(key) for key in ("task", "task_id", "episode_index")},
        "api_configuration": manifest["api"],
        "shared_policy_configuration": {key: manifest["policy"].get(key) for key in (
            "grid", "planner", "field_execution_steps", "adapter_max_rotation_rad",
            "recent_image_observations", "initial_history_settings", "reasoning_effort",
            "success_feedback_to_policy", "reward_feedback_to_policy", "object_state_feedback_to_policy")},
        "manifest_sha256": _file_sha256(directory / "manifest.json"),
        "simulator_physical_metadata": simulator_metadata,
    }


def compare(output, *, hybrid_directory=None, direct_directory=None, interrupted_direct_directory=None,
            matched_libero_guidance=False):
    output = Path(output)
    if matched_libero_guidance and any((hybrid_directory, direct_directory, interrupted_direct_directory)):
        raise ValueError("Matched guidance comparison requires its own fresh paired run directories")
    directories = {"hybrid": Path(hybrid_directory or output / "hybrid"),
                   "direct": Path(direct_directory or output / "direct")}
    arms = {name: (arm_metrics(path) if not matched_libero_guidance or (path / "summary.json").exists()
                   else None) for name, path in directories.items()}
    if matched_libero_guidance and any(arm is None for arm in arms.values()):
        result = {"kind": "matched_guidance_libero_single_pair_timing", "arms": arms,
            "checks": {"both_arms_recorded": False}, "valid_comparison": False,
            "missing_arms": [name for name, arm in arms.items() if arm is None],
            "hybrid_seconds_saved": None, "hybrid_time_reduction_percent": None, "hybrid_speedup": None,
            "limitations": ["An arm was not executed; recorded failed/incomplete arms are retained. No completed-task ratio is claimed."]}
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "comparison.json", result)
        return result
    h, d = arms["hybrid"], arms["direct"]
    checks = {
        "both_completed": h["completed"] and d["completed"],
        "same_initial_observation": h["initial_observation_sha256"] is not None and
            h["initial_observation_sha256"] == d["initial_observation_sha256"],
        "same_simulator": h["simulator_configuration"] == d["simulator_configuration"],
        "same_stopping_rule": h["stopping_rule"] == d["stopping_rule"],
        "same_sources": h["source_sha256"] == d["source_sha256"],
        "same_runner": h["runner_sha256"] == d["runner_sha256"],
        "correct_arms": (h["controller_mode"] == "hybrid" and h["execution_mode"] == "tracking"
            and d["controller_mode"] == "direct" and d["execution_mode"] == "fixed"
            and d["field_actions"] == d["local_plans"] == 0),
        "same_controller_cadence": (h["direct_action_steps"] == d["direct_action_steps"] == 5
            and h["field_update_steps"] == d["field_update_steps"] == 30),
        "same_compact_representation": h["field_representation"] == d["field_representation"] == "analytic-goal-rbf-v1",
        "same_summary_history": h["feedback_history"] == d["feedback_history"] == "block_summary",
        "same_medium_subscription_model": all(a["all_requests_and_responses_match_model_effort_route"] for a in arms.values()),
        "zero_api_key_calls": h["api_key_calls"] == d["api_key_calls"] == 0,
    }
    if matched_libero_guidance:
        pair_path = output / "comparison_manifest.json"
        pair = json.loads(pair_path.read_text()) if pair_path.exists() else {}
        expected_guidance_hash = hashlib.sha256(_matched_guidance().encode()).hexdigest()
        checks.update({
            "both_actual_libero_success": h["actual_libero_success"] and d["actual_libero_success"],
            "both_guidance_flags_enabled": all(a["centered_placement_guidance"] is True
                and a["grasp_settle_guidance"] is True for a in arms.values()),
            "identical_operational_guidance_in_every_request": all(
                a["all_requests_have_exact_shared_guidance_once"]
                and a["operational_guidance_sha256"] == expected_guidance_hash for a in arms.values()),
            "libero_only_completion": all(a["stop_on_model_complete"] is False
                and a["stopping_rule"] == runner._stopping_rule(False) for a in arms.values()),
            "same_task_and_initial_state_index": h["task_identity"] == d["task_identity"],
            "same_api_configuration": h["api_configuration"] == d["api_configuration"],
            "same_shared_policy_configuration": h["shared_policy_configuration"] == d["shared_policy_configuration"],
            "two_image_pairs_and_no_hidden_feedback": all(
                a["shared_policy_configuration"].get("recent_image_observations") == 2
                and a["shared_policy_configuration"].get("initial_history_settings", {}).get("image_history_limit") == 2
                and all(a["shared_policy_configuration"].get(key) is False for key in (
                    "success_feedback_to_policy", "reward_feedback_to_policy", "object_state_feedback_to_policy"))
                for a in arms.values()),
            "same_simulator_physical_metadata": h["simulator_physical_metadata"] is not None
                and h["simulator_physical_metadata"] == d["simulator_physical_metadata"],
            "fresh_pair_provenance": (pair.get("matched_libero_guidance") is True
                and pair.get("order") == ["hybrid", "direct"]
                and pair.get("operational_guidance_sha256") == expected_guidance_hash
                and pair.get("source_sha256") == h["source_sha256"]
                and pair.get("runner_sha256") == h["runner_sha256"]
                and pair.get("arm_directories") == {name: str(path.resolve()) for name, path in directories.items()}
                and pair.get("arm_manifest_sha256") == {name: arm["manifest_sha256"] for name, arm in arms.items()}),
            "positive_completed_control_times": all(isinstance(a["control_elapsed_s"], (int, float))
                and 0 < a["control_elapsed_s"] < float("inf") for a in arms.values()),
        })
    valid = all(checks.values())
    result = {"kind": ("matched_guidance_libero_single_pair_timing" if matched_libero_guidance else
                       "matched_single_pair_full_task_timing"), "arms": arms, "checks": checks,
        "report_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "valid_comparison": valid,
        "hybrid_seconds_saved": d["control_elapsed_s"] - h["control_elapsed_s"] if valid else None,
        "hybrid_time_reduction_percent": 100 * (1 - h["control_elapsed_s"] / d["control_elapsed_s"]) if valid else None,
        "hybrid_speedup": d["control_elapsed_s"] / h["control_elapsed_s"] if valid else None,
        "limitations": ["One sequential pair; service latency and model behavior vary.",
            "Completion is the first LIBERO or model signal; model declaration alone is not verified LIBERO success.",
            "Control time includes inference waits, local control, simulation, logging and inline video; excludes startup and finalization.",
            "Field expansion is included in model pipeline; model pipeline and planning are included in decision time."]}
    if matched_libero_guidance:
        result["limitations"][1] = "Only actual LIBERO success completes an arm; no model-declared or historical completion is substituted."
        result["limitations"].append("Both requests contain identical operational guidance; controller schemas, routing and trajectory execution intentionally differ.")
    if h["completed"] and not d["completed"]:
        result["censored_direct_observation"] = {
            "hybrid_completed_in_s": h["control_elapsed_s"],
            "direct_observed_unfinished_for_s": d["control_elapsed_s"],
            "direct_completion_time_s": None,
            "observed_time_beyond_hybrid_completion_s": (max(0., d["control_elapsed_s"] - h["control_elapsed_s"])
                if isinstance(d["control_elapsed_s"], (int, float)) and isinstance(h["control_elapsed_s"], (int, float))
                else None),
            "interpretation": "Direct had not completed at its stop. Its observed time is not a completed-task duration; no completion speedup ratio is claimed."}
    if interrupted_direct_directory is not None:
        result["interrupted_direct_attempt"] = arm_metrics(interrupted_direct_directory)
        result["limitations"].append(
            "The first Direct attempt stopped on a subscription error. Its incomplete time is reported separately; "
            "the comparison uses a fresh Direct trial from the identical initial state, without per-request retry or fallback.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--matched-libero-guidance", action="store_true",
                        help="Fresh matched grasp/placement guidance in both arms; require actual LIBERO success")
    parser.add_argument("--hybrid-run", type=Path)
    parser.add_argument("--direct-run", type=Path)
    parser.add_argument("--interrupted-direct-run", type=Path)
    parser.add_argument("--sim-python", default=runner.DEFAULT_SIM_PYTHON)
    parser.add_argument("--libero-root", default=runner.DEFAULT_LIBERO_ROOT)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--profile")
    args = parser.parse_args(argv)
    if not args.report_only and any((args.hybrid_run, args.direct_run, args.interrupted_direct_run)):
        parser.error("Explicit recorded run paths require --report-only")
    if args.matched_libero_guidance and any((args.hybrid_run, args.direct_run, args.interrupted_direct_run)):
        parser.error("Matched guidance requires its own fresh paired runs, without substituted historical directories")
    if args.report_only:
        result = compare(args.output, hybrid_directory=args.hybrid_run, direct_directory=args.direct_run,
                         interrupted_direct_directory=args.interrupted_direct_run,
                         matched_libero_guidance=args.matched_libero_guidance)
    else:
        if args.output.exists():
            parser.error("Fresh comparison requires a new output directory")
        config = runner.configuration()
        sim_config = LiberoConfig(suite="libero_spatial", seed=7, max_steps=None,
            libero_root=str(Path(args.libero_root).expanduser().resolve()),
            python_executable=os.path.abspath(os.path.expanduser(args.sim_python)))
        completion_and_guidance = {"stop_on_model_complete": not args.matched_libero_guidance}
        if args.matched_libero_guidance:
            completion_and_guidance.update(centered_placement_guidance=True, grasp_settle_guidance=True)
        for mode in ("hybrid", "direct"):
            runner.preflight(config, sim_config, field_representation="compact", reasoning_effort="medium",
                controller_mode=mode, execution_mode="tracking" if mode == "hybrid" else "fixed",
                feedback_history="block_summary", **completion_and_guidance)
        from gpt_field.chatgpt_auth import SubscriptionCredentials
        credentials = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile)
        args.output.mkdir(parents=True)
        pair_manifest = {
            "order": ["hybrid", "direct"], "model": "gpt-6-astra", "reasoning_effort": "medium",
            "billing": "chatgpt_subscription", "simulator": asdict(sim_config),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pair_count": 1, "api_key_fallback": False}
        if args.matched_libero_guidance:
            pair_manifest.update(matched_libero_guidance=True, **completion_and_guidance,
                stopping_rule=runner._stopping_rule(False), feedback_history="block_summary",
                operational_guidance=_matched_guidance(),
                operational_guidance_sha256=hashlib.sha256(_matched_guidance().encode()).hexdigest(),
                source_sha256=source_manifest()["source_sha256"], runner_sha256=_file_sha256(runner.__file__),
                api_configuration=asdict(config), arm_manifest_sha256={},
                arm_directories={mode: str((args.output / mode).resolve()) for mode in ("hybrid", "direct")},
                process_environment={name: os.environ.get(name) for name in (
                    "MUJOCO_EGL_DEVICE_ID", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
        write_json(args.output / "comparison_manifest.json", pair_manifest)
        for mode in ("hybrid", "direct"):
            summary = runner.execute_episode(args.output / mode, config=config, sim_config=sim_config,
                token_provider=credentials.access_token, field_representation="compact", reasoning_effort="medium",
                controller_mode=mode, execution_mode="tracking" if mode == "hybrid" else "fixed",
                feedback_history="block_summary", **completion_and_guidance)
            if args.matched_libero_guidance:
                pair_manifest["arm_manifest_sha256"][mode] = _file_sha256(args.output / mode / "manifest.json")
                write_json(args.output / "comparison_manifest.json", pair_manifest)
            if not summary["benchmark_completed"]:
                if args.matched_libero_guidance:
                    compare(args.output, matched_libero_guidance=True)
                else:
                    write_json(args.output / "comparison.json", {"valid_comparison": False,
                        "incomplete_arm": mode, "status": summary["status"], "error": summary.get("error")})
                return 3
        result = compare(args.output, matched_libero_guidance=args.matched_libero_guidance)
    print(json.dumps({key: value for key, value in result.items() if key != "arms"}), flush=True)
    return 0 if result["valid_comparison"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
