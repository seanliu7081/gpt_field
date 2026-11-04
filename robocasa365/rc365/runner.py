"""One RoboCasa365 RestockPantry pilot with UC7.1 transit50 and native mobile control.

prepare makes a model draft without policy actions; approve records an explicit
review of every step; run consumes that review once. No LIBERO registry, hidden
task feedback, automatic episode retry, or API-key billing is used.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "robocasa365"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_agent_speedup_v2_user_check as reviewed
from gpt_field.agent_speedup_v2_user_check import (
    ReviewError, ReviewStore, canonical_sha256, initial_observation_sha256,
    validate_review_plan,
)
from gpt_field.agent_speedup_v2_user_check_agent import AgentFirstV2Policy
from gpt_field.agent_speedup_v2_user_check_agent_client import AgentFirstV2Client
from gpt_field.agent_speedup_v2_user_check_client import PlanLock, UserCheckV2Client
from gpt_field.agent_speedup_v2_user_check_spatial import DIRECTION_SEMANTICS, build_view_frame
from gpt_field.agent_speedup_v2_types import jsonable
from gpt_field.contact_skills import StepView
from gpt_field.experiment import write_json
from .sim import RoboCasa365Config as RoboCasaConfig, RoboCasa365SimProcess as RoboCasaSimProcess
from .policy import (RoboCasa365PlanClient as RoboCasaPlanClient,
    RoboCasa365Client as RoboCasaAgentClient, RoboCasa365Policy as RoboCasaUserCheckPolicy, MACRO_REASON)

METHOD = "speedup_v2_uc71_transit50_robocasa365_mobile"
TASK_KEY = "robocasa365_RestockPantry"
TASK_PROMPT = "Pick the cans from the counter and place them in their designated side in the cabinet."
RUNNER = "robocasa365/rc365/runner.py"
OFFICIAL_HORIZON = 2550


def asset_tree_sha256(directory):
    """Bind installed assets by content without copying 11GB into each trial."""
    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(directory)).encode() + b"\0")
        with path.open("rb") as stream:
            digest.update(hashlib.file_digest(stream, "sha256").digest())
    return digest.hexdigest()


def sensor_record(obs):
    return {**obs.to_dict(), "geometry": obs.geometry,
            "joint_position": list(obs.joint_position),
            "joint_velocity": list(obs.joint_velocity), "mobile_state": obs.mobile_state}


def runtime_inventory(config, sim_config):
    """Bind code and settings, including the simulator checkout, to the review."""
    paths = set((ROOT / "src/gpt_field").rglob("*.py"))
    paths.update((PROJECT / "rc365").rglob("*.py"))
    paths.update([PROJECT / "run.py", PROJECT / "configs/restock_pantry.json"])
    paths.update(PROJECT / "setup" / name for name in (
        "assets_manifest.json", "requirements-runtime.lock", "provisioning_report.json"))
    paths.update(ROOT / name for name in (
        RUNNER, reviewed.RUNNER_PATH, reviewed.HTML_PATH,
        reviewed.base.RUNNER_PATH, reviewed.base.V1_RUNNER_PATH,
        *reviewed.base.V1_HELPER_SCRIPTS, *reviewed.base.HELPER_CHAIN,
    ))
    # Hash actual simulator source files, not just git HEAD (checkouts can be dirty).
    for directory in (Path(sim_config.robocasa_root) / "robocasa",
                      Path(sim_config.robosuite_root) / "robosuite"):
        paths.update(directory.rglob("*.py"))
    suite = Path(sim_config.robosuite_root) / "robosuite"
    paths.update((suite / "controllers/config").rglob("*.json"))
    paths.update((suite / "models/assets").rglob("*.xml"))
    casa = Path(sim_config.robocasa_root) / "robocasa"
    paths.update(casa.rglob("*.yaml"))
    result = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(paths)}
    result["@configuration"] = canonical_sha256({
        "api": asdict(config), "simulator": asdict(sim_config),
        "method": METHOD, "policy": "RoboCasa365Policy", "transit_action_m": 0.05,
        "mobile": True, "evaluation_profile": "uncapped_exploratory_pilot",
        "official_reference_horizon": OFFICIAL_HORIZON,
        "native_macros": False, "reasoning_effort": "high",
        "direction_semantics": DIRECTION_SEMANTICS,
        "python_executable": sys.executable,
    })
    result["@robocasa_assets_tree"] = asset_tree_sha256(casa / "models/assets")
    result["@robosuite_assets_tree"] = asset_tree_sha256(suite / "models/assets")
    return result


def freeze_sources(directory, runtime):
    root = Path(directory) / "frozen_source"
    for name in runtime:
        if name.startswith("@"):
            continue
        path = Path(name)
        target = root / path.relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())


def validate_budget(client):
    budget = client.budget_snapshot()
    if budget.get("billing") != "chatgpt_subscription" or budget.get("api_key_calls") != 0:
        raise ValueError("Only the ChatGPT subscription route with zero API-key calls is permitted")
    return budget


def terminal_budget(client):
    """Keep cleanup and terminal evidence possible even when budget checks fail."""
    if client is None:
        return None
    try:
        value = client.budget_snapshot()
        if value.get("billing") != "chatgpt_subscription" or value.get("api_key_calls") != 0:
            return {**value, "validation_error": "subscription_only_budget_invariant"}
        return value
    except Exception as exc:
        return {"validation_error": type(exc).__name__}


def new_sim(sim_config, factory):
    with reviewed.base._simulator_without_credentials():
        sim = factory(sim_config)
        try:
            sim.start()
        except BaseException:
            try:
                sim.close()
            finally:
                raise
    return sim


def prepare(directory, *, config, sim_config, token_provider,
            sim_factory=RoboCasaSimProcess, client_factory=RoboCasaPlanClient,
            auto_approve=False):
    from gpt_field.agent_speedup_v2_client import plan_images
    from gpt_field.agent_speedup_v2_geometry import frame_alignment_turns
    from gpt_field.agent_speedup_v2_types import FINGERTIP_EDGES
    from PIL import Image
    import numpy as np

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    runtime = runtime_inventory(config, sim_config)
    freeze_sources(directory, runtime)
    sim = None
    started = time.monotonic()
    try:
        sim = new_sim(sim_config, sim_factory)
        obs = sim.reset()
        if obs.prompt != TASK_PROMPT or sim.status.get("success"):
            raise ValueError("Expected the official, initially incomplete RestockPantry task")
        write_json(directory / "initial_sensors.json", sensor_record(obs))
        write_json(directory / "simulator.json", {"metadata": sim.metadata, "status": sim.status})
        reset = sim.snapshot(include_xml=True)
        write_json(directory / "initial_physics.json", reset)
        view = StepView(obs)
        Image.fromarray(np.asarray(view.external_rgb_upright)).save(directory / "external.png")
        Image.fromarray(np.asarray(view.wrist_rgb_upright)).save(directory / "wrist.png")
        turns = int(frame_alignment_turns(view.R, None))
        client = client_factory(config, token_provider=token_provider,
                                task_prompt=obs.prompt, log_dir=directory / "requests")
        client.check_observation(obs)
        model_started = time.monotonic()
        draft = client.request_plan(view.external_rgb_upright, view.wrist_rgb_upright,
                                    quarter_turns=turns, fingertip_edge=FINGERTIP_EDGES[turns])
        model_s = time.monotonic() - model_started
        validate_review_plan(draft)
        preparation = {
            "kind": "robocasa_model_draft_before_policy_actions", "policy_actions": 0,
            "approval_mode": "user_authorized_automatic" if auto_approve else "manual_browser_or_chat_review",
            "manual_review": not auto_approve,
            "execution": {"api": asdict(config), "simulator": asdict(sim_config),
                          "direction_semantics": DIRECTION_SEMANTICS},
            "view_frame": build_view_frame(getattr(obs, "geometry", None)),
            "initial_sensor_sha256": canonical_sha256(sensor_record(obs)),
            "initial_physics_sha256": reset["physical_sha256"],
            "model_xml_sha256": reset["model_xml_sha256"],
            "draft_model_s": model_s, "elapsed_s": time.monotonic() - started,
            "model_budget": validate_budget(client), "model_history": client.history_stats(),
            "quarter_turns": turns, "fingertip_edge": FINGERTIP_EDGES[turns],
            "review_images": [{"label": label, "data_url": url} for label, url in
                              plan_images(view.external_rgb_upright, view.wrist_rgb_upright, turns)],
            "native_macros": False, "native_macro_reason": MACRO_REASON,
            "stopping_rule": "official RoboCasa success, user stop, or infrastructure/subscription error",
            "automatic_episode_caps": None,
        }
        if runtime != runtime_inventory(config, sim_config):
            raise ReviewError("Runtime changed during preparation; prepare a fresh review")
        store = ReviewStore.create(directory, task_key=TASK_KEY, episode_index=0,
            task_prompt=obs.prompt, initial_observation=obs.to_dict(), plan=draft,
            runtime_sha256=runtime, preparation=preparation)
        write_json(directory / "preparation_summary.json", {
            "status": "review_required", "plan": draft,
            "validation": store.load()["validation"], "model_budget": preparation["model_budget"],
            "policy_actions": 0,
            **reviewed.direction_review_output(store.load()),
        })
        return store.load()
    finally:
        if sim is not None:
            sim.close()


def execute(output, *, review_dir, config, sim_config, token_provider,
            sim_factory=RoboCasaSimProcess, client_factory=RoboCasaAgentClient,
            policy_factory=RoboCasaUserCheckPolicy, video=True):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Every trial requires a fresh output directory")
    store = ReviewStore(review_dir)
    review = store.load()
    runtime = runtime_inventory(config, sim_config)
    review = store.claim(output=output, task_key=TASK_KEY, episode_index=0,
        initial_observation_sha256=review["initial_observation_sha256"], runtime_sha256=runtime)
    approval_mode = review.get("preparation", {}).get("approval_mode", "manual_browser_or_chat_review")
    human_reviewed = approval_mode != "user_authorized_automatic"
    output.mkdir(parents=True, exist_ok=False)
    freeze_sources(output, runtime)
    write_json(output / "user_check_review.json", review)
    write_json(output / "manifest.json", {
        "method": METHOD, "task": "RestockPantry", "suite": "RoboCasa365-1.0.1-mobile-pilot",
        "api": asdict(config), "simulator": asdict(sim_config), "runtime_sha256": runtime,
        "billing": "chatgpt_subscription", "reasoning_effort": "high", "api_key_fallback": False,
        "automatic_episode_caps": None, "automatic_episode_retries": 0,
        "evaluation_profile": "uncapped_exploratory_pilot",
        "official_reference_horizon": OFFICIAL_HORIZON, "official_benchmark_score": False,
        "transit_action_m": 0.05, "transit_target_limit_m": 0.50, "mobile_control": True,
        "native_macros": False, "native_macro_reason": MACRO_REASON,
        "evaluator": "RestockPantry._check_success", "contacts": "record_only",
        "policy_inputs": "two RGB, state8, arm7 joints, measured base/torso state, camera and fixed robot calibration",
        "hidden_scene_truth_to_policy": False, "human_reviewed_plan": human_reviewed,
        "approval_mode": approval_mode,
        "review_id": review["id"], "video": video,
        "version": reviewed.VERSION, "direction_semantics": DIRECTION_SEMANTICS,
        **reviewed.direction_review_output(review),
    })
    lock = PlanLock(review["plan"], task_prompt=TASK_PROMPT,
                    view_frame=(review.get("preparation") or {}).get("view_frame"))
    write_json(output / "direction_review.json", reviewed.direction_review_output(review))
    sim = client = policy = recorder = None
    obs = None
    steps = feedback_count = 0
    status, error = "infrastructure_error", None
    pending_action = None
    started = time.monotonic()
    control_started = control_stopped = None
    timings = {"decision_s": 0., "simulator_step_s": 0., "observe_transition_s": 0., "video_render_s": 0.}
    raw_success = False
    cleanup_errors = []

    def state(stage):
        write_json(output / "run_state.json", {
            "method": METHOD, "stage": stage, "status": status, "steps": steps,
            "feedback_count": feedback_count, "pending_action": pending_action,
            "elapsed_s": time.monotonic() - started, "robocasa_success": raw_success,
            "budget": terminal_budget(client),
        })

    with (output / "trace.jsonl").open("w") as trace, reviewed.base.termination_as_interrupt():
        def append(row):
            trace.write(json.dumps(jsonable(row), allow_nan=False) + "\n")
            trace.flush()
        append({"event": "direction_review", **reviewed.direction_review_output(review)})
        try:
            state("starting")
            sim = new_sim(sim_config, sim_factory)
            obs = sim.reset()
            if obs.prompt != TASK_PROMPT or initial_observation_sha256(obs.to_dict()) != review["initial_observation_sha256"]:
                raise ReviewError("Reset differs from the reviewed observation; no execution model call sent")
            if canonical_sha256(sensor_record(obs)) != review["preparation"]["initial_sensor_sha256"]:
                raise ReviewError("Camera calibration or robot sensors differ from the reviewed reset")
            reset = sim.snapshot(include_xml=True)
            write_json(output / "initial_physics.json", reset)
            if (reset["physical_sha256"] != review["preparation"]["initial_physics_sha256"]
                    or reset["model_xml_sha256"] != review["preparation"]["model_xml_sha256"]):
                raise ReviewError("Simulator physics/model differs from the reviewed reset")
            if sim.status.get("success"):
                raise ReviewError("Task already successful at reset; refusing a scored trial")
            write_json(output / "initial_sensors.json", sensor_record(obs))
            write_json(output / "simulator.json", {"metadata": sim.metadata, "status": sim.status})
            client = client_factory(config, token_provider=token_provider, task_prompt=obs.prompt,
                log_dir=output / "requests", approved_plan=review["plan"], lock=lock)
            policy = policy_factory(client, lock=lock)
            policy.reset()
            if video:
                from gpt_field.agent_speedup_v2_video import AgentSpeedUpV2VideoRecorder
                recorder = AgentSpeedUpV2VideoRecorder(output / "episode.mp4",
                    episode_id="ROBOCASA365 RESTOCKPANTRY / UC7.1 TRANSIT50 / MOBILE")
                recorder.observe(obs, step=0)
            append({"event": "reset", "observation": sensor_record(obs), "evaluator": sim.status})
            status = "running"
            control_started = time.monotonic()
            while True:
                if (output / reviewed.base.OPERATOR_STOP_FILE).exists():
                    status = "user_stop"
                    break
                if obs.prompt != TASK_PROMPT:
                    raise ValueError("Task prompt changed during execution")
                state("deciding")
                began = time.monotonic()
                decision = policy.act(obs)
                latency = time.monotonic() - began
                timings["decision_s"] += latency
                validate_budget(client)
                append({"event": "action", "step": steps, "action": decision.action,
                        "diagnostics": decision.diagnostics, "decision_latency_s": latency})
                if recorder is not None:
                    began = time.monotonic()
                    recorder.decision(obs, decision, decision_latency_s=latency)
                    timings["video_render_s"] += time.monotonic() - began
                before = obs
                pending_action = steps
                state("executing")
                began = time.monotonic()
                mobile = decision.diagnostics.get("mobile_command")
                obs, success = (sim.step(decision.action) if mobile is None else
                                sim.mobile_step(decision.action, mobile))
                timings["simulator_step_s"] += time.monotonic() - began
                steps += 1
                pending_action = None
                raw_success = bool(success)
                if type(success) is not bool or success != sim.status.get("robocasa_success"):
                    raise ValueError("Inconsistent RoboCasa evaluator status")
                append({"event": "executed", "step": steps - 1,
                        "observation": sensor_record(obs), "evaluator": sim.status})
                began = time.monotonic()
                feedback = policy.observe_transition(before, decision, obs)
                timings["observe_transition_s"] += time.monotonic() - began
                feedback_count += 1
                append({"event": "feedback", "step": steps - 1, "feedback": feedback})
                if recorder is not None:
                    began = time.monotonic()
                    recorder.executed(obs, step=steps - 1, success=success,
                        sim_status=sim.status, feedback=feedback, history_stats=client.history_stats())
                    timings["video_render_s"] += time.monotonic() - began
                write_json(output / "user_check_framework.json", lock.framework_snapshot())
                state("observed")
                if steps % 10 == 0 or success:
                    print(json.dumps({"event": "progress", "steps": steps, "robocasa_success": success,
                        "api_calls": validate_budget(client).get("api_calls"),
                        "control_s": time.monotonic() - control_started}), flush=True)
                if success:
                    status = "success"
                    break
                if sim.status.get("terminated"):
                    raise RuntimeError("Simulator terminated before official task success")
        except KeyboardInterrupt:
            status = "user_stop"
        except Exception as exc:
            status = "subscription_error" if isinstance(exc, (reviewed.base.APIError, reviewed.base.MissingCredentials)) else "infrastructure_error"
            error = {"type": type(exc).__name__, "message": reviewed.base.redact_secrets(str(exc))}
            append({"event": "error", "step": steps, "error": error})
        finally:
            if control_started is not None:
                control_stopped = time.monotonic()
            if sim is not None and obs is not None:
                try:
                    write_json(output / "final_physics.json", sim.snapshot())
                except Exception as exc:
                    cleanup_errors.append({"component": "final_physics", "type": type(exc).__name__})
            for name, owner in (("simulator", sim), ("video", recorder)):
                if owner is not None:
                    try:
                        owner.close(**({"status": status, "error": json.dumps(error)} if name == "video" else {}))
                    except Exception as exc:
                        cleanup_errors.append({"component": name, "type": type(exc).__name__})
            # Export images after closing the simulator, so export failures
            # cannot leak a worker process or bypass video finalization.
            if obs is not None:
                try:
                    write_json(output / "final_sensors.json", sensor_record(obs))
                    from PIL import Image
                    view = StepView(obs)
                    Image.fromarray(view.external_rgb_upright).save(output / "final_external.png")
                    Image.fromarray(view.wrist_rgb_upright).save(output / "final_wrist.png")
                except Exception as exc:
                    cleanup_errors.append({"component": "final_sensors", "type": type(exc).__name__})
            try:
                unchanged = runtime == runtime_inventory(config, sim_config)
            except Exception as exc:
                unchanged = False
                cleanup_errors.append({"component": "terminal_inventory", "type": type(exc).__name__})
            elapsed = None if control_started is None else control_stopped - control_started
            budget = terminal_budget(client)
            completed = (status == "success" and raw_success and unchanged
                         and steps == feedback_count and pending_action is None
                         and budget is not None and "validation_error" not in budget)
            summary = {
                "method": METHOD, "task": "RestockPantry", "status": status, "error": error,
                "version": reviewed.VERSION, "direction_semantics": DIRECTION_SEMANTICS,
                "steps": steps, "feedback_count": feedback_count, "pending_action": pending_action,
                "robocasa_success": raw_success, "task_success": completed,
                "evaluation_profile": "uncapped_exploratory_pilot",
                "official_reference_horizon": OFFICIAL_HORIZON,
                "success_within_reference_horizon": bool(completed and steps <= OFFICIAL_HORIZON),
                "official_benchmark_score": False,
                "evaluation_completed": completed, "completion_time_eligible": completed,
                "control_elapsed_s": elapsed, "timing_components": timings,
                "control_without_video_s": None if elapsed is None else elapsed - timings["video_render_s"],
                "preparation_model_s": review["preparation"].get("draft_model_s"),
                "execution_budget": budget,
                "preparation_budget": review["preparation"].get("model_budget"),
                "billing": budget.get("billing") if budget else "chatgpt_subscription",
                "api_key_calls": budget.get("api_key_calls") if budget else 0,
                "runtime_unchanged": unchanged, "cleanup_errors": cleanup_errors,
                "video": "episode.mp4" if recorder is not None else None,
                "evaluator_terminal": dict(sim.status) if sim is not None else None,
                "approval_mode": approval_mode, "human_reviewed_plan": human_reviewed,
                "interpretation": "One authorized mobile RoboCasa365 exploratory pilot, not an official benchmark score; incomplete times are not completion latency.",
            }
            write_json(output / "summary.json", summary)
            write_json(output / "user_check_plan_history.json", lock.snapshot())
            if policy is not None:
                write_json(output / "agent_stats.json", jsonable(policy.agent_stats()))
            state("terminal")
            append({"event": "terminal", "summary": summary})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "prepare", "review", "approve", "run"))
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--reviewed-steps", type=int, nargs="+")
    parser.add_argument("--revision", type=int)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--test-approved", action="store_true",
                        help="Use only after the user authorizes this one test; never supplied during setup")
    args = parser.parse_args(argv)
    config = reviewed.base.configuration()
    sim_config = RoboCasaConfig(seed=args.seed)
    if args.command == "preflight":
        inventory = runtime_inventory(config, sim_config)
        report = {"method": METHOD, "simulator": asdict(sim_config), "api": asdict(config),
                  "reasoning_effort": "high", "billing": "chatgpt_subscription",
                  "api_key_fallback": False, "native_macros": False,
                  "transit_action_m": .05, "transit_target_limit_m": .50,
                  "code_files": sum(not key.startswith("@") for key in inventory),
                  "runtime_sha256": inventory, "network_calls": 0,
                  "real_environment_creations": 0, "model_calls": 0,
                  "test_approval_status": json.loads(
                      (PROJECT / "configs/restock_pantry.json").read_text())["test_approval_status"]}
        if args.output is not None:
            write_json(args.output, report)
            print(json.dumps({key: val for key, val in report.items() if key != "runtime_sha256"}))
        else:
            print(json.dumps({key: val for key, val in report.items() if key != "runtime_sha256"}))
        return 0
    if args.command in ("prepare", "run") and not args.test_approved:
        parser.error("Task testing is pending user approval. --test-approved is required for prepare/run.")
    if args.review_dir is None:
        parser.error("--review-dir is required")
    if args.command == "review":
        reviewed._show_review(ReviewStore(args.review_dir), host="127.0.0.1", port=args.port,
                              until_approved=False)
        return 0
    if args.command == "approve":
        if args.revision is None or args.reviewed_steps is None:
            parser.error("--revision and --reviewed-steps are required for explicit plan review")
        result = ReviewStore(args.review_dir).approve(expected_revision=args.revision,
                                                     reviewed_steps=args.reviewed_steps)
        print(json.dumps({"status": result["status"], "approved": result["approved"],
                          **reviewed.direction_review_output(result)}))
        return 0
    from gpt_field.chatgpt_auth import SubscriptionCredentials
    token_provider = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile).access_token
    if args.command == "prepare":
        result = prepare(args.review_dir, config=config, sim_config=sim_config, token_provider=token_provider)
        print(json.dumps({"status": result["status"], "review_dir": str(args.review_dir),
                          "plan": result["plan"], "revision": result["revision"],
                          **reviewed.direction_review_output(result)}), flush=True)
        return 0
    if args.output is None:
        parser.error("--output is required for run")
    summary = execute(args.output, review_dir=args.review_dir, config=config,
                      sim_config=sim_config, token_provider=token_provider, video=not args.no_video)
    print(json.dumps(summary), flush=True)
    return 0 if summary["task_success"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
