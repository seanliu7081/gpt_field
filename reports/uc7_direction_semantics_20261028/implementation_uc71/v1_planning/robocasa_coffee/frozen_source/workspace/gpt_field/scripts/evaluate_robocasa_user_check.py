"""One arm-only RoboCasa task with the current speedup v2 user-check agent.

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

ROOT = Path(__file__).resolve().parents[1]
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
from gpt_field.robocasa_user_check_sim import RoboCasaConfig, RoboCasaSimProcess

METHOD = "speedup_v2_user_check_robocasa_arm_only"
TASK_KEY = "robocasa_CoffeePressButton"
TASK_PROMPT = "press the button on the coffee machine to serve coffee"
RUNNER = "scripts/evaluate_robocasa_user_check.py"
MACRO_REASON = "Native macro robot-base geometry has not been validated for RoboCasa PandaOmron."
ROBOT_CONTEXT = (
    "ROBOCASA ARM-ONLY CONFIGURATION. The Panda arm and parallel-jaw gripper operate "
    "with the mobile base and torso held at their reset configuration. No base or "
    "torso commands are available. Native macros are unavailable in this adapter; "
    "use the existing observed motion and execution fragments. Position targets "
    "refer to gripper0_right_grip_site. Use the synchronized calibration supplied "
    "with each image; the external view is robot0_agentview_left."
)
PLAN_ROBOT_CONTEXT = ROBOT_CONTEXT.replace(
    "Use the synchronized calibration supplied with each image; ",
    "No camera calibration is supplied to this planning request; ",
)


class RoboCasaPlanClient(UserCheckV2Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for kind in ("plan", "replan"):
            self._instruction_text[kind] += "\n" + PLAN_ROBOT_CONTEXT


class RoboCasaAgentClient(AgentFirstV2Client):
    def build_payload(self, request, grid=None, mode="plan"):
        payload = super().build_payload(request, grid, mode)
        if mode == "agent":
            payload["instructions"] = payload["instructions"].replace(
                "gripper0_grip_site", "gripper0_right_grip_site") + "\n" + ROBOT_CONTEXT
        return payload


class RoboCasaUserCheckPolicy(AgentFirstV2Policy):
    """Use unchanged direct fragments; fail closed on unverified native macros."""

    def _capabilities(self):
        value = super()._capabilities()
        value["native_macros"] = {"implemented": [], "available": False,
                                  "reason": MACRO_REASON}
        value["robot_configuration"] = {
            "robot": "PandaOmron", "base_motion": "disabled",
            "torso_motion": "disabled", "arm_commands": "world_frame",
        }
        return value

    def _apply_agent(self, answer, view):
        if answer["type"] == "macro":
            self._record_feedback({"action": "macro", "status": "unavailable",
                                   "reason": MACRO_REASON, "executed_steps": 0,
                                   "continuation": "Use observed motion or execution fragments."})
            self._need_agent = True
            return
        return super()._apply_agent(answer, view)


def sensor_record(obs):
    return {**obs.to_dict(), "geometry": obs.geometry,
            "joint_position": list(obs.joint_position),
            "joint_velocity": list(obs.joint_velocity)}


def runtime_inventory(config, sim_config):
    """Bind code and settings, including the simulator checkout, to the review."""
    paths = set((ROOT / "src/gpt_field").rglob("*.py"))
    paths.update(ROOT / name for name in (
        RUNNER, reviewed.RUNNER_PATH, reviewed.HTML_PATH,
        reviewed.base.RUNNER_PATH, reviewed.base.V1_RUNNER_PATH,
        *reviewed.base.V1_HELPER_SCRIPTS, *reviewed.base.HELPER_CHAIN,
    ))
    # Hash actual simulator source files, not just git HEAD (checkouts can be dirty).
    for directory in (Path("/workspace/past_action_robocasa/robocasa/robocasa"),
                      Path("/workspace/past_action_robocasa/robosuite/robosuite")):
        paths.update(directory.rglob("*.py"))
    suite = Path("/workspace/past_action_robocasa/robosuite/robosuite")
    paths.update((suite / "controllers/config").rglob("*.json"))
    for relative in ("models/assets/grippers/panda_gripper.xml",
                     "models/assets/robots/panda/robot.xml"):
        paths.add(suite / relative)
    result = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(paths)}
    result["@configuration"] = canonical_sha256({
        "api": asdict(config), "simulator": asdict(sim_config),
        "method": METHOD, "policy": "RoboCasaUserCheckPolicy",
        "native_macros": False, "reasoning_effort": "high",
        "direction_semantics": DIRECTION_SEMANTICS,
        "python_executable": sys.executable,
    })
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
            sim_factory=RoboCasaSimProcess, client_factory=RoboCasaPlanClient):
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
            raise ValueError("Expected the official, initially incomplete CoffeePressButton task")
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
            "approval_mode": "manual_browser_or_chat_review", "manual_review": True,
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
    output.mkdir(parents=True, exist_ok=False)
    freeze_sources(output, runtime)
    write_json(output / "user_check_review.json", review)
    write_json(output / "manifest.json", {
        "method": METHOD, "task": "CoffeePressButton", "suite": "RoboCasa-0.2-arm-only",
        "api": asdict(config), "simulator": asdict(sim_config), "runtime_sha256": runtime,
        "billing": "chatgpt_subscription", "reasoning_effort": "high", "api_key_fallback": False,
        "automatic_episode_caps": None, "automatic_episode_retries": 0,
        "native_macros": False, "native_macro_reason": MACRO_REASON,
        "evaluator": "CoffeePressButton._check_success", "contacts": "record_only",
        "policy_inputs": "two RGB, state8, seven arm joint sensors, camera and fixed robot calibration",
        "hidden_scene_truth_to_policy": False, "human_reviewed_plan": True,
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
                    episode_id="ROBOCASA COFFEE / SPEEDUP V2 USER CHECK / ARM ONLY")
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
                obs, success = sim.step(decision.action)
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
                "method": METHOD, "task": "CoffeePressButton", "status": status, "error": error,
                "version": reviewed.VERSION, "direction_semantics": DIRECTION_SEMANTICS,
                "steps": steps, "feedback_count": feedback_count, "pending_action": pending_action,
                "robocasa_success": raw_success, "task_success": completed,
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
                "interpretation": "One manually reviewed arm-only RoboCasa attempt; incomplete times are not completion latency.",
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
    parser.add_argument("--layout-id", type=int, default=0)
    parser.add_argument("--style-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--reviewed-steps", type=int, nargs="+")
    parser.add_argument("--revision", type=int)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    config = reviewed.base.configuration()
    sim_config = RoboCasaConfig(layout_id=args.layout_id, style_id=args.style_id, seed=args.seed)
    if args.command == "preflight":
        print(json.dumps({"method": METHOD, "simulator": asdict(sim_config), "api": asdict(config),
                          "reasoning_effort": "high", "native_macros": False,
                          "code_files": len(runtime_inventory(config, sim_config)), "network_calls": 0}))
        return 0
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
