"""Fresh matched-reset RoboCasa pilot using the Astranomy short-target loop.

No UC7.1 plan, fragments, route, demonstrations or previous policy transcript
enter this loop. Previous initial records are evaluator-only reset assertions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time

from PIL import Image, ImageDraw

from gpt_field.astra import APIError, AstraConfig, MissingCredentials
from gpt_field.astranomy_agentic_motion import (
    ShortTargetConfig, ShortTargetExecutor, TargetLimitError, rejected_feedback,
)
from gpt_field.contact_skills import StepView
from gpt_field.experiment import write_json
from gpt_field.video import MP4Stream, WIDTH, HEIGHT, _font
from . import runner as common
from .astranomy import RoboCasaAstranomyClient, MobileExecutor, model_request
from .sim import RoboCasa365Config, RoboCasa365SimProcess

METHOD = "astranomy_direct_robocasa365_mobile"
VERSION = "robocasa365-astranomy-direct-runner-1"
ROOT, PROJECT = common.ROOT, common.PROJECT
TASK_PROMPT = common.TASK_PROMPT
OFFICIAL_HORIZON = common.OFFICIAL_HORIZON


def configuration():
    return AstraConfig(model="gpt-6-astra", timeout_s=180,
                       max_api_calls=None, max_wall_time_s=None, max_cost_usd=None)


def runtime_inventory(config, sim_config, reference_run):
    value = common.runtime_inventory(config, sim_config)
    for path in (PROJECT / "run_astranomy.py", PROJECT / "configs/restock_pantry_astranomy.json",
                 Path(reference_run) / "initial_sensors.json", Path(reference_run) / "initial_physics.json"):
        value[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    value["@configuration"] = common.canonical_sha256({
        "api": asdict(config), "simulator": asdict(sim_config), "method": METHOD,
        "version": VERSION, "reasoning_effort": "high", "motion": ShortTargetConfig().to_dict(),
        "mobile": True, "history": "all", "reference_run": str(Path(reference_run).resolve()),
        "separate_planning_call": False, "overload_retries": 0,
    })
    return value


def check_initial(obs, snapshot, reference_run):
    reference_run = Path(reference_run)
    expected_sensors = json.loads((reference_run / "initial_sensors.json").read_text())
    expected_physics = json.loads((reference_run / "initial_physics.json").read_text())
    sensors = common.sensor_record(obs)
    checks = {
        "physical_sha256": snapshot["physical_sha256"] == expected_physics["physical_sha256"],
        "model_xml_sha256": snapshot["model_xml_sha256"] == expected_physics["model_xml_sha256"],
        "complete_sensors": sensors == expected_sensors,
        "official_prompt": obs.prompt == TASK_PROMPT,
    }
    return {"reference_run": str(reference_run.resolve()), "checks": checks,
            "matched": all(checks.values()),
            "actual_sensor_sha256": common.canonical_sha256(sensors),
            "expected_sensor_sha256": common.canonical_sha256(expected_sensors),
            "physical_sha256": snapshot["physical_sha256"],
            "model_xml_sha256": snapshot["model_xml_sha256"],
            "prior_trajectory_or_object_state_to_policy": False}


class Recorder:
    def __init__(self, output):
        self.stream = MP4Stream(Path(output) / "episode.mp4", fps=20)

    def record(self, obs, step, calls, tool, feedback):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), (12, 18, 30))
        draw = ImageDraw.Draw(canvas)
        draw.text((30, 20), "ASTRANOMY DIRECT / ROBOCASA365 / ASTRA HIGH SUBSCRIPTION",
                  font=_font(25, True), fill="white")
        draw.text((30, 65), TASK_PROMPT, font=_font(20), fill="white")
        draw.text((30, 105), f"Step {step} | model calls {calls} | 1-5 step tool segments",
                  font=_font(20), fill="#66e3a1")
        view = StepView(obs)
        for label, pixels, x in (("EXTERNAL RGB", view.external_rgb_upright, 40),
                                 ("WRIST RGB", view.wrist_rgb_upright, 840)):
            draw.text((x, 150), label, font=_font(18), fill="white")
            canvas.paste(Image.fromarray(pixels).resize((640, 640)), (x, 185))
        lines = [json.dumps(tool, ensure_ascii=True)[:175],
                 json.dumps(feedback, ensure_ascii=True)[:175] if feedback else "",
                 "20 Hz simulation; inference waits omitted. Only raw RoboCasa determines task success."]
        for i, line in enumerate(lines):
            draw.text((30, 842 + 34*i), line, font=_font(15), fill="white")
        self.stream.write(canvas)

    def close(self):
        self.stream.close()


def execute(output, *, reference_run, config, sim_config, token_provider,
            sim_factory=RoboCasa365SimProcess, client_factory=RoboCasaAstranomyClient,
            video=True):
    if not callable(token_provider):
        raise ValueError("Subscription token provider required")
    output, reference_run = Path(output).resolve(), Path(reference_run).resolve()
    if output.exists():
        raise FileExistsError("A fresh output is required; no resume or automatic retry")
    runtime = runtime_inventory(config, sim_config, reference_run)
    output.mkdir(parents=True, exist_ok=False)
    common.freeze_sources(output, runtime)
    write_json(output / "manifest.json", {
        "method": METHOD, "version": VERSION, "task": "RestockPantry",
        "api": asdict(config), "simulator": asdict(sim_config), "runtime_sha256": runtime,
        "reference_run": str(reference_run), "reasoning_effort": "high",
        "billing": "chatgpt_subscription", "api_key_fallback": False,
        "motion": ShortTargetConfig().to_dict(), "mobile": True,
        "history": "all accepted text/images/tools/robot feedback", "zero_shot": True,
        "prior_episode_history_to_policy": False, "separate_planning_call": False,
        "uc71_fragments": False, "runtime_harness": "subscription_native_tool_loop",
        "automatic_episode_caps": None, "automatic_episode_retries": 0, "overload_retries": 0,
        "collision_policy": "record_only", "evaluation_profile": "uncapped_exploratory_pilot",
        "official_reference_horizon": OFFICIAL_HORIZON, "official_benchmark_score": False,
        "authorization": "retest this using the astranomy direct control",
        "adaptations": ["RoboCasa PandaOmron and robot-mounted external camera",
                        "same native 20 Hz OSC arm controller as the reference run",
                        "additional bounded native base/torso tool, unchanged gripper",
                        "RoboCasa-only raw task evaluator outside policy"],
    })
    sim = recorder = obs = client = executor = None
    status, stage, error = "starting", "starting", None
    steps = feedback_count = calls = moves = rejected = arm_steps = mobile_steps = 0
    sim_status, last_tool, last_feedback = {}, None, None
    last_gripper, pending_action, pending_tool = -1., None, False
    started = time.monotonic()
    control_started = control_stopped = None
    timings = {key: 0. for key in ("model_pipeline_s", "local_planning_s", "local_control_s",
                                    "simulator_step_s", "video_render_s")}
    cleanup_errors = []
    trace = (output / "trace.jsonl").open("w")

    def log(event, **data):
        trace.write(json.dumps({"event": event, "step": steps, **data}, allow_nan=False) + "\n")
        trace.flush()

    def state(new_stage):
        nonlocal stage
        stage = new_stage
        write_json(output / "run_state.json", {
            "stage": stage, "status": status, "steps": steps, "feedback_count": feedback_count,
            "model_responses": calls, "moves": moves, "arm_steps": arm_steps,
            "mobile_steps": mobile_steps, "pending_action": pending_action,
            "pending_tool": pending_tool, "last_tool": last_tool, "last_feedback": last_feedback,
            "robocasa_success": bool(sim_status.get("robocasa_success")),
            "elapsed_s": time.monotonic() - started,
            "budget": common.terminal_budget(client), "timing_components": timings,
        })

    def check_stop():
        if (output / "operator_stop.request.json").exists():
            raise KeyboardInterrupt("Operator stop file")

    def finish_tool(feedback):
        nonlocal pending_tool, last_feedback
        last_feedback = feedback
        log("tool_result", feedback=feedback)
        client.record_result(feedback)
        pending_tool = False

    try:
        state("starting")
        sim = common.new_sim(sim_config, sim_factory)
        obs = sim.reset()
        sim_status = deepcopy(sim.status)
        initial = sim.snapshot(include_xml=True)
        write_json(output / "initial_physics.json", initial)
        write_json(output / "initial_sensors.json", common.sensor_record(obs))
        write_json(output / "simulator.json", {"metadata": sim.metadata, "status": sim_status})
        comparison = check_initial(obs, initial, reference_run)
        write_json(output / "initial_match.json", comparison)
        if not comparison["matched"] or sim_status.get("success"):
            raise ValueError("Initial reset does not match the incomplete reference; no model or policy action")
        client = client_factory(config, token_provider=token_provider, log_dir=output / "requests")
        if video:
            recorder = Recorder(output)
            recorder.record(obs, 0, 0, None, None)
        log("reset", observation=common.sensor_record(obs), evaluator=sim_status)
        control_started = time.monotonic()
        status = "running"
        while True:
            check_stop()
            if obs.prompt != TASK_PROMPT:
                raise ValueError("Official task prompt changed")
            state("deciding")
            t = time.monotonic()
            try:
                last_tool = client.predict(model_request(obs, steps))
            finally:
                timings["model_pipeline_s"] += time.monotonic() - t
            calls += 1
            pending_tool = True
            executor = None
            common.validate_budget(client)
            log("model_tool", tool=last_tool)
            args = last_tool["arguments"]
            is_mobile = last_tool["name"] == "execute_mobile"
            t = time.monotonic()
            try:
                executor = (MobileExecutor(obs, args, last_gripper) if is_mobile else
                            ShortTargetExecutor(obs.state, args["target"], args["steps"]))
            except TargetLimitError as exc:
                rejected += 1
                finish_tool(rejected_feedback(obs.state, args["target"], args["steps"], str(exc)))
            finally:
                timings["local_planning_s"] += time.monotonic() - t
            moves += 1
            while executor is not None and not executor.done:
                check_stop()
                t = time.monotonic()
                action = executor.next_action(obs if is_mobile else obs.state)
                timings["local_control_s"] += time.monotonic() - t
                before = obs
                mobile = {key: args[key] for key in ("base", "torso")} if is_mobile else None
                pending_action = steps
                log("action", action=list(action), mobile_command=mobile)
                state("executing")
                t = time.monotonic()
                obs, success = sim.mobile_step(action, mobile) if is_mobile else sim.step(action)
                timings["simulator_step_s"] += time.monotonic() - t
                steps += 1
                sim_status = deepcopy(sim.status)
                if type(success) is not bool or success != sim_status.get("robocasa_success"):
                    raise ValueError("Inconsistent raw RoboCasa evaluator status")
                log("executed", action_index=steps-1, observation=common.sensor_record(obs), evaluator=sim_status)
                t = time.monotonic()
                if is_mobile:
                    executor.observe_transition(before, action, obs)
                    mobile_steps += 1
                else:
                    executor.observe_transition(before.state, action, obs.state)
                    last_gripper = float(action[-1])
                    arm_steps += 1
                feedback_count += 1
                pending_action = None
                log("feedback", action_index=steps-1,
                    feedback=executor.feedback(obs if is_mobile else obs.state))
                timings["local_control_s"] += time.monotonic() - t
                if recorder:
                    t = time.monotonic()
                    recorder.record(obs, steps, calls, last_tool, last_feedback)
                    timings["video_render_s"] += time.monotonic() - t
                if success:
                    status = "success"
                    if not executor.done:
                        executor.interrupt("Host ended this segment before the selected steps completed")
                    break
                if sim_status.get("terminated"):
                    raise RuntimeError("Simulator terminated before official success")
            if executor is not None:
                finish_tool(executor.feedback(obs if is_mobile else obs.state))
            state("observed")
            view = StepView(obs)
            Image.fromarray(view.external_rgb_upright).save(output / "latest_external.png")
            Image.fromarray(view.wrist_rgb_upright).save(output / "latest_wrist.png")
            print(json.dumps({"event": "segment", "steps": steps, "calls": calls,
                              "tool": last_tool, "feedback": last_feedback,
                              "robocasa_success": bool(sim_status.get("robocasa_success"))}), flush=True)
            if status == "success":
                break
    except KeyboardInterrupt:
        status = "user_stop"
        log("user_stop", stage=stage)
    except Exception as exc:
        status = "subscription_error" if isinstance(exc, (APIError, MissingCredentials)) else "infrastructure_error"
        error = {"type": type(exc).__name__, "message": common.reviewed.base.redact_secrets(
            str(exc), getattr(client, "_used_access_tokens", ()) if client else ())}
        log("error", error=error)
    finally:
        if control_started is not None:
            control_stopped = time.monotonic()
        handlers = {sig: signal.signal(sig, signal.SIG_IGN) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if pending_tool and executor is not None and pending_action is None:
                try:
                    executor.interrupt("Host interrupted the segment")
                    finish_tool(executor.feedback(obs if last_tool["name"] == "execute_mobile" else obs.state))
                except Exception as exc:
                    cleanup_errors.append({"component": "pending_tool_feedback", "type": type(exc).__name__})
            if sim is not None and obs is not None:
                try:
                    write_json(output / "final_physics.json", sim.snapshot())
                except Exception as exc:
                    cleanup_errors.append({"component": "final_physics", "type": type(exc).__name__})
            for name, owner in (("simulator", sim), ("video", recorder)):
                if owner is not None:
                    try:
                        owner.close()
                    except Exception as exc:
                        cleanup_errors.append({"component": name, "type": type(exc).__name__})
            if obs is not None:
                try:
                    write_json(output / "final_sensors.json", common.sensor_record(obs))
                    view = StepView(obs)
                    Image.fromarray(view.external_rgb_upright).save(output / "final_external.png")
                    Image.fromarray(view.wrist_rgb_upright).save(output / "final_wrist.png")
                except Exception as exc:
                    cleanup_errors.append({"component": "final_sensors", "type": type(exc).__name__})
            try:
                unchanged = runtime == runtime_inventory(config, sim_config, reference_run)
            except Exception as exc:
                unchanged = False
                cleanup_errors.append({"component": "source_inventory", "type": type(exc).__name__})
            budget = common.terminal_budget(client)
            raw = bool(sim_status.get("robocasa_success"))
            complete = bool(status == "success" and raw and unchanged and pending_action is None
                            and not pending_tool and steps == feedback_count and budget is not None
                            and "validation_error" not in budget)
            elapsed = None if control_started is None else control_stopped-control_started
            result = {
                "method": METHOD, "version": VERSION, "task": "RestockPantry", "status": status,
                "error": error, "robocasa_success": raw, "task_success": complete,
                "steps": steps, "feedback_count": feedback_count, "model_responses": calls,
                "moves": moves, "rejected_moves": rejected, "arm_steps": arm_steps, "mobile_steps": mobile_steps,
                "pending_action": pending_action, "pending_tool": pending_tool,
                "execution_budget": budget, "preparation_model_calls": 0,
                "history": client.history_stats() if client is not None else None,
                "billing": "chatgpt_subscription", "api_key_calls": budget.get("api_key_calls") if budget else 0,
                "reasoning_effort": "high", "runtime_unchanged": unchanged,
                "cleanup_errors": cleanup_errors, "evaluator_terminal": sim_status,
                "control_elapsed_s": elapsed, "timing_components": timings,
                "control_without_video_s": None if elapsed is None else elapsed-timings["video_render_s"],
                "elapsed_s": time.monotonic()-started, "completion_time_eligible": complete,
                "evaluation_completed": complete, "evaluation_profile": "uncapped_exploratory_pilot",
                "official_reference_horizon": OFFICIAL_HORIZON,
                "success_within_reference_horizon": bool(complete and steps <= OFFICIAL_HORIZON),
                "official_benchmark_score": False, "reference_run": str(reference_run),
                "timing_note": "Control wall includes all in-loop inference and tool feedback; only measured synchronous video recording is subtracted. Initial reset/first video frame and final cleanup/encoding are excluded. No separate planning call. On an incomplete stop, pending-tool bookkeeping in finally occurs after the control clock stops.",
                "video": "episode.mp4" if recorder is not None else None,
            }
            write_json(output / "summary.json", result)
            state("terminal")
            log("terminal", summary=result)
            trace.close()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--test-approved", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    if not args.test_approved:
        parser.error("An explicit user authorization is required before a full-task test")
    from gpt_field.chatgpt_auth import SubscriptionCredentials
    credentials = SubscriptionCredentials()
    with common.reviewed.base.termination_as_interrupt():
        summary = execute(args.output, reference_run=args.reference_run, config=configuration(),
                          sim_config=RoboCasa365Config(seed=7),
                          token_provider=credentials.access_token, video=not args.no_video)
    print(json.dumps(summary), flush=True)
    return 0 if summary["task_success"] else 3
