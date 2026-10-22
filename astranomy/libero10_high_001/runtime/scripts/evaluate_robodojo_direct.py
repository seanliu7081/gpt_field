"""RoboDojo-style zero-shot EEF baseline adapted to single-arm LIBERO.

Only generic move_eef / give_up tools. Astra medium on the ChatGPT subscription;
no demonstrations, task skills, depth, object state, or evaluator feedback.
The adapter retains LIBERO's 20 Hz OSC physics instead of RoboDojo's 25 Hz
CuRobo joint trajectory backend. Robot-only IK checks the requested path before
motion. A tool's bounded executor returns measured residuals on non-arrival;
there is no full-episode action, call, or wall-time cap.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

from gpt_field.astra import AstraConfig
from gpt_field.distance_tasks import DISTANCE_TASK_KEYS, resolve_distance_task
from gpt_field.experiment import digest, source_manifest, write_json
from gpt_field.libero import LiberoConfig
from gpt_field.observation import decode_image, image_to_data_url
from gpt_field.video import MP4Stream, WIDTH, HEIGHT, _font

ROOT = Path(__file__).resolve().parents[1]
METHOD = "robodojo_direct"
spec = importlib.util.spec_from_file_location("_robodojo_lifecycle", ROOT / "scripts/evaluate_agent_speedup_v1.py")
life = importlib.util.module_from_spec(spec)
spec.loader.exec_module(life)


def configuration():
    return AstraConfig(model="gpt-6-astra", timeout_s=180,
                       max_api_calls=None, max_wall_time_s=None, max_cost_usd=None)


def sim_configuration(task_key):
    task = resolve_distance_task(task_key)
    return LiberoConfig(suite=task.suite, seed=7, image_size=512, render_size=512,
        max_steps=None, settle_steps=10, python_executable=life.DEFAULT_SIM_PYTHON,
        libero_root=life.DEFAULT_LIBERO_ROOT, stop_on_success=True,
        collision_check_mode="task_manipulation_v1", stop_on_collision=False,
        include_arm_joints=True)


def upright(url):
    """Undo legacy horizontal reversal, retaining native 512-pixel RGB."""
    pixels = np.asarray(decode_image(url).transpose(Image.Transpose.FLIP_LEFT_RIGHT))
    return image_to_data_url(pixels, size=512, rotate=False)


def model_request(obs, step):
    from gpt_field.robodojo_direct_motion import robot_state
    return {"external_image": upright(obs.image), "wrist_image": upright(obs.wrist_image),
            "robot_state": robot_state(obs.state), "joint_positions": list(obs.joint_position), "step": step}


def initial_identity(obs):
    # Match existing 512-pixel reset references before adding read-only joints.
    core = {k: v for k, v in obs.to_dict().items() if "joint_" not in k}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Recorder:
    def __init__(self, output, task, smoke=False):
        self.output, self.task, self.smoke = output, task, smoke
        self.stream = MP4Stream(output / "episode.mp4", fps=20)

    def record(self, obs, step, calls, tool, phase, feedback=None):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), (12, 18, 30))
        draw = ImageDraw.Draw(canvas)
        title = "ROBODOJO DIRECT / LIBERO / " + ("SCRIPTED DIAGNOSTIC" if self.smoke else "ASTRA MEDIUM SUBSCRIPTION")
        draw.text((30, 20), title, font=_font(25, True), fill="white")
        draw.text((30, 65), self.task, font=_font(20), fill="white")
        draw.text((30, 105), f"Step {step} | model calls {calls} | {phase}", font=_font(20), fill="#66e3a1")
        for label, url, x in (("EXTERNAL RGB", obs.image, 40), ("WRIST RGB", obs.wrist_image, 840)):
            draw.text((x, 150), label, font=_font(18), fill="white")
            frame = decode_image(url).transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize((640, 640))
            canvas.paste(frame, (x, 185))
        for i, line in enumerate((json.dumps(tool, ensure_ascii=True)[:175],
                                  json.dumps(feedback, ensure_ascii=True)[:175] if feedback else "",
                                  "20 Hz simulation; inference waits omitted. Only LIBERO determines task success.")):
            draw.text((30, 842 + 34*i), line, font=_font(15), fill="white")
        self.stream.write(canvas)

    def close(self):
        self.stream.close()


def freeze_sources(output):
    destination = output / "frozen_source"
    records = []
    for folder in ("src/gpt_field", "scripts"):
        for path in sorted((ROOT / folder).glob("*.py")):
            target = destination / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            target.write_bytes(data)
            records.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest()})
    write_json(output / "source_provenance.json", records)


def execute(output, *, task_key, episode_index=0, token_provider=None, smoke=False,
            video=True, robot_xml=None, client_factory=None, sim_factory=None):
    from gpt_field.robodojo_direct_client import RoboDojoDirectClient
    from gpt_field.robodojo_direct_motion import AbsoluteMotionExecutor, robot_state, MotionConfig
    from gpt_field.robodojo_direct_sim import PandaKinematics
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory; existing results cannot be resumed or overwritten")
    if not smoke and not callable(token_provider):
        raise ValueError("A subscription token provider is required; API keys are unsupported")
    if not 0 <= episode_index < 50:
        raise ValueError("episode index must be 0..49")
    task, config, sim_config = resolve_distance_task(task_key), configuration(), sim_configuration(task_key)
    kinematics = PandaKinematics.from_xml(robot_xml, python_executable=sim_config.python_executable)
    output.mkdir(parents=True)
    freeze_sources(output)
    client = None if smoke else (client_factory or RoboDojoDirectClient)(
        config, token_provider=token_provider, task_key=task_key, log_dir=output / "requests")
    identity = {"method": METHOD, "zero_shot": True, "task": task.prompt, "task_id": task.task_id,
        "episode_index": episode_index, "distance_task": task_key, "simulator": asdict(sim_config),
        "api": asdict(config), "reasoning_effort": "medium", "billing": "offline_scripted" if smoke else "chatgpt_subscription",
        "api_key_fallback": False, "model_fallback": False, "software": source_manifest(),
        "motion": asdict(MotionConfig()), "robot_kinematics": kinematics.metadata,
        "image_history": 2, "text_history": "all",
        "camera_input": "two native 512 RGB views, upright, no depth or calibration",
        "robot_input": "grasp-point pose, gripper aperture, seven read-only arm joint positions",
        "official_source": "https://github.com/XPolicyLab/XPolicyLab/tree/562c14cf08b9e6e411e7d76f192740afced77160/policy/GPT_6_Astra_Direct_EEF",
        "adaptations": ["single Panda arm and two LIBERO cameras", "20 Hz native OSC tracking instead of 25 Hz CuRobo joint trajectories",
            "robot-only IK plus Cartesian path; no scene-obstacle planner or task skills",
            "Panda downward orientation reference and native grasp point", "no episode call/action/time cap per existing user protocol"],
        "stopping_rule": ["raw LIBERO success", "model give_up (unsuccessful)", "operator stop", "infrastructure/subscription failure"],
        "collision_policy": "record_only", "success_qualification": "raw_libero_goal",
        "simulation_pauses_for_api": True, "prior_budgets_inherited": False}
    write_json(output / "manifest.json", {"configuration": identity, "identity_hash": digest(identity),
        "created_utc": datetime.now(timezone.utc).isoformat()})
    sim = recorder = obs = executor = None
    started = time.monotonic()
    control_started = control_stopped = None
    status, stage, error = "running", "starting", None
    steps = calls = moves = rejected = 0
    sim_status, last_tool, last_feedback = {}, None, None
    # The initial reset settles the native gripper open. Keep its command latched during transport.
    previous_gripper = 1.0
    timings = {k: 0.0 for k in ("model_pipeline_s", "local_planning_s", "local_control_s", "simulator_step_s", "video_render_s")}
    cleanup_errors, summaries = [], []
    smoke_index = 0
    trace = (output / "trace.jsonl").open("w")

    def log(event, **data):
        trace.write(json.dumps({"event": event, "step": steps, **data}, allow_nan=False) + "\n")
        trace.flush()

    def state(new_stage):
        nonlocal stage
        stage = new_stage
        budget = client.budget_snapshot() if client else None
        write_json(output / "run_state.json", {"status": status, "stage": stage, "method": METHOD,
            "steps": steps, "api_calls": budget.get("api_calls", calls) if budget else 0,
            "model_responses": calls, "moves": moves, "last_tool": last_tool,
            "last_feedback": last_feedback, "timing_components": timings,
            "budget": budget})

    def check_stop():
        if (output / "operator_stop.request.json").exists():
            raise KeyboardInterrupt("Operator stop file")

    def render(phase):
        if recorder:
            t = time.monotonic()
            recorder.record(obs, steps, calls, last_tool, phase, last_feedback)
            timings["video_render_s"] += time.monotonic() - t

    def summary():
        budget = client.budget_snapshot() if client else {"api_calls": 0, "api_key_calls": 0}
        success = bool(sim_status.get("libero_success", sim_status.get("success", False)))
        completed = not smoke and status == "success" and success
        elapsed = None if control_started is None else (control_stopped or time.monotonic()) - control_started
        return {"method": METHOD, "status": status, "task": task.prompt, "suite": task.suite,
            "distance_task": task_key, "task_id": task.task_id, "episode_index": episode_index, "seed": 7,
            "zero_shot": True, "reasoning_effort": "medium", "billing": identity["billing"],
            "api_calls": budget.get("api_calls", calls), "api_key_calls": 0, "api_key_fallback": False,
            "model_responses": calls, "moves": moves, "rejected_moves": rejected, "steps": steps,
            "simulated_motion_s": steps / 20, "timing_components": dict(timings),
            "control_elapsed_s": elapsed,
            "control_without_video_s": None if elapsed is None else elapsed - timings["video_render_s"],
            "elapsed_s": time.monotonic() - started,
            "timing_note": "Control wall time includes model, local planning/control and simulator overhead; subtract only recorder time. Reset and final video encoding excluded.",
            "budget": budget, "libero_success": success, "libero_success_observed": success,
            "benchmark_completed": completed, "evaluation_completed": completed,
            "completion_time_eligible": completed, "success_rate": 1.0 if completed else 0.0 if status == "model_give_up" else None,
            "scored_episodes": int(completed or (not smoke and status == "model_give_up")),
            "incomplete_episodes": int(not completed), "diagnostic_complete": smoke and status == "diagnostic_complete",
            "model_declared_complete": False, "last_tool": last_tool, "last_feedback": last_feedback,
            "collision": sim_status.get("collision"), "max_api_calls": None, "max_policy_steps": None,
            "max_wall_time_s": None, "error": error, "cleanup_errors": cleanup_errors,
            "video": "episode.mp4" if (output / "episode.mp4").exists() else None,
            "trace": "trace.jsonl", "output": str(output)}

    print(json.dumps({"event": "start", "method": METHOD, "task": task_key, "output": str(output),
        "operator_stop": f"touch {output / 'operator_stop.request.json'}", "smoke": smoke}), flush=True)
    with life.termination_as_interrupt():
        try:
            state("starting")
            with life._simulator_without_credentials():
                sim = (sim_factory or life._SessionSimProcess)(sim_config)
                sim.start()
            write_json(output / "simulator.json", sim.metadata)
            obs = sim.reset(task.task_id, episode_index)
            sim_status = dict(sim.status)
            if obs.prompt != task.prompt:
                raise ValueError("Official task mismatch")
            write_json(output / "initial_observation.json", obs.to_dict())
            actual = initial_identity(obs)
            expected = life.DEFAULT_EXPECTED_SHA256.get((task_key, episode_index))
            write_json(output / "initial_observation_check.json", {"sha256": actual, "expected": expected,
                       "matched": actual == expected if expected else None})
            if expected and actual != expected:
                raise ValueError("Initial RGB/state differs from the frozen historical reset; no inference performed")
            if video:
                recorder = Recorder(output, task.prompt, smoke=smoke)
            log("reset", observation=obs.to_dict(), sim_status=sim_status)
            control_started = time.monotonic()
            render("reset")
            while not sim_status.get("terminated"):
                check_stop()
                state("deciding")
                if smoke:
                    measured = robot_state(obs.state)
                    # Exercise translation, orientation, post-arm grip, and rejection without task knowledge.
                    choices = [
                        {"z": measured["z"] + .025, "note": "offline upward motion"},
                        {"yaw_deg": measured["yaw_deg"] + 8, "note": "offline orientation-only motion"},
                        {"gripper": 0.0, "note": "offline close"},
                        {"gripper": 1.0, "note": "offline open"},
                        {"x": 10.0, "note": "offline invalid target"}]
                    if smoke_index >= len(choices):
                        status = "diagnostic_complete"
                        break
                    last_tool = {"name": "move_eef", "arguments": choices[smoke_index]}
                    smoke_index += 1
                else:
                    t = time.monotonic()
                    try:
                        last_tool = client.predict(model_request(obs, steps))
                    finally:
                        # Failed and interrupted subscription waits still consume wall time.
                        timings["model_pipeline_s"] += time.monotonic() - t
                    calls += 1
                log("model_tool" if not smoke else "scripted_tool", tool=last_tool)
                print(json.dumps({"event": "tool", "call": calls, "step": steps, "tool": last_tool}), flush=True)
                if last_tool["name"] == "give_up":
                    status = "model_give_up"
                    break
                check_stop()
                state("planning")
                t = time.monotonic()
                executor = AbsoluteMotionExecutor(obs.state, last_tool["arguments"],
                    joint_positions=obs.joint_position, kinematics=kinematics, previous_gripper=previous_gripper)
                timings["local_planning_s"] += time.monotonic() - t
                log("motion_preflight", preflight=executor.preflight)
                moves += 1
                start_steps = steps
                while not executor.done and not sim_status.get("terminated"):
                    check_stop()
                    t = time.monotonic()
                    action = executor.next_action(obs.state)
                    timings["local_control_s"] += time.monotonic() - t
                    if action is None:
                        break
                    state("executing")
                    before = obs
                    t = time.monotonic()
                    obs, _ = sim.step(action)
                    timings["simulator_step_s"] += time.monotonic() - t
                    sim_status = dict(sim.status)
                    steps += 1
                    t = time.monotonic()
                    executor.observe_transition(before.state, action, obs.state)
                    timings["local_control_s"] += time.monotonic() - t
                    log("step", action=list(action), before_state=list(before.state), observation=obs.to_dict(), sim_status=sim_status)
                    render("executing")
                last_feedback = executor.feedback(obs.state)
                previous_gripper = executor.current_gripper
                if steps == start_steps:
                    rejected += 1
                summaries.append(last_feedback)
                log("tool_result", feedback=last_feedback)
                if client:
                    client.record_result(last_feedback)
                state("feedback")
                print(json.dumps({"event": "result", "call": calls, "step": steps, "feedback": last_feedback}), flush=True)
                decode_image(obs.image).transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(output / "latest_external.png")
                decode_image(obs.wrist_image).transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(output / "latest_wrist.png")
            if sim_status.get("libero_success", sim_status.get("success", False)):
                status = "diagnostic_complete" if smoke else "success"
            elif status == "running":
                status = "simulator_terminated"
        except KeyboardInterrupt:
            status = "user_stop"
            log("user_stop", stage=stage)
        except Exception as exc:
            status = "infrastructure_error"
            secrets = getattr(client, "_used_access_tokens", ()) if client else ()
            error = {"type": type(exc).__name__, "stage": stage,
                     "message": life.redact_secrets(str(exc), secrets)}
            (output / "error_traceback.txt").write_text(life.redact_secrets(traceback.format_exc(), secrets))
            log("error", error=error)
        finally:
            control_stopped = time.monotonic()
            trace.close()
            write_json(output / "summary.json", summary())
            write_json(output / "tool_results.json", summaries)
            # Preserve summary before cleanup; late signals must not interrupt video finalization.
            old_handlers = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                for label, obj in (("simulator", sim), ("video", recorder)):
                    if obj is not None:
                        try:
                            obj.close()
                        except Exception as exc:
                            cleanup_errors.append({"stage": label, "type": type(exc).__name__})
                write_json(output / "summary.json", summary())
                state("complete")
            finally:
                for s, handler in old_handlers.items():
                    signal.signal(s, handler)
    result = summary()
    print(json.dumps({"event": "finished", **result}), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", choices=DISTANCE_TASK_KEYS, default="bowl_plate")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--robot-xml", type=Path)
    parser.add_argument("--no-video", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--sim-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight:
        from gpt_field.robodojo_direct_client import RoboDojoDirectClient
        pixels = image_to_data_url(np.zeros((512, 512, 3), dtype=np.uint8), size=512, rotate=False)
        request = {"external_image": pixels, "wrist_image": pixels, "robot_state": {
            "x": 0., "y": 0., "z": 1., "pitch_deg": 0., "roll_deg": 0., "yaw_deg": 0., "gripper": 1.},
            "joint_positions": [0.] * 7, "step": 0}
        client = RoboDojoDirectClient(configuration(), token_provider=lambda: None, task_key=args.task)
        payload = client.build_payload(request)
        print(json.dumps({"method": METHOD, "simulator": asdict(sim_configuration(args.task)), "payload": payload}))
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.sim_smoke and args.store_dir:
        parser.error("Offline smoke never accesses credentials")
    from gpt_field.chatgpt_auth import SubscriptionCredentials
    token_provider = None if args.sim_smoke else SubscriptionCredentials(store_dir=args.store_dir).access_token
    result = execute(args.output, task_key=args.task, episode_index=args.episode_index,
        token_provider=token_provider, smoke=args.sim_smoke, video=not args.no_video, robot_xml=args.robot_xml)
    return 0 if result["evaluation_completed"] or result["diagnostic_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
