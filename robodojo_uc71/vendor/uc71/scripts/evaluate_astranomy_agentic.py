"""Astranomy Agentic-inspired subscription baseline for single-arm LIBERO.

One complete absolute EEF target held for 1--5 native 20 Hz OSC steps,
with concurrent gripper control, all visual/text history, robot/camera geometry
and measured feedback. This is the user-approved custom subscription harness;
it does not run Codex. No demonstrations, skills or full-episode caps.

Reference: https://research.nvidia.com/labs/srl/projects/astranomy/ (final
Agentic configuration / expanded interface). The original uses DROID, three
384x216 cameras and 15 Hz stepwise DLS joint control. This port uses LIBERO's
Panda, two upright 512x512 views and native 20 Hz OSC. Camera calibration and
fixed gripper geometry are policy inputs; scene truth and subtask progress are
not. The subscription transport is the same OAuth route as prior baselines,
with no API-key billing or fallback. Medium reasoning remains fixed.

Run one fresh task:
  PYTHONPATH=src .venv/bin/python scripts/evaluate_astranomy_agentic.py \
      --task bowl_plate --output runs/astranomy_agentic_bowl_trial \
      --store-dir /workspace/gpt_field/.private/chatgpt_plan
Use --task drawer_bowl for the drawer task. Use --sim-smoke without --store-dir
for a generic offline protocol diagnostic. Every output directory must be new.
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
METHOD = "astranomy_agentic"
spec = importlib.util.spec_from_file_location("_astranomy_lifecycle", ROOT / "scripts/evaluate_agent_speedup_v1.py")
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


class CalibratedSimProcess(life._SessionSimProcess):
    def __init__(self, config):
        super().__init__(config, command=[config.python_executable or sys.executable, "-u", "-m",
            "gpt_field.astranomy_agentic_sim", "--worker", "--config-json", json.dumps(asdict(config))])

    def geometry(self):
        from gpt_field.astranomy_agentic_geometry import validate_geometry_metadata
        return validate_geometry_metadata(self._request({"op": "geometry"}))


def model_request(obs, step, geometry):
    from gpt_field.astranomy_agentic_motion import measured_pose
    return {"external_image": upright(obs.image), "wrist_image": upright(obs.wrist_image),
            "robot_state": measured_pose(obs.state), "joint_positions": list(obs.joint_position),
            "geometry": geometry, "step": step}


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
        title = "ASTRANOMY AGENTIC ADAPTATION / LIBERO / " + ("SCRIPTED DIAGNOSTIC" if self.smoke else "ASTRA MEDIUM SUBSCRIPTION")
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
            video=True, client_factory=None, sim_factory=None):
    from gpt_field.astranomy_agentic_client import AstranomyAgenticClient
    from gpt_field.astranomy_agentic_motion import (ShortTargetExecutor, ShortTargetConfig,
        TargetLimitError, measured_pose, rejected_feedback)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory; existing results cannot be resumed or overwritten")
    if not smoke and not callable(token_provider):
        raise ValueError("A subscription token provider is required; API keys are unsupported")
    if not 0 <= episode_index < 50:
        raise ValueError("episode index must be 0..49")
    task, config, sim_config = resolve_distance_task(task_key), configuration(), sim_configuration(task_key)
    output.mkdir(parents=True)
    freeze_sources(output)
    client = None if smoke else (client_factory or AstranomyAgenticClient)(
        config, token_provider=token_provider, task_key=task_key, log_dir=output / "requests")
    identity = {"method": METHOD, "zero_shot": True, "task": task.prompt, "task_id": task.task_id,
        "episode_index": episode_index, "distance_task": task_key, "simulator": asdict(sim_config),
        "api": asdict(config), "reasoning_effort": "medium", "billing": "offline_scripted" if smoke else "chatgpt_subscription",
        "api_key_fallback": False, "model_fallback": False, "software": source_manifest(),
        "motion": ShortTargetConfig().to_dict(),
        "image_history": "all", "text_history": "all", "runtime_harness": "custom_subscription_tool_loop",
        "codex_runtime": False, "user_requested_subscription_harness": True,
        "camera_input": "two native 512 RGB views, upright; synchronized camera calibration; no depth",
        "robot_input": "grip-site position, right-hand quaternion, measured jaw closure, seven arm joints; fixed gripper geometry",
        "official_source": "https://research.nvidia.com/labs/srl/projects/astranomy/",
        "adaptations": ["existing subscription harness in place of Codex, explicitly requested by user",
            "LIBERO Panda and two 512 RGB cameras instead of DROID and three 384x216 cameras",
            "20 Hz native OSC instead of 15 Hz stepwise DLS joint control; no per-joint increment guarantee",
            "grip-site position and right-hand orientation instead of Robotiq mounting flange",
            "no simulator subtask-progress input; raw task outcome remains evaluator-only",
            "no whole-episode action cap per existing user protocol"],
        "stopping_rule": ["raw LIBERO success", "operator stop", "infrastructure/subscription failure"],
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
    timings = {k: 0.0 for k in ("model_pipeline_s", "local_planning_s", "local_control_s", "simulator_step_s", "geometry_s", "video_render_s")}
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
            "completion_time_eligible": completed, "success_rate": 1.0 if completed else None,
            "scored_episodes": int(completed),
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
                sim = (sim_factory or CalibratedSimProcess)(sim_config)
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
                t = time.monotonic()
                geometry = sim.geometry()
                timings["geometry_s"] += time.monotonic() - t
                if not (output / "initial_geometry.json").exists():
                    write_json(output / "initial_geometry.json", geometry)
                if smoke:
                    measured = measured_pose(obs.state)
                    if smoke_index >= 5:
                        status = "diagnostic_complete"
                        break
                    target = list(measured)
                    if smoke_index == 0:
                        target[2] += .015
                        target[7] = 1.0
                    elif smoke_index == 1:
                        from gpt_field.robodojo_direct_sim import rotation_matrix, matrix_quaternion
                        q = matrix_quaternion(rotation_matrix([0., 0., .1]) @ rotation_matrix(obs.state[3:6]))
                        target[3:7] = [float(q[3]), *map(float, q[:3])]
                        target[7] = 0.0
                    elif smoke_index == 2:
                        target[7] = 1.0
                    elif smoke_index == 3:
                        target[7] = 0.0
                    else:
                        target[0] += .1
                        target[7] = 1.0
                    last_tool = {"name": "execute_eef_target", "arguments": {"target": target, "steps": 5}}
                    smoke_index += 1
                else:
                    t = time.monotonic()
                    try:
                        last_tool = client.predict(model_request(obs, steps, geometry))
                    finally:
                        timings["model_pipeline_s"] += time.monotonic() - t
                    calls += 1
                log("model_tool" if not smoke else "scripted_tool", tool=last_tool)
                print(json.dumps({"event": "tool", "call": calls, "step": steps, "tool": last_tool}), flush=True)
                check_stop()
                state("planning")
                t = time.monotonic()
                args = last_tool["arguments"]
                executor = None
                try:
                    executor = ShortTargetExecutor(obs.state, args["target"], steps=args["steps"])
                except TargetLimitError as exc:
                    last_feedback = rejected_feedback(obs.state, args["target"], args["steps"], str(exc))
                timings["local_planning_s"] += time.monotonic() - t
                moves += 1
                start_steps = steps
                while executor is not None and not executor.done and not sim_status.get("terminated"):
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
                if executor is not None:
                    if not executor.done:
                        executor.interrupt(detail="Host ended segment before the selected steps completed")
                    last_feedback = executor.feedback(obs.state)
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
            if executor is not None and not executor.done and obs is not None:
                executor.interrupt(detail="Operator interrupted the short target segment")
                last_feedback = executor.feedback(obs.state)
                summaries.append(last_feedback)
                log("tool_result", feedback=last_feedback)
                if client:
                    client.record_result(last_feedback)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=DISTANCE_TASK_KEYS, default="bowl_plate")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--sim-smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.sim_smoke and args.store_dir:
        parser.error("Offline smoke never accesses credentials")
    from gpt_field.chatgpt_auth import SubscriptionCredentials
    token_provider = None if args.sim_smoke else SubscriptionCredentials(store_dir=args.store_dir).access_token
    result = execute(args.output, task_key=args.task, episode_index=args.episode_index,
        token_provider=token_provider, smoke=args.sim_smoke, video=not args.no_video)
    return 0 if result["evaluation_completed"] or result["diagnostic_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
