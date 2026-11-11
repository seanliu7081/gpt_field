"""One autonomous episode with an evaluator isolated from policy inputs.

The runner never uses raw task success to terminate control. A policy finish or
blocked claim, an operator stop, or an infrastructure error ends the episode.
The optional ten-step physical check holds the last commanded gripper state and
pose; it does not release, lift, or repair a terminal state.
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
import math
from pathlib import Path
import time

from ..agent_speedup_v2_sim import initial_observation_sha256
from ..agent_speedup_v1_video import json_safe
from ..contact_skills import HoldingActionAdapter, StepView

VERSION = "foundation_vision_harness_V1"
OPERATOR_STOP_FILE = "operator_stop.request.json"


def _write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False)+"\n")


def _terminal(policy):
    value = getattr(policy, "terminal", None)
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("kind") not in ("finish", "blocked"):
        raise ValueError("policy terminal must be a finish or blocked claim")
    return deepcopy(value)


def _raw(status, fallback=False):
    return bool(status.get("libero_success", status.get("success", fallback)))


def _truth(sim, label, directory):
    # An interrupted JSONL request leaves an unread reply in the pipe. Never
    # send a new evaluator request into that stream.
    if getattr(sim, "_in_flight", False):
        result = {"available": False, "error": "simulator request interrupted"}
    else:
        try:
            result = sim.truth()
            if not isinstance(result, dict):
                raise ValueError("truth response must be an object")
        except Exception as error:
            result = {"available": False, "error": f"{type(error).__name__}: {error}"}
    _write(directory / f"truth_{label}.json", result)
    return result


def execute_episode(output, *, task, episode_index, sim_factory, client_factory,
                    policy_factory, expected_initial_observation_sha256,
                    expected_sha_source="argument", recorder_factory=None,
                    simulator_start_context=None, clock=time.monotonic):
    """Run exactly one task; factories make the loop CPU/mock-testable.

    ``sim_factory()`` returns a public-observation simulator configured with
    stop_on_success=False. ``client_factory(task_prompt=..., log_dir=...)`` and
    ``policy_factory(client)`` supply the new V1 implementation. Recorder
    factories accept an output path and episode_id and return AsyncVideoRecorder.
    No rollout caps or retry loop are present.
    """
    directory = Path(output).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    stop_path = directory / OPERATOR_STOP_FILE
    overall_started = clock()
    sim = client = policy = recorder = observation = None
    control_started = control_stopped = None
    steps = withheld = hold_steps = 0
    status, error, terminal = "error", None, None
    first_raw = None
    raw_at_terminal = False
    truth_at_terminal = truth_after_hold = None
    hold_started = hold_stopped = None
    cleanup_errors = []
    stats = {}
    client_stats = {}
    video_stats = None
    trace = (directory / "trace.jsonl").open("x")

    def append(event, **fields):
        trace.write(json.dumps(json_safe({"event": event, "step": steps,
                    "online_wall_s": None if control_started is None else
                    (control_stopped if control_stopped is not None else clock())-control_started,
                    **fields}), ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")
        trace.flush()

    def run_state(phase):
        _write(directory / "run_state.json", {"version": VERSION, "phase": phase,
            "steps": steps, "online_wall_s": None if control_started is None else
            (control_stopped if control_stopped is not None else clock())-control_started,
            "first_raw_success": first_raw, "operator_stop_file": str(stop_path),
            "terminal": terminal, "error": error})

    try:
        run_state("starting")
        if stop_path.is_file():
            raise KeyboardInterrupt("operator stop before simulator startup")
        with (simulator_start_context() if simulator_start_context else nullcontext()):
            sim = sim_factory()
            if getattr(getattr(sim, "config", None), "stop_on_success", False):
                raise ValueError("autonomous V1 requires stop_on_success=False")
            sim.start()
        if (getattr(sim, "metadata", {}) or {}).get("oracle_enabled") is True:
            raise ValueError("live policy simulator must not enable the oracle")
        tasks = sim.list_tasks()
        selected = [item for item in tasks if item.get("task_id") == task.task_id]
        if len(selected) != 1 or selected[0].get("prompt") != task.prompt:
            raise ValueError("simulator task does not match selected public task prompt")
        _write(directory / "simulator.json", {"metadata": getattr(sim, "metadata", {}), "tasks": tasks})
        observation = sim.reset(task.task_id, episode_index)
        if observation.prompt != task.prompt:
            raise ValueError("reset observation has a different task prompt")
        initial = observation.to_dict()
        observed_sha = initial_observation_sha256(initial)
        _write(directory / "initial_observation.json", initial)
        _write(directory / "initial_observation_check.json", {"sha256": observed_sha,
            "expected_sha256": expected_initial_observation_sha256,
            "expected_source": expected_sha_source,
            "matched": observed_sha == expected_initial_observation_sha256})
        if not expected_initial_observation_sha256 or observed_sha != expected_initial_observation_sha256:
            raise ValueError("reset observation differs from the required frozen reset; no model call made")
        client = client_factory(task_prompt=observation.prompt, log_dir=directory / "requests")
        policy = policy_factory(client)
        if recorder_factory is not None:
            recorder = recorder_factory(directory / "episode.mp4", episode_id=VERSION)
        control_started = clock()
        append("reset", task=task.key, episode_index=episode_index, initial_sha256=observed_sha)
        if recorder is not None:
            recorder.submit(observation, step=0, planning=policy.planning_snapshot(), event="initial")
        if _raw(getattr(sim, "status", {})):
            first_raw = {"step": 0, "online_wall_s": clock()-control_started}
            append("raw_success_first_reached", **first_raw)
        while True:
            if stop_path.is_file():
                raise KeyboardInterrupt("operator stop before deciding")
            terminal = _terminal(policy)
            if terminal is not None:
                status = terminal["kind"]
                append("policy_terminal", claim=terminal)
                break
            run_state("deciding")
            decision_started = clock()
            decision = policy.act(observation)
            decision_latency = clock()-decision_started
            # No action produced by a late inference may pass a stop request.
            if stop_path.is_file():
                withheld += 1
                raise KeyboardInterrupt("operator stop during decision; action withheld")
            terminal = _terminal(policy)
            if terminal is not None:
                status = terminal["kind"]
                append("policy_terminal", claim=terminal, decision_latency_s=decision_latency,
                       action_withheld=True)
                break
            action = tuple(float(value) for value in decision.action)
            if len(action) != 7 or not all(math.isfinite(v) and abs(v) <= 1.0 for v in action):
                raise ValueError("policy action must contain seven finite normalized controls")
            if recorder is not None:
                recorder.submit(observation, step=steps, planning=policy.planning_snapshot(),
                                decision=decision, decision_latency_s=decision_latency, event="decision")
            if stop_path.is_file():
                withheld += 1
                raise KeyboardInterrupt("operator stop before action dispatch; action withheld")
            before = observation
            dispatch_started = clock()
            observation, step_success = sim.step(action)
            dispatch_latency = clock()-dispatch_started
            steps += 1
            # Only public RGB, robot sensors and camera geometry enter policy.
            policy.observe_transition(before, decision, observation)
            evaluator_status = deepcopy(getattr(sim, "status", {}) or {})
            raw = _raw(evaluator_status, step_success)
            if raw and first_raw is None:
                first_raw = {"step": steps, "online_wall_s": clock()-control_started}
                append("raw_success_first_reached", **first_raw)
            append("action", action=action, before_state=before.state, after_state=observation.state,
                   decision=decision.diagnostics, decision_latency_s=decision_latency,
                   simulator_step_wall_s=dispatch_latency, evaluator={"raw_success": raw,
                   "collision": evaluator_status.get("collision")})
            if recorder is not None:
                recorder.submit(observation, step=steps, planning=policy.planning_snapshot())
            if evaluator_status.get("terminated"):
                raise RuntimeError("simulator terminated before an autonomous policy terminal claim")
    except KeyboardInterrupt as exc:
        status, error = "user_stop", str(exc) or "operator interrupt"
        append("operator_stop", reason=error, withheld_decisions=withheld)
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
        append("error", reason=error)
    finally:
        # Online timing stops only after the terminal event and its log writes.
        # Neither video overhead nor policy/perception waiting is subtracted.
        if control_started is not None:
            control_stopped = clock()
        try:
            if sim is not None and observation is not None:
                raw_at_terminal = _raw(getattr(sim, "status", {}))
                _write(directory / "terminal_observation.json", observation.to_dict())
                _write(directory / "terminal_evaluator_status.json", getattr(sim, "status", {}))
                truth_at_terminal = _truth(sim, "terminal", directory)
                if status == "finish" and not stop_path.is_file() and not getattr(sim, "_in_flight", False):
                    hold_started = clock()
                    anchor = StepView(observation)
                    command = policy.gripper_command
                    if command not in ("open", "close"):
                        raise ValueError("terminal gripper command must be explicit")
                    adapter = HoldingActionAdapter()
                    for hold_index in range(10):
                        if stop_path.is_file():
                            raise KeyboardInterrupt("operator stop during evaluator hold")
                        current = StepView(observation)
                        delta = tuple(a-b for a, b in zip(anchor.position, current.position))
                        hold = adapter.adapt(delta, anchor.quaternion_xyzw, command, current.state)
                        observation, _ = sim.step(hold)
                        hold_steps += 1
                        append("evaluator_same_gripper_hold", hold_index=hold_index+1,
                               action=hold, gripper_command=command,
                               raw_success=_raw(getattr(sim, "status", {})))
                    truth_after_hold = _truth(sim, "after_same_gripper_hold", directory)
                    _write(directory / "after_hold_observation.json", observation.to_dict())
        except KeyboardInterrupt as exc:
            status, error = "user_stop", str(exc)
        except Exception as exc:
            status, error = "error", f"terminal evaluation failed: {type(exc).__name__}: {exc}"
        finally:
            if hold_started is not None:
                hold_stopped = clock()
        if policy is not None:
            try:
                stats = policy.agent_stats()
            except Exception as exc:
                cleanup_errors.append(f"policy stats: {type(exc).__name__}: {exc}")
        if client is not None:
            for name in ("history_stats", "budget_snapshot"):
                read = getattr(client, name, None)
                if callable(read):
                    try:
                        client_stats[name] = read()
                    except Exception as exc:
                        cleanup_errors.append(f"client {name}: {type(exc).__name__}: {exc}")
        for label, resource in (("policy", policy), ("client", client), ("simulator", sim)):
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    cleanup_errors.append(f"{label}: {type(exc).__name__}: {exc}")
        if recorder is not None:
            try:
                video_stats = recorder.close(status=status, error=error)
            except Exception as exc:
                cleanup_errors.append(f"recorder: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            status, error = "error", "resource finalization failed: "+"; ".join(cleanup_errors)
        try:
            run_state(status)
        finally:
            trace.close()

    strict_terminal = isinstance(truth_at_terminal, dict) and truth_at_terminal.get("strict_conjunction") is True
    strict_hold = isinstance(truth_after_hold, dict) and truth_after_hold.get("strict_conjunction") is True
    raw_hold = isinstance(truth_after_hold, dict) and truth_after_hold.get("check_success") is True
    # A visible finish claim is not an evaluator pass. Require the independent
    # conjunction at the claimed finish AND after the unchanged-gripper hold.
    strict_success = status == "finish" and raw_at_terminal and strict_terminal and raw_hold and strict_hold
    online_wall = None if control_started is None else control_stopped-control_started
    hold_wall = 0. if hold_started is None else hold_stopped-hold_started
    finished = clock()
    result = {"version": VERSION, "status": status, "error": error, "task": task.key,
        "episode_index": episode_index, "steps": steps, "terminal_claim": terminal,
        "withheld_decisions_after_stop": withheld, "first_raw_success": first_raw,
        "raw_success_at_terminal": raw_at_terminal, "strict_success_at_terminal": strict_terminal,
        "raw_success_after_same_gripper_hold": raw_hold,
        "strict_success_after_same_gripper_hold": strict_hold,
        "same_gripper_hold_steps": hold_steps, "strict_success": bool(strict_success),
        "completion_time_eligible": bool(strict_success),
        "completion_wall_s": hold_stopped-control_started if strict_success else None,
        "online_wall_s": online_wall, "evaluation_hold_wall_s": hold_wall,
        "startup_wall_s": (control_started if control_started is not None else finished)-overall_started,
        "postprocessing_wall_s": 0. if control_stopped is None else finished-control_stopped,
        "total_wall_s": finished-overall_started, "video_time_subtracted": False,
        "agent_stats": stats, "client_stats": client_stats,
        "video": video_stats, "cleanup_errors": cleanup_errors,
        "automatic_retry": False, "raw_success_terminates_control": False,
        "evaluation_hold_changes_gripper": False, "operator_stop_file": str(stop_path)}
    _write(directory / "summary.json", result)
    return result
