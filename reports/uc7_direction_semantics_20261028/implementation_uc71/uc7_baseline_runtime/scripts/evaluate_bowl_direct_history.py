"""Fresh Astra direct-action + history baseline on the requested LIBERO bowl task.

Live execution requires its own explicit --max-cost-usd. No action, API-call,
or wall-clock episode limit is imposed. --offline-smoke uses real LIBERO with
exactly two scripted mock responses and is unscored plumbing validation.
--preflight only builds a synthetic-observation payload, without API or simulator.
Existing runs are never resumed or overwritten by this runner.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np

from gpt_field.astra import APIError, AstraConfig, BudgetExceeded, urllib_transport
from gpt_field.direct_history import DirectHistoryAstraClient, DirectHistoryPolicy
from gpt_field.direct_history_video import DirectHistoryVideoRecorder
from gpt_field.experiment import digest, source_manifest, write_json
from gpt_field.libero import LiberoConfig, SimProcess
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.schema import GridSpec


ROOT = Path(__file__).resolve().parents[1]
TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
DEFAULT_SIM_PYTHON = "/workspace/venvs/libero-plus/bin/python"
DEFAULT_LIBERO_ROOT = "/workspace/heading_flow_pub/third_party/LIBERO"
MOCK_ACTION_COUNT = 10
STOPPING_RULE = ["LIBERO success", "user stop", "API budget reached", "infrastructure error"]


def redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, (list, tuple)):
        return [redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {redact(key, secret): redact(item, secret) for key, item in value.items()}
    return value


def redacting_transport(secret, base_transport=None):
    base_transport = base_transport or urllib_transport

    def send(endpoint, payload, headers, timeout_s):
        try:
            return redact(base_transport(endpoint, payload, headers, timeout_s), secret)
        except APIError as exc:
            raise APIError(redact(str(exc), secret),
                           response_body=redact(exc.response_body, secret)) from None
        except Exception as exc:
            raise APIError(redact(f"Transport {type(exc).__name__}: {exc}", secret)) from None

    return send


def configuration(max_cost_usd=None):
    """Explicit rates retain the field run's accounting convention, not its cap."""
    return AstraConfig(model="gpt-6-astra", max_output_tokens=8192, timeout_s=180,
                       max_api_calls=None, max_wall_time_s=None, max_cost_usd=max_cost_usd,
                       input_price_per_million=10, cached_input_price_per_million=1,
                       cache_write_price_per_million=12.5, output_price_per_million=50)


def validate_config(config, sim_config, *, offline_smoke=False, preflight=False):
    if config.model != "gpt-6-astra":
        raise ValueError("This baseline requires gpt-6-astra")
    if config.max_api_calls is not None or config.max_wall_time_s is not None:
        raise ValueError("The live baseline has no total API-call or wall-time limit")
    if config.max_output_tokens != 8192 or config.timeout_s != 180:
        raise ValueError("Per-request limits must be 8192 output tokens and 180 seconds")
    rates = (config.input_price_per_million, config.cached_input_price_per_million,
             config.cache_write_price_per_million, config.output_price_per_million)
    if rates != (10, 1, 12.5, 50):
        raise ValueError("Accounting rates must be input=10, cached=1, cache-write=12.5, output=50")
    cost = config.max_cost_usd
    if cost is not None and (isinstance(cost, bool) or not isinstance(cost, (int, float))
                             or not math.isfinite(cost) or cost <= 0):
        raise ValueError("max_cost_usd must be finite and positive")
    if not (offline_smoke or preflight) and cost is None:
        raise ValueError("A live baseline requires its own explicit max_cost_usd")
    if offline_smoke and cost is not None:
        raise ValueError("Offline smoke does not accept or consume an API budget")
    if (sim_config.suite != "libero_spatial" or sim_config.seed != 7
            or sim_config.max_steps is not None or sim_config.settle_steps != 10
            or sim_config.render_size != 256 or sim_config.image_size != 224):
        raise ValueError("Use official libero_spatial, seed 7, no action limit, 10 settling steps, 256/224 images")


class ScriptedSmokeTransport:
    """No-network fixture: move slightly while open, then close without moving.

    These commands only exercise plumbing. They do not attempt to solve the task.
    The third request is forbidden, even if the runner's diagnostic stop regresses.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls += 1
        if self.calls > 2:
            raise RuntimeError("Offline smoke fixture only contains two mock responses")
        action = [0.02, 0, 0, 0, 0, 0, -1] if self.calls == 1 else [0, 0, 0, 0, 0, 0, 1]
        return {"id": f"offline-mock-{self.calls}", "status": "completed", "offline_mock": True,
                "output": [{"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": json.dumps({"actions": [action[:] for _ in range(5)]})}]}],
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}}


def preflight(config, sim_config, output=None):
    """Read-only preview; synthetic camera data cannot be mistaken for evaluation."""
    validate_config(config, sim_config, preflight=True)
    if output is not None and Path(output).exists():
        raise FileExistsError("Fresh baseline requires a new output directory")
    camera = image_to_data_url(np.zeros((224, 224, 3), dtype=np.uint8), rotate=False)
    observation = PolicyObservation(camera, camera, (0, 0, 1, 0, 0, 0, 0.04, -0.04), TASK)
    client = DirectHistoryAstraClient(config, api_key="")
    payload = client.build_payload(observation.to_dict(), GridSpec(), mode="direct")
    executable = Path(sim_config.python_executable or "")
    checkout = Path(sim_config.libero_root or "")
    result = {"status": "preflight_only", "task": TASK, "suite": sim_config.suite,
              "task_id": 0, "episode_index": 0, "seed": 7,
              "synthetic_observation": True, "api_calls": 0, "simulator_started": False,
              "credentials_checked": False, "files_written": False,
              "simulator": asdict(sim_config), "api": asdict(config),
              "sim_python_exists": executable.is_file(),
              "libero_checkout_exists": (checkout / "libero/libero/benchmark/__init__.py").is_file(),
              "payload": {"model": payload["model"], "service_tier": payload["service_tier"],
                          "store": payload["store"], "max_output_tokens": payload["max_output_tokens"],
                          "instructions": payload["instructions"],
                          "text": payload["text"], "input_message_count": len(payload["input"]),
                          "initial_image_count": sum(part.get("type") == "input_image"
                              for message in payload["input"] for part in message.get("content", [])
                              if isinstance(part, dict))},
              "reservation_per_call_usd": client.budget_snapshot()["reservation_per_call_usd"],
              "live_budget_required": config.max_cost_usd is None,
              "old_field_episode_budget_inherited": False}
    if config.max_cost_usd is not None:
        result["budget_can_reserve_first_request"] = (
            config.max_cost_usd >= result["reservation_per_call_usd"])
    return result


@contextmanager
def termination_as_interrupt():
    """Finalize on SIGTERM as on Ctrl-C; SIGKILL cannot be handled by any runner."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _append(stream, value):
    stream.write(json.dumps(value, allow_nan=False) + "\n")
    stream.flush()


def execute_episode(output, *, config, sim_config, api_key=None, offline_smoke=False,
                    sim_factory=None, recorder_factory=None, transport=None):
    """Execute one fresh initial-state-zero episode; never continue old field runs.

    Factories/transport are test injection points. Production CLI always uses the
    real simulator and recorder; offline mode always selects its fixed mock API.
    """
    validate_config(config, sim_config, offline_smoke=offline_smoke)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Fresh baseline requires a new output directory; resume is unsupported")
    # Simulator inherits neither the live credential nor any old episode budget.
    os.environ.pop("OPENAI_API_KEY", None)
    secret = None if offline_smoke else api_key
    if offline_smoke:
        if transport is not None:
            raise ValueError("Offline smoke always uses its fixed scripted transport")
        api_key, transport = "OFFLINE-MOCK-NOT-A-CREDENTIAL", ScriptedSmokeTransport()
    else:
        transport = redacting_transport(secret, transport)
    client = DirectHistoryAstraClient(config, api_key=api_key, transport=transport,
                                     log_dir=output / "api")
    client.preflight()
    policy = DirectHistoryPolicy(client)
    sim_factory = sim_factory or SimProcess
    recorder_factory = recorder_factory or DirectHistoryVideoRecorder
    output.mkdir(parents=True, exist_ok=False)
    mode = "offline_mock_smoke" if offline_smoke else "live_astra_direct_history"
    episode_id = "libero_spatial_t00_i00_direct_history"
    identity = {"kind": "fresh_direct_history_baseline", "mode": mode,
                "task": TASK, "task_id": 0, "episode_index": 0,
                "simulator": asdict(sim_config), "api": asdict(config),
                "software": source_manifest(),
                "runner": {"path": "scripts/evaluate_bowl_direct_history.py",
                           "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                "policy": {"class": "gpt_field.direct_history.DirectHistoryPolicy",
                           "method": "direct_history", "update_steps": 5,
                           "recent_image_observations": 2,
                           "initial_history_settings": client.history_stats(),
                           "history": "all observation/feedback text and ordered accepted Responses output "
                                      "items, including IDs, phase, and opaque encrypted reasoning",
                           "field_generation": False, "local_field_planning": False,
                           "success_feedback_to_policy": False},
                "stopping_rule": (["ten diagnostic actions", "user stop", "infrastructure error"]
                                  if offline_smoke else STOPPING_RULE),
                "diagnostic_action_count": MOCK_ACTION_COUNT if offline_smoke else None,
                "max_policy_steps": None,
                "budget_scope": "this fresh run only; no prior budget is inherited",
                "evaluation_scored": not offline_smoke}
    write_json(output / "manifest.json", {"identity_hash": digest(identity), "configuration": identity,
        "created_utc": datetime.now(timezone.utc).isoformat(), "simulation_pauses_for_api": True})
    status, error, steps, close_actions, open_actions, neutral_actions = "running", None, 0, 0, 0, 0
    success_seen, pending_action, path_length = False, None, 0.0
    started = time.monotonic()
    recorder, sim, video_name = None, None, "episode.mp4"
    feedback_count, valid_chunks = 0, 0
    cleanup_errors = []

    def state(stage):
        write_json(output / "run_state.json", redact({"status": status, "stage": stage,
            "mode": mode, "step": steps, "pending_action_index": pending_action,
            "budget": client.budget_snapshot(), "error": error}, secret))

    state("starting")
    with termination_as_interrupt():
        try:
            sim = sim_factory(sim_config)
            sim.start()
            tasks = sim.list_tasks()
            if not tasks or tasks[0].get("task_id") != 0 or tasks[0].get("prompt") != TASK:
                raise ValueError("Official task zero does not match the requested bowl task")
            write_json(output / "simulator.json", {"metadata": sim.metadata, "tasks": tasks})
            obs = sim.reset(0, 0)
            if obs.prompt != TASK:
                raise ValueError("Reset observation does not match the requested bowl task")
            policy.reset()
            recorder = recorder_factory(output / video_name, episode_id=episode_id,
                                        offline_smoke=offline_smoke)
            recorder.observe(obs, step=0)
            with (output / "trace.jsonl").open("w") as log:
                _append(log, {"event": "reset", "observation": obs.to_dict(),
                              "sim_status": dict(sim.status), "mode": mode})
                while True:
                    if offline_smoke and steps == MOCK_ACTION_COUNT:
                        status = "diagnostic_complete"
                        break
                    if sim.status.get("success"):
                        success_seen = True
                        status = "diagnostic_sim_terminated" if offline_smoke else "success"
                        break
                    if sim.status.get("terminated"):
                        raise RuntimeError("Simulator terminated without LIBERO success and without an action limit")
                    if obs.prompt != TASK:
                        raise ValueError("Policy observation task changed during episode")
                    state("deciding")
                    decision_started = time.monotonic()
                    decision = policy.act(obs)
                    latency = time.monotonic() - decision_started
                    valid_chunks += int(bool(decision.diagnostics.get("refreshed")))
                    _append(log, {"event": "action", "step": steps, "state": list(obs.state),
                                  "action": list(decision.action), "diagnostics": decision.diagnostics,
                                  "decision_latency_s": latency, "mode": mode})
                    recorder.decision(obs, decision, decision_latency_s=latency)
                    pending_action = steps
                    state("executing")
                    before = obs
                    obs, success = sim.step(decision.action)
                    executed_step = steps
                    steps += 1
                    pending_action = None
                    success_seen = success_seen or success
                    close_actions += int(decision.action[6] > 0)
                    open_actions += int(decision.action[6] < 0)
                    neutral_actions += int(decision.action[6] == 0)
                    path_length += float(np.linalg.norm(np.asarray(obs.state[:3]) - np.asarray(before.state[:3])))
                    # Persist the simulator reply before any policy feedback/video work.
                    # Success/predicates remain exclusively in evaluator artifacts.
                    _append(log, {"event": "executed", "step": executed_step,
                                  "next_state": list(obs.state), "observation": obs.to_dict(),
                                  "action": list(decision.action), "success": success,
                                  "sim_status": dict(sim.status), "mode": mode})
                    feedback = policy.observe_transition(before, decision, obs)
                    feedback_count += 1
                    _append(log, {"event": "feedback", "step": executed_step, "feedback": feedback})
                    recorder.executed(obs, step=executed_step, success=success,
                                      sim_status=dict(sim.status), feedback=feedback)
                    state("observed")
                    print(json.dumps({"event": "executed", "mode": mode, "steps": steps,
                                      "close_actions": close_actions, "success_observed": success,
                                      "api_attempts": client.budget_snapshot()["api_calls"]}), flush=True)
        except KeyboardInterrupt:
            status, error = "user_stop", "Interrupted before successful completion"
        except BudgetExceeded as exc:
            status, error = "budget_exceeded", redact(str(exc), secret)
        except Exception as exc:
            status, error = "infrastructure_error", redact(f"{type(exc).__name__}: {exc}", secret)
        finally:
            if recorder is not None:
                try:
                    recorder.close(status=status, error=error)
                except Exception as exc:
                    cleanup_errors.append(redact(f"Video finalization {type(exc).__name__}: {exc}", secret))
            if sim is not None:
                try:
                    sim.close()
                except Exception as exc:
                    cleanup_errors.append(redact(f"Simulator close {type(exc).__name__}: {exc}", secret))
            if cleanup_errors:
                error = "; ".join(filter(None, [error, *cleanup_errors]))
                if status in ("success", "diagnostic_complete"):
                    status = "infrastructure_error"
            completed = status == "success" and not offline_smoke
            summary = redact({"status": status, "mode": mode, "task": TASK, "suite": "libero_spatial",
                "task_id": 0, "episode_index": 0, "seed": 7, "requested_episodes": 1,
                "steps": steps, "validated_action_chunks": valid_chunks,
                "measured_feedback_count": feedback_count,
                "gripper_close_actions": close_actions, "gripper_open_actions": open_actions,
                "gripper_neutral_actions": neutral_actions, "path_length_m": path_length,
                "pending_action_index": pending_action, "libero_success_observed": success_seen,
                "libero_success": True if completed else None,
                "evaluation_completed": completed, "success_rate": 1.0 if completed else None,
                "scored_episodes": int(completed), "incomplete_episodes": int(not completed),
                "api_calls": 0 if offline_smoke else client.budget_snapshot()["api_calls"],
                "mock_response_calls": client.budget_snapshot()["api_calls"] if offline_smoke else 0,
                "budget": client.budget_snapshot(), "max_cost_usd": config.max_cost_usd,
                "budget_ledger_kind": "mock_zero_usage" if offline_smoke else "live_usage_and_reservations",
                "budget_scope": "this fresh run only", "elapsed_s": time.monotonic() - started,
                "video": video_name if (output / video_name).is_file() else None,
                "trace": "trace.jsonl", "error": error, "output": str(output),
                "interpretation": ("Scripted mock responses with real simulation; plumbing check only. "
                    "Not Astra-generated actions and not a task-performance evaluation." if offline_smoke else
                    "One unbounded episode; budget, interruption, and infrastructure stops remain unscored.")}, secret)
            write_json(output / "runner_summary.json", summary)
            write_json(output / "summary.json", summary)
            with (output / "episodes.jsonl").open("w") as results:
                _append(results, summary)
            state("finished")
    print(json.dumps({"event": "finished", **summary}), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Fresh output directory; required except for --preflight")
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--libero-root", default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--max-cost-usd", type=float,
                        help="Explicit cap for this NEW live run; never inherited from the field episode")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline-smoke", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if not args.preflight and not args.output:
        parser.error("--output is required for execution")
    if not (args.offline_smoke or args.preflight) and args.max_cost_usd is None:
        parser.error("live execution requires --max-cost-usd for this new baseline")
    if args.offline_smoke and args.max_cost_usd is not None:
        parser.error("--offline-smoke does not accept an API budget")
    config = configuration(args.max_cost_usd)
    # abspath retains the virtualenv executable symlink; resolve() selects base Python.
    sim_config = LiberoConfig(suite="libero_spatial", seed=7, max_steps=None,
        libero_root=str(Path(args.libero_root).expanduser().resolve()),
        python_executable=os.path.abspath(os.path.expanduser(args.sim_python)))
    if args.preflight:
        print(json.dumps(preflight(config, sim_config, args.output), indent=2), flush=True)
        return 0
    validate_config(config, sim_config, offline_smoke=args.offline_smoke)
    if Path(args.output).exists():
        parser.error("output already exists; baseline runs require a fresh directory")
    key = None
    if not args.offline_smoke:
        key = os.environ.pop("OPENAI_API_KEY", None) or getpass.getpass("API key (hidden, memory only): ").strip()
    result = execute_episode(args.output, config=config, sim_config=sim_config,
                             api_key=key, offline_smoke=args.offline_smoke)
    return (0 if result["status"] in ("success", "diagnostic_complete") else
            130 if result["status"] == "user_stop" else 3 if result["status"] == "budget_exceeded" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
