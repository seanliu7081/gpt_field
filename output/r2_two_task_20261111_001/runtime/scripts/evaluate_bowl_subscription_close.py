"""Run the bowl field policy on a ChatGPT subscription until one close action.

Fresh LIBERO-Spatial task 0, initial state 0, seed 7. No action, inference-call,
or episode wall-time horizon. Stop only after the simulator acknowledges the
first actual action[6] > 0 and its measured feedback is saved. This diagnostic
is always unscored; a close command does not establish full closure or a grasp.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from gpt_field.astra import APIError, AstraConfig, BudgetExceeded, MissingCredentials, SchemaError
from gpt_field.chatgpt_plan import ChatGPTPlanFieldClient
from gpt_field.experiment import digest, source_manifest, write_json
from gpt_field.field_history import FieldHistoryAstraClient, FieldHistoryPolicy
from gpt_field.field_history_video import FieldHistoryVideoRecorder
from gpt_field.libero import LiberoConfig, SimProcess
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.planner import PlannerConfig
from gpt_field.schema import GridSpec


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_frozen_field_history_runner",
    ROOT / "scripts/evaluate_bowl_field_history.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
TASK = _base.TASK
DEFAULT_SIM_PYTHON = _base.DEFAULT_SIM_PYTHON
DEFAULT_LIBERO_ROOT = _base.DEFAULT_LIBERO_ROOT
STOP_REASON = "first_gripper_close_executed"
STOPPING_RULE = [STOP_REASON, "LIBERO success before close", "user stop",
                 "subscription authorization or quota error", "infrastructure error"]


def configuration():
    return AstraConfig(model="gpt-6-astra", timeout_s=180,
                       max_api_calls=None, max_wall_time_s=None, max_cost_usd=None)


def validate_config(config, sim_config):
    if config.model != "gpt-6-astra":
        raise ValueError("This diagnostic requires exactly gpt-6-astra")
    if config.max_api_calls is not None or config.max_wall_time_s is not None:
        raise ValueError("This diagnostic has no inference-call or episode wall-time horizon")
    if any(getattr(config, name) is not None for name in (
        "max_cost_usd", "input_price_per_million", "output_price_per_million",
        "cached_input_price_per_million", "cache_write_price_per_million",
    )):
        raise ValueError("Subscription usage must not use API-key prices or dollar reservations")
    if config.timeout_s != 180:
        raise ValueError("Keep the existing 180-second per-request timeout")
    if (sim_config.suite != "libero_spatial" or sim_config.seed != 7
            or sim_config.max_steps is not None or sim_config.settle_steps != 10
            or sim_config.render_size != 256 or sim_config.image_size != 224):
        raise ValueError("Use official libero_spatial, seed 7, no action limit, 10 settling steps, 256/224 images")


def preflight(config, sim_config, output=None):
    """Inspect the exact subscription payload; no auth, network, files or sim."""
    validate_config(config, sim_config)
    if output is not None and Path(output).exists():
        raise FileExistsError("Fresh diagnostic requires a new output directory")
    camera = image_to_data_url(np.zeros((224, 224, 3), dtype=np.uint8), rotate=False)
    observation = PolicyObservation(camera, camera, (0, 0, 1, 0, 0, 0, .04, -.04), TASK)
    client = ChatGPTPlanFieldClient(config, token_provider=lambda: None)
    payload = client.build_payload(observation.to_dict())
    return {"status": "preflight_only", "task": TASK, "task_id": 0, "episode_index": 0,
        "billing": "chatgpt_subscription", "simulator": asdict(sim_config), "api": asdict(config),
        "synthetic_observation": True, "api_calls": 0, "api_key_calls": 0,
        "credentials_checked": False, "simulator_started": False, "files_written": False,
        "sim_python_exists": Path(sim_config.python_executable or "").is_file(),
        "libero_checkout_exists": (Path(sim_config.libero_root or "") / "libero/libero/benchmark/__init__.py").is_file(),
        "payload": {key: payload[key] for key in ("model", "store", "stream", "instructions", "text")},
        "max_policy_steps": None, "max_api_calls": None, "max_wall_time_s": None,
        "output_token_limit_sent": None, "api_key_fallback": False, "model_fallback": False,
        "stopping_rule": STOPPING_RULE, "evaluation_scored": False,
        "policy": {"method": "field_history", "grid": GridSpec().to_dict(),
            "planner": asdict(PlannerConfig()), "field_update_steps": 5,
            "recent_image_observations": 2, "success_feedback_to_policy": False,
            "reward_feedback_to_policy": False}}


class _SmokeClient(ChatGPTPlanFieldClient):
    """Identical policy interface with explicitly labelled, zero-network fixtures."""

    def budget_snapshot(self):
        usage = super().budget_snapshot()
        return {**usage, "billing": "offline_mock", "api_calls": 0,
                "subscription_inference_calls": 0, "mock_response_calls": self._api_calls}

    def _write_log(self, path, data):
        if isinstance(data, dict):
            data = {**data, "billing": "offline_mock", "offline_smoke": True,
                    "transport_version": "scripted-no-network-fixture"}
        FieldHistoryAstraClient._write_log(self, path, data)


@contextmanager
def _simulator_without_credentials():
    names = ("OPENAI_API_KEY", "ACCESS_TOKEN", "OPENAI_ACCESS_TOKEN", "CHATGPT_ACCESS_TOKEN")
    previous = {name: os.environ.pop(name) for name in names if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(previous)


def _safe_error(exc, stage):
    result = {"type": type(exc).__name__, "stage": stage,
              "message": "Diagnostic stopped; no retry or API-key fallback"}
    status, code = getattr(exc, "http_status", None), getattr(exc, "error_code", None)
    if type(status) is int and 100 <= status <= 599:
        result["http_status"] = status
    if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", code):
        result["error_code"] = code
    return result


def execute_episode(output, *, config, sim_config, token_provider=None, offline_smoke=False,
                    sim_factory=None, recorder_factory=None, transport=None):
    validate_config(config, sim_config)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Fresh diagnostic requires a new output directory; no resume")
    if offline_smoke:
        if transport is not None or token_provider is not None:
            raise ValueError("Offline smoke uses only its fixed scripted transport and no credentials")
        client = _SmokeClient(config, token_provider=lambda: "OFFLINE-OAUTH-FIXTURE",
            transport=_base.ScriptedSmokeTransport(), log_dir=output / "requests")
    else:
        if not callable(token_provider):
            raise MissingCredentials("Live execution requires a ChatGPT subscription token provider")
        client = ChatGPTPlanFieldClient(config, token_provider=token_provider,
            transport=transport, log_dir=output / "requests")
    policy = FieldHistoryPolicy(client)
    sim_factory, recorder_factory = sim_factory or SimProcess, recorder_factory or FieldHistoryVideoRecorder
    output.mkdir(parents=True, exist_ok=False)
    mode = "offline_mock_smoke" if offline_smoke else "live_chatgpt_subscription_field_history"
    identity = {"kind": "subscription_first_close_diagnostic", "mode": mode,
        "task": TASK, "task_id": 0, "episode_index": 0, "simulator": asdict(sim_config),
        "api": asdict(config), "billing": "offline_mock" if offline_smoke else "chatgpt_subscription",
        "software": source_manifest(), "runner": {"path": "scripts/evaluate_bowl_subscription_close.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "policy": {"class": "gpt_field.field_history.FieldHistoryPolicy", "method": "field_history",
            "update_steps": 5, "recent_image_observations": 2, "grid": policy.grid.to_dict(),
            "planner": asdict(policy.planner.config), "adapter_max_rotation_rad": policy.adapter.max_rotation_rad,
            "initial_history_settings": client.history_stats(), "local_plans_per_action": 1,
            "field_generation": True, "local_field_planning": True,
            "success_feedback_to_policy": False, "reward_feedback_to_policy": False,
            "object_state_feedback_to_policy": False},
        "stopping_rule": STOPPING_RULE, "max_policy_steps": None, "max_api_calls": None,
        "max_wall_time_s": None, "output_token_limit_sent": None,
        "api_key_fallback": False, "model_fallback": False, "evaluation_scored": False,
        "prior_budgets_inherited": False,
        "close_definition": "The first actual normalized action[6] > 0 acknowledged by sim.step; not proof of full closure or grasp"}
    write_json(output / "manifest.json", {"identity_hash": digest(identity), "configuration": identity,
        "created_utc": datetime.now(timezone.utc).isoformat(), "simulation_pauses_for_api": True})
    status, stage, error = "running", "starting", None
    steps = valid_fields = local_plans = feedback_count = close_actions = open_actions = neutral_actions = 0
    pending_action, close_event, success_seen, path_length = None, None, False, 0.0
    recorder, sim, cleanup_errors = None, None, []
    started = time.monotonic()
    video_name = "episode.mp4"

    def state(new_stage):
        nonlocal stage
        stage = new_stage
        write_json(output / "run_state.json", {"status": status, "stage": stage, "mode": mode,
            "step": steps, "pending_action_index": pending_action, "budget": client.budget_snapshot(),
            "close_event": close_event, "error": error})

    state("starting")
    with _base.termination_as_interrupt():
        try:
            with _simulator_without_credentials():
                sim = sim_factory(sim_config)
                sim.start()
            tasks = sim.list_tasks()
            if not tasks or tasks[0].get("task_id") != 0 or tasks[0].get("prompt") != TASK:
                raise ValueError("Official task zero does not match the bowl task")
            write_json(output / "simulator.json", {"metadata": sim.metadata, "tasks": tasks})
            obs = sim.reset(0, 0)
            if obs.prompt != TASK:
                raise ValueError("Reset observation has a different task")
            policy.reset()
            recorder = recorder_factory(output / video_name,
                episode_id="CHATGPT SUBSCRIPTION / libero_spatial_t00_i00 / FIRST CLOSE DIAGNOSTIC",
                offline_smoke=offline_smoke)
            recorder.observe(obs, step=0)
            with (output / "trace.jsonl").open("w") as log:
                _base._append(log, {"event": "reset", "observation": obs.to_dict(),
                    "sim_status": dict(sim.status), "mode": mode})
                while True:
                    if sim.status.get("success"):
                        success_seen, status = True, "libero_success_before_close"
                        break
                    if sim.status.get("terminated"):
                        raise RuntimeError("Simulator terminated before diagnostic completion")
                    if obs.prompt != TASK:
                        raise ValueError("Policy task changed during rollout")
                    state("deciding")
                    decision_started = time.monotonic()
                    decision = policy.act(obs)
                    latency = time.monotonic() - decision_started
                    valid_fields += int(bool(decision.diagnostics.get("refreshed")))
                    local_plans += int("plan" in decision.diagnostics)
                    _base._append(log, {"event": "action", "step": steps, "state": list(obs.state),
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
                    is_close = decision.action[6] > 0
                    close_actions += int(is_close)
                    open_actions += int(decision.action[6] < 0)
                    neutral_actions += int(decision.action[6] == 0)
                    path_length += float(np.linalg.norm(np.asarray(obs.state[:3]) - np.asarray(before.state[:3])))
                    _base._append(log, {"event": "executed", "step": executed_step,
                        "next_state": list(obs.state), "observation": obs.to_dict(),
                        "action": list(decision.action), "success": success,
                        "sim_status": dict(sim.status), "mode": mode})
                    if is_close:
                        close_event = {"action_index": executed_step, "action_ordinal": steps,
                            "normalized_gripper_command": decision.action[6],
                            "close_execution_acknowledged": True,
                            "gripper_joints_before": list(before.state[6:8]),
                            "gripper_joints_after": list(obs.state[6:8]),
                            "gripper_joint_delta": [after - prior for prior, after in zip(before.state[6:8], obs.state[6:8])],
                            "full_closure_verified": False, "grasp_verified": False}
                    feedback = policy.observe_transition(before, decision, obs)
                    feedback_count += 1
                    _base._append(log, {"event": "feedback", "step": executed_step, "feedback": feedback})
                    recorder.executed(obs, step=executed_step, success=success, sim_status=dict(sim.status),
                        feedback=feedback, history_stats=client.history_stats())
                    if is_close:
                        status = STOP_REASON
                    state("observed")
                    print(json.dumps({"event": "executed", "mode": mode, "steps": steps,
                        "close_actions": close_actions, "success_observed": success,
                        "api_attempts": client.budget_snapshot()["api_calls"]}), flush=True)
                    if is_close:
                        break  # No extra action, settling step, or model request after this point.
        except KeyboardInterrupt:
            status, error = "user_stop", {"type": "KeyboardInterrupt", "stage": stage,
                                           "message": "User interrupted the diagnostic"}
        except (APIError, MissingCredentials, BudgetExceeded, SchemaError) as exc:
            status, error = "subscription_error", _safe_error(exc, stage)
        except Exception as exc:
            status, error = "infrastructure_error", _safe_error(exc, stage)
        finally:
            if recorder is not None:
                try:
                    recorder.close(status=status, error=json.dumps(error) if error else None)
                except Exception as exc:
                    cleanup_errors.append(_safe_error(exc, "video_finalization"))
            if sim is not None:
                try:
                    sim.close()
                except Exception as exc:
                    cleanup_errors.append(_safe_error(exc, "simulator_close"))
            if cleanup_errors:
                status = "infrastructure_error"
            summary = {"status": status, "mode": mode, "task": TASK, "suite": "libero_spatial",
                "task_id": 0, "episode_index": 0, "seed": 7, "requested_episodes": 1,
                "steps": steps, "validated_field_refreshes": valid_fields, "local_plan_count": local_plans,
                "measured_feedback_count": feedback_count, "gripper_close_actions": close_actions,
                "gripper_open_actions": open_actions, "gripper_neutral_actions": neutral_actions,
                "path_length_m": path_length, "pending_action_index": pending_action,
                "close_event": close_event, "close_execution_acknowledged": close_event is not None,
                "diagnostic_complete": status == STOP_REASON,
                "libero_success_observed": success_seen, "libero_success": None,
                "evaluation_completed": False, "success_rate": None, "scored_episodes": 0,
                "incomplete_episodes": 1, "api_calls": client.budget_snapshot()["api_calls"],
                "mock_response_calls": client._api_calls if offline_smoke else 0,
                "billing": "offline_mock" if offline_smoke else "chatgpt_subscription",
                "budget": client.budget_snapshot(), "api_key_calls": 0, "api_key_fallback": False,
                "max_policy_steps": None, "max_api_calls": None, "max_wall_time_s": None,
                "elapsed_s": time.monotonic() - started,
                "video": video_name if (output / video_name).is_file() else None,
                "trace": "trace.jsonl", "error": error, "cleanup_errors": cleanup_errors,
                "output": str(output), "interpretation": (
                    "Scripted mock fields with simulator execution; no live inference." if offline_smoke else
                    ("The simulator acknowledged one close action; this does not verify full closure, grasp, "
                     "or task completion." if close_event is not None else
                     "The diagnostic ended before any close action was acknowledged; no task-performance score."))}
            write_json(output / "runner_summary.json", summary)
            write_json(output / "summary.json", summary)
            with (output / "episodes.jsonl").open("w") as results:
                _base._append(results, summary)
            state("finished")
    print(json.dumps({"event": "finished", **summary}), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--libero-root", default=DEFAULT_LIBERO_ROOT)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--profile")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline-smoke", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if not args.preflight and args.output is None:
        parser.error("--output is required for execution")
    config = configuration()
    sim_config = LiberoConfig(suite="libero_spatial", seed=7, max_steps=None,
        libero_root=str(Path(args.libero_root).expanduser().resolve()),
        python_executable=os.path.abspath(os.path.expanduser(args.sim_python)))
    if args.preflight:
        print(json.dumps(preflight(config, sim_config, args.output), indent=2), flush=True)
        return 0
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    token_provider = None
    if not args.offline_smoke:
        from gpt_field.chatgpt_auth import SubscriptionCredentials
        credentials = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile)
        token_provider = credentials.access_token
    result = execute_episode(args.output, config=config, sim_config=sim_config,
        token_provider=token_provider, offline_smoke=args.offline_smoke)
    return 0 if result["diagnostic_complete"] else 130 if result["status"] == "user_stop" else 2


if __name__ == "__main__":
    raise SystemExit(main())
