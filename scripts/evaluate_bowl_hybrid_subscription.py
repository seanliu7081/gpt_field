"""Evaluate model-routed coarse fields and direct contact actions on one bowl task.

Fresh LIBERO-Spatial task 0 / initial state 0 / seed 7. Live execution has no
action, call-count or episode-time horizon and continues through gripper closure
until LIBERO success, user interruption, subscription failure or infrastructure
failure. Only the isolated scripted smoke stops after fifteen diagnostic steps.
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
import time

import numpy as np

from gpt_field.astra import APIError, AstraClient, BudgetExceeded, MissingCredentials, SchemaError
from gpt_field.direct_history import HISTORY_PROTOCOL
from gpt_field.experiment import digest, source_manifest, write_json
from gpt_field.hybrid_policy import HybridFieldDirectPolicy
from gpt_field.hybrid_subscription import ChatGPTPlanHybridClient, HYBRID_PROMPT_VERSION
from gpt_field.hybrid_video import HybridVideoRecorder
from gpt_field.libero import LiberoConfig, SimProcess
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.planner import PlannerConfig
from gpt_field.schema import GridSpec


ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_frozen_subscription_close_runner",
    ROOT / "scripts/evaluate_bowl_subscription_close.py")
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
TASK = _old.TASK
DEFAULT_SIM_PYTHON = _old.DEFAULT_SIM_PYTHON
DEFAULT_LIBERO_ROOT = _old.DEFAULT_LIBERO_ROOT
configuration = _old.configuration
validate_config = _old.validate_config
SMOKE_STEPS = 15
STOPPING_RULE = ["LIBERO success", "user stop", "subscription authorization or quota error", "infrastructure error"]


def preflight(config, sim_config, output=None):
    validate_config(config, sim_config)
    if output is not None and Path(output).exists():
        raise FileExistsError("Fresh hybrid experiment requires a new output directory")
    camera = image_to_data_url(np.zeros((224, 224, 3), dtype=np.uint8), rotate=False)
    obs = PolicyObservation(camera, camera, (0, 0, 1, 0, 0, 0, .04, -.04), TASK)
    client = ChatGPTPlanHybridClient(config, token_provider=lambda: None)
    payload = client.build_payload(obs.to_dict())
    return {"status": "preflight_only", "task": TASK, "task_id": 0, "episode_index": 0,
        "billing": "chatgpt_subscription", "simulator": asdict(sim_config), "api": asdict(config),
        "synthetic_observation": True, "api_calls": 0, "api_key_calls": 0,
        "credentials_checked": False, "simulator_started": False, "files_written": False,
        "sim_python_exists": Path(sim_config.python_executable or "").is_file(),
        "libero_checkout_exists": (Path(sim_config.libero_root or "") / "libero/libero/benchmark/__init__.py").is_file(),
        "payload": {key: payload[key] for key in ("model", "store", "stream", "instructions", "text")},
        "max_policy_steps": None, "max_api_calls": None, "max_wall_time_s": None,
        "output_token_limit_sent": None, "api_key_fallback": False, "model_fallback": False,
        "stopping_rule": STOPPING_RULE, "evaluation_scored": True,
        "policy": {"method": "hybrid_field_direct", "grid": GridSpec().to_dict(),
            "planner": asdict(PlannerConfig()), "controller_update_steps": 5,
            "routing_source": "model RGB/state/history prediction, not simulator contact",
            "controllers": ["coarse_field", "contact_direct"], "recent_image_observations": 2,
            "success_feedback_to_policy": False, "reward_feedback_to_policy": False,
            "object_state_feedback_to_policy": False}}


class ScriptedHybridTransport:
    """Three explicit synthetic envelopes, never a live model request."""

    def __init__(self):
        self.calls = 0

    def __call__(self, endpoint, payload, headers, timeout):
        index = self.calls
        self.calls += 1
        if index >= 3:
            raise AssertionError("Smoke must stop before a fourth scripted response")
        if index == 1:
            envelope = {"controller": "contact_direct", "target_region": "black_bowl", "near_target": True,
                "routing_reason": "Scripted smoke: exercise direct close commands and measured feedback.",
                "field": None, "direct": {"actions": [[0, 0, 0, 0, 0, 0, 1] for _ in range(5)]}}
        else:
            grid = GridSpec()
            field = {"stage": "scripted_coarse_approach" if index == 0 else "scripted_closed_gripper_hold",
                "orientation_target_xyzw": [0, 0, 0, 1], "gripper_command": "open" if index == 0 else "hold",
                "motion_mode": "move" if index == 0 else "hold",
                "values": (-grid.coordinates((0, 0, 0))[:, 0]).tolist(),
                "blocked_indices": [], "unknown_indices": []}
            envelope = {"controller": "coarse_field", "target_region": "black_bowl" if index == 0 else "plate",
                "near_target": False, "routing_reason": "Scripted smoke: exercise field planning and mode return.",
                "field": field, "direct": None}
        return {"id": f"scripted-hybrid-{index}", "status": "completed", "offline_mock": True,
            "model": "gpt-6-astra", "output": [{"id": f"scripted-hybrid-message-{index}",
                "type": "message", "role": "assistant", "status": "completed", "phase": "final_answer",
                "content": [{"type": "output_text", "text": json.dumps(envelope)}]}],
            "usage": {"input_tokens": 0, "output_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}


class _SmokeClient(ChatGPTPlanHybridClient):
    def budget_snapshot(self):
        usage = super().budget_snapshot()
        return {**usage, "billing": "offline_mock", "api_calls": 0,
                "subscription_inference_calls": 0, "mock_response_calls": self._api_calls}

    def _write_log(self, path, data):
        if isinstance(data, dict):
            data = {**data, "billing": "offline_mock", "offline_smoke": True,
                    "transport_version": "scripted-hybrid-no-network-fixture"}
            if "payload" in data and "request_ordinal" in data:
                data.update(prompt_version=HYBRID_PROMPT_VERSION, history_protocol=HISTORY_PROTOCOL,
                    image_history=self.image_history, history_stats_before_request=self.history_stats())
                self._history_request_settled = False
        AstraClient._write_log(self, path, data)


def execute_episode(output, *, config, sim_config, token_provider=None, offline_smoke=False,
                    sim_factory=None, recorder_factory=None, transport=None):
    validate_config(config, sim_config)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Fresh hybrid experiment requires a new output directory; no resume")
    if offline_smoke:
        if transport is not None or token_provider is not None:
            raise ValueError("Offline smoke uses only its scripted responses, without credentials")
        client = _SmokeClient(config, token_provider=lambda: "OFFLINE-OAUTH-FIXTURE",
                             transport=ScriptedHybridTransport(), log_dir=output / "requests")
    else:
        if not callable(token_provider):
            raise MissingCredentials("Live hybrid execution requires a ChatGPT subscription token provider")
        client = ChatGPTPlanHybridClient(config, token_provider=token_provider, transport=transport,
                                        log_dir=output / "requests")
    policy = HybridFieldDirectPolicy(client)
    sim_factory = sim_factory or SimProcess
    recorder_factory = recorder_factory or HybridVideoRecorder
    output.mkdir(parents=True, exist_ok=False)
    mode = "offline_mock_hybrid_smoke" if offline_smoke else "live_chatgpt_subscription_hybrid"
    identity = {"kind": "subscription_hybrid_full_task", "mode": mode,
        "task": TASK, "task_id": 0, "episode_index": 0, "simulator": asdict(sim_config),
        "api": asdict(config), "billing": "offline_mock" if offline_smoke else "chatgpt_subscription",
        "software": source_manifest(), "runner": {"path": "scripts/evaluate_bowl_hybrid_subscription.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "policy": {"class": "gpt_field.hybrid_policy.HybridFieldDirectPolicy", "method": "hybrid_field_direct",
            "update_steps": 5, "recent_image_observations": 2, "grid": policy.grid.to_dict(),
            "planner": asdict(policy.planner.config), "adapter_max_rotation_rad": policy.adapter.max_rotation_rad,
            "initial_history_settings": client.history_stats(),
            "controllers": ["coarse_field", "contact_direct"],
            "routing_source": "model RGB/state/history prediction, not simulator contact",
            "local_plans_per_field_action": 1, "local_plans_per_direct_action": 0,
            "field_generation": True, "local_field_planning": True, "direct_action_chunks": 5,
            "success_feedback_to_policy": False, "reward_feedback_to_policy": False,
            "object_state_feedback_to_policy": False},
        "stopping_rule": ["fifteen scripted diagnostic actions", *STOPPING_RULE] if offline_smoke else STOPPING_RULE,
        "max_policy_steps": None, "max_api_calls": None, "max_wall_time_s": None,
        "offline_diagnostic_steps": SMOKE_STEPS if offline_smoke else None,
        "output_token_limit_sent": None, "api_key_fallback": False, "model_fallback": False,
        "evaluation_scored": not offline_smoke, "prior_budgets_inherited": False}
    write_json(output / "manifest.json", {"identity_hash": digest(identity), "configuration": identity,
        "created_utc": datetime.now(timezone.utc).isoformat(), "simulation_pauses_for_api": True})
    status, stage, error = "running", "starting", None
    steps = refreshes = fields = direct_chunks = plans = feedback_count = close_actions = open_actions = neutral_actions = 0
    coarse_actions = direct_actions = switches = 0
    pending_action, first_close_event, success_seen, path_length = None, None, False, 0.0
    controller = None
    recorder, sim, cleanup_errors = None, None, []
    started = time.monotonic()

    def state(new_stage):
        nonlocal stage
        stage = new_stage
        write_json(output / "run_state.json", {"status": status, "stage": stage, "mode": mode,
            "step": steps, "controller": controller, "controller_switches": switches,
            "pending_action_index": pending_action, "budget": client.budget_snapshot(),
            "first_close_event": first_close_event, "error": error})

    state("starting")
    with _old._base.termination_as_interrupt():
        try:
            with _old._simulator_without_credentials():
                sim = sim_factory(sim_config)
                sim.start()
            tasks = sim.list_tasks()
            if not tasks or tasks[0].get("task_id") != 0 or tasks[0].get("prompt") != TASK:
                raise ValueError("Official task zero does not match the requested bowl task")
            write_json(output / "simulator.json", {"metadata": sim.metadata, "tasks": tasks})
            obs = sim.reset(0, 0)
            if obs.prompt != TASK:
                raise ValueError("Reset observation has a different task")
            policy.reset()
            recorder = recorder_factory(output / "episode.mp4", grid=policy.grid,
                episode_id="CHATGPT SUBSCRIPTION / libero_spatial_t00_i00 / HYBRID FULL TASK", offline_smoke=offline_smoke)
            recorder.observe(obs, step=0)
            with (output / "trace.jsonl").open("w") as log:
                append = lambda row: _old._base._append(log, row)
                append({"event": "reset", "observation": obs.to_dict(), "sim_status": dict(sim.status), "mode": mode})
                while True:
                    if sim.status.get("success"):
                        success_seen, status = True, "diagnostic_sim_terminated" if offline_smoke else "success"
                        break
                    if offline_smoke and steps == SMOKE_STEPS:
                        status = "diagnostic_complete"
                        break
                    if sim.status.get("terminated"):
                        raise RuntimeError("Simulator terminated without task success or an action horizon")
                    if obs.prompt != TASK:
                        raise ValueError("Policy task changed during rollout")
                    state("deciding")
                    began = time.monotonic()
                    decision = policy.act(obs)
                    latency = time.monotonic() - began
                    details = decision.diagnostics
                    previous_controller, controller = controller, details["controller"]
                    if details.get("refreshed"):
                        refreshes += 1
                        fields += int(controller == "coarse_field")
                        direct_chunks += int(controller == "contact_direct")
                        append({"event": "routing_decision", "step": steps, "controller": controller,
                                "routing": details["routing"], "previous_controller": previous_controller,
                                "source": "model_prediction", "mode": mode})
                        if previous_controller is not None and previous_controller != controller:
                            switches += 1
                            append({"event": "controller_switch", "step": steps, "from_controller": previous_controller,
                                    "to_controller": controller, "routing": details["routing"], "source": "model_prediction"})
                    plans += int("plan" in details)
                    append({"event": "action", "step": steps, "state": list(obs.state),
                            "action": list(decision.action), "diagnostics": details,
                            "decision_latency_s": latency, "mode": mode})
                    recorder.decision(obs, decision, decision_latency_s=latency)
                    pending_action = steps
                    state("executing")
                    before = obs
                    obs, success = sim.step(decision.action)
                    executed_step, steps = steps, steps + 1
                    pending_action = None
                    success_seen = success_seen or success
                    close_actions += int(decision.action[6] > 0)
                    open_actions += int(decision.action[6] < 0)
                    neutral_actions += int(decision.action[6] == 0)
                    coarse_actions += int(controller == "coarse_field")
                    direct_actions += int(controller == "contact_direct")
                    path_length += float(np.linalg.norm(np.asarray(obs.state[:3]) - np.asarray(before.state[:3])))
                    append({"event": "executed", "step": executed_step, "controller": controller,
                            "next_state": list(obs.state), "observation": obs.to_dict(), "action": list(decision.action),
                            "success": success, "sim_status": dict(sim.status), "mode": mode})
                    if first_close_event is None and decision.action[6] > 0:
                        first_close_event = {"action_index": executed_step, "action_ordinal": steps,
                            "controller": controller, "gripper_joint_delta": [obs.state[i] - before.state[i] for i in (6, 7)],
                            "full_closure_verified": False, "grasp_verified": False}
                    feedback = policy.observe_transition(before, decision, obs)
                    feedback_count += 1
                    append({"event": "feedback", "step": executed_step, "feedback": feedback})
                    recorder.executed(obs, step=executed_step, success=success, sim_status=dict(sim.status),
                        feedback=feedback, history_stats=client.history_stats())
                    if success:
                        status = "diagnostic_sim_terminated" if offline_smoke else "success"
                    state("observed")
                    print(json.dumps({"event": "executed", "mode": mode, "steps": steps, "controller": controller,
                        "controller_switches": switches, "close_actions": close_actions, "success_observed": success,
                        "api_attempts": client.budget_snapshot()["api_calls"]}), flush=True)
                    if success:
                        break
        except KeyboardInterrupt:
            status, error = "user_stop", {"type": "KeyboardInterrupt", "stage": stage, "message": "User interrupted the hybrid episode"}
        except (APIError, MissingCredentials, BudgetExceeded, SchemaError) as exc:
            status, error = "subscription_error", _old._safe_error(exc, stage)
        except Exception as exc:
            status, error = "infrastructure_error", _old._safe_error(exc, stage)
        finally:
            if sim is not None:
                try:
                    sim.close()
                except Exception as exc:
                    cleanup_errors.append(_old._safe_error(exc, "simulator_close"))
            if cleanup_errors:
                status = "infrastructure_error"
            if recorder is not None:
                try:
                    recorder.close(status=status, error=json.dumps(error or cleanup_errors) if error or cleanup_errors else None)
                except Exception as exc:
                    cleanup_errors.append(_old._safe_error(exc, "video_finalization"))
                    status = "infrastructure_error"
            completed = status == "success" and not offline_smoke
            summary = {"status": status, "mode": mode, "task": TASK, "suite": "libero_spatial", "task_id": 0,
                "episode_index": 0, "seed": 7, "requested_episodes": 1, "steps": steps,
                "validated_hybrid_refreshes": refreshes, "validated_field_refreshes": fields,
                "validated_direct_chunks": direct_chunks, "local_plan_count": plans,
                "coarse_field_actions": coarse_actions, "contact_direct_actions": direct_actions,
                "controller_switches": switches, "final_controller": controller,
                "measured_feedback_count": feedback_count, "gripper_close_actions": close_actions,
                "gripper_open_actions": open_actions, "gripper_neutral_actions": neutral_actions,
                "path_length_m": path_length, "pending_action_index": pending_action,
                "first_close_event": first_close_event,
                "diagnostic_complete": offline_smoke and status == "diagnostic_complete",
                "libero_success_observed": success_seen, "libero_success": True if completed else None,
                "evaluation_completed": completed, "success_rate": 1.0 if completed else None,
                "scored_episodes": int(completed), "incomplete_episodes": int(not completed),
                "api_calls": client.budget_snapshot()["api_calls"], "mock_response_calls": client._api_calls if offline_smoke else 0,
                "billing": "offline_mock" if offline_smoke else "chatgpt_subscription", "budget": client.budget_snapshot(),
                "api_key_calls": 0, "api_key_fallback": False, "max_policy_steps": None, "max_api_calls": None,
                "max_wall_time_s": None, "elapsed_s": time.monotonic() - started,
                "video": "episode.mp4" if (output / "episode.mp4").is_file() else None,
                "trace": "trace.jsonl", "error": error, "cleanup_errors": cleanup_errors, "output": str(output),
                "interpretation": ("Scripted field/direct/field smoke with real simulator execution; no model performance score."
                    if offline_smoke else "One hybrid episode. Only actual LIBERO success completes and scores the episode; other stops remain incomplete and unscored.")}
            write_json(output / "runner_summary.json", summary)
            write_json(output / "summary.json", summary)
            with (output / "episodes.jsonl").open("w") as results:
                _old._base._append(results, summary)
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
        token_provider = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile).access_token
    result = execute_episode(args.output, config=config, sim_config=sim_config,
        token_provider=token_provider, offline_smoke=args.offline_smoke)
    return 0 if result["evaluation_completed"] or result["diagnostic_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
