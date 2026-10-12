"""Evaluate model-routed coarse fields and direct contact actions on a registered task.

Defaults to fresh LIBERO-Spatial task 0 / initial state 0 / seed 7; distance
control also supports --task drawer_bowl (LIBERO-Goal task 3). Live execution has no
action, call-count or episode-time horizon and continues through gripper closure
until LIBERO success, user interruption, subscription failure or infrastructure
failure. An opt-in compact protocol also stops immediately when Astra declares
completion, without treating that declaration as LIBERO success. Only
the isolated scripted smoke stops after two thirty-action field blocks and one
five-action direct block.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from gpt_field.astra import APIError, AstraClient, BudgetExceeded, MissingCredentials, SchemaError
from gpt_field.direct_history import HISTORY_PROTOCOL
from gpt_field.compact_subscription import ChatGPTPlanCompactClient
from gpt_field.experiment import digest, source_manifest, write_json
from gpt_field.hybrid_policy import HybridFieldDirectPolicy, ModelTaskComplete
from gpt_field.hybrid_subscription import (
    ChatGPTPlanHybridClient, HYBRID_CHUNK_SIZE, HYBRID_FIELD_UPDATE_STEPS, HYBRID_GRID, HYBRID_PROMPT_VERSION,
)
from gpt_field.hybrid_video import HybridVideoRecorder
from gpt_field.distance_subscription import ChatGPTPlanDistanceClient
from gpt_field.distance_policy import DistanceHybridPolicy
from gpt_field.distance_video import DistanceHybridVideoRecorder
from gpt_field.distance_tasks import DISTANCE_TASK_KEYS, resolve_distance_task
from gpt_field.libero import LiberoConfig, SimProcess
from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.planner import PlannerConfig


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
SMOKE_STEPS = 2 * HYBRID_FIELD_UPDATE_STEPS + HYBRID_CHUNK_SIZE
STOPPING_RULE = ["LIBERO success", "user stop", "subscription authorization or quota error", "infrastructure error"]


def _client_type(field_representation, distance_field_control=False):
    if distance_field_control:
        if field_representation != "compact":
            raise ValueError("Distance control requires compact representation")
        return ChatGPTPlanDistanceClient
    if field_representation == "compact":
        return ChatGPTPlanCompactClient
    if field_representation == "dense":
        return ChatGPTPlanHybridClient
    raise ValueError("field_representation must be compact or dense")


def _control_options(field_representation, reasoning_effort, stop_on_model_complete, *, offline_smoke=False,
                     controller_mode="hybrid", execution_mode="fixed", feedback_history="full",
                     centered_placement_guidance=False, grasp_settle_guidance=False, field_improvements=False,
                     distance_field_control=False, task_key="bowl_plate"):
    task_spec = resolve_distance_task(task_key)
    if task_spec.key != "bowl_plate" and not distance_field_control:
        raise ValueError("Explicit task selection requires --distance-field-control")
    _client_type(field_representation, distance_field_control)
    if type(distance_field_control) is not bool:
        raise ValueError("distance_field_control must be a boolean")
    if distance_field_control and (not field_improvements or feedback_history != "block_summary"
            or not centered_placement_guidance or not grasp_settle_guidance):
        raise ValueError("Distance control requires field improvements and shared guidance with block summaries")
    if reasoning_effort not in ("medium", "high"):
        raise ValueError("reasoning_effort must be medium or high")
    if controller_mode not in ("hybrid", "direct"):
        raise ValueError("full-task controller_mode must be hybrid or direct")
    if execution_mode not in ("fixed", "tracking") or feedback_history not in ("full", "block_summary"):
        raise ValueError("Unsupported field execution or feedback history mode")
    if type(centered_placement_guidance) is not bool:
        raise ValueError("centered_placement_guidance must be a boolean")
    if centered_placement_guidance and (offline_smoke or field_representation != "compact"
            or stop_on_model_complete):
        raise ValueError("Centered placement guidance requires live compact Hybrid or Direct with LIBERO-only completion")
    if type(grasp_settle_guidance) is not bool:
        raise ValueError("grasp_settle_guidance must be a boolean")
    if grasp_settle_guidance and not centered_placement_guidance:
        raise ValueError("Grasp settling guidance requires centered placement guidance")
    if type(field_improvements) is not bool:
        raise ValueError("field_improvements must be a boolean")
    if field_improvements and (offline_smoke or field_representation != "compact"
            or controller_mode != "hybrid" or execution_mode != "tracking"
            or reasoning_effort != "medium" or stop_on_model_complete):
        raise ValueError("Field improvements require compact Hybrid, tracking, medium effort and LIBERO-only completion")
    if (offline_smoke or field_representation == "dense") and (
            controller_mode != "hybrid" or execution_mode != "fixed" or feedback_history != "full"):
        raise ValueError("Optimized execution and forced Direct require live compact control")
    if (offline_smoke or field_representation == "dense") and (
            reasoning_effort != "high" or stop_on_model_complete):
        raise ValueError("Medium effort and model-completion stopping require live compact control")
    return ({"reasoning_effort": reasoning_effort, "stop_on_model_complete": stop_on_model_complete,
             "controller_mode": controller_mode, "execution_mode": execution_mode, "feedback_history": feedback_history,
             "centered_placement_guidance": centered_placement_guidance,
             "grasp_settle_guidance": grasp_settle_guidance, "field_improvements": field_improvements,
             **({"distance_field_control": True} if distance_field_control else {}),
             **({"task_key": task_spec.key} if task_spec.key != "bowl_plate" else {})}
            if field_representation == "compact" else {})


def _stopping_rule(stop_on_model_complete):
    return (["Astra declares task_complete=true"] if stop_on_model_complete else []) + STOPPING_RULE


def _validate_task_configuration(config, sim_config, task_spec):
    if sim_config.suite != task_spec.suite:
        raise ValueError("Simulator suite must match the selected task")
    # Reuse the frozen runner's effort-independent transport, image, seed and
    # horizon checks. Only its old hard-coded suite check needs substitution;
    # actual simulator construction and every artifact retain the real suite.
    validate_config(config, replace(sim_config, suite="libero_spatial"))


def preflight(config, sim_config, output=None, *, field_representation="compact",
              reasoning_effort="high", stop_on_model_complete=False,
              controller_mode="hybrid", execution_mode="fixed", feedback_history="full",
              centered_placement_guidance=False, grasp_settle_guidance=False, field_improvements=False,
              distance_field_control=False, task_key="bowl_plate"):
    task_spec = resolve_distance_task(task_key)
    client_options = _control_options(field_representation, reasoning_effort, stop_on_model_complete,
        controller_mode=controller_mode, execution_mode=execution_mode, feedback_history=feedback_history,
        centered_placement_guidance=centered_placement_guidance, grasp_settle_guidance=grasp_settle_guidance,
        field_improvements=field_improvements, distance_field_control=distance_field_control, task_key=task_key)
    _validate_task_configuration(config, sim_config, task_spec)
    if output is not None and Path(output).exists():
        raise FileExistsError("Fresh hybrid experiment requires a new output directory")
    camera = image_to_data_url(np.zeros((224, 224, 3), dtype=np.uint8), rotate=False)
    obs = PolicyObservation(camera, camera, (0, 0, 1, 0, 0, 0, .04, -.04), task_spec.prompt)
    client_type = _client_type(field_representation, distance_field_control)
    client = client_type(config, token_provider=lambda: None, **client_options)
    payload = client.build_payload(obs.to_dict())
    return {"status": "preflight_only", "task": task_spec.prompt, "task_id": task_spec.task_id, "episode_index": 0,
        **({"distance_task": task_spec.key, "task_spec": asdict(task_spec)} if task_spec.key != "bowl_plate" else {}),
        "billing": "chatgpt_subscription", "simulator": asdict(sim_config), "api": asdict(config),
        "synthetic_observation": True, "api_calls": 0, "api_key_calls": 0,
        "credentials_checked": False, "simulator_started": False, "files_written": False,
        "sim_python_exists": Path(sim_config.python_executable or "").is_file(),
        "libero_checkout_exists": (Path(sim_config.libero_root or "") / "libero/libero/benchmark/__init__.py").is_file(),
        "payload": {key: payload[key] for key in ("model", "store", "stream", "instructions", "text", "reasoning")
                    if key in payload},
        "max_policy_steps": None, "max_api_calls": None, "max_wall_time_s": None,
        "output_token_limit_sent": None, "api_key_fallback": False, "model_fallback": False,
        "stopping_rule": _stopping_rule(stop_on_model_complete), "evaluation_scored": True,
        "stop_on_model_complete": stop_on_model_complete,
        "model_completion_is_libero_success": False,
        "policy": {"method": "hybrid_field_direct", "grid": client.grid.to_dict(),
            "field_representation": getattr(client, "field_representation", "dense"),
            "controller_mode": getattr(client, "controller_mode", "hybrid"), "prompt_version": client.prompt_version,
            "planner": asdict(PlannerConfig(horizon=client.planner_horizon)), "controller_update_steps": client.field_update_steps,
            "field_update_steps": client.field_update_steps, "direct_action_steps": client.chunk_size,
            "field_execution_steps": client.field_execution_steps,
            "reasoning_effort": client.reasoning_effort,
            "execution_mode": execution_mode, "feedback_history": feedback_history,
            "centered_placement_guidance": centered_placement_guidance,
            "grasp_settle_guidance": grasp_settle_guidance,
            "field_improvements": field_improvements,
            "distance_field_control": distance_field_control,
            "field_reuse_limit": 120 if distance_field_control else None,
            "field_refresh_policy": ("visual_progress_guarded_30_to_120" if distance_field_control else
                                     "events_or_30_actions" if field_improvements else "fixed_interval"),
            "local_path_strategy": ("dijkstra_distance_field" if distance_field_control else
                                    "analytic_pure_goal_else_lattice" if field_improvements else "lattice"),
            "goal_tolerance_m": 0.005 if field_improvements else None,
            "goal_orientation_tolerance_rad": 0.05 if field_improvements else None,
            "tracking_lookahead_m": .03 if execution_mode == "tracking" else None,
            "tracking_max_command_m": .05 if execution_mode == "tracking" else None,
            "direct_action_chunks": client.chunk_size,
            "routing_source": ("model RGB/state/history prediction, not simulator contact"
                               if controller_mode == "hybrid" else "fixed Direct comparison arm"),
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
                "field": None, "direct": {"actions": [[0, 0, 0, 0, 0, 0, 1] for _ in range(HYBRID_CHUNK_SIZE)]}}
        else:
            grid = HYBRID_GRID
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
                data.update(prompt_version=self.prompt_version, history_protocol=HISTORY_PROTOCOL,
                    image_history=self.image_history, history_stats_before_request=self.history_stats())
                self._history_request_settled = False
        AstraClient._write_log(self, path, data)


def execute_episode(output, *, config, sim_config, token_provider=None, offline_smoke=False,
                    sim_factory=None, recorder_factory=None, transport=None, field_representation="compact",
                    reasoning_effort="high", stop_on_model_complete=False,
                    controller_mode="hybrid", execution_mode="fixed", feedback_history="full",
                    centered_placement_guidance=False, grasp_settle_guidance=False, field_improvements=False,
                    distance_field_control=False, task_key="bowl_plate"):
    task_spec = resolve_distance_task(task_key)
    client_options = _control_options(field_representation, reasoning_effort, stop_on_model_complete,
        offline_smoke=offline_smoke, controller_mode=controller_mode,
        execution_mode=execution_mode, feedback_history=feedback_history,
        centered_placement_guidance=centered_placement_guidance, grasp_settle_guidance=grasp_settle_guidance,
        field_improvements=field_improvements, distance_field_control=distance_field_control, task_key=task_key)
    _validate_task_configuration(config, sim_config, task_spec)
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
        client = _client_type(field_representation, distance_field_control)(config, token_provider=token_provider, transport=transport,
                                        log_dir=output / "requests", **client_options)
    policy = (DistanceHybridPolicy if distance_field_control else HybridFieldDirectPolicy)(client)
    sim_factory = sim_factory or SimProcess
    recorder_factory = recorder_factory or (DistanceHybridVideoRecorder if distance_field_control else HybridVideoRecorder)
    output.mkdir(parents=True, exist_ok=False)
    mode = "offline_mock_hybrid_smoke" if offline_smoke else "live_chatgpt_subscription_hybrid"
    identity = {"kind": "subscription_hybrid_full_task", "mode": mode,
        "task": task_spec.prompt, "task_id": task_spec.task_id, "episode_index": 0, "simulator": asdict(sim_config),
        **({"distance_task": task_spec.key, "task_spec": asdict(task_spec)} if task_spec.key != "bowl_plate" else {}),
        "api": asdict(config), "billing": "offline_mock" if offline_smoke else "chatgpt_subscription",
        "software": source_manifest(), "runner": {"path": "scripts/evaluate_bowl_hybrid_subscription.py",
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "policy": {"class": ("gpt_field.distance_policy.DistanceHybridPolicy" if distance_field_control else
                             "gpt_field.hybrid_policy.HybridFieldDirectPolicy"), "method": "hybrid_field_direct",
            "field_representation": getattr(client, "field_representation", "dense"),
            "controller_mode": getattr(client, "controller_mode", "hybrid"), "prompt_version": client.prompt_version,
            "update_steps": policy.update_steps, "field_update_steps": client.field_update_steps,
            "field_execution_steps": client.field_execution_steps,
            "direct_action_steps": client.chunk_size, "recent_image_observations": 2, "grid": policy.grid.to_dict(),
            "reasoning_effort": client.reasoning_effort,
            "execution_mode": execution_mode, "feedback_history": feedback_history,
            "centered_placement_guidance": centered_placement_guidance,
            "grasp_settle_guidance": grasp_settle_guidance,
            "field_improvements": field_improvements,
            "distance_field_control": distance_field_control,
            "field_reuse_limit": 120 if distance_field_control else None,
            "field_refresh_policy": ("visual_progress_guarded_30_to_120" if distance_field_control else
                                     "events_or_30_actions" if field_improvements else "fixed_interval"),
            "local_path_strategy": ("dijkstra_distance_field" if distance_field_control else
                                    "analytic_pure_goal_else_lattice" if field_improvements else "lattice"),
            "goal_tolerance_m": 0.005 if field_improvements else None,
            "goal_orientation_tolerance_rad": 0.05 if field_improvements else None,
            "tracking_lookahead_m": .03 if execution_mode == "tracking" else None,
            "tracking_max_command_m": .05 if execution_mode == "tracking" else None,
            "planner": asdict(policy.planner.config), "adapter_max_rotation_rad": policy.adapter.max_rotation_rad,
            "initial_history_settings": client.history_stats(),
            "controllers": ["coarse_field", "contact_direct"],
            "routing_source": ("model RGB/state/history prediction, not simulator contact"
                               if controller_mode == "hybrid" else "fixed Direct comparison arm"),
            "local_plans_per_field_action": None if field_improvements else 1 / client.field_execution_steps,
            "local_plans_per_direct_action": 0,
            "field_generation": controller_mode == "hybrid", "local_field_planning": controller_mode == "hybrid", "direct_action_chunks": client.chunk_size,
            "success_feedback_to_policy": False, "reward_feedback_to_policy": False,
            "object_state_feedback_to_policy": False},
        "stopping_rule": ([f"{SMOKE_STEPS} scripted diagnostic actions", *STOPPING_RULE]
                          if offline_smoke else _stopping_rule(stop_on_model_complete)),
        "stop_on_model_complete": stop_on_model_complete,
        "model_completion_is_libero_success": False,
        "max_policy_steps": None, "max_api_calls": None, "max_wall_time_s": None,
        "offline_diagnostic_steps": SMOKE_STEPS if offline_smoke else None,
        "output_token_limit_sent": None, "api_key_fallback": False, "model_fallback": False,
        "evaluation_scored": not offline_smoke, "prior_budgets_inherited": False}
    write_json(output / "manifest.json", {"identity_hash": digest(identity), "configuration": identity,
        "created_utc": datetime.now(timezone.utc).isoformat(), "simulation_pauses_for_api": True})
    frozen = output / "frozen_source"
    frozen.mkdir()
    for filename in identity["software"]["source_sha256"]:
        shutil.copyfile(ROOT / "src/gpt_field" / filename, frozen / filename)
    shutil.copyfile(Path(__file__), frozen / Path(__file__).name)
    status, stage, error = "running", "starting", None
    steps = refreshes = fields = direct_chunks = plans = feedback_count = close_actions = open_actions = neutral_actions = 0
    coarse_actions = direct_actions = switches = 0
    pending_action, first_close_event, success_seen, path_length = None, None, False, 0.0
    controller = None
    recorder, sim, cleanup_errors = None, None, []
    started = time.monotonic()
    control_started = control_stopped = None
    model_completion = None
    timings = {key: 0.0 for key in ("decision_s", "model_pipeline_s", "field_expansion_s",
        "local_planning_s", "simulator_step_s", "video_render_s")}

    def record_decision_time(latency, details):
        timings["decision_s"] += latency
        for key, detail in (("model_pipeline_s", "api_latency_s"),
                            ("field_expansion_s", "field_expansion_latency_s"),
                            ("local_planning_s", "plan_latency_s")):
            timings[key] += float(details.get(detail) or 0.0)

    def state(new_stage):
        nonlocal stage
        stage = new_stage
        write_json(output / "run_state.json", {"status": status, "stage": stage, "mode": mode,
            "step": steps, "controller": controller, "controller_switches": switches,
            "pending_action_index": pending_action, "budget": client.budget_snapshot(),
            "first_close_event": first_close_event, "model_completion": model_completion, "error": error})

    state("starting")
    with _old._base.termination_as_interrupt():
        try:
            with _old._simulator_without_credentials():
                sim = sim_factory(sim_config)
                sim.start()
            tasks = sim.list_tasks()
            selected = [task for task in tasks if task.get("task_id") == task_spec.task_id]
            if len(selected) != 1 or selected[0].get("prompt") != task_spec.prompt:
                raise ValueError("Official task does not match the selected task profile")
            write_json(output / "simulator.json", {"metadata": sim.metadata, "tasks": tasks})
            obs = sim.reset(task_spec.task_id, 0)
            if obs.prompt != task_spec.prompt:
                raise ValueError("Reset observation has a different task")
            policy.reset()
            recorder = recorder_factory(output / "episode.mp4", grid=policy.grid,
                episode_id=f"CHATGPT SUBSCRIPTION / {task_spec.suite}_t{task_spec.task_id:02d}_i00 / {controller_mode.upper()} FULL TASK", offline_smoke=offline_smoke)
            recorder.observe(obs, step=0)
            control_started = time.monotonic()
            with (output / "trace.jsonl").open("w") as log:
                append = lambda row: _old._base._append(log, row)
                append({"event": "reset", "observation": obs.to_dict(), "sim_status": dict(sim.status), "mode": mode})
                while True:
                    if sim.status.get("success"):
                        success_seen = True
                        status = "diagnostic_sim_terminated" if offline_smoke else "success"
                        break
                    if offline_smoke and steps == SMOKE_STEPS:
                        status = "diagnostic_complete"
                        break
                    if sim.status.get("terminated"):
                        raise RuntimeError("Simulator terminated without task success or an action horizon")
                    if obs.prompt != task_spec.prompt:
                        raise ValueError("Policy task changed during rollout")
                    state("deciding")
                    began = time.monotonic()
                    try:
                        decision = policy.act(obs)
                    except ModelTaskComplete as completion:
                        if not stop_on_model_complete:
                            raise RuntimeError("Model completion was not enabled for this experiment") from completion
                        control_stopped = time.monotonic()
                        latency = control_stopped - began
                        details = completion.details
                        record_decision_time(latency, details)
                        previous_controller, controller = controller, completion.prediction["controller"]
                        refreshes += 1
                        fields += int(controller == "coarse_field")
                        direct_chunks += int(controller == "contact_direct")
                        model_completion = {
                            "step": steps, "acknowledged_actions": steps, "task_complete": True,
                            "completion_reason": details["completion_reason"], "controller": controller,
                            "previous_controller": previous_controller,
                            "decision_latency_s": latency, "elapsed_s": control_stopped - started,
                            "control_elapsed_s": control_stopped - control_started,
                            "api_latency_s": details.get("api_latency_s"),
                            "api_request_key": details.get("api_request_key"),
                            "api_request_log": details.get("api_request_log"),
                            "reasoning_effort": client.reasoning_effort,
                            "libero_success_observed": success_seen,
                            "action_proposed": False, "action_executed": False,
                        }
                        append({"event": "model_completion", **model_completion,
                                "diagnostics": details, "prediction": completion.prediction, "mode": mode})
                        status = "model_declared_complete"
                        state("model_declared_complete")
                        recorder.model_completion(obs, step=steps,
                            completion_reason=details["completion_reason"],
                            decision_latency_s=latency, diagnostics=details)
                        print(json.dumps({"event": "model_completion", **model_completion}), flush=True)
                        break
                    latency = time.monotonic() - began
                    details = decision.diagnostics
                    record_decision_time(latency, details)
                    previous_controller, controller = controller, details["controller"]
                    if details.get("refreshed"):
                        refreshes += 1
                        fields += int(controller == "coarse_field")
                        direct_chunks += int(controller == "contact_direct")
                        append({"event": "routing_decision", "step": steps, "controller": controller,
                                "routing": details["routing"], "previous_controller": previous_controller,
                                "source": "model_prediction" if controller_mode == "hybrid" else "forced_direct", "mode": mode})
                        if previous_controller is not None and previous_controller != controller:
                            switches += 1
                            append({"event": "controller_switch", "step": steps, "from_controller": previous_controller,
                                    "to_controller": controller, "routing": details["routing"], "source": "model_prediction"})
                    plans += int(details.get("plan_refreshed", "plan" in details))
                    append({"event": "action", "step": steps, "state": list(obs.state),
                            "action": list(decision.action), "diagnostics": details,
                            "decision_latency_s": latency, "mode": mode})
                    video_began = time.monotonic()
                    recorder.decision(obs, decision, decision_latency_s=latency)
                    timings["video_render_s"] += time.monotonic() - video_began
                    pending_action = steps
                    state("executing")
                    before = obs
                    sim_began = time.monotonic()
                    obs, success = sim.step(decision.action)
                    timings["simulator_step_s"] += time.monotonic() - sim_began
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
                    if field_improvements and policy.pending_field_event is not None:
                        event_name = "distance_control_event" if distance_field_control else "field_control_event"
                        append({"event": event_name, "step": steps,
                                event_name: policy.pending_field_event})
                    if distance_field_control:
                        append({"event": "local_visual_feedback", "step": executed_step,
                                "feedback": policy.local_feedback,
                                "local_field_revision": policy.local_revision,
                                "local_recovery_active": policy._recovery_active})
                    video_began = time.monotonic()
                    recorder.executed(obs, step=executed_step, success=success, sim_status=dict(sim.status),
                        feedback=feedback, history_stats=client.history_stats())
                    timings["video_render_s"] += time.monotonic() - video_began
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
            if control_started is not None and control_stopped is None:
                control_stopped = time.monotonic()
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
            completed = (not offline_smoke and (status == "success" or
                (status == "model_declared_complete" and success_seen)))
            if distance_field_control:
                timings["local_visual_s"] = policy.local_visual_total_s
                timings["local_field_update_s"] = policy.local_update_total_s
            summary = {"status": status, "mode": mode, "task": task_spec.prompt,
                "suite": task_spec.suite, "task_id": task_spec.task_id,
                **({"distance_task": task_spec.key} if task_spec.key != "bowl_plate" else {}),
                "episode_index": 0, "seed": 7, "requested_episodes": 1, "steps": steps,
                "validated_hybrid_refreshes": refreshes, "validated_field_refreshes": fields,
                "validated_direct_chunks": direct_chunks, "local_plan_count": plans, "distance_field_control": distance_field_control,
                "coarse_field_actions": coarse_actions, "contact_direct_actions": direct_actions,
                "controller_switches": switches, "final_controller": controller,
                "measured_feedback_count": feedback_count, "gripper_close_actions": close_actions,
                "gripper_open_actions": open_actions, "gripper_neutral_actions": neutral_actions,
                "path_length_m": path_length, "pending_action_index": pending_action,
                "first_close_event": first_close_event,
                "diagnostic_complete": offline_smoke and status == "diagnostic_complete",
                "stop_on_model_complete": stop_on_model_complete,
                "model_declared_complete": status == "model_declared_complete",
                "model_completion": model_completion,
                "benchmark_completed": not offline_smoke and status in ("success", "model_declared_complete"),
                "reasoning_effort": client.reasoning_effort,
                "controller_mode": controller_mode, "execution_mode": execution_mode,
                "feedback_history": feedback_history, "timing_components": timings,
                "centered_placement_guidance": centered_placement_guidance,
                "grasp_settle_guidance": grasp_settle_guidance,
                "field_improvements": field_improvements,
                "timing_components_note": "field_expansion_s is included in model_pipeline_s; model_pipeline_s and local_planning_s are included in decision_s. "
                    + ("local_visual_s includes both decision-time setup and post-action tracking; local_field_update_s measures post-action updates. These components overlap and must not all be added. " if distance_field_control else "")
                    + "Terminal video and cleanup are excluded from control_elapsed_s.",
                "libero_success_observed": success_seen, "libero_success": True if completed else None,
                "evaluation_completed": completed, "success_rate": 1.0 if completed else None,
                "scored_episodes": int(completed), "incomplete_episodes": int(not completed),
                "api_calls": client.budget_snapshot()["api_calls"], "mock_response_calls": client._api_calls if offline_smoke else 0,
                "billing": "offline_mock" if offline_smoke else "chatgpt_subscription", "budget": client.budget_snapshot(),
                "api_key_calls": 0, "api_key_fallback": False, "max_policy_steps": None, "max_api_calls": None,
                "max_wall_time_s": None, "elapsed_s": time.monotonic() - started,
                "startup_to_initial_observation_s": (control_started - started if control_started is not None else None),
                "control_elapsed_s": (control_stopped - control_started if control_started is not None else None),
                "video": "episode.mp4" if (output / "episode.mp4").is_file() else None,
                "trace": "trace.jsonl", "error": error, "cleanup_errors": cleanup_errors, "output": str(output),
                "interpretation": ("Scripted field/direct/field smoke with real simulator execution; no model performance score."
                    if offline_smoke else "One distance-field hybrid episode. Only actual LIBERO success completes and scores the evaluation; visual agreement and model statements do not stop the task. Other stops remain incomplete and unscored."
                    if distance_field_control else "One hybrid episode. A model completion declaration completes the requested timing run, but only actual LIBERO success completes and scores the evaluation; other stops remain incomplete and unscored.")}
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
    parser.add_argument("--field-representation", choices=("compact", "dense"), default="compact")
    parser.add_argument("--reasoning-effort", choices=("medium", "high"))
    parser.add_argument("--stop-on-model-complete", action="store_true")
    parser.add_argument("--controller-mode", choices=("hybrid", "direct"), default="hybrid")
    parser.add_argument("--execution-mode", choices=("fixed", "tracking"))
    parser.add_argument("--feedback-history", choices=("full", "block_summary"))
    parser.add_argument("--distance-field-control", action="store_true",
                        help="Enable v7 Dijkstra distance fields, local RGB feedback and placement recovery; implies field improvements")
    parser.add_argument("--task", choices=DISTANCE_TASK_KEYS, default="bowl_plate",
                        help="Select a registered task; nondefault tasks require --distance-field-control")
    parser.add_argument("--field-improvements", action="store_true",
                        help="Enable visual goal checks, grasp-offset targets and variable analytic paths; defaults to medium/tracking/block_summary")
    parser.add_argument("--centered-placement-guidance", action="store_true",
                        help="Ask Hybrid or Direct to visually center the bowl; requires LIBERO-only completion")
    parser.add_argument("--grasp-settle-guidance", action="store_true",
                        help="Explain control timing and measured grasp settling; requires centered placement guidance")
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--actions-per-block", type=int, default=30)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--offline-smoke", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--timing-comparison", action="store_true")
    args = parser.parse_args(argv)
    args.field_improvements = args.field_improvements or args.distance_field_control
    args.reasoning_effort = args.reasoning_effort or ("medium" if args.field_improvements else "high")
    args.execution_mode = args.execution_mode or ("tracking" if args.field_improvements else "fixed")
    args.feedback_history = args.feedback_history or ("block_summary" if args.field_improvements else "full")
    if args.field_improvements:
        args.centered_placement_guidance = args.grasp_settle_guidance = True
    if args.timing_comparison and args.field_representation != "compact":
        parser.error("--timing-comparison requires --field-representation compact")
    if args.timing_comparison and (args.reasoning_effort != "high" or args.stop_on_model_complete):
        parser.error("--timing-comparison supports high effort and no model-completion stop")
    if args.timing_comparison and (args.controller_mode != "hybrid" or args.execution_mode != "fixed"
                                   or args.feedback_history != "full" or args.centered_placement_guidance
                                   or args.grasp_settle_guidance or args.field_improvements):
        parser.error("Legacy equal-action timing comparison requires its fixed/full defaults")
    try:
        _control_options(args.field_representation, args.reasoning_effort, args.stop_on_model_complete,
            offline_smoke=args.offline_smoke, controller_mode=args.controller_mode,
            execution_mode=args.execution_mode, feedback_history=args.feedback_history,
            centered_placement_guidance=args.centered_placement_guidance,
            grasp_settle_guidance=args.grasp_settle_guidance, field_improvements=args.field_improvements,
            distance_field_control=args.distance_field_control, task_key=args.task)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.preflight and args.output is None:
        parser.error("--output is required for execution")
    config = configuration()
    task_spec = resolve_distance_task(args.task)
    sim_config = LiberoConfig(suite=task_spec.suite, seed=7, max_steps=None,
        libero_root=str(Path(args.libero_root).expanduser().resolve()),
        python_executable=os.path.abspath(os.path.expanduser(args.sim_python)))
    if args.preflight:
        print(json.dumps(preflight(config, sim_config, args.output, field_representation=args.field_representation,
            reasoning_effort=args.reasoning_effort, stop_on_model_complete=args.stop_on_model_complete,
            controller_mode=args.controller_mode, execution_mode=args.execution_mode,
            feedback_history=args.feedback_history,
            centered_placement_guidance=args.centered_placement_guidance,
            grasp_settle_guidance=args.grasp_settle_guidance, field_improvements=args.field_improvements,
            distance_field_control=args.distance_field_control, task_key=args.task), indent=2), flush=True)
        return 0
    if args.output.exists():
        parser.error("output already exists; choose a fresh directory")
    token_provider = None
    if not args.offline_smoke:
        from gpt_field.chatgpt_auth import SubscriptionCredentials
        token_provider = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile).access_token
    if args.timing_comparison:
        spec = importlib.util.spec_from_file_location("_compact_timing_comparison", ROOT / "scripts/benchmark_compact_subscription.py")
        benchmark = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(benchmark)
        result = benchmark.run_benchmark(args.output, config=config, sim_config=sim_config,
            token_provider=token_provider, blocks=args.blocks, actions_per_block=args.actions_per_block)
        print(json.dumps(result), flush=True)
        return 0 if result["comparison"]["complete"] else 3
    result = execute_episode(args.output, config=config, sim_config=sim_config,
        token_provider=token_provider, offline_smoke=args.offline_smoke, field_representation=args.field_representation,
        reasoning_effort=args.reasoning_effort, stop_on_model_complete=args.stop_on_model_complete,
        controller_mode=args.controller_mode, execution_mode=args.execution_mode, feedback_history=args.feedback_history,
        centered_placement_guidance=args.centered_placement_guidance,
        grasp_settle_guidance=args.grasp_settle_guidance, field_improvements=args.field_improvements,
        distance_field_control=args.distance_field_control, task_key=args.task)
    return 0 if result["evaluation_completed"] or result["diagnostic_complete"] or result["model_declared_complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
