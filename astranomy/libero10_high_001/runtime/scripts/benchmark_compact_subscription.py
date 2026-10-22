"""Paired, finite subscription timing comparison: compact field versus direct actions."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np

from gpt_field.astra import APIError, BudgetExceeded, MissingCredentials, SchemaError
from gpt_field.compact_subscription import ChatGPTPlanCompactClient
from gpt_field.experiment import digest, source_manifest, write_json
from gpt_field.hybrid_policy import HybridFieldDirectPolicy
from gpt_field.libero import LiberoConfig, SimProcess

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_compact_timing_runtime", ROOT / "scripts/evaluate_bowl_subscription_close.py")
_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runtime)
TASK = _runtime.TASK
TIMING_DEFINITIONS = {
    "model_transport_s": "Wall time inside each subscription transport call, including network, server generation and stream parsing; excludes OAuth retrieval, payload construction and field expansion.",
    "field_expansion_s": "Client-measured compact-to-dense field expansion time for accepted field responses.",
    "local_planning_s": "Policy-reported planner time, counted only when a new local plan is created.",
    "policy_decision_s": "Wall time in policy.act, including model transport, expansion, planning, validation, adaptation and request logging.",
    "feedback_s": "Wall time in policy.observe_transition after acknowledged simulator execution.",
    "policy_total_s": "policy_decision_s + feedback_s; component transport/expansion/planning times overlap this total.",
    "simulator_step_s": "Wall time in sim.step, including simulator communication, execution and returned observation processing.",
    "control_loop_s": "policy_total_s + simulator_step_s; excludes simulator startup/reset, client construction, trace/summary writes and terminal cleanup. Request logging inside policy.act remains included.",
}


class TimedTransport:
    def __init__(self, transport, clock=time.perf_counter):
        self.transport, self.clock, self.calls = transport, clock, []

    def __call__(self, *args, **kwargs):
        started, completed = self.clock(), False
        try:
            response = self.transport(*args, **kwargs)
            completed = True
            return response
        finally:
            self.calls.append({"transport_s": max(0., self.clock() - started), "completed": completed})


def _comparison(arms, blocks, requested):
    pairs = []
    for replicate in range(blocks):
        pair = {arm["controller_mode"]: arm for arm in arms if arm["replicate"] == replicate}
        comparable = len(pair) == 2 and all(arm["status"] == "completed" and arm["actions"] == requested for arm in pair.values())
        item = {"replicate": replicate, "comparable": comparable}
        if comparable:
            field, direct = pair["field"], pair["direct"]
            item.update(field_control_loop_s=field["timing"]["control_loop_s"],
                        direct_control_loop_s=direct["timing"]["control_loop_s"])
            item["direct_over_field_ratio"] = (item["direct_control_loop_s"] / item["field_control_loop_s"]
                                                if item["field_control_loop_s"] else None)
        pairs.append(item)
    complete = len(arms) == 2 * blocks and all(pair["comparable"] for pair in pairs)
    aggregates = {}
    if complete:
        for mode in ("field", "direct"):
            subset = [arm for arm in arms if arm["controller_mode"] == mode]
            aggregates[mode] = {"actions": sum(arm["actions"] for arm in subset),
                "api_calls": sum(arm["api_calls"] for arm in subset),
                "local_plans": sum(arm["local_plans"] for arm in subset),
                "translation_hold_actions": sum(arm["translation_hold_actions"] for arm in subset),
                "timing_totals": {key: sum(arm["timing"][key] for arm in subset) for key in TIMING_DEFINITIONS},
                "median_control_loop_s_per_block": statistics.median(arm["timing"]["control_loop_s"] for arm in subset),
                "tokens": {key: sum(arm["budget"].get(key, 0) for arm in subset)
                           for key in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens")}}
        f = aggregates["field"]["timing_totals"]["control_loop_s"]
        d = aggregates["direct"]["timing_totals"]["control_loop_s"]
        aggregates["direct_over_field_ratio"] = d / f if f else None
    return {"complete": complete, "pairs": pairs, "aggregates": aggregates}


def _report(summary):
    lines = ["# Compact field versus direct: timing comparison", "", f"Status: **{summary['status']}**.", "",
             "Same task, initial state and seed; fresh model history per arm. Arm order alternates by paired replicate. Task success is not the comparison metric.", "",
             "| Replicate | Controller | Actions | Calls | Plans | Model transport (s) | Policy total (s) | Simulation (s) | Control loop (s) | Status |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm in summary["arms"]:
        t = arm["timing"]
        lines.append(f"| {arm['replicate'] + 1} | {arm['controller_mode']} | {arm['actions']} | {arm['api_calls']} | {arm['local_plans']} | {t['model_transport_s']:.3f} | {t['policy_total_s']:.3f} | {t['simulator_step_s']:.3f} | {t['control_loop_s']:.3f} | {arm['status']} |")
    comparison = summary["comparison"]
    lines += [""]
    if comparison["complete"]:
        ratio = comparison["aggregates"]["direct_over_field_ratio"]
        if ratio is not None:
            lines.append(f"Total direct / compact-field control-loop time: **{ratio:.3f}×** (values above 1 mean compact field used less time).")
    else:
        lines.append("No overall speed comparison: every paired arm must finish the same requested action count.")
    lines += ["", "Transport, expansion and planning are overlapping parts of policy time; do not add them to policy total.", ""]
    lines += [f"- `{key}`: {definition}" for key, definition in TIMING_DEFINITIONS.items()]
    return "\n".join(lines) + "\n"


def run_benchmark(output, *, config, sim_config, token_provider, blocks=3, actions_per_block=30,
                  sim_factory=None, client_factory=None, policy_factory=None, transport=None,
                  clock=time.perf_counter):
    """Run finite paired arms; stop the complete comparison on the first failure."""
    _runtime.validate_config(config, sim_config)
    for name, value in (("blocks", blocks), ("actions_per_block", actions_per_block)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not callable(token_provider):
        raise MissingCredentials("Timing comparison requires an explicit ChatGPT OAuth token provider")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Timing comparison requires a fresh output directory")
    sim_factory = sim_factory or SimProcess
    client_factory = client_factory or ChatGPTPlanCompactClient
    policy_factory = policy_factory or HybridFieldDirectPolicy
    output.mkdir(parents=True)
    overall_started, startup_s = clock(), 0.
    order = [{"replicate": index, "controller_mode": mode} for index in range(blocks)
             for mode in (("field", "direct") if index % 2 == 0 else ("direct", "field"))]
    identity = {"kind": "compact_subscription_timing_comparison", "task": TASK, "task_id": 0,
        "episode_index": 0, "seed": 7, "blocks": blocks, "actions_per_block": actions_per_block,
        "order": order, "runner_sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT / "scripts/evaluate_bowl_hybrid_subscription.py")}, "api": asdict(config), "simulator": asdict(sim_config), "software": source_manifest(),
        "billing": "chatgpt_subscription", "reasoning_effort": "high", "api_key_calls": 0,
        "api_key_fallback": False, "retries": False, "video": False, "success_is_metric": False,
        "field_representation": "analytic-goal-rbf-v1", "timing_definitions": TIMING_DEFINITIONS}
    write_json(output / "manifest.json", {"identity_hash": digest(identity), "configuration": identity,
        "created_utc": datetime.now(timezone.utc).isoformat()})
    arms, simulator, current, client, timed = [], None, None, None, None
    status, error, initial_state, initial_observation_digest = "running", None, None, None

    def snapshot(stage):
        comparison = _comparison(arms, blocks, actions_per_block)
        if status != "completed":
            comparison["complete"], comparison["aggregates"] = False, {}
        summary = {"status": status, "task": TASK, "blocks": blocks, "actions_per_block": actions_per_block,
            "arms": arms, "comparison": comparison, "error": error, "api_key_calls": 0,
            "success_is_metric": False, "timing_definitions": TIMING_DEFINITIONS,
            "simulator_startup_s": startup_s, "wall_elapsed_s": max(0., clock() - overall_started),
            "overhead_definition": "simulator_startup_s covers construction/start/task listing; each arm reset_s covers sim.reset. wall_elapsed_s includes all setup, control, trace writes and cleanup.",
            "output": str(output)}
        write_json(output / "summary.json", summary)
        write_json(output / "run_state.json", {"status": status, "stage": stage,
            "completed_arms": sum(arm["status"] == "completed" for arm in arms),
            "current_arm": {key: current[key] for key in ("replicate", "controller_mode", "actions")} if current else None,
            "error": error})
        (output / "comparison.md").write_text(_report(summary))
        return summary

    def settle_arm():
        if current is None:
            return
        timing = current["timing"]
        timing["model_transport_s"] = sum(call["transport_s"] for call in timed.calls) if timed else 0.
        timing["policy_total_s"] = timing["policy_decision_s"] + timing["feedback_s"]
        timing["control_loop_s"] = timing["policy_total_s"] + timing["simulator_step_s"]
        current["transport_calls"] = list(timed.calls) if timed else []
        if client is not None:
            current["budget"] = client.budget_snapshot()
            current["api_calls"] = current["budget"].get("api_calls", len(current["transport_calls"]))

    snapshot("starting")
    with _runtime._base.termination_as_interrupt():
        try:
            setup_started = clock()
            try:
                with _runtime._simulator_without_credentials():
                    simulator = sim_factory(sim_config)
                    simulator.start()
                tasks = simulator.list_tasks()
            finally:
                startup_s = max(0., clock() - setup_started)
            if not tasks or tasks[0].get("task_id") != 0 or tasks[0].get("prompt") != TASK:
                raise ValueError("Official task zero does not match the benchmark task")
            write_json(output / "simulator.json", {"metadata": simulator.metadata, "tasks": tasks})
            for index, specification in enumerate(order):
                client = timed = None
                current = {**specification, "arm_index": index, "status": "running", "actions": 0,
                    "api_calls": 0, "local_plans": 0, "translation_hold_actions": 0,
                    "zero_action_count": 0, "timing": {key: 0. for key in TIMING_DEFINITIONS}, "budget": {}}
                arms.append(current)
                arm_dir = output / f"arm_{index:02d}_{current['controller_mode']}"
                arm_dir.mkdir()
                snapshot("resetting")
                reset_started = clock()
                try:
                    obs = simulator.reset(0, 0)
                finally:
                    current["reset_s"] = max(0., clock() - reset_started)
                if obs.prompt != TASK:
                    raise ValueError("Reset observation has the wrong task")
                current["initial_state"] = list(obs.state)
                current["initial_observation_digest"] = digest(obs.to_dict())
                if initial_state is None:
                    initial_state = np.asarray(obs.state)
                    initial_observation_digest = current["initial_observation_digest"]
                elif (not np.allclose(initial_state, obs.state, rtol=0., atol=1e-9)
                      or initial_observation_digest != current["initial_observation_digest"]):
                    raise ValueError("Paired arms did not reset to identical initial observations")
                client = client_factory(config, controller_mode=current["controller_mode"],
                    token_provider=token_provider, transport=transport, log_dir=arm_dir / "requests")
                timed = TimedTransport(client.transport, clock)
                client.transport = timed
                policy = policy_factory(client)
                policy.reset()
                current.update(prompt_version=client.prompt_version, field_representation=client.field_representation,
                    planner=asdict(policy.planner.config), field_update_steps=policy.field_update_steps,
                    direct_action_steps=policy.direct_action_steps, field_execution_steps=policy.field_execution_steps)
                with (arm_dir / "trace.jsonl").open("w") as trace:
                    for step in range(actions_per_block):
                        if simulator.status.get("terminated") or simulator.status.get("success"):
                            current["status"] = status = "incomplete_simulator_termination"
                            break
                        snapshot("deciding")
                        began = clock()
                        try:
                            decision = policy.act(obs)
                        finally:
                            current["timing"]["policy_decision_s"] += max(0., clock() - began)
                        details = decision.diagnostics
                        expected = "coarse_field" if current["controller_mode"] == "field" else "contact_direct"
                        if details["controller"] != expected:
                            raise SchemaError("Forced benchmark controller changed mode")
                        if details.get("refreshed") and expected == "coarse_field":
                            current["timing"]["field_expansion_s"] += client.last_field_expansion_s
                        planned = details.get("plan_refreshed", "plan" in details)
                        current["local_plans"] += int(planned)
                        if planned:
                            current["timing"]["local_planning_s"] += details.get("plan_latency_s", 0.)
                        before = obs
                        began = clock()
                        try:
                            obs, _ = simulator.step(decision.action)
                        finally:
                            current["timing"]["simulator_step_s"] += max(0., clock() - began)
                        current["actions"] += 1
                        current["translation_hold_actions"] += int(all(abs(value) <= 1e-12 for value in decision.action[:3]))
                        current["zero_action_count"] += int(all(abs(value) <= 1e-12 for value in decision.action))
                        began = clock()
                        try:
                            feedback = policy.observe_transition(before, decision, obs)
                        finally:
                            current["timing"]["feedback_s"] += max(0., clock() - began)
                        recorded = {key: details[key] for key in (
                            "controller", "refreshed", "chunk_index", "chunk_step", "plan_refreshed", "plan_start_step",
                            "plan_step", "plan_execution_steps", "planned_delta_xyz", "plan_latency_s", "api_latency_s") if key in details}
                        if planned:
                            recorded["plan"] = details["plan"]
                        if details.get("refreshed"):
                            recorded["compact_prediction"] = client.last_compact_prediction
                        trace.write(json.dumps({"step": step, "state_before": list(before.state), "state_after": list(obs.state),
                            "action": list(decision.action), "diagnostics": recorded, "feedback": feedback}, allow_nan=False) + "\n")
                        trace.flush()
                        settle_arm()
                        snapshot("executed")
                        print(json.dumps({"event": "benchmark_action", "replicate": current["replicate"],
                            "controller": current["controller_mode"], "actions": current["actions"],
                            "calls": current["api_calls"], "control_loop_s": current["timing"]["control_loop_s"]}), flush=True)
                settle_arm()
                if current["status"] != "running":
                    break
                current["status"] = "completed"
                write_json(arm_dir / "summary.json", current)
                snapshot("arm_completed")
            else:
                status = "completed"
        except KeyboardInterrupt as exc:
            status, error = "user_stop", _runtime._safe_error(exc, "benchmark")
        except (APIError, BudgetExceeded, MissingCredentials, SchemaError) as exc:
            status, error = "subscription_error", _runtime._safe_error(exc, "benchmark")
        except Exception as exc:
            status, error = "infrastructure_error", _runtime._safe_error(exc, "benchmark")
        finally:
            settle_arm()
            if current is not None and current["status"] == "running":
                current["status"] = status
            if simulator is not None:
                try:
                    simulator.close()
                except Exception as exc:
                    status, error = "infrastructure_error", _runtime._safe_error(exc, "simulator_close")
            if current is not None:
                write_json(output / f"arm_{current['arm_index']:02d}_{current['controller_mode']}" / "summary.json", current)
            summary = snapshot("finished")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--actions-per-block", type=int, default=30)
    parser.add_argument("--sim-python", default=_runtime.DEFAULT_SIM_PYTHON)
    parser.add_argument("--libero-root", default=_runtime.DEFAULT_LIBERO_ROOT)
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--profile")
    args = parser.parse_args(argv)
    from gpt_field.chatgpt_auth import SubscriptionCredentials
    credentials = SubscriptionCredentials(store_dir=args.store_dir, profile=args.profile)
    sim = LiberoConfig(suite="libero_spatial", seed=7, max_steps=None,
        libero_root=str(Path(args.libero_root).expanduser().resolve()),
        python_executable=os.path.abspath(os.path.expanduser(args.sim_python)))
    result = run_benchmark(args.output, config=_runtime.configuration(), sim_config=sim,
        token_provider=credentials.access_token, blocks=args.blocks, actions_per_block=args.actions_per_block)
    print(json.dumps(result), flush=True)
    return 0 if result["comparison"]["complete"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
