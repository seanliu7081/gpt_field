"""LIBERO-10 worker with record-only contacts and sensor-only geometry.

The frozen V2 worker supplies generic contact records to evaluator status. Its
truth, answer-key, oracle and counterfactual operations are not exposed here.
Policy geometry uses the unchanged Astranomy extractor and the state sampled
with the latest RGB observation, never the V2 evaluator geometry operation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from gpt_field.agent_speedup_v2_sim import V2LiberoWorker
from gpt_field.astranomy_agentic_geometry import extract_geometry_metadata
from gpt_field.libero import LiberoConfig


PROTOCOL_OPS = ("metadata", "tasks", "reset", "step", "geometry")
MONITOR_OPTIONS = {"goal_aware": True, "truth_window": False}


def validate_config(config):
    """Keep monitoring and termination within the approved run protocol."""
    if (config.suite != "libero_10" or config.collision_check_mode != "none"
            or config.stop_on_collision or not config.stop_on_success
            or config.max_steps is not None or not config.include_arm_joints):
        raise ValueError("LIBERO-10 requires raw-success stopping, record-only contacts, "
                         "no horizon, base collision mode none and measured arm joints")
    if config.initial_state_override is not None or config.task_bddl_override is not None:
        raise ValueError("LIBERO-10 uses official task files and initial states")
    return config


def serve(config, requests, protocol, *, worker_factory=None):
    """Serve the original JSONL sensor protocol; injectable streams allow offline tests."""
    validate_config(config)
    worker = None
    observation_state = None
    try:
        for line in requests:
            try:
                request = json.loads(line)
                if not isinstance(request, dict) or request.get("op") not in PROTOCOL_OPS:
                    raise ValueError("unsupported operation")
                op = request["op"]
                if worker is None:
                    worker = (worker_factory or V2LiberoWorker)(
                        config, monitor_options=dict(MONITOR_OPTIONS))
                if op == "metadata":
                    result = {**worker.metadata, "agentic_geometry_extension": True,
                              "evaluator_ops": [], "oracle_enabled": False,
                              "worker_protocol_ops": list(PROTOCOL_OPS),
                              "geometry_contract": "astranomy_synchronized_camera_and_fixed_robot"}
                elif op == "tasks":
                    result = {"tasks": worker.tasks()}
                elif op == "reset":
                    # This adapter is scoped to the three approved fresh episodes.
                    if (type(request.get("task_id")) is not int
                            or request["task_id"] not in (2, 6, 8)
                            or type(request.get("episode_index")) is not int
                            or request["episode_index"] != 0):
                        raise ValueError("approved scope is LIBERO-10 task 2, 6 or 8, initial state 0")
                    observation_state = None
                    result = worker.reset(request["task_id"], request["episode_index"])
                    observation_state = list(result["observation"]["observation/state"])
                elif op == "step":
                    if observation_state is None:
                        raise ValueError("Reset is required before stepping")
                    result = worker.step(request["action"])
                    observation_state = list(result["observation"]["observation/state"])
                else:
                    if observation_state is None or worker.env is None:
                        raise ValueError("Reset is required before requesting camera calibration")
                    result = extract_geometry_metadata(
                        worker.env, image_size=config.image_size,
                        observation_state=observation_state)
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            protocol.write(json.dumps(response, allow_nan=False) + "\n")
            protocol.flush()
    finally:
        if worker is not None:
            worker.close()
    return 0


def worker_main(config):
    # Keep JSON protocol on the original stdout and library logs on stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    try:
        return serve(config, sys.stdin, protocol)
    finally:
        protocol.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args(argv)
    return worker_main(LiberoConfig(**json.loads(args.config_json)))


if __name__ == "__main__":
    raise SystemExit(main())
