"""Isolated LIBERO worker extension exposing calibrated camera/robot facts.

The historical monitored tasks retain their worker. New tasks use the generic
record-only monitor without changing controller limits or policy observations.
Camera calibration and evaluator-only truth use separate protocol operations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .libero import LiberoConfig, _LiberoWorker


def _make_worker(config):
    if config.collision_check_mode == "none":
        from .agent_speedup_v2_sim import V2LiberoWorker
        return V2LiberoWorker(config)
    return _LiberoWorker(config)


def _dispatch(worker, request, config, observation_state):
    """Return one reply and the last observed pose; truth never enters observations."""
    op = request["op"]
    if op == "metadata":
        truth_available = callable(getattr(worker, "truth", None))
        result = {**worker.metadata, "agentic_geometry_extension": True,
                  "agentic_worker_backend": ("V2LiberoWorker" if config.collision_check_mode == "none"
                                             else "_LiberoWorker"),
                  "evaluator_truth_available": truth_available,
                  "evaluator_ops": ["truth"] if truth_available else []}
    elif op == "tasks":
        result = {"tasks": worker.tasks()}
    elif op == "reset":
        result = worker.reset(request["task_id"], request["episode_index"])
        observation_state = list(result["observation"]["observation/state"])
    elif op == "step":
        result = worker.step(request["action"])
        observation_state = list(result["observation"]["observation/state"])
    elif op == "geometry":
        if worker.env is None:
            raise ValueError("Reset is required before requesting camera calibration")
        from .astranomy_agentic_geometry import extract_geometry_metadata
        # robosuite caches RGB and proprioception at the same sensor tick,
        # before the interval's final physics state. Use that observed pose.
        result = extract_geometry_metadata(worker.env, image_size=config.image_size,
                                           observation_state=observation_state)
    elif op == "truth":
        truth = getattr(worker, "truth", None)
        if not callable(truth):
            raise ValueError("Evaluator truth is unavailable for this worker")
        result = truth()
    else:
        raise ValueError(f"unknown operation: {op}")
    return result, observation_state


def worker_main(config):
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker = None
    observation_state = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if worker is None:
                    worker = _make_worker(config)
                result, observation_state = _dispatch(worker, request, config, observation_state)
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            protocol.write(json.dumps(response, allow_nan=False) + "\n")
            protocol.flush()
    finally:
        if worker is not None:
            worker.close()
        protocol.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    return worker_main(LiberoConfig(**json.loads(args.config_json)))


if __name__ == "__main__":
    raise SystemExit(main())
