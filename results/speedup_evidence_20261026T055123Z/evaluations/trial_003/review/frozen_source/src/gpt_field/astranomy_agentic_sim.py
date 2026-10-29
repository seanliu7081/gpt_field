"""Isolated LIBERO worker extension exposing calibrated camera/robot facts.

The historical simulator and observation protocol remain unchanged. Only the
new Agentic runner can ask for geometry, separately from evaluator status.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .libero import LiberoConfig, _LiberoWorker


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
                    worker = _LiberoWorker(config)
                op = request["op"]
                if op == "metadata":
                    result = {**worker.metadata, "agentic_geometry_extension": True}
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
                    # robosuite caches RGB and proprioception at the same sensor
                    # tick, before the interval's final physics state. Derive
                    # moving-camera calibration from that observed robot pose.
                    result = extract_geometry_metadata(worker.env, image_size=config.image_size,
                                                       observation_state=observation_state)
                else:
                    raise ValueError(f"unknown operation: {op}")
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
