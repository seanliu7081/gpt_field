"""User-check simulator adapter: native v2 evaluation plus calibrated RGB facts.

The model-visible extension is separate from v2's evaluator-only ``geometry``
operation. It contains only the allowlisted Astranomy camera/fixed robot facts,
with wrist calibration derived from the exact state8 accompanying the RGB.
Native observation serialization is unchanged, including its frozen reset hash.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from numbers import Real
import os
import sys

from .agent_speedup_v2_sim import V2LiberoWorker, V2SimProcess, _json_line
from .astranomy_agentic_geometry import extract_geometry_metadata, validate_geometry_metadata
from .libero import LiberoConfig
from .observation import PolicyObservation


def _robot_proprioception(value):
    """Validate only the two measured seven-joint channels; reject extra fields."""
    if not isinstance(value, dict) or set(value) != {"joint_position", "joint_velocity"}:
        raise ValueError("Robot proprioception requires only joint position and velocity")
    clean = {}
    for key, values in value.items():
        if (not isinstance(values, (list, tuple)) or len(values) != 7
                or any(isinstance(x, bool) or not isinstance(x, Real)
                       or not math.isfinite(float(x)) for x in values)):
            raise ValueError("Robot joint sensors require seven finite real values")
        clean[key] = tuple(float(x) for x in values)
    return clean


class UserCheckLiberoWorker(V2LiberoWorker):
    def _response(self, obs):
        response = super()._response(obs)
        # Read the same cached observation supplied by env.step(), never the
        # later simulator endpoint. No scene/object channels are copied.
        response["robot_proprioception"] = _robot_proprioception({
            "joint_position": list(obs["robot0_joint_pos"]),
            "joint_velocity": list(obs["robot0_joint_vel"]),
        })
        return response


@dataclass(frozen=True)
class CalibratedObservation(PolicyObservation):
    geometry: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "geometry", validate_geometry_metadata(self.geometry))

    def to_dict(self):
        # Approval/frozen reset identity remains the original state8 protocol.
        value = super().to_dict()
        value.pop("observation/joint_position", None)
        value.pop("observation/joint_velocity", None)
        return value


class UserCheckSimProcess(V2SimProcess):
    """Keep evaluator transport/cleanup unchanged; attach a separate sensor fact."""

    def __init__(self, config, *, monitor_options=None):
        super().__init__(config, monitor_options=monitor_options)
        self._robot_sensors = None
        self._command = [config.python_executable or sys.executable, "-u", "-m",
                         "gpt_field.agent_speedup_v2_user_check_sim", "--worker",
                         "--config-json", json.dumps(asdict(config)),
                         "--monitor-json", json.dumps(self.monitor_options)]

    def _request(self, request):
        result = super()._request(request)
        if request.get("op") in ("reset", "step"):
            self._robot_sensors = _robot_proprioception(result["robot_proprioception"])
        return result

    def _observation(self, data):
        native = super()._observation(data)
        geometry = validate_geometry_metadata(self._request({"op": "agent_geometry"}))
        return CalibratedObservation(
            image=native.image, wrist_image=native.wrist_image, state=native.state,
            prompt=native.prompt, joint_position=self._robot_sensors["joint_position"],
            joint_velocity=self._robot_sensors["joint_velocity"], geometry=geometry)


def worker_main(config, monitor_options):
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker = None
    observation_state = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if worker is None:
                    worker = UserCheckLiberoWorker(config, monitor_options)
                op = request["op"]
                if op == "metadata":
                    result = {**worker.metadata, "user_check_camera_geometry": True,
                              "model_geometry_op": "agent_geometry",
                              "model_robot_proprioception": "cached_observation_seven_arm_joints"}
                elif op == "tasks":
                    result = {"tasks": worker.tasks()}
                elif op == "reset":
                    result = worker.reset(request["task_id"], request["episode_index"])
                    observation_state = list(result["observation"]["observation/state"])
                elif op == "step":
                    result = worker.step(request["action"])
                    observation_state = list(result["observation"]["observation/state"])
                elif op == "agent_geometry":
                    if observation_state is None:
                        raise ValueError("Reset is required before camera calibration")
                    result = extract_geometry_metadata(
                        worker.env, image_size=config.image_size,
                        observation_state=observation_state)
                elif op == "truth":
                    result = worker.truth()
                elif op == "settle_truth":
                    result = worker.settle_truth(request.get("protocol"))
                elif op == "answer_key":
                    result = worker.answer_key()
                elif op == "geometry":
                    result = worker.geometry()
                elif op == "oracle":
                    result = worker.oracle(request.get("request"))
                else:
                    raise ValueError(f"unknown operation: {op}")
                response = {"ok": True, "result": result}
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            protocol.write(_json_line(response) + "\n")
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
    parser.add_argument("--monitor-json", required=True)
    args = parser.parse_args()
    return worker_main(LiberoConfig(**json.loads(args.config_json)),
                       json.loads(args.monitor_json))


if __name__ == "__main__":
    raise SystemExit(main())
