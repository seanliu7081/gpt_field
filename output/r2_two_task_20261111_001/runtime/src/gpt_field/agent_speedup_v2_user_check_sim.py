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

from .action import TRANSLATION_SCALE_M
from .agent_speedup_v2_sim import V2LiberoWorker, V2SimProcess, _json_line
from .astranomy_agentic_geometry import extract_geometry_metadata
from .robocasa_user_check_geometry import validate_geometry_metadata
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
    def __init__(self, config, monitor_options=None):
        self._ordered_progress = None
        self._ordered_task_key = None
        self._ordered_observed_steps = 0
        super().__init__(config, monitor_options)
        self.metadata["user_check_osc_translation_scale_m"] = TRANSLATION_SCALE_M
        if config.custom_task_prompt is not None:
            from .agent_speedup_v2_tasks import custom_v2_task_contract, resolve_custom_v2_task
            from .ordered_task_evaluation import ORDERED_KIND, ORDERED_EVALUATOR_VERSION
            spec = resolve_custom_v2_task(config.suite, config.custom_task_id, config.custom_task_prompt)
            contract = custom_v2_task_contract(spec.key)
            if contract and contract.get("ordered_success_kind") == ORDERED_KIND:
                self._ordered_task_key = spec.key
                self.metadata.update(success_qualification="ordered_custom_goal",
                                     ordered_task_key=spec.key,
                                     ordered_evaluator_version=ORDERED_EVALUATOR_VERSION,
                                     ordered_progress_policy_visible=False)

    def reset(self, task_id, episode_index):
        # Base reset performs its ten passive settling actions. They never
        # establish ordered milestones, including on a reused worker.
        self._ordered_progress = None
        self._ordered_observed_steps = 0
        response = super().reset(task_id, episode_index)
        if TRANSLATION_SCALE_M != 0.05:
            # Apply after the stock zero-action reset/settle, preserving the
            # saved initial observation. Gains, rotations and timing stay fixed.
            for robot in self.env.robots:
                controller = robot.controller
                controller.output_min[:3] = -TRANSLATION_SCALE_M
                controller.output_max[:3] = TRANSLATION_SCALE_M
                # Settling has already populated robosuite's scaling cache.
                controller.action_scale = None
                controller.action_input_transform = None
                controller.action_output_transform = None
            self.metadata["controller"] = {
                **self.metadata["controller"],
                "output_min": self.env.robots[0].controller.output_min.tolist(),
                "output_max": self.env.robots[0].controller.output_max.tolist(),
            }
            self.metadata["action_order"][:3] = [
                f"d{axis}/{TRANSLATION_SCALE_M:g}" for axis in "xyz"]
            self.metadata["user_check_controller_override"] = {
                "translation_scale_m": TRANSLATION_SCALE_M,
                "applied_after_reset": True,
                "source_config_unchanged": True,
            }
        if self._ordered_task_key is not None:
            from .ordered_task_evaluation import OrderedProgress, drawer_bowl_moka_predicates
            self._ordered_progress = OrderedProgress()
            baseline = self._ordered_progress.reset(drawer_bowl_moka_predicates(self.env))
            if not baseline["initial_state_valid"]:
                raise RuntimeError("Registered ordered task reset violates its initial predicates")
            self._qualify_ordered_status()
            response["status"] = self.status.copy()
        return response

    def _raw_success(self):
        raw = super()._raw_success()
        progress = getattr(self, "_ordered_progress", None)
        if progress is not None and not self.counterfactual_applied:
            steps = self.status["steps"]
            if steps > self._ordered_observed_steps:
                if steps != self._ordered_observed_steps + 1:
                    raise RuntimeError("Ordered evaluator missed an actual policy step")
                from .ordered_task_evaluation import drawer_bowl_moka_predicates
                # V2 calls here after base step and before its monitor refresh,
                # matching the kinematics of the raw success evaluation.
                progress.observe(drawer_bowl_moka_predicates(self.env))
                self._ordered_observed_steps = steps
        return raw

    def _qualify_ordered_status(self):
        from .ordered_task_evaluation import qualify_status
        self.status = qualify_status(self.status, self._ordered_progress, self._ordered_task_key,
                                     stop_on_success=self.config.stop_on_success)

    def step(self, action):
        response = super().step(action)
        if self._ordered_progress is not None:
            self._qualify_ordered_status()
            response["status"] = self.status.copy()
        return response

    def truth(self):
        result = super().truth()
        if self._ordered_progress is not None:
            from .ordered_task_evaluation import qualify_truth
            result = qualify_truth(result, self._ordered_progress, self._ordered_task_key)
        return result

    def settle_truth(self, protocol=None):
        result = super().settle_truth(protocol)
        if self._ordered_progress is not None:
            from .ordered_task_evaluation import qualify_truth
            result = qualify_truth(result, self._ordered_progress, self._ordered_task_key)
        return result

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

    def reset(self, task_id, episode_index):
        observation = super().reset(task_id, episode_index)
        if TRANSLATION_SCALE_M != 0.05:
            self.metadata = self._request({"op": "metadata"})
        return observation

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
