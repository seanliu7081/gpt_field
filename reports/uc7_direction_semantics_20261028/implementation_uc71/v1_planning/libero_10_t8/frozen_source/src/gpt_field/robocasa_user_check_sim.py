"""Isolated RoboCasa CoffeePressButton transport for the v2 user-check policy.

Only RGB, state8, seven measured arm joints and robot/camera calibration cross
the policy boundary. Task predicates, contacts and reset snapshots stay in
evaluator responses. The mobile base receives zero velocity; the torso holds
its reset position with its own controller. Nothing is teleported after reset.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import selectors
import subprocess
import sys
import time

import numpy as np

from .libero import SimProcess, SimulatorError
from .observation import PolicyObservation, image_to_data_url, quaternion_to_axisangle


SIM_VERSION = "robocasa-coffee-user-check-sim-v1"
EXTERNAL_CAMERAS = ("robot0_agentview_left", "robot0_agentview_center", "robot0_agentview_right")
ARM_ONLY_TRANSLATION_TOLERANCE_M = 0.001
ARM_ONLY_YAW_TOLERANCE_RAD = 0.001


@dataclass(frozen=True)
class RoboCasaConfig:
    task: str = "CoffeePressButton"
    seed: int = 7
    layout_id: int = 0
    style_id: int = 0
    image_size: int = 512
    external_camera: str = "robot0_agentview_left"
    python_executable: str = "/venv/oat/bin/python"
    robocasa_root: str = "/workspace/past_action_robocasa/robocasa"
    robosuite_root: str = "/workspace/past_action_robocasa/robosuite"
    response_timeout_s: float = 300.0
    stop_on_success: bool = True

    def __post_init__(self):
        if self.task != "CoffeePressButton":
            raise ValueError("This adapter is scoped to CoffeePressButton")
        if self.image_size != 512 or self.external_camera not in EXTERNAL_CAMERAS:
            raise ValueError("Use two native 512px RGBs with a standard RoboCasa camera")
        for name in ("seed", "layout_id", "style_id"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not math.isfinite(self.response_timeout_s) or self.response_timeout_s <= 0:
            raise ValueError("response_timeout_s must be positive")
        if type(self.stop_on_success) is not bool:
            raise ValueError("stop_on_success must be boolean")


def validate_arm_action(action):
    if not isinstance(action, (list, tuple, np.ndarray)) or len(action) != 7:
        raise ValueError("Arm action requires seven finite values")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.number))
           or not math.isfinite(float(x)) or not -1 <= float(x) <= 1 for x in action):
        raise ValueError("Arm actions must be finite real values in [-1,1]")
    return [float(x) for x in action]


def expand_arm_action(action, indexes, action_dim, torso_position):
    """Map arm7 using declared controller parts; never infer mobile indices."""
    values = validate_arm_action(action)
    expected = {"right": 6, "right_gripper": 1, "base": 3, "torso": 1}
    if set(indexes) != set(expected) or action_dim != 12:
        raise ValueError("Unexpected PandaOmron composite controller layout")
    occupied = []
    for part, width in expected.items():
        start, end = indexes[part]
        if type(start) is not int or type(end) is not int or end - start != width:
            raise ValueError("Unexpected controller part dimensions")
        occupied.extend(range(start, end))
    if sorted(occupied) != list(range(action_dim - 1)):
        raise ValueError("Controller parts overlap or include mobile-mode slot")
    torso = np.asarray(torso_position, dtype=float)
    if torso.shape != (1,) or not np.isfinite(torso).all():
        raise ValueError("Torso hold requires one finite reset position")
    full = np.zeros(action_dim, dtype=float)
    full[slice(*indexes["right"])] = values[:6]
    full[slice(*indexes["right_gripper"])] = values[6:]
    full[slice(*indexes["torso"])] = torso
    full[-1] = -1.0  # achieved-pose arm mode, not desired-pose mobile mode
    return full


def validate_arm_controller(controller):
    """Fail closed if the installed OSC has different action semantics."""
    actual = {"class": type(controller).__name__, "input_type": controller.input_type,
              "input_ref_frame": controller.input_ref_frame,
              "control_dim": int(controller.control_dim),
              "input_min": np.asarray(controller.input_min).tolist(),
              "input_max": np.asarray(controller.input_max).tolist(),
              "output_min": np.asarray(controller.output_min).tolist(),
              "output_max": np.asarray(controller.output_max).tolist()}
    expected_scale = np.array([0.05] * 3 + [0.5] * 3)
    if (actual["class"] != "OperationalSpaceController" or actual["input_type"] != "delta"
            or actual["input_ref_frame"] != "world" or actual["control_dim"] != 6
            or not np.array_equal(actual["input_min"], [-1] * 6)
            or not np.array_equal(actual["input_max"], [1] * 6)
            or not np.array_equal(actual["output_min"], -expected_scale)
            or not np.array_equal(actual["output_max"], expected_scale)):
        raise ValueError("Actual OSC controller does not match world-delta arm7 contract")
    return actual


def validate_arm_only_displacement(base_delta, torso_delta):
    base = np.asarray(base_delta, dtype=float)
    torso = np.asarray(torso_delta, dtype=float)
    if (base.shape != (3,) or torso.shape != (1,) or not np.isfinite(base).all()
            or not np.isfinite(torso).all()
            or np.linalg.norm(base[:2]) > ARM_ONLY_TRANSLATION_TOLERANCE_M
            or abs(base[2]) > ARM_ONLY_YAW_TOLERANCE_RAD
            or abs(torso[0]) > ARM_ONLY_TRANSLATION_TOLERANCE_M):
        raise RuntimeError("Arm-only contract violated: base/torso moved beyond 1mm or yaw beyond 0.001rad")


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class RoboCasaWorker:
    def __init__(self, config):
        self.config = config
        self.env = None
        self.status = {}
        self._initial = None
        self._metadata = {
            "benchmark": "RoboCasa", "task": config.task, "robot": "PandaOmron",
            "sim_version": SIM_VERSION, "seed": config.seed,
            "layout_id": config.layout_id, "style_id": config.style_id,
            "image_size": config.image_size, "external_camera": config.external_camera,
            "wrist_camera": "robot0_eye_in_hand", "control_freq_hz": 20,
            "action_interface": "world_delta_xyz_axisangle_gripper_7",
            "translation_scale_m": 0.05, "rotation_scale_rad": 0.5,
            "base_control": "zero_velocity", "torso_control": "absolute_reset_position",
            "mobile_mode": -1, "success_qualification": "raw_robocasa_check_success",
            "episode_action_cap": None, "episode_time_cap": None,
            "contact_policy": "record_only", "contact_sampling": "control_endpoint",
            "observation_sync": "fresh_endpoint_rgb_robot_sensors_camera_calibration",
            "arm_only_tolerance": {"base_translation_m": ARM_ONLY_TRANSLATION_TOLERANCE_M,
                "base_yaw_rad": ARM_ONLY_YAW_TOLERANCE_RAD,
                "torso_translation_m": ARM_ONLY_TRANSLATION_TOLERANCE_M,
                "violation": "infrastructure_error_not_task_outcome"},
        }

    @property
    def metadata(self):
        return dict(self._metadata)

    def reset(self):
        self.close()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        import robosuite
        import robocasa
        import mujoco
        from robosuite import macros
        from robosuite.controllers import load_composite_controller_config

        for package, expected in ((robosuite, self.config.robosuite_root),
                                  (robocasa, self.config.robocasa_root)):
            if not Path(package.__file__).resolve().is_relative_to(Path(expected).resolve()):
                raise RuntimeError(f"Unexpected {package.__name__} installation: {package.__file__}")

        # Native OpenCV convention is the raw MuJoCo RGB flipped vertically.
        macros.IMAGE_CONVENTION = "opencv"
        cfg = load_composite_controller_config(robot="PandaOmron")
        cfg["body_parts"]["right"]["input_ref_frame"] = "world"
        # Keep standard delta torso semantics during RoboCasa's own reset
        # settling. A zero absolute action there would lower the torso.
        self.env = robosuite.make(
            env_name=self.config.task, robots="PandaOmron", controller_configs=cfg,
            camera_names=[self.config.external_camera, "robot0_eye_in_hand"],
            camera_widths=512, camera_heights=512, camera_depths=False,
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            use_object_obs=False, ignore_done=True, reward_shaping=False,
            control_freq=20, seed=self.config.seed, layout_ids=[self.config.layout_id],
            style_ids=[self.config.style_id], initialization_noise=None,
            randomize_cameras=False, generative_textures=None, translucent_robot=False)
        self.env.reset()
        robot = self.env.robots[0]
        actual_arm = validate_arm_controller(robot.part_controllers["right"])
        self.indexes = {k: tuple(int(x) for x in v)
                        for k, v in robot.composite_controller._action_split_indexes.items()}
        self.torso_controller = robot.part_controllers["torso"]
        self.torso_controller.input_type = "absolute"
        self.torso_position = np.array(self.env.sim.data.qpos[self.torso_controller.qpos_index], copy=True)
        self.torso_controller.set_goal(self.torso_position.copy())
        self.base_controller = robot.part_controllers["base"]
        base_joints = [self.env.sim.model.joint_name2id(name) for name in robot.robot_model.base_joints]
        if np.asarray(self.env.sim.model.jnt_type)[base_joints].tolist() != [2, 2, 3]:
            raise ValueError("Unsupported base joint order; expected slide, slide, hinge")
        self.base_position = np.array(self.env.sim.data.qpos[self.base_controller.qpos_index], copy=True)
        expand_arm_action([0] * 7, self.indexes, self.env.action_dim, self.torso_position)
        self.prompt = self.env.get_ep_meta()["lang"]  # only official task language
        self.status = {"steps": 0, "terminated": False}
        self._metadata.update(robosuite_source=robosuite.__file__, robocasa_source=robocasa.__file__,
                              controller=cfg, action_split_indexes=self.indexes,
                              simulator_action_dimension=self.env.action_dim,
                              actual_arm_controller=actual_arm,
                              actual_torso_input_type=self.torso_controller.input_type,
                              base_joint_names=robot.robot_model.base_joints,
                              robosuite_version=getattr(robosuite, "__version__", "unknown"),
                              robocasa_version=getattr(robocasa, "__version__", "unknown"),
                              mujoco_version=mujoco.__version__)
        response = self._response()
        self._initial = self.snapshot(include_xml=True)
        return response

    def _fresh_observation(self):
        from .agent_speedup_v2_contacts import forward_keeping_warmstart
        forward_keeping_warmstart(self.env.sim)
        return self.env._get_observations(force_update=True)

    def _contacts(self):
        # Raw endpoint contacts, without an invented task-specific collision rule.
        model, data = self.env.sim.model, self.env.sim.data
        records = []
        for i in range(int(data.ncon)):
            contact = data.contact[i]
            if float(contact.dist) <= float(contact.includemargin):
                records.append({"geom1": model.geom_id2name(int(contact.geom1)),
                                "geom2": model.geom_id2name(int(contact.geom2)),
                                "distance_m": float(contact.dist)})
        return {"mode": "raw_endpoint_contacts", "policy": "record_only",
                "count": len(records), "records": records}

    def _response(self):
        from .robocasa_user_check_geometry import extract_robocasa_geometry
        obs = self._fresh_observation()
        state = tuple(float(x) for x in (*obs["robot0_eef_pos"],
                      *quaternion_to_axisangle(obs["robot0_eef_quat"]), *obs["robot0_gripper_qpos"]))
        observation = PolicyObservation(
            # The existing v2 transport stores horizontally mirrored upright
            # images; StepView and its video layer undo that mirror exactly.
            image=image_to_data_url(obs[f"{self.config.external_camera}_image"][:, ::-1], 512, rotate=False),
            wrist_image=image_to_data_url(obs["robot0_eye_in_hand_image"][:, ::-1], 512, rotate=False),
            state=state, prompt=self.prompt,
            joint_position=tuple(float(x) for x in obs["robot0_joint_pos"]),
            joint_velocity=tuple(float(x) for x in obs["robot0_joint_vel"]))
        geometry = extract_robocasa_geometry(self.env, image_size=512,
                    observation_state=state, external_camera=self.config.external_camera)
        raw = bool(self.env._check_success())
        base_delta = np.asarray(self.env.sim.data.qpos[self.base_controller.qpos_index]) - self.base_position
        torso_delta = np.asarray(self.env.sim.data.qpos[self.torso_controller.qpos_index]) - self.torso_position
        self.status.update(success=raw, robocasa_success=raw,
                           terminated=bool(raw and self.config.stop_on_success),
                           contacts=self._contacts(),
                           base_displacement_from_reset=base_delta.tolist(),
                           torso_displacement_from_reset_m=torso_delta.tolist())
        try:
            validate_arm_only_displacement(base_delta, torso_delta)
        except RuntimeError:
            self.status.update(terminated=True, infrastructure_error="arm_only_displacement_limit")
            raise
        return {"observation": observation.to_dict(), "geometry": geometry,
                "status": dict(self.status)}

    def step(self, action):
        if self.env is None or self.status.get("terminated"):
            raise RuntimeError("Reset is required before stepping")
        full = expand_arm_action(action, self.indexes, self.env.action_dim, self.torso_position)
        self.env.step(full)
        self.status["steps"] += 1
        return self._response()

    def snapshot(self, include_xml=False):
        """Evaluator-only physical replay evidence; never attached to observations."""
        if self.env is None:
            raise RuntimeError("Reset is required before snapshot")
        data = self.env.sim.data
        native = getattr(data, "_data", data)
        physical = {name: np.asarray(getattr(native, name)).tolist()
                    for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")}
        physical["time"] = float(data.time)
        xml = self.env.sim.model.get_xml()
        result = {"physical": physical, "physical_sha256": _digest(physical),
                  "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
                  "episode_metadata": self.env.get_ep_meta(), "status": self.status,
                  "base_reset_qpos": self.base_position.tolist(),
                  "torso_reset_qpos": self.torso_position.tolist(),
                  "task_internal_state": self.env.coffee_machine.get_state()}
        if include_xml:
            result["model_xml"] = xml
        return _jsonable(result)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None


class RoboCasaSimProcess(SimProcess):
    """Reuse JSONL framing only; no LIBERO config, tasks, status or reset logic."""

    def __init__(self, config=None, *, command=None):
        config = config or RoboCasaConfig()
        super().__init__(config, command=command or [config.python_executable, "-u", "-m",
                         "gpt_field.robocasa_user_check_sim", "--worker", "--config-json",
                         json.dumps(asdict(config))])
        self._in_flight = False

    def _request(self, request):
        self._in_flight = True
        try:
            result = super()._request(request)
        except Exception:
            self._in_flight = False
            raise
        self._in_flight = False
        return result

    def start(self):
        if self._process is not None:
            return self
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).resolve().parents[1]),
                             self.config.robosuite_root, self.config.robocasa_root,
                             env.get("PYTHONPATH", "")])
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        self._process = subprocess.Popen(self._command, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, env=env, bufsize=0, start_new_session=True)
        try:
            self.metadata = self._request({"op": "metadata"})
        except BaseException:
            self.close()
            raise
        return self

    def _consume_observation(self, result):
        from .agent_speedup_v2_user_check_sim import CalibratedObservation
        native = PolicyObservation.from_dict(result["observation"])
        self.status = result["status"]
        return CalibratedObservation(image=native.image, wrist_image=native.wrist_image,
                state=native.state, prompt=native.prompt, joint_position=native.joint_position,
                joint_velocity=native.joint_velocity, geometry=result["geometry"])

    def reset(self):
        self.start()
        result = self._request({"op": "reset"})
        observation = self._consume_observation(result)
        self.metadata = self._request({"op": "metadata"})
        return observation

    def step(self, action):
        if not self.status or self.status.get("terminated"):
            raise SimulatorError("Reset is required before stepping an inactive episode")
        result = self._request({"op": "step", "action": validate_arm_action(action)})
        observation = self._consume_observation(result)
        return observation, bool(self.status["robocasa_success"])

    def list_tasks(self):
        return [{"task": self.config.task, "benchmark": "RoboCasa"}]

    def snapshot(self, *, include_xml=False):
        return self._request({"op": "snapshot", "include_xml": bool(include_xml)})

    def save_reset(self, path):
        result = self._request({"op": "initial_snapshot"})
        path = Path(path)
        with path.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        return {"path": str(path), "physical_sha256": result["physical_sha256"],
                "model_xml_sha256": result["model_xml_sha256"]}

    def close(self):
        proc = self._process
        if proc is not None and proc.stdin is not None and proc.stdout is not None:
            try:
                proc.stdin.close()
                deadline = time.monotonic() + 3.0
                with selectors.DefaultSelector() as selector:
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    while (remaining := deadline - time.monotonic()) > 0 and selector.select(remaining):
                        if not os.read(proc.stdout.fileno(), 1 << 20):
                            break
            except (OSError, ValueError):
                pass
        super().close()


def worker_main(config):
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker = RoboCasaWorker(config)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request["op"]
                if op == "metadata":
                    result = worker.metadata
                elif op == "reset":
                    result = worker.reset()
                elif op == "step":
                    result = worker.step(request["action"])
                elif op == "snapshot":
                    result = worker.snapshot(include_xml=request.get("include_xml", False))
                elif op == "initial_snapshot":
                    if worker._initial is None:
                        raise ValueError("Reset is required before initial snapshot")
                    result = worker._initial
                else:
                    raise ValueError(f"Unknown operation: {op}")
                response = {"ok": True, "result": result}
            except Exception as exc:
                import traceback
                traceback.print_exc(file=sys.stderr)
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            protocol.write(json.dumps(_jsonable(response), allow_nan=False) + "\n")
            protocol.flush()
    finally:
        worker.close()
        protocol.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, action="store_true")
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    worker_main(RoboCasaConfig(**json.loads(args.config_json)))


if __name__ == "__main__":
    main()
