"""Isolated RoboCasa365 UC7.1 transport. Importing never starts a simulator.

Arm commands retain world-delta arm7. A separate mobile_step carries bounded
native base/torso commands while keeping the gripper. Task truth stays in status.
The native HYBRID_MOBILE_BASE controller uses base-relative desired arm goals
while moving; ordinary arm mode uses world-relative achieved goals.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import numpy as np

from gpt_field.agent_speedup_v2_user_check_sim import CalibratedObservation
from gpt_field.libero import SimProcess, SimulatorError
from gpt_field.observation import PolicyObservation, image_to_data_url, quaternion_to_axisangle
from gpt_field.robocasa_user_check_sim import (
    RoboCasaWorker, RoboCasaSimProcess, EXTERNAL_CAMERAS, validate_arm_action,
    expand_arm_action, validate_arm_controller, _jsonable, _digest,
)

ROOT = Path(__file__).resolve().parents[1]
SIM_VERSION = "robocasa365-mobile-uc71-sim-3"
MOBILE_SCHEMA = "robocasa365-measured-mobile-state-1"
MOBILE_COMMAND_FRAME = (
    "Native LegacyMobileBaseJointVelocityController current-base axes: forward, side, yaw; "
    "controller transforms into initial-base joint coordinates. Inputs are bounded "
    "normalized commands. After the native frame transform, verified actuator scales are "
    "[1m/s,1m/s,1.5rad/s], not world displacement or measured speed. "
    "Torso is a normalized joint-position delta with 0.05m scale; positive raises torso."
)
MOBILE_LIMIT = 0.2
EXPECTED_GRIPPER_SHA256 = "066fd7ae45afb11e1844869360e587646e9768210ba5389aa3cd35292cde29c1"


@dataclass(frozen=True)
class RoboCasa365Config:
    task: str = "RestockPantry"
    split: str = "target"
    seed: int = 7
    image_size: int = 512
    external_camera: str = "robot0_agentview_left"
    python_executable: str = str(ROOT / ".venv/bin/python")
    robocasa_root: str = str(ROOT / "vendor/robocasa")
    robosuite_root: str = str(ROOT / "vendor/robosuite")
    response_timeout_s: float = 600.0
    stop_on_success: bool = True
    mobile_enabled: bool = True

    def __post_init__(self):
        if not isinstance(self.task, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.task):
            raise ValueError("task must be an official registry identifier")
        if self.split not in ("target", "pretrain", "all"):
            raise ValueError("Use an official RoboCasa365 split")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.image_size != 512 or self.external_camera not in EXTERNAL_CAMERAS:
            raise ValueError("Use two native512 standard robot cameras")
        if (type(self.response_timeout_s) not in (int, float)
                or not math.isfinite(self.response_timeout_s) or self.response_timeout_s <= 0):
            raise ValueError("response_timeout_s must be positive")
        if type(self.stop_on_success) is not bool or type(self.mobile_enabled) is not bool:
            raise ValueError("stop_on_success and mobile_enabled must be booleans")


def _finite_vector(value, count, label):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != count:
        raise ValueError(f"{label} requires {count} finite values")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.number))
           or isinstance(x, (complex, np.complexfloating)) or not math.isfinite(float(x)) for x in value):
        raise ValueError(f"{label} requires real finite numbers")
    return [float(x) for x in value]


def validate_mobile_command(value):
    if not isinstance(value, dict) or set(value) != {"base", "torso"}:
        raise ValueError("mobile command requires exactly base and torso")
    base = _finite_vector(value["base"], 3, "base")
    torso = _finite_vector([value["torso"]], 1, "torso")[0]
    if max(abs(x) for x in [*base, torso]) > MOBILE_LIMIT:
        raise ValueError("mobile normalized command exceeds +/-0.2")
    return {"base": base, "torso": torso}


def expand_mobile_action(arm_action, command, indexes, action_dim, previous_gripper):
    arm = validate_arm_action(arm_action)
    mobile = validate_mobile_command(command)
    if any(x != 0 for x in arm[:6]) or arm[6] != previous_gripper:
        raise ValueError("mobile_step requires zero arm delta and unchanged gripper command")
    # Shared validator verifies every native split, dimension and occupied slot.
    full = expand_arm_action(arm, indexes, action_dim, [0.0])
    full[slice(*indexes["base"])] = mobile["base"]
    full[slice(*indexes["torso"])] = mobile["torso"]
    full[-1] = 1.0
    return full


def validate_mobile_state(value):
    required = {"schema_version", "base_joint_position", "base_joint_velocity",
                "torso_joint_position", "torso_joint_velocity", "base_position_world_m",
                "base_rotation_world", "command_frame", "control_mode", "step"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("mobile_state must contain only measured robot fields")
    if value["schema_version"] != MOBILE_SCHEMA or value["command_frame"] != MOBILE_COMMAND_FRAME:
        raise ValueError("Unsupported mobile state schema/frame")
    if value["control_mode"] not in ("arm", "mobile") or type(value["step"]) is not int or value["step"] < 0:
        raise ValueError("Invalid mobile mode/timestamp")
    clean = dict(value)
    for name, count in (("base_joint_position", 3), ("base_joint_velocity", 3),
                        ("torso_joint_position", 1), ("torso_joint_velocity", 1),
                        ("base_position_world_m", 3)):
        clean[name] = _finite_vector(value[name], count, name)
    rotation = value["base_rotation_world"]
    if not isinstance(rotation, (list, tuple, np.ndarray)) or len(rotation) != 3:
        raise ValueError("base rotation requires a 3x3 matrix")
    matrix = np.array([_finite_vector(row, 3, "base rotation row") for row in rotation])
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(matrix), 1, atol=1e-6):
        raise ValueError("base rotation must be proper orthonormal")
    clean["base_rotation_world"] = matrix.tolist()
    return clean


@dataclass(frozen=True)
class MobileObservation(CalibratedObservation):
    mobile_state: dict = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "mobile_state", validate_mobile_state(self.mobile_state))


def validate_native_controllers(robot, *, initial=False):
    arm = robot.part_controllers["right"]
    expected_frame = "base" if initial else "world"
    if arm.input_ref_frame != expected_frame:
        raise ValueError(f"Expected native arm frame {expected_frame}")
    # Use unchanged strict numerical OSC check, restoring the native frame even on error.
    try:
        arm.input_ref_frame = "world"
        actual_arm = validate_arm_controller(arm)
    finally:
        arm.input_ref_frame = expected_frame
    actual_arm["input_ref_frame"] = expected_frame
    if type(robot.composite_controller).__name__ != "HybridMobileBase":
        raise ValueError("Expected native HybridMobileBase")
    if arm.interpolator_pos is not None or arm.interpolator_ori is not None:
        raise ValueError("Mode transitions require native no-interpolation OSC")
    records = {"arm": actual_arm}
    for part, class_name, dim, scale in (("base", "LegacyMobileBaseJointVelocityController", 3, 1.),
                                       ("torso", "JointPositionController", 1, .05)):
        ctrl = robot.part_controllers[part]
        if (type(ctrl).__name__ != class_name or ctrl.control_dim != dim
                or ctrl.impedance_mode != "fixed"
                or not np.array_equal(ctrl.input_min, [-1] * dim)
                or not np.array_equal(ctrl.input_max, [1] * dim)
                or not np.array_equal(ctrl.output_min, [-scale] * dim)
                or not np.array_equal(ctrl.output_max, [scale] * dim)):
            raise ValueError(f"Unexpected native {part} controller scale/type")
        if part == "base" and (
                not np.array_equal(ctrl.actuator_min, [-1., -1., -1.5])
                or not np.array_equal(ctrl.actuator_max, [1., 1., 1.5])):
            raise ValueError("Unexpected native base actuator velocity ranges")
        if part == "torso" and initial and ctrl.input_type != "delta":
            raise ValueError("Native torso must initialize in delta mode")
        records[part] = {"class": class_name, "control_dim": dim,
                         "input_min": [-1] * dim, "input_max": [1] * dim,
                         "output_min": [-scale] * dim, "output_max": [scale] * dim}
        if part == "base":
            records[part].update(actuator_min=np.asarray(ctrl.actuator_min).tolist(),
                                 actuator_max=np.asarray(ctrl.actuator_max).tolist(),
                                 normalized_to_actuator_velocity_scale=[1., 1., 1.5],
                                 velocity_units=["m/s", "m/s", "rad/s"],
                                 native_scaling="frame transform then actuator midrange plus half-range times command")
    return records


def switch_arm_reference(arm, frame):
    """Rebase the desired goal from measured robot state; never move physics."""
    if frame not in ("world", "base"):
        raise ValueError("Unsupported arm reference frame")
    arm.update(force=True)
    arm.input_ref_frame = frame
    arm.goal_pos = None
    arm.goal_ori = None
    arm.set_goal_update_mode("desired" if frame == "base" else "achieved")
    # Native compute_goal_* initialises missing goals in the chosen frame.
    arm.set_goal(np.zeros(6))


class RoboCasa365Worker(RoboCasaWorker):
    def __init__(self, config):
        self.config, self.env, self.status, self._initial = config, None, {}, None
        self._mode = None
        self._last_gripper = -1.0
        self._metadata = {"benchmark": "RoboCasa365", "task": config.task,
                          "split": config.split, "seed": config.seed, "robot": "PandaOmron",
                          "sim_version": SIM_VERSION, "image_size": 512,
                          "external_camera": config.external_camera, "wrist_camera": "robot0_eye_in_hand",
                          "control_freq_hz": 20, "translation_scale_m": .05, "rotation_scale_rad": .5,
                          "success_qualification": "raw_robocasa_check_success",
                          "episode_action_cap": None, "episode_time_cap": None,
                          "contact_policy": "record_only", "contact_sampling": "control_endpoint",
                          "mobile_enabled": config.mobile_enabled,
                          "mobile_command_frame": MOBILE_COMMAND_FRAME,
                          "mobile_command_cap": MOBILE_LIMIT,
                          "native_macros": False,
                          "observation_sync": "fresh_endpoint_rgb_robot_sensors_camera_calibration"}

    def reset(self):
        self.close()
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        import robosuite
        import robocasa
        import mujoco
        from robosuite import macros
        from robocasa.utils.env_utils import create_env
        from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS
        from gpt_field.action import TRANSLATION_SCALE_M, USER_CHECK_TRANSIT_ACTION_M
        if TRANSLATION_SCALE_M != .05 or USER_CHECK_TRANSIT_ACTION_M != .05:
            raise RuntimeError("RoboCasa365 profile requires explicit transit cap0.05/OSC scale0.05")
        for package, expected in ((robosuite, self.config.robosuite_root), (robocasa, self.config.robocasa_root)):
            if not Path(package.__file__).resolve().is_relative_to(Path(expected).resolve()):
                raise RuntimeError(f"Unexpected {package.__name__} import path")
        gripper_path = Path(self.config.robosuite_root) / "robosuite/models/assets/grippers/panda_gripper.xml"
        if hashlib.sha256(gripper_path.read_bytes()).hexdigest() != EXPECTED_GRIPPER_SHA256:
            raise RuntimeError("Panda gripper asset differs from validated geometry schema")
        registry = {**ATOMIC_TASK_DATASETS, **COMPOSITE_TASK_DATASETS}
        if self.config.task not in registry:
            raise ValueError("Task is absent from official RoboCasa365 registry")
        self.official_horizon = int(registry[self.config.task]["horizon"])
        macros.IMAGE_CONVENTION = "opencv"
        self.env = create_env(
            env_name=self.config.task, robots="PandaOmron", split=self.config.split,
            seed=self.config.seed, camera_names=[self.config.external_camera, "robot0_eye_in_hand"],
            camera_widths=512, camera_heights=512, camera_depths=False,
            render_onscreen=False, randomize_cameras=False, generative_textures=None,
            translucent_robot=False, control_freq=20, horizon=self.official_horizon,
            initialization_noise=None, reward_shaping=False)
        self.env.reset()
        if (self.env.control_freq != 20 or not self.env.ignore_done
                or self.env.horizon != self.official_horizon):
            raise RuntimeError("Native control rate, ignore_done or reference horizon differs from the configured pilot")
        robot = self.env.robots[0]
        native = validate_native_controllers(robot, initial=True)
        self.arm_controller = robot.part_controllers["right"]
        self.base_controller = robot.part_controllers["base"]
        self.torso_controller = robot.part_controllers["torso"]
        self.indexes = {key: tuple(int(x) for x in value)
                        for key, value in robot.composite_controller._action_split_indexes.items()}
        expand_arm_action([0] * 7, self.indexes, self.env.action_dim, [0.0])
        base_ids = [self.env.sim.model.joint_name2id(name) for name in robot.robot_model.base_joints]
        if (np.asarray(self.env.sim.model.jnt_type)[base_ids].tolist() != [2, 2, 3]
                or not np.array_equal(np.asarray(self.env.sim.model.jnt_axis)[base_ids], np.eye(3))):
            raise ValueError("Native base must use forward/side/yaw joint axes and order")
        torso_ids = self.torso_controller.joint_indexes["joints"]
        if (len(torso_ids) != 1 or int(self.env.sim.model.jnt_type[torso_ids[0]]) != 2
                or not np.array_equal(self.env.sim.model.jnt_axis[torso_ids[0]], [0, 0, 1])):
            raise ValueError("Native torso must use one positive-up slide joint")
        self._mode = None
        self._last_gripper = -1.0
        self._enter_mode("arm")
        validate_native_controllers(robot)
        self.base_position = np.array(self.env.sim.data.qpos[self.base_controller.qpos_index], copy=True)
        self.prompt = self.env.get_ep_meta()["lang"]
        self.status = {"steps": 0, "terminated": False, "last_native_action": None}
        self._metadata.update(
            official_horizon=self.official_horizon, horizon_policy="record_only_ignore_done_true",
            native_controller_contract=native, actual_arm_frame="world",
            action_split_indexes=self.indexes, simulator_action_dimension=self.env.action_dim,
            base_joint_names=list(robot.robot_model.base_joints),
            robosuite_source=robosuite.__file__, robocasa_source=robocasa.__file__,
            robosuite_version=getattr(robosuite, "__version__", "unknown"),
            robocasa_version=getattr(robocasa, "__version__", "unknown"), mujoco_version=mujoco.__version__,
            controller_frame_switch="world/achieved arm; base/desired mobile; rebase to measured pose at transition",
            object_observations="native create_env default internally; discarded at boundary")
        response = self._response()
        self._initial = self.snapshot(include_xml=True)
        return response

    def _enter_mode(self, mode):
        if mode == self._mode:
            return
        robot = self.env.robots[0]
        pos, ori = robot.composite_controller.get_controller_base_pose("right")
        self.arm_controller.update_origin(pos, ori)
        switch_arm_reference(self.arm_controller, "world" if mode == "arm" else "base")
        self.torso_controller.input_type = "absolute" if mode == "arm" else "delta"
        if mode == "arm":
            self.torso_position = np.array(self.env.sim.data.qpos[self.torso_controller.qpos_index], copy=True)
            self.torso_controller.set_goal(self.torso_position.copy())
        self._mode = mode

    def _mobile_state(self):
        data = self.env.sim.data
        pos, ori = self.base_controller.get_base_pose()
        return validate_mobile_state({
            "schema_version": MOBILE_SCHEMA,
            "base_joint_position": np.asarray(data.qpos[self.base_controller.qpos_index]).tolist(),
            "base_joint_velocity": np.asarray(data.qvel[self.base_controller.qvel_index]).tolist(),
            "torso_joint_position": np.asarray(data.qpos[self.torso_controller.qpos_index]).tolist(),
            "torso_joint_velocity": np.asarray(data.qvel[self.torso_controller.qvel_index]).tolist(),
            "base_position_world_m": np.asarray(pos).tolist(), "base_rotation_world": np.asarray(ori).tolist(),
            "command_frame": MOBILE_COMMAND_FRAME, "control_mode": self._mode, "step": self.status["steps"]})

    def _response(self):
        from gpt_field.robocasa_user_check_geometry import extract_robocasa_geometry
        obs = self._fresh_observation()
        state = tuple(float(x) for x in (*obs["robot0_eef_pos"],
                      *quaternion_to_axisangle(obs["robot0_eef_quat"]), *obs["robot0_gripper_qpos"]))
        observation = PolicyObservation(
            image=image_to_data_url(obs[f"{self.config.external_camera}_image"][:, ::-1], 512, rotate=False),
            wrist_image=image_to_data_url(obs["robot0_eye_in_hand_image"][:, ::-1], 512, rotate=False),
            state=state, prompt=self.prompt,
            joint_position=tuple(float(x) for x in obs["robot0_joint_pos"]),
            joint_velocity=tuple(float(x) for x in obs["robot0_joint_vel"]))
        geometry = extract_robocasa_geometry(self.env, image_size=512,
                    observation_state=state, external_camera=self.config.external_camera)
        raw = bool(self.env._check_success())
        self.status.update(success=raw, robocasa_success=raw,
                           terminated=bool(raw and self.config.stop_on_success),
                           contacts=self._contacts(), official_horizon=self.official_horizon,
                           horizon_hit=self.status["steps"] >= self.official_horizon,
                           native_timestep=int(self.env.timestep))
        return {"observation": observation.to_dict(), "geometry": geometry,
                "mobile_state": self._mobile_state(), "status": dict(self.status)}

    def _advance(self, full):
        # Native base controller transforms its slice in place. Preserve sent action first.
        self.status["last_native_action"] = np.asarray(full).tolist()
        self.env.step(np.array(full, copy=True))
        self.status["steps"] += 1
        return self._response()

    def step(self, action):
        if self.env is None or self.status.get("terminated"):
            raise RuntimeError("Reset is required before stepping")
        arm = validate_arm_action(action)
        self._enter_mode("arm")
        full = expand_arm_action(arm, self.indexes, self.env.action_dim, self.torso_position)
        self._last_gripper = arm[6]
        return self._advance(full)

    def mobile_step(self, action, command):
        if self.env is None or self.status.get("terminated"):
            raise RuntimeError("Reset is required before stepping")
        if not self.config.mobile_enabled:
            raise ValueError("Mobile capability disabled")
        full = expand_mobile_action(action, command, self.indexes, self.env.action_dim, self._last_gripper)
        self._enter_mode("mobile")
        return self._advance(full)

    def snapshot(self, include_xml=False):
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
                  "episode_metadata": self.env.get_ep_meta(), "status": dict(self.status),
                  "mobile_state": self._mobile_state(), "control_mode": self._mode,
                  "base_reset_qpos": self.base_position.tolist(),
                  "torso_arm_hold_qpos": self.torso_position.tolist()}
        if include_xml:
            result["model_xml"] = xml
        return _jsonable(result)


class RoboCasa365SimProcess(RoboCasaSimProcess):
    def __init__(self, config=None, *, command=None):
        config = config or RoboCasa365Config()
        SimProcess.__init__(self, config, command=command or [config.python_executable, "-u", "-m",
                            "rc365.sim", "--worker", "--config-json", json.dumps(asdict(config))])
        self._in_flight = False

    def start(self):
        if self._process is not None:
            return self
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT.parent / "src"),
                            self.config.robosuite_root, self.config.robocasa_root])
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        self._process = subprocess.Popen(self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        env=env, bufsize=0, start_new_session=True)
        try:
            self.metadata = self._request({"op": "metadata"})
        except BaseException:
            self.close()
            raise
        return self

    def _consume_observation(self, result):
        native = PolicyObservation.from_dict(result["observation"])
        self.status = result["status"]
        return MobileObservation(image=native.image, wrist_image=native.wrist_image,
                state=native.state, prompt=native.prompt, joint_position=native.joint_position,
                joint_velocity=native.joint_velocity, geometry=result["geometry"],
                mobile_state=result["mobile_state"])

    def mobile_step(self, action, command):
        if not self.status or self.status.get("terminated"):
            raise SimulatorError("Reset is required before stepping an inactive episode")
        result = self._request({"op": "mobile_step", "action": validate_arm_action(action),
                                "command": validate_mobile_command(command)})
        return self._consume_observation(result), bool(self.status["robocasa_success"])

    def list_tasks(self):
        return [{"task": self.config.task, "benchmark": "RoboCasa365", "split": self.config.split}]


def worker_main(config):
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    worker = RoboCasa365Worker(config)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request["op"]
                if op == "metadata": result = worker.metadata
                elif op == "reset": result = worker.reset()
                elif op == "step": result = worker.step(request["action"])
                elif op == "mobile_step": result = worker.mobile_step(request["action"], request["command"])
                elif op == "snapshot": result = worker.snapshot(include_xml=request.get("include_xml", False))
                elif op == "initial_snapshot":
                    if worker._initial is None: raise ValueError("Reset is required before initial snapshot")
                    result = worker._initial
                else: raise ValueError(f"Unknown operation: {op}")
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
    worker_main(RoboCasa365Config(**json.loads(args.config_json)))


if __name__ == "__main__":
    main()
