"""Candidate robot/camera measurements; no stepping, rendering, control or policy.

The caller must finish native measurement/render synchronization before capture.
The fixed-base Jacobian is world-frame and COM-referenced; ArmState requires the
link actor origin. This helper makes that reference-point correction explicitly.
Neither source geometry nor one successful snapshot certifies live calibration.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


ARMS = ("left", "right")
CAMERAS = {"cam_head": "head", "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"}


class NativeMeasurementError(RuntimeError):
    """Sanitized operator-only failure; no native exception text is exposed."""
    def __init__(self, stage: str, error_type: str):
        self.stage, self.error_type = stage, error_type
        super().__init__(f"native_measurement_failed:{stage}:{error_type}")


def _array(value, shape):
    if callable(getattr(value, "detach", None)):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
        raise ValueError("Invalid finite measurement shape/type")
    return value.astype(float, copy=True)


def _rotation(quat):
    quat = _array(quat, (4,))
    norm = np.linalg.norm(quat)
    if abs(norm - 1) > 1e-3:
        raise ValueError("Nonunit measured orientation")
    w, x, y, z = quat / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _skew(value):
    x, y, z = value
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _rigid(value):
    value = _array(value, (4, 4))
    r = value[:3, :3]
    if (not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=2e-5, rtol=0)
            or not np.isclose(np.linalg.det(r), 1., atol=2e-5, rtol=0)):
        raise ValueError("Invalid rigid transform")
    return value


def _physics_time(env):
    value = env.sim.sim.current_time
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Invalid physics time")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Invalid physics time")
    return value


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_link6_tool_candidate(native_root) -> dict:
    """Declare upstream link6 actor origin, NOT a physical fingertip/grasp TCP.

    The audited native get_real_endpose reads ee_link=link6 directly; its
    gripper_bias metadata is not applied. The caller must explicitly supply this
    candidate to NativeMeasurements; no implicit offset or calibrated default.
    Native source pin verification belongs to the enclosing provisioning gate.
    """
    root = Path(native_root).resolve(strict=True)
    sources = [{"path": str(root / name), "sha256": _sha(root / name)} for name in
               ("Assets/Robots/x5/robot_config.yml", "env/robot_manager/robot_manager.py")]
    source_sha = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return {"schema": "robodojo-native-tool-candidate-1", "calibration_status": "uncalibrated_candidate",
            "standard_ready": False, "definition": "native_link6_actor_origin", "source_artifacts": sources,
            "tools": {arm: {"link_name": "link6", "link6_from_tool": np.eye(4).tolist(),
                "source_sha256": source_sha, "finger_gap_formula": "sum_of_measured_finger_joint_positions",
                "finger_gap_semantics": "joint_separation_proxy"} for arm in ARMS}}


def _candidate(value):
    value = deepcopy(value)
    if (not isinstance(value, dict) or set(value) != {"schema", "calibration_status", "standard_ready",
            "definition", "source_artifacts", "tools"}
            or value["schema"] != "robodojo-native-tool-candidate-1"
            or value["calibration_status"] != "uncalibrated_candidate" or value["standard_ready"] is not False
            or value["definition"] != "native_link6_actor_origin" or set(value["tools"]) != set(ARMS)):
        raise ValueError("An explicit uncalibrated native-link6 candidate is required")
    sources = value["source_artifacts"]
    if (not isinstance(sources, list) or len(sources) != 2
            or any(set(row) != {"path", "sha256"} or _sha(row["path"]) != row["sha256"] for row in sources)
            or len({str(Path(row["path"]).resolve()) for row in sources}) != 2):
        raise ValueError("Tool candidate source provenance changed")
    source_sha = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    for tool in value["tools"].values():
        if (set(tool) != {"link_name", "link6_from_tool", "source_sha256", "finger_gap_formula", "finger_gap_semantics"}
                or tool["link_name"] != "link6" or tool["source_sha256"] != source_sha
                or tool["finger_gap_formula"] != "sum_of_measured_finger_joint_positions"
                or tool["finger_gap_semantics"] != "joint_separation_proxy"
                or not np.array_equal(_rigid(tool["link6_from_tool"]), np.eye(4))):
            raise ValueError("This candidate supports only the declared native link6 origin")
    return value


@dataclass(frozen=True)
class NativeRobotSample:
    arms: dict
    finger_position_m: dict
    provenance: dict  # operator-only; never merge into model observations


class NativeMeasurements:
    """Read both arms/finger pairs in one sample; never infer a calibration pass."""
    calibration_status = "uncalibrated_candidate"
    ready_for_full_test = False

    def __init__(self, env, *, tool_candidate):
        self._env = env
        try:
            self.tool_candidate = _candidate(tool_candidate)
        except Exception as error:
            raise NativeMeasurementError("tool_candidate", type(error).__name__) from None

    def require_ready(self):
        raise NativeMeasurementError("calibration_not_validated", "ReadinessError")

    def arm_states(self):
        """Convenience view; use sample() to share a capture with measured fingers."""
        return self.sample().arms

    def sample(self) -> NativeRobotSample:
        from .controller import ArmState, measured_tcp_pose
        stage = "manager"
        try:
            env, manager = self._env, self._env.robot_manager
            before = _physics_time(env)
            robots, articulations = list(manager.robot_list), list(manager.robot_key)
            if (env.num_envs != 1 or manager.num_envs != 1 or len(robots) != 2
                    or len(articulations) != 2 or articulations[0] is articulations[1]):
                raise ValueError("Exactly one environment and two distinct robots required")
            arms, fingers, evidence = {}, {}, {}
            for robot, articulation in zip(robots, articulations):
                stage = "robot_identity"
                arm = {"left_arm": "left", "right_arm": "right"}[robot.arm_name]
                joints, bodies = list(articulation.joint_names), list(articulation.body_names)
                n, b = len(joints), len(bodies)
                if (arm in arms or not articulation.is_fixed_base or robot.ee_link_name != "link6"
                        or robot.base_link != "base_link" or bodies[0] != "base_link"
                        or n != 8 or len(set(joints)) != n or len(set(bodies)) != b
                        or articulation.num_joints != n or articulation.num_bodies != b):
                    raise ValueError("Unsupported or ambiguous fixed-base ARX identity")
                arm_ids, finger_ids = list(robot.arm_joint_indices), list(robot.gripper_joint_indices)
                ids = arm_ids + finger_ids
                if (len(arm_ids) != 6 or len(finger_ids) != 2 or len(set(ids)) != n
                        or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                               or not 0 <= i < n for i in ids)
                        or [joints[i] for i in arm_ids] != list(robot.arm_joints_name)
                        or set(robot.arm_joints_name) != {f"joint{i}" for i in range(1, 7)}
                        or [joints[i] for i in finger_ids] != list(robot.gripper_joints_name)
                        or set(robot.gripper_joints_name) != {"joint7", "joint8"}):
                    raise ValueError("Live joint indices/names do not identify six arm and two finger joints")
                body_id = bodies.index("link6")
                if body_id < 1:
                    raise ValueError("Fixed root cannot be the end link")
                stage = "robot_measurements"
                data = articulation.data
                q, dq = _array(data.joint_pos, (1, n))[0], _array(data.joint_vel, (1, n))[0]
                limits = _array(data.joint_pos_limits, (1, n, 2))[0]
                velocity = _array(data.joint_vel_limits, (1, n))[0]
                pose = _array(data.body_link_pose_w, (1, b, 7))[0, body_id]
                com_local = _array(data.body_com_pos_b, (1, b, 3))[0, body_id]
                raw = _array(articulation.root_physx_view.get_jacobians(), (1, b-1, 6, n))
                # Separate indexing avoids NumPy advanced-index axis permutation.
                selected = raw[0, body_id-1][:, arm_ids]
                jacobian = selected.copy()
                com_world_offset = _rotation(pose[3:]) @ com_local
                jacobian[:3] += _skew(com_world_offset) @ jacobian[3:]
                state = ArmState(q[arm_ids], dq[arm_ids], pose[:3], pose[3:], jacobian,
                    limits[arm_ids, 0], limits[arm_ids, 1], velocity[arm_ids],
                    np.array(self.tool_candidate["tools"][arm]["link6_from_tool"])[:3, 3])
                measured_tcp_pose(state)  # validate without selecting or sending any target
                arms[arm], fingers[arm] = state, tuple(float(q[i]) for i in finger_ids)
                evidence[arm] = {"arm_joint_indices": [int(i) for i in arm_ids],
                    "arm_joint_names": [joints[i] for i in arm_ids], "finger_joint_indices": [int(i) for i in finger_ids],
                    "finger_joint_names": [joints[i] for i in finger_ids], "link6_body_index": body_id,
                    "jacobian_body_row": body_id-1, "raw_jacobian_shape": list(raw.shape),
                    "raw_com_world_jacobian": selected.tolist(), "link6_com_position_local_m": com_local.tolist(),
                    "jacobian_reference_correction": "J_actor_linear=J_COM_linear+skew(R_world_link6*c_COM_local)@J_angular"}
            stage = "snapshot_clock"
            after = _physics_time(env)
            if set(arms) != set(ARMS) or before != after:
                raise ValueError("Robot identities or physics changed during measurement")
            return NativeRobotSample(arms, fingers, {"schema": "robodojo-native-robot-sample-1",
                "scope": "setup_evaluator_only", "calibration_status": self.calibration_status,
                "standard_ready": False, "model_input_eligible": False,
                "physics_time_s": before, "captured_monotonic_s": time.monotonic(), "robots": evidence})
        except Exception as error:
            raise NativeMeasurementError(stage, type(error).__name__) from None


def capture_camera_inventory(env) -> dict:
    """Capture three candidate optical calibrations after native render sync.

    Physics time equality is checked, but dispatcher freshness/pixel alignment
    still needs rendered landmark validation. No camera render/update is called.
    """
    stage = "camera_manager"
    try:
        before = _physics_time(env)
        manager = env.camera_manager
        names, cameras = list(manager.camera_names[0]), list(manager.cameras[0])
        if (env.num_envs != 1 or manager.num_envs != 1 or len(names) != 3
                or set(names) != set(CAMERAS) or len(cameras) != 3 or len({id(c) for c in cameras}) != 3):
            raise ValueError("Exactly three uniquely named native cameras required")
        rows, paths = {}, set()
        for name, camera in zip(names, cameras):
            stage = "camera_" + CAMERAS[name]
            width, height = camera.get_resolution()
            if (type(width) not in (int, np.int32, np.int64) or type(height) not in (int, np.int32, np.int64)
                    or (width, height) != (640, 480) or "pinhole" not in camera.get_lens_distortion_model()):
                raise ValueError("Native 640x480 pinhole camera required")
            path = camera.prim_path
            if not isinstance(path, str) or not path.startswith("/") or path in paths or any(ord(c) < 32 for c in path):
                raise ValueError("Invalid or duplicate actual camera prim path")
            paths.add(path)
            mount = manager.camera_config[name]["camera"].get("mount_link")
            expected_mount = {"cam_head": None, "cam_left_wrist": "robot0/camera", "cam_right_wrist": "robot1/camera"}[name]
            if mount != expected_mount:
                raise ValueError("Native head/wrist mount identity differs from reviewed configuration")
            k = _array(camera.get_intrinsics_matrix(device="cpu"), (3, 3))
            if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-8, rtol=0):
                raise ValueError("Invalid measured pinhole intrinsic matrix")
            view = _rigid(camera.get_view_matrix_ros(device="cpu"))
            world = _rigid(np.linalg.inv(view))
            rows[CAMERAS[name]] = {"native_name": name, "native_camera_index": names.index(name),
                "prim_path": path, "mount_link": mount, "width": int(width), "height": int(height),
                "intrinsics": k.tolist(), "world_from_optical": world.tolist(),
                "optical_from_world": view.tolist(), "optical_axes": "+X right, +Y down, +Z forward",
                "pose_source": "synchronized native Camera.get_view_matrix_ros inverse; candidate",
                "fixed_world_frame_candidate": name == "cam_head",
                "static_across_frames_validated": False, "frame_alignment_validated": False}
        stage = "camera_snapshot_clock"
        after = _physics_time(env)
        if before != after:
            raise ValueError("Physics advanced during camera metadata capture")
        return {"schema": "robodojo-native-camera-inventory-1", "scope": "setup_evaluator_only",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(), "captured_monotonic_s": time.monotonic(),
            "calibration_status": "uncalibrated_candidate", "standard_ready": False, "model_input_eligible": False,
            "rendered_landmark_validation_passed": False, "native_physics_time_s": before,
            "synchronization_source": "caller after native render_for_capture; no independent freshness proof",
            "source_interfaces": ["Camera.get_intrinsics_matrix", "Camera.get_view_matrix_ros",
                "CameraManager.camera_names/cameras/camera_config"], "cameras": rows}
    except Exception as error:
        raise NativeMeasurementError(stage, type(error).__name__) from None


def capture_paused_physics_state(env, native_sim_step_calls: int) -> dict:
    """Read raw PhysX DOFs, bypassing timestamp-cached ArticulationData positions."""
    stage = "paused_physics_state"
    try:
        before = _physics_time(env)
        manager = env.robot_manager
        robots, articulations = list(manager.robot_list), list(manager.robot_key)
        if (type(native_sim_step_calls) is not int or native_sim_step_calls < 0
                or env.num_envs != 1 or manager.num_envs != 1 or len(robots) != 2
                or len(articulations) != 2 or articulations[0] is articulations[1]):
            raise ValueError("Unsupported paused-state inventory")
        positions = {}
        for robot, articulation in zip(robots, articulations):
            arm = {"left_arm": "left", "right_arm": "right"}[robot.arm_name]
            if arm in positions or articulation.num_joints != 8:
                raise ValueError("Ambiguous native arm identity")
            positions[arm] = _array(articulation.root_physx_view.get_dof_positions(), (1, 8)).tolist()
        if set(positions) != set(ARMS) or before != _physics_time(env):
            raise ValueError("Physics advanced or an arm was absent during paused-state capture")
        return {"native_physics_time_s": before, "native_sim_step_calls": native_sim_step_calls,
                "articulation_dof_positions": positions}
    except Exception as error:
        raise NativeMeasurementError(stage, type(error).__name__) from None
