"""Read-only, setup/evaluator-only inventory of the initialized dual ARX robots.

Call capture_robot_inventory(env) after native reset/measurement synchronization.
It accesses only RobotManager.robot_list/robot_key and articulation measurement
getters. IsaacLab getters may populate their measurement caches; this helper does
not call step, update, reset, render, action, target-write, observation, reward,
contact, or scene-object APIs. No simulator or torch modules are imported here.

Raw PhysX Jacobians retain their complete shape and values: no body index, joint
column convention, reference frame, or TCP conversion is inferred. Declared bias
and transforms are copied as metadata, never applied to poses. Native calibration
and approval remain external. The caller must persist an extraction failure as a
separate optional-inventory result, without changing a successful RGB result into
a claimed inventory success or silently dropping the error.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math

import numpy as np


MAX_ARRAY_ELEMENTS = 65536
MAX_NAMES = 64
MAX_JSON_BYTES = 2 * 1024 * 1024


class RobotInventoryError(RuntimeError):
    """Sanitized extraction failure, with no raw native exception payload."""
    def __init__(self, stage: str, error_type: str):
        self.stage, self.error_type = stage, error_type
        super().__init__(f"robot_inventory_failed:{stage}:{error_type}")


def _name(value) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256 or any(ord(c) < 32 for c in value):
        raise ValueError("Invalid native name")
    return value


def _names(value) -> list[str]:
    if isinstance(value, str):
        raise ValueError("Expected a native name sequence")
    names = [_name(item) for item in value]
    if not 1 <= len(names) <= MAX_NAMES or len(set(names)) != len(names):
        raise ValueError("Invalid native name count or uniqueness")
    return names


def _array(value, *, shape=None) -> dict:
    """Detach/read back tensors, preserving dtype and raw shape in JSON."""
    device = str(getattr(value, "device", "cpu"))
    numel = getattr(value, "numel", None)
    if callable(numel) and not 0 < numel() <= MAX_ARRAY_ELEMENTS:
        raise ValueError("Tensor exceeds inventory size bound")
    if callable(getattr(value, "detach", None)):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if (array.dtype.kind not in "iuf" or not 0 < array.size <= MAX_ARRAY_ELEMENTS
            or array.ndim > 4 or not np.isfinite(array).all()
            or (shape is not None and array.shape != shape)):
        raise ValueError("Invalid finite bounded numeric measurement")
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "source_device": device, "values": array.tolist()}


def _indices(value, count: int) -> list[int]:
    indices = list(value)
    if (not indices or len(indices) > MAX_NAMES
            or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                   or not 0 <= i < count for i in indices)):
        raise ValueError("Invalid native joint indices")
    result = [int(i) for i in indices]
    if len(result) != len(set(result)):
        raise ValueError("Duplicate native joint indices")
    return result


def _finite_scalar(value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError("Invalid declared numeric metadata")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite declared numeric metadata")
    return number


def capture_robot_inventory(env) -> dict:
    """Return JSON-safe robot measurements or raise RobotInventoryError.

    Exactly one native environment and two distinct initialized articulations
    are required. Native arm names define identity; list order is recorded, not
    interpreted as left/right. Tensor arrays retain the environment dimension.
    Mixed arm/finger joint units are not relabeled as universally radians.
    """
    stage = "manager"
    try:
        manager = env.robot_manager
        if env.num_envs != 1 or manager.num_envs != 1:
            raise ValueError("Inventory requires one environment")
        robots, articulations = list(manager.robot_list), list(manager.robot_key)
        if len(robots) != 2 or len(articulations) != 2 or articulations[0] is articulations[1]:
            raise ValueError("Inventory requires exactly two distinct initialized robots")
        result = {"schema": "robodojo-native-robot-inventory-1", "scope": "setup_evaluator_only",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_status": "uncalibrated_candidate", "num_envs": 1,
            "tcp_calibration_inferred": False, "jacobian_index_or_frame_inferred": False,
            "model_input_eligible": False, "robots": [],
            "pose_contract": "IsaacLab root_link_pose_w/body_link_pose_w: world actor-link [x,y,z,qw,qx,qy,qz]",
            "jacobian_contract": "Raw root_physx_view.get_jacobians(); indexing and coordinate convention unvalidated",
            "declared_transform_contract": "Native robot metadata only; no transform is applied or certified as TCP",
            "source_interfaces": ["env/robot_manager/robot_manager.py:RobotManager",
                "env/robot_manager/robot_class/x5.py:X5",
                "isaaclab/assets/articulation/articulation.py:Articulation",
                "isaaclab/assets/articulation/articulation_data.py:ArticulationData"]}
        for index, (robot, articulation) in enumerate(zip(robots, articulations)):
            stage = f"robot_{index}_identity"
            joint_names, body_names = _names(articulation.joint_names), _names(articulation.body_names)
            count, bodies = len(joint_names), len(body_names)
            if int(articulation.num_joints) != count or int(articulation.num_bodies) != bodies:
                raise ValueError("Native articulation name/count mismatch")
            arm_indices = _indices(robot.arm_joint_indices, count)
            gripper_indices = _indices(robot.gripper_joint_indices, count)
            arm_names, gripper_names = _names(robot.arm_joints_name), _names(robot.gripper_joints_name)
            if ([joint_names[i] for i in arm_indices] != arm_names
                    or [joint_names[i] for i in gripper_indices] != gripper_names):
                raise ValueError("Native declared joint names/indices disagree")
            ee_link = _name(robot.ee_link_name)
            if ee_link not in body_names:
                raise ValueError("Declared native end link is absent from live articulation")
            record = {"native_list_index": index, "robot_name": _name(robot.robot_name),
                "robot_type": _name(robot.robot_type), "arm_name": _name(robot.arm_name),
                "gripper_name": _name(robot.gripper_name), "ee_type": _name(robot.ee_type),
                "ee_joint_name": _name(robot.ee_joint_name), "ee_link_name": ee_link,
                "base_link_name": _name(robot.base_link), "joint_names": joint_names,
                "body_names": body_names, "num_joints": count, "num_bodies": bodies,
                "is_fixed_base": bool(articulation.is_fixed_base),
                "native_arm_joint_indices": arm_indices, "native_gripper_joint_indices": gripper_indices,
                "native_arm_joint_names": arm_names, "native_gripper_joint_names": gripper_names}
            stage = f"robot_{index}_joint_measurements"
            data = articulation.data
            record["joint_positions"] = _array(data.joint_pos, shape=(1, count))
            record["joint_velocities"] = _array(data.joint_vel, shape=(1, count))
            record["joint_position_limits_lower_upper"] = _array(data.joint_pos_limits, shape=(1, count, 2))
            record["joint_velocity_limits"] = _array(data.joint_vel_limits, shape=(1, count))
            stage = f"robot_{index}_link_poses"
            record["root_link_pose_w"] = _array(data.root_link_pose_w, shape=(1, 7))
            record["body_link_pose_w"] = _array(data.body_link_pose_w, shape=(1, bodies, 7))
            stage = f"robot_{index}_raw_jacobians"
            record["root_physx_view_jacobians"] = _array(articulation.root_physx_view.get_jacobians())
            stage = f"robot_{index}_declared_metadata"
            mimic = list(robot.gripper_move["mimic"])
            if len(mimic) != 3:
                raise ValueError("Unexpected declared gripper mimic metadata")
            record["declared_metadata"] = {
                "gripper_bias": _finite_scalar(robot.gripper_bias),
                "gripper_scale": _array(robot.gripper_scale, shape=(2,)),
                "gripper_move": {"base": _name(robot.gripper_move["base"]),
                    "sign": _finite_scalar(robot.gripper_move["sign"]),
                    "mimic": [_name(mimic[0]), _finite_scalar(mimic[1]), _finite_scalar(mimic[2])]},
                "delta_matrix": _array(robot.delta_matrix, shape=(3, 3)),
                "global_trans_matrix": _array(robot.global_trans_matrix, shape=(3, 3)),
                "default_root_pos": _array(robot.default_root_pos, shape=(3,)),
                "default_root_rot": _array(robot.default_root_rot, shape=(4,))}
            result["robots"].append(record)
        stage = "serialization"
        if (len({r["arm_name"] for r in result["robots"]}) != 2
                or len({r["gripper_name"] for r in result["robots"]}) != 2):
            raise ValueError("Native robot names are not unique")
        if len(json.dumps(result, allow_nan=False).encode()) > MAX_JSON_BYTES:
            raise ValueError("Inventory JSON exceeds size bound")
        return result
    except Exception as error:
        raise RobotInventoryError(stage, type(error).__name__) from None
