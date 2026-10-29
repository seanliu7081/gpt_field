"""Exact robot/camera reset binding plus a declared bounded RGB comparison.

This is an offline/fresh-reset check, not calibration or full-test approval.
Raw new image hashes and comparison metrics remain in the audit result. Only
the invariant reference, exact robot/camera signatures and a successful image
comparison enter the normalized start signature. No observed image is replaced.
The caller must capture live data without advancing physics and bind this code,
the reference and its files in the approval manifest before model execution.
"""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
from PIL import Image

from .approval import canonical_sha256, file_sha256

CAMERAS = {"head": "cam_head", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
IMAGE_POLICY = {"version": "native-rgb-reset-comparison-1", "shape": [480, 640, 3],
                "mean_absolute_channel_difference_max": 1.0,
                "percentile_99_channel_difference_max": 4.0,
                "maximum_channel_difference_max": 32,
                "resizing_alignment_cropping_or_color_adjustment": False}
ROBOT_FIELDS = ("joint_names", "body_names", "native_arm_joint_indices", "native_gripper_joint_indices",
                "is_fixed_base", "joint_positions", "joint_velocities", "root_link_pose_w",
                "body_link_pose_w", "body_com_pos_b", "joint_position_limits_lower_upper",
                "joint_velocity_limits", "root_physx_view_jacobians")


class ResetBindingError(ValueError):
    pass


def _finite(value):
    # Canonical encoding rejects NaN/Inf anywhere, including nested arrays.
    canonical_sha256(value)
    return deepcopy(value)


def robot_signature(inventory):
    if inventory.get("schema") != "robodojo-native-robot-inventory-1" or inventory.get("num_envs") != 1:
        raise ResetBindingError("native_robot_inventory_required")
    robots = {}
    for robot in inventory["robots"]:
        name = robot["arm_name"]
        if name not in ("left_arm", "right_arm") or name in robots:
            raise ResetBindingError("exact_dual_arm_inventory_required")
        row = {}
        for key in ROBOT_FIELDS:
            value = robot[key]
            # Tensor metadata can contain capture details; numerical values are exact.
            row[key] = value["values"] if isinstance(value, dict) and "values" in value else value
        if (row["joint_names"] != [f"joint{i}" for i in range(1, 9)]
                or row["native_arm_joint_indices"] != list(range(6))
                or row["native_gripper_joint_indices"] != [6, 7]
                or row["is_fixed_base"] is not True):
            raise ResetBindingError("unexpected_native_robot_structure")
        for key in ("joint_positions", "joint_velocities", "joint_velocity_limits"):
            if np.asarray(row[key], dtype=float).shape != (1, 8):
                raise ResetBindingError("unexpected_native_joint_array")
        robots[name] = _finite(row)
    if set(robots) != {"left_arm", "right_arm"}:
        raise ResetBindingError("exact_dual_arm_inventory_required")
    return {"schema": "robodojo-exact-robot-reset-1", "robots": robots}


def camera_signature(inventory):
    if (inventory.get("schema") != "robodojo-native-camera-inventory-1"
            or set(inventory.get("cameras", {})) != set(CAMERAS)):
        raise ResetBindingError("exact_native_camera_inventory_required")
    result = {}
    for name, native in CAMERAS.items():
        camera = inventory["cameras"][name]
        row = {key: camera[key] for key in ("native_name", "prim_path", "mount_link", "width", "height",
                                          "intrinsics", "world_from_optical", "optical_axes")}
        if (row["native_name"] != native or row["width"] != 640 or row["height"] != 480
                or row["optical_axes"] != "+X right, +Y down, +Z forward"
                or np.asarray(row["intrinsics"]).shape != (3, 3)
                or np.asarray(row["world_from_optical"]).shape != (4, 4)):
            raise ResetBindingError("unexpected_native_sensor_structure")
        result[name] = _finite(row)
    return {"schema": "robodojo-exact-initial-camera-1", "cameras": result}


def _image(path):
    with Image.open(path) as im:
        if im.format != "PNG" or im.mode != "RGB" or im.size != (640, 480):
            raise ResetBindingError("unaltered_native_rgb_png_required")
        value = np.asarray(im).copy()
    if value.dtype != np.uint8 or list(value.shape) != IMAGE_POLICY["shape"]:
        raise ResetBindingError("native_rgb_shape_or_dtype_mismatch")
    return value


def _entry(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def build_reference(robot_path, camera_path, images):
    if set(images) != set(CAMERAS):
        raise ResetBindingError("exact_three_images_required")
    robot, camera = _entry(robot_path), _entry(camera_path)
    image_files = {}
    for name, path in images.items():
        _image(path)
        image_files[name] = _entry(path)
    return {"schema": "robodojo-reset-reference-1", "image_policy": deepcopy(IMAGE_POLICY),
            "robot_file": robot, "camera_file": camera, "image_files": image_files,
            "robot_signature": robot_signature(json.loads(Path(robot["path"]).read_text())),
            "camera_signature": camera_signature(json.loads(Path(camera["path"]).read_text())),
            "full_test_authorized": False, "calibration_certified": False}


def verify_fresh_reset(reference, robot_inventory, camera_inventory, images):
    if (reference.get("schema") != "robodojo-reset-reference-1"
            or reference.get("image_policy") != IMAGE_POLICY
            or reference.get("full_test_authorized") is not False
            or reference.get("calibration_certified") is not False
            or set(reference.get("image_files", {})) != set(CAMERAS) or set(images) != set(CAMERAS)):
        raise ResetBindingError("frozen_reset_reference_required")
    for entry in [reference["robot_file"], reference["camera_file"], *reference["image_files"].values()]:
        if file_sha256(entry["path"]) != entry["sha256"]:
            raise ResetBindingError("reset_reference_file_changed")
    expected_robot = robot_signature(json.loads(Path(reference["robot_file"]["path"]).read_text()))
    expected_camera = camera_signature(json.loads(Path(reference["camera_file"]["path"]).read_text()))
    actual_robot, actual_camera = robot_signature(robot_inventory), camera_signature(camera_inventory)
    if (reference["robot_signature"] != expected_robot or reference["camera_signature"] != expected_camera
            or canonical_sha256(actual_robot) != canonical_sha256(expected_robot)
            or canonical_sha256(actual_camera) != canonical_sha256(expected_camera)):
        raise ResetBindingError("exact_robot_or_camera_reset_mismatch")
    rows = {}
    for name, path in images.items():
        original = _image(reference["image_files"][name]["path"])
        actual = _image(path)
        difference = np.abs(actual.astype(np.int16) - original.astype(np.int16))
        metrics = {"mean_absolute_channel_difference": float(difference.mean()),
                   "percentile_99_channel_difference": float(np.percentile(difference, 99)),
                   "maximum_channel_difference": int(difference.max())}
        if any(value > IMAGE_POLICY[key + "_max"] for key, value in metrics.items()):
            raise ResetBindingError(f"rgb_reset_comparison_failed:{name}:{json.dumps(metrics, sort_keys=True)}")
        rows[name] = {**_entry(path), **metrics, "passed": True}
    signature = {"schema": "robodojo-normalized-start-1",
                 "reference_sha256": canonical_sha256(reference),
                 "robot_sha256": canonical_sha256(actual_robot),
                 "camera_sha256": canonical_sha256(actual_camera),
                 "image_comparison_passed": True}
    return {"schema": "robodojo-reset-comparison-1", "passed": True,
            "start_state_signature": signature, "start_state_sha256": canonical_sha256(signature),
            "actual_image_comparisons": rows, "full_test_authorized": False,
            "calibration_certified": False}
