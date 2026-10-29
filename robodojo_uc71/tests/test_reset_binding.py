from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from robodojo_uc71.reset_binding import (CAMERAS, IMAGE_POLICY, ResetBindingError,
    build_reference, verify_fresh_reset)


@pytest.fixture
def reset(tmp_path):
    robot = {"schema": "robodojo-native-robot-inventory-1", "num_envs": 1, "robots": []}
    for name in ("left_arm", "right_arm"):
        row = {"arm_name": name, "joint_names": [f"joint{i}" for i in range(1, 9)],
               "body_names": ["base_link", *[f"link{i}" for i in range(1, 9)]],
               "native_arm_joint_indices": list(range(6)), "native_gripper_joint_indices": [6, 7],
               "is_fixed_base": True}
        for key, shape in {"joint_positions": (1, 8), "joint_velocities": (1, 8),
            "joint_velocity_limits": (1, 8), "joint_position_limits_lower_upper": (1, 8, 2),
            "root_link_pose_w": (1, 7), "body_link_pose_w": (1, 9, 7),
            "body_com_pos_b": (1, 9, 3), "root_physx_view_jacobians": (1, 8, 6, 8)}.items():
            row[key] = {"values": np.zeros(shape).tolist()}
        robot["robots"].append(row)
    camera = {"schema": "robodojo-native-camera-inventory-1", "cameras": {name: {
        "native_name": native, "prim_path": f"/World/{native}", "mount_link": None,
        "width": 640, "height": 480, "intrinsics": np.eye(3).tolist(),
        "world_from_optical": np.eye(4).tolist(), "optical_axes": "+X right, +Y down, +Z forward"}
        for name, native in CAMERAS.items()}}
    rp, cp = tmp_path / "robot.json", tmp_path / "camera.json"
    rp.write_text(json.dumps(robot)); cp.write_text(json.dumps(camera))
    images = {}
    for name in CAMERAS:
        images[name] = tmp_path / f"{name}.png"
        Image.fromarray(np.full((480, 640, 3), 100, np.uint8)).save(images[name])
    reference = build_reference(rp, cp, images)
    return reference, robot, camera, images, tmp_path


def modified_images(reset, transform):
    images = dict(reset[3])
    data = np.full((480, 640, 3), 100, np.uint8)
    transform(data)
    images["head"] = reset[4] / "new_head.png"
    Image.fromarray(data).save(images["head"])
    return images


def test_bounded_pixel_noise_retains_normalized_signature_and_raw_new_hash(reset):
    ref, robot, camera, images, _ = reset
    original = verify_fresh_reset(ref, robot, camera, images)
    new_images = modified_images(reset, lambda data: data.__setitem__((0, 0, 0), 120))
    actual = verify_fresh_reset(ref, robot, camera, new_images)
    assert original["start_state_sha256"] == actual["start_state_sha256"]
    assert original["actual_image_comparisons"]["head"]["sha256"] != actual["actual_image_comparisons"]["head"]["sha256"]
    assert actual["actual_image_comparisons"]["head"]["maximum_channel_difference"] == 20
    assert actual["calibration_certified"] is actual["full_test_authorized"] is False


@pytest.mark.parametrize("mutation", [lambda data: data.fill(102),
    lambda data: data.__setitem__((0, 0, 0), 133),
    lambda data: data.__setitem__((slice(0, 20), slice(None), slice(None)), 105)])
def test_each_declared_pixel_metric_can_reject(reset, mutation):
    ref, robot, camera, _, _ = reset
    with pytest.raises(ResetBindingError, match="rgb_reset_comparison_failed"):
        verify_fresh_reset(ref, robot, camera, modified_images(reset, mutation))


@pytest.mark.parametrize("kind", ["joint", "camera", "jacobian"])
def test_no_numerical_tolerance_for_robot_or_camera_state(reset, kind):
    ref, robot, camera, images, _ = deepcopy(reset)
    if kind == "joint": robot["robots"][0]["joint_positions"]["values"][0][0] += 1e-12
    if kind == "jacobian": robot["robots"][0]["root_physx_view_jacobians"]["values"][0][0][0][0] += 1e-12
    if kind == "camera": camera["cameras"]["head"]["world_from_optical"][0][3] += 1e-12
    with pytest.raises(ResetBindingError, match="exact_robot_or_camera_reset_mismatch"):
        verify_fresh_reset(ref, robot, camera, images)


def test_reference_evidence_change_is_rejected(reset):
    ref, robot, camera, images, _ = reset
    images["head"].write_bytes(b"changed")
    with pytest.raises(ResetBindingError, match="reset_reference_file_changed"):
        verify_fresh_reset(ref, robot, camera, images)


def test_relaxed_policy_or_forged_signature_is_rejected(reset):
    ref, robot, camera, images, _ = deepcopy(reset)
    ref["image_policy"]["maximum_channel_difference_max"] = 255
    with pytest.raises(ResetBindingError, match="frozen_reset_reference_required"):
        verify_fresh_reset(ref, robot, camera, images)
    ref["image_policy"] = deepcopy(IMAGE_POLICY)
    ref["robot_signature"]["robots"]["left_arm"]["joint_positions"][0][0] = 1
    with pytest.raises(ResetBindingError, match="exact_robot_or_camera_reset_mismatch"):
        verify_fresh_reset(ref, robot, camera, images)


def test_wrong_image_geometry_is_not_resized(reset):
    ref, robot, camera, images, directory = reset
    images = dict(images)
    images["head"] = directory / "small.png"
    Image.fromarray(np.zeros((240, 320, 3), np.uint8)).save(images["head"])
    with pytest.raises(ResetBindingError, match="unaltered_native_rgb_png_required"):
        verify_fresh_reset(ref, robot, camera, images)
