"""Opt-in camera calibration and fixed Panda geometry for the Agentic adapter.

The extractor reads only two named cameras. It never reads objects, contacts,
rewards, task predicates, or simulator joint arrays and never advances physics.
The returned calibration describes upright, uncropped RGB (raw render flipped
vertically), with OpenGL camera axes and integer pixel centers. For policy use,
pass the native observation state so the wrist camera uses the same cached
measurement as its RGB, rather than a slightly later simulator endpoint.
"""
from __future__ import annotations

import copy
import math
from numbers import Real
from typing import Mapping

import numpy as np

from .robot_tool_geometry import ROBOT_ASSET_PROVENANCE, _rotation_matrix


GEOMETRY_VERSION = "astranomy-agentic-panda-camera-v2"
CAMERA_NAMES = {"external": "agentview", "wrist": "robot0_eye_in_hand"}
_WRIST_POSITION_HAND_M = np.array([0.05, 0.0, 0.0])
_HAND_TO_GRIP_M = np.array([0.0, 0.0, 0.097])
# robot.xml: fixed eye_in_hand quaternion wxyz=[0,.707108,.707108,0].
_WRIST_ROTATION_HAND_FROM_CAMERA = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
_IMAGE_CONSTANTS = {
    "pixel_axes": "u_right_v_down",
    "pixel_centers": "integer",
    "source_transform": "raw_mujoco_rgb_vertical_flip_no_crop",
}
_GRIPPER = {
    "model": "PandaGripper",
    "position_reference": "gripper0_grip_site",
    "orientation_reference": "robot0_right_hand",
    "hand_to_grip_site_m": [0.0, 0.0, 0.097],
    "approach_axis_hand": [0.0, 0.0, 1.0],
    "jaw_axis_hand": [0.0, 1.0, 0.0],
    "finger_joint_ranges_m": [[0.0, 0.04], [-0.04, 0.0]],
    "symmetric_pad_midpoint_from_grip_site_hand_m": [0.0, 0.0, -0.0036],
    "pad_full_size_m": [0.016, 0.008, 0.016],
    "nominal_inner_gap_from_finger_q_m": "q1 - q2 - 0.001",
    "nominal_open_inner_gap_m": 0.079,
    "pad_midpoint_lateral_offset_hand_m": "(q1 + q2) / 2",
    "asset_provenance": [dict(x) for x in ROBOT_ASSET_PROVENANCE
                         if x["path"].endswith("panda_gripper.xml")],
}
_CAMERA_KEYS = {"mujoco_name", "pose_source", "fovy_deg", "focal_length_px", "principal_point_px",
                "position_world_m", "rotation_world_from_camera"}


def _keys(value, expected, name):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{name} must contain exactly the allowed fields")


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _vector(value, size, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != size:
        raise ValueError(f"{name} must have {size} entries")
    return [_number(x, name) for x in value]


def _dimension(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8192:
        raise ValueError(f"{name} must be an integer in [1, 8192]")
    return value


def validate_geometry_metadata(value) -> dict:
    """Reject extra fields and return independent, JSON-safe camera/robot data."""
    _keys(value, {"schema_version", "frame", "image", "gripper", "cameras"}, "geometry")
    if value["schema_version"] != GEOMETRY_VERSION or value["frame"] != "simulation_world":
        raise ValueError("unsupported geometry schema or frame")
    img = value["image"]
    _keys(img, {"width", "height", *_IMAGE_CONSTANTS}, "geometry.image")
    width, height = _dimension(img["width"], "width"), _dimension(img["height"], "height")
    if width != height or any(img[k] != v for k, v in _IMAGE_CONSTANTS.items()):
        raise ValueError("geometry requires upright square uncropped images")
    # Static facts cannot become an unbounded conduit for arbitrary scene data.
    if value["gripper"] != _GRIPPER:
        raise ValueError("gripper geometry must match the fixed Panda asset contract")
    _keys(value["cameras"], CAMERA_NAMES, "geometry.cameras")
    cameras = {}
    for label, name in CAMERA_NAMES.items():
        cam = value["cameras"][label]
        _keys(cam, _CAMERA_KEYS, f"geometry.cameras.{label}")
        if cam["mujoco_name"] != name:
            raise ValueError("unexpected camera identity")
        permitted_sources = ({"fixed_world_camera"} if label == "external" else
                             {"observed_robot_pose_fixed_mount", "current_simulator_camera_state"})
        if cam["pose_source"] not in permitted_sources:
            raise ValueError("unexpected camera pose source")
        fovy = _number(cam["fovy_deg"], "fovy_deg")
        if not 0 < fovy < 180:
            raise ValueError("camera fovy must be between 0 and 180 degrees")
        focal = _vector(cam["focal_length_px"], 2, "focal_length_px")
        center = _vector(cam["principal_point_px"], 2, "principal_point_px")
        expected_focal = height / (2 * math.tan(math.radians(fovy) / 2))
        if not all(math.isclose(x, expected_focal, rel_tol=1e-10, abs_tol=1e-10) for x in focal):
            raise ValueError("camera focal length does not match image size and fovy")
        if center != [(width - 1) / 2, (height - 1) / 2]:
            raise ValueError("camera principal point does not match integer pixel centers")
        position = _vector(cam["position_world_m"], 3, "position_world_m")
        raw_rotation = cam["rotation_world_from_camera"]
        if not isinstance(raw_rotation, (list, tuple, np.ndarray)) or len(raw_rotation) != 3:
            raise ValueError("camera rotation must be a 3 by 3 matrix")
        rotation = np.asarray([_vector(row, 3, "rotation row") for row in raw_rotation])
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
                or not math.isclose(float(np.linalg.det(rotation)), 1, abs_tol=1e-7)):
            raise ValueError("camera rotation must be a proper orthonormal matrix")
        cameras[label] = {"mujoco_name": name, "pose_source": cam["pose_source"], "fovy_deg": fovy,
                          "focal_length_px": focal, "principal_point_px": center,
                          "position_world_m": position,
                          "rotation_world_from_camera": rotation.tolist()}
    return {"schema_version": GEOMETRY_VERSION, "frame": "simulation_world",
            "image": {"width": width, "height": height, **_IMAGE_CONSTANTS},
            "gripper": copy.deepcopy(_GRIPPER), "cameras": cameras}


def wrist_camera_pose_from_observation(observation_state) -> tuple[list, list]:
    """Derive the wrist camera pose from native state8 and its fixed robot mount.

    Native state8 is grip-site XYZ, right-hand world rotation vector, and two
    finger joints. Camera and robot sensors sample in the same physics update.
    This conversion preserves that timestamp without changing the observations.
    """
    state = np.asarray(_vector(observation_state, 8, "observation_state"))
    rotation = _rotation_matrix(state[3:6])
    position = state[:3] + rotation @ (_WRIST_POSITION_HAND_M - _HAND_TO_GRIP_M)
    return position.tolist(), (rotation @ _WRIST_ROTATION_HAND_FROM_CAMERA).tolist()


def extract_geometry_metadata(env, image_size: int = 512, observation_state=None) -> dict:
    """Read camera-only calibration at the current observation, without stepping.

    ``env`` is the worker's existing OffScreenRenderEnv exposing ``env.sim``.
    Its wrapper model provides camera_name2id. No second environment is created.
    For policy observations, supply native observation_state from the same RGB
    response. Omitting it exports endpoint wrist calibration for diagnostics only.
    """
    size = _dimension(image_size, "image_size")
    model, data = env.sim.model, env.sim.data
    observed_wrist = (None if observation_state is None else
                      wrist_camera_pose_from_observation(observation_state))
    cameras = {}
    for label, name in CAMERA_NAMES.items():
        cid = model.camera_name2id(name)
        if not isinstance(cid, (int, np.integer)) or cid < 0:
            raise ValueError(f"required camera is unavailable: {name}")
        fovy = _number(model.cam_fovy[cid], "fovy_deg")
        if not 0 < fovy < 180:
            raise ValueError("camera fovy must be between 0 and 180 degrees")
        focal = size / (2 * math.tan(math.radians(fovy) / 2))
        if label == "wrist" and observed_wrist is not None:
            position, rotation = observed_wrist
            pose_source = "observed_robot_pose_fixed_mount"
        else:
            position = np.asarray(data.cam_xpos[cid]).tolist()
            rotation = np.asarray(data.cam_xmat[cid]).reshape(3, 3).tolist()
            pose_source = "fixed_world_camera" if label == "external" else "current_simulator_camera_state"
        cameras[label] = {"mujoco_name": name, "pose_source": pose_source, "fovy_deg": fovy,
                          "focal_length_px": [focal, focal],
                          "principal_point_px": [(size - 1) / 2] * 2,
                          "position_world_m": position,
                          "rotation_world_from_camera": rotation}
    return validate_geometry_metadata({"schema_version": GEOMETRY_VERSION,
        "frame": "simulation_world", "image": {"width": size, "height": size, **_IMAGE_CONSTANTS},
        "gripper": copy.deepcopy(_GRIPPER), "cameras": cameras})


def project_world_point(point, camera: Mapping) -> list[float] | None:
    """Project a supplied point for geometry tests; no simulator state is read."""
    point = np.asarray(_vector(point, 3, "point"))
    rotation = np.asarray(camera["rotation_world_from_camera"], dtype=float)
    local = rotation.T @ (point - np.asarray(camera["position_world_m"]))
    depth = -float(local[2])
    if depth <= 0:
        return None
    fx, fy = camera["focal_length_px"]
    cx, cy = camera["principal_point_px"]
    return [float(cx + fx * local[0] / depth), float(cy - fy * local[1] / depth)]
