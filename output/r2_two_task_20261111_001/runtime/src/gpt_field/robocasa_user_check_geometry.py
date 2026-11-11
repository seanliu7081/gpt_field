"""RoboCasa camera and known Panda geometry for the current user-check adapter.

Only named robot sites/bodies, camera calibration and fixed robot asset facts
are read. No objects, depth, contacts, reward or task predicates cross this
boundary. The worker must force fresh RGB and proprioception at the current
physics endpoint before calling the extractor, without intervening physics.
The endpoint check rejects the stale cached-observation case explicitly.

The existing LIBERO schema and validator stay unchanged. RoboCasa has its own
schema because its robot-mounted external camera, gripper site identity and
asset provenance differ even where the Panda's physical dimensions agree.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import numpy as np

from . import astranomy_agentic_geometry as libero_geometry
from .robot_tool_geometry import _rotation_matrix


GEOMETRY_VERSION = "robocasa-user-check-panda-camera-v1"
EXTERNAL_CAMERAS = frozenset({"robot0_agentview_center", "robot0_agentview_left",
                              "robot0_agentview_right"})
DEFAULT_EXTERNAL_CAMERA = "robot0_agentview_left"
WRIST_CAMERA = "robot0_eye_in_hand"
HAND_BODY = "robot0_right_hand"
GRIP_SITE = "gripper0_right_grip_site"
WRIST_POSITION_HAND_M = np.array([0.05, 0.0, 0.0])
HAND_TO_GRIP_M = np.array([0.0, 0.0, 0.097])
WRIST_ROTATION_HAND_FROM_CAMERA = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
ROBOCASA_GRIPPER = copy.deepcopy(libero_geometry._GRIPPER)
ROBOCASA_GRIPPER.update(position_reference=GRIP_SITE, asset_provenance=[{
    "path": "models/assets/grippers/panda_gripper.xml",
    "sha256": "066fd7ae45afb11e1844869360e587646e9768210ba5389aa3cd35292cde29c1",
    "size_bytes": 4546,
}])
EXTERNAL_POSE_SOURCE = "synchronized_robot_mounted_camera"


def validate_geometry_metadata(value) -> dict:
    """Dispatch only the two explicit allowlisted camera/robot schemas.

    The old validator supplies the shared numerical camera checks after all
    RoboCasa-specific identities and fixed asset facts are independently checked.
    Its temporary canonical identities are never returned or sent to a model.
    """
    if not isinstance(value, Mapping):
        raise ValueError("geometry must be an object")
    if value.get("schema_version") == libero_geometry.GEOMETRY_VERSION:
        return libero_geometry.validate_geometry_metadata(value)
    if value.get("schema_version") != GEOMETRY_VERSION:
        raise ValueError("unsupported geometry schema")
    libero_geometry._keys(value, {"schema_version", "frame", "image", "gripper", "cameras"},
                          "geometry")
    if value["gripper"] != ROBOCASA_GRIPPER:
        raise ValueError("gripper geometry must match the RoboCasa Panda asset contract")
    libero_geometry._keys(value["cameras"], {"external", "wrist"}, "geometry.cameras")
    for label in ("external", "wrist"):
        libero_geometry._keys(value["cameras"][label], libero_geometry._CAMERA_KEYS,
                              f"geometry.cameras.{label}")
    external, wrist = value["cameras"]["external"], value["cameras"]["wrist"]
    if (not isinstance(external["mujoco_name"], str)
            or external["mujoco_name"] not in EXTERNAL_CAMERAS):
        raise ValueError("unexpected RoboCasa external camera identity")
    if external["pose_source"] != EXTERNAL_POSE_SOURCE:
        raise ValueError("RoboCasa external camera requires synchronized robot-mounted calibration")
    if (wrist["mujoco_name"] != WRIST_CAMERA
            or wrist["pose_source"] != "observed_robot_pose_fixed_mount"):
        raise ValueError("RoboCasa wrist camera requires its observed robot pose and checked fixed mount")
    canonical = copy.deepcopy(dict(value))
    canonical["schema_version"] = libero_geometry.GEOMETRY_VERSION
    canonical["gripper"] = copy.deepcopy(libero_geometry._GRIPPER)
    canonical["cameras"]["external"].update(mujoco_name="agentview", pose_source="fixed_world_camera")
    clean = libero_geometry.validate_geometry_metadata(canonical)
    clean["schema_version"] = GEOMETRY_VERSION
    clean["gripper"] = copy.deepcopy(ROBOCASA_GRIPPER)
    clean["cameras"]["external"].update(mujoco_name=external["mujoco_name"],
                                          pose_source=EXTERNAL_POSE_SOURCE)
    return clean


def _required_id(model, kind, name):
    index = getattr(model, f"{kind}_name2id")(name)
    if not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError(f"required RoboCasa {kind} is unavailable: {name}")
    return int(index)


def _check_robot_and_synchronization(model, data, state, wrist_id):
    """Verify the actual compiled mount and matching robot sensor timestamp."""
    hand_id = _required_id(model, "body", HAND_BODY)
    grip_id = _required_id(model, "site", GRIP_SITE)
    if int(model.cam_bodyid[wrist_id]) != hand_id or int(model.cam_mode[wrist_id]) != 0:
        raise ValueError("RoboCasa wrist camera must be fixed to the named Panda hand")
    if not np.allclose(model.cam_pos[wrist_id], WRIST_POSITION_HAND_M, atol=1e-9, rtol=0):
        raise ValueError("RoboCasa wrist camera position does not match the supported fixed mount")
    mount_quat = np.asarray(model.cam_quat[wrist_id], dtype=float)
    expected_quat = np.array([0., 1 / math.sqrt(2), 1 / math.sqrt(2), 0.])
    if (mount_quat.shape != (4,) or not np.isfinite(mount_quat).all()
            or not math.isclose(float(np.linalg.norm(mount_quat)), 1.0, abs_tol=1e-7)
            or min(np.linalg.norm(mount_quat - expected_quat),
                   np.linalg.norm(mount_quat + expected_quat)) > 1e-7):
        raise ValueError("RoboCasa wrist camera rotation does not match the supported fixed mount")
    hand_rotation = np.asarray(data.body_xmat[hand_id], dtype=float).reshape(3, 3)
    hand_position = np.asarray(data.body_xpos[hand_id], dtype=float)
    grip_position = np.asarray(data.site_xpos[grip_id], dtype=float)
    if not np.allclose(hand_rotation.T @ (grip_position - hand_position),
                       HAND_TO_GRIP_M, atol=1e-8, rtol=0):
        raise ValueError("RoboCasa grip site is not at the supported Panda hand offset")
    observed_rotation = _rotation_matrix(state[3:6])
    # robosuite's observation quaternion conversion can include float32 rounding.
    if (not np.allclose(state[:3], grip_position, atol=1e-6, rtol=0)
            or not np.allclose(observed_rotation, hand_rotation, atol=1e-6, rtol=0)):
        raise ValueError("RoboCasa camera calibration and robot observation are not synchronized")
    return observed_rotation


def extract_robocasa_geometry(env, image_size: int = 512, observation_state=None, *,
                              external_camera: str = DEFAULT_EXTERNAL_CAMERA) -> dict:
    """Extract calibration after fresh endpoint RGB/proprioception, never step.

    Images consumed by ``StepView`` must be upright, uncropped, square RGB.
    ``observation_state`` is required: world grip-site XYZ, right-hand rotation
    vector, and two finger joint positions. The raw observation data URLs use the
    existing horizontal-mirror transport convention, undone by ``StepView``.
    """
    size = libero_geometry._dimension(image_size, "image_size")
    state = np.asarray(libero_geometry._vector(observation_state, 8, "observation_state"))
    if not isinstance(external_camera, str) or external_camera not in EXTERNAL_CAMERAS:
        raise ValueError("unsupported RoboCasa external camera")
    model, data = env.sim.model, env.sim.data
    wrist_id = _required_id(model, "camera", WRIST_CAMERA)
    observed_rotation = _check_robot_and_synchronization(model, data, state, wrist_id)
    cameras = {}
    for label, name in (("external", external_camera), ("wrist", WRIST_CAMERA)):
        cid = _required_id(model, "camera", name)
        fovy = libero_geometry._number(model.cam_fovy[cid], "fovy_deg")
        if not 0 < fovy < 180:
            raise ValueError("camera fovy must be between 0 and 180 degrees")
        focal = size / (2 * math.tan(math.radians(fovy) / 2))
        if label == "wrist":
            position = state[:3] + observed_rotation @ (WRIST_POSITION_HAND_M - HAND_TO_GRIP_M)
            rotation = observed_rotation @ WRIST_ROTATION_HAND_FROM_CAMERA
            source = "observed_robot_pose_fixed_mount"
        else:
            position = np.asarray(data.cam_xpos[cid], dtype=float)
            rotation = np.asarray(data.cam_xmat[cid], dtype=float).reshape(3, 3)
            source = EXTERNAL_POSE_SOURCE
        cameras[label] = {"mujoco_name": name, "pose_source": source, "fovy_deg": fovy,
                          "focal_length_px": [focal, focal],
                          "principal_point_px": [(size - 1) / 2] * 2,
                          "position_world_m": position.tolist(),
                          "rotation_world_from_camera": rotation.tolist()}
    return validate_geometry_metadata({"schema_version": GEOMETRY_VERSION,
        "frame": "simulation_world", "image": {"width": size, "height": size,
                                                  **libero_geometry._IMAGE_CONSTANTS},
        "gripper": copy.deepcopy(ROBOCASA_GRIPPER), "cameras": cameras})
