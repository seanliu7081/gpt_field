"""Robot/camera boundary checks without loading RoboCasa or making model calls."""
import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field import astranomy_agentic_geometry as legacy
from gpt_field.agent_speedup_v2_user_check_sim import CalibratedObservation
from gpt_field.contact_skills import StepView
from gpt_field.observation import image_to_data_url
from gpt_field.robocasa_user_check_geometry import (
    DEFAULT_EXTERNAL_CAMERA, EXTERNAL_POSE_SOURCE, GEOMETRY_VERSION,
    GRIP_SITE, HAND_BODY, HAND_TO_GRIP_M, ROBOCASA_GRIPPER,
    WRIST_CAMERA, WRIST_POSITION_HAND_M, WRIST_ROTATION_HAND_FROM_CAMERA,
    extract_robocasa_geometry, validate_geometry_metadata,
)
from gpt_field.robot_tool_geometry import _rotation_matrix


class RobotCameraOnlyModel:
    def __init__(self):
        self.cam_fovy = np.array([60., 75.])
        self.cam_bodyid = np.array([1, 0])
        self.cam_mode = np.array([0, 0])
        self.cam_pos = np.array([[0., 0., 0.], WRIST_POSITION_HAND_M])
        self.cam_quat = np.array([[1., 0., 0., 0.], [0., 2 ** -.5, 2 ** -.5, 0.]])

    def camera_name2id(self, name):
        return {DEFAULT_EXTERNAL_CAMERA: 0, WRIST_CAMERA: 1}[name]

    def body_name2id(self, name):
        assert name == HAND_BODY
        return 0

    def site_name2id(self, name):
        assert name == GRIP_SITE
        return 0

    def __getattr__(self, name):
        raise AssertionError(f"Forbidden model field read: {name}")


class RobotCameraOnlyData:
    def __init__(self, state):
        rotation = _rotation_matrix(np.asarray(state[3:6]))
        grip = np.asarray(state[:3])
        hand = grip - rotation @ HAND_TO_GRIP_M
        self.body_xmat = np.array([rotation.ravel()])
        self.body_xpos = np.array([hand])
        self.site_xpos = np.array([grip])
        self.cam_xpos = np.array([[1., 2., 3.], hand + rotation @ WRIST_POSITION_HAND_M])
        self.cam_xmat = np.array([np.eye(3).ravel(),
                                  (rotation @ WRIST_ROTATION_HAND_FROM_CAMERA).ravel()])

    def __getattr__(self, name):
        raise AssertionError(f"Forbidden data field read: {name}")


STATE = [1., 2., 3., 0., 0., math.pi / 2, .04, -.04]


def simulator(state=STATE):
    return SimpleNamespace(sim=SimpleNamespace(model=RobotCameraOnlyModel(),
                                               data=RobotCameraOnlyData(state)))


def metadata():
    return extract_robocasa_geometry(simulator(), observation_state=STATE)


def test_extractor_reads_only_named_robot_and_camera_fields_and_uses_real_provenance():
    env = simulator()
    before = env.sim.data.cam_xpos.copy()
    result = extract_robocasa_geometry(env, observation_state=STATE)
    assert result["schema_version"] == GEOMETRY_VERSION
    assert result["cameras"]["external"]["mujoco_name"] == DEFAULT_EXTERNAL_CAMERA
    assert result["cameras"]["external"]["pose_source"] == EXTERNAL_POSE_SOURCE
    assert result["gripper"]["position_reference"] == GRIP_SITE
    assert result["gripper"]["asset_provenance"] != legacy._GRIPPER["asset_provenance"]
    result["cameras"]["external"]["position_world_m"][0] = 99
    np.testing.assert_array_equal(env.sim.data.cam_xpos, before)
    json.dumps(result, allow_nan=False)


def test_wrist_mount_is_verified_then_transformed_from_the_observation():
    result = metadata()["cameras"]["wrist"]
    np.testing.assert_allclose(result["position_world_m"], [1, 2.05, 2.903], atol=1e-14)
    np.testing.assert_allclose(result["rotation_world_from_camera"],
                               [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], atol=1e-14)
    assert result["pose_source"] == "observed_robot_pose_fixed_mount"


def test_external_projection_matches_actual_camera_pose_and_upright_pixel_axes():
    env = simulator()
    rotation = _rotation_matrix(np.array([.3, -.4, .2]))
    env.sim.data.cam_xmat[0] = rotation.ravel()
    camera = extract_robocasa_geometry(env, observation_state=STATE)["cameras"]["external"]
    origin = np.asarray(camera["position_world_m"])
    f = 512 / (2 * math.tan(math.radians(60) / 2))
    assert legacy.project_world_point(origin + rotation @ [0, 0, 1], camera) is None
    np.testing.assert_allclose(legacy.project_world_point(origin + rotation @ [.1, .2, -1], camera),
                               [255.5 + .1 * f, 255.5 - .2 * f], atol=1e-10)


@pytest.mark.parametrize("mutate", [
    lambda env: env.sim.model.cam_pos.__setitem__((1, 0), .03),
    lambda env: env.sim.model.cam_quat.__setitem__(1, [1, 0, 0, 0]),
    lambda env: env.sim.model.cam_bodyid.__setitem__(1, 1),
    lambda env: env.sim.model.cam_mode.__setitem__(1, 1),
    lambda env: env.sim.data.site_xpos.__setitem__((0, 2), 3.01),
])
def test_unrecognized_actual_mount_or_grip_offset_is_rejected(mutate):
    env = simulator()
    mutate(env)
    with pytest.raises(ValueError):
        extract_robocasa_geometry(env, observation_state=STATE)


@pytest.mark.parametrize("index", [0, 3])
def test_stale_position_or_orientation_cannot_be_paired_with_current_camera_pose(index):
    changed_state = list(STATE)
    changed_state[index] += .01
    with pytest.raises(ValueError, match="not synchronized"):
        extract_robocasa_geometry(simulator(), observation_state=changed_state)


@pytest.mark.parametrize("state", [None, [0] * 7, [0] * 7 + [True], [0] * 7 + [float("nan")]])
def test_observation_is_required_and_validated(state):
    with pytest.raises(ValueError):
        extract_robocasa_geometry(simulator(), observation_state=state)


@pytest.mark.parametrize("path", [(), ("image",), ("gripper",), ("cameras",),
                                  ("cameras", "external"), ("cameras", "wrist")])
def test_scene_truth_cannot_be_smuggled_into_metadata(path):
    result = metadata()
    node = result
    for key in path:
        node = node[key]
    node["object_pose"] = [1, 2, 3]
    with pytest.raises(ValueError):
        validate_geometry_metadata(result)


@pytest.mark.parametrize("field,value", [
    ("position_world_m", [0, float("nan"), 0]),
    ("fovy_deg", 0), ("fovy_deg", 180),
    ("rotation_world_from_camera", [[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    ("focal_length_px", [200, 200]),
    ("principal_point_px", [256, 256]),
    ("mujoco_name", "object_tracker"),
    ("pose_source", "fixed_world_camera"),
])
def test_camera_numerics_and_identity_are_validated(field, value):
    result = metadata()
    result["cameras"]["external"][field] = value
    with pytest.raises(ValueError):
        validate_geometry_metadata(result)


def test_unknown_schema_and_legacy_asset_claim_are_rejected():
    result = metadata()
    result["schema_version"] = "arbitrary-camera"
    with pytest.raises(ValueError, match="unsupported"):
        validate_geometry_metadata(result)
    result = metadata()
    result["gripper"] = copy.deepcopy(legacy._GRIPPER)
    with pytest.raises(ValueError, match="RoboCasa Panda asset"):
        validate_geometry_metadata(result)


def test_legacy_validation_is_byte_identical_and_copies_are_independent():
    # Construct a real legacy-schema document with the existing extractor.
    model = SimpleNamespace(cam_fovy=np.array([60., 75.]),
                            camera_name2id=lambda name: {"agentview": 0, WRIST_CAMERA: 1}[name])
    env = SimpleNamespace(sim=SimpleNamespace(model=model, data=RobotCameraOnlyData(STATE)))
    original = legacy.extract_geometry_metadata(env, observation_state=STATE)
    expected = json.dumps(legacy.validate_geometry_metadata(original))
    clean = validate_geometry_metadata(original)
    assert json.dumps(clean) == expected
    clean["gripper"]["asset_provenance"][0]["sha256"] = "changed"
    assert json.dumps(original) == expected
    new = metadata()
    copied = validate_geometry_metadata(new)
    copied["gripper"]["asset_provenance"][0]["sha256"] = "changed"
    assert new["gripper"] == ROBOCASA_GRIPPER


def test_calibrated_observation_keeps_upright_projection_and_original_serialization():
    upright = np.zeros((32, 32, 3), dtype=np.uint8)
    upright[3:6, 4:8] = [255, 0, 0]
    upright[20:24, 25:28] = [0, 255, 0]
    # Existing StepView reverses the observation transport's horizontal mirror.
    url = image_to_data_url(upright[:, ::-1], size=32, rotate=False)
    observation = CalibratedObservation(image=url, wrist_image=url, state=tuple(STATE),
        prompt="press the coffee machine button", joint_position=(0.,) * 7,
        joint_velocity=(0.,) * 7,
        geometry=extract_robocasa_geometry(simulator(), image_size=32, observation_state=STATE))
    np.testing.assert_array_equal(StepView(observation).external_rgb_upright, upright)
    np.testing.assert_array_equal(StepView(observation).wrist_rgb_upright, upright)
    assert set(observation.to_dict()) == {"observation/image", "observation/wrist_image",
                                           "observation/state", "prompt"}
    assert observation.geometry["schema_version"] == GEOMETRY_VERSION
