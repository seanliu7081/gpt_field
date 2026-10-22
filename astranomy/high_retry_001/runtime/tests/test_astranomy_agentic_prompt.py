"""Robot-only geometry and prompt boundary tests; no simulator or model calls."""
import copy
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.astranomy_agentic_geometry import (
    extract_geometry_metadata, project_world_point, validate_geometry_metadata,
    wrist_camera_pose_from_observation,
)
from gpt_field.astranomy_agentic_prompt import instructions
from gpt_field.distance_tasks import resolve_distance_task


class CameraOnlyModel:
    cam_fovy = np.array([60.0, 75.0])

    def camera_name2id(self, name):
        return {"agentview": 0, "robot0_eye_in_hand": 1}[name]

    def __getattr__(self, name):
        raise AssertionError(f"Forbidden model field read: {name}")


class CameraOnlyData:
    def __init__(self):
        self.cam_xpos = np.array([[1.0, 2.0, 3.0], [-0.1, 0.2, 0.5]])
        self.cam_xmat = np.array([np.eye(3).ravel(), np.eye(3).ravel()])

    def __getattr__(self, name):
        raise AssertionError(f"Forbidden data field read: {name}")


def env():
    return SimpleNamespace(sim=SimpleNamespace(model=CameraOnlyModel(), data=CameraOnlyData()))


def test_extractor_uses_only_camera_fields_and_copies_data():
    simulator = env()
    before = simulator.sim.data.cam_xpos.copy()
    geometry = extract_geometry_metadata(simulator)
    assert set(geometry["cameras"]) == {"external", "wrist"}
    assert geometry["image"]["source_transform"] == "raw_mujoco_rgb_vertical_flip_no_crop"
    assert geometry["gripper"]["position_reference"] == "gripper0_grip_site"
    geometry["cameras"]["external"]["position_world_m"][0] = 200
    np.testing.assert_array_equal(simulator.sim.data.cam_xpos, before)
    json.dumps(geometry, allow_nan=False)


def test_camera_projection_signs_and_integer_pixel_centers():
    camera = extract_geometry_metadata(env())["cameras"]["external"]
    assert project_world_point([1, 2, 2], camera) == [255.5, 255.5]
    right = project_world_point([1.1, 2, 2], camera)
    up = project_world_point([1, 2.1, 2], camera)
    assert right[0] > 255.5 and right[1] == 255.5
    assert up[0] == 255.5 and up[1] < 255.5
    assert project_world_point([1, 2, 4], camera) is None


def test_rotated_camera_uses_columns_as_camera_axes():
    simulator = env()
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    simulator.sim.data.cam_xmat[1] = R.ravel()
    camera = extract_geometry_metadata(simulator)["cameras"]["wrist"]
    origin = np.array(camera["position_world_m"])
    projected = project_world_point(origin + R @ [0.1, 0.2, -1], camera)
    f = 256 / math.tan(math.radians(75) / 2)
    np.testing.assert_allclose(projected, [255.5 + 0.1 * f, 255.5 - 0.2 * f])


def test_geometry_reexport_tracks_moving_wrist_without_mutating_prior_observation():
    simulator = env()
    old = extract_geometry_metadata(simulator)
    simulator.sim.data.cam_xpos[1] += [0.1, 0, 0]
    new = extract_geometry_metadata(simulator)
    assert old["cameras"]["wrist"]["position_world_m"] == [-0.1, 0.2, 0.5]
    assert new["cameras"]["wrist"]["position_world_m"] == [0, 0.2, 0.5]


def test_observation_synchronization_uses_fixed_mount_in_observed_hand_frame():
    state = [1, 2, 3, 0, 0, math.pi / 2, 0.04, -0.04]
    position, rotation = wrist_camera_pose_from_observation(state)
    np.testing.assert_allclose(position, [1, 2.05, 2.903], atol=1e-14)
    np.testing.assert_allclose(rotation, [[-1, 0, 0], [0, 1, 0], [0, 0, -1]], atol=1e-14)
    simulator = env()
    before = extract_geometry_metadata(simulator, observation_state=state)
    # An endpoint camera pose can differ from the cached image's pose. It must
    # not leak into the calibration paired with that image and proprioception.
    simulator.sim.data.cam_xpos[1] += [0.5, 0.7, -0.3]
    simulator.sim.data.cam_xmat[1] *= -1
    after = extract_geometry_metadata(simulator, observation_state=state)
    assert before == after
    assert after['cameras']['wrist']['pose_source'] == 'observed_robot_pose_fixed_mount'
    np.testing.assert_allclose(after['cameras']['wrist']['position_world_m'], position)


@pytest.mark.parametrize('state', [[0] * 7, [0] * 7 + [True], [0] * 7 + [float('nan')]])
def test_invalid_observation_state_cannot_supply_wrist_calibration(state):
    with pytest.raises(ValueError):
        extract_geometry_metadata(env(), observation_state=state)


@pytest.mark.parametrize("path", [(), ("image",), ("gripper",), ("cameras",),
                                  ("cameras", "external"), ("cameras", "wrist")])
def test_unknown_scene_truth_field_rejected_at_every_mapping_boundary(path):
    geometry = extract_geometry_metadata(env())
    node = geometry
    for key in path:
        node = node[key]
    node["object_pose"] = [1, 2, 3]
    with pytest.raises(ValueError):
        validate_geometry_metadata(geometry)


@pytest.mark.parametrize("field,value", [
    ("position_world_m", [0, float("nan"), 0]),
    ("position_world_m", [0, True, 0]),
    ("fovy_deg", 0), ("fovy_deg", 180),
    ("rotation_world_from_camera", [[1, 0, 0], [0, 1, 0], [0, 0, -1]]),
    ("rotation_world_from_camera", [[2, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ("focal_length_px", [200, 200]),
    ("principal_point_px", [256, 256]),
    ("mujoco_name", "hidden_object_camera"),
    ("pose_source", "object_tracker"),
])
def test_invalid_geometry_rejected(field, value):
    geometry = extract_geometry_metadata(env())
    geometry["cameras"]["external"][field] = value
    with pytest.raises(ValueError):
        validate_geometry_metadata(geometry)


def test_validator_returns_deep_independent_copy():
    source = extract_geometry_metadata(env())
    result = validate_geometry_metadata(source)
    result["gripper"]["asset_provenance"][0]["sha256"] = "changed"
    result["cameras"]["wrist"]["rotation_world_from_camera"][0][0] = 0
    assert validate_geometry_metadata(source) == source
    assert result != source


def test_prompt_differs_only_in_official_task_sentence_and_adds_no_recipe():
    bowl, drawer = instructions("bowl_plate"), instructions("drawer_bowl")
    assert bowl.replace(resolve_distance_task("bowl_plate").prompt, "TASK") == drawer.replace(
        resolve_distance_task("drawer_bowl").prompt, "TASK")
    assert "pull the drawer" not in drawer
    assert "brief sequence of subgoals" in bowl
    assert "full chronological text and visual history" in bowl
    assert "0=open or 1=closed" in bowl
    assert "20 Hz" in bowl and "0.02 m" in bowl and "0.1 rad" in bowl
    assert "0.05 m" in bowl and "0.35 rad" in bowl and "1 through 5" in bowl
    assert "no done or give_up tool" in bowl
    assert "Code" not in bowl and "Codex" not in bowl


def test_prompt_validates_geometry_without_copying_observation_into_instructions():
    geometry = extract_geometry_metadata(env())
    assert instructions("bowl_plate", geometry) == instructions("bowl_plate")
    corrupted = copy.deepcopy(geometry)
    corrupted["gripper"]["current_object"] = "bowl"
    with pytest.raises(ValueError):
        instructions("bowl_plate", corrupted)
