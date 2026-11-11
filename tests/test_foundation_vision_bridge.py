"""Coordinate, synchronization and public-input boundaries without a simulator."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from gpt_field import astranomy_agentic_geometry as public_geometry
from gpt_field.foundation_vision.bridge import (
    MANIFEST_VERSION, frame_from_observation, frame_from_step_view, load_manifest,
)
from gpt_field.observation import PolicyObservation, image_to_data_url


def metadata(state=None, size=8):
    state = state or [.2, -.1, 1., .2, -.3, .1, .04, -.04]
    position, rotation = public_geometry.wrist_camera_pose_from_observation(state)
    def camera(name, source, origin, r):
        return {"mujoco_name": name, "pose_source": source, "fovy_deg": 60.,
                "focal_length_px": [size / (2 * np.tan(np.pi / 6))] * 2,
                "principal_point_px": [(size - 1) / 2] * 2,
                "position_world_m": origin, "rotation_world_from_camera": r}
    return {"schema_version": public_geometry.GEOMETRY_VERSION, "frame": "simulation_world",
            "image": {"width": size, "height": size, **public_geometry._IMAGE_CONSTANTS},
            "gripper": deepcopy(public_geometry._GRIPPER),
            "cameras": {"external": camera("agentview", "fixed_world_camera", [0., 0., 2.], np.eye(3).tolist()),
                        "wrist": camera("robot0_eye_in_hand", "observed_robot_pose_fixed_mount", position, rotation)}}


def view(size=8):
    rgb = np.arange(size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    return SimpleNamespace(state=np.array([.2, -.1, 1., .2, -.3, .1, .04, -.04]),
                           external_rgb_upright=rgb, wrist_rgb_upright=rgb[::-1].copy())


def write_manifest(tmp_path, *, frame_count=1):
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(8)
    Image.fromarray(rgb).save(tmp_path / "left.png")
    Image.fromarray(rgb[:, ::-1]).save(tmp_path / "right.png")
    k = [[10., 0., 3.5], [0., 10., 3.5], [0., 0., 1.]]
    left, right = np.eye(4), np.eye(4)
    right[0, 3] = -.1
    frames = []
    for index in range(frame_count):
        frames.append({"frame_id": index, "captured_at_s": float(index), "cameras": [
            {"name": name, "rgb_path": image, "intrinsics": k,
             "world_to_camera": e.tolist(), "observed_at_s": float(index)}
            for name, image, e in (("external", "left.png", left), ("wrist", "right.png", right))]})
    data = {"schema_version": MANIFEST_VERSION, "source_kind": "synthetic_offline_fixture",
            "targets": [{"key": "target", "text": "a round object"}], "frames": frames}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data))
    return path, data


def test_opengl_to_opencv_preserves_public_pixel_projection_and_hand_origin():
    observed = view()
    calibration = metadata()
    frame = frame_from_step_view(observed, calibration, 12, 0.6)
    for camera in frame.cameras:
        old = calibration["cameras"][camera.name]
        world = np.asarray(old["position_world_m"]) + np.asarray(old["rotation_world_from_camera"]) @ [.12, -.08, -1.2]
        cv_point = camera.world_to_camera[:3, :3] @ world + camera.world_to_camera[:3, 3]
        projection = camera.intrinsics @ cv_point
        assert cv_point[2] > 0
        assert np.allclose(projection[:2] / projection[2], public_geometry.project_world_point(world, old))
        assert not camera.rgb.flags.writeable
    assert np.allclose(frame.hand_world[:3, 3] + frame.hand_world[:3, :3] @ [0, 0, .097], observed.state[:3])


def test_bridge_rejects_endpoint_calibration_or_calibration_from_another_state():
    calibration = metadata()
    calibration["cameras"]["wrist"]["pose_source"] = "current_simulator_camera_state"
    with pytest.raises(ValueError, match="same observed"):
        frame_from_step_view(view(), calibration, 0, 0.)
    calibration = metadata()
    calibration["cameras"]["wrist"]["position_world_m"][0] += .01
    with pytest.raises(ValueError, match="does not match"):
        frame_from_step_view(view(), calibration, 0, 0.)


def test_public_transport_mirror_is_undone_exactly_once():
    observed = view()
    wire = PolicyObservation(
        image_to_data_url(observed.external_rgb_upright[:, ::-1], size=8, rotate=False),
        image_to_data_url(observed.wrist_rgb_upright[:, ::-1], size=8, rotate=False),
        tuple(observed.state), "move the chosen object")
    frame = frame_from_observation(wire.to_dict(), metadata(), 0, 0.)
    assert np.array_equal(frame.cameras[0].rgb, observed.external_rgb_upright)
    assert np.array_equal(frame.cameras[1].rgb, observed.wrist_rgb_upright)


def test_public_truth_and_extra_geometry_cannot_cross_the_bridge():
    calibration = metadata()
    calibration["object_positions"] = {"secret": [1, 2, 3]}
    with pytest.raises(ValueError, match="exactly"):
        frame_from_step_view(view(), calibration, 0, 0.)
    with pytest.raises(ValueError, match="allowlisted"):
        frame_from_observation({"truth": True}, metadata(), 0, 0.)


def test_manifest_loads_upright_images_and_hashes_every_explicit_source(tmp_path):
    path, _ = write_manifest(tmp_path, frame_count=2)
    loaded = load_manifest(path)
    assert len(loaded.frames) == 2
    assert loaded.source_kind == "synthetic_offline_fixture"
    assert loaded.frames[0].cameras[0].rgb[0, 0, 0] == 0
    assert loaded.frames[0].cameras[0].rgb[0, -1, 0] == 7
    assert set(loaded.source_hashes) == {str(path), str(tmp_path / "left.png"), str(tmp_path / "right.png")}
    assert all(len(value) == 64 for value in loaded.source_hashes.values())


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(truth={"success": True}), "extra fields"),
    (lambda d: d["frames"][0]["cameras"][1].update(observed_at_s=.2), "timestamp"),
    (lambda d: d["frames"][1].update(captured_at_s=0.), "timestamp"),
    (lambda d: d["targets"].append(d["targets"][0].copy()), "unique"),
    (lambda d: d["targets"][0].update(seed_pixels={"external": [9., 2.]}), "outside"),
    (lambda d: d["frames"][0]["cameras"][0].update(contact=True), "extra fields"),
])
def test_manifest_rejects_ambiguous_or_nonpublic_input(tmp_path, mutation, match):
    path, data = write_manifest(tmp_path, frame_count=2)
    mutation(data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_manifest(path)


def test_manifest_duplicate_keys_fail_instead_of_silently_replacing(tmp_path):
    path, _ = write_manifest(tmp_path)
    path.write_text('{"schema_version":"x","schema_version":"y"}')
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(path)


def test_manifest_supports_saved_public_observation_and_checked_geometry(tmp_path):
    observed = view()
    observation = PolicyObservation(
        image_to_data_url(observed.external_rgb_upright[:, ::-1], size=8, rotate=False),
        image_to_data_url(observed.wrist_rgb_upright[:, ::-1], size=8, rotate=False),
        tuple(observed.state), "move the chosen object")
    (tmp_path / "observation.json").write_text(json.dumps(observation.to_dict()))
    (tmp_path / "geometry.json").write_text(json.dumps(metadata()))
    data = {"schema_version": MANIFEST_VERSION, "source_kind": "saved_public_observations",
            "targets": [{"key": "t", "text": "object"}], "frames": [
                {"frame_id": 9, "captured_at_s": .45, "observation_path": "observation.json", "geometry_path": "geometry.json"}]}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(data))
    result = load_manifest(path)
    assert result.frames[0].frame_id == 9
    assert np.array_equal(result.frames[0].cameras[0].rgb, observed.external_rgb_upright)
    assert len(result.source_hashes) == 3
