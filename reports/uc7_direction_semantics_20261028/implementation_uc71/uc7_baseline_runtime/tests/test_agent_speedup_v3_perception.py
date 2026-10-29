"""Synthetic public RGB exercises real stereo matching and temporal tracking.

The known renderer motion is never supplied to the perception implementation.
No simulator, model, hidden depth, segmentation, or object-pose API is used.
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gpt_field.agent_speedup_v3_perception import LocalRGBGeometryEvidence, _fit_rigid, _triangulate


_SIDE = 192
_FOCAL = 192.
_CENTRE = (_SIDE-1)/2
_TEXTURE = cv2.GaussianBlur(np.random.default_rng(22).integers(35, 245, (128, 128), dtype=np.uint8), (3, 3), .5)


def sensor(object_x=0., camera_shift=0., *, blank=False, missing=False, wrist_half_turn=False):
    origins = {"external": np.array([-.12, 0., 1.]), "wrist": np.array([.12+camera_shift, 0., 1.])}
    cameras = {}
    images = {}
    for name, origin in origins.items():
        y, x = np.indices((_SIDE, _SIDE), dtype=np.float32)
        world_x = origin[0]+(x-_CENTRE)/_FOCAL
        world_y = origin[1]-(y-_CENTRE)/_FOCAL
        u = (world_x-object_x+.22)/.44*127
        v = (.22-world_y)/.44*127
        gray = cv2.remap(_TEXTURE, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=20)
        if blank:
            gray[:] = 20
        camera_rotation = np.eye(3)
        if name == "wrist" and wrist_half_turn:
            gray = np.rot90(gray, 2).copy()
            camera_rotation = np.diag([-1., -1., 1.])
        images[name] = np.repeat(gray[:, :, None], 3, axis=2)
        cameras[name] = {"position_world_m": origin.tolist(), "rotation_world_from_camera": camera_rotation.tolist(),
                         "focal_length_px": [_FOCAL, _FOCAL], "principal_point_px": [_CENTRE, _CENTRE]}
    result = SimpleNamespace(external_rgb_upright=images["external"], wrist_rgb_upright=images["wrist"],
        observation=SimpleNamespace(geometry={"image": {"width": _SIDE, "height": _SIDE}, "cameras": cameras}))
    if missing:
        del result.wrist_rgb_upright
    return result


def boxes():
    return [{"center": np.zeros(3), "half": np.array([.22, .22, .005]),
             "rotation": np.eye(3), "uncertainty": .008}]


def test_calibrated_stereo_tracks_real_material_displacement_without_model_depth():
    tracker = LocalRGBGeometryEvidence()
    baseline = tracker.capture("package", boxes(), sensor(), 0, 1)
    assert baseline["status"] == "reference", baseline
    moved = tracker.assess("package", boxes(), sensor(object_x=.024, camera_shift=.009), 3, 1)
    assert moved["status"] == "supported", moved
    assert moved["center_displacement_m"] == pytest.approx([.024, 0, 0], abs=.006)
    assert moved["position_uncertainty_m"] > 0
    assert not moved["free_space_verified"] and not moved["attachment_verified"]
    assert moved["reference_geometry_revision"] == 1


def test_camera_motion_alone_does_not_relocate_object_or_refresh_geometry():
    tracker = LocalRGBGeometryEvidence()
    assert tracker.capture("package", boxes(), sensor(), 0, 1)["status"] == "reference"
    report = tracker.assess("package", boxes(), sensor(camera_shift=.025), 2, 1)
    assert report["status"] == "consistent", report
    assert report["pose_updated"] is False
    assert report["center_displacement_m"] == pytest.approx([0, 0, 0], abs=.004)


def test_missing_camera_and_material_loss_do_not_invent_displacement():
    tracker = LocalRGBGeometryEvidence()
    assert tracker.capture("package", boxes(), sensor(missing=True), 0, 1)["status"] == "unavailable"
    tracker.capture("package", boxes(), sensor(), 1, 1)
    for step, v in ((2, sensor(blank=True)), (3, sensor(missing=True))):
        report = tracker.assess("package", boxes(), v, step, 1)
        assert report["status"] != "supported"
        assert "world_translation_delta_m" not in report
    covered = [{"center": np.array([0., 0., .3]), "half": np.array([.8, .8, .02]), "rotation": np.eye(3)}]
    report = tracker.assess("package", boxes(), sensor(), 4, 1, occluder_boxes=covered)
    assert report["status"] == "occluded"


def test_stereo_weak_baseline_and_inconsistent_rigid_material_are_rejected():
    camera = {"origin": np.array([0., 0., 1.]), "rotation": np.eye(3),
              "focal": np.array([192., 192.]), "principal": np.array([96., 96.])}
    _, good, _, _ = _triangulate(np.array([[96., 96.]]), np.array([[96., 96.]]), [camera, camera])
    assert not good[0]
    rng = np.random.default_rng(4)
    original, unrelated = rng.normal(0, .1, (20, 3)), rng.normal(0, .1, (20, 3))
    with pytest.raises(ValueError, match="nonrigid_or_ambiguous"):
        _fit_rigid(original, unrelated, .003)


@pytest.mark.parametrize("wrist_half_turn", [False, True])
def test_explicit_local_anchor_repairs_origin_without_assuming_feature_is_center(wrist_half_turn):
    tracker = LocalRGBGeometryEvidence()
    v = sensor(wrist_half_turn=wrist_half_turn)
    point = np.array([.06, -.03, 0.])
    pixels = {}
    for name, camera in v.observation.geometry["cameras"].items():
        ray = (point-np.array(camera["position_world_m"]))@np.asarray(camera["rotation_world_from_camera"])
        pixels[name+"_uv"] = [_CENTRE+_FOCAL*ray[0]/-ray[2], _CENTRE-_FOCAL*ray[1]/-ray[2]]
    entity = {"position_m": [.03, 0., 0.], "quaternion_wxyz": [1, 0, 0, 0], "geometry_revision": 1,
              "uncertainty_m": .01, "boxes": [{"center_local_m": [0, 0, 0], "half_extents_m": [.22, .22, .005],
                                                "quaternion_wxyz": [1, 0, 0, 0]}],
              "anchors": {**pixels, "same_feature": True, "anchor_local_m": point.tolist()}}
    report = tracker.anchor_pose("package", entity, v, 0)
    assert report["status"] == "supported", report
    assert report["world_translation_delta_m"] == pytest.approx([-.03, 0., 0.], abs=.002)
    assert report["orientation_observed"] is False
    assert report["anchor_correspondence_method"] == "calibrated_local_surface_rectification_with_unique_material_peak"
    entity["anchors"]["anchor_local_m"] = None
    assert tracker.anchor_pose("package", entity, v, 0)["status"] == "unavailable"
