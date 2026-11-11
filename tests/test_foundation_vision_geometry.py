"""Sensor-only geometry tests; no model, simulator, or hidden scene inputs."""
import json

import numpy as np
import pytest

from gpt_field.foundation_vision.geometry import GeometryConfig, build_geometry
from gpt_field.foundation_vision.types import CameraFrame, DepthEstimate, VisionFrame


def scene(*, shape=(21, 21), depth_a=2.0, depth_b=2.0, baseline=0.2, resized=None,
          scores=None):
    height, width = shape
    k = np.array([[10.0, 0, (width - 1) / 2], [0, 10.0, (height - 1) / 2], [0, 0, 1]])
    cameras = []
    depths = {}
    for name, x, depth in (("a", 0, depth_a), ("b", baseline, depth_b)):
        extrinsic = np.eye(4)
        extrinsic[0, 3] = -x
        camera = CameraFrame(name, np.zeros((*shape, 3), dtype=np.uint8), k, extrinsic, 5.0)
        cameras.append(camera)
        depth_shape = resized or shape
        sx, sy = depth_shape[1] / width, depth_shape[0] / height
        affine = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        raster = np.full(depth_shape, depth, dtype=float) if np.ndim(depth) == 0 else depth
        depths[name] = DepthEstimate(name, raster, affine @ k, extrinsic, score_map=scores)
    return VisionFrame(7, 5.0, tuple(cameras)), depths


def full_masks(frame):
    return {"target": {camera.name: np.ones(camera.rgb.shape[:2], dtype=bool) for camera in frame.cameras}}


def test_calibrated_world_points_round_trip_nonidentity_transform():
    frame, depths = scene()
    angle = 0.31
    transform = np.array([[np.cos(angle), -np.sin(angle), 0, 0.4],
                          [np.sin(angle), np.cos(angle), 0, -0.3],
                          [0, 0, 1, 0.1], [0, 0, 0, 1]])
    original = frame.cameras[0]
    camera = CameraFrame("a", original.rgb, original.intrinsics, transform, 5.0)
    frame = VisionFrame(7, 5.0, (camera, frame.cameras[1]))
    depths["a"] = DepthEstimate("a", depths["a"].depth_m, camera.intrinsics, transform)
    evidence = build_geometry(frame, {"a": depths["a"]}, {})
    points = np.asarray(evidence.to_dict()["surface_points_world"])
    camera_points = points @ transform[:3, :3].T + transform[:3, 3]
    projected = camera_points @ camera.intrinsics.T
    uv = projected[:, :2] / projected[:, 2, None]
    rows, cols = np.indices((21, 21))
    np.testing.assert_allclose(uv, np.column_stack((cols.ravel(), rows.ravel())), atol=1e-12)
    np.testing.assert_allclose(camera_points[:, 2], 2.0)


def test_supported_plane_geometry_is_partial_and_json_safe():
    frame, depths = scene()
    evidence = build_geometry(frame, depths, full_masks(frame))
    obj = evidence.object_geometry["target"]
    assert obj["status"] == "supported_surface_hypothesis"
    assert obj["association_status"] == "crossview_supported"
    assert obj["supported_point_count"] > 100
    assert obj["centroid_world_m"][2] == pytest.approx(2.0)
    assert obj["bounds_world_m"]["min"][2] == pytest.approx(2.0)
    np.testing.assert_allclose(obj["normal_world"], [0, 0, -1])
    assert obj["error_bound_m"] is None
    assert obj["bounds_are_complete_object_extent"] is False
    json.dumps(evidence.to_dict(), allow_nan=False)


def test_query_requires_supported_visible_rays_and_preserves_unknown():
    frame, depths = scene()
    evidence = build_geometry(frame, depths, {})
    result = evidence.query([[0, 0, 1], [0, 0, 2], [0, 0, 3], [100, 0, 1],
                             [0, 0, -1], [np.nan, 0, 1]], now_s=5.1, max_age_s=0.5)
    assert [item["state"] for item in result] == ["estimated_free", "surface", "unknown", "unknown", "unknown", "unknown"]
    assert result[2]["reason"] == "behind_observed_surface"
    assert result[-1]["reason"] == "invalid_query_point"


def test_behind_occluder_is_not_a_contradiction_but_front_disagreement_is():
    frame, depths = scene(depth_a=2, depth_b=1)
    evidence = build_geometry(frame, depths, {})
    details = evidence.to_dict()["views"]
    assert details["a"]["occluded_in_other_views_points"] > 0
    assert details["a"]["contradicted_points"] == 0
    assert details["b"]["contradicted_points"] > 0
    # A nearer conflicting surface cannot be erased by the farther prediction.
    assert evidence.query([[0, 0, 1]], 5, 1)[0]["state"] == "surface"
    assert evidence.query([[0, 0, 0.5]], 5, 1)[0]["state"] == "unknown"


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -1.0, 0.0])
def test_invalid_depth_cannot_create_free_space(invalid):
    frame, depths = scene(depth_a=invalid)
    evidence = build_geometry(frame, depths, {})
    assert evidence.to_dict()["views"]["a"]["valid_depth_pixels"] == 0
    assert evidence.query([[0, 0, 1]], 5, 1)[0]["state"] == "unknown"
    assert evidence.query([[0, 0, 2]], 5, 1)[0]["state"] == "surface"
    json.dumps(evidence.to_dict(), allow_nan=False)


def test_all_invalid_and_missing_depth_are_unknown_not_empty_free_scene():
    frame, depths = scene(depth_a=np.nan, depth_b=np.nan)
    for estimates in (depths, {}):
        evidence = build_geometry(frame, estimates, full_masks(frame))
        assert evidence.object_geometry["target"]["centroid_world_m"] is None
        assert evidence.to_dict()["surface_hypothesis_count"] == 0
        assert evidence.query([[0, 0, 1]], 5, 1)[0]["state"] == "unknown"


def test_duplicate_camera_poses_do_not_count_as_independent_support():
    frame, depths = scene(baseline=0)
    evidence = build_geometry(frame, depths, full_masks(frame))
    assert evidence.query([[0, 0, 1]], 5, 1)[0]["state"] == "unknown"
    assert evidence.object_geometry["target"]["status"] == "unknown"


def test_stale_and_future_observations_never_clear_space():
    frame, depths = scene()
    evidence = build_geometry(frame, depths, {})
    for now in (4.0, 7.0):
        assert evidence.query([[0, 0, 1], [0, 0, 2]], now, 0.5)[0]["state"] == "stale"
        assert evidence.assess_path([[0, 0, 1], [0.1, 0, 1]], 0.01, now, 0.5)["status"] == "stale"


def test_resized_depth_pixel_centres_map_to_original_mask_precisely():
    frame, depths = scene(shape=(20, 20), resized=(10, 10))
    # Depth pixel (1,1) corresponds to original continuous pixel (2.5,2.5),
    # whose nearest original pixel is (3,3), not the naive scaled pixel (2,2).
    mask = np.zeros((20, 20), dtype=bool)
    mask[3, 3] = True
    evidence = build_geometry(frame, {"a": depths["a"]}, {"target": {"a": mask}})
    assert evidence.object_geometry["target"]["surface_hypothesis_count"] == 1
    wrong_mask = np.zeros_like(mask)
    wrong_mask[2, 2] = True
    wrong = build_geometry(frame, {"a": depths["a"]}, {"target": {"a": wrong_mask}})
    assert wrong.object_geometry["target"]["surface_hypothesis_count"] == 0


def test_incorrect_resized_intrinsics_and_changed_extrinsics_are_rejected():
    frame, depths = scene(shape=(20, 20), resized=(10, 10))
    camera = frame.cameras[0]
    naive_k = camera.intrinsics.copy()
    naive_k[:2] *= 0.5
    bad = DepthEstimate("a", depths["a"].depth_m, naive_k, camera.world_to_camera)
    with pytest.raises(ValueError, match="pixel-centre"):
        build_geometry(frame, {"a": bad}, {})
    changed = camera.world_to_camera.copy()
    changed[0, 3] += 1e-9
    bad = DepthEstimate("a", depths["a"].depth_m, depths["a"].intrinsics, changed)
    with pytest.raises(ValueError, match="extrinsics"):
        build_geometry(frame, {"a": bad}, {})


def test_crossview_masks_of_different_instances_are_not_fused():
    frame, depths = scene()
    left = np.zeros((21, 21), dtype=bool)
    right = np.zeros_like(left)
    left[7:14, 3:8] = True
    right[7:14, 13:18] = True
    evidence = build_geometry(frame, depths, {"same_key": {"a": left, "b": right}})
    obj = evidence.object_geometry["same_key"]
    assert obj["association_status"] == "rejected_mixed_instance"
    assert obj["centroid_world_m"] is None
    assert obj["supported_point_count"] == 0


def test_unselected_scene_obstacle_blocks_between_clear_endpoints():
    frame, depths = scene(scores=np.zeros((21, 21)))
    # Only one depth map means low support. Empty targets must not remove the
    # scene, and the deliberately tiny JSON output must not remove obstacles.
    evidence = build_geometry(frame, {"a": depths["a"]}, {},
                              GeometryConfig(max_serialized_points=1))
    output = evidence.to_dict()
    assert len(output["surface_points_world"]) == 1
    assert output["surface_hypothesis_count"] == 441
    start, end = [-0.4, 0, 1.8], [0.4, 0, 2.2]
    assert all(item["state"] != "surface" for item in evidence.query([start, end], 5, 1))
    result = evidence.assess_path([start, end], 0.01, 5, 1)
    assert result["status"] == "blocked"
    assert result["reason"] == "observed_surface_intersects_tube"
    assert result["full_arm_checked"] is False


def test_path_distinguishes_supported_free_from_occluded_and_unseen():
    frame, depths = scene()
    evidence = build_geometry(frame, depths, {})
    free = evidence.assess_path([[-0.2, 0, 1], [0.2, 0, 1]], 0.01, 5, 1)
    assert free["status"] == "estimated_clear"
    assert free["continuous_clearance_certified"] is False
    assert evidence.assess_path([[-0.2, 0, 3], [0.2, 0, 3]], 0.01, 5, 1)["status"] == "unknown"
    assert evidence.assess_path([[100, 0, 1], [101, 0, 1]], 0.01, 5, 1)["status"] == "unknown"


def test_masks_must_use_original_boolean_raster_and_named_cameras():
    frame, depths = scene()
    with pytest.raises(ValueError, match="original RGB"):
        build_geometry(frame, depths, {"target": {"a": np.ones((10, 10), dtype=bool)}})
    with pytest.raises(ValueError, match="boolean"):
        build_geometry(frame, depths, {"target": {"a": np.ones((21, 21), dtype=float)}})
    with pytest.raises(ValueError, match="unknown camera"):
        build_geometry(frame, depths, {"target": {"missing": np.ones((21, 21), dtype=bool)}})


def test_serialized_object_evidence_is_detached_and_scores_never_clear_space():
    frame, depths = scene(scores=np.full((21, 21), np.nan))
    evidence = build_geometry(frame, depths, full_masks(frame))
    result = evidence.to_dict()
    json.dumps(result, allow_nan=False)
    assert result["views"]["a"]["uncalibrated_score_range"] is None
    result["object_geometry"]["target"]["centroid_world_m"][2] = 999
    assert evidence.object_geometry["target"]["centroid_world_m"][2] == pytest.approx(2)
    assert evidence.query([[0, 0, 2]], 5, 1)[0]["state"] == "surface"
