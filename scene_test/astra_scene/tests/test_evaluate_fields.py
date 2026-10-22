"""Independent geometry/metric checks for the new field evaluator."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("astra_scene_evaluate_test_module", HERE / "evaluate_fields.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def sphere(center=(.2, -.1, 1.4), radius=.3):
    return {"schema_version": 1, "units": "m", "coordinate_frame": "camera_cv_xright_ydown_zforward",
            "estimated_focal_px": 500., "nodes": [
                {"name": "sphere", "op": "sphere", "center_m": list(center),
                 "rotation_xyzw": [0., 0., 0., 1.], "size_m": [radius, 0., 0.],
                 "left": None, "right": None}], "root": 0, "uncertainty_notes": "synthetic test only"}


def test_six_virtual_cameras_are_right_handed_and_centered():
    lower, upper = np.array([-.3, -.5, .8]), np.array([.7, .3, 2.])
    center = (lower + upper) / 2
    views = list(evaluation.six_views((lower, upper), side=65))
    assert len(views) == 6 and len({x["name"] for x in views}) == 6
    for view in views:
        R = view["rotation"]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-14)
        assert np.isclose(np.linalg.det(R), 1., atol=1e-14)
        local_center = (center - view["eye"]) @ R
        assert np.allclose(local_center[:2], 0., atol=1e-14)
        assert local_center[2] > 0
        projected = view["K"] @ local_center
        assert np.allclose(projected[:2] / projected[2], [32., 32.])
        local_ray = evaluation.camera_rays(view["shape"], view["K"])[32 * 65 + 32]
        world_ray = local_ray @ R.T
        assert np.allclose(world_ray, (center - view["eye"]) / local_center[2])


def test_virtual_view_analytic_parameters_are_axial_depth_not_normalized_range():
    field = sphere()
    center = np.asarray(field["nodes"][0]["center_m"])
    radius = field["nodes"][0]["size_m"][0]
    for view in evaluation.six_views(evaluation.field_bounds(field), side=9):
        local_rays = evaluation.camera_rays(view["shape"], view["K"])
        directions = local_rays @ view["rotation"].T
        origins = np.broadcast_to(view["eye"], directions.shape)
        got = evaluation.analytic_depth(field, origins, directions, view["shape"]).ravel()
        # Independent ray-sphere quadratic with arbitrary direction lengths.
        origin = origins - center
        a = np.sum(directions ** 2, axis=1)
        b = 2 * np.sum(origin * directions, axis=1)
        c = np.sum(origin ** 2, axis=1) - radius ** 2
        discriminant = b*b - 4*a*c
        expected = np.full(len(got), np.nan)
        hit = discriminant >= 0
        expected[hit] = (-b[hit] - np.sqrt(discriminant[hit])) / (2*a[hit])
        assert np.array_equal(np.isfinite(got), np.isfinite(expected))
        assert np.allclose(got[hit], expected[hit], atol=1e-12)
        points = origins[hit] + got[hit, None] * directions[hit]
        local_points = (points - view["eye"]) @ view["rotation"]
        assert np.allclose(local_points[:, 2], got[hit], atol=1e-12)


def test_representation_scores_keep_misses_and_new_hits_visible():
    original = np.array([[1., 1., np.nan, np.nan]])
    decoded = np.array([[1.0005, np.nan, 2., np.nan]])
    score = evaluation.representation_scores(decoded, original)
    assert score["reference_pixels"] == 2 and score["hit_pixels"] == 1
    assert score["miss_pixels"] == 1 and score["coverage"] == .5
    assert score["mae_on_hits_m"] == pytest.approx(.0005)
    assert score["within_threshold_all_reference_pixels"]["1mm"] == .5
    assert score["new_hit_where_original_misses"] == 1
    assert score["new_hit_fraction_all_test_rays"] == .25
    assert score["total_test_rays"] == 4


def test_empty_reference_does_not_report_perfect_accuracy():
    original = np.full((2, 2), np.nan)
    decoded = np.array([[1., np.nan], [np.nan, np.nan]])
    score = evaluation.representation_scores(decoded, original)
    assert score["reference_pixels"] == 0
    assert score["coverage"] is None and score["mae_on_hits_m"] is None
    assert score["within_threshold_all_reference_pixels"]["5mm"] is None
    assert score["new_hit_where_original_misses"] == 1


def test_all_decoded_misses_fail_all_reference_thresholds():
    score = evaluation.representation_scores(np.full((2, 2), np.nan), np.ones((2, 2)))
    assert score["coverage"] == 0 and score["miss_pixels"] == 4
    assert score["mae_on_hits_m"] is None
    assert set(score["within_threshold_all_reference_pixels"].values()) == {0.}


def test_primary_regions_use_exact_mask_denominator():
    reference = np.array([[1., 1., 1., np.nan]])
    predicted = np.array([[1., 1.006, np.nan, 1.]])
    target_mask = np.array([[False, True, True, True]])
    score = evaluation.depth_hit_scores(predicted, reference, target_mask)
    assert score["reference_pixels"] == 2
    assert score["hit_pixels"] == 1 and score["miss_pixels"] == 1
    assert score["mae_on_hits_m"] == pytest.approx(.006)
    assert score["within_threshold_all_reference_pixels"]["5mm"] == 0
    assert score["within_threshold_all_reference_pixels"]["10mm"] == .5
