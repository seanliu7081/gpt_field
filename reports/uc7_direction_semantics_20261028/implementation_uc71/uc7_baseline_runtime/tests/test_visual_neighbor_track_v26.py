"""Exact unchanged descriptors, with actual recorded neighbor RGB replay."""
import json
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import decode_image
from gpt_field.visual_field import _gray
from gpt_field.visual_monitor_v5 import LocalFeatureBundle, _sample
from gpt_field.visual_neighbor_track_v26 import ExactDescriptorNeighborTrackV26
from gpt_field.visual_translation_guard_v11 import ProjectedTranslationGuardV11
from test_visual_monitor_v5 import textured, transform


@pytest.mark.parametrize("point", [[55., 55.], [55.2, 55.3], [55.5, 55.5], [55.75, 54.8]])
def test_exact_integer_and_fractional_descriptor_has_no_parabolic_motion(point):
    pixels = textured()
    track = ExactDescriptorNeighborTrackV26(pixels, [.5, .5], "neighbor")
    point = np.array(point)
    legacy = LocalFeatureBundle._match(track, pixels, pixels.copy(), point)
    assert legacy is not None
    assert np.linalg.norm(legacy[0]-point) > 1e-5
    result = track._match(pixels, pixels.copy(), point)
    assert np.array_equal(result[0], point) and result[1] == 1.


def test_repeated_exact_image_keeps_geometry_and_initial_identity_bank():
    pixels = textured()
    track = ExactDescriptorNeighborTrackV26(pixels, [.5, .5], "neighbor")
    initial = track.initial_points.copy()
    templates = [x.copy() for x in track.initial_templates]
    for _ in range(30):
        assert track.track(pixels.copy())
        assert np.array_equal(track.position, track.seed)
        assert np.array_equal(track.matrix, np.eye(2))
        assert np.array_equal(track.translation, np.zeros(2))
    assert np.array_equal(track.initial_points, initial)
    assert all(np.array_equal(a, b) for a, b in zip(templates, track.initial_templates))
    assert len(track.active) == len(initial)
    assert not track.summary()["exact_descriptor_matching"]["whole_object_stationarity_certified"]


@pytest.mark.parametrize("shift", [.2, .5, 1., 1.4, 2.5])
def test_actual_changed_descriptor_translation_uses_unchanged_legacy_match(shift):
    pixels = textured()
    moved = transform(pixels, (shift, .2))
    track = ExactDescriptorNeighborTrackV26(pixels, [.5, .5], "neighbor")
    point = np.array([55., 55.])
    assert not np.array_equal(_sample(pixels, point+track.offsets), _sample(moved, point+track.offsets))
    legacy = LocalFeatureBundle._match(track, pixels, moved, point)
    actual = track._match(pixels, moved, point)
    assert legacy is not None and actual is not None
    assert np.array_equal(actual[0], legacy[0]) and actual[1] == legacy[1]
    assert np.linalg.norm(actual[0]-point) > .05


def test_small_nonzero_photometric_change_is_not_rounded_into_exact_equality():
    pixels = textured()
    changed = pixels+1e-10
    track = ExactDescriptorNeighborTrackV26(pixels, [.5, .5], "neighbor")
    point = np.array([55.2, 55.3])
    legacy = LocalFeatureBundle._match(track, pixels, changed, point)
    actual = track._match(pixels, changed, point)
    assert not np.array_equal(_sample(pixels, point+track.offsets), _sample(changed, point+track.offsets))
    assert np.array_equal(actual[0], legacy[0]) and actual[1] == legacy[1]


@pytest.mark.parametrize("case", ["low_texture", "repeated_texture", "occlusion"])
def test_exactness_never_rescues_inherited_texture_ambiguity_or_occlusion_failure(case):
    pixels = textured()
    track = ExactDescriptorNeighborTrackV26(pixels, [.5, .5], "neighbor")
    if case == "low_texture":
        first = second = np.full_like(pixels, 100.)
    elif case == "repeated_texture":
        first = second = np.tile(pixels[:4, :4], (28, 28))
    else:
        first, second = pixels, np.full_like(pixels, 100.)
    assert LocalFeatureBundle._match(track, first, second, np.array([55., 55.])) is None
    assert track._match(first, second, np.array([55., 55.])) is None


def test_success_outside_rounding_plus_parabola_bound_is_never_overridden(monkeypatch):
    pixels = textured()
    track = ExactDescriptorNeighborTrackV26(pixels, [.5, .5], "neighbor")
    point = np.array([55., 55.])
    distant = (point+np.array([1.0001, 0.]), .93)
    monkeypatch.setattr(LocalFeatureBundle, "_match", lambda *a, **kw: distant)
    assert track._match(pixels, pixels.copy(), point) is distant


def test_failed_identity_cannot_be_resurrected_by_restoring_exact_pixels():
    pixels = textured()
    track = ExactDescriptorNeighborTrackV26(pixels, [.5, .5], "neighbor")
    assert not track.track(np.full_like(pixels, 100.))
    assert not track.track(pixels)
    assert not track.reliable


def test_real_recorded_cabinet_descriptors_do_not_invent_scale_or_box_drift():
    path = Path(__file__).parents[1]/"runs/visual_field_v26_design_001/controller_diagnostic_001/trace.jsonl"
    if not path.exists():
        pytest.skip("Actual V26 diagnostic RGB archive unavailable")
    rows = [json.loads(line) for line in path.open()]
    initial = next(row["observation"] for row in rows if row["event"] == "reset")
    prediction = next(row["diagnostics"]["prediction"] for row in rows
                      if row["event"] == "action" and row["diagnostics"].get("prediction"))
    roi = np.array(next(row["external_roi_uv"] for row in prediction["scene_inventory"]
                        if row["id"] == "cabinet"))
    pixels = _gray(decode_image(initial["observation/image"]))
    legacy = ProjectedTranslationGuardV11._seed_neighbor(pixels, roi, "cabinet")
    corrected = ExactDescriptorNeighborTrackV26(pixels, legacy.seed/legacy.scale, "cabinet",
        support_radius_px=12., maximum_features=8, roi_uv=roi)
    assert np.array_equal(corrected.initial_points, legacy.initial_points)
    initial_points = corrected.initial_points.copy()
    count = 0
    for row in rows:
        if row["event"] != "executed":
            continue
        current = _gray(decode_image(row["observation"]["observation/image"]))
        # These are current coordinate descriptor measurements, without any
        # simulator truth, object-category stationarity assumption, or policy.
        assert all(np.array_equal(_sample(pixels, point+corrected.offsets),
                                  _sample(current, point+corrected.offsets))
                   for point in initial_points)
        assert legacy.track(current)
        assert corrected.track(current)
        assert np.array_equal(corrected.matrix, np.eye(2))
        assert np.array_equal(corrected.translation, np.zeros(2))
        assert np.array_equal(corrected.position, corrected.seed)
        assert corrected.last_inliers == 8
        count += 1
    assert count == 28
    assert np.linalg.norm(legacy.position-legacy.seed) > .6
    assert np.linalg.norm(legacy.matrix[0]) > 1.04
    assert not np.array_equal(legacy.matrix, corrected.matrix)
