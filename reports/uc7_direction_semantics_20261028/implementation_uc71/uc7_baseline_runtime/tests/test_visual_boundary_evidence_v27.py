"""Outline sensitivity is distinct from local correspondence-fit precision."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_boundary_evidence_v27 import capture_boundary_evidence, compare_gap_evidence
from gpt_field.visual_monitor_v5 import _similarity


def corners(left=0., top=0., right=100., bottom=100.):
    return np.array([[left, top], [right, top], [right, bottom], [left, bottom]])


def track(center=(50., 50.), spread=40., identity="tool"):
    xy = np.array([(x, y) for x in (-1., 0., 1.) for y in (-1., 0., 1.)])
    points = xy*spread+center
    return SimpleNamespace(reliable=True, ident=identity, initial_points=points,
                           active=np.arange(len(points)), points=points.copy(),
                           matrix=np.eye(2), translation=np.zeros(2))


def move(item, shift=(0., 0.), angle=0., scale=1., errors=None):
    c, s = np.cos(angle), np.sin(angle)
    m = scale*np.array([[c, s], [-s, c]])
    item.points = item.initial_points[item.active]@m+shift
    if errors is not None:
        item.points += errors
    item.matrix, item.translation = _similarity(item.initial_points[item.active], item.points)


def snapshot(item, box=None, epoch=1):
    return capture_boundary_evidence(item, corners() if box is None else box,
                                     representation_key=epoch)


def compare_shift(spread, shift):
    tool, neighbor = track(spread=spread), track(center=(200., 50.), identity="neighbor")
    before, obstacle = snapshot(tool), snapshot(neighbor, corners(150, 0, 250, 100))
    move(tool, (shift, 0.))
    return before, snapshot(tool), compare_gap_evidence(before, snapshot(tool), obstacle, obstacle)


def test_well_spread_correspondences_resolve_actual_closing_and_opening():
    _, _, closing = compare_shift(40., 4.)
    assert closing["comparable"] and closing["classification"] == "closing"
    assert closing["nominal_gap_change_px"] == pytest.approx(-4.)
    assert closing["gap_change_interval_px"][1] < 0
    _, _, opening = compare_shift(40., -4.)
    assert opening["classification"] == "opening"
    assert not closing["clearance_certified"]


def test_small_local_support_cannot_inherit_full_outline_precision():
    broad, _, broad_result = compare_shift(40., 4.)
    local, _, local_result = compare_shift(2., 4.)
    assert local["local_fit_residual_rms_px"] < 1e-10
    assert broad["local_fit_residual_rms_px"] < 1e-10
    assert np.max(local["corner_coordinate_radius_px"]) > 8*np.max(broad["corner_coordinate_radius_px"])
    assert broad_result["classification"] == "closing"
    assert local_result["classification"] == "inconclusive"


def test_scale_and_rotation_are_propagated_without_translation_only_substitution():
    item = track()
    move(item, (3., -2.), angle=.1, scale=1.07)
    evidence = snapshot(item)
    expected = corners()@item.matrix+item.translation
    assert evidence["available"]
    assert np.allclose(evidence["nominal_corners_px"], expected)
    assert not np.allclose(expected, corners()+item.translation)
    lo, hi = np.asarray(evidence["bounds_interval_px"])
    nominal = np.r_[expected.min(axis=0), expected.max(axis=0)]
    assert np.all(lo <= nominal) and np.all(hi >= nominal)


def test_actual_fit_residual_and_leave_one_out_instability_are_reported():
    item = track(spread=3.)
    clean = snapshot(item)
    errors = np.zeros_like(item.points);errors[0] = [1., -.7]
    move(item, errors=errors)
    evidence = snapshot(item)
    assert evidence["available"]
    assert evidence["local_fit_residual_rms_px"] > .1
    assert np.max(evidence["corner_coordinate_radius_px"]) > np.max(clean["corner_coordinate_radius_px"])
    assert np.max(evidence["leave_one_out_corner_spread_px"]) > .5


@pytest.mark.parametrize("change", ["epoch", "new_track", "corners", "feature_bank", "identity"])
def test_changed_representation_is_unavailable_not_progress(change):
    item, neighbor = track(), track(center=(200., 50.), identity="neighbor")
    before, obstacle = snapshot(item), snapshot(neighbor, corners(150, 0, 250, 100))
    if change == "epoch":
        current = snapshot(item, epoch=2)
    elif change == "new_track":
        current = snapshot(copy.deepcopy(item))
    elif change == "corners":
        current = snapshot(item, corners(0, 0, 110, 100))
    else:
        if change == "feature_bank":
            item.initial_points += .1;move(item)
        else:
            item.ident = "renamed"
        current = snapshot(item)
    result = compare_gap_evidence(before, current, obstacle, obstacle)
    assert not result["comparable"] and result["classification"] == "unavailable"
    assert result["reason"] == "measurement_representation_changed"


@pytest.mark.parametrize("case", ["lost", "too_few", "degenerate", "nonfinite", "stale_fit"])
def test_unusable_or_stale_fit_does_not_supply_evidence(case):
    item = track()
    if case == "lost":item.reliable = False
    elif case == "too_few":item.active = item.active[:2];item.points = item.points[:2]
    elif case == "degenerate":item.initial_points[:] = 50.;item.points[:] = 50.
    elif case == "nonfinite":item.points[0, 0] = np.nan
    elif case == "stale_fit":item.translation += 2.
    evidence = snapshot(item)
    assert not evidence["available"]
    assert compare_gap_evidence(evidence, evidence, evidence, evidence)["classification"] == "unavailable"


def test_common_image_translation_is_not_relative_gap_progress():
    tool, neighbor = track(), track(center=(200., 50.), identity="neighbor")
    old_tool, old_neighbor = snapshot(tool), snapshot(neighbor, corners(150, 0, 250, 100))
    move(tool, (5., -3.));move(neighbor, (5., -3.))
    result = compare_gap_evidence(old_tool, snapshot(tool), old_neighbor, snapshot(neighbor, corners(150, 0, 250, 100)))
    assert result["nominal_gap_change_px"] == pytest.approx(0.)
    assert result["classification"] == "inconclusive"
    assert not result["interval_narrowing_is_new_motion"]


def test_capture_and_compare_are_json_safe_and_do_not_mutate_track_or_outline():
    item = track();box = corners();original = copy.deepcopy(item.__dict__)
    evidence = capture_boundary_evidence(item, box, representation_key={"epoch": 7, "view": "external"})
    json.dumps(evidence, allow_nan=False)
    for name, before in original.items():
        after = getattr(item, name)
        assert np.array_equal(before, after) if isinstance(before, np.ndarray) else before == after
    assert np.array_equal(box, corners())


def test_correspondence_subset_change_with_same_bank_remains_comparable():
    item, neighbor = track(), track(center=(200., 50.), identity="neighbor")
    previous, obstacle = snapshot(item), snapshot(neighbor, corners(150, 0, 250, 100))
    item.active = item.active[:-1];item.points = item.points[:-1]
    move(item, (5., 0.))
    current = snapshot(item)
    assert current["available"]
    assert compare_gap_evidence(previous, current, obstacle, obstacle)["classification"] == "closing"


def test_actual_saved_approach_boundary_extrapolation_remains_inconclusive():
    path = Path(__file__).parents[1]/"runs/visual_field_v27_design_001/boundary_evidence_archive_audit.json"
    if not path.exists():
        pytest.skip("Actual saved-RGB boundary replay artifact unavailable")
    artifact = json.loads(path.read_text())
    assert artifact["actual_matching_transitions"] == 16
    assert artifact["simulator_calls"] == artifact["model_calls"] == 0
    snapshots = artifact["snapshots"]
    for ending, expected in (("after14", -4.82330744050671), ("after15", -4.53537485957753)):
        a, b = snapshots["before8"], snapshots[ending]
        current = compare_gap_evidence(a["tool"], b["tool"], a["cabinet"], b["cabinet"])
        assert current["comparable"] and current["classification"] == "inconclusive"
        assert current["nominal_gap_change_px"] == pytest.approx(expected)
        assert current["gap_change_interval_px"][0] < 0 < current["gap_change_interval_px"][1]
    previous, current = snapshots["after14"], snapshots["after15"]
    increment = compare_gap_evidence(previous["tool"], current["tool"], previous["cabinet"], current["cabinet"])
    assert increment["nominal_gap_change_px"] > 0
    assert increment["classification"] == "inconclusive"
