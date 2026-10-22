"""Actual image-flow checks; all images are generated, no model/simulator calls."""
import copy

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.operation_visual import (
    OperationVisualFeedback, tracking_reference_schema, validate_tracking_reference,
)

SIZE = 192
CENTERS = {"object": np.array([118, 90]), "tool": np.array([55, 85]), "target": np.array([120, 145])}
TEXTURES = {role: np.random.default_rng(index + 22).integers(30, 225, (29, 29), dtype=np.uint8)
            for index, role in enumerate(CENTERS)}


def observation(shifts=None, *, state_dx=0., blank=(), rotation=0., fingers=(.02, -.02)):
    shifts = shifts or {}
    images = {}
    for view in ("external", "wrist"):
        image = np.full((SIZE, SIZE), 15, dtype=np.uint8)
        for role, center in CENTERS.items():
            if (view, role) in blank:
                continue
            x, y = center + np.asarray(shifts.get((view, role), (0, 0)), dtype=int)
            image[y-14:y+15, x-14:x+15] = TEXTURES[role]
        images[view] = image_to_data_url(np.repeat(image[:, :, None], 3, axis=2), SIZE, rotate=False)
    return PolicyObservation(images["external"], images["wrist"],
        (.3 + state_dx, 0., .8, rotation, 0., 0., *fingers), "Manipulate the selected object")


def reference(*, external=True, wrist=True, identifier="ref-1"):
    result = {"reference_id": identifier}
    for view, enabled in (("external", external), ("wrist", wrist)):
        result[view] = None if not enabled else {role + "_roi_uv": (np.r_[center - 14, center + 14] / (SIZE - 1)).tolist()
                                                  for role, center in CENTERS.items()}
    return result


def configured(ref=None):
    first = observation()
    monitor = OperationVisualFeedback()
    diagnostic = monitor.configure(first, {"kind": "verify_grasp"},
                                   {"tracking_reference": reference() if ref is None else ref})
    assert diagnostic["tracking_valid"], diagnostic
    return monitor, first


def test_schema_validation_and_independent_role_view_availability():
    ref = reference(wrist=False)
    ref["external"]["target_roi_uv"] = None
    assert validate_tracking_reference(ref) == ref
    assert validate_tracking_reference(None) is None
    assert tracking_reference_schema()["anyOf"][0] == {"type": "null"}
    monitor, _ = configured(ref)
    diag = monitor.snapshot()
    assert diag["active_views"] == ["external"]
    assert set(diag["views"]["external"]) == {"tool", "object"}
    assert not diag["grasp_certified"] and not diag["task_success_certified"]


@pytest.mark.parametrize("defect", ["extra", "nan", "bool", "reverse", "empty_view", "empty_both", "id"])
def test_bad_reference_rejected(defect):
    ref = reference()
    if defect == "extra":
        ref["depth"] = 1.
    elif defect == "nan":
        ref["external"]["object_roi_uv"][0] = float("nan")
    elif defect == "bool":
        ref["external"]["object_roi_uv"][0] = True
    elif defect == "reverse":
        ref["external"]["object_roi_uv"] = [.9, .1, .2, .2]
    elif defect == "empty_view":
        ref["external"] = {name: None for name in ref["external"]}
    elif defect == "empty_both":
        ref["external"] = ref["wrist"] = None
    else:
        ref["reference_id"] = " "
    with pytest.raises(ValueError):
        validate_tracking_reference(ref)


def test_material_features_measure_motion_without_metric_object_pose_or_grasp_claim():
    monitor, previous = configured()
    for index in range(1, 8):
        current = observation({("external", "object"): (index * 2, index)}, state_dx=index * .002)
        diag = monitor.observe(previous, current)
        previous = current
    obj = diag["views"]["external"]["object"]
    assert obj["valid"] and obj["age"] == 7
    assert obj["displacement_px"] == pytest.approx([14., 7.], abs=.35)
    assert diag["verification_evidence"] == "unknown"
    assert diag["verification_reason"] == "verification_not_requested"
    assert diag["tracking_elapsed_s"] > 0.


def test_explicit_verification_detects_supporting_comotion_but_never_certifies_grasp():
    monitor, previous = configured()
    monitor.begin_verification(previous)
    for index in range(1, 5):
        shifts = {("external", role): (index * 2, 0) for role in ("object", "tool")}
        current = observation(shifts, state_dx=.003 * index)
        diag = monitor.observe(previous, current)
        previous = current
    assert diag["verification_evidence"] == "supporting", diag
    assert diag["measured_robot_displacement_m"] == pytest.approx(.012)
    assert not diag["grasp_certified"] and not diag["task_success_certified"]


def test_testlift_stationary_object_contradicts_attachment_with_actual_flow():
    monitor, previous = configured()
    monitor.begin_verification()
    for index in range(1, 5):
        current = observation({("external", "tool"): (index * 2, 0)}, state_dx=.003 * index)
        diag = monitor.observe(previous, current)
        previous = current
    assert diag["verification_evidence"] == "contradicted", diag
    assert "stationary" in diag["verification_reason"]


def test_wrist_only_tracking_cannot_prove_attachment():
    monitor, previous = configured(reference(external=False))
    monitor.begin_verification()
    for index in range(1, 5):
        current = observation({("wrist", "target"): (index, 0)}, state_dx=.003 * index)
        diag = monitor.observe(previous, current)
        previous = current
    assert diag["tracking_valid"] and diag["active_views"] == ["wrist"]
    assert diag["verification_evidence"] == "unknown"
    assert "wrist_is_indeterminate" in diag["verification_reason"]


def test_one_occluded_region_does_not_discard_other_views_or_roles():
    monitor, previous = configured()
    current = observation(blank=(("external", "tool"),))
    diag = monitor.observe(previous, current)
    assert not diag["views"]["external"]["tool"]["valid"]
    assert diag["views"]["external"]["object"]["valid"]
    assert diag["views"]["wrist"]["tool"]["valid"]
    assert diag["tracking_valid"]
    # Reappearance must not automatically reseed another surface with old identity.
    diag = monitor.observe(current, previous)
    assert not diag["views"]["external"]["tool"]["valid"]


def test_fixed_view_target_motion_reported_as_image_evidence_only():
    monitor, previous = configured()
    for index in range(1, 4):
        current = observation({("external", "target"): (index * 2, 0)})
        diag = monitor.observe(previous, current)
        previous = current
    assert diag["external_target_image_motion_observed"] is True
    assert not diag["task_success_certified"]


def test_same_reference_persists_but_changed_geometry_requires_new_identifier():
    monitor, previous = configured()
    current = observation({("external", "object"): (2, 0)})
    monitor.observe(previous, current)
    same = monitor.configure(current, {"kind": "transport"}, {"tracking_reference": reference()})
    assert same["views"]["external"]["object"]["age"] == 1
    ref = reference()
    ref["external"]["object_roi_uv"][0] += .01
    changed = monitor.configure(current, {"kind": "transport"}, {"tracking_reference": ref})
    assert not changed["tracking_valid"]
    assert "requires_new_id" in changed["status"]
    ref["reference_id"] = "ref-2"
    reseeded = monitor.configure(current, {"kind": "transport"}, {"tracking_reference": ref})
    assert reseeded["tracking_valid"] and reseeded["epoch"] == 2


def test_null_reference_clears_and_missing_backend_falls_back(monkeypatch):
    monitor, previous = configured()
    assert not monitor.configure(previous, {}, {"tracking_reference": None})["tracking_valid"]
    def unavailable():
        raise ImportError("No optional backend")
    monkeypatch.setattr("gpt_field.operation_visual._backend", unavailable)
    diag = monitor.configure(previous, {}, {"tracking_reference": reference()})
    assert diag["status"] == "optical_flow_backend_unavailable"
    assert not diag["tracking_valid"]


def test_discontinuous_observations_invalidate_without_action_or_reseed():
    monitor, _ = configured()
    not_previous = observation(state_dx=.001)
    after = observation(state_dx=.002)
    diag = monitor.observe(not_previous, after)
    assert not diag["tracking_valid"]
    assert "discontinuity" in diag["status"]
    assert diag["verification_evidence"] == "unknown"


@pytest.mark.parametrize("change,reason", [(dict(rotation=.3), "rotation"), (dict(fingers=(.04, -.04)), "finger")])
def test_verification_geometry_change_keeps_evidence_unknown(change, reason):
    monitor, previous = configured()
    monitor.begin_verification()
    for index in range(1, 5):
        current = observation({("external", role): (index * 2, 0) for role in ("object", "tool")},
                              state_dx=.003 * index, **change)
        diag = monitor.observe(previous, current)
        previous = current
    assert diag["verification_evidence"] == "unknown"
    assert reason in diag["verification_reason"]


def test_same_pixel_region_for_object_and_tool_is_not_attachment_evidence():
    ref = reference()
    ref["external"]["object_roi_uv"] = copy.deepcopy(ref["external"]["tool_roi_uv"])
    monitor, previous = configured(ref)
    monitor.begin_verification()
    for index in range(1, 5):
        current = observation({("external", "tool"): (index * 2, 0)}, state_dx=.003 * index)
        diag = monitor.observe(previous, current)
        previous = current
    assert diag["verification_evidence"] == "unknown"
    assert "overlap" in diag["verification_reason"]


def test_release_clears_verification_without_restarting_persistent_tracks():
    monitor, previous = configured()
    monitor.begin_verification()
    current = observation({("external", "tool"): (2, 0)}, state_dx=.003)
    monitor.observe(previous, current)
    diag = monitor.configure(current, {"kind": "release"}, {"tracking_reference": reference()})
    assert diag["verification_active"] is False
    assert diag["verification_evidence"] == "unknown"
    assert diag["views"]["external"]["tool"]["age"] == 1


def test_actual_rgb_similarity_change_and_partial_occlusion_keep_surviving_features():
    import cv2
    from gpt_field.observation import decode_image
    raw = observation()
    # Smooth material texture avoids aliasing independent random pixels during
    # fractional-pixel image resampling.
    original = cv2.GaussianBlur(np.asarray(decode_image(raw.image)), (3, 3), .8)
    encoded = image_to_data_url(original, SIZE, rotate=False)
    previous = PolicyObservation(encoded, raw.wrist_image, raw.state, raw.prompt)
    monitor = OperationVisualFeedback()
    monitor.configure(previous, {"kind": "approach"}, {"tracking_reference": reference()})
    center = tuple(float(x) for x in CENTERS["object"])
    for index in range(1, 7):
        warp = cv2.getRotationMatrix2D(center, index * 1., 1. + index * .008)
        warp[:, 2] += (index * 1.5, 0.)
        image = cv2.warpAffine(original, warp, (SIZE, SIZE), borderValue=(15, 15, 15))
        # One corner is briefly obscured. The surviving body support must carry
        # the estimate; no feature re-detection or identity replacement occurs.
        if index == 4:
            image[75:82, 105:114] = 15
        encoded = image_to_data_url(image, SIZE, rotate=False)
        current = PolicyObservation(encoded, previous.wrist_image,
                                    (.3 + index * .001, 0., .8, 0., 0., 0., .02, -.02), previous.prompt)
        diag = monitor.observe(previous, current)
        previous = current
    region = diag["views"]["external"]["object"]
    assert region["valid"], region
    assert region["displacement_px"] == pytest.approx([9., 0.], abs=.6)
    assert 4 <= region["support_count"] <= region["seed_support_count"]


def test_same_reference_cannot_transfer_evidence_to_a_changed_operation_target():
    monitor = OperationVisualFeedback()
    first = observation()
    monitor.configure(first, {"kind": "verify_grasp", "target_id": "item-a"},
                      {"tracking_reference": reference()})
    monitor.begin_verification()
    changed = monitor.configure(first, {"kind": "transport", "target_id": "item-b"},
                                {"tracking_reference": reference()})
    assert changed["tracking_valid"] is False
    assert changed["verification_active"] is False
    assert "target_changed" in changed["status"]


def test_verification_rotation_guard_is_invariant_to_pi_axis_angle_wrap():
    import math
    first = observation(rotation=math.pi - .001)
    monitor = OperationVisualFeedback()
    monitor.configure(first, {"kind": "verify_grasp"}, {"tracking_reference": reference()})
    monitor.begin_verification()
    previous = first
    for index in range(1, 5):
        shifts = {("external", role): (index * 2, 0) for role in ("object", "tool")}
        current = observation(shifts, state_dx=.003 * index, rotation=-math.pi + .001)
        diagnostic = monitor.observe(previous, current)
        previous = current
    # Representation difference is almost 2*pi; physical change is only .002rad.
    assert abs(first.state[3] - current.state[3]) > 6.
    assert diagnostic["verification_reason"] != "verification_rotation_changed_relation"
    assert diagnostic["verification_evidence"] == "supporting"
    assert diagnostic["grasp_certified"] is False
