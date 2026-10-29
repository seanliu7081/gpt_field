"""Offline synthetic RGB/state checks; no simulator or model invocation."""
import copy
import json

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.speed_first_pregrasp import (
    PREGRASP_VERSION, PregraspVisualFeedback, pregrasp_instructions,
    pregrasp_reference_schema, validate_pregrasp_reference,
)

SIZE = 192
BASE = np.array([.3, -.1, .8, 0., 0., 0., .04, -.04])
CENTERS = {"external_tool": [60, 66], "external_target": [104, 88],
           "wrist_tool": [60, 66], "wrist_target": [104, 94]}
JEXT = np.array([[1000., 0., 0.], [0., 1000., 0.]])
JWRIST = np.array([[0., 0., 1000.], [500., 500., 500.]])
TEXTURES = {key: np.random.default_rng(i + 41).integers(30, 230, (33, 33), dtype=np.uint8)
            for i, key in enumerate(CENTERS)}


def sample(delta=(0., 0., 0.), *, external_target_shift=(0., 0.), wrist_extra=(0., 0.),
           orientation=None, fingers=None, blank=None, duplicate=None, tool_extra=(0., 0.), wrist_tool_shift=(0., 0.)):
    delta = np.asarray(delta, float)
    state = BASE.copy()
    state[:3] += delta
    if orientation is not None:
        state[3:6] = orientation
    if fingers is not None:
        state[6:] = fingers
    points = {key: np.asarray(value, float).copy() for key, value in CENTERS.items()}
    points["external_tool"] += JEXT @ delta + tool_extra
    points["external_target"] += external_target_shift
    points["wrist_tool"] += wrist_tool_shift
    points["wrist_target"] -= JWRIST @ delta + wrist_extra
    images = {}
    for view in ("external", "wrist"):
        image = np.full((SIZE, SIZE), 12, dtype=np.uint8)
        for role in ("tool", "target"):
            key = view + "_" + role
            if blank == key:
                continue
            x, y = np.rint(points[key]).astype(int)
            image[y-16:y+17, x-16:x+17] = TEXTURES[key]
            if duplicate == key:
                image[y-16:y+17, x+17:x+50] = TEXTURES[key]
        images[view] = image_to_data_url(np.repeat(image[:, :, None], 3, axis=2), size=SIZE, rotate=False)
    return PolicyObservation(images["external"], images["wrist"], tuple(state), "Manipulate the selected object"), points


def reference(points=None, identifier="epoch-a"):
    points = CENTERS if points is None else points
    result = {"reference_id": identifier, "target_id": "selected_object"}
    for view in ("external", "wrist"):
        tool, target = (np.asarray(points[view + "_" + role], float) for role in ("tool", "target"))
        result[view] = {"tool_reference_mode": "tracked_material", "tool_contact_uv": (tool / (SIZE-1)).tolist(),
                       "target_contact_uv": (target / (SIZE-1)).tolist(),
                       "tool_roi_uv": (np.r_[tool-16, tool+16] / (SIZE-1)).tolist(),
                       "target_roi_uv": (np.r_[target-16, target+16] / (SIZE-1)).tolist(),
                       "desired_offset_uv": (np.array([-40., -20.]) / (SIZE-1)).tolist()}
    return result


TRAJECTORY = [(0., 0., 0.), (.002, 0., 0.), (.002, .002, 0.), (.002, .002, .002),
              (.004, .002, .002), (.004, .004, .002), (.004, .004, .004),
              (.002, .004, .004), (.002, .002, .004), (.002, .002, .002)]


def trained(trajectory=TRAJECTORY):
    tracker = PregraspVisualFeedback()
    initial, _ = sample(trajectory[0])
    configured = tracker.configure(initial, reference(), gripper_open=True, target_id="selected_object")
    assert configured["tracking_valid"], configured
    for delta in trajectory[1:]:
        latest, points = sample(delta)
        diagnostic = tracker.observe(latest)
    return tracker, latest, points, diagnostic


def test_schema_and_instructions_distinguish_semantic_intent_from_measurements():
    ref = reference()
    assert validate_pregrasp_reference(ref) == ref
    assert validate_pregrasp_reference(None) is None
    assert pregrasp_reference_schema()["anyOf"][0] == {"type": "null"}
    instructions = pregrasp_instructions()
    assert PREGRASP_VERSION in instructions
    assert "VISIBLE MATERIAL" in instructions and "not automatically zero" in instructions
    assert "invisible pinch midpoint" in instructions
    assert "No calibrated camera" in instructions
    assert "no extra model calls" in instructions


@pytest.mark.parametrize("defect", ["extra", "missing", "nan", "boolean", "outside_point", "reversed_roi", "long_id", "offset"])
def test_malformed_references_are_rejected_before_configuration(defect):
    ref = reference()
    if defect == "extra":
        ref["depth"] = 1.
    elif defect == "missing":
        del ref["wrist"]
    elif defect == "nan":
        ref["external"]["target_contact_uv"][0] = float("nan")
    elif defect == "boolean":
        ref["external"]["tool_roi_uv"][0] = True
    elif defect == "outside_point":
        ref["external"]["tool_contact_uv"] = [0., 0.]
    elif defect == "reversed_roi":
        ref["wrist"]["tool_roi_uv"] = [.7, .7, .2, .2]
    elif defect == "long_id":
        ref["reference_id"] = "a" * 65
    else:
        ref["wrist"]["desired_offset_uv"] = [.3, 0.]
    with pytest.raises(ValueError):
        validate_pregrasp_reference(ref)


def test_natural_measured_xyz_identifies_depth_only_after_held_out_validation():
    tracker = PregraspVisualFeedback()
    initial, _ = sample()
    tracker.configure(initial, reference(), gripper_open=True)
    for index, delta in enumerate(TRAJECTORY[1:]):
        obs, _ = sample(delta)
        diag = tracker.observe(obs)
        if index < 3:
            assert diag["response_validated"] is False
    assert diag["tracking_valid"] and diag["response_validated"], diag
    assert diag["observed_rank"] == 3 and diag["depth_support"] == pytest.approx(1., abs=1e-6)
    assert diag["held_out_samples"] >= 2
    fitted = np.asarray(diag["response_px_per_m"]).T
    assert fitted == pytest.approx(np.vstack((JEXT, JWRIST)), abs=8.)
    assert diag["grasp_certified"] is diag["task_success_certified"] is False
    assert diag["probe_requested"] is False


def test_rank_one_motion_only_corrects_supported_axis_without_depth_claim():
    trajectory = [(i * .002, 0., 0.) for i in range(7)]
    tracker, obs, _, diag = trained(trajectory)
    assert diag["response_validated"], diag
    assert diag["observed_rank"] == 1 and diag["depth_support"] < 1e-8
    nominal = np.asarray(obs.state[:3]) + [0., .003, .004]
    goal, result = tracker.corrected_goal(obs, nominal)
    assert result["proposal_available"] and goal is not None
    assert goal[1:] == pytest.approx(nominal[1:], abs=1e-9)


def test_goal_correction_is_bounded_uses_immutable_nominal_and_preserves_input():
    tracker, obs, _, _ = trained()
    nominal = np.asarray(obs.state[:3]) + [.002, .001, .001]
    before = nominal.copy()
    goal, diag = tracker.corrected_goal(obs, nominal)
    assert goal is not None and diag["proposal_available"]
    assert np.linalg.norm(goal - nominal) <= tracker.maximum_correction_m + 1e-12
    assert np.linalg.norm(goal - nominal) <= tracker.maximum_nominal_deviation_m
    assert np.linalg.norm(goal - BASE[:3]) <= tracker.maximum_local_radius_m
    assert nominal == pytest.approx(before)
    again, _ = tracker.corrected_goal(obs, nominal)
    assert again == pytest.approx(goal)  # No accumulating corrections at a fixed observation.
    assert tracker.corrected_goal(obs, BASE[:3] + [.06, 0., 0.])[0] is None


def test_pushed_target_in_fixed_view_invalidates_before_adding_robot_response():
    tracker, obs, _, prior = trained()
    changed, _ = sample(TRAJECTORY[-1], external_target_shift=[3., 0.])
    diagnostic = tracker.observe(changed)
    assert diagnostic["status"] == "fixed_camera_target_moved_no_response_learning"
    assert not diagnostic["response_validated"] and not diagnostic["tracking_valid"]
    assert diagnostic["sample_count"] == 0
    assert tracker.corrected_goal(changed, changed.state[:3])[0] is None


def test_small_fixed_target_tracking_noise_is_not_fit_as_tool_jacobian():
    tracker = PregraspVisualFeedback()
    first, _ = sample()
    tracker.configure(first, reference(), gripper_open=True)
    for index, delta in enumerate(TRAJECTORY[1:]):
        obs, _ = sample(delta, external_target_shift=[float(index % 2), 0.])
        diag = tracker.observe(obs)
    assert diag["response_validated"], diag
    fitted = np.asarray(diag["response_px_per_m"]).T
    assert fitted[:2] == pytest.approx(JEXT, abs=8.)


@pytest.mark.parametrize("mutation,status", [
    ({"orientation": [.08, 0., 0.]}, "measured_orientation_changed"),
    ({"fingers": [.025, -.025]}, "measured_finger_width_changed"),
])
def test_pose_or_finger_change_invalidates_epoch_and_same_id_cannot_restart(mutation, status):
    tracker, _, _, _ = trained()
    obs, points = sample(TRAJECTORY[-1], **mutation)
    diag = tracker.observe(obs)
    assert diag["status"] == status
    assert tracker.configure(obs, reference(points), gripper_open=True)["new_reference_id_required"]
    new = tracker.configure(obs, reference(points, "new-epoch"), gripper_open=True)
    assert new["tracking_valid"] and not new["response_validated"]
    assert new["sample_count"] == 0


def test_epoch_domain_limit_and_no_motion_pixel_drift_only_disable_local_correction():
    tracker, _, _, _ = trained()
    far, _ = sample((.045, 0., 0.))
    assert tracker.observe(far)["status"] == "measured_translation_left_local_domain"
    other = PregraspVisualFeedback()
    initial, _ = sample()
    other.configure(initial, reference(), gripper_open=True)
    moved_pixels, _ = sample(tool_extra=[3., 0.])
    diag = other.observe(moved_pixels)
    assert diag["views"]["external"]["status"] == "pixel_drift_without_measured_robot_motion"
    assert diag["active_views"] == ["wrist"]
    assert diag["fallback"] == "continue_model_selected_field_or_direct"
    assert diag["probe_requested"] is False


def test_occlusion_ambiguous_texture_and_textureless_rois_do_not_fake_contact_tracking():
    tracker, _, _, _ = trained()
    occluded, _ = sample(TRAJECTORY[-1], blank="wrist_target")
    diagnostic = tracker.observe(occluded)
    assert diagnostic["tracking_valid"] and diagnostic["active_views"] == ["external"]
    assert diagnostic["validated_views"] == ["external"]
    blank, _ = sample(blank="external_tool")
    assert PregraspVisualFeedback().configure(blank, reference(), gripper_open=True)["active_views"] == ["wrist"]
    # Four identical periodic patches create alternate perfect NCC peaks.
    initial, _ = sample()
    gray = np.tile(np.random.default_rng(91).integers(0, 255, (12, 12), dtype=np.uint8), (16, 16))
    url = image_to_data_url(np.repeat(gray[:, :, None], 3, axis=2), size=SIZE, rotate=False)
    ambiguous = PolicyObservation(url, url, initial.state, initial.prompt)
    assert not PregraspVisualFeedback().configure(ambiguous, reference(), gripper_open=True)["tracking_valid"]


def test_same_epoch_preserves_fit_only_for_same_current_points_target_and_objective():
    tracker, obs, points, diag = trained()
    retained = tracker.configure(obs, reference(points), gripper_open=True, target_id="selected_object")
    assert retained["mapping_preserved_across_model_call"] and retained["response_validated"]
    assert retained["sample_count"] == diag["sample_count"]
    wrong = reference(points)
    wrong["external"]["desired_offset_uv"][0] += .02
    assert tracker.configure(obs, wrong, gripper_open=True)["status"] == "same_id_desired_relation_changed"
    assert tracker.corrected_goal(obs, obs.state[:3])[0] is None


def test_null_reference_wrong_target_or_closed_command_immediately_falls_back():
    obs, _ = sample()
    for ref, kwargs in ((None, {"gripper_open": True}),
                        (reference(), {"gripper_open": False}),
                        (reference(), {"gripper_open": True, "target_id": "different"})):
        tracker = PregraspVisualFeedback()
        diagnostic = tracker.configure(obs, ref, **kwargs)
        assert not diagnostic["tracking_valid"]
        assert tracker.corrected_goal(obs, obs.state[:3])[0] is None


def test_inconsistent_held_out_response_is_not_a_validated_fit():
    tracker = PregraspVisualFeedback()
    obs, _ = sample()
    tracker.configure(obs, reference(), gripper_open=True)
    for index, delta in enumerate(TRAJECTORY[1:]):
        obs, _ = sample(delta, wrist_extra=[8. if index % 2 else -8., 0.])
        diag = tracker.observe(obs)
        if not diag["tracking_valid"]:
            break
    assert not diag["views"]["wrist"]["response_validated"]
    assert "wrist" not in diag["correction_views"]
    # The independently validated external camera remains usable.
    assert diag["views"]["external"]["response_validated"]


def test_diagnostics_are_detached_and_serializable_and_no_observation_is_mutated():
    tracker, obs, _, _ = trained()
    before = obs.to_dict()
    diag = tracker.diagnostics()
    json.dumps(diag, allow_nan=False)
    diag["status"] = "tampered"
    assert tracker.diagnostics()["status"] != "tampered"
    assert obs.to_dict() == before


def test_every_frozen_patch_stays_inside_declared_material_roi():
    tracker = PregraspVisualFeedback()
    obs, _ = sample()
    assert tracker.configure(obs, reference(), gripper_open=True)["tracking_valid"]
    for track in tracker._tracks.values():
        for patch in track.patches:
            assert np.all(patch["seed"] - track.radius >= track.roi[0])
            assert np.all(patch["seed"] + track.radius <= track.roi[1])
    narrow = reference()
    center = np.asarray(narrow["external"]["tool_contact_uv"])
    narrow["external"]["tool_roi_uv"] = np.r_[center-.025, center+.025].tolist()
    small = PregraspVisualFeedback()
    assert small.configure(obs, narrow, gripper_open=True)["tracking_valid"]
    assert small._tracks["external_tool"].radius == 2


def test_same_tool_target_support_cannot_create_zero_error_self_tracking():
    obs, _ = sample()
    ref = reference()
    for view in ("external", "wrist"):
        ref[view]["target_contact_uv"] = ref[view]["tool_contact_uv"].copy()
        ref[view]["target_roi_uv"] = ref[view]["tool_roi_uv"].copy()
        ref[view]["desired_offset_uv"] = [0., 0.]
    tracker = PregraspVisualFeedback()
    diagnostic = tracker.configure(obs, ref, gripper_open=True)
    assert diagnostic["status"] == "tool_target_body_support_overlaps"
    assert not diagnostic["image_alignment_observed"]
    assert tracker.corrected_goal(obs, obs.state[:3])[0] is None


def test_proposed_goal_reduces_predicted_image_error_relative_to_uncorrected_nominal():
    tracker, obs, _, diagnostic = trained()
    current = np.asarray(obs.state[:3])
    nominal = current + [.002, .001, -.001]
    goal, proposed = tracker.corrected_goal(obs, nominal)
    assert proposed["proposal_available"]
    response = np.asarray(diagnostic["response_px_per_m"])
    error = np.asarray(diagnostic["image_error_px"])
    uncorrected_error = error + (nominal-current) @ response
    corrected_error = error + (goal-current) @ response
    assert np.linalg.norm(corrected_error) < np.linalg.norm(uncorrected_error)
    assert proposed["grasp_certified"] is False


def test_pixel_alignment_includes_uncertainty_and_never_certifies_grasp():
    obs, points = sample()
    ref = reference()
    for view in ("external", "wrist"):
        tool, target = (np.asarray(points[view + "_" + role]) for role in ("tool", "target"))
        ref[view]["desired_offset_uv"] = ((tool-target)/(SIZE-1)).tolist()
    tracker = PregraspVisualFeedback()
    initial = tracker.configure(obs, ref, gripper_open=True)
    assert initial["image_alignment_observed"]
    assert initial["grasp_certified"] is initial["task_success_certified"] is False
    for track in tracker._tracks.values():
        track.uncertainty_px = 2.
    uncertain = tracker._refresh("synthetic_uncertainty_check")
    assert not uncertain["image_alignment_observed"]
    assert uncertain["pixel_error_norm"] < .1


def test_wrist_tool_material_support_must_remain_fixed_in_hand_camera():
    tracker, _, _, _ = trained()
    changed, _ = sample(TRAJECTORY[-1], wrist_tool_shift=[3., 0.])
    diagnostic = tracker.observe(changed)
    assert diagnostic["views"]["wrist"]["status"] == "wrist_tool_moved_in_fixed_hand_frame"
    assert diagnostic["active_views"] == ["external"]
    assert diagnostic["response_validated"]


def test_actual_depth_motion_with_no_image_response_stays_unobservable(monkeypatch):
    monkeypatch.setitem(globals(), "JWRIST", np.zeros((2, 3)))
    tracker, obs, _, diagnostic = trained([(0., 0., i*.002) for i in range(7)])
    assert diagnostic["tracking_valid"]
    assert not diagnostic["response_validated"]
    assert diagnostic["observed_rank"] == 0 and diagnostic["depth_support"] == 0.
    assert tracker.corrected_goal(obs, obs.state[:3])[0] is None


def test_independent_nullable_views_and_fixed_mode_schema_contract():
    ref = reference()
    ref["external"] = None
    ref["wrist"]["tool_reference_mode"] = "fixed_wrist"
    assert validate_pregrasp_reference(ref) == ref
    legacy = reference()
    for view in ("external", "wrist"):
        legacy[view].pop("tool_reference_mode")
    assert validate_pregrasp_reference(legacy) == legacy
    with pytest.raises(ValueError):
        validate_pregrasp_reference({**ref, "wrist": None})
    illegal = reference()
    illegal["external"]["tool_reference_mode"] = "fixed_wrist"
    with pytest.raises(ValueError):
        validate_pregrasp_reference(illegal)
    schema = pregrasp_reference_schema()["anyOf"][1]["properties"]
    assert schema["wrist"]["anyOf"][0] == {"type": "null"}
    assert "coarse_field with phase=align" in pregrasp_instructions()


def test_losing_external_view_preserves_only_independently_validated_wrist_rows():
    tracker, obs, _, _ = trained()
    before = tracker._models["wrist"]
    count = len(before.samples)
    lost, _ = sample(TRAJECTORY[-1], blank="external_tool")
    diagnostic = tracker.observe(lost)
    assert diagnostic["active_views"] == diagnostic["validated_views"] == ["wrist"]
    assert tracker._models["wrist"] is before and len(before.samples) == count
    assert "external" not in tracker._models
    assert diagnostic["observed_rank"] == 2
    current = np.asarray(lost.state[:3])
    nominal = current + [.002, -.001, .001]
    goal, proposal = tracker.corrected_goal(lost, nominal)
    assert goal is not None and proposal["correction_views"] == ["wrist"]
    null = np.array([1., -1., 0.])/np.sqrt(2.)
    assert float((goal-nominal)@null) == pytest.approx(0., abs=2e-5)
    assert np.linalg.norm(goal-nominal) <= .006+1e-12
    # Returning pixels cannot silently resurrect a dropped response model.
    restored, points = sample(TRAJECTORY[-1])
    still = tracker.configure(restored, reference(points), gripper_open=True)
    assert still["active_views"] == ["wrist"]


def test_fixed_wrist_reference_requires_visible_seed_but_survives_short_occlusion():
    ref = reference()
    ref["external"] = None
    ref["wrist"]["tool_reference_mode"] = "fixed_wrist"
    hidden, _ = sample(blank="wrist_tool")
    rejected = PregraspVisualFeedback().configure(hidden, ref, gripper_open=True)
    assert not rejected["tracking_valid"]
    assert rejected["status"] == "fixed_wrist_requires_visible_initial_tool_point"
    tracker = PregraspVisualFeedback()
    visible, _ = sample()
    assert tracker.configure(visible, ref, gripper_open=True)["tracking_valid"]
    for index in range(1, 13):
        hidden, _ = sample((0., index*.00001, 0.), blank="wrist_tool")
        diag = tracker.observe(hidden)
        assert diag["active_views"] == ["wrist"]
        assert diag["point_sources"]["wrist_tool"]["unchecked_observations"] == index
        assert diag["point_uncertainty_px"]["wrist_tool"] == pytest.approx(.75+.15*index)
        # The same observation/corrected_goal/configure does not age twice.
        tracker.corrected_goal(hidden, hidden.state[:3])
        assert tracker.configure(hidden, ref, gripper_open=True)["point_sources"]["wrist_tool"]["unchecked_observations"] == index
    expired, _ = sample((0., .00013, 0.), blank="wrist_tool")
    result = tracker.observe(expired)
    assert not result["tracking_valid"]
    assert result["status"] == "fixed_wrist_reference_expired_without_texture_check"


def test_fixed_wrist_visible_texture_reacquisition_resets_finite_retention():
    ref = reference()
    ref["external"] = None
    ref["wrist"]["tool_reference_mode"] = "fixed_wrist"
    tracker = PregraspVisualFeedback()
    initial, _ = sample()
    tracker.configure(initial, ref, gripper_open=True)
    hidden, _ = sample((.00001, 0., 0.), blank="wrist_tool")
    tracker.observe(hidden)
    seen, _ = sample((.00002, 0., 0.))
    diag = tracker.observe(seen)
    assert diag["point_sources"]["wrist_tool"]["unchecked_observations"] == 0
    assert diag["point_sources"]["wrist_tool"]["source"] == "visible_seed_retained_in_fixed_wrist_frame"


def test_external_target_push_invalidates_wrist_even_after_external_tool_loss():
    tracker, _, _, _ = trained()
    lost, _ = sample(TRAJECTORY[-1], blank="external_tool")
    assert tracker.observe(lost)["active_views"] == ["wrist"]
    pushed, _ = sample(TRAJECTORY[-1], blank="external_tool", external_target_shift=(3., 0.))
    result = tracker.observe(pushed)
    assert result["status"] == "fixed_camera_target_moved_no_response_learning"
    assert result["active_views"] == [] and not result["response_validated"]
    assert tracker.corrected_goal(pushed, pushed.state[:3])[0] is None


def test_auxiliary_support_is_explicit_local_inference_with_growing_uncertainty():
    obs, _ = sample()
    ref = reference()
    ref["wrist"] = None
    # Contact (60,66) remains visible; support is a different nearby part of same body.
    ref["external"]["tool_roi_uv"] = (np.array([44., 50., 54., 82.])/(SIZE-1)).tolist()
    tracker = PregraspVisualFeedback()
    diag = tracker.configure(obs, ref, gripper_open=True)
    assert diag["tracking_valid"], diag
    assert diag["point_sources"]["external_tool"]["local_inference"]
    assert diag["point_sources"]["external_tool"]["source"] == "local_same_body_support_inference"
    start = diag["point_uncertainty_px"]["external_tool"]
    moved, _ = sample((.004, 0., 0.))
    after = tracker.observe(moved)
    assert after["point_uncertainty_px"]["external_tool"] > start
    far = reference()
    far["wrist"] = None
    far["external"]["tool_roi_uv"] = (np.array([15., 40., 29., 80.])/(SIZE-1)).tolist()
    with pytest.raises(ValueError):
        PregraspVisualFeedback().configure(obs, far, gripper_open=True)


def test_small_eleven_pixel_body_roi_has_unique_material_support_without_background():
    obs, _ = sample()
    ref = reference()
    ref["wrist"] = None
    center = np.asarray(CENTERS["external_tool"])
    ref["external"]["tool_roi_uv"] = (np.r_[center-5.5, center+5.5]/(SIZE-1)).tolist()
    tracker = PregraspVisualFeedback()
    diag = tracker.configure(obs, ref, gripper_open=True)
    assert diag["tracking_valid"] and tracker._tracks["external_tool"].radius == 2
    moved, _ = sample((.002, 0., 0.))
    assert tracker.observe(moved)["tracking_valid"]
    for patch in tracker._tracks["external_tool"].patches:
        assert np.all(patch["seed"]-2 >= center-5.5)
        assert np.all(patch["seed"]+2 <= center+5.5)


def test_independent_unexcited_directions_are_unknown_not_zero_cross_responses():
    from gpt_field.speed_first_pregrasp import _ViewResponse
    tracker = PregraspVisualFeedback()
    obs, _ = sample()
    tracker.configure(obs, reference(), gripper_open=True)
    for view, delta, response in (("external", np.array([.002, 0., 0.]), np.array([2., 0.])),
                                  ("wrist", np.array([0., .002, 0.]), np.array([0., 2.]))):
        model = _ViewResponse()
        for _ in range(7):
            model.append(delta, response)
        assert model.validated()
        tracker._models[view] = model
    assert tracker._combined(["external", "wrist"]) == (None, None)
    diag = tracker._refresh("independent_partial_support")
    assert len(diag["correction_views"]) == 1
    assert diag["observed_rank"] == 1
    assert diag["depth_support"] == 0.


def test_affine_support_transport_handles_modest_scale_without_constant_offset(monkeypatch):
    from gpt_field.speed_first_pregrasp import _FrozenSupport
    from gpt_field.observation import decode_image
    obs, _ = sample()
    image = np.asarray(decode_image(obs.image).convert("L"), dtype=np.uint8).copy()
    point = np.array([79., 66.])  # just outside support, still on visible material boundary
    # Seed contact visibility is explicitly supplied in this synthetic frame.
    image[63:70, 76:83] = np.random.default_rng(213).integers(30, 230, (7, 7), dtype=np.uint8)
    track = _FrozenSupport(image, point/(SIZE-1), np.array([44., 50., 75., 82.])/(SIZE-1))
    linear = np.array([[1.05, .015], [-.01, .98]])
    center = np.array([60., 66.])
    shift = np.array([2., -1.])
    monkeypatch.setattr(track, "_match", lambda image, patch: (patch["seed"]-center)@linear+center+shift)
    actual = track.update(image)
    expected = (point-center)@linear+center+shift
    assert actual == pytest.approx(expected, abs=1e-6)
    assert not np.allclose(actual, point+track.translation)
    assert track.inferred and track.uncertainty_px > .5
    # Larger scale is outside the local rigid-perspective approximation.
    monkeypatch.setattr(track, "_match", lambda image, patch: (patch["seed"]-center)*1.3+center)
    with pytest.raises(ValueError, match="inconsistent_or_deformed"):
        track.update(image)


def test_affine_transport_rejects_nonrigid_inconsistent_supports(monkeypatch):
    from gpt_field.speed_first_pregrasp import _FrozenSupport
    from gpt_field.observation import decode_image
    obs, _ = sample()
    image = np.asarray(decode_image(obs.image).convert("L"), dtype=np.uint8)
    track = _FrozenSupport(image, np.array([60., 66.])/(SIZE-1), np.array([44., 50., 76., 82.])/(SIZE-1))
    bad_seed = track.patches[0]["seed"].copy()
    monkeypatch.setattr(track, "_match", lambda image, patch: patch["seed"]+([8., -5.] if np.array_equal(patch["seed"], bad_seed) else [1., 0.]))
    with pytest.raises(ValueError):
        track.update(image)


def test_rendered_small_scale_change_uses_rgb_supports_without_expanding_roi():
    from PIL import Image
    from gpt_field.speed_first_pregrasp import _FrozenSupport
    from gpt_field.observation import decode_image
    obs, _ = sample()
    image = np.asarray(decode_image(obs.image).convert("L"), dtype=np.uint8)
    point = np.array([60., 66.])
    track = _FrozenSupport(image, point/(SIZE-1), np.array([44., 50., 76., 82.])/(SIZE-1))
    linear = np.array([[1.04, .012], [-.008, 1.02]])
    shift = np.array([2., -1.])
    inverse = np.linalg.inv(linear)
    offset = point-inverse@(point+shift)
    mapping = tuple(np.c_[inverse, offset].reshape(-1))
    warped = np.asarray(Image.fromarray(image).transform((SIZE, SIZE), Image.Transform.AFFINE,
        mapping, resample=Image.Resampling.BILINEAR), dtype=np.uint8)
    actual = track.update(warped)
    assert actual == pytest.approx(point+shift, abs=.8)
    for patch in track.patches:
        assert np.all(patch["seed"]-track.radius >= track.roi[0])
        assert np.all(patch["seed"]+track.radius <= track.roi[1])


def test_same_id_one_view_coordinate_jitter_only_drops_that_view():
    tracker, obs, points, before = trained()
    changed = reference(points)
    changed["external"]["tool_contact_uv"][0] += 5./(SIZE-1)
    result = tracker.configure(obs, changed, gripper_open=True)
    assert result["active_views"] == ["wrist"]
    assert result["views"]["external"]["status"] == "same_id_semantic_point_changed"
    assert result["views"]["wrist"]["sample_count"] == before["views"]["wrist"]["sample_count"]
    assert result["response_validated"]
