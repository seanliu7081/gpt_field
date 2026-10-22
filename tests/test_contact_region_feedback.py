"""Behavioral RGB/state tests; no model, simulator or evaluator inputs."""
import copy

import numpy as np
import pytest

from gpt_field.contact_region_feedback import (
    CONTACT_REGION_VERSION, ContactRegionFeedback, _FrozenResponse,
    contact_region_reference_schema, validate_contact_region_reference,
)
from gpt_field.observation import PolicyObservation, image_to_data_url

SIZE = 192
BASE = np.array([.3, -.1, .8, 0., 0., 0., .04, -.04])
CENTERS = {"external_tool": np.array([60., 66.]), "external_target": np.array([104., 88.]),
           "wrist_tool": np.array([60., 66.]), "wrist_target": np.array([104., 94.])}
TEXTURES = {key: np.random.default_rng(i + 51).integers(30, 225, (33, 33), dtype=np.uint8)
            for i, key in enumerate(CENTERS)}
JEXT = np.array([[1000., 0., 0.], [0., 1000., 0.]])
JWRIST = np.array([[500., 0., 1000.], [0., 1000., 0.]])
TRAJECTORY = [(0., 0., 0.), (.002, 0., 0.), (.002, .002, 0.), (.004, .002, 0.),
              (.004, .004, 0.), (.006, .004, 0.), (.006, .006, 0.)]


def observation(delta=(0., 0., 0.), *, blank=(), noise=0, tool_extra=(0., 0.), target_shift=(0., 0.),
                wrist_extra=(0., 0.), rotation=0., fingers=(.04, -.04)):
    delta = np.asarray(delta, float)
    state = BASE.copy()
    state[:3] += delta
    state[3] = rotation
    state[6:] = fingers
    points = {key: value.copy() for key, value in CENTERS.items()}
    points["external_tool"] += JEXT @ delta + np.asarray(tool_extra)
    points["external_target"] += target_shift
    points["wrist_target"] -= JWRIST @ delta + np.asarray(wrist_extra)
    images = {}
    for index, view in enumerate(("external", "wrist")):
        frame = np.full((SIZE, SIZE), 12, dtype=np.uint8)
        for role in ("tool", "target"):
            key = view + "_" + role
            if key in blank:
                continue
            x, y = np.rint(points[key]).astype(int)
            frame[y - 16:y + 17, x - 16:x + 17] = TEXTURES[key]
        if noise:
            rng = np.random.default_rng(int(abs(delta.sum()) * 1e6) + index)
            frame = np.clip(frame.astype(float) + rng.normal(0., noise, frame.shape), 0, 255).astype(np.uint8)
        images[view] = image_to_data_url(np.repeat(frame[:, :, None], 3, axis=2), SIZE, rotate=False)
    return PolicyObservation(images["external"], images["wrist"], tuple(state), "Manipulate selected target")


def reference(*, external=True, wrist=True, identifier="epoch-1"):
    ref = {"reference_id": identifier, "target_id": "selected_target"}
    for view, enabled in (("external", external), ("wrist", wrist)):
        if not enabled:
            ref[view] = None
            continue
        ref[view] = {role + "_region_uv": (np.r_[CENTERS[view + "_" + role] - 16,
                                                                  CENTERS[view + "_" + role] + 16] / (SIZE - 1)).tolist()
                     for role in ("tool", "target")}
        ref[view]["desired_offset_uv"] = (np.array([-40., -20.]) / (SIZE - 1)).tolist()
    return ref


def trained(*, external=True, wrist=True, trajectory=TRAJECTORY, noise=0):
    monitor = ContactRegionFeedback()
    first = observation(trajectory[0], noise=noise)
    diag = monitor.configure(first, reference(external=external, wrist=wrist), gripper_open=True,
                             target_id="selected_target")
    assert diag["tracking_valid"], diag
    for delta in trajectory[1:]:
        last = observation(delta, noise=noise)
        diag = monitor.observe(last)
    return monitor, last, diag


def test_schema_anchor_contract_is_smaller_and_no_semantic_contact_claims():
    ref = reference(wrist=False)
    assert validate_contact_region_reference(ref) == ref
    assert validate_contact_region_reference(None) is None
    schema = contact_region_reference_schema()
    assert schema["anyOf"][0] == {"type": "null"}
    properties = schema["anyOf"][1]["properties"]["external"]["anyOf"][1]["properties"]
    assert set(properties) == {"tool_region_uv", "target_region_uv", "desired_offset_uv"}
    tracker = ContactRegionFeedback()
    diag = tracker.configure(observation(), ref, gripper_open=True)
    assert diag["version"] == CONTACT_REGION_VERSION
    assert diag["anchor_semantics"] == "transported_model_region_centers_not_physical_contact_points"
    assert "tool_region_anchor_minus_target_region_anchor" == diag["desired_offset_semantics"]
    assert not diag["grasp_certified"] and not diag["task_success_certified"]


@pytest.mark.parametrize("bad", ["extra", "missing", "nan", "bool", "complex", "order", "offset", "both_null", "id"])
def test_bad_references_rejected(bad):
    ref = reference()
    if bad == "extra":
        ref["depth"] = 1.
    elif bad == "missing":
        del ref["target_id"]
    elif bad == "nan":
        ref["wrist"]["target_region_uv"][0] = float("nan")
    elif bad == "bool":
        ref["external"]["tool_region_uv"][0] = True
    elif bad == "complex":
        ref["external"]["tool_region_uv"][0] = 1j
    elif bad == "order":
        ref["external"]["tool_region_uv"] = [.5, .5, .4, .8]
    elif bad == "offset":
        ref["external"]["desired_offset_uv"] = [-2., 0.]
    elif bad == "both_null":
        ref["external"] = ref["wrist"] = None
    else:
        ref["reference_id"] = " "
    with pytest.raises(ValueError):
        validate_contact_region_reference(ref)


def test_real_texture_motion_learns_frozen_fit_then_validates_later_samples():
    tracker = ContactRegionFeedback()
    tracker.configure(observation(), reference(wrist=False), gripper_open=True)
    for index, delta in enumerate(TRAJECTORY[1:], 1):
        diag = tracker.observe(observation(delta))
        if index < 5:
            assert not diag["response_validated"]
    assert diag["response_validated"], diag
    model = diag["views"]["external"]
    assert model["validated_rank"] == 2
    assert set(model["fit_training_sample_ids"]).isdisjoint(model["held_out_sample_ids"])
    assert model["fit_training_sample_ids"] == [1, 2, 3]
    assert model["held_out_sample_ids"] == [4, 5, 6]
    assert np.asarray(model["response_px_per_m"]).T == pytest.approx(JEXT, abs=4.)
    assert model["unknown_input_directions_are_zero"] is False
    assert not diag["probe_requested"]


def test_frozen_coefficients_do_not_refit_using_validation_and_require_later_direction_coverage():
    model = _FrozenResponse()
    for delta in ([.002, 0., 0.], [0., .002, 0.], [.002, 0., 0.]):
        delta = np.asarray(delta)
        model.append(delta, delta @ JEXT.T)
    matrix = model.matrix.copy()
    for _ in range(2):
        delta = np.array([.002, 0., 0.])
        model.append(delta, delta @ JEXT.T + [.05, .02])
    assert model.matrix == pytest.approx(matrix, abs=0.)
    basis = model.validated_basis()
    assert basis.shape == (3, 1)
    assert basis @ basis.T == pytest.approx(np.diag([1., 0., 0.]), abs=1e-6)
    assert set(model.fit_training_ids).isdisjoint(r["sample_id"] for r in model.validation)


def test_unknown_new_axis_requires_new_frozen_fit_and_discards_old_validation():
    model = _FrozenResponse()
    for _ in range(5):
        model.append(np.array([.002, 0., 0.]), np.array([2., 0.]))
    assert model.validated_basis().shape[1] == 1
    version = model.fit_version
    model.append(np.array([0., .002, 0.]), np.array([0., 2.]))
    assert model.fit_version == version + 1
    assert model.validated_basis().shape[1] == 0
    assert not model.validation
    assert model.fit_training_ids == [1, 2, 3, 6]


def test_correction_solves_residual_at_nominal_not_current_error_twice():
    tracker, obs, diag = trained(external=True, wrist=False)
    nominal = np.array(obs.state[:3]) + [.002, -.001, 0.]
    current_error = np.asarray(diag["views"]["external"]["image_error_px"])
    predicted_nominal = current_error + JEXT @ (nominal - obs.state[:3])
    goal, result = tracker.corrected_goal(obs, nominal)
    assert goal is not None, result
    assert result["predicted_nominal_error_norm"] == pytest.approx(np.linalg.norm(predicted_nominal), abs=.05)
    actual_projected = current_error + JEXT @ (goal - obs.state[:3])
    assert np.linalg.norm(actual_projected) < np.linalg.norm(predicted_nominal)
    assert result["predicted_corrected_error_norm"] == pytest.approx(np.linalg.norm(actual_projected), abs=.05)
    assert np.linalg.norm(goal - nominal) <= .006 + 1e-12
    assert goal[2] == pytest.approx(nominal[2], abs=1e-10)


def test_repeated_proposals_do_not_accumulate_and_nominal_argument_is_immutable():
    tracker, obs, _ = trained(wrist=False)
    nominal = np.array(obs.state[:3])
    saved = nominal.copy()
    first, diag = tracker.corrected_goal(obs, nominal)
    assert first is not None, diag
    for _ in range(5):
        again, _ = tracker.corrected_goal(obs, nominal)
        assert again == pytest.approx(first)
        assert np.linalg.norm(again - saved) <= .006 + 1e-12
    assert nominal == pytest.approx(saved)
    assert tracker.corrected_goal(obs, first)[0] is None
    assert tracker.snapshot()["status"] == "nominal_changed_requires_new_intent"


def test_single_observed_axis_does_not_claim_or_correct_depth():
    path = [(i * .002, 0., 0.) for i in range(7)]
    tracker, obs, diag = trained(wrist=False, trajectory=path)
    assert diag["response_validated"]
    assert diag["views"]["external"]["validated_rank"] == 1
    goal, diag = tracker.corrected_goal(obs, obs.state[:3])
    assert goal is not None, diag
    assert goal[1:] == pytest.approx(obs.state[1:3], abs=1e-10)
    other, obs2, _ = trained(wrist=False, trajectory=path)
    goal, diag = other.corrected_goal(obs2, np.asarray(obs2.state[:3]) + [0., 0., .002])
    assert goal is None
    assert diag["status"] == "nominal_motion_has_unknown_image_response"


def test_expected_post_application_motion_is_measured_not_claimed_grasp_improvement():
    tracker, obs, _ = trained(wrist=False)
    nominal = np.asarray(obs.state[:3])
    goal, diag = tracker.corrected_goal(obs, nominal)
    assert goal is not None, diag
    tracker.notify_correction_applied(obs, nominal, goal)
    after = observation((.004, .006, 0.))
    diag = tracker.observe(after)
    check = diag["post_application"]
    assert check["status"] == "prediction_supported_not_grasp_evidence", diag
    assert check["measured_checks"] == 1
    assert check["views"]["external"]["prediction_error_px"] < .1
    assert check["views"]["external"]["observed_error_decreased"]
    assert check["physical_grasp_improved"] == "unknown"
    assert check["causal_correction_benefit"] == "unmeasured"


def test_reversed_actual_response_revokes_applied_goal_and_requests_nominal_restore():
    tracker, obs, _ = trained(wrist=False)
    nominal = np.asarray(obs.state[:3])
    goal, _ = tracker.corrected_goal(obs, nominal)
    tracker.notify_correction_applied(obs, nominal, goal)
    # Robot really moved -2mm X, but observed image moves +2px instead.
    after = observation((.004, .006, 0.), tool_extra=(4., 0.))
    diag = tracker.observe(after)
    assert diag["status"] == "applied_correction_prediction_mismatch", diag
    assert diag["restore_nominal_requested"] and not diag["response_validated"]
    assert diag["nominal_goal_xyz"] == pytest.approx(nominal)
    assert diag["post_application"]["views"]["external"]["prediction_error_px"] > 3.5
    assert tracker.corrected_goal(after, nominal)[0] is None


def test_shadow_proposal_has_no_registered_motion_application_or_restore():
    tracker, obs, _ = trained(wrist=False)
    goal, _ = tracker.corrected_goal(obs, obs.state[:3])
    assert goal is not None
    after = observation((.004, .006, 0.), tool_extra=(4., 0.))
    diag = tracker.observe(after)
    assert diag["application_count"] == 0
    assert diag["post_application"]["status"] == "not_applied"
    assert not diag["restore_nominal_requested"]
    assert not diag["response_validated"]


def test_application_registration_rejects_stale_or_manually_changed_proposals():
    tracker, obs, _ = trained(wrist=False)
    nominal = np.asarray(obs.state[:3])
    goal, _ = tracker.corrected_goal(obs, nominal)
    with pytest.raises(ValueError, match="exact offered proposal"):
        tracker.notify_correction_applied(obs, nominal, goal + [.001, 0., 0.])
    with pytest.raises(ValueError, match="current observation"):
        tracker.notify_correction_applied(observation((.004, .006, 0.)), nominal, goal)


def test_lost_one_view_leaves_independent_other_view_available():
    tracker, _, diag = trained()
    assert diag["response_validated"], diag
    obs = observation(TRAJECTORY[-1], blank=("external_tool",))
    diag = tracker.observe(obs)
    assert diag["active_views"] == ["wrist"], diag
    assert diag["response_validated"]
    assert diag["target_stationarity_observed"]
    diag = tracker.observe(observation(TRAJECTORY[-1]))
    assert diag["active_views"] == ["wrist"]  # no implicit reseeding


def test_wrist_only_uses_relative_motion_without_claiming_stationary_target():
    tracker, obs, diag = trained(external=False)
    assert diag["response_validated"], diag
    assert diag["active_views"] == ["wrist"]
    assert not diag["target_stationarity_observed"]
    goal, diag = tracker.corrected_goal(obs, obs.state[:3])
    assert goal is not None, diag


def test_fixed_view_target_drift_revokes_both_view_models():
    tracker, _, _ = trained()
    diag = tracker.observe(observation(TRAJECTORY[-1], target_shift=(3., 0.)))
    assert diag["status"] == "fixed_camera_target_moved", diag
    assert not diag["tracking_valid"] and not diag["response_validated"]


@pytest.mark.parametrize("kwargs,reason", [({"rotation": .08}, "measured_orientation_changed"),
                                          ({"fingers": (.02, -.02)}, "measured_finger_width_changed")])
def test_changed_geometry_invalidates_and_same_id_cannot_reseed(kwargs, reason):
    tracker, _, _ = trained()
    obs = observation(TRAJECTORY[-1], **kwargs)
    diag = tracker.observe(obs)
    assert diag["status"] == reason
    diag = tracker.configure(obs, reference(), gripper_open=True)
    assert not diag["tracking_valid"] and diag["new_reference_id_required"]


def test_no_motion_image_shift_drops_view_not_mistakenly_fit_as_response():
    tracker, _, _ = trained()
    diag = tracker.observe(observation(TRAJECTORY[-1], tool_extra=(3., 0.)))
    assert diag["active_views"] == ["wrist"], diag
    assert diag["views"]["external"]["status"] == "image_drift_without_robot_motion"


def test_target_identity_and_same_id_geometry_changes_rejected():
    tracker, obs, _ = trained()
    assert not tracker.configure(obs, reference(), gripper_open=True, target_id="other")["tracking_valid"]
    tracker = ContactRegionFeedback()
    tracker.configure(observation(), reference(), gripper_open=True)
    changed = reference()
    changed["external"]["desired_offset_uv"][0] += .01
    diag = tracker.configure(observation(), changed, gripper_open=True)
    assert diag["status"] == "same_id_reference_changed_requires_new_id"


def test_regions_are_persistent_seed_descriptions_across_model_calls():
    tracker, obs, diag = trained()
    same = tracker.configure(obs, reference(), gripper_open=True)
    assert same["response_validated"]
    assert same["sample_count"] == diag["sample_count"]
    tracker.configure(obs, reference(identifier="epoch-2"), gripper_open=True)
    again = tracker.configure(obs, reference(), gripper_open=True)
    assert again["status"] == "retired_reference_id_cannot_restart"


def test_small_image_noise_keeps_response_usable_and_correction_bounded():
    tracker, obs, diag = trained(wrist=False, noise=2)
    assert diag["response_validated"], diag
    goal, diag = tracker.corrected_goal(obs, obs.state[:3])
    assert goal is not None, diag
    assert np.linalg.norm(goal - obs.state[:3]) <= .006 + 1e-12


def test_overlap_textureless_and_missing_backend_disable_without_actions(monkeypatch):
    tracker = ContactRegionFeedback()
    ref = reference(wrist=False)
    ref["external"]["target_region_uv"] = copy.deepcopy(ref["external"]["tool_region_uv"])
    diag = tracker.configure(observation(), ref, gripper_open=True)
    assert not diag["tracking_valid"]
    assert diag["views"]["external"]["status"] == "tool_target_regions_overlap"
    diag = tracker.configure(observation(blank=("external_tool",)), reference(wrist=False, identifier="blank"), gripper_open=True)
    assert not diag["tracking_valid"]
    def missing():
        raise ImportError("missing")
    monkeypatch.setattr("gpt_field.contact_region_feedback._backend", missing)
    diag = tracker.configure(observation(), reference(identifier="backend"), gripper_open=True)
    assert diag["status"] == "optical_flow_backend_unavailable"
    assert not diag["probe_requested"]


def test_nominal_and_measured_local_domain_limits():
    tracker, obs, _ = trained(wrist=False)
    goal, diag = tracker.corrected_goal(obs, BASE[:3] + [.05, 0., 0.])
    assert goal is None and diag["status"] == "nominal_goal_outside_local_domain"
    other, _, _ = trained()
    diag = other.observe(observation((.041, 0., 0.)))
    assert diag["status"] == "measured_translation_left_local_domain"


def test_new_model_goal_preserves_validated_fit_but_has_fresh_immutable_nominal_budget():
    tracker, obs, diag = trained(wrist=False)
    nominal_a = np.asarray(obs.state[:3])
    tracker.begin_nominal_goal(nominal_a, "accepted-response-1")
    goal_a, result = tracker.corrected_goal(obs, nominal_a)
    tracker.notify_correction_applied(obs, nominal_a, goal_a)
    nominal_b = nominal_a + [.002, 0., 0.]
    begun = tracker.begin_nominal_goal(nominal_b, "accepted-response-2")
    assert begun["response_validated"]
    assert begun["sample_count"] == diag["sample_count"]
    assert not begun["restore_nominal_requested"]
    goal_b, result = tracker.corrected_goal(obs, nominal_b)
    assert goal_b is not None, result
    assert np.linalg.norm(goal_b - nominal_b) <= .006 + 1e-12
    assert result["nominal_goal_xyz"] == pytest.approx(nominal_b)
    with pytest.raises(ValueError, match="same nominal intent"):
        tracker.begin_nominal_goal(goal_b, "accepted-response-2")
    with pytest.raises(ValueError, match="retired nominal"):
        tracker.begin_nominal_goal(nominal_a, "accepted-response-1")


def test_trained_but_unvalidated_nominal_axis_is_not_silently_projected_to_zero():
    path = [(0., 0., 0.), (.002, 0., 0.), (.002, .002, 0.), (.004, .002, 0.),
            (.006, .002, 0.), (.008, .002, 0.)]
    tracker, obs, diag = trained(wrist=False, trajectory=path)
    model = diag["views"]["external"]
    assert model["observed_rank"] == 2 and model["validated_rank"] == 1
    # Y was in training but not subsequent validation. Its 2px nominal effect
    # must not disappear simply because the correction itself is X-only.
    goal, diag = tracker.corrected_goal(obs, np.asarray(obs.state[:3]) + [0., .002, 0.])
    assert goal is None
    assert diag["status"] == "nominal_motion_has_unknown_image_response"


def test_submillimeter_replanning_retains_pending_application_prediction_baseline():
    path = [(i * .002, 0., 0.) for i in range(7)]
    tracker, obs, _ = trained(wrist=False, trajectory=path)
    nominal = np.asarray(obs.state[:3])
    for index in range(3):
        goal, diag = tracker.corrected_goal(obs, nominal)
        assert goal is not None, diag
        tracker.notify_correction_applied(obs, nominal, goal)
        obs = observation((.012 - .0004 * (index + 1), 0., 0.))
        diag = tracker.observe(obs)
    assert diag["post_application"]["measured_checks"] == 1, diag
    assert diag["post_application"]["status"] == "prediction_supported_not_grasp_evidence"


def test_restoration_acknowledges_original_nominal_but_does_not_reenable_rejected_fit():
    tracker, obs, _ = trained(wrist=False)
    nominal = np.asarray(obs.state[:3])
    goal, _ = tracker.corrected_goal(obs, nominal)
    tracker.notify_correction_applied(obs, nominal, goal)
    tracker.observe(observation((.004, .006, 0.), tool_extra=(4., 0.)))
    with pytest.raises(ValueError, match="immutable original"):
        tracker.notify_nominal_restored(goal)
    diag = tracker.notify_nominal_restored(nominal)
    assert not diag["restore_nominal_requested"]
    assert not diag["tracking_valid"] and not diag["response_validated"]


def test_occluding_applied_view_restores_goal_even_if_other_view_survives():
    tracker, obs, _ = trained()
    nominal = np.asarray(obs.state[:3])
    goal, diag = tracker.corrected_goal(obs, nominal)
    assert goal is not None, diag
    tracker.notify_correction_applied(obs, nominal, goal)
    diag = tracker.observe(observation(TRAJECTORY[-1], blank=("external_tool",)))
    assert diag["status"] == "applied_correction_view_lost"
    assert diag["restore_nominal_requested"]

    assert diag["active_views"] == ["wrist"]
    assert diag["response_validated"]
    tracker.notify_nominal_restored(nominal)
    goal, diag = tracker.corrected_goal(observation(TRAJECTORY[-1], blank=("external_tool",)), nominal)
    assert goal is not None and diag["correction_views"] == ["wrist"]


def test_accepted_tiny_out_of_plane_motion_does_not_destroy_validation_coverage():
    model = _FrozenResponse()
    for _ in range(3):
        model.append(np.array([.002, 0., 0.]), np.array([2., 0.]))
    for _ in range(2):
        model.append(np.array([.002, 0., .0001]), np.array([2., 0.]))
    basis = model.validated_basis()
    assert basis.shape == (3, 1)
    assert basis @ basis.T == pytest.approx(np.diag([1., 0., 0.]), abs=1e-6)
    assert model.snapshot()["held_out_max_unsupported_input_m"] == pytest.approx(.0001)
    assert model.validation[0]["delta"][2] == pytest.approx(.0001)
    assert model.validation[0]["supported_delta"][2] == pytest.approx(0.)
    # Larger unsupported directions remain new training, not held-out evidence.
    model.append(np.array([.002, 0., .0004]), np.array([2., 0.]))
    assert not model.validation
    assert model.validated_basis().shape[1] == 0


def edge_wrist_observation(delta=(0., 0., 0.), *, edge_shift=0, erase=False, **kwargs):
    from gpt_field.observation import decode_image
    obs = observation(delta, **kwargs)
    image = np.asarray(decode_image(obs.wrist_image)).copy()
    image[50:83, 44:77] = 20
    if not erase:
        image[50:83, 60 + edge_shift:77] = 160
    return PolicyObservation(obs.image, image_to_data_url(image, SIZE, rotate=False), obs.state, obs.prompt)


def test_visible_low_texture_wrist_tool_uses_checked_rigid_anchor_and_target_flow():
    tracker = ContactRegionFeedback()
    diag = tracker.configure(edge_wrist_observation(), reference(external=False), gripper_open=True)
    assert diag["tracking_valid"], diag
    tool = diag["views"]["wrist"]["regions"]["tool"]
    assert tool["measurement"] == "conditional_fixed_wrist_region_anchor_not_measured_optical_flow"
    assert not tool["image_motion_measured"] and not tool["semantic_identity_certified"]
    for delta in TRAJECTORY[1:]:
        obs = edge_wrist_observation(delta)
        diag = tracker.observe(obs)
    assert diag["response_validated"], diag
    assert diag["views"]["wrist"]["regions"]["tool"]["age"] == len(TRAJECTORY) - 1
    assert diag["views"]["wrist"]["regions"]["target"]["measurement"] == "model_region_feature_transport_not_object_pose"
    goal, diag = tracker.corrected_goal(obs, obs.state[:3])
    assert goal is not None, diag
    assert not diag["grasp_certified"]


@pytest.mark.parametrize("mutation", [{"edge_shift": 3}, {"erase": True}])
def test_shifted_or_occluded_fixed_wrist_edge_is_rejected_without_reseeding(mutation):
    tracker = ContactRegionFeedback()
    tracker.configure(edge_wrist_observation(), reference(external=False), gripper_open=True)
    diag = tracker.observe(edge_wrist_observation((.002, 0., 0.), **mutation))
    assert not diag["tracking_valid"]
    assert diag["views"] == {}  # invalidated epoch retains no authorization
    assert not tracker.configure(edge_wrist_observation(), reference(external=False), gripper_open=True)["tracking_valid"]


@pytest.mark.parametrize("kwargs,reason", [({"rotation": .08}, "measured_orientation_changed"),
                                          ({"fingers": (.02, -.02)}, "measured_finger_width_changed")])
def test_fixed_wrist_reference_does_not_outlive_robot_geometry_assumptions(kwargs, reason):
    tracker = ContactRegionFeedback()
    tracker.configure(edge_wrist_observation(), reference(external=False), gripper_open=True)
    diag = tracker.observe(edge_wrist_observation((.002, 0., 0.), **kwargs))
    assert diag["status"] == reason and not diag["tracking_valid"]


def test_fixed_wrist_blank_invisible_and_uniform_regions_are_not_valid_seeds():
    from gpt_field.contact_region_feedback import _FixedWristRegion, _backend
    cv = _backend()
    for brightness in (0, 20, 180):
        with pytest.raises(ValueError, match="visible_region_edge"):
            _FixedWristRegion(np.full((96, 96), brightness, dtype=np.uint8), [.2, .2, .8, .8], cv)


def test_fixed_wrist_visible_pattern_checked_each_frame_and_same_epoch_expiry_as_flow():
    from gpt_field.contact_region_feedback import _FixedWristRegion, _backend
    before = np.full((96, 96), 20, dtype=np.uint8)
    before[:, 48:] = 160
    fixed = _FixedWristRegion(before, [.2, .2, .8, .8], _backend())
    rng = np.random.default_rng(8)
    for _ in range(fixed.maximum_age):
        after = np.clip(before.astype(float) + rng.normal(0., .5, before.shape), 0, 255).astype(np.uint8)
        fixed.update(before, after)
        assert fixed.valid, fixed.snapshot()
    fixed.update(before, before)
    assert not fixed.valid and fixed.reason == "local_tracking_epoch_expired"


def test_fixed_anchor_not_used_for_external_tool_or_for_wrist_target():
    tracker = ContactRegionFeedback()
    from gpt_field.observation import decode_image
    first = edge_wrist_observation()
    # The same corner-poor finger appearance is not assumed world-fixed in the
    # external camera, and a target is never declared hand-fixed automatically.
    external_ref = reference(wrist=False)
    external_ref["external"] = copy.deepcopy(reference()["wrist"])
    obs = PolicyObservation(first.wrist_image, first.wrist_image, first.state, first.prompt)
    diag = tracker.configure(obs, external_ref, gripper_open=True)
    assert not diag["tracking_valid"]
    image = np.asarray(decode_image(first.wrist_image)).copy()
    image[78:111, 88:121] = 20
    image[78:111, 104:121] = 160
    obs = PolicyObservation(first.image, image_to_data_url(image, SIZE, rotate=False), first.state, first.prompt)
    diag = ContactRegionFeedback().configure(obs, reference(external=False), gripper_open=True)
    assert not diag["tracking_valid"]
