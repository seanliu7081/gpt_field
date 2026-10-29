"""Measured-motion and RGB-only coverage for the opt-in local correction."""

import copy

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.speed_first_visual import (
    SpeedFirstVisualFeedback, validate_visual_reference, visual_reference_schema,
)


SIZE = 512
MOVING = np.array([180., 220.])
TARGET = np.array([230., 210.])
OFFSET = np.array([-12., 12.])
J = np.array([[1200., 400.], [-400., 800.]])
PATCHES = {name: np.random.default_rng(seed).integers(0, 256, (15, 15, 3), dtype=np.uint8)
           for name, seed in (("moving", 17), ("target", 43))}
PATH = [(.005, 0.), (.005, .005), (.010, .005), (.010, .010)]


def reference(xy=(0., 0.), *, identifier="object-target-1", desired_offset=OFFSET):
    return {"reference_id": identifier,
            "moving_point_uv": (np.rint(MOVING + J @ xy)/(SIZE-1)).tolist(),
            "target_point_uv": (TARGET/(SIZE-1)).tolist(),
            "desired_offset_uv": (np.asarray(desired_offset)/(SIZE-1)).tolist(),
            "planar_alignment": True, "rigid_grasp": True}


def observation(xy=(0., 0.), *, z=1., rotation=(0., 0., 0.), gripper=(.005, -.005),
                moving_delta=(0., 0.), target_delta=(0., 0.), missing=None,
                wrist_shift=0, textureless=None):
    pixels = np.full((SIZE, SIZE, 3), 25, dtype=np.uint8)
    points = {"moving": np.rint(MOVING + J @ xy + moving_delta),
              "target": np.rint(TARGET + target_delta)}
    for name, point in points.items():
        if name == missing:
            continue
        x, y = (int(v) for v in point)
        patch = PATCHES[name] if textureless != name else np.full((15, 15, 3), 80, dtype=np.uint8)
        pixels[y-7:y+8, x-7:x+8] = patch
    image = image_to_data_url(pixels, size=SIZE, rotate=False)
    wrist = image_to_data_url(np.roll(pixels, wrist_shift, axis=1), size=SIZE, rotate=False)
    state = (.1+xy[0], .2+xy[1], z, *rotation, *gripper)
    return PolicyObservation(image, wrist, state, "place the held object at its target")


def calibrated():
    feedback = SpeedFirstVisualFeedback()
    initial = feedback.configure(observation(), reference(), gripper_closed=True)
    assert initial["reliable"] and not initial["calibration_valid"]
    for xy in PATH:
        diagnostic = feedback.observe(observation(xy))
    assert diagnostic["calibration_valid"]
    return feedback, diagnostic


def test_measured_rgb_jacobian_and_desired_feature_offset_produce_actual_error_reduction():
    feedback, diagnostic = calibrated()
    assert np.asarray(diagnostic["jacobian_px_per_m"]) == pytest.approx(J)
    assert diagnostic["pixel_error"] == pytest.approx(TARGET + OFFSET - MOVING - J @ PATH[-1])
    current = observation(PATH[-1])
    goal, proposal = feedback.corrected_goal(current, current.state[:3])
    assert proposal["proposal_available"] and not proposal["task_success_certified"]
    assert goal[2] == current.state[2]
    delta = goal[:2] - current.state[:2]
    assert np.linalg.norm(delta) <= .01 + 1e-12
    # This is closed-loop measured feedback, not a test that merely re-solves J.
    # Render the moving feature at the newly measured world position and check
    # the next independent RGB tracking observation reduces task image error.
    next_xy = goal[:2] - [.1, .2]
    updated = feedback.observe(observation(next_xy))
    assert updated["calibration_valid"]
    assert updated["pixel_error_norm"] < .65 * diagnostic["pixel_error_norm"]
    assert updated["image_error_reduction_px"] > 0.
    assert updated["sample_count"] == 5


def test_fit_persists_across_model_calls_with_same_current_visible_features():
    feedback, diagnostic = calibrated()
    repeated = feedback.configure(observation(PATH[-1]), reference(PATH[-1]), gripper_closed=True)
    assert repeated["mapping_preserved_across_model_call"]
    assert repeated["epoch"] == diagnostic["epoch"]
    assert repeated["sample_count"] == 4
    assert repeated["calibration_valid"]
    assert repeated["goal_correction_xy_m"] == diagnostic["goal_correction_xy_m"]
    # A second model configuration at precisely the same observation adds no
    # fictitious sample or elapsed robot motion.
    assert feedback.configure(observation(PATH[-1]), reference(PATH[-1]),
                              gripper_closed=True)["sample_count"] == 4


def test_subpixel_offset_rounding_cannot_accumulate_objective_drift_across_calls():
    feedback, diagnostic = calibrated()
    for _ in range(6):
        ref = reference(PATH[-1], desired_offset=OFFSET + [.4, 0.])
        updated = feedback.configure(observation(PATH[-1]), ref, gripper_closed=True)
        assert updated["mapping_preserved_across_model_call"]
        assert updated["desired_offset_px"] == pytest.approx(OFFSET)
        assert updated["goal_correction_xy_m"] == pytest.approx(diagnostic["goal_correction_xy_m"])


@pytest.mark.parametrize("extra,reason", [
    ({"z": 1.004}, "height_changed"),
    ({"rotation": (0., 0., .04)}, "orientation_changed"),
    ({"gripper": (.008, -.008)}, "gripper_changed"),
    ({"missing": "moving"}, "moving_match_lost"),
    ({"missing": "target"}, "target_match_lost"),
    ({"moving_delta": (4., 0.)}, "point_moved_without_robot_motion"),
    ({"target_delta": (4., 0.)}, "stationary_target_moved"),
])
def test_changed_assumptions_or_lost_features_fall_back_and_require_new_epoch(extra, reason):
    feedback, _ = calibrated()
    result = feedback.observe(observation(PATH[-1], **extra))
    assert result["status"] == reason
    assert result["goal_correction_xy_m"] is None
    proposal, result = feedback.corrected_goal(observation(PATH[-1]), [.11, .21, 1.])
    assert proposal is None
    assert result["fallback"] == "continue_model_selected_field_or_direct"
    assert not result["probe_requested"]
    result = feedback.configure(observation(PATH[-1]), reference(PATH[-1]), gripper_closed=True)
    assert result["new_reference_id_required"] and not result["calibration_valid"]
    fresh = feedback.configure(observation(PATH[-1]), reference(PATH[-1], identifier="replacement"),
                               gripper_closed=True)
    assert fresh["epoch"] == 2 and fresh["reliable"] and not fresh["calibration_valid"]


def test_new_reference_id_does_not_reuse_previous_mapping():
    feedback, _ = calibrated()
    result = feedback.configure(observation(PATH[-1]), reference(PATH[-1], identifier="second-placement"),
                                gripper_closed=True)
    assert result["epoch"] == 2
    assert result["sample_count"] == 0
    assert not result["calibration_valid"]


@pytest.mark.parametrize("change,reason", [
    ({"moving_point_uv": [float((MOVING[0]+8)/(SIZE-1)), float(MOVING[1]/(SIZE-1))]},
     "reference_id_feature_changed"),
    ({"desired_offset_uv": [0., 0.]}, "reference_id_desired_offset_changed"),
])
def test_same_identifier_cannot_silently_change_feature_identity_or_objective(change, reason):
    feedback, _ = calibrated()
    ref = reference(PATH[-1])
    ref.update(change)
    result = feedback.configure(observation(PATH[-1]), ref, gripper_closed=True)
    assert result["status"] == reason and not result["calibration_valid"]


def test_single_axis_motion_never_manufactures_2d_feedback_or_requests_probes():
    feedback = SpeedFirstVisualFeedback()
    feedback.configure(observation(), reference(), gripper_closed=True)
    for n in range(1, 6):
        obs = observation((.005*n, 0.))
        goal, result = feedback.corrected_goal(obs, obs.state[:3])
        assert goal is None and not result["calibration_valid"]
        assert not result["probe_requested"]
        assert result["fallback"] == "continue_model_selected_field_or_direct"
    assert result["status"] == "insufficient_planar_excitation"


def test_no_reference_or_no_closed_gripper_cannot_start_local_correction():
    feedback = SpeedFirstVisualFeedback()
    assert feedback.configure(observation(), None, gripper_closed=True)["status"] == "visual_reference_not_requested"
    assert feedback.configure(observation(), reference(), gripper_closed=False)["status"] == "closed_gripper_hypothesis_missing"
    assert feedback.corrected_goal(observation(), observation().state[:3])[0] is None


def test_textureless_anchor_falls_back_without_an_action_or_probe():
    feedback = SpeedFirstVisualFeedback()
    obs = observation(textureless="moving")
    result = feedback.configure(obs, reference(), gripper_closed=True)
    assert result["status"] == "moving_textureless_anchor"
    assert not result["probe_requested"]
    assert feedback.corrected_goal(obs, obs.state[:3])[0] is None


def test_nonplanar_nominal_goal_is_not_silently_overridden():
    feedback, _ = calibrated()
    obs = observation(PATH[-1])
    goal, result = feedback.corrected_goal(obs, [.11, .21, 1.01])
    assert goal is None
    assert result["status"] == "nominal_goal_changes_working_height"
    assert not result["proposal_available"]
    assert feedback.diagnostics()["calibration_valid"]


def test_wrist_motion_is_not_treated_as_fixed_camera_object_motion():
    first, _ = calibrated()
    second, _ = calibrated()
    a = first.observe(observation((.015, .010)))
    b = second.observe(observation((.015, .010), wrist_shift=60))
    assert a == b


def test_explicit_release_invalidation_erases_mapping_without_certifying_grasp():
    feedback, _ = calibrated()
    result = feedback.invalidate("model_selected_open")
    assert result["status"] == "model_selected_open"
    assert not result["task_success_certified"] and not result["calibration_valid"]
    assert feedback.corrected_goal(observation(PATH[-1]), [.11, .21, 1.])[0] is None


@pytest.mark.parametrize("key,value", [
    ("moving_point_uv", [True, .2]), ("target_point_uv", [float("nan"), .2]),
    ("desired_offset_uv", [.5, 0.]), ("reference_id", ""),
    ("reference_id", "x"*65), ("rigid_grasp", False), ("planar_alignment", 1),
    ("simulator_pose", [0., 0., 0.]),
])
def test_reference_schema_rejects_malformed_or_extra_data(key, value):
    ref = reference()
    ref[key] = value
    with pytest.raises(ValueError):
        validate_visual_reference(ref)


def test_schema_and_validator_return_independent_objects():
    ref = reference()
    validated = validate_visual_reference(ref)
    validated["moving_point_uv"][0] = 0.
    assert ref["moving_point_uv"][0] != 0.
    schema = visual_reference_schema()
    assert schema["anyOf"][0] == {"type": "null"}
    assert set(schema["anyOf"][1]["required"]) == set(ref)
    assert not schema["anyOf"][1]["additionalProperties"]
    assert validate_visual_reference(None) is None


def test_returned_diagnostics_cannot_mutate_internal_mapping():
    feedback, diagnostic = calibrated()
    changed = copy.deepcopy(diagnostic)
    changed["jacobian_px_per_m"][0][0] = 999999.
    result = feedback.diagnostics()
    result["goal_correction_xy_m"][0] = 999.
    assert feedback.diagnostics() == diagnostic
