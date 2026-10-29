"""Synthetic, RGB-only tests for the optional local planar visual guidance."""

import copy
import math

import numpy as np
import pytest

from gpt_field.local_visual_feedback import ANCHOR_SEMANTICS, LocalVisualFeedback
from gpt_field.observation import PolicyObservation, image_to_data_url


SIZE = 160
MOVING = np.array([50., 65.])
TARGET = np.array([108., 108.])
JACOBIAN = np.array([[800., 200.], [-200., 600.]])
PATCHES = {name: np.random.default_rng(seed).integers(0, 256, size=(11, 11, 3), dtype=np.uint8)
           for name, seed in (("moving", 17), ("target", 43))}


def reference(*, kind="end_effector", alignment=True):
    result = {"kind": kind, "object_center_offset_m": None,
              "desired_object_center_offset_m": None,
              "visual_anchor": {"moving_point_uv": (MOVING/(SIZE-1)).tolist(),
                                "target_point_uv": (TARGET/(SIZE-1)).tolist()}}
    if alignment:
        result["local_alignment"] = {"semantic": ANCHOR_SEMANTICS, "planar_alignment": True}
    return result


def observation(xy=(0., 0.), *, moving_delta=(0., 0.), target_delta=(0., 0.),
                missing=None, duplicate=None, textureless=None, z=1.,
                rotation=(0., 0., 0.), gripper=(.005, -.005), wrist_shift=0):
    pixels = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    points = {"moving": np.rint(MOVING + JACOBIAN @ np.array(xy) + moving_delta),
              "target": np.rint(TARGET + np.array(target_delta))}
    for name, point in points.items():
        if name == missing:
            continue
        patch = PATCHES[name] if textureless != name else np.full((11, 11, 3), 80, dtype=np.uint8)
        for center in (point, *((point + np.array([18, 0]),) if duplicate == name else ())):
            x, y = (int(v) for v in center)
            left, right = max(0, x-5), min(SIZE, x+6)
            top, bottom = max(0, y-5), min(SIZE, y+6)
            if left < right and top < bottom:
                pixels[top:bottom, left:right] = patch[top-y+5:bottom-y+5, left-x+5:right-x+5]
    image = image_to_data_url(pixels, size=SIZE, rotate=False)
    wrist = image_to_data_url(np.roll(pixels, wrist_shift, axis=1), size=SIZE, rotate=False)
    state = (.1+xy[0], .2+xy[1], z, *rotation, *gripper)
    return PolicyObservation(image, wrist, state, "move the held object to the target")


CALIBRATION_PATH = [(.005, 0.), (.005, .005), (.010, .005), (.010, .010)]


def calibrate(instance=None, *, ref=None):
    check = instance or LocalVisualFeedback()
    configured = check.configure(observation(), reference() if ref is None else ref)
    assert configured["reliable"] and not configured["calibration_valid"]
    for xy in CALIBRATION_PATH:
        result = check.observe(observation(xy))
    return check, result


def test_independent_xy_excitation_learns_measured_image_mapping_and_bounds_correction():
    check, result = calibrate()
    assert result["status"] == "local_guidance_ready"
    assert result["reliable"] and result["calibration_valid"]
    assert np.array(result["jacobian_px_per_m"]) == pytest.approx(JACOBIAN)
    error = TARGET - MOVING - JACOBIAN @ CALIBRATION_PATH[-1]
    raw = np.linalg.solve(JACOBIAN, error)
    expected = raw / np.linalg.norm(raw) * .01
    assert result["goal_correction_xy_m"] == pytest.approx(expected)
    assert result["pixel_error"] == pytest.approx(error)
    assert result["residual_px"] < 1e-10
    assert result["correction_was_bounded"]
    assert result["sample_count"] == 4
    assert result["task_success_certified"] is False
    assert "3-D" in result["limitations"][0]


def test_target_is_tracked_independently_and_new_target_position_changes_correction():
    check, previous = calibrate()
    result = check.observe(observation(CALIBRATION_PATH[-1], target_delta=(-4., 3.)))
    assert result["calibration_valid"]
    assert result["target_point_px"] == pytest.approx(TARGET + [-4., 3.])
    assert result["pixel_error"] == pytest.approx(np.array(previous["pixel_error"]) + [-4., 3.])
    assert result["goal_correction_xy_m"] != previous["goal_correction_xy_m"]
    assert result["sample_count"] == previous["sample_count"]


def test_wrist_camera_motion_does_not_change_external_feedback():
    first, _ = calibrate()
    second, _ = calibrate()
    xy = (.015, .010)
    assert first.observe(observation(xy)) == second.observe(observation(xy, wrist_shift=30))


def test_single_axis_motion_cannot_identify_two_dimensional_correction():
    check = LocalVisualFeedback()
    check.configure(observation(), reference())
    for n in range(1, 7):
        result = check.observe(observation((.005*n, 0.)))
        assert result["goal_correction_xy_m"] is None
        assert result["calibration_valid"] is False
        assert result["reliable"]
    assert result["status"] == "insufficient_planar_excitation"


def test_repeated_identical_observations_do_not_count_as_excitation():
    check = LocalVisualFeedback()
    check.configure(observation(), reference())
    for _ in range(5):
        result = check.observe(observation())
        assert result["sample_count"] == 0
        assert result["status"] == "awaiting_excitation"
        assert not result["calibration_valid"]


@pytest.mark.parametrize("extra,status", [
    ({"z": 1.004}, "height_changed"),
    ({"rotation": (0., 0., .04)}, "orientation_changed"),
    ({"gripper": (.008, -.008)}, "gripper_changed"),
    ({"xy": (.061, 0.)}, "outside_local_calibration_region"),
])
def test_control_assumption_changes_permanently_invalidate_mapping(extra, status):
    check, _ = calibrate(ref=reference(kind="held_object"))
    kwargs = {"xy": CALIBRATION_PATH[-1], **extra}
    result = check.observe(observation(**kwargs))
    assert result["status"] == status
    assert not result["reliable"] and not result["calibration_valid"]
    assert result["goal_correction_xy_m"] is None
    assert check.observe(observation(CALIBRATION_PATH[-1]))["status"] == status


def test_rotation_vector_wrap_does_not_invalidate_same_orientation():
    check, _ = calibrate()
    result = check.observe(observation(CALIBRATION_PATH[-1], rotation=(0., 0., 2*math.pi)))
    assert result["calibration_valid"]


@pytest.mark.parametrize("kwargs,status", [
    ({"textureless": "moving"}, "moving_textureless_anchor"),
    ({"textureless": "target"}, "target_textureless_anchor"),
    ({"duplicate": "moving"}, "moving_ambiguous_match"),
    ({"duplicate": "target"}, "target_ambiguous_match"),
])
def test_uninformative_or_ambiguous_initial_anchor_is_explicitly_unavailable(kwargs, status):
    check = LocalVisualFeedback()
    result = check.configure(observation(**kwargs), reference())
    assert result["status"] == status
    assert result["goal_correction_xy_m"] is None
    assert check.observe(observation())["status"] == status


@pytest.mark.parametrize("name", ["moving", "target"])
def test_occlusion_of_either_patch_invalidates_and_cannot_silently_resume(name):
    check, _ = calibrate()
    result = check.observe(observation(CALIBRATION_PATH[-1], missing=name))
    assert result["status"] == f"{name}_match_lost"
    assert not result["reliable"] and not result["calibration_valid"]
    assert check.observe(observation(CALIBRATION_PATH[-1]))["status"] == result["status"]


def test_out_of_view_patch_cannot_produce_a_correction():
    check, _ = calibrate()
    result = check.observe(observation(CALIBRATION_PATH[-1], moving_delta=(-60., 0.)))
    assert not result["reliable"] and result["goal_correction_xy_m"] is None


def test_stationary_robot_with_moving_rigid_anchor_detects_possible_slip():
    check, _ = calibrate(ref=reference(kind="held_object"))
    result = check.observe(observation(CALIBRATION_PATH[-1], moving_delta=(3., 0.)))
    assert result["status"] == "point_moved_without_robot_motion"
    assert not result["calibration_valid"]
    assert result["goal_correction_xy_m"] is None


def test_post_calibration_response_inconsistent_with_mapping_invalidates():
    check, _ = calibrate()
    result = check.observe(observation((.015, .01), moving_delta=(5., 0.)))
    assert result["status"] == "jacobian_prediction_inconsistent"
    assert result["prediction_error_px"] == pytest.approx(5.)
    assert result["goal_correction_xy_m"] is None


def test_inconsistent_initial_correspondences_fail_fit_residual_check():
    check = LocalVisualFeedback()
    check.configure(observation(), reference())
    for xy, shift in zip(CALIBRATION_PATH, [(0, 0), (0, 0), (7, 0), (0, 0)]):
        result = check.observe(observation(xy, moving_delta=shift))
    assert result["status"] == "inconsistent_planar_response"
    assert result["residual_px"] > check.maximum_residual_px
    assert result["goal_correction_xy_m"] is None


def test_robot_motion_without_image_response_does_not_establish_a_mapping():
    check = LocalVisualFeedback()
    check.configure(observation(), reference())
    for xy in CALIBRATION_PATH:
        result = check.observe(observation(xy, moving_delta=-JACOBIAN @ xy))
    assert result["status"] == "ill_conditioned_image_jacobian"
    assert not result["calibration_valid"]
    assert result["goal_correction_xy_m"] is None


@pytest.mark.parametrize("alignment", [None, {}, {"semantic": "bowl_plate_centers", "planar_alignment": True},
                                       {"semantic": ANCHOR_SEMANTICS, "planar_alignment": False},
                                       {"semantic": ANCHOR_SEMANTICS, "planar_alignment": 1}])
def test_without_explicit_planar_same_point_assertion_only_tracking_is_available(alignment):
    check = LocalVisualFeedback()
    ref = reference(alignment=False)
    ref["local_alignment"] = alignment
    check.configure(observation(), ref)
    for xy in CALIBRATION_PATH:
        result = check.observe(observation(xy))
        assert result["reliable"] and result["status"] == "tracking_only"
        assert not result["calibration_valid"] and result["goal_correction_xy_m"] is None
        assert result["sample_count"] == 0


def test_tracking_only_supports_release_and_camera_viewpoint_invariant_robot_changes():
    check = LocalVisualFeedback()
    check.configure(observation(), reference(kind="held_object", alignment=False))
    result = check.observe(observation(z=1.2, rotation=(.3, 0., 0.), gripper=(.04, -.04)))
    assert result["reliable"] and result["status"] == "tracking_only"
    assert not result["calibration_valid"]
    assert result["task_success_certified"] is False


@pytest.mark.parametrize("ref,status", [
    (None, "visual_anchor_unavailable"), ({"kind": "end_effector", "visual_anchor": None}, "visual_anchor_unavailable"),
    ({"kind": "simulator_pose"}, "unsupported_goal_reference"),
])
def test_missing_or_unsupported_references_are_unavailable(ref, status):
    check = LocalVisualFeedback()
    assert check.configure(observation(), ref)["status"] == status
    assert check.observe(observation())["status"] == status


def test_overlapping_initial_templates_are_not_independent_evidence():
    ref = reference()
    ref["visual_anchor"]["target_point_uv"] = ref["visual_anchor"]["moving_point_uv"].copy()
    result = LocalVisualFeedback().configure(observation(), ref)
    assert result["status"] == "overlapping_anchors"
    assert not result["reliable"]


@pytest.mark.parametrize("point,status", [([0., .5], "moving_anchor_at_image_boundary"),
                                          ([True, .5], "invalid_visual_anchor"),
                                          ([float("nan"), .5], "invalid_visual_anchor"),
                                          ([-.01, .5], "invalid_visual_anchor")])
def test_bad_normalized_anchors_do_not_enter_tracker(point, status):
    ref = reference()
    ref["visual_anchor"]["moving_point_uv"] = point
    assert LocalVisualFeedback().configure(observation(), ref)["status"] == status


def test_external_phase_invalidation_clears_every_calibration_sample():
    check, _ = calibrate()
    result = check.invalidate("release_phase")
    assert result["sample_count"] == 0 and not result["calibration_valid"]
    assert check.observe(observation())["status"] == "release_phase"


def test_metric_goal_validation_exposes_and_enforces_proposed_calibration_domain():
    check, result = calibrate()
    assert result["calibration_domain"] == {
        "origin_xyz_m": [.1, .2, 1.], "maximum_xy_radius_m": .06,
        "maximum_height_change_m": .003,
    }
    assert check.validate_metric_goal([.1, .2, 1.])
    assert check.validate_metric_goal([.16, .2, 1.003])
    # A one-centimeter step starting at radius 5.9 cm ends outside the map's
    # six-centimeter domain, even though it remains in the 16-centimeter grid.
    assert not check.validate_metric_goal([.1 + .059 + .01, .2, 1.])
    # Domain is a disk, not independent six-centimeter per-axis bounds.
    assert not check.validate_metric_goal([.15, .25, 1.])
    # Keeping the old model goal height must not move a calibrated planar goal
    # beyond its actual stable working plane.
    assert not check.validate_metric_goal([.115, .21, 1.004])
    assert not check.validate_metric_goal([.115, .21, .996])


def test_metric_goal_validation_requires_current_valid_mapping_and_is_read_only():
    check = LocalVisualFeedback()
    assert not check.validate_metric_goal([.1, .2, 1.])
    check.configure(observation(), reference())
    assert not check.validate_metric_goal([.1, .2, 1.])
    check, _ = calibrate(check)
    snapshot = check.observe(observation(CALIBRATION_PATH[-1]))
    assert check.validate_metric_goal([.115, .215, 1.])
    assert check.observe(observation(CALIBRATION_PATH[-1])) == snapshot
    check.observe(observation(CALIBRATION_PATH[-1], z=1.004))
    assert not check.validate_metric_goal([.115, .215, 1.])
    check.configure(observation(), reference(alignment=False))
    assert not check.validate_metric_goal([.1, .2, 1.])


@pytest.mark.parametrize("goal", [None, [1., 2.], [1., 2., 3., 4.],
                                   [float("nan"), .2, 1.], [.1, float("inf"), 1.],
                                   {"x": .1, "y": .2, "z": 1.}, "not a goal"])
def test_metric_goal_validation_rejects_malformed_or_nonfinite_goals(goal):
    check, _ = calibrate()
    assert not check.validate_metric_goal(goal)


def test_configure_restarts_epoch_without_mutating_goal_reference_or_using_old_fit():
    check, _ = calibrate()
    ref = reference()
    before = copy.deepcopy(ref)
    check.configure(observation(), ref)
    result = check.observe(observation((.005, 0.)))
    assert ref == before
    assert result["sample_count"] == 1 and not result["calibration_valid"]


def test_mapping_and_hidden_state_dictionaries_are_not_an_input_path():
    check = LocalVisualFeedback()
    with pytest.raises(TypeError, match="PolicyObservation"):
        check.configure({**observation().to_dict(), "object_pose": [1., 2., 3.]}, reference())
    with pytest.raises(TypeError, match="PolicyObservation"):
        check.observe(observation().to_dict())


@pytest.mark.parametrize("options", [{"patch_radius": True}, {"search_radius": 4}, {"minimum_samples": 2},
                                     {"maximum_samples": 3}, {"minimum_score": 1.1},
                                     {"maximum_residual_px": float("nan")}, {"maximum_correction_m": .1},
                                     {"maximum_motion_condition": .5}, {"minimum_motion_m": True}])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        LocalVisualFeedback(**options)
