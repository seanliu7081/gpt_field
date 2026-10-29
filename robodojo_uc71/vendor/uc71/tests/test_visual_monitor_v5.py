"""V5 fixed-feature evidence: deformation, occlusion, identity and budget cases."""
import numpy as np
import pytest

from gpt_field.visual_monitor_v5 import LocalFeatureBundle, RoleAwareVisualMonitorV5, _sample


def textured(seed=12):
    rng = np.random.default_rng(seed)
    # Smoothed random texture approximates image edges without ambiguous periodic
    # tiles. Independent seeds make the distractor identity test nontrivial.
    values = rng.uniform(20, 230, (112, 112))
    return np.lib.stride_tricks.sliding_window_view(np.pad(values, 1), (3, 3)).mean(axis=(-1, -2))


def transform(image, shift=(0., 0.), angle=0., scale=1.):
    c, s = np.cos(angle), np.sin(angle)
    matrix = scale*np.array([[c, s], [-s, c]])
    center = np.array([55., 55.])
    rows, cols = np.indices(image.shape)
    destination = np.stack((cols, rows), axis=-1)
    origin = (destination-center-np.asarray(shift))@np.linalg.inv(matrix)+center
    valid = (origin[..., 0] >= 0) & (origin[..., 1] >= 0) & (origin[..., 0] < 111) & (origin[..., 1] < 111)
    result = np.zeros_like(image)
    result[valid] = _sample(image, origin[valid])
    return result


def test_translation_scale_and_rotation_preserve_seed_with_fixed_feature_identities():
    image = textured()
    tracker = LocalFeatureBundle(image, [55/111, 55/111], "tool")
    initial = tracker.initial_points.copy()
    assert len(initial) >= 5
    for n in range(1, 9):
        moved = transform(image, (.55*n, -.3*n), .006*n, 1+.004*n)
        assert tracker.track(moved), tracker.failure_reason
        assert np.linalg.norm(tracker.position-[55+.55*n, 55-.3*n]) < .7
        assert set(tracker.active).issubset(set(range(len(initial))))
    assert np.array_equal(tracker.initial_points, initial)
    assert tracker.summary()["automatic_reseeding"] is False


def test_partial_feature_occlusion_keeps_the_same_consensus_seed():
    image = textured()
    tracker = LocalFeatureBundle(image, [55/111, 55/111], "tool")
    moved = transform(image, (1., 1.))
    # Occlude one side, leaving enough original features on the rigid region.
    moved[43:52, 44:66] = 110
    assert tracker.track(moved), tracker.failure_reason
    assert len(tracker.active) < len(tracker.initial_points)
    assert np.linalg.norm(tracker.position-[56, 56]) < .8


def test_texture_replacement_does_not_reacquire_a_different_identity():
    image = textured()
    tracker = LocalFeatureBundle(image, [.5, .5], "moving_object")
    assert not tracker.track(textured(94))
    assert not tracker.reliable
    # Even restoration cannot resurrect evidence without explicit model reseed.
    assert not tracker.track(image)


def test_declared_roi_contains_all_original_descriptors_and_bounds_search_support():
    image = textured()
    roi = np.array([44, 44, 66, 66])/111
    tracker = LocalFeatureBundle(image, [55/111, 55/111], "tool", roi_uv=roi, support_radius_px=16)
    assert np.all(tracker.initial_points-3 >= [44, 44])
    assert np.all(tracker.initial_points+3 <= [66, 66])
    assert np.all(np.linalg.norm(tracker.initial_points-tracker.seed, axis=1) <= 16)
    assert tracker.summary()["seed_roi_uv"] == pytest.approx(roi)


@pytest.mark.parametrize("roi", [[.7, .7, .9, .9], [.8, .2, .1, .7], [-.1, 0, 1, 1]])
def test_bad_roi_cannot_silently_select_an_outside_feature(roi):
    with pytest.raises(ValueError):
        LocalFeatureBundle(textured(), [.5, .5], "tool", roi_uv=roi)


def test_untextured_seed_does_not_use_distant_textured_background():
    image = textured()
    image[35:77, 35:77] = 110
    with pytest.raises(ValueError, match="insufficient_seed_local_corners"):
        LocalFeatureBundle(image, [.5, .5], "tool")


def refs():
    return {"external": {"moving_id": "tool", "moving_kind": "tool",
            "moving_point_uv": [.3, .4], "moving_roi_uv": [.2, .3, .4, .5],
            "target_id": "object", "target_anchor_uv": [.7, .7],
            "target_point_uv": [.7, .7], "target_roi_uv": [.6, .6, .8, .8],
            "tool_point_uv": None, "tool_roi_uv": None}, "wrist": None}


def rgb(gray):
    return np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=2)


def test_static_background_cannot_validate_robot_motion_or_renew_budget():
    image = rgb(textured())
    images = {"external": image, "wrist": image}
    state = np.zeros(8)
    monitor = RoleAwareVisualMonitorV5()
    monitor.configure(images, refs(), state, intent_id=1)
    assert monitor.control_limit() == (.005, None)  # No mandatory calibration.
    for n in range(1, 9):
        state[0] = n*.002
        monitor.observe(images, state, [.1, 0, 0, 0, 0, 0, 0])
    assert not monitor.response_ready
    assert monitor.maximum_feature_displacement_px < 1
    assert monitor.unvalidated_path_m == pytest.approx(.016)
    assert monitor.control_limit()[0] == 0
    assert monitor.status == "controlled_feature_unresponsive"


def test_so3_wrap_and_domain_reset_still_use_invariant_and_finite_budget():
    image = rgb(textured())
    images = {"external": image, "wrist": image}
    state = np.zeros(8); state[3] = np.pi
    monitor = RoleAwareVisualMonitorV5()
    monitor.configure(images, refs(), state, intent_id=1)
    state[0] = .003; state[3] = -np.pi
    monitor.observe(images, state, [.1, 0, 0, 0, 0, 0, 0])
    assert monitor.path_m == pytest.approx(.003)
    state[0] = .006; state[3] += .08
    monitor.observe(images, state, [.1, 0, 0, 0, 0, 0, 0])
    assert monitor.path_m == 0
    assert monitor.unvalidated_path_m == pytest.approx(.006)


def test_coherent_subpixel_motion_is_validated_by_prediction_against_static_baseline():
    base = textured()
    image = rgb(base)
    state = np.zeros(8)
    monitor = RoleAwareVisualMonitorV5()
    monitor.configure({"external": image, "wrist": image}, refs(), state, intent_id=1)
    for n in range(1, 13):
        moved = base.copy()
        warped = transform(base, (.05*n, 0.))
        moved[25:59, 16:50] = warped[25:59, 16:50]
        state[0] = .001*n
        image = rgb(moved)
        monitor.observe({"external": image, "wrist": image}, state, [.1, 0, 0, 0, 0, 0, 0])
    assert .35 <= monitor.maximum_feature_displacement_px < 1.
    assert monitor.response_ready, monitor.summary()
    assert monitor.control_limit() == (None, None)
    summary = monitor.summary()
    assert summary["held_out_prediction_squared_error"] <= .7*summary["held_out_zero_motion_squared_error"]


def test_random_subpixel_jitter_cannot_validate_a_measured_robot_response():
    base = textured()
    image = rgb(base)
    state = np.zeros(8)
    monitor = RoleAwareVisualMonitorV5()
    monitor.configure({"external": image, "wrist": image}, refs(), state, intent_id=1)
    rng = np.random.default_rng(72)
    for n in range(1, 17):
        # Adversarial image jitter on the nominal moving region has enough
        # displacement to pass the pixel gate; it is unrelated to robot motion.
        moved = base.copy()
        warped = transform(base, (float(rng.uniform(-.5, .5)), 0.))
        moved[25:59, 16:50] = warped[25:59, 16:50]
        state[0] = .001*n
        image = rgb(moved)
        monitor.observe({"external": image, "wrist": image}, state, [.1, 0, 0, 0, 0, 0, 0])
    assert monitor.maximum_feature_displacement_px > .35
    assert not monitor.response_ready
    assert monitor.control_limit()[0] == 0
    assert monitor.unvalidated_path_m >= .015
