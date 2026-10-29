"""RoboCasa's mobile action dimensions must never receive policy arm values."""
import numpy as np
import pytest

from gpt_field.robocasa_user_check_sim import (
    RoboCasaConfig, expand_arm_action, validate_arm_action, validate_arm_only_displacement,
    validate_arm_controller,
)


INDEXES = {"right": (0, 6), "right_gripper": (6, 7), "base": (7, 10), "torso": (10, 11)}


def test_arm_mapping_holds_mobile_parts_and_selects_arm_mode():
    action = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1]
    full = expand_arm_action(action, INDEXES, 12, [0.141])
    np.testing.assert_allclose(full, action + [0, 0, 0, 0.141, -1])


def test_mapping_uses_declared_parts_instead_of_hardcoded_positions():
    reordered = {"base": (0, 3), "torso": (3, 4), "right": (4, 10), "right_gripper": (10, 11)}
    full = expand_arm_action([0.1] * 6 + [-1], reordered, 12, [0.2])
    np.testing.assert_allclose(full, [0, 0, 0, 0.2] + [0.1] * 6 + [-1, -1])


@pytest.mark.parametrize("action", [[0] * 6, [0] * 8, [True] + [0] * 6,
                                     [float("nan")] + [0] * 6,
                                     [float("inf")] + [0] * 6, [1.01] + [0] * 6])
def test_invalid_policy_actions_rejected(action):
    with pytest.raises(ValueError):
        validate_arm_action(action)


@pytest.mark.parametrize("indexes", [
    {**INDEXES, "torso": (9, 10)},
    {**INDEXES, "base": (8, 11)},
    {**INDEXES, "head": (11, 12)},
    {**INDEXES, "right": (0, 7)},
])
def test_rejects_unexpected_or_overlapping_robot_controller_parts(indexes):
    with pytest.raises(ValueError):
        expand_arm_action([0] * 7, indexes, 12, [0.1])


def test_config_rejects_other_tasks_and_non_native_images():
    with pytest.raises(ValueError):
        RoboCasaConfig(task="OpenDrawer")
    with pytest.raises(ValueError):
        RoboCasaConfig(image_size=224)
    assert RoboCasaConfig().external_camera == "robot0_agentview_left"


@pytest.mark.parametrize("base,torso", [([0.002, 0, 0], [0]), ([0, 0, 0.002], [0]),
                                          ([0, 0, 0], [0.002]),
                                          ([0.0008, 0.0008, 0], [0]),
                                          ([float("nan"), 0, 0], [0])])
def test_arm_only_trial_fails_if_uncommanded_mobile_motion_is_meaningful(base, torso):
    with pytest.raises(RuntimeError, match="Arm-only contract"):
        validate_arm_only_displacement(base, torso)
    validate_arm_only_displacement([1e-5, -1e-5, 1e-5], [1e-5])


def test_actual_arm_controller_must_match_world_delta_scales():
    actual = type("OperationalSpaceController", (), {
        "input_type": "delta", "input_ref_frame": "world", "control_dim": 6,
        "input_min": np.full(6, -1), "input_max": np.ones(6),
        "output_min": -np.array([0.05] * 3 + [0.5] * 3),
        "output_max": np.array([0.05] * 3 + [0.5] * 3)})()
    assert validate_arm_controller(actual)["input_ref_frame"] == "world"
    actual.input_ref_frame = "base"
    with pytest.raises(ValueError, match="Actual OSC"):
        validate_arm_controller(actual)
    actual.input_ref_frame = "world"
    actual.output_max = [0.1] * 6
    with pytest.raises(ValueError, match="Actual OSC"):
        validate_arm_controller(actual)
