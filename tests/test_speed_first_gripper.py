"""Feedback-driven settling tests; no simulator truth or model I/O."""

import numpy as np
import pytest

from gpt_field.action import ActionAdapter
from gpt_field.speed_first_gripper import GripperSettleConfig, LocalGripperSettle


def state(fingers=(0.04, -0.04)):
    return np.array([0.1, 0.2, 0.9, 0., 0., 0., *fingers])


def tick(executor, before, *, fingers=None, xyz=None, rotvec=None):
    action = executor.next_action(before)
    after = before.copy()
    if fingers is not None:
        after[6:8] = fingers
    if xyz is not None:
        after[:3] = xyz
    if rotvec is not None:
        after[3:6] = rotvec
    return after, action, executor.observe_transition(before, action, after)


@pytest.mark.parametrize("command,initial,target,minimum,sign", [
    ("close", (.04, -.04), (.015, -.015), 15, 1.),
    ("open", (.015, -.015), (.04, -.04), 10, -1.),
])
def test_explicit_command_requires_measured_settling_and_command_minimum(command, initial, target, minimum, sign):
    measured = state(initial)
    executor = LocalGripperSettle(command, measured)
    for step in range(minimum):
        measured, action, feedback = tick(executor, measured, fingers=target)
        assert action == (0., 0., 0., 0., 0., 0., sign)
        assert executor.done is (step == minimum - 1)
    assert feedback["stop_reason"] == "measured_fingers_stable"
    assert feedback["actions_executed"] == minimum
    assert feedback["finger_motion_observed"]
    assert not feedback["grasp_success_certified"]
    assert not feedback["task_success_certified"]
    with pytest.raises(RuntimeError, match="new model decision"):
        executor.next_action(measured)


def test_no_motion_cannot_be_confused_with_early_successful_closure():
    measured = state()
    executor = LocalGripperSettle("close", measured)
    for step in range(30):
        measured, _, feedback = tick(executor, measured)
        assert executor.done is (step == 29)
    assert feedback["fingers_stable"]
    assert not feedback["finger_motion_observed"]
    assert feedback["stop_reason"] == "action_budget_exhausted"


def test_slowly_moving_fingers_do_not_pass_stable_window_even_with_small_per_step_changes():
    measured = state()
    executor = LocalGripperSettle("close", measured)
    for step in range(30):
        # Each difference is below .2 mm, but the full four-step window is not.
        finger = .04 - (step + 1) * .0001
        measured, _, feedback = tick(executor, measured, fingers=(finger, -finger))
        assert executor.done is (step == 29)
    assert feedback["finger_motion_observed"]
    assert not feedback["fingers_stable"]
    assert feedback["stop_reason"] == "action_budget_exhausted"


def test_stability_requires_both_fingers_and_a_full_feedback_window():
    config = GripperSettleConfig(min_close_actions=5, min_open_actions=5, max_actions=20)
    measured = state()
    executor = LocalGripperSettle("close", measured, config=config)
    for index in range(7):
        measured, _, _ = tick(executor, measured, fingers=(.02, -.04 + index * .001))
        assert not executor.done
    for index in range(4):
        measured, _, _ = tick(executor, measured)
        assert executor.done is (index == 3)
    assert executor.actions_executed == 11


def test_pose_hold_uses_fixed_measured_start_and_direction_preserving_bounded_correction():
    measured = state()
    start = measured.copy()
    adapter = ActionAdapter(max_rotation_rad=.15)
    executor = LocalGripperSettle("close", measured, adapter=adapter)
    measured, _, _ = tick(executor, measured, xyz=start[:3] + [.03, .04, 0.], rotvec=[0., 0., .3])
    action = executor.next_action(measured)
    np.testing.assert_allclose(action[:3], [-.12, -.16, 0.], atol=1e-14)
    np.testing.assert_allclose(action[3:6], [0., 0., -.3], atol=1e-14)
    assert np.linalg.norm(np.array(action[:3]) * .05) == pytest.approx(.01)
    assert action[6] == adapter._gripper == 1.
    assert executor.diagnostics()["pose_error_m"] == pytest.approx(.05)
    np.testing.assert_array_equal(executor.diagnostics()["pose_target_xyz_m"], start[:3])


def test_pose_hold_corrects_small_drift_without_minimum_movement():
    measured = state()
    executor = LocalGripperSettle("open", measured)
    measured, _, _ = tick(executor, measured, xyz=measured[:3] + [.001, -.002, .003])
    action = executor.next_action(measured)
    np.testing.assert_allclose(action[:3], [-.02, .04, -.06], atol=1e-14)
    assert action[6] == -1.


def test_actions_only_advance_with_matching_feedback():
    measured = state()
    executor = LocalGripperSettle("close", measured)
    with pytest.raises(RuntimeError, match="no pending"):
        executor.observe_transition(measured, [0.] * 7, measured)
    action = executor.next_action(measured)
    assert executor.actions_executed == 0
    with pytest.raises(RuntimeError, match="observe_transition"):
        executor.next_action(measured)
    incorrect = measured.copy()
    incorrect[0] += .01
    with pytest.raises(ValueError, match="before-state"):
        executor.observe_transition(incorrect, action, measured)
    with pytest.raises(ValueError, match="action must match"):
        executor.observe_transition(measured, [0.] * 7, measured)
    executor.observe_transition(measured, action, incorrect)
    assert executor.actions_executed == 1
    with pytest.raises(RuntimeError, match="no pending"):
        executor.observe_transition(measured, action, incorrect)
    with pytest.raises(ValueError, match="preceding measured feedback"):
        executor.next_action(measured)
    executor.next_action(incorrect)


@pytest.mark.parametrize("command", ["hold", "grasp", "release", None, True])
def test_only_explicit_open_and_close_are_accepted(command):
    with pytest.raises(ValueError, match="explicit open or close"):
        LocalGripperSettle(command, state())


@pytest.mark.parametrize("kwargs", [
    {"min_close_actions": 31}, {"min_open_actions": 0}, {"max_actions": True},
    {"stable_feedback_steps": 31}, {"finger_stability_tolerance_m": 0.},
    {"finger_motion_threshold_m": .0001}, {"max_pose_correction_m": .1},
    {"max_pose_correction_m": float("nan")},
])
def test_config_rejects_invalid_budgets_and_bounds(kwargs):
    with pytest.raises(ValueError):
        GripperSettleConfig(**kwargs)


@pytest.mark.parametrize("invalid", [[0.] * 7, [False] * 8, [0.] * 7 + [float("nan")], ["0"] * 8])
def test_non_proprioceptive_state_values_are_rejected(invalid):
    with pytest.raises(ValueError, match="eight finite real"):
        LocalGripperSettle("close", invalid)
