"""Synthetic affine robot only; no simulator/model, calibration, or task result."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from robodojo_uc71.controller import ArmState, ControllerConfig
from robodojo_uc71.fragment_executor import (ActionReceipt, FragmentExecutionError,
    FragmentExecutor, MonitoringSettings, RobotMeasurement)
from robodojo_uc71.lifecycle import EpisodeLifecycle, LifecycleError, NativeResult
from robodojo_uc71.schema import public_json, validate_feedback


def running():
    lifecycle = EpisodeLifecycle()
    lifecycle.ready()
    lifecycle.start({"schema": "robodojo-uc71-consumed-1", "attempts": 1,
                     "claim_nonce": "synthetic-unit-test", "manifest_sha256": "a"*64,
                     "ready_sha256": "b"*64})
    return lifecycle


def monitoring(**changes):
    # Deliberately synthetic thresholds; these are NOT ARX calibration values.
    settings = MonitoringSettings({"transit": .0001, "contact": .0001},
        {"transit": .001, "contact": .001}, 2, 2, .0001, .01, 2, 3, 4, .00001, .05)
    return replace(settings, **changes)


def decision(*segments, arm="left", kind="execute", motion=None):
    return {"belief": "Synthetic unit test", "plan": "Requested bounded motion",
            "why": "Exercise accounting", "type": kind,
            "arm_id": None if kind in ("observe", "finish") else arm,
            "execution": {"segments": list(segments)} if kind == "execute" else None,
            "motion": motion}


def segment(skill="move_relative", xyz=(.01, 0, 0), steps=25, grip="keep", checkpoint=False):
    return {"skill": skill, "xyz_m": list(xyz) if xyz is not None else None,
            "quaternion_wxyz": None, "gripper": grip, "profile": "transit",
            "max_steps": steps, "observation_after": checkpoint}


def short_motion(xyz=(.04, 0, 0), steps=5, grip="keep"):
    return decision(kind="motion", motion={"frame": "world", "position_mode": "relative",
        "xyz_m": list(xyz), "quaternion_wxyz": None, "gripper": grip, "steps": steps})


class AffineRobot:
    def __init__(self, *, follow=True):
        self.q = {arm: np.zeros(6) for arm in ("left", "right")}
        self.fingers = {arm: [.02, .02] for arm in self.q}
        self.actions = []
        self.follow = follow
        self.success_at = None
        self.after_action = lambda ticket: None

    def measure(self):
        arms = {}
        for arm, q in self.q.items():
            base = np.array([0. if arm == "left" else .5, 0., 0.])
            arms[arm] = ArmState(q.copy(), np.zeros(6), base + q[:3], [1., 0., 0., 0.],
                np.eye(6), -np.ones(6)*4, np.ones(6)*4, np.ones(6)*5, np.zeros(3))
        return RobotMeasurement(arms, deepcopy(self.fingers))

    def send(self, ticket, action):
        self.actions.append((ticket, deepcopy(action)))
        if self.follow:
            for arm in self.q:
                self.q[arm] = np.array(action[f"{arm}_arm_joint_state"])
        self.after_action(ticket)
        success = ticket == self.success_at
        return ActionReceipt(self.measure(), NativeResult(ticket, float(success), success,
            True, True, f"synthetic-{ticket}", 37.5))


def executor(robot, lifecycle=None, **kwargs):
    return FragmentExecutor(lifecycle=lifecycle or running(), measure=robot.measure,
        send_native_joint_action=robot.send, gripper_commands={"left": "open", "right": "open"},
        monitoring=monitoring(), controller_config=ControllerConfig(), **kwargs)


@pytest.mark.parametrize("kind", ["observe", "finish"])
def test_observe_and_finish_issue_no_measurement_or_action_and_cannot_certify_success(kind):
    robot = AffineRobot()
    robot.measure = lambda: pytest.fail("zero-action decision must not read a robot")
    lifecycle = EpisodeLifecycle()
    result = executor(robot, lifecycle).run(decision(kind=kind))
    assert not robot.actions
    assert lifecycle.state == "CREATED"
    assert result.confirmed_steps == result.submitted_steps == 0
    assert result.feedback["status"] == "pending_verification"
    assert result.observation_required
    assert result.calibration_status == "uncalibrated_candidate"
    assert "success" not in json.dumps(result.feedback)


def test_physical_execution_requires_running_caller_lifecycle():
    robot = AffineRobot()
    with pytest.raises(LifecycleError):
        executor(robot, EpisodeLifecycle()).run(short_motion())
    assert robot.actions == []


def test_relative_stage_targets_use_new_measured_entry_and_native_action_shape():
    robot = AffineRobot()
    result = executor(robot).run(decision(segment(), segment(xyz=(.02, 0, 0))))
    feedback = result.feedback
    assert feedback["reason"] == "fragment_complete"
    assert feedback["completed_segment_indices"] == [0, 1]
    assert feedback["pending_segment_indices"] == []
    assert feedback["target_tool_pose_wxyz"][0] - feedback["entry_tool_pose_wxyz"][0] == pytest.approx(.02)
    assert result.confirmed_steps == result.submitted_steps == len(robot.actions) <= 50
    for ticket, action in robot.actions:
        assert ticket >= 1
        assert set(action) == {"left_arm_joint_state", "right_arm_joint_state", "left_ee_joint_state", "right_ee_joint_state"}
        assert len(action["left_arm_joint_state"]) == 6
        assert action["right_arm_joint_state"] == [0.]*6
        assert action["left_ee_joint_state"] == action["right_ee_joint_state"] == [1.]
    assert validate_feedback(public_json(feedback)) == feedback
    assert "partial_score" not in json.dumps(feedback)


@pytest.mark.parametrize("first", [segment(checkpoint=True), segment("wait", None, 3)])
def test_checkpoint_and_wait_discard_pending_stages_without_autoresume(first):
    robot = AffineRobot()
    engine = executor(robot)
    result = engine.run(decision(first, segment("set_gripper", None, 15, "close")))
    assert result.feedback["reason"] == "observation_required"
    assert result.feedback["completed_segment_indices"] == [0]
    assert result.feedback["pending_segment_indices"] == [1]
    count = len(robot.actions)
    engine.run(decision(kind="observe"))
    assert len(robot.actions) == count
    assert all(action["left_ee_joint_state"] == [1.] for _, action in robot.actions)


def test_gripper_minimum_and_latch_survive_next_fragment_and_arm_handoff():
    robot = AffineRobot()
    engine = executor(robot)
    result = engine.run(decision(segment("set_gripper", None, 25, "close")))
    assert result.confirmed_steps == 15
    assert result.feedback["reason"] == "fragment_complete"
    assert result.feedback["gripper_command"] == "close"
    engine.run(short_motion(steps=1))
    engine.run(decision(segment("wait", None, 1), arm="right"))
    assert all(action["left_ee_joint_state"] == [0.] and action["right_ee_joint_state"] == [1.]
               for _, action in robot.actions)


def test_inactive_arm_keeps_handoff_reference_across_same_arm_decisions():
    robot = AffineRobot()
    robot.after_action = lambda ticket: robot.q["right"].__setitem__(0, .1) if ticket == 1 else None
    engine = executor(robot)
    engine.run(short_motion(steps=1))
    engine.run(short_motion(steps=1))
    assert robot.actions[0][1]["right_arm_joint_state"][0] == 0.
    assert robot.actions[1][1]["right_arm_joint_state"][0] == pytest.approx(.05)


@pytest.mark.parametrize("model_decision", [decision(segment("move_to_pose", (.31, 0, 0))),
    decision(segment("move_to_pose", (1e300, 0, 0))),
    decision({**segment(), "quaternion_wxyz": [0., 1., 0., 0.]})])
def test_measured_entry_envelope_rejects_before_any_dispatch(model_decision):
    robot = AffineRobot()
    result = executor(robot).run(model_decision)
    assert result.feedback["reason"] == "target_envelope_rejected"
    assert result.confirmed_steps == result.submitted_steps == 0
    assert robot.actions == []


def test_later_bad_target_does_not_erase_already_confirmed_fragment_actions():
    robot = AffineRobot()
    result = executor(robot).run(decision(segment(), segment("move_to_pose", (.8, 0, 0))))
    assert result.feedback["reason"] == "target_envelope_rejected"
    assert result.feedback["completed_segment_indices"] == [0]
    assert 0 < result.confirmed_steps == len(robot.actions) < 25


def test_short_motion_is_fixed_target_and_yields_after_stable_measured_arrival():
    robot = AffineRobot()
    result = executor(robot).run(short_motion())
    assert 1 <= result.confirmed_steps <= 5
    assert result.feedback["target_tool_pose_wxyz"][:3] == [.04, 0., 0.]
    assert result.feedback["reason"] == "pose_arrived"
    assert result.feedback["last_tool_pose_wxyz"][0] == pytest.approx(.04, abs=.0001)
    assert "pending_segment_indices" not in result.feedback


def test_short_motion_without_arrival_uses_only_requested_allowance():
    robot = AffineRobot(follow=False)
    result = executor(robot).run(short_motion(steps=2))
    assert result.confirmed_steps == len(robot.actions) == 2
    assert result.feedback["reason"] == "segment_budget_exhausted"


@pytest.mark.parametrize("steps, reason, count", [(3, "segment_budget_exhausted", 3), (25, "segment_stall", 4)])
def test_no_progress_returns_to_model_without_retry(steps, reason, count):
    robot = AffineRobot(follow=False)
    result = executor(robot).run(decision(segment(steps=steps)))
    assert result.feedback["reason"] == reason
    assert result.confirmed_steps == len(robot.actions) == count


def test_closed_gap_rise_yields_without_claiming_a_contact_or_grasp_cause():
    robot = AffineRobot()
    engine = executor(robot)
    engine.run(decision(segment("set_gripper", None, 15, "close")))
    robot.after_action = lambda ticket: robot.fingers.__setitem__("left", [.03, .03])
    result = engine.run(decision(segment("wait", None, 20)))
    assert result.feedback["reason"] == "maintained_grip_gap_changed"
    assert result.confirmed_steps == 2
    assert validate_feedback(public_json(result.feedback)) == result.feedback


def test_native_success_stops_dispatch_but_is_not_copied_to_control_feedback():
    robot = AffineRobot()
    robot.success_at = 1
    lifecycle = running()
    result = executor(robot, lifecycle).run(decision(segment()))
    assert len(robot.actions) == 1
    assert lifecycle.terminal_record()["official_success"] is True
    assert result.feedback["reason"] == "interrupted"
    assert validate_feedback(public_json(result.feedback)) == result.feedback
    assert "37.5" not in json.dumps(result.feedback)


def test_standard_lifecycle_stops_exactly_at_1400_without_1401_or_reset():
    robot = AffineRobot()
    lifecycle = running()
    engine = executor(robot, lifecycle)
    for _ in range(9):
        engine.run(decision(segment("wait", None, 150)))
    result = engine.run(decision(segment("wait", None, 150)))
    assert result.confirmed_steps == 50
    assert len(robot.actions) == lifecycle.confirmed_actions == lifecycle.submitted_actions == 1400
    assert lifecycle.terminal_record()["status"] == "horizon_exhausted"
    with pytest.raises(LifecycleError):
        engine.run(short_motion())
    assert len(robot.actions) == 1400


def test_lost_dispatch_confirmation_is_infrastructure_terminal_and_never_resent():
    robot = AffineRobot()
    def lost(ticket, action):
        robot.actions.append((ticket, action))
        raise RuntimeError("privileged object truth must not reach feedback")
    robot.send = lost
    lifecycle = running()
    engine = executor(robot, lifecycle)
    with pytest.raises(FragmentExecutionError, match="^native_action_or_measurement_unavailable$"):
        engine.run(short_motion())
    terminal = lifecycle.terminal_record()
    assert terminal["status"] == "infrastructure_error"
    assert terminal["unconfirmed_action"] == 1
    assert terminal["confirmed_actions"] == 0
    with pytest.raises(LifecycleError):
        engine.run(short_motion())
    assert len(robot.actions) == 1


def test_user_stop_at_feedback_boundary_sends_no_action():
    robot = AffineRobot()
    lifecycle = running()
    result = executor(robot, lifecycle, stop_requested=lambda: True).run(short_motion())
    assert result.feedback["reason"] == "interrupted"
    assert not robot.actions
    assert lifecycle.terminal_record()["status"] == "user_stop"


def test_unverified_native_callback_cannot_confirm_or_succeed():
    robot = AffineRobot()
    native_send = robot.send
    def unverified(ticket, action):
        receipt = native_send(ticket, action)
        return replace(receipt, native_result=replace(receipt.native_result,
            native_reward=1., end_flag=True, callback_verified=False))
    robot.send = unverified
    lifecycle = running()
    with pytest.raises(FragmentExecutionError):
        executor(robot, lifecycle).run(short_motion())
    terminal = lifecycle.terminal_record()
    assert terminal["status"] == "infrastructure_error"
    assert terminal["official_success"] is None
    assert terminal["confirmed_actions"] == 0
    assert terminal["unconfirmed_action"] == 1


def test_unexpected_callback_exception_is_sanitized_and_stops_infrastructure():
    robot = AffineRobot()
    lifecycle = running()
    def broken_stop():
        raise ValueError("privileged data must not escape callback")
    with pytest.raises(FragmentExecutionError, match="^executor_infrastructure_error$"):
        executor(robot, lifecycle, stop_requested=broken_stop).run(short_motion())
    assert not robot.actions
    assert lifecycle.terminal_record()["status"] == "infrastructure_error"
