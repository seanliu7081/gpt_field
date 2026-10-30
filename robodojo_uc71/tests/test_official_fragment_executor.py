"""Synthetic native-EEF receipts verify scheduler behavior, not physical IK."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from robodojo_uc71.fragment_executor import (ActionReceipt, FragmentExecutionError,
    RobotMeasurement)
from robodojo_uc71.lifecycle import EpisodeLifecycle, NativeResult
from robodojo_uc71.official_control import OFFICIAL_EE, validate_official_ee_action
from robodojo_uc71.schema import public_json, validate_feedback
from test_controller import arm
from test_fragment_executor import decision, executor, running, segment, short_motion


class EERobot:
    def __init__(self):
        self.states = {"left": arm(link6_world_jacobian=np.zeros((6, 6)), tcp_offset_link6=[0., 0., .1]),
                       "right": arm(link6_position=[.5, 0., 0.], tcp_offset_link6=[0., .1, 0.])}
        self.fingers = {side: [.02, .02] for side in self.states}
        self.actions = []
        self.ik_status = {"left": "Success", "right": "Success"}
        self.after_action = lambda ticket: None
        self.success_at = None

    def measure(self):
        return RobotMeasurement(deepcopy(self.states), deepcopy(self.fingers))

    def send(self, ticket, action):
        clean = validate_official_ee_action(action)
        self.actions.append((ticket, deepcopy(clean)))
        for side in self.states:
            if self.ik_status is not None and self.ik_status.get(side) == "Success":
                pose = clean[f"{side}_ee_pose"]
                self.states[side] = replace(self.states[side], link6_position=pose[:3], link6_quaternion_wxyz=pose[3:])
            # Native gripper execution remains possible even if that arm's IK failed.
            self.fingers[side] = [.02, .02] if clean[f"{side}_ee_joint_state"] == [1.] else [0., 0.]
        self.after_action(ticket)
        success = ticket == self.success_at
        return ActionReceipt(self.measure(), NativeResult(ticket, float(success), success,
            True, True, f"synthetic-ee-{ticket}", 37.5), deepcopy(self.ik_status))


def official_executor(robot, lifecycle=None, **kwargs):
    return executor(robot, lifecycle, control_backend=OFFICIAL_EE, **kwargs)


def test_final_tcp_targets_go_to_official_interface_without_local_dls():
    robot = EERobot()
    result = official_executor(robot).run(decision(segment(xyz=(.2, 0., 0.))))
    assert result.feedback["reason"] == "fragment_complete"
    assert result.confirmed_steps == result.submitted_steps == 2
    assert robot.actions[0][1]["left_ee_pose"] == pytest.approx([.2, 0., 0., 1., 0., 0., 0.])
    assert result.feedback["target_tool_pose_wxyz"][:3] == [.2, 0., .1]
    assert robot.actions[1][1] == robot.actions[0][1]
    assert validate_feedback(public_json(result.feedback)) == result.feedback


def test_official_inactive_pose_persists_across_same_arm_decisions():
    robot = EERobot()
    robot.after_action = lambda ticket: robot.states.__setitem__("right",
        replace(robot.states["right"], link6_position=[.6, 0., 0.]))
    engine = official_executor(robot)
    engine.run(short_motion(steps=1))
    engine.run(short_motion(steps=1))
    assert robot.actions[0][1]["right_ee_pose"][:3] == [.5, 0., 0.]
    assert robot.actions[1][1]["right_ee_pose"][:3] == [.5, 0., 0.]


@pytest.mark.parametrize("failed_arm", ["left", "right"])
def test_ik_failure_yields_after_actual_action_and_gripper_without_retry_or_rollback(failed_arm):
    robot, lifecycle = EERobot(), running()
    robot.ik_status[failed_arm] = "Fail"
    # The source can keep pursuing a previous command on failed IK. Its actual
    # resulting pose, rather than a fictional zero-motion hold, is reported.
    robot.after_action = lambda _: robot.states.__setitem__("left",
        replace(robot.states["left"], link6_position=[.003, 0., 0.]))
    result = official_executor(robot, lifecycle).run(short_motion(grip="close"))
    assert result.feedback["reason"] == "ik_unavailable"
    assert result.feedback["status"] == "blocked"
    assert result.confirmed_steps == result.submitted_steps == len(robot.actions) == 1
    assert lifecycle.confirmed_actions == 1 and lifecycle.state == "RUNNING"
    assert robot.actions[0][1]["left_ee_joint_state"] == [0.]
    assert robot.fingers["left"] == [0., 0.]
    assert result.feedback["gripper_command"] == "close"
    assert result.feedback["last_tool_pose_wxyz"][:3] == [.003, 0., .1]
    assert "ik_status" not in result.feedback


@pytest.mark.parametrize("status", [None, {"left": "Success"}, {"left": "Success", "right": "collision"}])
def test_official_receipt_requires_complete_sanitized_native_ik_status(status):
    robot, lifecycle = EERobot(), running()
    robot.ik_status = status
    with pytest.raises(FragmentExecutionError, match="native_action_or_measurement_unavailable"):
        official_executor(robot, lifecycle).run(short_motion())
    assert lifecycle.confirmed_actions == len(robot.actions) == 1
    assert lifecycle.terminal_record()["status"] == "infrastructure_error"


@pytest.mark.parametrize("kind", ["observe", "finish"])
def test_official_observe_finish_do_not_call_ik_or_advance_lifecycle(kind):
    robot, lifecycle = EERobot(), EpisodeLifecycle()
    robot.measure = lambda: pytest.fail("observation-only decision must not measure")
    result = official_executor(robot, lifecycle).run(decision(kind=kind))
    assert result.confirmed_steps == 0 and robot.actions == []
    assert lifecycle.state == "CREATED"


def test_official_target_envelope_still_rejects_before_dispatch():
    robot = EERobot()
    result = official_executor(robot).run(decision(segment("move_to_pose", xyz=(.31, 0., .1))))
    assert result.feedback["reason"] == "target_envelope_rejected"
    assert not robot.actions


def test_official_handoff_preserves_closed_gripper_latch():
    robot = EERobot()
    engine = official_executor(robot)
    result = engine.run(decision(segment("set_gripper", None, 25, "close")))
    assert result.confirmed_steps == 15
    engine.run(decision(segment(xyz=(.01, 0., 0.)), arm="right"))
    assert all(action["left_ee_joint_state"] == [0.] for _, action in robot.actions)
    assert robot.actions[-1][1]["left_ee_pose"][:3] == [0., 0., 0.]


def test_official_native_success_ends_actions_without_becoming_control_feedback():
    robot, lifecycle = EERobot(), running()
    robot.success_at = 1
    result = official_executor(robot, lifecycle).run(short_motion())
    assert len(robot.actions) == lifecycle.confirmed_actions == 1
    assert lifecycle.terminal_record()["official_success"] is True
    assert result.feedback["reason"] == "interrupted"
    assert "official_success" not in result.feedback


def test_official_backend_keeps_1400_native_action_horizon():
    robot, lifecycle = EERobot(), running()
    engine = official_executor(robot, lifecycle)
    for _ in range(9):
        engine.run(decision(segment("wait", None, 150)))
    result = engine.run(decision(segment("wait", None, 150)))
    assert result.confirmed_steps == 50
    assert len(robot.actions) == lifecycle.confirmed_actions == 1400
    assert lifecycle.terminal_record()["status"] == "horizon_exhausted"
