"""Robot-only action conversion checks; no native IK or simulator is invoked."""
from dataclasses import replace
import math

import numpy as np
import pytest

from robodojo_uc71.controller import ControllerError, measured_tcp_pose
from robodojo_uc71.official_control import (EE_ACTION_KEYS, LEGACY_JOINT, OFFICIAL_EE,
    OfficialEEController, validate_control_action, validate_control_backend,
    validate_ik_status, validate_official_ee_action, world_link6_target)
from test_controller import arm


def ee_action():
    return {"left_ee_pose": [.1, .2, .3, 1., 0., 0., 0.],
            "right_ee_pose": [-.1, .2, .3, 1., 0., 0., 0.],
            "left_ee_joint_state": [0.], "right_ee_joint_state": [1.]}


def test_tcp_target_uses_target_orientation_instead_of_current_orientation():
    state = arm(link6_position=[.1, .2, .3], tcp_offset_link6=[.2, 0., 0.])
    target_rotation = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
    target = world_link6_target(state, [.4, .6, .7], target_rotation)
    assert target == pytest.approx([.4, .4, .7, *target_rotation])
    target_state = replace(state, link6_position=target[:3], link6_quaternion_wxyz=target[3:])
    # Independent forward transform closes the tool→link6→tool conversion.
    tcp, orientation = measured_tcp_pose(target_state)
    assert tcp == pytest.approx([.4, .6, .7])
    assert orientation == pytest.approx(target_rotation)


def test_official_controller_sends_final_target_without_dls_or_joint_limits_on_increment():
    states = {"left": arm(link6_world_jacobian=np.zeros((6, 6)), tcp_offset_link6=[0., 0., .1]),
              "right": arm(link6_position=[.5, .1, .4])}
    controller = OfficialEEController(gripper_commands={"left": 1., "right": 0.})
    controller.handoff("left", states)
    action = controller.command(states, [.2, 0., .1], [1., 0., 0., 0.], profile="contact")
    assert set(action) == EE_ACTION_KEYS
    assert action["left_ee_pose"] == pytest.approx([.2, 0., 0., 1., 0., 0., 0.])
    assert action["right_ee_pose"] == pytest.approx([.5, .1, .4, 1., 0., 0., 0.])
    assert action["right_ee_joint_state"] == [0.]


def test_inactive_pose_is_frozen_until_handoff_and_grippers_remain_latched():
    states = {"left": arm(link6_position=[.1, .2, .3]),
              "right": arm(link6_position=[.5, .6, .7])}
    controller = OfficialEEController(gripper_commands={"left": 1., "right": 1.})
    controller.handoff("left", states)
    drifted = {**states, "right": replace(states["right"], link6_position=[.51, .62, .73])}
    action = controller.command(drifted, [.15, .2, .3], [1., 0., 0., 0.], gripper_commands={"left": 0.})
    assert action["right_ee_pose"][:3] == [.5, .6, .7]
    action["right_ee_pose"][0] = 999.  # Returned actions do not mutate the hold.
    assert controller.command(drifted, [.15, .2, .3], [1., 0., 0., 0.])["right_ee_pose"][:3] == [.5, .6, .7]
    newer = {**drifted, "left": replace(states["left"], link6_position=[.12, .2, .3])}
    controller.handoff("right", newer)
    action = controller.command(newer, [.55, .6, .7], [1., 0., 0., 0.])
    assert action["left_ee_pose"][:3] == [.12, .2, .3]
    assert action["left_ee_joint_state"] == [0.]
    assert action["right_ee_joint_state"] == [1.]


@pytest.mark.parametrize("bad", [None, "ee", "joint", [], True, {}])
def test_backend_names_are_explicit(bad):
    with pytest.raises(ValueError, match="unsupported_control_backend"):
        validate_control_backend(bad)


def test_action_validation_has_exact_backend_keys_and_is_copying():
    source = ee_action()
    clean = validate_official_ee_action(source)
    assert clean == source
    clean["left_ee_pose"][0] = 99.
    assert source["left_ee_pose"][0] == .1
    with pytest.raises(ValueError, match="exact_joint_mode_keys_required"):
        validate_control_action(source, LEGACY_JOINT)
    joints = {"left_arm_joint_state": [0.] * 6, "right_arm_joint_state": [0.] * 6,
              "left_ee_joint_state": [1.], "right_ee_joint_state": [0.]}
    assert validate_control_action(joints, LEGACY_JOINT) == joints
    with pytest.raises(ValueError, match="exact_official_ee_keys_required"):
        validate_control_action(joints, OFFICIAL_EE)


@pytest.mark.parametrize("key, value", [
    ("left_ee_pose", [0.] * 7),
    ("left_ee_pose", [0., 0., 0., 2., 0., 0., 0.]),
    ("left_ee_pose", [float("nan"), 0., 0., 1., 0., 0., 0.]),
    ("left_ee_pose", [0., 0., 0., 1., 0., 0.]),
    ("left_ee_pose", [True, 0., 0., 1., 0., 0., 0.]),
    ("right_ee_joint_state", [.5]),
    ("right_ee_joint_state", [False]),
])
def test_malformed_official_actions_are_rejected_before_dispatch(key, value):
    source = ee_action()
    source[key] = value
    with pytest.raises(ValueError, match="invalid_native_ee_values"):
        validate_official_ee_action(source)


def test_unknown_native_metadata_cannot_enter_action_or_ik_status():
    with pytest.raises(ValueError, match="exact_official_ee_keys_required"):
        validate_official_ee_action({**ee_action(), "scene_model": {}})
    assert validate_ik_status({"left": "Success", "right": "Fail"}) == {"left": "Success", "right": "Fail"}
    for value in (None, {}, {"left": "Success"}, {"left": "Success", "right": "self_collision"},
                  {"left": "Success", "right": "Fail", "reason": "table_collision"}):
        with pytest.raises(ValueError, match="invalid_native_ik_status"):
            validate_ik_status(value)


def test_handoff_and_binary_gripper_contracts():
    states = {"left": arm(), "right": arm()}
    controller = OfficialEEController(gripper_commands={"left": 1., "right": 1.})
    with pytest.raises(ControllerError, match="arm_handoff_required"):
        controller.command(states, [0., 0., 0.], [1., 0., 0., 0.])
    controller.handoff("left", states)
    with pytest.raises(ControllerError, match="invalid_gripper_command"):
        controller.command(states, [0., 0., 0.], [1., 0., 0., 0.], gripper_commands={"left": .5})
