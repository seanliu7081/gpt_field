"""Native004 regression and numerical holds; no simulator or native dispatch."""
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from robodojo_uc71.controller import (ArmState, ControllerConfig, ControllerError,
    DualArmController, joint_target, measured_tcp_pose, POSITION_NUMERICAL_RESOLUTION_M,
    ORIENTATION_NUMERICAL_RESOLUTION_RAD)


def native_fixture():
    value = json.loads((Path(__file__).parent / "fixtures/controller_native004_near_target.json").read_text())
    return value, {arm: ArmState(**fields) for arm, fields in value["states"].items()}


def world_rotation(q, axis, angle):
    a = np.zeros(3); a[axis] = math.sin(angle / 2)
    w, xyz = q[0], np.asarray(q[1:])
    return np.r_[math.cos(angle/2)*w-a@xyz,
                 math.cos(angle/2)*xyz+w*a+np.cross(a, xyz)].tolist()


def test_exact_native004_rejected_state_is_a_numerical_hold_with_gripper_commands():
    fixture, states = native_fixture()
    assert set(fixture["native_pose_dtypes"].values()) == {"float32"}
    target = fixture["target_tool_pose_wxyz"]
    config = ControllerConfig(**fixture["controller_config"])
    result = joint_target(states["left"], target[:3], target[3:], config=config)
    assert result.status == "at_target"
    assert result.position_error_m == pytest.approx(3.096960041597894e-8, rel=1e-5)
    assert result.orientation_error_rad == pytest.approx(8.946420489969285e-8, rel=1e-5)
    assert np.array_equal(result.joint_positions, states["left"].joint_positions)
    assert result.predicted_translation_m == result.predicted_rotation_rad == (0.,)*3
    controller = DualArmController(gripper_commands={"left":1., "right":1.}, config=config)
    controller.handoff("left", states)
    for grip in (0., 1.):
        command = controller.command(states, target[:3], target[3:], gripper_commands={"left":grip})
        assert command.active_result.status == "at_target"
        assert command.gripper_commands == {"left":grip, "right":1.}
        for arm in ("left", "right"):
            assert np.array_equal(command.joint_positions[arm], states[arm].joint_positions)


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("sign", [-1., 1.])
def test_native_both_arm_translation_and_rotation_quantization_hold(arm, axis, sign):
    _, states = native_fixture()
    state = states[arm]; position, quaternion = measured_tcp_pose(state)
    position[axis] += sign*.5*POSITION_NUMERICAL_RESOLUTION_M
    target_q = world_rotation(quaternion, axis, sign*.5*ORIENTATION_NUMERICAL_RESOLUTION_RAD)
    result = joint_target(state, position, target_q)
    assert result.status == "at_target"
    assert np.array_equal(result.joint_positions, state.joint_positions)


@pytest.mark.parametrize("kind", ["translation", "rotation"])
def test_outside_numerical_floor_still_commands_and_zero_jacobian_still_fails(kind):
    state = ArmState(np.zeros(6), np.zeros(6), np.zeros(3), [1.,0.,0.,0.], np.eye(6),
                     -np.ones(6)*2, np.ones(6)*2, np.ones(6)*5, np.zeros(3))
    position = [1.1e-6, 0., 0.] if kind == "translation" else [0., 0., 0.]
    quaternion = world_rotation([1.,0.,0.,0.], 2, 1.1e-6) if kind == "rotation" else [1.,0.,0.,0.]
    assert joint_target(state, position, quaternion).status == "command"
    with pytest.raises(ControllerError, match="no_cartesian_progress"):
        joint_target(replace(state, link6_world_jacobian=np.zeros((6,6))), position, quaternion)


def test_joint_limit_block_remains_rejected_above_numerical_floor():
    state = ArmState(np.ones(6)*2, np.zeros(6), np.zeros(3), [1.,0.,0.,0.], np.eye(6),
                     -np.ones(6)*2, np.ones(6)*2, np.ones(6)*5, np.zeros(3))
    with pytest.raises(ControllerError, match="no_cartesian_progress"):
        joint_target(state, [2e-6,0.,0.], [1.,0.,0.,0.])
