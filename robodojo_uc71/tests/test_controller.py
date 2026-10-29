from dataclasses import replace
import math

import numpy as np
import pytest

from robodojo_uc71.controller import (ArmState, ControllerConfig, ControllerError,
    DualArmController, joint_target, measured_tcp_pose, orientation_error_world,
    tcp_world_jacobian)


def arm(**changes):
    base = ArmState(np.zeros(6), np.zeros(6), np.zeros(3), [1.,0.,0.,0.],
                    np.eye(6), -np.ones(6)*2, np.ones(6)*2,
                    np.ones(6)*5, np.zeros(3))
    return replace(base, **changes)


def test_identity_jacobian_moves_in_requested_direction_with_candidate_physical_caps():
    target = joint_target(arm(), [1.,0.,0.], [1.,0.,0.,0.])
    assert target.calibration_status == "uncalibrated_candidate"
    assert target.status == "command"
    assert target.joint_positions[0] == pytest.approx(.016/(1+.02**2))
    assert target.joint_positions[1:] == pytest.approx((0.,)*5)
    contact = joint_target(arm(), [-1.,0.,0.], [1.,0.,0.,0.], profile="contact")
    assert contact.joint_positions[0] == pytest.approx(-.004/(1+.02**2))


def test_world_tcp_offset_and_jacobian_cross_product_sign():
    # Link6 offset +X rotated90deg aboutZ gives world +Y. A world +Z
    # angular velocity must then give TCP linear velocity -X.
    state = arm(link6_quaternion_wxyz=[math.sqrt(.5),0,0,math.sqrt(.5)],
                tcp_offset_link6=[.2,0,0])
    position, _ = measured_tcp_pose(state)
    assert position == pytest.approx([0,.2,0], abs=1e-12)
    jacobian = tcp_world_jacobian(state)
    assert jacobian[:3,5] == pytest.approx([-.2,0,0], abs=1e-12)
    assert jacobian[3:,5] == pytest.approx([0,0,1])


def test_shortest_quaternion_error_world_axis_and_antipodal_equivalence():
    a = math.radians(170)/2
    b = math.radians(-170)/2
    current, target = [math.cos(a),0,0,math.sin(a)], [math.cos(b),0,0,math.sin(b)]
    error = orientation_error_world(current, target)
    assert error == pytest.approx([0,0,math.radians(20)])
    assert orientation_error_world(current, -np.array(target)) == pytest.approx(error)
    assert orientation_error_world(current, -np.array(current)) == pytest.approx([0,0,0])


def test_joint_lead_live_velocity_and_position_limits():
    q = np.zeros(6); q[0] = .499
    upper = np.ones(6)*2; upper[0] = .5
    velocity = np.ones(6)*5; velocity[1] = .01
    state = arm(joint_positions=q, joint_upper=upper, joint_velocity_limits=velocity)
    target = joint_target(state, [1,1,0], [math.cos(.5),0,0,math.sin(.5)])
    delta = np.array(target.joint_positions)-q
    assert delta[0] <= .001+1e-12
    assert abs(delta[1]) <= .0004+1e-12
    assert abs(delta[5]) <= .05+1e-12
    assert np.linalg.norm(target.predicted_translation_m) <= .016+1e-12
    assert np.linalg.norm(target.predicted_rotation_rad) <= .08+1e-12


def test_blocked_or_rank_zero_ik_is_explicit_failure_not_silent_hold():
    with pytest.raises(ControllerError, match="no_cartesian_progress"):
        joint_target(arm(link6_world_jacobian=np.zeros((6,6))), [1,0,0], [1,0,0,0])
    state = arm(joint_positions=np.ones(6)*2)
    with pytest.raises(ControllerError, match="no_cartesian_progress"):
        joint_target(state, [1,0,0], [1,0,0,0])
    done = joint_target(arm(), [0,0,0], [1,0,0,0])
    assert done.status == "at_target"


@pytest.mark.parametrize("changes", [
    {"joint_positions":[float("nan")]*6},
    {"joint_positions":["0"]*6},
    {"joint_positions":[False]*6},
    {"joint_velocities":[float("inf")]*6},
    {"link6_world_jacobian":np.eye(7)},
    {"joint_velocity_limits":[0]*6},
    {"joint_lower":[3]*6},
    {"link6_quaternion_wxyz":[0,0,0,0]},
    {"joint_positions":[3]*6},
])
def test_invalid_measured_schema_and_limits_fail_closed(changes):
    with pytest.raises(ControllerError):
        joint_target(arm(**changes), [1,0,0], [1,0,0,0])


def test_inactive_arm_holds_measured_handoff_q_and_latches_requested_gripper():
    states = {"left":arm(), "right":arm(joint_positions=np.ones(6)*.2)}
    controller = DualArmController(gripper_commands={"left":1., "right":.25})
    controller.handoff("left", states)
    moved = {**states, "right":arm(joint_positions=np.ones(6)*.21)}
    command = controller.command(moved, [.1,0,0], [1,0,0,0], gripper_commands={"left":0.})
    assert command.joint_positions["right"] == pytest.approx([.2]*6)
    assert command.gripper_commands == {"left":0., "right":.25}
    # New handoff captures the actual left q, not its previous motion command.
    newer = {"left":arm(joint_positions=np.ones(6)*.3), "right":moved["right"]}
    controller.handoff("right", newer)
    command = controller.command(newer, [.1,0,0], [1,0,0,0])
    assert command.joint_positions["left"] == pytest.approx([.3]*6)
    assert command.gripper_commands == {"left":0., "right":.25}


def test_uncalibrated_configuration_cannot_claim_calibrated():
    with pytest.raises(ControllerError):
        joint_target(arm(), [1,0,0], [1,0,0,0], config=ControllerConfig(calibration_status="calibrated"))
