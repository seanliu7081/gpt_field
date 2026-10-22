"""Generic robot motion tests; no model requests or task simulations."""
import math
from pathlib import Path

import numpy as np
import pytest

from gpt_field.robodojo_direct_motion import (
    AbsoluteMotionExecutor, MotionConfig, orientation_matrix, resolve_target, robot_state)
from gpt_field.robodojo_direct_sim import IKResult, PandaKinematics, rotation_matrix, rotation_vector


def state_at(xyz=(.1, -.1, .9), *, roll=0., pitch=0., yaw=0., grip=1.):
    R = orientation_matrix(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw)
    return np.r_[xyz, rotation_vector(R), .04*grip, -.04*grip]


class FakeKinematics:
    def __init__(self, reject=False):
        self.reject = reject
        self.checked = []

    def bind(self, state, joints):
        self.bound = (np.array(state), np.array(joints))
        return self

    def solve(self, position, rotation, seed):
        self.checked.append((np.array(position), np.array(rotation)))
        return IKResult(not self.reject, tuple(np.zeros(7)), 0. if not self.reject else .3, 0., 1)


def kinematic_step(state, action):
    after = np.array(state, copy=True)
    after[:3] += np.asarray(action[:3])*.05
    after[3:6] = rotation_vector(rotation_matrix(np.asarray(action[3:6])*.5) @ rotation_matrix(state[3:6]))
    after[6] = np.clip(after[6]-.004*action[6], 0., .04)
    after[7] = -after[6]
    return after


def run_motion(motion, state, step=kinematic_step):
    records = []
    while not motion.done:
        action = motion.next_action(state)
        after = step(state, action)
        record = motion.observe_transition(state, action, after)
        records.append((action, record))
        state = after
        assert len(records) <= motion.config.max_arm_steps+motion.config.gripper_steps
    return state, records


@pytest.mark.parametrize("angles", [(0,0,0), (25,-30,70), (-70,25,-170), (90,34,78), (-90,34,78)])
def test_named_euler_roundtrip_and_downward_reference(angles):
    roll, pitch, yaw = angles
    state = state_at(roll=roll, pitch=pitch, yaw=yaw)
    exported = robot_state(state)
    R = orientation_matrix(**{k: exported[k] for k in ("roll_deg", "pitch_deg", "yaw_deg")})
    np.testing.assert_allclose(R, rotation_matrix(state[3:6]), atol=1e-8)
    np.testing.assert_allclose(orientation_matrix()[:,2], [0,0,-1], atol=1e-10)
    # RoboDojo's pitch is world X, rather than conventional pitch-about-Y.
    np.testing.assert_allclose(orientation_matrix(pitch_deg=90)[:,2], [0,1,0], atol=1e-10)
    np.testing.assert_allclose(orientation_matrix(roll_deg=90)[:,2], [-1,0,0], atol=1e-10)


def test_partial_targets_retain_measured_values_not_previous_target():
    state = state_at(roll=20, pitch=-8, yaw=40, grip=.4)
    target = resolve_target(state, {"z":1.02, "x":None, "note":"move above"})
    measured = robot_state(state)
    assert target == {**measured, "z":1.02}


@pytest.mark.parametrize("arguments", [{}, {"note":"only"}, {"x":True}, {"x":float("nan")},
                                         {"gripper":1.1}, {"target":{"x":0.}}, {"pitch_deg":"3"}])
def test_invalid_targets_rejected(arguments):
    with pytest.raises(ValueError):
        resolve_target(state_at(), arguments)


def test_ik_rejection_emits_no_physical_action_or_gripper_change():
    state = state_at(grip=.3)
    motion = AbsoluteMotionExecutor(state, {"x":.9, "gripper":1.}, np.zeros(7), FakeKinematics(True),
                                    previous_gripper=0.)
    assert motion.done and motion.feedback(state)["status"] == "rejected"
    assert motion.current_gripper == 0. and not motion.gripper_applied
    assert motion.executed_steps == 0
    with pytest.raises(RuntimeError, match="finished"):
        motion.next_action(state)


def test_full_path_checked_before_first_action_and_native_commands_bounded():
    state, kin = state_at(), FakeKinematics()
    motion = AbsoluteMotionExecutor(state, {"x":.3, "yaw_deg":90.}, np.zeros(7), kin)
    assert len(kin.checked) >= 10
    assert motion.executed_steps == 0
    np.testing.assert_allclose(kin.checked[-1][0], [.3,-.1,.9])
    after, records = run_motion(motion, state)
    assert motion.feedback(after)["status"] == "reached"
    assert all(np.linalg.norm(a[:3])*.05 <= .012+1e-12 for a, _ in records)
    assert all(np.linalg.norm(a[3:6])*.5 <= math.radians(4)+1e-12 for a, _ in records)
    assert all(len(a) == 7 and max(abs(v) for v in a) <= 1 for a, _ in records)
    assert motion.feedback(after)["position_error_m"] <= .004


def test_gripper_applied_only_after_arm_arrival_and_close_latch_persists():
    state = state_at(grip=.4)
    motion = AbsoluteMotionExecutor(state, {"x":.2, "gripper":1.}, np.zeros(7), FakeKinematics(),
                                    previous_gripper=0.)
    after, records = run_motion(motion, state)
    arm = [a for a,r in records if r["phase"] == "arm"]
    fingers = [a for a,r in records if r["phase"] == "gripper"]
    assert arm and fingers and all(a[6] == 1. for a in arm)
    assert all(a[6] == -1. for a in fingers)
    assert motion.current_gripper == 1. and motion.feedback(after)["gripper_applied"]
    assert len(fingers) == 30


def test_tracking_timeout_reports_actual_pose_and_skips_new_gripper():
    state = state_at(grip=.4)
    cfg = MotionConfig(max_arm_steps=10, tracking_extra_steps=3)
    motion = AbsoluteMotionExecutor(state, {"x":.12,"gripper":1.}, np.zeros(7), FakeKinematics(),
                                    previous_gripper=0., config=cfg)
    after, records = run_motion(motion, state, step=lambda before, action: before.copy())
    feedback = motion.feedback(after)
    assert feedback["status"] == "incomplete" and feedback["position_error_m"] == pytest.approx(.02)
    assert not feedback["gripper_applied"] and motion.current_gripper == 0.
    assert all(a[6] == 1. for a, _ in records)
    assert all(r["phase"] == "arm" for _,r in records)


def test_omitted_gripper_keeps_commanded_close_despite_measured_object_width():
    state = state_at(grip=.4)
    motion = AbsoluteMotionExecutor(state, {"z":.91}, np.zeros(7), FakeKinematics(), previous_gripper=0.)
    assert motion.target["gripper"] == 0.
    assert motion.feedback(state)["requested_target"]["gripper"] == 0.
    assert motion.feedback(state)["measured_state"]["gripper"] == pytest.approx(.4)
    after, records = run_motion(motion, state)
    assert all(a[6] == 1. for a,_ in records)
    assert motion.current_gripper == 0. and not motion.gripper_applied


def test_feedback_requires_acknowledged_actions_and_contains_only_robot_fields():
    state = state_at()
    motion = AbsoluteMotionExecutor(state, {"x":.11}, np.zeros(7), FakeKinematics())
    a = motion.next_action(state)
    with pytest.raises(RuntimeError, match="acknowledge"):
        motion.next_action(state)
    with pytest.raises(ValueError, match="does not match"):
        motion.observe_transition(state, np.zeros(7), state)
    motion.observe_transition(state, a, kinematic_step(state,a))
    motion.interrupt()
    assert motion.feedback(state)["status"] == "user_stop"
    assert set(motion.feedback(state)) == {"status","requested_target","measured_state","position_error_m",
        "orientation_error_deg","executed_steps","detail","gripper_applied","gripper_error"}


def panda_fixture(tmp_path):
    # Static robot-only joint origins from the Panda asset; no scene/object data.
    positions = [(0,0,.333),(0,0,0),(0,-.316,0),(.0825,0,0),(-.0825,.384,0),(0,0,0),(.088,0,0)]
    quats = [(1,0,0,0),(.707107,-.707107,0,0),(.707107,.707107,0,0),(.707107,.707107,0,0),
             (.707107,-.707107,0,0),(.707107,.707107,0,0),(.707107,.707107,0,0)]
    ranges = [(-2.8973,2.8973),(-1.7628,1.7628),(-2.8973,2.8973),(-3.0718,-.0698),
              (-2.8973,2.8973),(-.0175,3.7525),(-2.8973,2.8973)]
    text = '<mujoco><worldbody><body name="base">'
    for i,(p,q,bounds) in enumerate(zip(positions,quats,ranges),1):
        text += (f'<body name="link{i}" pos="{" ".join(map(str,p))}" quat="{" ".join(map(str,q))}">'
                 f'<joint name="joint{i}" axis="0 0 1" range="{bounds[0]} {bounds[1]}"/>')
    text += '<body name="right_hand" pos="0 0 .1065" quat=".924 0 0 -.383"/>'
    text += '</body>'*8+'</worldbody></mujoco>'
    robot = tmp_path/'robot.xml'; robot.write_text(text)
    gripper = tmp_path/'gripper.xml'
    gripper.write_text('<mujoco><worldbody><body name="right_gripper" quat=".707107 0 0 -.707107">'
                      '<body name="eef" pos="0 0 .097"><site name="grip_site"/></body>'
                      '</body></worldbody></mujoco>')
    return PandaKinematics.from_xml(robot, gripper)


Q = np.array([0., -.4, 0., -2., 0., 1.6, .8])


def test_robot_xml_fk_jacobian_matches_numerical_differentiation(tmp_path):
    kin = panda_fixture(tmp_path)
    T,J = kin.forward(Q)
    for i in range(7):
        delta = np.zeros(7); delta[i]=1e-6
        perturbed,_ = kin.forward(Q+delta)
        numeric = np.r_[(perturbed[:3,3]-T[:3,3])/1e-6,
                        rotation_vector(perturbed[:3,:3] @ T[:3,:3].T)/1e-6]
        np.testing.assert_allclose(J[:,i], numeric, atol=1e-6)
    assert kin.metadata['grip_offset_in_right_hand_m'] == [0.,0.,.097]


def test_robot_only_binding_ik_reachable_goal_and_unreachable_rejection(tmp_path):
    kin = panda_fixture(tmp_path)
    state = state_at()
    bound = kin.bind(state,Q)
    pose,_ = bound.forward(Q)
    np.testing.assert_allclose(pose[:3,3], state[:3], atol=1e-12)
    np.testing.assert_allclose(pose[:3,:3], rotation_matrix(state[3:6]), atol=1e-12)
    reference,_ = bound.forward(Q+np.array([.05,.03,-.02,-.08,.03,.02,-.04]))
    result = bound.solve(reference[:3,3], reference[:3,:3], Q)
    assert result.ok and result.position_error_m < .0015 and result.orientation_error_deg < 1.
    failed = bound.solve([10.,10.,10.], reference[:3,:3], Q)
    assert not failed.ok and failed.position_error_m > 10.
    assert np.all(np.array(failed.joints) >= kin.low) and np.all(np.array(failed.joints) <= kin.high)
    # Bind is isolated: probing cannot change the source model's world pose.
    np.testing.assert_array_equal(kin.world_from_base,np.eye(4))
