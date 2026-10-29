"""Offline control-contract tests; no model calls or task simulator."""
import json
import math

import numpy as np
import pytest

from gpt_field.action import _quaternion_product, _rotvec_quaternion
from gpt_field.astranomy_agentic_motion import (
    CONTROL_HZ, ShortTargetConfig, ShortTargetExecutor, TargetValidationError,
    measured_pose, rotation_error_rad, validate_target)


STATE = np.array([.1, -.1, .9, math.pi, 0., 0., .04, -.04])


def target_for(state=STATE, *, delta=(0.,0.,0.), rotation=(0.,0.,0.), grip=0):
    current = _rotvec_quaternion(np.asarray(state[3:6]))
    change = _rotvec_quaternion(np.asarray(rotation, dtype=float))
    q = _quaternion_product(change, current)
    return [*(np.asarray(state[:3])+delta), q[3], *q[:3], grip]


def acknowledge(executor, before, after):
    action = executor.next_action(before)
    result = executor.observe_transition(before, action, after)
    return action, result


def test_measured_pose_is_grip_position_hand_quaternion_and_continuous_closure():
    state = STATE.copy()
    state[6:] = [.025,-.015]
    pose = measured_pose(state)
    assert pose[:3] == pytest.approx(state[:3])
    assert pose[3:7] == pytest.approx([0.,1.,0.,0.], abs=1e-10)
    assert pose[7] == pytest.approx(.5)
    assert len(pose) == 8


def test_complete_pose_is_normalized_without_moving_input_or_changing_target():
    target = target_for(delta=(.03,.04,0.), rotation=(0.,0.,.35), grip=True)
    target[3:7] = (np.array(target[3:7])*7).tolist()
    original = list(target)
    normalized = validate_target(STATE,target,5)
    assert target == original
    assert np.linalg.norm(normalized[3:7]) == pytest.approx(1.)
    assert normalized[:3] == pytest.approx(target[:3])
    assert normalized[7] == 1.
    assert np.linalg.norm(rotation_error_rad(normalized[3:7],STATE[3:6])) == pytest.approx(.35)


@pytest.mark.parametrize("steps", [0,6,-1,1.0,True,None])
def test_steps_must_be_explicit_integer_one_through_five(steps):
    with pytest.raises(TargetValidationError,match="steps"):
        ShortTargetExecutor(STATE,target_for(),steps)


@pytest.mark.parametrize("target", [[], [0.]*7, [0.]*9, [None]*8,
    [True,0,0,1,0,0,0,0], [.1,-.1,.9,0,0,0,0,0],
    [.1,-.1,.9,float('nan'),0,0,0,0], [.1,-.1,.9,0,1,0,0,.3],
    [.1,-.1,.9,0,1,0,0,float('nan')], [.1,-.1,.9,0,1,0,0,'1']])
def test_incomplete_or_malformed_targets_are_rejected_without_action(target):
    with pytest.raises(TargetValidationError):
        ShortTargetExecutor(STATE,target,1)


@pytest.mark.parametrize("target", [target_for(delta=(.05001,0,0)),
                                    target_for(delta=(.04,.04,0)),
                                    target_for(rotation=(.35001,0,0))])
def test_outside_local_target_envelope_is_rejected_not_clipped(target):
    with pytest.raises(TargetValidationError,match="exceeds"):
        ShortTargetExecutor(STATE,target,1)


def test_quaternion_sign_does_not_change_angle_limit_or_osc_action():
    target = target_for(rotation=(.08,-.06,.09))
    negative = target.copy()
    negative[3:7] = (-np.array(target[3:7])).tolist()
    first = ShortTargetExecutor(STATE,target,1)
    second = ShortTargetExecutor(STATE,negative,1)
    assert first.next_action(STATE) == pytest.approx(second.next_action(STATE))


def test_same_target_reused_with_measured_feedback_and_no_time_interpolation():
    target = target_for(delta=(.05,0,0))
    executor = ShortTargetExecutor(STATE,target,3)
    first = executor.next_action(STATE)
    # The first requested local step is the full capped 2 cm error, rather
    # than 1/3 of the target or an interpolated trajectory sample.
    assert first[:3] == pytest.approx([.4,0,0])
    after1 = STATE.copy(); after1[0] += .01
    executor.observe_transition(STATE,first,after1)
    second = executor.next_action(after1)
    assert second[:3] == pytest.approx([.4,0,0])
    after2 = STATE.copy(); after2[0] += .047
    executor.observe_transition(after1,second,after2)
    third = executor.next_action(after2)
    assert third[:3] == pytest.approx([.003/.05,0,0])
    executor.observe_transition(after2,third,after2)
    assert executor.target == pytest.approx(target)
    assert executor.done and executor.executed_steps == 3
    assert executor.feedback()['position_error_m'] == pytest.approx(.003)


def test_each_step_caps_cartesian_errors_and_applies_gripper_even_without_arrival():
    target = target_for(delta=(.03,.04,0), rotation=(.21,.14,.07), grip=1)
    executor = ShortTargetExecutor(STATE,target,5)
    for i in range(5):
        action,result = acknowledge(executor,STATE,STATE)
        assert np.linalg.norm(action[:3])*.05 == pytest.approx(.02)
        assert np.linalg.norm(action[3:6])*.5 == pytest.approx(.1)
        assert action[-1] == 1.
        assert result['executed_steps'] == i+1
        assert executor.done == (i == 4)
    feedback = executor.feedback()
    assert feedback['status'] == 'segment_complete'
    assert feedback['position_error_m'] == pytest.approx(.05)
    assert feedback['orientation_error_rad'] > .1
    assert feedback['gripper_command'] == 1
    assert feedback['simulation_time_s'] == pytest.approx(.25)
    assert CONTROL_HZ == 20


def test_arrival_does_not_shorten_requested_segment_or_add_settling_steps():
    executor = ShortTargetExecutor(STATE,target_for(grip=0),5)
    for i in range(5):
        action,_ = acknowledge(executor,STATE,STATE)
        assert action[:3] == (0.,0.,0.) and action[-1] == -1.
        assert executor.done == (i == 4)
    with pytest.raises(RuntimeError,match='finished'):
        executor.next_action(STATE)
    assert executor.feedback()['orientation_error_rad'] == pytest.approx(0.)


def test_one_step_target_is_exactly_one_action_with_immediate_gripper():
    executor = ShortTargetExecutor(STATE,target_for(delta=(.04,0,0),grip=1),1)
    action,_ = acknowledge(executor,STATE,STATE)
    assert action[-1] == 1. and executor.done and executor.executed_steps == 1
    assert executor.feedback()['position_error_m'] == pytest.approx(.04)


def test_zero_rotation_refreshes_native_osc_goal_without_a_macro():
    executor = ShortTargetExecutor(STATE,target_for(),1)
    action,_ = acknowledge(executor,STATE,STATE)
    assert action[3:6] == pytest.approx([2e-9,0,0])
    assert executor.executed_steps == 1


def test_acknowledgement_rejects_stale_state_duplicate_or_rewritten_action():
    executor = ShortTargetExecutor(STATE,target_for(delta=(.01,0,0)),2)
    stale = STATE.copy(); stale[0] += .001
    with pytest.raises(ValueError,match='latest acknowledged'):
        executor.next_action(stale)
    action = executor.next_action(STATE)
    with pytest.raises(RuntimeError,match='acknowledge'):
        executor.next_action(STATE)
    with pytest.raises(ValueError,match='does not match'):
        executor.observe_transition(STATE,np.zeros(7),STATE)
    with pytest.raises(ValueError,match='does not match'):
        executor.observe_transition(stale,action,STATE)
    executor.observe_transition(STATE,action,STATE)
    with pytest.raises(RuntimeError,match='no issued'):
        executor.observe_transition(STATE,action,STATE)
    assert executor.executed_steps == 1


def test_partial_interrupt_keeps_exact_executed_count_and_robot_only_feedback():
    executor = ShortTargetExecutor(STATE,target_for(delta=(.02,0,0)),5)
    after = STATE.copy(); after[0] += .007
    acknowledge(executor,STATE,after)
    executor.interrupt('operator stopped')
    feedback = executor.feedback()
    assert feedback['status'] == 'interrupted' and feedback['executed_steps'] == 1
    assert feedback['requested_steps'] == 5 and feedback['position_error_m'] == pytest.approx(.013)
    assert feedback['measured_pose'][:3] == pytest.approx(after[:3])
    assert set(feedback) == {'status','requested_target','measured_pose','position_error_m',
        'orientation_error_rad','requested_steps','executed_steps','simulation_time_s','control_hz',
        'gripper_command','detail'}
    assert not {'success','reward','contact','task_progress','object_pose'} & set(feedback)
    json.dumps(feedback,allow_nan=False)


def test_large_finite_quaternion_normalizes_but_far_target_is_rejected_cleanly():
    target=target_for()
    target[3:7]=[0.,1e300,0.,0.]
    assert validate_target(STATE,target,1)[3:7] == pytest.approx([0,1,0,0])
    target[0]=1e308
    with pytest.raises(TargetValidationError,match='translation'):
        validate_target(STATE,target,1)


@pytest.mark.parametrize('kwargs',[{'max_step_translation_m':.021},{'max_step_rotation_rad':.101},
    {'max_target_translation_m':.051},{'max_target_rotation_rad':.351},{'max_step_rotation_rad':False},
    {'max_step_translation_m':float('nan')},{'max_target_translation_m':0}])
def test_config_cannot_silently_enlarge_paper_limits(kwargs):
    with pytest.raises(ValueError):
        ShortTargetConfig(**kwargs)


def test_geometric_rejection_is_distinct_from_schema_failure_and_has_zero_action_feedback():
    from gpt_field.astranomy_agentic_motion import TargetLimitError, rejected_feedback
    target = target_for(delta=(.08,0,0),grip=1)
    with pytest.raises(TargetLimitError) as error:
        ShortTargetExecutor(STATE,target,4)
    result = rejected_feedback(STATE,target,4,str(error.value))
    assert result['status'] == 'rejected' and result['executed_steps'] == 0
    assert result['simulation_time_s'] == 0. and result['requested_steps'] == 4
    assert result['position_error_m'] == pytest.approx(.08)
    assert result['measured_pose'] == pytest.approx(measured_pose(STATE))
    assert 'no arm or gripper action' in result['detail']
    json.dumps(result,allow_nan=False)
    with pytest.raises(TargetValidationError) as malformed:
        rejected_feedback(STATE,[0]*7,4,'not valid')
    assert not isinstance(malformed.value,TargetLimitError)
