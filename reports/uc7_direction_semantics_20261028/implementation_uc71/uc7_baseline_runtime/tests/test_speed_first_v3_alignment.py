"""Behavioral integration regressions for local alignment and Field handoff."""

from dataclasses import replace

import numpy as np
import pytest

from gpt_field.action import _rotvec_quaternion
from test_speed_first_v2 import prediction
from test_speed_first_v3 import controller
from test_speed_first_visual import (
    J, MOVING, OFFSET, PATH, TARGET, observation, reference,
)


def calibrated_policy(*, desired_offset=OFFSET, field_goal=(.01, .005, 0.),
                      field_rotation=None):
    direct = prediction(direct=True, phase="align")
    direct["control_context"]["visual_reference"] = reference(desired_offset=desired_offset)
    path = [(0., 0.), *PATH, PATH[-1]]
    direct["direct"]["actions"] = [[*(np.subtract(b, a)/.05), 0., 0., 0., 0., 1.]
                                    for a, b in zip(path[:-1], path[1:])]
    field = prediction(goal=field_goal, phase="align")
    field["field"]["gripper_command"] = "hold"
    if field_rotation is not None:
        field["field"]["orientation_target_xyzw"] = _rotvec_quaternion(np.array(field_rotation)).tolist()
    field["control_context"]["visual_reference"] = reference(PATH[-1], desired_offset=desired_offset)
    policy, transport = controller(direct, field)
    policy.adapter._gripper = 1.
    obs = replace(observation(), prompt=policy.client.task_prompt)
    for xy, expected in zip(path[1:], direct["direct"]["actions"]):
        decision = policy.act(obs)
        assert decision.action == tuple(expected)
        after = replace(observation(xy), prompt=obs.prompt)
        policy.observe_transition(obs, decision, after)
        obs = after
    assert policy.local_feedback["calibration_valid"]
    return policy, transport, obs


def measured_tick(policy, obs):
    decision = policy.act(obs)
    # Robot response is one fifth of the commanded displacement, as in the
    # controller regression. The following image independently renders this
    # measured position; it is not computed from feedback's proposed goal.
    xy = np.array(obs.state[:2]) - [.1, .2] + .01*np.array(decision.action[:2])
    after = replace(observation(xy), prompt=obs.prompt)
    policy.observe_transition(obs, decision, after)
    return after, decision


def test_intermediate_metric_arrival_does_not_force_gpt_refresh_before_pixel_correction():
    policy, transport, obs = calibrated_policy()
    initial_error = policy.local_feedback["pixel_error_norm"]
    executed = 0
    for _ in range(30):
        obs, decision = measured_tick(policy, obs)
        executed += 1
        event = policy.pending_field_event
        if event:
            break
    assert len(transport.calls) == 2
    assert policy.local_goal_updates >= 3
    assert policy.local_feedback["pixel_error_norm"] < initial_error
    assert event is not None
    # Regression: previously step14 emitted goal_reached with error11.045px
    # because the last accepted local waypoint was only4.858mm away. This must
    # instead continue bounded local execution until convergence or its budget.
    if event["reason"] == "goal_reached":
        assert policy.local_feedback["calibration_valid"]
        assert policy.local_feedback["pixel_error_norm"] <= policy.visual.tracker.image_tolerance_px
    else:
        assert event["reason"] == "field_budget_exhausted"
        assert executed == 30


def test_sub_2_5mm_visual_correction_can_generate_field_motion():
    desired_offset = MOVING + J @ PATH[-1] + [5., 0.] - TARGET
    policy, transport, obs = calibrated_policy(desired_offset=desired_offset, field_goal=(0., 0., 0.))
    correction = np.array(policy.local_feedback["goal_correction_xy_m"])
    assert 0.001 < np.linalg.norm(correction) < .0025
    motions = []
    for _ in range(3):
        obs, decision = measured_tick(policy, obs)
        motions.append(float(np.linalg.norm(decision.action[:2])))
        if policy.pending_field_event:
            break
    assert len(transport.calls) == 2
    assert policy.local_goal_updates >= 1
    assert max(motions) > 0.
    if policy.pending_field_event:
        assert policy.local_feedback["pixel_error_norm"] <= policy.visual.tracker.image_tolerance_px


def test_rotating_nominal_field_keeps_normal_motion_and_disables_planar_proposal():
    policy, _, obs = calibrated_policy(field_rotation=(0., 0., .1))
    obs, decision = measured_tick(policy, obs)
    assert abs(decision.action[5]) > 0.
    assert np.linalg.norm(decision.action[:2]) > 0.
    assert policy.local_goal_updates == 0
    assert policy.local_feedback["status"] == "nominal_goal_changes_orientation"
    assert not policy.local_feedback["proposal_available"]


def test_invalid_feedback_does_not_mutate_a_valid_visual_epoch():
    policy, _, obs = calibrated_policy()
    decision = policy.act(obs)
    previous = policy.visual.diagnostics()
    wrong_before = replace(obs, state=(obs.state[0] + .001, *obs.state[1:]))
    lost = replace(observation(PATH[-1], missing="moving"), prompt=obs.prompt)
    with pytest.raises(ValueError, match="exact pending"):
        policy.observe_transition(wrong_before, decision, lost)
    assert policy.visual.diagnostics() == previous
    assert policy._pending[1] is decision
    policy.observe_transition(obs, decision, obs)


def test_executed_open_command_invalidates_before_fingers_have_moved():
    direct = prediction(direct=True, phase="align")
    direct["control_context"]["visual_reference"] = reference()
    path = [(0., 0.), *PATH, PATH[-1]]
    direct["direct"]["actions"] = [[*(np.subtract(b, a)/.05), 0., 0., 0., 0., 1.]
                                    for a, b in zip(path[:-1], path[1:])]
    direct["direct"]["actions"][-1][-1] = -1.
    policy, transport = controller(direct)
    policy.adapter._gripper = 1.
    obs = replace(observation(), prompt=policy.client.task_prompt)
    for index, (xy, expected) in enumerate(zip(path[1:], direct["direct"]["actions"])):
        decision = policy.act(obs)
        assert decision.action == tuple(expected)
        # Deliberately leave observed fingers unchanged after the open command;
        # invalidation must follow that executed intent immediately.
        after = replace(observation(xy), prompt=obs.prompt)
        policy.observe_transition(obs, decision, after)
        obs = after
        if index == 3:
            assert policy.local_feedback["calibration_valid"]
    assert len(transport.calls) == 1
    assert policy.local_feedback["status"] == "executed_open_command"
    assert not policy.local_feedback["calibration_valid"]
