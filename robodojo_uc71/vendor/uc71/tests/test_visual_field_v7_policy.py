"""Cross-layer V7 regressions with actual RGB tracking and measured test motion.

Textures form a deterministic test plant, not extra policy inputs. No model or
simulator is called, and local arrival/settling never stand in for task success.
"""
import copy
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.visual_field_policy_v7 import VisualFieldPolicyV7
from test_visual_field_v6_policy import (
    Client, MOVING, TARGET, TOOL, observation, prediction, projection,
)


def move_once(policy, obs, offset, *, slip=False, payload=False, gain=.25):
    decision = policy.act(obs)
    offset = offset+np.asarray(decision.action[:3])*.05*gain
    after = observation(offset, slip=slip, payload=payload)
    row = policy.observe_transition(obs, decision, after)
    return after, offset, decision, row


def block_static_reference(policy, *, payload=False):
    offset = np.zeros(3)
    obs = observation(offset, slip=True, payload=payload)
    for _ in range(20):
        obs, offset, decision, row = move_once(policy, obs, offset, slip=True, payload=payload, gain=1.)
        if policy.refresh_reason:
            break
    else:
        pytest.fail("Stationary image evidence must not permit unbounded measured motion")
    health = row["visual"]["reference_health"]
    assert health["repair_required"] and health["unvalidated_travel_m"] > 0
    assert health["status"].startswith("controlled_reference_")
    assert row["controller_handoff"]["reference_repair_required"]
    return obs, offset, copy.deepcopy(health)


def test_valid_controlled_rgb_reference_reuses_one_field_beyond_cold_start_allowance():
    client = Client(prediction(goal=(.08, 0., 0.)))
    policy = VisualFieldPolicyV7(client)
    obs, offset = observation(), np.zeros(3)
    count = 0
    for _ in range(45):
        obs, offset, decision, row = move_once(policy, obs, offset)
        if decision.diagnostics["selection"]["accepted"]:
            count += 1
        assert decision.action[3:] == (0.,)*4
        assert decision.diagnostics["policy_version"] == "visual-field-policy-v7"
        if policy.refresh_reason:
            assert policy.refresh_reason == "measured_local_goal_arrival"
            break
    assert client.calls == 1 and count > 20
    assert offset[0] > .06
    assert row["visual"]["reference_health"]["unvalidated_travel_m"] < .015
    assert policy.gripper_actions == 0 and not policy.monitor.attachment_evidence


@pytest.mark.parametrize("update", ["retain", "rename_same_seed"])
def test_static_correspondence_cannot_renew_travel_by_request_retain_or_renaming(update):
    client = Client(prediction(goal=(.08, 0., 0.)))
    policy = VisualFieldPolicyV7(client)
    obs, offset, rejected = block_static_reference(policy)
    client.value = prediction(goal=(.06, 0., 0.), retain=update == "retain")
    if update == "rename_same_seed":
        client.value["visual_references"]["external"]["moving_id"] = "new_name_same_pixels"
    for _ in range(3):
        decision = policy.act(obs)
        assert decision.action == (0.,)*7
        assert not decision.diagnostics["selection"]["accepted"]
        row = policy.observe_transition(obs, decision, obs)
        health = row["visual"]["reference_health"]
        assert health["repair_required"]
        assert health["correspondence_id"] == rejected["correspondence_id"]
        assert health["unvalidated_travel_m"] == pytest.approx(rejected["unvalidated_travel_m"])
        assert row["controller_handoff"]["reference_repair_required"]
    assert client.calls == 4
    assert np.linalg.norm(offset) <= .015+1e-9


def test_explicitly_corrected_moving_part_recovers_field_after_stationary_reference_failure():
    client = Client(prediction(goal=(.08, 0., 0.)))
    policy = VisualFieldPolicyV7(client)
    # The first model seed is a static image patch; the other visible rigid
    # patch follows measured robot motion. No ground-truth label is supplied.
    obs, offset, rejected = block_static_reference(policy, payload=True)
    corrected = prediction(goal=(.06, 0., 0.))
    ext = corrected["visual_references"]["external"]
    ext["seed"]["moving_point_uv"] = ((TOOL+projection(offset))/223).tolist()
    ext["goal_offset_uv"] = ((TOOL+projection(offset+np.array([.06, 0., 0.]))-TARGET)/223).tolist()
    client.value = corrected
    start = offset.copy()
    for _ in range(25):
        obs, offset, decision, row = move_once(policy, obs, offset, payload=True, slip=True)
        if policy.refresh_reason:
            assert policy.refresh_reason == "measured_local_goal_arrival"
            break
        assert decision.diagnostics["selection"]["accepted"]
    health = row["visual"]["reference_health"]
    assert client.calls == 2
    assert health["correspondence_id"] != rejected["correspondence_id"]
    assert not health["repair_required"]
    assert np.linalg.norm(offset-start) > .03
    assert health["recent_window"]["predictive_response_validated"]
    assert not policy.monitor.attachment_evidence


@pytest.mark.parametrize("command,sign", [("close", 1.), ("open", -1.)])
def test_local_gripper_still_executes_seventeen_measured_steps_from_one_model_intent(command, sign):
    value = prediction("gripper", phase="grasp", gripper=command)
    value["visual_references"] = {"external": None, "wrist": None}
    client = Client(value)
    policy = VisualFieldPolicyV7(client)
    initial = .04 if command == "close" else .012
    obs = observation(fingers=(initial, -initial))
    for index in range(17):
        decision = policy.act(obs)
        assert decision.action == (0.,)*6+(sign,)
        assert decision.diagnostics["selection"]["accepted"]
        aperture = (max(.012, .04-.002*(index+1)) if command == "close"
                    else min(.04, .012+.002*(index+1)))
        after = replace(obs, state=(*obs.state[:6], aperture, -aperture))
        row = policy.observe_transition(obs, decision, after)
        obs = after
        assert policy.refresh_reason is None if index < 16 else policy.refresh_reason == "gripper_settled"
    assert client.calls == 1 and policy.gripper_actions == 17
    assert row["gripper_execution"]["executed_actions"] == 17
    assert not row["gripper_execution"]["grasp_certified"]
    assert not policy.monitor.attachment_evidence


def test_direct_five_requested_contact_actions_remain_exact_without_automatic_gripper_routing():
    commands = [[.01*i, -.02, .01, .0, .01*i, .0, -1.] for i in range(5)]
    value = prediction("direct", phase="grasp", actions=commands)
    value["visual_references"] = {"external": None, "wrist": None}
    client = Client(value)
    policy = VisualFieldPolicyV7(client)
    obs = observation()
    for command in commands:
        decision = policy.act(obs)
        assert list(decision.action) == command
        policy.observe_transition(obs, decision, obs)
    assert client.calls == 1
    assert policy.gripper_actions == 0 and policy.gripper.intent is None
    assert policy.attempts == []


def test_terminal_field_stall_never_closes_and_allows_explicit_direct_contact_correction():
    client = Client(prediction(goal=(0., 0., -.03), motion_axes="z"))
    policy = VisualFieldPolicyV7(client)
    obs = observation()
    for _ in range(5):
        decision = policy.act(obs)
        assert decision.action[6] == 0.
        row = policy.observe_transition(obs, decision, obs)
    assert policy.refresh_reason == "measured_motion_stall"
    assert policy.gripper_actions == 0 and policy.attempts == []
    assert not row["task_success_certified"]
    client.value = prediction("direct", phase="grasp", actions=[[.02, 0., 0., 0., 0., 0., -1.]]*5)
    client.value["visual_references"] = {"external": None, "wrist": None}
    for _ in range(5):
        decision = policy.act(obs)
        assert decision.diagnostics["controller"] == "direct"
        assert decision.action == (.02, 0., 0., 0., 0., 0., -1.)
        policy.observe_transition(obs, decision, obs)
    assert client.calls == 2 and policy.gripper_actions == 0
