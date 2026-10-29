"""Cross-layer V8 field reuse, measured hold and auditable rotation vetoes."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.visual_field_policy_v8 import VisualFieldPolicyV8
from gpt_field.visual_field_v8 import MetricFieldExecutorV8
from test_visual_field_v6_policy import Client, observation, prediction


def context(purpose="task_translation", *, neighbor=False):
    return {"purpose": purpose, "tool_external_roi_uv": [.3, .2, .5, .4],
            "nearby_obstacles": ([{"id": "fixture", "external_roi_uv": [.57, .25, .67, .45]}]
                                 if neighbor else []),
            "obstacle_assessment": "present" if neighbor else "none_relevant"}


def value(controller="visual_field", **kwargs):
    result = prediction(controller, **kwargs)
    result["motion_context"] = context("contact_correction" if controller == "direct" else "task_translation")
    return result


def test_one_field_intent_has_sustained_rgb_validated_motion_and_bounded_pose_hold():
    client = Client(value(goal=(.08, 0., 0.)))
    policy = VisualFieldPolicyV8(client)
    obs, offset = observation(), np.zeros(3)
    for _ in range(40):
        decision = policy.act(obs)
        assert decision.diagnostics["policy_version"] == "visual-field-policy-v8"
        assert np.linalg.norm(decision.action[3:6])*.5 <= .02+1e-12
        assert decision.action[6] == 0.
        offset += np.asarray(decision.action[:3])*.05*.25
        after = observation(offset)
        row = policy.observe_transition(obs, decision, after)
        obs = after
        if policy.refresh_reason:
            assert policy.refresh_reason == "measured_local_goal_arrival"
            break
    assert client.calls == 1 and policy.field_effective_actions > 20
    assert offset[0] > .06
    assert "response_evidence" in row["visual"]
    assert not row["task_success_certified"]


def test_measured_rotation_drift_is_corrected_without_model_refresh_or_translation_rewrite():
    client = Client(value(goal=(.08, 0., 0.)))
    policy = VisualFieldPolicyV8(client)
    obs = observation()
    first = policy.act(obs)
    offset = np.asarray(first.action[:3])*.05*.25
    after = observation(offset)
    after = replace(after, state=(*after.state[:3], 0., 0., .01, *after.state[6:]))
    policy.observe_transition(obs, first, after)
    second = policy.act(after)
    assert second.diagnostics["selection"]["accepted"]
    assert second.action[5] < 0. and np.linalg.norm(second.action[3:6])*.5 <= .02
    assert second.action[0] > 0. and second.action[6] == 0.
    assert client.calls == 1
    # Acknowledgement includes the actual corrected command, not the original
    # zero-rotation base-executor command hidden from the real robot.
    policy.observe_transition(after, second, after)


def test_accepted_task_direct_chunk_executes_exactly_the_five_model_commands():
    commands = [[.01*i, -.02, .01, 0., .01*i, 0., -1.] for i in range(5)]
    pred = value("direct", phase="grasp", actions=commands)
    pred["visual_references"] = {"external": None, "wrist": None}
    client = Client(pred); policy = VisualFieldPolicyV8(client); obs = observation()
    for command in commands:
        decision = policy.act(obs)
        assert list(decision.action) == command
        assert decision.diagnostics["projected_rotation_risk"]["accepted"]
        policy.observe_transition(obs, decision, obs)
    assert client.calls == 1 and policy.rotation_vetoes == 0
    assert policy.gripper_actions == 0


@pytest.mark.parametrize("reference_only", [False, True])
def test_rotation_veto_logs_requested_action_and_discards_entire_chunk_before_recovery(reference_only):
    commands = [[0., 0., 0., 0., 0., .16, 1.]]*5
    pred = value("direct", phase="recover", actions=commands)
    pred["visual_references"] = {"external": None, "wrist": None}
    pred["motion_context"] = context("reference_repair" if reference_only else "task_reorientation", neighbor=True)
    client = Client(pred); policy = VisualFieldPolicyV8(client); obs = observation()
    rejected = policy.act(obs)
    assert rejected.action == (0., 0., 0., MetricFieldExecutorV8.OSC_GOAL_REFRESH_RAD/.5, 0., 0., 0.)
    assert rejected.diagnostics["rejection_hold"]["requested_motion_executed"] is False
    assert rejected.diagnostics["rejected_model_action"] == commands[0]
    assert not rejected.diagnostics["selection"]["accepted"]
    assert policy.chunk == [] and policy.events[-1]["step"] == 0
    row = policy.observe_transition(obs, rejected, obs)
    assert policy.attempts == [] and not row["task_success_certified"]
    assert policy.rotation_vetoes == 1
    replacement = [[.02, 0., 0., 0., 0., 0., -1.]]*5
    client.value = value("direct", phase="grasp", actions=replacement)
    client.value["visual_references"] = {"external": None, "wrist": None}
    for command in replacement:
        decision = policy.act(obs)
        assert list(decision.action) == command
        policy.observe_transition(obs, decision, obs)
    assert client.calls == 2 and policy.rotation_vetoes == 1
    assert client.blocks[0]["steps"] == 1
    assert client.blocks[0]["end_visual_feedback"]["projected_rotation_risk"]["accepted"] is False


def test_explicit_gripper_settling_does_not_require_unrelated_rotation_geometry():
    pred = value("gripper", phase="grasp")
    pred["visual_references"] = {"external": None, "wrist": None}
    pred["motion_context"].update(tool_external_roi_uv=None, obstacle_assessment="unobservable")
    client = Client(pred); policy = VisualFieldPolicyV8(client); obs = observation()
    for index in range(17):
        decision = policy.act(obs)
        assert decision.action[6] == 1. and np.linalg.norm(decision.action[:3]) == 0.
        gap = max(.012, .04-.002*(index+1))
        after = replace(obs, state=(*obs.state[:6], gap, -gap))
        policy.observe_transition(obs, decision, after)
        obs = after
    assert policy.refresh_reason == "gripper_settled" and client.calls == 1
    assert policy.rotation_vetoes == 0 and policy.gripper_actions == 17


def test_declared_unknown_scene_has_a_finite_field_budget_even_after_rgb_response_validates():
    pred = value(goal=(.08, 0., 0.))
    pred["motion_context"]["obstacle_assessment"] = "unobservable"
    original = copy.deepcopy(pred)
    client = Client(pred); policy = VisualFieldPolicyV8(client)
    obs, offset = observation(), np.zeros(3)
    for _ in range(45):
        decision = policy.act(obs)
        assert np.linalg.norm(decision.action[:3])*.05 <= .005+1e-12
        offset += np.asarray(decision.action[:3])*.05*.25
        after = observation(offset)
        row = policy.observe_transition(obs, decision, after)
        obs = after
        if policy.refresh_reason:
            break
    assert client.calls == 1 and np.linalg.norm(offset) <= .015+1e-9
    assert policy.refresh_reason in ("intent_travel_budget", "prospective_travel_budget", "field_endpoint_reached")
    assert policy.local.intent["max_travel_m"] == .015
    assert policy.local.intent["goal_offset_m"] == pred["field_intent"]["goal_offset_m"]
    assert policy.local.intent["obstacles"] == pred["field_intent"]["obstacles"]
    assert row["scene_execution_limits"]["goal_and_regions_preserved"]
    assert policy.prediction == original
