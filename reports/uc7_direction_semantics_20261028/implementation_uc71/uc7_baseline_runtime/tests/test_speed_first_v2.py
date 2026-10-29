"""Independent integration tests for explicit operations, provenance and routing."""

import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from gpt_field.astra import APIError, AstraConfig, SchemaError
from gpt_field.hybrid_subscription import HYBRID_GRID
from gpt_field.speed_first_v2 import (
    ChatGPTPlanSpeedFirstV2Client, SpeedFirstV2Policy,
    SPEED_FIRST_V2_VERSION, speed_first_v2_instructions, speed_first_v2_schema,
)
from test_distance_policy import intent
from test_field_history import FakeTransport, envelope, observation


def prediction(**kwargs):
    return {**intent(**kwargs), "gripper_operation": None}


def operation(command="close"):
    raw = prediction(direct=True, phase="grasp" if command == "close" else "release")
    raw["direct"] = None
    raw["gripper_operation"] = {"command": command}
    return raw


def controller(*responses, **kwargs):
    transport = FakeTransport(*(envelope(raw, id=f"v2-{i}", model="gpt-6-astra",
                                        reasoning={"effort": "medium"})
                                for i, raw in enumerate(responses)))
    client = ChatGPTPlanSpeedFirstV2Client(token_provider=lambda: "offline-oauth-v2",
                                          transport=transport, **kwargs)
    return SpeedFirstV2Policy(client), transport


def tick(policy, before, *, fingers=None, actual_position=None, rotvec=None, fraction=1.):
    decision = policy.act(before)
    state = list(before.state)
    state[:3] = (np.asarray(state[:3]) + .05 * np.asarray(decision.action[:3]) * fraction
                 if actual_position is None else actual_position)
    if fingers is not None:
        state[6:8] = fingers
    if rotvec is not None:
        state[3:6] = rotvec
    after = replace(before, state=tuple(state))
    feedback = policy.observe_transition(before, decision, after)
    return after, decision, feedback


def records(messages):
    found = []
    for message in messages:
        for part in message.get("content", []):
            if part.get("type") in ("input_text", "output_text"):
                try:
                    found.append(json.loads(part["text"]))
                except json.JSONDecodeError:
                    pass
    return found


def test_mixed_field_close_operation_direct_five_field_recovery_without_extra_requests():
    direct = prediction(direct=True, phase="verify")
    direct["direct"]["actions"] = [[.01 * i, -.02, .03, .04, 0., -.01, 1.] for i in range(5)]
    recovery = prediction(goal=(0., -.03, 0.), phase="recover")
    recovery["field"]["gripper_command"] = "hold"
    policy, transport = controller(prediction(), operation(), direct, recovery)
    obs, field, _ = tick(policy, observation())
    assert field.diagnostics["execution_kind"] == "field_tracking"
    assert policy.pending_field_event["reason"] == "goal_reached"
    for index in range(15):
        obs, close, feedback = tick(policy, obs, fingers=(.015, -.015))
        assert close.action == (0., 0., 0., 0., 0., 0., 1.)
        assert close.diagnostics["execution_kind"] == "local_gripper_settle"
        assert close.diagnostics["refreshed"] is (index == 0)
        assert feedback["chunk_step"] == index
        assert len(transport.calls) == 2
        assert policy.gripper_operation.done is (index == 14)
    assert policy.gripper_operation.stop_reason == "measured_fingers_stable"
    assert policy.gripper_operations == 1 and policy.gripper_operation_actions == 15
    assert policy.client.history_stats()["feedback_records"] == 16
    for index, expected in enumerate(direct["direct"]["actions"]):
        obs, decision, _ = tick(policy, obs)
        assert decision.action == tuple(expected)
        assert decision.diagnostics["execution_kind"] == "direct_five_actions"
        assert decision.diagnostics["refreshed"] is (index == 0)
        assert len(transport.calls) == 3
    obs, recovery_decision, _ = tick(policy, obs)
    assert recovery_decision.diagnostics["execution_kind"] == "field_tracking"
    assert recovery_decision.diagnostics["control_context"]["phase"] == "recover"
    assert recovery_decision.action[1] < 0.
    assert recovery_decision.action[6] == 1.
    assert len(transport.calls) == 4
    assert policy.client.history_stats()["feedback_records"] == 22
    summaries = [row for row in records(transport.calls[2]["input"])
                 if row.get("type") == "local_gripper_result"]
    assert len(summaries) == 1 and summaries[0]["actions_executed"] == 15
    assert not summaries[0]["task_success_certified"]


@pytest.mark.parametrize("command,initial,target,minimum,sign", [
    ("close", (.04, -.04), (.02, -.02), 15, 1.),
    ("open", (.02, -.02), (.04, -.04), 10, -1.),
])
def test_gripper_minimum_actions_feedback_and_unchanged_raw_output(command, initial, target, minimum, sign, tmp_path):
    raw = operation(command)
    policy, transport = controller(raw, prediction(direct=True), log_dir=tmp_path)
    obs = observation(state=[.1, .2, 1., 0., 0., 0., *initial])
    for index in range(minimum):
        obs, decision, feedback = tick(policy, obs, fingers=target)
        assert decision.action[6] == sign
        assert len(transport.calls) == 1
        assert feedback["chunk_step"] == index
        assert decision.diagnostics["update_steps"] == 30
        assert decision.diagnostics["hybrid_prediction"] == raw
        if index == 0:
            assert decision.diagnostics["compact_prediction"] == raw
    assert policy.gripper_operation.done
    assert policy.client.last_compact_prediction == raw
    assert raw["direct"] is None
    assert json.loads(policy.client.last_response["output"][-1]["content"][0]["text"]) == raw
    saved = json.loads(next(tmp_path.glob("*.response.json")).read_text())
    assert json.loads(saved["output"][-1]["content"][0]["text"]) == raw
    request = json.loads(next(tmp_path.glob("*.request.json")).read_text())
    assert request["prompt_version"] == SPEED_FIRST_V2_VERSION
    assert request["billing"] == "chatgpt_subscription"
    _, decision, _ = tick(policy, obs)
    assert len(transport.calls) == 2
    assert decision.diagnostics["execution_kind"] == "direct_five_actions"


@pytest.mark.parametrize("moving", [False, True])
def test_no_motion_or_continuous_motion_ends_at_30_action_budget_without_success_claim(moving):
    policy, transport = controller(operation(), prediction(direct=True))
    obs = observation()
    for index in range(30):
        fingers = (.04 - .0003 * index, -.04 + .0003 * index) if moving else None
        obs, decision, _ = tick(policy, obs, fingers=fingers)
        assert len(transport.calls) == 1
        assert policy.gripper_operation.done is (index == 29)
        assert not decision.diagnostics["task_success_certified"]
    assert policy.gripper_operation.stop_reason == "action_budget_exhausted"
    assert policy.client.history_stats()["feedback_records"] == 30
    assert policy._completion is None
    _, next_decision, _ = tick(policy, obs)
    assert len(transport.calls) == 2
    assert next_decision.diagnostics["execution_kind"] == "direct_five_actions"


def test_operation_replaces_internal_placeholders_with_measured_pose_correction():
    policy, transport = controller(operation())
    original = observation()
    moved, first, _ = tick(policy, original, actual_position=[.13, .24, 1.], rotvec=[0., 0., .3])
    assert first.action[:6] == (0.,) * 6
    correction = policy.act(moved)
    np.testing.assert_allclose(correction.action[:3], [-.12, -.16, 0.], atol=1e-12)
    np.testing.assert_allclose(correction.action[3:6], [0., 0., -.3], atol=1e-12)
    assert len(transport.calls) == 1
    assert policy.gripper_operation.actions_executed == 1
    with pytest.raises(RuntimeError, match="observe_transition"):
        policy.act(moved)
    policy.observe_transition(moved, correction, moved)
    assert policy.gripper_operation.actions_executed == 2


@pytest.mark.parametrize("mutate", [
    lambda row: row.pop("gripper_operation"),
    lambda row: row.update(gripper_operation={"command": "hold"}),
    lambda row: row.update(gripper_operation={"command": "close", "actions": 30}),
    lambda row: row.update(near_target=False),
    lambda row: row.update(controller="coarse_field"),
    lambda row: row.update(direct={"actions": [[0.] * 7] * 5}),
    lambda row: row.update(field=prediction()["field"]),
])
def test_invalid_operation_never_returns_action_or_commits_history(mutate):
    raw = operation()
    mutate(raw)
    policy, transport = controller(raw)
    with pytest.raises(SchemaError):
        policy.act(observation())
    assert policy.step_index == 0 and policy._pending is None
    assert policy.gripper_operations == policy.gripper_operation_actions == 0
    assert policy.client.history_snapshot() == []
    assert policy.client.history_stats()["observations"] == 0
    assert policy.client.last_gripper_operation is None
    assert len(transport.calls) == 1


def test_bad_response_after_completed_operation_does_not_reexecute_operation():
    bad = prediction()
    bad["field"]["orientation_target_xyzw"] = [0., 0., 0., 0.]
    policy, transport = controller(operation(), bad)
    obs = observation()
    for _ in range(15):
        obs, _, _ = tick(policy, obs, fingers=(.02, -.02))
    history = policy.client.history_snapshot()
    with pytest.raises(SchemaError):
        policy.act(obs)
    assert policy.step_index == 15 and policy._pending is None
    assert policy.gripper_operation_actions == 15
    assert policy.client.history_snapshot() == history
    assert len(transport.calls) == 2


def test_real_failed_quaternion_rounding_is_normalized_without_mutating_raw_provenance(tmp_path):
    raw = prediction()
    quaternion = [.9728, -.2292, .0125, .0865]
    raw["field"]["orientation_target_xyzw"] = quaternion
    original = copy.deepcopy(raw)
    policy, transport = controller(raw, log_dir=tmp_path)
    decision = policy.act(observation())
    diagnostics = decision.diagnostics["quaternion_normalization"]
    assert diagnostics["raw"] == quaternion
    assert diagnostics["raw_norm"] == pytest.approx(1.0032502080737387)
    assert diagnostics["max_norm_error"] == .01
    assert diagnostics["orientation_direction_changed"] is False
    normalized = np.asarray(quaternion) / np.linalg.norm(quaternion)
    np.testing.assert_allclose(diagnostics["normalized"], normalized)
    np.testing.assert_allclose(policy.client.last_effective_compact_field["orientation_target_xyzw"], normalized)
    assert policy.client.last_compact_prediction == original == raw
    assert decision.diagnostics["compact_prediction"] == original
    assert json.loads(policy.client.last_response["output"][-1]["content"][0]["text"]) == original
    assert json.loads(next(tmp_path.glob("*.response.json")).read_text())["output"] == policy.client.last_response["output"]
    assert policy.client.history_snapshot()[-1] == policy.client.last_response["output"][-1]
    assert len(transport.calls) == 1


@pytest.mark.parametrize("quaternion", [
    [0., 0., 0., 0.], [0., 0., 0., 1.02], [0., 0., 0., 2.],
    [float("nan"), 0., 0., 1.], [float("inf"), 0., 0., 1.],
    [True, 0., 0., 0.], ["0", 0., 0., 1.],
])
def test_quaternion_repair_does_not_accept_zero_nonfinite_or_gross_errors(quaternion):
    raw = prediction()
    raw["field"]["orientation_target_xyzw"] = quaternion
    policy, transport = controller(raw)
    with pytest.raises(SchemaError):
        policy.act(observation())
    assert policy.step_index == 0 and policy._pending is None
    assert policy.client.last_quaternion_normalization is None
    assert policy.client.history_snapshot() == []
    assert len(transport.calls) == 1


def test_v2_schema_prompt_and_subscription_contract_are_explicit():
    policy, transport = controller()
    payload = policy.client.build_payload(observation().to_dict())
    assert payload["model"] == "gpt-6-astra"
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["text"]["format"]["schema"] == speed_first_v2_schema()
    assert payload["text"]["format"]["strict"] is True
    prompt = payload["instructions"]
    for phrase in (SPEED_FIRST_V2_VERSION, "ZERO-SHOT", "multiple five-action chunks",
                   "0.25 seconds", "at least 15", "at least 10", "at most 30",
                   "gripper_operation=null", "phase=recover", "no automatic lift",
                   "LIBERO success", "collisions are recorded separately"):
        assert phrase in prompt
    assert not transport.calls
    assert policy.client.budget_snapshot()["api_key_calls"] == 0
    assert policy.grid == HYBRID_GRID
    assert (policy.direct_action_steps, policy.field_update_steps, policy.field_execution_steps) == (5, 30, 10)
    assert policy.client.demonstration is None


@pytest.mark.parametrize("kwargs", [
    {"reasoning_effort": "high"}, {"stop_on_model_complete": True}, {"chunk_size": 30},
    {"demonstration": object()}, {"zero_shot_collision_free": True},
    {"config": AstraConfig(model="other-model")},
])
def test_v2_inherits_experiment_contract_guards(kwargs):
    with pytest.raises((ValueError, TypeError)):
        controller(**kwargs)


@pytest.mark.parametrize("reported", [{"model": "other-model"}, {"reasoning": {"effort": "high"}}])
def test_provider_model_or_effort_mismatch_never_executes(reported):
    fake = FakeTransport(envelope(operation(), **reported))
    client = ChatGPTPlanSpeedFirstV2Client(token_provider=lambda: "offline-oauth", transport=fake)
    policy = SpeedFirstV2Policy(client)
    with pytest.raises(APIError):
        policy.act(observation())
    assert policy.step_index == 0 and policy.gripper_operation is None
    assert client.history_snapshot() == []
    assert client.budget_snapshot()["api_key_calls"] == 0


def test_reset_discards_operation_and_pending_pose_without_changing_direct_contract():
    direct = prediction(direct=True)
    policy, transport = controller(operation(), direct)
    obs, _, _ = tick(policy, observation(), fingers=(.02, -.02))
    assert policy.gripper_operations == 1
    policy.reset()
    assert policy.gripper_operation is None
    assert policy.gripper_operations == policy.gripper_operation_actions == 0
    assert policy.step_index == 0
    decision = policy.act(observation())
    assert decision.action == tuple(direct["direct"]["actions"][0])
    assert decision.diagnostics["execution_kind"] == "direct_five_actions"
    assert len(transport.calls) == 2
