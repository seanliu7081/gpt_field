"""Controller handoffs preserve execution semantics and measured history."""

import copy
from dataclasses import replace

import pytest

from gpt_field.astra import SchemaError
from gpt_field.hybrid_policy import HybridFieldDirectPolicy
from gpt_field.policy import Decision, FieldPolicy
from test_field_history import observation, prediction


def field(gripper="open", axis=2):
    return dict(controller="coarse_field", target_region="black_bowl", near_target=False,
                routing_reason="Object is ahead of the gripper.",
                field=prediction(gripper=gripper, axis=axis), direct=None)


def direct(gripper=1):
    return dict(controller="contact_direct", target_region="black_bowl", near_target=True,
                routing_reason="Gripper is next to the bowl rim.", field=None,
                direct={"actions": [[.1 * i, -.05, .1, 0, .02, 0, gripper if i == 0 else 0]
                                    for i in range(5)]})


class Client:
    last_request_key = "offline-key"
    last_log_path = "offline.json"

    def __init__(self, *predictions):
        self.predictions = copy.deepcopy(list(predictions))
        self.calls, self.feedback, self.resets = [], [], 0

    def reset_history(self):
        self.resets += 1
        self.feedback.clear()

    def history_stats(self):
        return {"feedback_records": len(self.feedback)}

    def record_transition(self, feedback):
        self.feedback.append(copy.deepcopy(feedback))

    def predict(self, obs, grid, mode):
        self.calls.append((copy.deepcopy(obs), grid, mode))
        return self.predictions.pop(0)


def execute(policy, obs, step):
    decision = policy.act(obs)
    state = list(obs.state)
    for i in range(3):
        state[i] += decision.action[i] * .025
    state[6] -= .001 * decision.action[6]
    state[7] += .001 * decision.action[6]
    after = observation(step + 1, state)
    feedback = policy.observe_transition(obs, decision, after)
    return after, decision, feedback


def test_coarse_actions_and_plans_equal_frozen_original_policy():
    data = field()
    client = Client(data)
    policy = HybridFieldDirectPolicy(client)
    original = FieldPolicy(Client(data["field"]))
    obs = observation()
    for i in range(5):
        reference = original.act(obs)
        obs, actual, _ = execute(policy, obs, i)
        assert actual.action == reference.action
        assert actual.diagnostics["plan"] == reference.diagnostics["plan"]
        assert actual.diagnostics["field_center"] == reference.diagnostics["field_center"]
    assert len(client.calls) == 1


@pytest.mark.parametrize("intent", [-1, 1])
def test_field_direct_field_handoffs_preserve_history_grip_and_fresh_center(intent):
    data = direct(intent)
    client = Client(field(), data, field("hold", axis=0))
    policy = HybridFieldDirectPolicy(client)
    obs = observation()
    decisions, centers = [], []
    for step in range(15):
        centers.append(list(obs.state[:3]))
        obs, decision, feedback = execute(policy, obs, step)
        decisions.append(decision)
        assert feedback["step"] == step
        assert decision.diagnostics["chunk_index"] == step // 5
        assert decision.diagnostics["chunk_step"] == step % 5
        assert (policy.field is None) == (5 <= step < 10)
        if 5 <= step < 10:
            assert decision.action == tuple(data["direct"]["actions"][step - 5])
            assert "plan" not in decision.diagnostics
        if step >= 10:
            assert decision.action[6] == intent
    assert [d.diagnostics["step"] for d in decisions if d.diagnostics["mode_changed"]] == [5, 10]
    assert [c[0]["observation/state"][:3] for c in client.calls] == [centers[i] for i in (0, 5, 10)]
    assert decisions[10].diagnostics["field_center"] == centers[10]
    assert centers[0] != centers[10]
    assert decisions[10].diagnostics["field_refreshed"]
    assert [x["step"] for x in client.feedback] == list(range(15))
    assert client.resets == 1
    assert len(client.calls) == 3
    assert policy.step_index == 15


def test_initial_direct_and_zero_do_not_open_remembered_close():
    client = Client(direct(), field("hold"))
    policy = HybridFieldDirectPolicy(client)
    obs = observation()
    for step in range(6):
        obs, decision, _ = execute(policy, obs, step)
        if step == 0:
            assert not decision.diagnostics["mode_changed"]
        if 1 <= step <= 4:
            assert decision.action[6] == 0
        else:
            assert decision.action[6] == 1


def test_feedback_required_exactly_once_with_matching_decision_and_observations():
    policy = HybridFieldDirectPolicy(Client(direct()))
    before = observation()
    decision = policy.act(before)
    after = observation(1)
    with pytest.raises(RuntimeError):
        policy.act(before)
    with pytest.raises(ValueError):
        policy.observe_transition(before, Decision(decision.action, decision.diagnostics), after)
    with pytest.raises(ValueError):
        policy.observe_transition(observation(2), decision, after)
    policy.observe_transition(before, decision, after)
    with pytest.raises(RuntimeError):
        policy.observe_transition(before, decision, after)
    with pytest.raises(ValueError):
        policy.act(before)
    policy.act(after)


def test_invalid_envelope_issues_no_action_or_step():
    data = direct()
    data["field"] = field()["field"]
    policy = HybridFieldDirectPolicy(Client(data))
    with pytest.raises(SchemaError):
        policy.act(observation())
    assert policy.step_index == 0
    assert policy._pending is None
    assert policy.controller is None


def test_reset_clears_episode_state_and_restores_initial_open_intent():
    client = Client(direct(), field("hold"))
    policy = HybridFieldDirectPolicy(client)
    execute(policy, observation(), 0)
    assert policy.adapter._gripper == 1
    policy.reset()
    assert client.resets == 2
    assert client.feedback == []
    assert policy.step_index == 0
    assert policy.controller is None
    assert policy.field is None
    assert policy.latest_feedback is None
    assert policy._progress == []
    assert policy.chunk == []
    assert policy._pending is None
    assert policy._expected_observation is None
    assert policy.adapter._gripper == -1
    assert policy.act(observation()).action[6] == -1
