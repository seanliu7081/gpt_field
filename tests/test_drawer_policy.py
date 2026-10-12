"""Task-aware policy plumbing uses synthetic observations and transport only."""
import copy
from dataclasses import replace
import json

import pytest

from gpt_field.direct_history import DirectHistoryAstraClient, DirectHistoryPolicy, TASK
from gpt_field.distance_policy import DistanceHybridPolicy
from gpt_field.distance_subscription import ChatGPTPlanDistanceClient
from gpt_field.distance_tasks import resolve_distance_task
from test_distance_policy import intent, tick
from test_field_history import FakeTransport, envelope, observation


DRAWER = resolve_distance_task("drawer_bowl")


def drawer_observation():
    return replace(observation(), prompt=DRAWER.prompt)


def drawer_response(*, target="drawer_handle", phase="approach", direct=False, hold=False):
    raw = intent(phase=phase, direct=direct, hold=hold)
    raw["target_region"] = target
    return raw


def drawer_controller(*responses):
    transport = FakeTransport(*(envelope(raw, id=f"drawer-policy-{i}", model="gpt-6-astra",
                                        reasoning={"effort": "medium"}) for i, raw in enumerate(responses)))
    client = ChatGPTPlanDistanceClient(task_key=DRAWER.key, token_provider=lambda: "offline-oauth",
                                      transport=transport)
    return DistanceHybridPolicy(client), transport


def test_direct_history_payload_uses_explicit_task_prompt_without_inference():
    client = DirectHistoryAstraClient(api_key=None, task_prompt=DRAWER.prompt)
    payload = client.build_payload(drawer_observation().to_dict())
    assert json.loads(payload["input"][-1]["content"][0]["text"])["prompt"] == DRAWER.prompt
    with pytest.raises(ValueError, match="configured task"):
        client.build_payload(observation().to_dict())
    with pytest.raises(ValueError, match="requested bowl task"):
        DirectHistoryAstraClient(api_key=None).build_payload(drawer_observation().to_dict())
    assert client.history_stats()["observations"] == 0


class ObservableDirectClient:
    """No transport, credentials or simulator; exercise DirectHistoryPolicy itself."""
    task_prompt = DRAWER.prompt
    last_request_key = last_log_path = None

    def __init__(self, actions):
        self.actions = copy.deepcopy(actions)
        self.reset_history()

    def reset_history(self):
        self.feedback, self.calls = [], []

    def predict(self, obs, grid, mode):
        self.calls.append(copy.deepcopy(obs))
        return {"actions": copy.deepcopy(self.actions)}

    def history_stats(self):
        return {"observations": len(self.calls), "feedback_records": len(self.feedback)}

    def record_transition(self, feedback):
        self.feedback.append(copy.deepcopy(feedback))


def test_direct_history_policy_checks_task_on_act_and_both_feedback_observations():
    actions = [[.01 * i, 0., 0., 0., 0., 0., 1.] for i in range(5)]
    client = ObservableDirectClient(actions)
    policy = DirectHistoryPolicy(client)
    correct = drawer_observation()
    with pytest.raises(ValueError, match="configured task"):
        policy.act(observation())
    assert client.calls == []
    decision = policy.act(correct)
    for wrong_before, wrong_after in ((observation(), correct), (correct, observation())):
        with pytest.raises(ValueError, match="configured task"):
            policy.observe_transition(wrong_before, decision, wrong_after)
    assert client.feedback == []
    feedback = policy.observe_transition(correct, decision, correct)
    assert feedback["step"] == 0
    assert client.calls[0]["prompt"] == DRAWER.prompt
    assert len(client.feedback) == 1


@pytest.mark.parametrize("target", DRAWER.target_regions)
def test_task_routes_survive_outer_hybrid_normalized_validation(target):
    raw = drawer_response(target=target, hold=True)
    policy, transport = drawer_controller(raw)
    obs, decision = tick(policy, drawer_observation())
    assert decision.diagnostics["routing"]["target_region"] == target
    assert decision.diagnostics["controller"] == "coarse_field"
    assert decision.diagnostics["plan"]["status"] == "hold"
    assert obs.prompt == DRAWER.prompt
    assert len(transport.calls) == 1
    assert policy.client.history_stats()["feedback_records"] == 1


def test_hybrid_task_guard_prevents_cross_task_actions_and_feedback():
    policy, transport = drawer_controller(drawer_response(hold=True))
    with pytest.raises(ValueError, match="configured task"):
        policy.act(observation())
    assert transport.calls == []
    correct = drawer_observation()
    decision = policy.act(correct)
    for before, after in ((observation(), correct), (correct, observation())):
        with pytest.raises(ValueError, match="configured task"):
            policy.observe_transition(before, decision, after)
    assert policy.client.history_stats()["feedback_records"] == 0
    policy.observe_transition(correct, decision, correct)
    assert policy.client.history_stats()["feedback_records"] == 1


@pytest.mark.parametrize("phase", ["verify", "approach"])
def test_drawer_destination_hold_requests_review_without_automatic_vertical_motion(phase):
    raw = drawer_response(target="drawer_interior", phase=phase, hold=True)
    policy, transport = drawer_controller(raw, copy.deepcopy(raw))
    obs = drawer_observation()
    initial = obs.state
    for i in range(10):
        obs, decision = tick(policy, obs)
        assert decision.action[:3] == (0., 0., 0.)
        assert obs.state[:3] == initial[:3]
        assert not decision.diagnostics["local_recovery_active"]
        assert not policy._recovery_active and policy.local_revision == 0
        assert (policy.pending_field_event is not None) is (i % 5 == 4)
        if i % 5 == 4:
            assert policy.pending_field_event["reason"] == "placement_unverified"
    assert len(transport.calls) == 2
    assert policy._completion is None


@pytest.mark.parametrize("phase", ["handle_grasp", "drawer_open", "handle_release"])
def test_handle_contact_executes_exactly_five_unmodified_actions(phase):
    raw = drawer_response(phase=phase, direct=True)
    actions = [[.03 * i, -.02 * i, 0., 0., .01 * i, 0., grip]
               for i, grip in enumerate([1., 1., 0., -1., 0.])]
    raw["direct"]["actions"] = copy.deepcopy(actions)
    policy, transport = drawer_controller(raw, drawer_response(hold=True))
    obs = drawer_observation()
    for i in range(5):
        obs, decision = tick(policy, obs)
        assert decision.action == tuple(actions[i])
        assert decision.diagnostics["routing"]["target_region"] == "drawer_handle"
        assert decision.diagnostics["refreshed"] is (i == 0)
        assert len(transport.calls) == 1
        assert "plan" not in decision.diagnostics
    assert policy.pending_field_event["reason"] == "field_budget_exhausted"
    assert policy.client.history_stats()["feedback_records"] == 5
    tick(policy, obs)
    assert len(transport.calls) == 2
