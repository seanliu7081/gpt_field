"""Offline direct-action subscription transport and exact history contracts."""

import copy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from gpt_field.astra import APIError, AstraConfig, BudgetExceeded, MissingCredentials, SchemaError
from gpt_field.chatgpt_direct import ChatGPTPlanDirectClient
from gpt_field.chatgpt_plan import TRANSPORT_VERSION, UNSUPPORTED_FIELDS
from gpt_field.direct_history import DirectHistoryAstraClient, DirectHistoryPolicy, PROMPT_VERSION
from gpt_field.schema import GridSpec
from test_direct_history import FakeTransport, envelope, execute, observation, prediction


def client(fake=None, **kwargs):
    return ChatGPTPlanDirectClient(token_provider=lambda: "oauth-fixture-direct",
                                  transport=fake, **kwargs)


def test_subscription_payload_is_exact_frozen_direct_payload_except_transport_fields():
    original = DirectHistoryAstraClient(api_key="unused").build_payload(observation().to_dict())
    expected = {key: value for key, value in original.items() if key not in UNSUPPORTED_FIELDS}
    expected["stream"] = True
    instance = client(FakeTransport())
    actual = instance.build_payload(observation().to_dict())
    assert actual == expected
    assert actual["text"]["format"]["name"] == "astra_direct_history_v1"
    assert actual["text"]["format"]["schema"]["required"] == ["actions"]
    assert "astra-field-history-v1" not in actual["instructions"]
    assert actual["model"] == "gpt-6-astra" and actual["store"] is False and actual["stream"] is True
    assert not instance.transport.calls


def test_exact_five_by_seven_actions_are_returned_and_policy_executes_original_values():
    actions = {"actions": [[index / 10, -.1, 0, .2, 0, -.3, -1 if index < 2 else 1] for index in range(5)]}
    fake = FakeTransport(envelope(actions, model="gpt-6-astra"))
    instance = client(fake)
    policy = DirectHistoryPolicy(instance)
    obs = observation()
    executed = []
    for step in range(5):
        obs, decision, feedback = execute(policy, obs, step)
        executed.append(list(decision.action))
        assert feedback["normalized_action"] == actions["actions"][step]
    assert executed == actions["actions"] and len(fake.calls) == 1
    assert instance.history_stats()["feedback_records"] == 5


@pytest.mark.parametrize("bad", [
    {"actions": []}, {"actions": [[0] * 7] * 4}, {"actions": [[0] * 6] * 5},
    {"actions": [[0, 0, 0, 0, 0, 0, 1.01]] * 5},
    {"actions": [[0, 0, 0, 0, 0, 0, True]] * 5},
    {"actions": [[0] * 7] * 5, "extra": "forbidden"},
])
def test_invalid_actions_never_retry_commit_or_execute(bad):
    fake = FakeTransport(envelope(bad), envelope())
    instance = client(fake)
    policy = DirectHistoryPolicy(instance)
    with pytest.raises(SchemaError):
        policy.act(observation())
    assert len(fake.calls) == 1
    assert instance.history_snapshot() == []
    assert policy.step_index == 0 and not policy.chunk
    assert instance.budget_snapshot()["api_calls"] == 1


def test_full_opaque_history_feedback_and_latest_two_image_pairs_survive_subscription():
    fake = FakeTransport(envelope(), envelope(prediction(-1)), envelope())
    instance = client(fake)
    policy = DirectHistoryPolicy(instance)
    obs = observation()
    initial_images = (obs.image, obs.wrist_image)
    for step in range(11):
        obs, _, _ = execute(policy, obs, step)
    third = fake.calls[2]
    assert third["input"][1:3] == envelope()["output"]
    reasoning = [item for item in third["input"] if item.get("type") == "reasoning"]
    assert reasoning == [envelope()["output"][0]] * 2
    assert all(item["phase"] == "final_answer" for item in third["input"] if item.get("role") == "assistant")
    images = [part["image_url"] for item in third["input"] for part in item.get("content", [])
              if part["type"] == "input_image"]
    assert len(images) == 4 and not any(image in images for image in initial_images)
    records = [json.loads(item["content"][0]["text"]) for item in third["input"] if item.get("role") == "user"]
    assert [record["step"] for record in records if record["type"] == "execution_feedback"] == list(range(10))
    assert [record["observation_index"] for record in records if record["type"] == "observation"] == [0, 1, 2]
    assert instance.history_stats()["feedback_records"] == 11


def test_direct_provenance_and_request_hash_are_correct_with_no_token_logs(tmp_path):
    token1, token2 = "oauth-token-one-private", "oauth-token-two-private"
    first, second = envelope(), envelope()
    first["ignored_server_echo"] = token1
    second["ignored_server_echo"] = [token1, token2]
    fake = FakeTransport(first, second)
    instance = ChatGPTPlanDirectClient(token_provider=Mock(side_effect=[token1, token2]),
                                       transport=fake, log_dir=tmp_path)
    instance.predict(observation().to_dict())
    instance.predict(observation(1).to_dict())
    requests = sorted(tmp_path.glob("*.request.json"))
    for index, path in enumerate(requests):
        record = json.loads(path.read_text())
        assert record["prompt_version"] == PROMPT_VERSION
        assert record["mode"] == "direct"
        assert record["billing"] == "chatgpt_subscription" and record["transport_version"] == TRANSPORT_VERSION
        assert record["request_key"] == DirectHistoryAstraClient.request_key(instance, fake.calls[index], GridSpec(), "direct")
    logs = "".join(path.read_text() for path in tmp_path.glob("*.json"))
    assert token1 not in logs and token2 not in logs
    assert "astra-field-history-v1" not in logs and "astra-potential-v1" not in logs


def test_subscription_accounting_has_no_api_prices_or_output_cap():
    instance = client(FakeTransport(envelope(usage={"input_tokens": 100, "output_tokens": 20_000})),
                      config=AstraConfig(max_output_tokens=1))
    instance.predict(observation().to_dict())
    usage = instance.budget_snapshot()
    assert usage["billing"] == "chatgpt_subscription" and usage["api_key_calls"] == 0
    assert usage["output_tokens"] == 20_000 and usage["cost_usd_upper_bound"] is None
    assert usage["subscription_cost_usd"] is None
    with pytest.raises(ValueError):
        client(config=AstraConfig(max_cost_usd=300))


def test_no_api_key_fallback_or_credentials_needed_for_preview(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-never-use")
    provider = Mock(return_value="")
    fake = FakeTransport()
    instance = ChatGPTPlanDirectClient(token_provider=provider, transport=fake)
    instance.build_payload(observation().to_dict())
    provider.assert_not_called()
    with pytest.raises(MissingCredentials):
        instance.predict(observation().to_dict())
    assert not fake.calls and instance._api_key == ""


@pytest.mark.parametrize("value", [APIError("quota"), envelope(status="incomplete"), envelope(model="different-model")])
def test_transport_incomplete_or_wrong_model_never_commits(value):
    fake = FakeTransport(value)
    instance = client(fake)
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert len(fake.calls) == 1 and instance.history_snapshot() == []


def test_refusal_is_not_a_direct_action():
    value = envelope()
    value["output"][1]["content"] = [{"type": "refusal", "refusal": "Cannot comply"}]
    instance = client(FakeTransport(value))
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert instance.history_snapshot() == []


def test_validation_happens_before_auth_and_wrong_mode_or_oracle_is_rejected():
    provider = Mock(return_value="oauth-fixture")
    instance = ChatGPTPlanDirectClient(token_provider=provider)
    with pytest.raises(ValueError):
        instance.predict(observation().to_dict(), mode="field")
    with pytest.raises(ValueError):
        instance.predict({**observation().to_dict(), "reward": 1})
    with pytest.raises(ValueError):
        instance.predict(observation().to_dict(), GridSpec(frame="robot"))
    with pytest.raises(ValueError):
        instance.predict(replace(observation(), prompt="wrong task").to_dict())
    provider.assert_not_called()


def test_failed_next_request_keeps_prior_history_and_snapshots_are_copies():
    instance = client(FakeTransport(envelope(), envelope({})))
    instance.predict(observation().to_dict())
    before = instance.history_snapshot()
    with pytest.raises(SchemaError):
        instance.predict(observation(1).to_dict())
    assert instance.history_snapshot() == before
    before[1]["encrypted_content"] = "mutated-copy"
    assert instance.history_snapshot()[1]["encrypted_content"] == "opaque-fixture"


def test_inherited_optional_call_guard_never_falls_back():
    fake = FakeTransport(envelope())
    instance = client(fake, config=AstraConfig(max_api_calls=1))
    instance.predict(observation().to_dict())
    with pytest.raises(BudgetExceeded):
        instance.predict(observation(1).to_dict())
    assert len(fake.calls) == 1
