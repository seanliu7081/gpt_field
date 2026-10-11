"""Offline contracts for one-request, shared-history visual hybrid routing."""

import copy
from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from gpt_field.astra import APIError, AstraClient, AstraConfig, BudgetExceeded, MissingCredentials, SchemaError, output_schema
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS, _protocol_error
from gpt_field.direct_history import DirectHistoryAstraClient, DirectHistoryPolicy
from gpt_field.hybrid_subscription import (
    ChatGPTPlanHybridClient, HYBRID_PROMPT_VERSION, hybrid_output_schema, validate_hybrid_prediction,
)
from gpt_field.schema import GridSpec
from test_direct_history import execute, prediction as actions
from test_field_history import FakeTransport, envelope, observation, prediction as field


def wrapper(controller="coarse_field", *, target="black_bowl"):
    contact = controller == "contact_direct"
    return {"controller": controller, "target_region": target, "near_target": contact,
            "routing_reason": "The gripper is beside the visible bowl rim." if contact else
                              "The gripper is above the scene with free space before the target.",
            "field": None if contact else field(), "direct": actions() if contact else None}


def client(fake=None, **kwargs):
    return ChatGPTPlanHybridClient(token_provider=lambda: "oauth-hybrid-fixture", transport=fake, **kwargs)


def test_schema_keeps_exact_original_nested_interfaces_and_nullable_branches():
    schema = hybrid_output_schema()
    assert set(schema["required"]) == {"controller", "target_region", "near_target", "routing_reason", "field", "direct"}
    assert schema["additionalProperties"] is False
    for mode in ("field", "direct"):
        assert schema["properties"][mode]["anyOf"] == [output_schema(GridSpec(), mode), {"type": "null"}]


def test_both_valid_branches_accepted_without_action_or_field_modification():
    for control in ("coarse_field", "contact_direct"):
        prediction = wrapper(control)
        assert validate_hybrid_prediction(prediction) is prediction


@pytest.mark.parametrize("change", [
    {"controller": "stage_grasp"}, {"target_region": "ramekin"}, {"near_target": 0},
    {"near_target": True}, {"routing_reason": ""}, {"routing_reason": "  "},
    {"routing_reason": "a" * 501}, {"routing_reason": None}, {"field": None},
    {"direct": actions()}, {"reward": 1}, {"success": False},
])
def test_strict_wrapper_routing_and_single_branch_validation(change):
    with pytest.raises(SchemaError):
        validate_hybrid_prediction({**wrapper(), **change})


@pytest.mark.parametrize("change", [
    {"near_target": False}, {"field": field()}, {"direct": None},
    {"direct": {"actions": []}}, {"direct": {"actions": [[0] * 7] * 4}},
    {"direct": {"actions": [[0] * 6] * 5}},
    {"direct": {"actions": [[0, 0, 0, 0, 0, 0, 1.1]] * 5}},
    {"direct": {"actions": [[0, 0, 0, 0, 0, 0, True]] * 5}},
    {"direct": {**actions(), "extra": 0}},
])
def test_direct_branch_has_exact_frozen_five_by_seven_finite_normalized_contract(change):
    with pytest.raises(SchemaError):
        validate_hybrid_prediction({**wrapper("contact_direct"), **change})


def test_field_branch_has_exact_frozen_grid_masks_and_orientation_validation():
    invalid_fields = [
        {**field(), "values": [0]}, {**field(), "orientation_target_xyzw": [0, 0, 0, 0]},
        {**field(), "blocked_indices": [729]}, {**field(), "blocked_indices": [1], "unknown_indices": [1]},
        {**field(), "success": False},
    ]
    for invalid in invalid_fields:
        with pytest.raises(SchemaError):
            validate_hybrid_prediction({**wrapper(), "field": invalid})
    for invalid in ({}, None, [], {key: value for key, value in wrapper().items() if key != "direct"}):
        with pytest.raises(SchemaError):
            validate_hybrid_prediction(invalid)


def test_preview_has_explicit_hybrid_prompt_original_field_math_and_no_credentials():
    provider = Mock(return_value="oauth-unused")
    instance = ChatGPTPlanHybridClient(token_provider=provider, transport=FakeTransport())
    payload = instance.build_payload(observation().to_dict())
    original = AstraClient._instructions(GridSpec(), "field")
    assert "State is " + original.split("State is ", 1)[1] in payload["instructions"]
    assert HYBRID_PROMPT_VERSION in payload["instructions"]
    assert "Routing is reversible" in payload["instructions"]
    assert "Holding the bowl alone does not mean contact_direct forever" in payload["instructions"]
    assert "across BOTH controllers, initially open" in payload["instructions"]
    assert "Direct gripper 0 does not change that remembered" in payload["instructions"]
    assert "not from a numeric potential value" in payload["instructions"]
    assert "at most 500 characters" in payload["instructions"]
    assert payload["text"]["format"]["name"] == "astra_hybrid_history_v1"
    assert payload["text"]["format"]["schema"] == hybrid_output_schema()
    assert payload["model"] == "gpt-6-astra" and payload["store"] is False and payload["stream"] is True
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert not UNSUPPORTED_FIELDS.intersection(payload)
    provider.assert_not_called()
    assert not instance.transport.calls and instance.history_snapshot() == []


def test_single_history_preserves_field_to_direct_to_field_full_outputs_feedback_and_opaque_items():
    predictions = [wrapper(), wrapper("contact_direct"), wrapper(target="plate")]
    fake = FakeTransport(*(envelope(prediction, model="gpt-6-astra") for prediction in predictions))
    instance = client(fake)
    # Produce allowed observable feedback using the exact frozen implementation.
    feedback_client = DirectHistoryAstraClient(api_key="offline-fixture",
        transport=FakeTransport(*(envelope(actions()) for _ in range(3))))
    feedback_policy = DirectHistoryPolicy(feedback_client)
    obs = observation()
    first_images = (obs.image, obs.wrist_image)
    for batch, prediction in enumerate(predictions):
        assert instance.predict(obs.to_dict()) == prediction
        for offset in range(5):
            obs, _, feedback = execute(feedback_policy, obs, batch * 5 + offset)
            instance.record_transition(feedback)
    assert len(fake.calls) == 3
    last = fake.calls[-1]
    assert last["input"][1:3] == envelope(predictions[0])["output"]
    prior_wrappers = [json.loads(item["content"][0]["text"]) for item in last["input"]
                      if item.get("role") == "assistant"]
    assert prior_wrappers == predictions[:2]
    assert all(item["phase"] == "final_answer" for item in last["input"] if item.get("role") == "assistant")
    assert [item for item in last["input"] if item.get("type") == "reasoning"] == [envelope()["output"][0]] * 2
    records = [json.loads(item["content"][0]["text"]) for item in last["input"] if item.get("role") == "user"]
    assert [record["observation_index"] for record in records if record["type"] == "observation"] == [0, 1, 2]
    assert [record["step"] for record in records if record["type"] == "execution_feedback"] == list(range(10))
    images = [part["image_url"] for item in last["input"] for part in item.get("content", []) if part["type"] == "input_image"]
    assert len(images) == 4 and not any(image in images for image in first_images)
    assert instance.history_stats()["feedback_records"] == 15
    assert instance.budget_snapshot()["api_calls"] == 3
    assert instance.budget_snapshot()["input_tokens"] == 300


def test_invalid_wrapper_never_retries_or_commits_after_valid_history():
    fake = FakeTransport(envelope(wrapper()), envelope({**wrapper(), "near_target": True}), envelope(wrapper()))
    instance = client(fake)
    instance.predict(observation().to_dict())
    before = instance.history_snapshot()
    with pytest.raises(SchemaError):
        instance.predict(observation(1).to_dict())
    assert len(fake.calls) == 2 and instance.history_snapshot() == before


def test_hybrid_provenance_request_hash_and_all_used_access_tokens_redacted(tmp_path):
    tokens = ["oauth-private-one", "oauth-private-two"]
    first, second = envelope(wrapper()), envelope(wrapper("contact_direct"))
    first["server_echo"] = tokens[0]
    second["server_echo"] = tokens
    fake = FakeTransport(first, second)
    instance = ChatGPTPlanHybridClient(token_provider=Mock(side_effect=tokens), transport=fake, log_dir=tmp_path)
    instance.predict(observation().to_dict())
    instance.predict(observation(1).to_dict())
    for index, path in enumerate(sorted(tmp_path.glob("*.request.json"))):
        record = json.loads(path.read_text())
        assert record["prompt_version"] == HYBRID_PROMPT_VERSION
        assert record["mode"] == "hybrid" and record["billing"] == "chatgpt_subscription"
        assert record["request_key"] == instance.request_key(fake.calls[index], GridSpec(), "hybrid")
        assert record["history_stats_before_request"]["observations"] == index
    logs = "".join(path.read_text() for path in tmp_path.glob("*.json"))
    assert all(token not in logs for token in tokens)
    assert "astra-field-history-v1" not in logs and "astra-direct-history-v1" not in logs


@pytest.mark.parametrize("bad", [envelope(wrapper(), status="incomplete"),
                                envelope(wrapper(), model="different-model"), APIError("failure")])
def test_invalid_completion_wrong_model_or_transport_error_never_commit(bad):
    fake = FakeTransport(bad)
    instance = client(fake)
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert len(fake.calls) == 1 and not instance.history_snapshot()


def test_safe_quota_codes_retained_and_interrupt_unknown_usage_accounted(tmp_path):
    instance = client(FakeTransport(_protocol_error("secret-server-message", http_status=429,
        error_code="subscription_sharing_usage_limit_exceeded")), log_dir=tmp_path / "quota")
    with pytest.raises(APIError) as caught:
        instance.predict(observation().to_dict())
    assert caught.value.error_code == "subscription_sharing_usage_limit_exceeded"
    assert caught.value.http_status == 429
    result = json.loads(next((tmp_path / "quota").glob("*.result.json")).read_text())
    assert "secret-server-message" not in json.dumps(result)
    assert result["budget"]["unknown_usage_calls"] == 1
    interrupted = client(FakeTransport(KeyboardInterrupt()), log_dir=tmp_path / "interrupt")
    with pytest.raises(KeyboardInterrupt):
        interrupted.predict(observation().to_dict())
    assert interrupted.budget_snapshot()["unknown_usage_calls"] == 1
    assert not interrupted.history_snapshot()


def test_refusal_is_never_committed():
    value = envelope(wrapper())
    value["output"][1]["content"] = [{"type": "refusal", "refusal": "No"}]
    instance = client(FakeTransport(value))
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert not instance.history_snapshot()


def test_no_credentials_or_fallback_on_bad_observation_mode_grid_or_auth(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-forbidden")
    provider = Mock(return_value="")
    fake = FakeTransport()
    instance = ChatGPTPlanHybridClient(token_provider=provider, transport=fake)
    with pytest.raises(ValueError):
        instance.predict({**observation().to_dict(), "reward": 1})
    with pytest.raises(ValueError):
        instance.predict(observation().to_dict(), mode="field")
    with pytest.raises(ValueError):
        instance.predict(observation().to_dict(), GridSpec(size=3))
    with pytest.raises(ValueError):
        instance.predict(replace(observation(), prompt="another task").to_dict())
    provider.assert_not_called()
    with pytest.raises(MissingCredentials):
        instance.predict(observation().to_dict())
    assert not fake.calls and instance.budget_snapshot()["api_key_calls"] == 0


def test_subscription_accounting_not_api_pricing_and_reset_never_clears_lifetime_calls():
    fake = FakeTransport(envelope(wrapper(), usage={"input_tokens": 2_000_000, "output_tokens": 20_000}))
    instance = client(fake, config=AstraConfig(max_output_tokens=1))
    instance.predict(observation().to_dict())
    usage = instance.budget_snapshot()
    assert usage["output_tokens"] == 20_000 and usage["subscription_cost_usd"] is None
    assert usage["cost_usd_upper_bound"] is None and usage["api_key_calls"] == 0
    instance.reset_history()
    assert instance.history_snapshot() == [] and instance.history_stats()["observations"] == 0
    assert instance.budget_snapshot()["api_calls"] == 1
    with pytest.raises(ValueError):
        client(config=AstraConfig(max_cost_usd=300))


def test_optional_call_guard_and_snapshot_copy_are_preserved():
    instance = client(FakeTransport(envelope(wrapper())), config=AstraConfig(max_api_calls=1))
    instance.predict(observation().to_dict())
    before = instance.history_snapshot()
    before[1]["encrypted_content"] = "changed-copy"
    with pytest.raises(BudgetExceeded):
        instance.predict(observation().to_dict())
    assert instance.history_snapshot()[1]["encrypted_content"] == "opaque-fixture"
