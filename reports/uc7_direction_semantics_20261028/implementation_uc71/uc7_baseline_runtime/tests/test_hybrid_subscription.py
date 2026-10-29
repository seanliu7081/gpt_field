"""Offline contracts for one-request, shared-history visual hybrid routing."""

import copy
from dataclasses import replace
import hashlib
import json
from unittest.mock import Mock

import pytest

from gpt_field.astra import APIError, AstraClient, AstraConfig, BudgetExceeded, MissingCredentials, SchemaError, output_schema
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS, _protocol_error
from gpt_field.direct_history import DirectHistoryAstraClient, DirectHistoryPolicy
from gpt_field.hybrid_subscription import (
    ChatGPTPlanHybridClient, HYBRID_CHUNK_SIZE, HYBRID_GRID, HYBRID_PROMPT_VERSION,
    HYBRID_FIELD_UPDATE_STEPS, HYBRID_PLANNER_HORIZON, HYBRID_FIELD_EXECUTION_STEPS, HYBRID_REASONING_EFFORT,
    HYBRID_SYMMETRIC_PROMPT_VERSION, HYBRID_ASYMMETRIC_PROMPT_VERSION, HYBRID_RECEDING_PROMPT_VERSION,
    LEGACY_HYBRID_PROMPT_VERSION, hybrid_output_schema, hybrid_prompt_version,
    validate_hybrid_prediction,
)
from gpt_field.schema import GridSpec
from test_direct_history import execute, prediction as legacy_actions
from test_field_history import FakeTransport, envelope, observation, prediction as legacy_field


def actions(chunk_size=HYBRID_CHUNK_SIZE):
    return {"actions": [legacy_actions()["actions"][0] for _ in range(chunk_size)]}


def field(grid=HYBRID_GRID):
    return {**legacy_field(), "values": grid.coordinates([0., 0., 0.])[:, 2].tolist()}


def wrapper(controller="coarse_field", *, target="black_bowl", grid=HYBRID_GRID,
            chunk_size=HYBRID_CHUNK_SIZE):
    contact = controller == "contact_direct"
    return {"controller": controller, "target_region": target, "near_target": contact,
            "routing_reason": "The gripper is beside the visible bowl rim." if contact else
                              "The gripper is above the scene with free space before the target.",
            "field": None if contact else field(grid),
            "direct": actions(chunk_size) if contact else None}


def client(fake=None, **kwargs):
    return ChatGPTPlanHybridClient(token_provider=lambda: "oauth-hybrid-fixture", transport=fake, **kwargs)


def test_schema_uses_five_mm_spacing_same_coverage_and_five_direct_actions():
    schema = hybrid_output_schema()
    assert set(schema["required"]) == {"controller", "target_region", "near_target", "routing_reason", "field", "direct"}
    assert schema["additionalProperties"] is False
    assert HYBRID_GRID.size == 33 and HYBRID_GRID.spacing_m == .005
    assert HYBRID_GRID.count == 35937 and HYBRID_GRID.half_width_m == GridSpec().half_width_m == .08
    assert schema["properties"]["field"]["anyOf"] == [output_schema(HYBRID_GRID, "field"), {"type": "null"}]
    field_values = schema["properties"]["field"]["anyOf"][0]["properties"]["values"]
    assert field_values["minItems"] == field_values["maxItems"] == 35937
    direct = schema["properties"]["direct"]["anyOf"]
    assert direct[1] == {"type": "null"}
    assert direct[0]["properties"]["actions"]["minItems"] == 5
    assert direct[0]["properties"]["actions"]["maxItems"] == 5
    assert output_schema(GridSpec(), "direct")["properties"]["actions"]["maxItems"] == 5


def test_explicit_legacy_schema_prompt_and_validation_are_unchanged():
    schema = hybrid_output_schema(GridSpec(), 5)
    for mode in ("field", "direct"):
        assert schema["properties"][mode]["anyOf"] == [output_schema(GridSpec(), mode), {"type": "null"}]
    assert hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest() == (
        "b68e1e02d63451236f89cd348ca7bc10435d97517562612560b892bebe2ac4b1")
    instructions = ChatGPTPlanHybridClient._instructions(GridSpec(), "hybrid", 5)
    assert hashlib.sha256(instructions.encode()).hexdigest() == (
        "bbe118fc0a5f94a7b3f34083b91dc6cfd220371b17946c707766726089f8b835")
    assert hybrid_prompt_version(GridSpec(), 5) == LEGACY_HYBRID_PROMPT_VERSION
    assert hybrid_prompt_version(HYBRID_GRID, 30) == HYBRID_SYMMETRIC_PROMPT_VERSION
    assert hybrid_prompt_version(HYBRID_GRID, 5) == HYBRID_PROMPT_VERSION
    instance = client(FakeTransport(), grid=GridSpec(), chunk_size=5, reasoning_effort=None)
    payload = instance.build_payload(observation().to_dict())
    assert payload["instructions"].startswith(instructions)
    assert payload["text"]["format"]["name"] == "astra_hybrid_history_v1"
    assert payload["text"]["format"]["schema"] == schema
    assert "reasoning" not in payload
    assert instance.prompt_version == LEGACY_HYBRID_PROMPT_VERSION
    for controller in ("coarse_field", "contact_direct"):
        prediction = wrapper(controller, grid=GridSpec(), chunk_size=5)
        assert validate_hybrid_prediction(prediction, GridSpec(), 5) is prediction
        if controller == "coarse_field":
            with pytest.raises(SchemaError):
                validate_hybrid_prediction(prediction)
        else:
            assert validate_hybrid_prediction(prediction) is prediction


def test_explicit_legacy_client_predicts_and_logs_v1_without_changing_current_defaults(tmp_path):
    prediction = wrapper("contact_direct", grid=GridSpec(), chunk_size=5)
    instance = client(FakeTransport(envelope(prediction)), grid=GridSpec(), chunk_size=5,
                      reasoning_effort=None, log_dir=tmp_path)
    assert instance.predict(observation().to_dict()) == prediction
    record = json.loads(next(tmp_path.glob("*.request.json")).read_text())
    assert record["prompt_version"] == LEGACY_HYBRID_PROMPT_VERSION
    assert record["grid"] == GridSpec().to_dict()
    assert record["request_key"] == instance.request_key(record["payload"], GridSpec(), "hybrid")
    assert record["payload"]["text"]["format"]["name"] == "astra_hybrid_history_v1"
    assert "reasoning" not in record["payload"]
    assert instance.chunk_size == instance.field_update_steps == 5
    assert instance.planner_horizon == 3
    assert instance.field_execution_steps == 1
    assert client(FakeTransport()).chunk_size == 5
    assert client(FakeTransport()).field_update_steps == 30


def test_explicit_symmetric_intervals_preserve_v2_prompt_schema_and_transport():
    instructions = ChatGPTPlanHybridClient._instructions(HYBRID_GRID, "hybrid", 30, 30)
    assert hashlib.sha256(instructions.encode()).hexdigest() == (
        "3e3fcb15466b56f78513170e2346eb1604c30eae271ad1355401d06db2206975")
    assert ChatGPTPlanHybridClient._instructions(HYBRID_GRID, "hybrid", 30) == instructions
    prediction = wrapper("contact_direct", chunk_size=30)
    instance = client(FakeTransport(envelope(prediction)), chunk_size=30, field_update_steps=30)
    assert instance.predict(observation().to_dict()) == prediction
    assert instance.prompt_version == HYBRID_SYMMETRIC_PROMPT_VERSION
    assert instance.planner_horizon == 3
    assert instance.field_execution_steps == 1
    payload = instance.transport.calls[0]
    assert payload["instructions"].startswith(instructions)
    assert payload["text"]["format"]["name"] == "astra_hybrid_history_v2"
    assert payload["text"]["format"]["schema"] == hybrid_output_schema(HYBRID_GRID, 30)
    with pytest.raises(SchemaError):
        validate_hybrid_prediction(prediction)


def test_explicit_three_step_asymmetric_client_preserves_v3_prompt_and_changes_fingerprint(tmp_path):
    instructions = ChatGPTPlanHybridClient._instructions(HYBRID_GRID, "hybrid", 5, 30, 3)
    assert hashlib.sha256(instructions.encode()).hexdigest() == (
        "d12d79a4a40eb885bdf977237cae794d9658849ae06ade1a69f5997858aff862")
    assert hybrid_prompt_version(HYBRID_GRID, 5, 30, 3) == HYBRID_ASYMMETRIC_PROMPT_VERSION
    prediction = wrapper("contact_direct")
    instance = client(FakeTransport(envelope(prediction)), planner_horizon=3, log_dir=tmp_path)
    assert instance.predict(observation().to_dict()) == prediction
    assert instance.planner_horizon == 3
    assert instance.field_execution_steps == 1
    assert instance.prompt_version == HYBRID_ASYMMETRIC_PROMPT_VERSION
    payload = instance.transport.calls[0]
    assert payload["instructions"].startswith(instructions)
    assert payload["text"]["format"]["name"] == "astra_hybrid_history_v3"
    current = client(FakeTransport())
    current_payload = current.build_payload(observation().to_dict())
    assert payload["text"]["format"]["schema"] == current_payload["text"]["format"]["schema"]
    assert instance.request_key(payload, HYBRID_GRID, "hybrid") != current.request_key(
        current_payload, HYBRID_GRID, "hybrid")
    record = json.loads(next(tmp_path.glob("*.request.json")).read_text())
    assert record["prompt_version"] == HYBRID_ASYMMETRIC_PROMPT_VERSION
    assert record["request_key"] == instance.request_key(payload, HYBRID_GRID, "hybrid")


def test_explicit_one_step_execution_preserves_v4_prompt_and_changes_fingerprint(tmp_path):
    instructions = ChatGPTPlanHybridClient._instructions(HYBRID_GRID, "hybrid", 5, 30, 10, 1)
    assert hashlib.sha256(instructions.encode()).hexdigest() == (
        "41b11546045481d20742d26c3588c6e4bb41f801a3c27cbd309439e9899e2126")
    assert hybrid_prompt_version(HYBRID_GRID, 5, 30, 10, 1) == HYBRID_RECEDING_PROMPT_VERSION
    prediction = wrapper("contact_direct")
    instance = client(FakeTransport(envelope(prediction)), field_execution_steps=1, log_dir=tmp_path)
    assert instance.predict(observation().to_dict()) == prediction
    assert instance.field_execution_steps == 1
    assert instance.prompt_version == HYBRID_RECEDING_PROMPT_VERSION
    payload = instance.transport.calls[0]
    assert payload["instructions"].startswith(instructions)
    assert payload["text"]["format"]["name"] == "astra_hybrid_history_v4"
    current = client(FakeTransport())
    current_payload = current.build_payload(observation().to_dict())
    assert payload["text"]["format"]["schema"] == current_payload["text"]["format"]["schema"]
    assert instance.request_key(payload, HYBRID_GRID, "hybrid") != current.request_key(
        current_payload, HYBRID_GRID, "hybrid")
    record = json.loads(next(tmp_path.glob("*.request.json")).read_text())
    assert record["prompt_version"] == HYBRID_RECEDING_PROMPT_VERSION
    assert record["request_key"] == instance.request_key(payload, HYBRID_GRID, "hybrid")


@pytest.mark.parametrize("grid, direct_steps, field_steps, horizon, expected", [
    (GridSpec(), 5, 5, 3, 1), (GridSpec(), 5, 30, 10, 1),
    (HYBRID_GRID, 30, 30, 10, 1), (HYBRID_GRID, 5, 30, 3, 1),
    (HYBRID_GRID, 5, 30, 4, 4), (HYBRID_GRID, 5, 30, 10, 10),
])
def test_execution_defaults_preserve_historical_modes_and_follow_current_horizon(
        grid, direct_steps, field_steps, horizon, expected):
    instance = client(FakeTransport(), grid=grid, chunk_size=direct_steps,
                      field_update_steps=field_steps, planner_horizon=horizon)
    assert instance.field_execution_steps == expected


@pytest.mark.parametrize("execution_steps", [0, -1, 11, True, 10.0, "10", {}])
def test_invalid_execution_window_is_rejected_before_auth(execution_steps):
    provider = Mock(return_value="oauth-unused")
    with pytest.raises(ValueError, match="field_execution_steps"):
        ChatGPTPlanHybridClient(token_provider=provider, transport=FakeTransport(),
                               field_execution_steps=execution_steps)
    with pytest.raises(ValueError, match="field_execution_steps"):
        hybrid_prompt_version(HYBRID_GRID, 5, 30, 10, execution_steps)
    provider.assert_not_called()


def test_execution_window_cannot_exceed_horizon_and_need_not_divide_field_interval():
    with pytest.raises(ValueError, match="field_execution_steps"):
        client(FakeTransport(), planner_horizon=3, field_execution_steps=4)
    instance = client(FakeTransport(), field_update_steps=25, field_execution_steps=10)
    payload = instance.build_payload(observation().to_dict())
    assert "coarse_field interval lasts 25 executed actions" in payload["instructions"]
    assert "executes the next 10 successive" in payload["instructions"]
    assert "field refresh or controller switch discards any unexecuted translation suffix" in payload["instructions"]
    assert "three local plans per complete 30-action field interval" not in payload["instructions"]


@pytest.mark.parametrize("grid, direct_steps, field_steps, expected", [
    (GridSpec(), 5, None, 3), (GridSpec(), 5, 30, 3),
    (HYBRID_GRID, 30, None, 3), (HYBRID_GRID, 5, 5, 3),
    (HYBRID_GRID, 5, None, 10), (HYBRID_GRID, 5, 30, 10),
])
def test_default_horizon_preserves_legacy_and_symmetric_configurations(grid, direct_steps, field_steps, expected):
    instance = client(FakeTransport(), grid=grid, chunk_size=direct_steps, field_update_steps=field_steps)
    assert instance.planner_horizon == expected


@pytest.mark.parametrize("horizon", [1, 6, 7, 10])
def test_explicit_horizon_prompt_describes_actual_search_and_first_move(horizon):
    instance = client(FakeTransport(), planner_horizon=horizon, field_execution_steps=1)
    payload = instance.build_payload(observation().to_dict())
    instructions = payload["instructions"]
    assert f"horizon {horizon}:" in instructions
    assert f"each candidate contains {horizon} one-centimeter moves" in instructions
    assert "only its first move is executed" in instructions
    assert ("Exact lattice dynamic programming" in instructions) == (horizon > 6)
    assert "contact_direct interval lasts 5 executed actions" in instructions
    assert "coarse_field interval lasts 30 executed actions" in instructions
    assert payload["reasoning"] == {"effort": "high"}


@pytest.mark.parametrize("horizon", [0, -1, 11, True, 10.0, "10"])
def test_invalid_planner_horizon_is_rejected_before_auth(horizon):
    provider = Mock(return_value="oauth-unused")
    with pytest.raises(ValueError, match="planner_horizon"):
        ChatGPTPlanHybridClient(token_provider=provider, transport=FakeTransport(), planner_horizon=horizon)
    with pytest.raises(ValueError, match="planner_horizon"):
        hybrid_prompt_version(HYBRID_GRID, 5, 30, horizon)
    provider.assert_not_called()


def test_field_refresh_interval_is_independent_of_direct_schema_and_changes_request_key():
    instance = client(FakeTransport())
    alternate = client(FakeTransport(), field_update_steps=25)
    payload = instance.build_payload(observation().to_dict())
    alternate_payload = alternate.build_payload(observation().to_dict())
    assert payload["text"]["format"]["schema"] == alternate_payload["text"]["format"]["schema"]
    assert instance.request_key(payload, HYBRID_GRID, "hybrid") != alternate.request_key(
        alternate_payload, HYBRID_GRID, "hybrid")
    assert "25 executed actions (1.25 seconds at 20 Hz)" in alternate_payload["instructions"]


@pytest.mark.parametrize("chunk_size", [0, -1, True, 30.0, None])
def test_chunk_configuration_rejects_nonpositive_or_noninteger_sizes(chunk_size):
    with pytest.raises(ValueError):
        client(FakeTransport(), chunk_size=chunk_size)
    with pytest.raises(ValueError):
        hybrid_output_schema(chunk_size=chunk_size)


@pytest.mark.parametrize("field_update_steps", [0, -1, True, 30.0])
def test_field_refresh_configuration_rejects_nonpositive_or_noninteger_sizes(field_update_steps):
    with pytest.raises(ValueError, match="field_update_steps"):
        client(FakeTransport(), field_update_steps=field_update_steps)
    with pytest.raises(ValueError, match="field_update_steps"):
        hybrid_prompt_version(HYBRID_GRID, 5, field_update_steps)


@pytest.mark.parametrize("effort", ["", "HIGH", "none", "unsupported", True, 1, {}, []])
def test_reasoning_effort_rejects_unsupported_configuration_before_auth(effort):
    provider = Mock(return_value="oauth-unused")
    with pytest.raises(ValueError, match="reasoning_effort"):
        ChatGPTPlanHybridClient(token_provider=provider, transport=FakeTransport(),
                               reasoning_effort=effort)
    provider.assert_not_called()


def test_high_reasoning_changes_request_fingerprint_and_none_omits_it():
    high = client(FakeTransport())
    omitted = client(FakeTransport(), reasoning_effort=None)
    high_payload = high.build_payload(observation().to_dict())
    omitted_payload = omitted.build_payload(observation().to_dict())
    assert high_payload["reasoning"] == {"effort": "high"}
    assert "reasoning" not in omitted_payload
    assert {key: value for key, value in high_payload.items() if key != "reasoning"} == omitted_payload
    assert high.request_key(high_payload, HYBRID_GRID, "hybrid") != omitted.request_key(
        omitted_payload, HYBRID_GRID, "hybrid")


@pytest.mark.parametrize("reported", [None, {}, {"effort": None}, {"effort": "high"}])
def test_high_reasoning_reaches_transport_and_logs_without_requiring_response_metadata(tmp_path, reported):
    prediction = wrapper("contact_direct")
    fake = FakeTransport(envelope(prediction, reasoning=reported))
    instance = client(fake, log_dir=tmp_path)
    assert instance.predict(observation().to_dict()) == prediction
    assert fake.calls[0]["reasoning"] == {"effort": "high"}
    assert fake.calls[0]["include"] == ["reasoning.encrypted_content"]
    request = json.loads(next(tmp_path.glob("*.request.json")).read_text())
    assert request["payload"]["reasoning"] == {"effort": "high"}
    assert request["billing"] == "chatgpt_subscription"
    assert instance.budget_snapshot()["api_key_calls"] == 0


def test_reported_effort_mismatch_rejects_without_retry_fallback_or_history_commit(tmp_path):
    prediction = wrapper("contact_direct")
    fake = FakeTransport(envelope(prediction, reasoning={"effort": "high"}),
                         envelope(prediction, reasoning={"effort": "medium"}),
                         envelope(prediction, reasoning={"effort": "high"}))
    instance = client(fake, log_dir=tmp_path)
    instance.predict(observation().to_dict())
    before = instance.history_snapshot()
    with pytest.raises(APIError, match="reasoning effort differs from requested effort") as caught:
        instance.predict(observation(1).to_dict())
    assert caught.value.error_code == "response_reasoning_effort_mismatch"
    assert len(fake.calls) == 2 and len(fake.responses) == 1
    assert instance.history_snapshot() == before
    assert instance.budget_snapshot()["api_key_calls"] == 0
    result = json.loads(sorted(tmp_path.glob("*.result.json"))[-1].read_text())
    assert result["status"] == "error" and result["error_type"] == "APIError"
    assert result["error_code"] == "response_reasoning_effort_mismatch"
    assert "no retry or API-key fallback" in result["error"]


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
    {"direct": {"actions": [[0] * 7] * 6}},
    {"direct": {"actions": [[0] * 7] * 29}},
    {"direct": {"actions": [[0] * 7] * 30}},
    {"direct": {"actions": [[0] * 7] * 31}},
    {"direct": {"actions": [[0] * 6] * 5}},
    {"direct": {"actions": [[0, 0, 0, 0, 0, 0, 1.1]] * 5}},
    {"direct": {"actions": [[0, 0, 0, 0, 0, 0, True]] * 5}},
    {"direct": {"actions": [[0, 0, 0, 0, 0, 0, float("nan")]] * 5}},
    {"direct": {**actions(), "extra": 0}},
])
def test_direct_branch_has_exact_five_by_seven_finite_normalized_contract(change):
    with pytest.raises(SchemaError):
        validate_hybrid_prediction({**wrapper("contact_direct"), **change})


def test_field_branch_has_exact_dense_grid_masks_and_orientation_validation():
    invalid_fields = [
        {**field(), "values": [0]}, {**field(), "orientation_target_xyzw": [0, 0, 0, 0]},
        {**field(), "blocked_indices": [35937]}, {**field(), "blocked_indices": [1], "unknown_indices": [1]},
        {**field(), "success": False},
    ]
    for invalid in invalid_fields:
        with pytest.raises(SchemaError):
            validate_hybrid_prediction({**wrapper(), "field": invalid})
    for invalid in ({}, None, [], {key: value for key, value in wrapper().items() if key != "direct"}):
        with pytest.raises(SchemaError):
            validate_hybrid_prediction(invalid)


def test_preview_has_field_thirty_direct_five_prompt_and_no_credentials():
    provider = Mock(return_value="oauth-unused")
    instance = ChatGPTPlanHybridClient(token_provider=provider, transport=FakeTransport())
    payload = instance.build_payload(observation().to_dict())
    original = AstraClient._instructions(HYBRID_GRID, "field")
    assert "State is " + original.split("State is ", 1)[1] in payload["instructions"]
    assert HYBRID_PROMPT_VERSION in payload["instructions"]
    assert "Routing is reversible" in payload["instructions"]
    assert "Holding the bowl alone does not mean contact_direct forever" in payload["instructions"]
    assert "across BOTH controllers, initially open" in payload["instructions"]
    assert "Direct gripper 0 does not change that remembered" in payload["instructions"]
    assert "not from a numeric potential value" in payload["instructions"]
    assert "at most 500 characters" in payload["instructions"]
    assert "coarse_field interval lasts 30 executed actions (1.5 seconds at 20 Hz)" in payload["instructions"]
    assert "contact_direct interval lasts 5 executed actions (0.25 seconds at 20 Hz)" in payload["instructions"]
    assert "Reuse this fixed-center field for 30 executed actions" in payload["instructions"]
    assert "exactly 5 consecutive actions" in payload["instructions"]
    assert "chunk_index identifies consecutive routing intervals in chronological order" in payload["instructions"]
    assert "30 actions for coarse_field and 5 for contact_direct" in payload["instructions"]
    assert "every 5-action boundary" not in payload["instructions"]
    assert "33 points on each axis with spacing 0.005 meters" in payload["instructions"]
    assert "five-action" not in payload["instructions"] and "FIVE" not in payload["instructions"]
    assert "original field" not in payload["instructions"]
    assert "translation primitives and horizon 10" in payload["instructions"]
    assert "10 one-centimeter moves" in payload["instructions"]
    assert "executes the next 10 successive fixed world-frame translation primitives" in payload["instructions"]
    assert "interval-relative steps 0, 10, and 20" in payload["instructions"]
    assert "three local plans per complete 30-action field interval" in payload["instructions"]
    assert "replans from the new actual measured pose after 10 executed actions" in payload["instructions"]
    assert "Remaining translations are not regenerated or corrected" in payload["instructions"]
    assert "does not repeat the first primitive for every action" in payload["instructions"]
    assert "no complete valid plan exists, translation stays zero for that execution window" in payload["instructions"]
    assert "latest measured robot state on every action" in payload["instructions"]
    assert "records execution feedback after every action" in payload["instructions"]
    assert "replans after each measured action" not in payload["instructions"]
    assert "first move" not in payload["instructions"]
    assert "only its first" not in payload["instructions"]
    assert instance.field_execution_steps == HYBRID_FIELD_EXECUTION_STEPS == 10
    assert "Exact lattice dynamic programming merges equivalent partial paths" in payload["instructions"]
    assert "fixed +/-0.08-meter cube" in payload["instructions"]
    assert "10 cm total path length" in payload["instructions"]
    assert "cannot travel straight 10 cm and must turn" in payload["instructions"]
    assert "horizon three" not in payload["instructions"]
    assert instance.planner_horizon == HYBRID_PLANNER_HORIZON == 10
    assert instance.grid == HYBRID_GRID and instance.chunk_size == 5
    assert instance.field_update_steps == HYBRID_FIELD_UPDATE_STEPS == 30
    assert instance.reasoning_effort == HYBRID_REASONING_EFFORT == "high"
    assert payload["reasoning"] == {"effort": "high"}
    assert instance.prompt_version == HYBRID_PROMPT_VERSION
    assert payload["text"]["format"]["name"] == "astra_hybrid_history_v5"
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
    # Reuse measured feedback; only its routing interval belongs to this hybrid client.
    feedback_client = DirectHistoryAstraClient(api_key="offline-fixture",
        transport=FakeTransport(*(envelope(legacy_actions()) for _ in range(13))))
    feedback_policy = DirectHistoryPolicy(feedback_client)
    obs = observation()
    first_images = (obs.image, obs.wrist_image)
    step = 0
    for batch, prediction in enumerate(predictions):
        assert instance.predict(obs.to_dict()) == prediction
        interval = HYBRID_FIELD_UPDATE_STEPS if prediction["controller"] == "coarse_field" else HYBRID_CHUNK_SIZE
        for offset in range(interval):
            obs, _, feedback = execute(feedback_policy, obs, step)
            feedback.update(chunk_index=batch, chunk_step=offset)
            instance.record_transition(feedback)
            step += 1
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
    feedback_records = [record for record in records if record["type"] == "execution_feedback"]
    assert [record["step"] for record in feedback_records] == list(range(35))
    assert [record["chunk_step"] for record in feedback_records] == list(range(30)) + list(range(5))
    assert [record["chunk_index"] for record in feedback_records] == [0] * 30 + [1] * 5
    images = [part["image_url"] for item in last["input"] for part in item.get("content", []) if part["type"] == "input_image"]
    assert len(images) == 4 and not any(image in images for image in first_images)
    assert instance.history_stats()["feedback_records"] == 65
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
        assert record["request_key"] == instance.request_key(fake.calls[index], HYBRID_GRID, "hybrid")
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
        instance.predict(observation().to_dict(), GridSpec())
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
