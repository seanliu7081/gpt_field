"""Offline contracts for compact outputs, local expansion, and subscription guards."""

import copy
import hashlib
import json
from unittest.mock import Mock, patch

import pytest

from gpt_field.astra import APIError, MissingCredentials, SchemaError, output_schema
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS
from gpt_field.compact_field import COMPACT_FIELD_VERSION, compact_field_schema, expand_compact_field
from gpt_field.compact_subscription import (
    ChatGPTPlanCompactClient, compact_instructions, compact_operational_guidance,
    compact_output_schema, compact_prompt_version,
)
from gpt_field.hybrid_subscription import HYBRID_GRID, HYBRID_KEYS, validate_hybrid_prediction
from gpt_field.schema import GridSpec
from test_field_history import FakeTransport, envelope, observation


def compact_field():
    return {"stage": "approach", "orientation_target_xyzw": [0., 0., 0., 1.],
            "gripper_command": "open", "motion_mode": "move", "goal_offset_m": [.05, 0., 0.],
            "attraction_weights": [1., 1., 1.], "repulsors": [], "blocked_boxes": [], "unknown_boxes": []}


def direct():
    return {"actions": [[.1, 0., 0., 0., 0., 0., -1.] for _ in range(5)]}


def hybrid(contact=False):
    return {"controller": "contact_direct" if contact else "coarse_field", "target_region": "black_bowl",
            "near_target": contact, "routing_reason": "Visible bowl rim is beside the gripper.",
            "field": None if contact else compact_field(), "direct": direct() if contact else None}


def client(fake=None, **kwargs):
    return ChatGPTPlanCompactClient(token_provider=lambda: "oauth-compact-fixture", transport=fake, **kwargs)


@pytest.mark.parametrize("mode", ["hybrid", "field", "direct"])
def test_defaults_and_preview_preserve_subscription_controls_without_auth(mode):
    provider = Mock(return_value="unused-oauth")
    fake = FakeTransport()
    instance = ChatGPTPlanCompactClient(controller_mode=mode, token_provider=provider, transport=fake)
    payload = instance.build_payload(observation().to_dict())
    assert instance.grid == HYBRID_GRID
    assert (instance.chunk_size, instance.field_update_steps, instance.planner_horizon,
            instance.field_execution_steps) == (5, 30, 10, 10)
    assert instance.field_representation == COMPACT_FIELD_VERSION
    assert instance.routing_is_model_prediction is (mode == "hybrid")
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "high"}
    assert payload["store"] is False and payload["stream"] is True
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert not set(payload).intersection(UNSUPPORTED_FIELDS)
    assert payload["text"]["format"] == {
        "type": "json_schema", "name": f"astra_compact_{mode}_v1", "strict": True,
        "schema": compact_output_schema(mode),
    }
    assert payload["instructions"] == compact_instructions(mode, instance.image_history)
    assert instance.prompt_version == compact_prompt_version(mode)
    assert not fake.calls and not instance.history_snapshot()
    provider.assert_not_called()


def test_forced_schemas_have_no_routing_or_dense_output_overhead():
    assert compact_output_schema("field") == compact_field_schema()
    assert compact_output_schema("direct") == output_schema(HYBRID_GRID, "direct")
    assert set(compact_output_schema("direct")["required"]) == {"actions"}
    schema = compact_output_schema("hybrid")
    assert schema["properties"]["field"]["anyOf"] == [compact_field_schema(), {"type": "null"}]
    assert "values" not in json.dumps(schema)
    assert "routing_reason" not in json.dumps(compact_output_schema("field"))


def test_prompts_explain_compact_semantics_and_actual_execution_without_contradictions():
    prompt = compact_instructions("field")
    for text in ("33x33x33", "0.005-meter", "fixed +/-0.08-meter", "uncalibrated preference potential",
                 "all ten ordered", "0, 10, 20", "30 executed actions", "every measured state",
                 "neither regenerated nor corrected", "blocked takes precedence", "at most four repulsors",
                 "their own preceding observation", "initially open", "raw accepted"):
        assert text in prompt
    assert "only its first move" not in prompt and "replans after each measured action" not in prompt
    assert "Output an object containing only actions" not in prompt
    assert "repulsors" not in compact_instructions("direct")
    assert "exactly five consecutive" in compact_instructions("direct")
    assert "Routing is reversible" in compact_instructions("hybrid")


@pytest.mark.parametrize("mode,raw", [
    ("field", compact_field()), ("direct", direct()),
    ("hybrid", hybrid()), ("hybrid", hybrid(True)),
])
def test_raw_output_normalizes_for_policy_but_history_remains_raw(mode, raw):
    response = envelope(raw, model="gpt-6-astra", reasoning={"effort": "high"})
    fake = FakeTransport(response, response)
    instance = client(fake, controller_mode=mode)
    normalized = instance.predict(observation().to_dict())
    validate_hybrid_prediction(normalized)
    assert instance.last_compact_prediction == raw
    assert instance.history_snapshot() == fake.calls[0]["input"] + response["output"]
    assert instance.history_snapshot()[-2:]==response["output"]
    if normalized["field"] is not None:
        source = raw["field"] if mode == "hybrid" else raw
        assert normalized["field"] == expand_compact_field(source, HYBRID_GRID)
        assert len(normalized["field"]["values"]) == 35937
        assert instance.last_field_expansion_s >= 0
    else:
        assert normalized["direct"] == (raw["direct"] if mode == "hybrid" else raw)
        assert instance.last_field_expansion_s == 0
    if mode != "hybrid":
        assert "routing was not predicted" in normalized["routing_reason"]
    # Mutating the caller's dense result cannot insert expanded values into history.
    normalized["field"] = {"values": [123.] * 35937}
    instance.predict(observation(1).to_dict())
    assert response["output"][0] in fake.calls[1]["input"]
    assert response["output"][1] in fake.calls[1]["input"]
    assert '"values"' not in json.dumps(fake.calls[1]["input"])
    assert instance.history_stats()["observations"] == 2


def test_expansion_runs_after_transport_and_exposes_separate_latency():
    fake = FakeTransport(envelope(compact_field()))
    instance = client(fake, controller_mode="field")
    def expand(data, grid):
        assert len(fake.calls) == 1
        return expand_compact_field(data, grid)
    with patch("gpt_field.compact_subscription.expand_compact_field", side_effect=expand), \
            patch("gpt_field.compact_subscription.time.perf_counter", side_effect=[10., 10.025]):
        instance.predict(observation().to_dict())
    assert instance.last_field_expansion_s == pytest.approx(.025)


@pytest.mark.parametrize("mode,raw", [
    ("field", hybrid()), ("direct", hybrid(True)),
    ("field", {**compact_field(), "values": [0.]}),
    ("field", {**compact_field(), "attraction_weights": [True, 1., 1.]}),
    ("direct", {"actions": direct()["actions"][:4]}),
    ("hybrid", {**hybrid(), "near_target": True}),
    ("hybrid", {**hybrid(), "direct": direct()}),
])
def test_invalid_raw_response_is_rejected_without_accepting_history(mode, raw):
    instance = client(FakeTransport(envelope(raw)), controller_mode=mode)
    with pytest.raises(SchemaError):
        instance.predict(observation().to_dict())
    assert instance.history_snapshot() == []
    assert instance.last_compact_prediction is None
    assert instance.last_field_expansion_s == 0


@pytest.mark.parametrize("updates", [
    {"model": "unexpected-model"}, {"reasoning": {"effort": "medium"}},
])
def test_inherited_model_and_high_effort_guards_run_before_expansion(updates):
    instance = client(FakeTransport(envelope(compact_field(), **updates)), controller_mode="field")
    with patch("gpt_field.compact_subscription.expand_compact_field") as expand:
        with pytest.raises(APIError):
            instance.predict(observation().to_dict())
    expand.assert_not_called()
    assert instance.last_compact_prediction is None and not instance.history_snapshot()
    assert instance.budget_snapshot()["api_key_calls"] == 0


def test_logs_and_request_keys_identify_raw_protocol(tmp_path):
    raw = compact_field()
    instance = client(FakeTransport(envelope(raw)), controller_mode="field", log_dir=tmp_path)
    instance.predict(observation().to_dict())
    request = json.loads(next(tmp_path.glob("*.request.json")).read_text())
    response = json.loads(next(tmp_path.glob("*.response.json")).read_text())
    assert request["prompt_version"] == instance.prompt_version
    assert request["request_key"] == instance.request_key(request["payload"], HYBRID_GRID, "hybrid")
    assert request["billing"] == "chatgpt_subscription"
    assert json.loads(response["output"][-1]["content"][0]["text"]) == raw
    assert "oauth-compact-fixture" not in "".join(path.read_text() for path in tmp_path.glob("*.json"))
    other = client(FakeTransport(), controller_mode="direct")
    assert other.request_key(request["payload"], HYBRID_GRID, "hybrid") != request["request_key"]


def test_only_oauth_is_accepted_and_environment_api_key_is_ignored(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-authorized")
    fake = FakeTransport(envelope(direct()))
    instance = ChatGPTPlanCompactClient(controller_mode="direct", token_provider=lambda: "sk-not-oauth", transport=fake)
    assert instance._api_key == ""
    with pytest.raises(MissingCredentials):
        instance.predict(observation().to_dict())
    assert not fake.calls and instance.budget_snapshot()["api_key_calls"] == 0


@pytest.mark.parametrize("kwargs", [
    {"controller_mode": "invalid"}, {"grid": GridSpec()}, {"chunk_size": 30},
    {"field_update_steps": 5}, {"planner_horizon": 3}, {"field_execution_steps": 1},
    {"reasoning_effort": None}, {"reasoning_effort": "low"}, {"reasoning_effort": "xhigh"},
])
def test_invalid_compact_configuration_rejected_without_auth(kwargs):
    provider = Mock(return_value="unused-oauth")
    with pytest.raises(ValueError):
        ChatGPTPlanCompactClient(token_provider=provider, transport=FakeTransport(), **kwargs)
    provider.assert_not_called()


def test_diagnostics_reset_on_a_failed_request_after_success():
    instance = client(FakeTransport(envelope(compact_field()), APIError("offline failure")), controller_mode="field")
    instance.predict(observation().to_dict())
    history = instance.history_snapshot()
    with pytest.raises(APIError):
        instance.predict(observation(1).to_dict())
    assert instance.last_compact_prediction is None and instance.last_field_expansion_s == 0
    assert instance.history_snapshot() == history


@pytest.mark.parametrize("mode", ["hybrid", "field", "direct"])
def test_explicit_medium_is_sent_and_validated_without_changing_control_settings(mode):
    raw = hybrid() if mode == "hybrid" else compact_field() if mode == "field" else direct()
    fake = FakeTransport(envelope(raw, model="gpt-6-astra", reasoning={"effort": "medium"}))
    instance = client(fake, controller_mode=mode, reasoning_effort="medium")
    normalized = instance.predict(observation().to_dict())
    validate_hybrid_prediction(normalized)
    assert fake.calls[0]["reasoning"] == {"effort": "medium"}
    assert instance.prompt_version == compact_prompt_version(mode)
    assert instance.budget_snapshot()["api_key_calls"] == 0
    assert instance.last_task_complete is False and instance.last_completion_reason == ""


def test_explicit_medium_rejects_a_high_effort_response_before_expansion():
    instance = client(FakeTransport(envelope(hybrid(), reasoning={"effort": "high"})),
                      reasoning_effort="medium")
    with patch("gpt_field.compact_subscription.expand_compact_field") as expand:
        with pytest.raises(APIError, match="reasoning effort differs") as failure:
            instance.predict(observation().to_dict())
    assert failure.value.error_code == "response_reasoning_effort_mismatch"
    expand.assert_not_called()
    assert instance.history_snapshot() == []
    assert instance.last_task_complete is False and instance.last_completion_reason == ""


def completion_response(*, contact=False, complete=True):
    raw = hybrid(contact)
    raw.update(task_complete=complete, completion_reason=(
        "The bowl rests on the plate, and the released gripper is open and clear." if complete else ""))
    if complete:
        raw["target_region"] = "plate"
        if contact:
            raw["direct"]["actions"] = [[0., 0., 0., 0., 0., 0., -1.] for _ in range(5)]
        else:
            raw["field"].update(motion_mode="hold", gripper_command="open")
    return raw


def test_model_completion_is_explicit_versioned_schema_and_visual_judgment():
    provider = Mock(return_value="unused-oauth")
    instance = ChatGPTPlanCompactClient(token_provider=provider, transport=FakeTransport(),
                                      reasoning_effort="medium", stop_on_model_complete=True)
    payload = instance.build_payload(observation().to_dict())
    assert instance.prompt_version == "astra-compact-field-direct-history-hybrid-completion-v2"
    assert payload["text"]["format"]["name"] == "astra_compact_hybrid_completion_v2"
    schema = payload["text"]["format"]["schema"]
    assert set(schema["required"]) == HYBRID_KEYS | {"task_complete", "completion_reason"}
    assert schema["properties"]["task_complete"] == {"type": "boolean"}
    assert schema["properties"]["completion_reason"] == {"type": "string", "maxLength": 500}
    assert schema["additionalProperties"] is False
    assert "CURRENT visual evidence" in payload["instructions"]
    assert "temporary hold" in payload["instructions"]
    assert "before any actions from that response" in payload["instructions"]
    assert "not a benchmark success label" in payload["instructions"]
    assert "task_complete" not in compact_instructions()
    assert "task_complete" not in compact_output_schema()["properties"]
    legacy = client()
    assert instance.request_key(payload, HYBRID_GRID, "hybrid") != legacy.request_key(
        legacy.build_payload(observation().to_dict()), HYBRID_GRID, "hybrid")
    provider.assert_not_called()


@pytest.mark.parametrize("kwargs", [
    {"controller_mode": "field", "stop_on_model_complete": True},
    {"stop_on_model_complete": 1}, {"stop_on_model_complete": "true"},
])
def test_completion_opt_in_requires_boolean_and_supported_mode(kwargs):
    provider = Mock(return_value="unused-oauth")
    with pytest.raises(ValueError):
        ChatGPTPlanCompactClient(token_provider=provider, **kwargs)
    provider.assert_not_called()


@pytest.mark.parametrize("contact", [False, True])
def test_terminal_response_normalizes_and_commits_raw_completion_evidence(contact):
    raw = completion_response(contact=contact)
    response = envelope(raw, reasoning={"effort": "medium"})
    next_raw = completion_response(complete=False)
    fake = FakeTransport(response, envelope(next_raw, reasoning={"effort": "medium"}))
    instance = client(fake, reasoning_effort="medium", stop_on_model_complete=True)
    normalized = instance.predict(observation().to_dict())
    assert set(normalized) == HYBRID_KEYS
    validate_hybrid_prediction(normalized)
    assert instance.last_task_complete is True
    assert instance.last_completion_reason == raw["completion_reason"]
    assert instance.last_compact_prediction == raw
    assert instance.history_snapshot()[-2:] == response["output"]
    instance.predict(observation(1).to_dict())
    assert response["output"][-1] in fake.calls[1]["input"]
    assert '"values"' not in json.dumps(fake.calls[1]["input"])
    assert instance.last_task_complete is False and instance.last_completion_reason == ""


@pytest.mark.parametrize("updates", [
    {"task_complete": 1}, {"task_complete": "true"}, {"task_complete": None},
    {"completion_reason": None}, {"completion_reason": ""}, {"completion_reason": "  "},
    {"completion_reason": "x" * 501}, {"target_region": "black_bowl"},
])
def test_invalid_completion_fields_cannot_be_accepted_or_leave_a_stop_signal(updates):
    raw = {**completion_response(), **updates}
    instance = client(FakeTransport(envelope(raw)), stop_on_model_complete=True)
    with pytest.raises(SchemaError):
        instance.predict(observation().to_dict())
    assert instance.history_snapshot() == []
    assert instance.last_compact_prediction is None and instance.last_field_expansion_s == 0
    assert instance.last_task_complete is False and instance.last_completion_reason == ""


@pytest.mark.parametrize("omitted", ["task_complete", "completion_reason"])
def test_completion_fields_are_required_on_every_opted_in_response(omitted):
    raw = completion_response(complete=False)
    del raw[omitted]
    instance = client(FakeTransport(envelope(raw)), stop_on_model_complete=True)
    with pytest.raises(SchemaError, match="missing or unexpected"):
        instance.predict(observation().to_dict())
    assert not instance.last_task_complete and not instance.history_snapshot()


@pytest.mark.parametrize("contact", [False, True])
@pytest.mark.parametrize("change", ["motion", "gripper"])
def test_terminal_response_rejects_contradictory_movement_or_closure(contact, change):
    raw = completion_response(contact=contact)
    if contact:
        raw["direct"]["actions"][0][0 if change == "motion" else 6] = .1 if change == "motion" else 1.
    else:
        raw["field"]["motion_mode" if change == "motion" else "gripper_command"] = (
            "move" if change == "motion" else "close")
    instance = client(FakeTransport(envelope(raw)), stop_on_model_complete=True)
    with pytest.raises(SchemaError, match="terminal"):
        instance.predict(observation().to_dict())
    assert instance.last_task_complete is False and instance.last_compact_prediction is None
    assert not instance.history_snapshot()


def test_hold_without_completion_does_not_set_terminal_flag():
    raw = completion_response()
    raw.update(task_complete=False, completion_reason="Waiting for the bowl to settle.")
    instance = client(FakeTransport(envelope(raw)), stop_on_model_complete=True)
    instance.predict(observation().to_dict())
    assert instance.last_task_complete is False
    assert instance.last_completion_reason == raw["completion_reason"]


def test_legacy_client_rejects_completion_fields_without_opt_in():
    instance = client(FakeTransport(envelope(completion_response())))
    with pytest.raises(SchemaError, match="missing or unexpected"):
        instance.predict(observation().to_dict())
    assert not instance.last_task_complete and not instance.history_snapshot()


@pytest.mark.parametrize("failure", [APIError("offline failure"), envelope({"invalid": True})])
def test_completion_state_clears_on_a_failed_request_and_episode_reset(failure):
    instance = client(FakeTransport(envelope(completion_response()), failure,
                                   envelope(completion_response())), stop_on_model_complete=True)
    instance.predict(observation().to_dict())
    assert instance.last_task_complete is True
    accepted_history = copy.deepcopy(instance.history_snapshot())
    with pytest.raises((APIError, SchemaError)):
        instance.predict(observation(1).to_dict())
    assert instance.last_task_complete is False and instance.last_completion_reason == ""
    assert instance.last_compact_prediction is None and instance.last_field_expansion_s == 0
    assert instance.history_snapshot() == accepted_history
    instance.predict(observation(2).to_dict())
    assert instance.last_task_complete is True
    instance.reset_history()
    assert instance.history_snapshot() == []
    assert instance.last_task_complete is False and instance.last_completion_reason == ""
    assert instance.last_compact_prediction is None and instance.last_field_expansion_s == 0


def test_failure_after_validation_clears_completion_before_propagating():
    instance = client(FakeTransport(envelope(completion_response())), stop_on_model_complete=True)
    def fail_result_log(path, data):
        if data.get("status") == "valid":
            raise OSError("offline log failure after validation")
    with patch.object(instance, "_write_log", side_effect=fail_result_log):
        with pytest.raises(APIError):
            instance.predict(observation().to_dict())
    assert instance.last_task_complete is False and instance.last_completion_reason == ""
    assert instance.last_compact_prediction is None and not instance.history_snapshot()


@pytest.mark.parametrize("feedback_history", ["full", "block_summary"])
def test_centered_guidance_is_explicit_versioned_visual_only_and_keeps_direct_schema(feedback_history):
    provider = Mock(return_value="unused-oauth")
    instance = ChatGPTPlanCompactClient(token_provider=provider, transport=FakeTransport(),
        controller_mode="direct", reasoning_effort="medium", feedback_history=feedback_history,
        centered_placement_guidance=True)
    payload = instance.build_payload(observation().to_dict())
    assert instance.centered_placement_guidance is True
    assert instance.stop_on_model_complete is False
    assert instance.prompt_version == (
        f"astra-compact-field-direct-history-direct-fixed-{feedback_history}-centered-placement-v4")
    assert payload["text"]["format"]["name"] == (
        f"astra_compact_direct_fixed_{feedback_history}_centered_placement_v4")
    assert len(payload["text"]["format"]["name"]) <= 64
    assert payload["text"]["format"]["schema"] == compact_output_schema("direct")
    assert set(payload["text"]["format"]["schema"]["required"]) == {"actions"}
    prompt = payload["instructions"]
    assert "Place the bowl centered within the plate rim" in prompt
    assert "Visual overlap" in prompt and "insufficient for centered placement" in prompt
    assert "Before releasing" in prompt and "retracting the open gripper" in prompt
    assert "both RGB camera views" in prompt and "correct any remaining visible offset" in prompt
    assert not any(text in prompt for text in ("3 cm", "0.03", "On(", "plate_1", "akita_black_bowl"))
    assert "exactly five consecutive actions" in prompt and "task_complete" not in prompt
    assert payload["reasoning"] == {"effort": "medium"}
    provider.assert_not_called()


@pytest.mark.parametrize("kwargs", [
    {"centered_placement_guidance": 1}, {"centered_placement_guidance": "true"},
    {"centered_placement_guidance": None},
    {"centered_placement_guidance": True, "controller_mode": "hybrid", "stop_on_model_complete": True},
    {"centered_placement_guidance": True, "controller_mode": "field"},
    {"centered_placement_guidance": True, "controller_mode": "direct", "stop_on_model_complete": True},
])
def test_centered_guidance_invalid_combination_rejected_before_auth(kwargs):
    provider = Mock(return_value="unused-oauth")
    with pytest.raises(ValueError):
        ChatGPTPlanCompactClient(token_provider=provider, transport=FakeTransport(), **kwargs)
    provider.assert_not_called()


@pytest.mark.parametrize("mode,completion,history", [
    ("hybrid", False, "full"), ("hybrid", True, "block_summary"),
    ("field", False, "full"), ("direct", False, "block_summary"), ("direct", True, "full"),
])
def test_disabled_centered_guidance_preserves_existing_protocol(mode, completion, history):
    options = {"stop_on_model_complete": completion, "feedback_history": history}
    assert compact_instructions(mode, **options) == compact_instructions(
        mode, **options, centered_placement_guidance=False)
    assert compact_prompt_version(mode, **options) == compact_prompt_version(
        mode, **options, centered_placement_guidance=False)
    assert "Visual overlap between the bowl and plate" not in compact_instructions(mode, **options)


def test_centered_direct_preserves_action_output_and_records_guidance_provenance(tmp_path):
    raw = direct()
    instance = client(FakeTransport(envelope(raw, reasoning={"effort": "medium"})),
        controller_mode="direct", reasoning_effort="medium", feedback_history="block_summary",
        centered_placement_guidance=True, log_dir=tmp_path)
    result = instance.predict(observation().to_dict())
    assert result["direct"] == raw
    assert instance.last_task_complete is False
    assert instance.last_compact_prediction == raw
    assert instance.history_stats()["feedback_records"] == 0
    request = json.loads(next(tmp_path.glob("*.request.json")).read_text())
    assert request["prompt_version"] == instance.prompt_version
    assert request["payload"]["reasoning"] == {"effort": "medium"}
    assert instance.budget_snapshot()["api_key_calls"] == 0
    previous = client(controller_mode="direct", reasoning_effort="medium", feedback_history="block_summary")
    assert request["request_key"] != previous.request_key(request["payload"], HYBRID_GRID, "hybrid")


@pytest.mark.parametrize("feedback_history", ["full", "block_summary"])
def test_grasp_settle_guidance_is_versioned_robot_feedback_and_rgb_only(feedback_history):
    provider = Mock(return_value="unused-oauth")
    instance = ChatGPTPlanCompactClient(token_provider=provider, transport=FakeTransport(),
        controller_mode="direct", reasoning_effort="medium", feedback_history=feedback_history,
        centered_placement_guidance=True, grasp_settle_guidance=True)
    payload = instance.build_payload(observation().to_dict())
    assert instance.grasp_settle_guidance is True and instance.centered_placement_guidance is True
    assert instance.stop_on_model_complete is False
    assert instance.prompt_version == (
        f"astra-compact-field-direct-history-direct-fixed-{feedback_history}-centered-placement-grasp-settle-v5")
    assert payload["text"]["format"]["name"] == (
        f"astra_compact_direct_fixed_{feedback_history}_grasp_settle_v5")
    assert len(payload["text"]["format"]["name"]) <= 64
    assert payload["text"]["format"]["schema"] == compact_output_schema("direct")
    prompt = payload["instructions"]
    for evidence in ("20 Hz", "0.25 seconds", "does not advance the physics",
                     "multiple five-action chunks", "measured finger joint positions settle",
                     "short test lift", "both RGB camera views", "whether the bowl follows",
                     "After a missed grasp", "visibly observed alignment"):
        assert evidence in prompt
    assert "Place the bowl centered within the plate rim" in prompt
    assert not any(text in prompt for text in ("0.03", "aperture threshold", "plate_1", "akita_black_bowl"))
    assert "exactly five consecutive actions" in prompt and "task_complete" not in prompt
    assert payload["reasoning"] == {"effort": "medium"}
    provider.assert_not_called()


@pytest.mark.parametrize("updates", [
    {"grasp_settle_guidance": 1}, {"grasp_settle_guidance": "true"},
    {"grasp_settle_guidance": None}, {"centered_placement_guidance": False},
    {"controller_mode": "hybrid", "stop_on_model_complete": True}, {"controller_mode": "field"},
    {"stop_on_model_complete": True},
])
def test_grasp_settle_guidance_rejects_unsupported_flags_before_auth(updates):
    provider = Mock(return_value="unused-oauth")
    options = {"controller_mode": "direct", "centered_placement_guidance": True,
               "grasp_settle_guidance": True, **updates}
    with pytest.raises(ValueError):
        ChatGPTPlanCompactClient(token_provider=provider, transport=FakeTransport(), **options)
    provider.assert_not_called()


@pytest.mark.parametrize("centered,history", [(False, "full"), (False, "block_summary"),
                                            (True, "full"), (True, "block_summary")])
def test_disabled_grasp_settle_guidance_leaves_v1_through_v4_unchanged(centered, history):
    options = {"centered_placement_guidance": centered, "feedback_history": history}
    assert compact_instructions("direct", **options) == compact_instructions(
        "direct", **options, grasp_settle_guidance=False)
    assert compact_prompt_version("direct", **options) == compact_prompt_version(
        "direct", **options, grasp_settle_guidance=False)
    assert "measured finger joint positions settle" not in compact_instructions("direct", **options)


def test_grasp_settle_guidance_keeps_model_actions_unmodified_and_auditable(tmp_path):
    raw = direct()
    instance = client(FakeTransport(envelope(raw, reasoning={"effort": "medium"})),
        controller_mode="direct", reasoning_effort="medium", feedback_history="block_summary",
        centered_placement_guidance=True, grasp_settle_guidance=True, log_dir=tmp_path)
    result = instance.predict(observation().to_dict())
    assert result["direct"] == raw
    assert not instance.last_task_complete
    assert instance.last_compact_prediction == raw
    request = json.loads(next(tmp_path.glob("*.request.json")).read_text())
    assert request["prompt_version"] == instance.prompt_version
    assert instance.budget_snapshot()["api_key_calls"] == 0
    v4 = client(controller_mode="direct", feedback_history="block_summary", centered_placement_guidance=True)
    assert request["request_key"] != v4.request_key(request["payload"], HYBRID_GRID, "hybrid")


@pytest.mark.parametrize("grasp", [False, True])
def test_operational_guidance_is_byte_identical_for_hybrid_and_direct(grasp):
    options = {"centered_placement_guidance": True, "grasp_settle_guidance": grasp}
    guidance = compact_operational_guidance(**options)
    prompts = [compact_instructions(mode, execution_mode=execution, feedback_history="block_summary", **options)
               for mode, execution in [("hybrid", "tracking"), ("direct", "fixed")]]
    for prompt in prompts:
        assert prompt.endswith(guidance)
        assert prompt.count(guidance) == 1
        assert prompt[prompt.index("Place the bowl centered within the plate rim."):] == guidance
    assert "Visual overlap" in guidance and "both RGB camera views" in guidance
    assert ("measured finger joint positions settle" in guidance) is grasp
    assert not any(token in guidance for token in ("0.03", "3 cm", "plate_1", "object_pose",
                                                   "contact sensor", "LIBERO", "success signal"))
    assert compact_operational_guidance() == ""


@pytest.mark.parametrize("grasp,expected_sha256", [
    # SHA-256 of exact Direct prompts from direct_medium_libero_success_002/frozen_source.
    (False, "583e19b14ded59323bdba80c7d20baea6d832d5ce568ebd260fa93b5153e7963"),
    (True, "7a82eae7bcaf870e0068afad30a70c05b2a214c962f46787f945d97c6d1bddb9"),
])
def test_existing_direct_guidance_prompt_bytes_match_frozen_v4_and_v5(grasp, expected_sha256):
    prompt = compact_instructions("direct", feedback_history="block_summary",
                                  centered_placement_guidance=True, grasp_settle_guidance=grasp)
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == expected_sha256


@pytest.mark.parametrize("kwargs", [{"centered_placement_guidance": 1}, {"grasp_settle_guidance": "true"},
                                    {"grasp_settle_guidance": True}])
def test_shared_guidance_helper_rejects_invalid_flags(kwargs):
    with pytest.raises(ValueError):
        compact_operational_guidance(**kwargs)


@pytest.mark.parametrize("grasp", [False, True])
def test_hybrid_guidance_version_schema_and_request_controls(grasp):
    instance = client(controller_mode="hybrid", execution_mode="tracking", feedback_history="block_summary",
                      reasoning_effort="medium", centered_placement_guidance=True, grasp_settle_guidance=grasp)
    payload = instance.build_payload(observation().to_dict())
    version = "centered-placement-grasp-settle-v5" if grasp else "centered-placement-v4"
    assert instance.prompt_version == f"astra-compact-field-direct-history-hybrid-tracking-block_summary-{version}"
    suffix = "grasp_settle_v5" if grasp else "centered_v4"
    assert payload["text"]["format"]["name"] == f"astra_compact_hybrid_tracking_block_summary_{suffix}"
    assert len(payload["text"]["format"]["name"]) <= 64
    assert payload["text"]["format"]["schema"] == compact_output_schema("hybrid")
    assert payload["reasoning"] == {"effort": "medium"}
    assert instance.routing_is_model_prediction is True
    assert instance.stop_on_model_complete is False
    assert (instance.chunk_size, instance.field_update_steps, instance.planner_horizon,
            instance.field_execution_steps) == (5, 30, 10, 10)


def test_shared_guidance_preserves_30_field_and_five_direct_actions_between_requests():
    from gpt_field.hybrid_policy import HybridFieldDirectPolicy
    from test_hybrid_policy import execute

    predictions = [hybrid(), hybrid(True), hybrid()]
    for index in (0, 2):
        predictions[index]["field"]["motion_mode"] = "hold"
    fake = FakeTransport(*(envelope(raw, reasoning={"effort": "medium"}) for raw in predictions))
    instance = client(fake, reasoning_effort="medium", centered_placement_guidance=True,
                      grasp_settle_guidance=True, feedback_history="block_summary")
    policy = HybridFieldDirectPolicy(instance)
    obs = observation()
    refresh_steps, planning_steps, controllers = [], [], []
    for step in range(36):
        obs, decision, feedback = execute(policy, obs, step)
        controllers.append(decision.diagnostics["controller"])
        if decision.diagnostics["refreshed"]:
            refresh_steps.append(step)
        if decision.diagnostics.get("plan_refreshed"):
            planning_steps.append(step)
        assert feedback["step"] == step
    assert refresh_steps == [0, 30, 35]
    assert planning_steps == [0, 10, 20, 35]
    assert controllers == ["coarse_field"] * 30 + ["contact_direct"] * 5 + ["coarse_field"]
    assert len(fake.calls) == 3
    assert instance.history_stats()["feedback_records"] == 36
    assert instance.budget_snapshot()["api_key_calls"] == 0
