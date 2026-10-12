"""Offline protocol, provenance and boundary-history tests for distance control."""

import copy
import json
from unittest.mock import Mock

import pytest

from gpt_field.astra import APIError, MissingCredentials, SchemaError
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS
from gpt_field.compact_subscription import ChatGPTPlanCompactClient, compact_operational_guidance
from gpt_field.distance_subscription import (
    ALIGNMENT_ASSUMPTION, DISTANCE_FIELD_VERSION, DISTANCE_PROMPT_VERSION,
    ChatGPTPlanDistanceClient, compact_distance_feedback_history,
    distance_instructions, distance_output_schema, validate_control_context, validate_control_event,
)
from gpt_field.hybrid_subscription import HYBRID_GRID, HYBRID_KEYS, validate_hybrid_prediction
from test_block_feedback_history import feedback, message, text_records
from test_compact_subscription import hybrid
from test_field_history import FakeTransport, envelope, observation


def context(phase="approach", visual=True):
    return {"phase": phase, "visual_reference": {
        "kind": "end_effector", "moving_point_uv": [.4, .5], "target_point_uv": [.6, .5],
        "alignment_assumption": ALIGNMENT_ASSUMPTION} if visual else None}


def prediction(contact=False, phase="approach", visual=True):
    raw = hybrid(contact)
    if raw["field"] is not None:
        raw["field"]["goal_reference"] = None
    raw["control_context"] = context(phase, visual)
    return raw


def event(step, chunk=0, **updates):
    return {"type": "distance_control_event", "step": step, "chunk_index": chunk,
            "reason": "goal_reached", **updates}


def response(raw, **updates):
    return envelope(raw, model="gpt-6-astra", reasoning={"effort": "medium"}, **updates)


def client(transport=None, **kwargs):
    return ChatGPTPlanDistanceClient(token_provider=lambda: "distance-offline-oauth",
                                    transport=transport, **kwargs)


def test_new_default_contract_is_explicit_and_preview_requires_no_credentials():
    provider, transport = Mock(return_value="unused"), FakeTransport()
    instance = ChatGPTPlanDistanceClient(token_provider=provider, transport=transport)
    payload = instance.build_payload(observation().to_dict())
    assert instance.distance_field_control and instance.field_improvements
    assert instance.field_representation == DISTANCE_FIELD_VERSION
    assert instance.prompt_version == DISTANCE_PROMPT_VERSION
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["model"] == "gpt-6-astra" and payload["stream"] and not payload["store"]
    assert not set(payload).intersection(UNSUPPORTED_FIELDS)
    assert payload["text"]["format"] == {"type": "json_schema", "name": "astra_distance_field_feedback_v7",
                                          "strict": True, "schema": distance_output_schema()}
    assert (instance.chunk_size, instance.field_update_steps, instance.field_execution_steps) == (5, 30, 10)
    assert instance.grid == HYBRID_GRID
    assert not instance.stop_on_model_complete
    assert not transport.calls and not instance.history_snapshot()
    provider.assert_not_called()
    assert instance.budget_snapshot()["api_key_calls"] == 0


@pytest.mark.parametrize("setting,value", [
    ("distance_field_control", False), ("distance_field_control", 1),
    ("controller_mode", "direct"), ("reasoning_effort", "high"),
    ("field_improvements", False), ("execution_mode", "fixed"),
    ("feedback_history", "full"), ("centered_placement_guidance", False),
    ("grasp_settle_guidance", False), ("stop_on_model_complete", True),
])
def test_incompatible_settings_are_rejected_without_transport(setting, value):
    transport = FakeTransport()
    with pytest.raises(ValueError, match="Distance-field control requires"):
        client(transport, **{setting: value})
    assert not transport.calls


def test_prompt_describes_actual_distance_execution_and_observational_limits():
    prompt = distance_instructions()
    for expected in ("26-neighbor", "Dijkstra", "average endpoint density", "120 total actions",
                     "exactly five consecutive", "0.005 m", "goal_reference", "R_target R_current.T",
                     "same_rigid_point_desired_projection", "FIXED EXTERNAL", "valid calibration",
                     "no episode action or time cap", "at most 2 cm", "once per release/recovery epoch",
                     "no task_complete output", "cannot certify LIBERO success"):
        assert expected in prompt
    assert "analytic straight-line" not in prompt and "lattice planning" not in prompt
    assert "uncalibrated preference potential" not in prompt
    assert prompt.count(compact_operational_guidance(centered_placement_guidance=True,
                                                    grasp_settle_guidance=True)) == 1
    schema = distance_output_schema()
    assert set(schema["required"]) == HYBRID_KEYS | {"control_context"}
    assert "task_complete" not in schema["properties"]
    assert not schema["properties"]["control_context"]["additionalProperties"]


@pytest.mark.parametrize("contact", [False, True])
def test_context_is_validated_then_stripped_only_from_normalized_control(contact):
    raw = prediction(contact, "release" if contact else "transport")
    original = copy.deepcopy(raw)
    returned = response(raw)
    transport = FakeTransport(returned, returned)
    instance = client(transport)
    normalized = instance.predict(observation().to_dict())
    validate_hybrid_prediction(normalized)
    assert set(normalized) == HYBRID_KEYS
    assert raw == original and instance.last_compact_prediction == original
    assert instance.last_control_context == raw["control_context"]
    assert instance.history_snapshot() == transport.calls[0]["input"] + returned["output"]
    if contact:
        assert normalized["direct"] == raw["direct"]
    else:
        assert len(normalized["field"]["values"]) == 35937
    instance.last_control_context["phase"] = "recover"
    assert instance.last_compact_prediction == original
    instance.predict(observation(1).to_dict())
    assert returned["output"][0] in transport.calls[1]["input"]
    assert returned["output"][1] in transport.calls[1]["input"]
    assert '"values"' not in json.dumps(transport.calls[1]["input"])
    instance.reset_history()
    assert instance.last_control_context is None and instance.last_compact_prediction is None


@pytest.mark.parametrize("change", [
    {"phase": "complete"}, {"phase": None}, {"success": True},
    {"visual_reference": {"kind": "held_object"}},
    {"visual_reference": {**context()["visual_reference"], "moving_point_uv": [True, .5]}},
    {"visual_reference": {**context()["visual_reference"], "target_point_uv": [float("nan"), .5]}},
    {"visual_reference": {**context()["visual_reference"], "target_point_uv": [1.1, .5]}},
    {"visual_reference": {**context()["visual_reference"], "alignment_assumption": "plate_center"}},
    {"visual_reference": {**context()["visual_reference"], "object_pose": [0, 0, 0]}},
])
def test_context_rejects_hidden_fields_nonfinite_coordinates_and_ambiguous_semantics(change):
    invalid = {**context(), **change}
    with pytest.raises(SchemaError):
        validate_control_context(invalid)


def test_null_visual_reference_is_unavailable_not_a_fabricated_anchor():
    assert validate_control_context(context("verify", False))["visual_reference"] is None
    raw = prediction(phase="verify", visual=False)
    instance = client(FakeTransport(response(raw)))
    instance.predict(observation().to_dict())
    assert instance.last_control_context == context("verify", False)


def test_invalid_response_does_not_commit_context_or_history():
    raw = prediction()
    raw["control_context"]["phase"] = "completed"
    instance = client(FakeTransport(response(raw)))
    with pytest.raises(SchemaError):
        instance.predict(observation().to_dict())
    assert not instance.history_snapshot()
    assert instance.last_control_context is None and instance.last_compact_prediction is None


@pytest.mark.parametrize("updates", [{"model": "different-model"}, {"reasoning": {"effort": "high"}}])
def test_server_model_or_effort_mismatch_retains_no_accepted_context(updates):
    returned = response(prediction())
    returned.update(updates)
    instance = client(FakeTransport(returned))
    with pytest.raises(APIError):
        instance.predict(observation().to_dict())
    assert instance.last_control_context is None and not instance.history_snapshot()


@pytest.mark.parametrize("contact,count", [(False, 1), (False, 30), (False, 31), (False, 120), (True, 5)])
def test_variable_controller_blocks_compact_only_after_observable_boundary(contact, count):
    raw = prediction(contact)
    records = [feedback(i) for i in range(count)]
    original = response(raw)["output"] + [message(record) for record in records]
    assert compact_distance_feedback_history(original) == original
    closing = message(event(count))
    source = original + [closing]
    snapshot = copy.deepcopy(source)
    compacted = compact_distance_feedback_history(source)
    assert source == snapshot
    assert compacted[:2] == source[:2] and compacted[-1] == closing
    texts = text_records(compacted)
    assert texts[0]["type"] == "execution_block_summary" and texts[0]["executed_actions"] == count
    assert texts[-1] == event(count)
    assert compact_distance_feedback_history(compacted) == compacted


@pytest.mark.parametrize("defect", ["missing_step", "wrong_chunk", "missing_measurement", "wrong_event_step",
                                    "field_over_budget", "direct_truncated", "hidden_event_feedback"])
def test_unclosed_or_malformed_blocks_remain_verbatim(defect):
    direct = defect == "direct_truncated"
    count = 121 if defect == "field_over_budget" else 4
    records = [feedback(i) for i in range(count)]
    closing = event(count)
    if defect == "missing_step":
        records[1]["step"] = 0
    elif defect == "wrong_chunk":
        records[2]["chunk_index"] = 7
    elif defect == "missing_measurement":
        del records[1]["measured_translation_norm_m"]
    elif defect == "wrong_event_step":
        closing["step"] += 1
    elif defect == "hidden_event_feedback":
        closing["feedback"] = {"libero_success": False}
    source = response(prediction(direct))["output"] + [message(record) for record in records] + [message(closing)]
    assert compact_distance_feedback_history(source) == source


def test_boundary_history_is_only_committed_after_a_valid_response_and_logs_remain_raw(tmp_path):
    first, next_response = response(prediction()), response(prediction(True, "align"))
    transport = FakeTransport(first, APIError("offline transport fixture"), next_response)
    instance = client(transport, log_dir=tmp_path)
    instance.predict(observation().to_dict())
    for i in range(37):
        instance.record_transition(feedback(i))
    closing = event(37, reason="visual_unavailable", feedback={"status": "match_lost", "reliable": False,
        "calibration_valid": False, "pixel_error_norm": None, "local_goal_updates": 0,
        "task_success_certified": False})
    instance.record_control_event(closing)
    original_history = instance.history_snapshot()
    preview = instance.build_payload(observation(1).to_dict())
    assert instance.history_snapshot() == original_history
    assert any(item.get("executed_actions") == 37 for item in text_records(preview["input"]))
    assert closing in text_records(preview["input"])
    with pytest.raises(APIError):
        instance.predict(observation(1).to_dict())
    assert instance.history_snapshot() == original_history
    instance.predict(observation(1).to_dict())
    assert instance.history_stats()["feedback_records"] == 37
    assert instance.history_stats()["feedback_block_summaries"] == 1
    assert len(transport.calls) == 3
    request = json.loads(sorted(tmp_path.glob("*.request.json"))[-1].read_text())
    assert request["prompt_version"] == DISTANCE_PROMPT_VERSION
    assert request["request_key"] == instance.request_key(request["payload"], HYBRID_GRID, "hybrid")
    assert request["billing"] == "chatgpt_subscription"
    assert "distance-offline-oauth" not in "".join(path.read_text() for path in tmp_path.glob("*.json"))


@pytest.mark.parametrize("changes", [
    {"success": False}, {"step": True}, {"reason": "task_complete"}, {"reason": []},
    {"feedback": {"object_pose": [0, 0, 0]}}, {"feedback": {"task_success_certified": True}},
    {"feedback": {"reliable": 1}}, {"feedback": {"local_goal_updates": -1}},
    {"feedback": {"pixel_error_norm": float("inf")}}, {"feedback": {"status": "x" * 81}},
])
def test_boundary_feedback_rejects_hidden_state_and_false_certification(changes):
    with pytest.raises(ValueError):
        validate_control_event({**event(5), **changes})


def test_boundary_must_follow_acknowledged_feedback_and_cannot_be_duplicated():
    instance = client(FakeTransport(response(prediction(True))))
    with pytest.raises(ValueError):
        instance.record_control_event(event(0))
    instance.predict(observation().to_dict())
    for i in range(5):
        instance.record_transition(feedback(i))
    with pytest.raises(ValueError):
        instance.record_control_event(event(4))
    with pytest.raises(ValueError):
        instance.record_control_event(event(5, chunk=1))
    instance.record_control_event(event(5))
    with pytest.raises(ValueError):
        instance.record_control_event(event(5))


@pytest.mark.parametrize("contact,limit", [(False, 120), (True, 5)])
def test_feedback_extension_is_field_only_and_nominal_interval_stays_thirty(contact, limit):
    instance = client(FakeTransport(response(prediction(contact))))
    instance.predict(observation().to_dict())
    for step in range(limit):
        instance.record_transition(feedback(step))
    assert instance.field_update_steps == 30 and instance.chunk_size == 5
    assert instance.history_stats()["feedback_records"] == limit
    with pytest.raises(ValueError, match="chunk_step exceeds"):
        instance.record_transition(feedback(limit))
    assert instance.history_stats()["feedback_records"] == limit


def test_extended_feedback_still_rejects_hidden_state_and_nonsequential_steps():
    instance = client(FakeTransport(response(prediction())))
    instance.predict(observation().to_dict())
    for bad in ({**feedback(0), "libero_success": False}, feedback(1),
                {**feedback(0), "rolling_progress": {"object_pose": [0, 0, 0]}}):
        with pytest.raises(ValueError):
            instance.record_transition(bad)
    assert instance.history_stats()["feedback_records"] == 0


def test_api_key_is_never_a_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-authorized")
    transport = FakeTransport(response(prediction()))
    instance = ChatGPTPlanDistanceClient(token_provider=lambda: "sk-not-oauth", transport=transport)
    with pytest.raises(MissingCredentials):
        instance.predict(observation().to_dict())
    assert not transport.calls and instance.budget_snapshot()["api_key_calls"] == 0


def test_constructing_new_client_does_not_change_existing_compact_defaults():
    before = ChatGPTPlanCompactClient(token_provider=lambda: "unused").build_payload(observation().to_dict())
    client().build_payload(observation().to_dict())
    after = ChatGPTPlanCompactClient(token_provider=lambda: "unused").build_payload(observation().to_dict())
    assert after == before and after["reasoning"] == {"effort": "high"}
