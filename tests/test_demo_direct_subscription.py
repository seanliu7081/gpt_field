"""Matched demonstration-conditioned Direct contract; no live inference."""

import copy
from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from gpt_field.astra import APIError, AstraConfig, SchemaError
from gpt_field.block_feedback_history import summarize_feedback_block
from gpt_field.demo_direct_subscription import (
    ChatGPTPlanDemoDirectClient, DEMO_DIRECT_KEYS, DEMO_DIRECT_PROMPT_VERSION,
    DEMO_DIRECT_REPRESENTATION, DEMO_DIRECT_SCHEMA_NAME, demo_direct_instructions,
    demo_direct_output_schema, shared_demo_task_guidance,
)
from gpt_field.demonstration import load_demonstration
from gpt_field.distance_subscription import distance_instructions
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.hybrid_subscription import HYBRID_GRID, HYBRID_KEYS
from test_block_feedback_history import feedback, text_records
from test_demonstration import write_bundle
from test_drawer_subscription import drawer_observation
from test_field_history import FakeTransport, envelope, observation


DRAWER = resolve_distance_task("drawer_bowl")


@pytest.fixture
def demo(tmp_path):
    return load_demonstration(write_bundle(tmp_path))


def prediction(*, phase="drawer_open", target="drawer_handle"):
    return {"actions": [[.02 + .01*i, -.01, .005, 0., 0., 0., [-1., 1., 0., 0., -1.][i]]
                        for i in range(5)],
            "target_region": target, "routing_reason": "Current visible handle engagement supports a small pull.",
            "control_context": {"phase": phase, "visual_reference": None}}


def response(raw=None, **updates):
    result = envelope(prediction() if raw is None else raw, model="gpt-6-astra", reasoning={"effort": "medium"})
    result.update(updates)
    return result


def client(demo, transport=None, **kwargs):
    return ChatGPTPlanDemoDirectClient(demonstration=demo, transport=transport,
                                      token_provider=lambda: "offline-demo-direct-oauth", **kwargs)


def test_preview_enforces_matched_inputs_and_direct_only_contract_without_auth(demo):
    provider, transport = Mock(), FakeTransport()
    instance = ChatGPTPlanDemoDirectClient(demonstration=demo, token_provider=provider, transport=transport)
    payload = instance.build_payload(drawer_observation().to_dict())
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
    assert payload["store"] is False and payload["stream"] is True
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert payload["text"]["format"] == {"type": "json_schema", "name": DEMO_DIRECT_SCHEMA_NAME,
        "strict": True, "schema": demo_direct_output_schema()}
    assert payload["instructions"] == demo_direct_instructions()
    assert instance.prompt_version == DEMO_DIRECT_PROMPT_VERSION
    assert instance.field_representation == DEMO_DIRECT_REPRESENTATION
    assert instance.demo_direct_control and not instance.distance_field_control
    assert instance.execution_mode == "fixed" and instance.feedback_history == "block_summary"
    assert instance.chunk_size == 5 and instance.image_history == 2
    assert not instance.field_improvements and not instance.stop_on_model_complete
    assert instance.controller_mode == "direct" and not instance.routing_is_model_prediction
    assert payload["input"][0] == demo.message()
    current = json.loads(payload["input"][1]["content"][0]["text"])
    assert current["prompt"] == DRAWER.prompt and current["observation_index"] == 0
    assert instance.history_snapshot() == [] and not transport.calls
    provider.assert_not_called()


def test_task_demo_grasp_and_recovery_guidance_is_verbatim_shared_with_hybrid():
    hybrid = distance_instructions(task_key="drawer_bowl", demonstration=True)
    direct = demo_direct_instructions()
    spans = shared_demo_task_guidance()
    assert {"grasp_settling", "placement", "recovery_and_stopping", "demonstration_reference"} <= set(spans)
    for text in spans.values():
        assert text in hybrid and text in direct
    assert "does NOT call for a vertical test lift" in direct
    assert "For the BOWL grasp only" in direct
    assert "Do not impose a closed-finger handle grasp" in direct
    assert "Closing the drawer is not required" in direct
    assert "No distance field, local motion planner, local visual servo" in direct
    assert "Choose any reobservation or recovery movement explicitly" in direct
    assert "Only the external evaluator can stop on actual LIBERO success" in direct
    assert "Continue to choose the normal Field/Direct schema" not in direct
    assert "For coarse_field, output stage" not in direct


def test_schema_requests_exactly_five_actions_and_no_field_or_completion_output():
    schema = demo_direct_output_schema()
    assert set(schema["properties"]) == set(schema["required"]) == DEMO_DIRECT_KEYS
    assert schema["additionalProperties"] is False
    assert schema["properties"]["actions"]["minItems"] == schema["properties"]["actions"]["maxItems"] == 5
    action = schema["properties"]["actions"]["items"]
    assert action["minItems"] == action["maxItems"] == 7
    assert schema["properties"]["target_region"]["enum"] == list(DRAWER.target_regions)
    assert schema["properties"]["control_context"]["properties"]["phase"]["enum"] == list(DRAWER.control_phases)
    assert not {"controller", "field", "task_complete"}.intersection(schema["properties"])


@pytest.mark.parametrize("phase,target", [
    ("approach", "drawer_handle"), ("handle_grasp", "drawer_handle"), ("drawer_open", "drawer_handle"),
    ("handle_release", "drawer_handle"), ("grasp", "black_bowl"), ("transport", "drawer_interior"),
    ("align", "drawer_interior"), ("release", "drawer_interior"), ("verify", "drawer_interior"),
    ("recover", "black_bowl"),
])
def test_every_phase_normalizes_to_unchanged_direct_actions_without_expansion(demo, phase, target):
    raw = prediction(phase=phase, target=target)
    source = copy.deepcopy(raw)
    returned = response(raw)
    transport = FakeTransport(returned)
    instance = client(demo, transport)
    normalized = instance.predict(drawer_observation().to_dict())
    assert set(normalized) == HYBRID_KEYS
    assert normalized["controller"] == "contact_direct" and normalized["field"] is None
    assert normalized["direct"]["actions"] == raw["actions"] == source["actions"]
    assert normalized["target_region"] == target
    assert instance.last_compact_prediction == raw and instance.last_control_context == raw["control_context"]
    assert instance.last_effective_compact_field is None and instance.last_goal_grounding is None
    assert instance.last_field_expansion_s == 0 and instance.last_task_complete is False
    assert instance.history_snapshot()[-2:] == returned["output"]
    assert instance.budget_snapshot()["api_key_calls"] == 0
    normalized["direct"]["actions"][0][0] = .9
    assert instance.last_compact_prediction == source


@pytest.mark.parametrize("change", [
    lambda d: d.update(task_complete=True), lambda d: d.update(field={}),
    lambda d: d.pop("control_context"), lambda d: d["actions"].pop(),
    lambda d: d["actions"].append([0.]*7), lambda d: d["actions"][0].pop(),
    lambda d: d["actions"][0].__setitem__(0, 1.1),
    lambda d: d["actions"][0].__setitem__(0, True),
    lambda d: d["actions"][0].__setitem__(0, float("nan")),
    lambda d: d.update(target_region="plate"), lambda d: d.update(routing_reason=" "),
    lambda d: d["control_context"].update(phase="completed"),
    lambda d: d["control_context"].update(object_pose=[0., 0., 0.]),
    lambda d: d["control_context"].update(phase="grasp"),
    lambda d: d["control_context"].update(phase="release"),
    lambda d: d.update(target_region="black_bowl"),
])
def test_invalid_model_outputs_never_commit_history_or_diagnostics(demo, change):
    raw = prediction()
    change(raw)
    instance = client(demo, FakeTransport(response(raw)))
    with pytest.raises(SchemaError):
        instance.predict(drawer_observation().to_dict())
    assert instance.history_snapshot() == []
    assert instance.last_control_context is None and instance.last_compact_prediction is None
    assert not instance.last_task_complete


def test_optional_visual_reference_is_preserved_without_local_calibration(demo):
    raw = prediction(phase="align", target="drawer_interior")
    raw["control_context"]["visual_reference"] = {
        "kind": "held_object", "moving_point_uv": [.25, .4], "target_point_uv": [.3, .5],
        "alignment_assumption": "same_rigid_point_desired_projection"}
    instance = client(demo, FakeTransport(response(raw)))
    instance.predict(drawer_observation().to_dict())
    assert instance.last_control_context == raw["control_context"]
    assert instance.last_goal_grounding is None
    assert not hasattr(instance, "record_control_event")


def test_demo_survives_live_image_trimming_and_complete_feedback_compaction(demo):
    replies = [response(id=f"direct-{i}") for i in range(4)]
    transport = FakeTransport(*replies)
    instance = client(demo, transport)
    all_records = []
    for chunk in range(4):
        instance.predict(drawer_observation(chunk).to_dict())
        if chunk < 3:
            records = [feedback(chunk*5+i, chunk=chunk, chunk_step=i) for i in range(5)]
            all_records.append(records)
            for record in records:
                instance.record_transition(record)
    for i, payload in enumerate(transport.calls):
        assert payload["input"][0] == demo.message()
        assert payload["input"].count(demo.message()) == 1
        live = payload["input"][1:]
        images = [p for m in live for p in m.get("content", []) if p.get("type") == "input_image"]
        assert len(images) == 2*min(i+1, 2)
        summaries = [r for r in text_records(live) if r["type"] == "execution_block_summary"]
        assert summaries == [summarize_feedback_block(records) for records in all_records[:i]]
    history = instance.history_snapshot()
    assert demo.message() not in history
    assert [m for m in history if m.get("type") == "reasoning"] == [r["output"][0] for r in replies]
    assert instance.history_stats()["observations"] == 4
    assert instance.history_stats()["retained_image_pairs"] == 2
    assert instance.history_stats()["feedback_records"] == 15
    assert instance.history_stats()["feedback_block_summaries"] == 3
    instance.reset_history()
    assert instance.history_snapshot() == [] and instance.last_control_context is None
    assert instance.build_payload(drawer_observation().to_dict())["input"][0] == demo.message()


def test_partial_feedback_stays_raw_and_failed_inference_does_not_commit_compaction(demo):
    transport = FakeTransport(response(), APIError("offline failure"), response())
    instance = client(demo, transport)
    instance.predict(drawer_observation().to_dict())
    records = [feedback(i) for i in range(5)]
    for record in records[:4]:
        instance.record_transition(record)
    partial = instance.build_payload(drawer_observation(1).to_dict())
    assert [r for r in text_records(partial["input"][1:]) if r["type"] == "execution_feedback"] == records[:4]
    instance.record_transition(records[-1])
    history = instance.history_snapshot()
    with pytest.raises(APIError, match="offline failure"):
        instance.predict(drawer_observation(1).to_dict())
    assert instance.history_snapshot() == history
    instance.predict(drawer_observation(1).to_dict())
    assert instance.history_stats()["feedback_block_summaries"] == 1
    assert transport.calls[1] == transport.calls[2]


@pytest.mark.parametrize("updates", [{"model": "gpt-other"}, {"reasoning": {"effort": "high"}}])
def test_server_model_and_effort_mismatch_cannot_be_accepted(demo, updates):
    instance = client(demo, FakeTransport(response(**updates)))
    with pytest.raises(APIError):
        instance.predict(drawer_observation().to_dict())
    assert instance.history_snapshot() == [] and instance.last_control_context is None


@pytest.mark.parametrize("changes", [
    {"task_key": "bowl_plate"}, {"demonstration": None}, {"demonstration": {}},
    {"controller_mode": "hybrid"}, {"reasoning_effort": "high"}, {"reasoning_effort": None},
    {"field_improvements": True}, {"execution_mode": "tracking"}, {"distance_field_control": True},
    {"feedback_history": "full"}, {"image_history": 1}, {"image_history": 2.0},
    {"centered_placement_guidance": False}, {"grasp_settle_guidance": False},
    {"stop_on_model_complete": True}, {"task_prompt": "different drawer task"},
    {"chunk_size": 10}, {"config": replace(AstraConfig(), model="gpt-other")},
])
def test_unmatched_conditions_fail_before_auth_or_transport(demo, changes):
    provider, transport = Mock(), FakeTransport()
    kwargs = {"demonstration": demo, "token_provider": provider, "transport": transport, **changes}
    with pytest.raises((ValueError, TypeError)):
        ChatGPTPlanDemoDirectClient(**kwargs)
    provider.assert_not_called()
    assert transport.calls == []


def test_wrong_task_observation_or_mutated_prefix_is_rejected(demo):
    instance = client(demo)
    with pytest.raises(ValueError):
        instance.build_payload(observation().to_dict())
    payload = instance.build_payload(drawer_observation().to_dict())
    payload["input"][0]["content"].clear()
    with pytest.raises(ValueError, match="prefix"):
        instance._history_input(payload)
    assert demo.message()["content"]
    assert instance.build_payload(drawer_observation().to_dict())["input"][0] == demo.message()
