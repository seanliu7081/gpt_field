"""Matched no-demo collision guidance and observable-only subscription inputs."""

import copy
from dataclasses import replace
import hashlib
import json
from unittest.mock import Mock

import pytest

from gpt_field.astra import APIError, AstraConfig, SchemaError
from gpt_field.block_feedback_history import summarize_feedback_block
from gpt_field.demo_direct_subscription import demo_direct_instructions, shared_demo_task_guidance
from gpt_field.distance_subscription import (
    ChatGPTPlanDistanceClient, ZERO_SHOT_DISTANCE_PROMPT_VERSION, ZERO_SHOT_DISTANCE_SCHEMA_NAME,
    collision_free_guidance, distance_instructions, distance_output_schema, distance_prompt_version,
    distance_schema_name, hybrid_collision_field_guidance, shared_zero_shot_task_guidance,
)
from gpt_field.hybrid_subscription import HYBRID_GRID, HYBRID_KEYS
from gpt_field.zero_shot_direct_subscription import (
    ChatGPTPlanZeroShotDirectClient, ZERO_SHOT_DIRECT_KEYS, ZERO_SHOT_DIRECT_PROMPT_VERSION,
    ZERO_SHOT_DIRECT_REPRESENTATION, ZERO_SHOT_DIRECT_SCHEMA_NAME,
    zero_shot_direct_instructions, zero_shot_direct_output_schema,
)
from test_block_feedback_history import feedback
from test_distance_subscription import prediction as hybrid_prediction
from test_field_history import FakeTransport, envelope, observation


def raw_direct(phase="approach", target="black_bowl"):
    return {
        "actions": [[.02+i*.01, -.01, .005, 0., 0., 0., [-1., 1., 0., 0., -1.][i]] for i in range(5)],
        "target_region": target, "routing_reason": "The current images support a clear bounded approach.",
        "control_context": {"phase": phase, "visual_reference": None},
    }


def response(raw=None, **updates):
    result = envelope(raw_direct() if raw is None else raw, model="gpt-6-astra", reasoning={"effort": "medium"})
    result.update(updates)
    return result


def direct(transport=None, **kwargs):
    return ChatGPTPlanZeroShotDirectClient(token_provider=lambda: "offline-zero-shot-oauth",
                                         transport=transport, **kwargs)


def hybrid(transport=None, **kwargs):
    return ChatGPTPlanDistanceClient(token_provider=lambda: "offline-zero-shot-oauth",
                                    transport=transport, zero_shot_collision_free=True, **kwargs)


@pytest.mark.parametrize("task,demo,expected", [
    ("bowl_plate", False, "22983492324f16da877d257e196e2b0913ab2de88c357699e1b1d253f913e810"),
    ("bowl_plate", True, "ca6e403a73e1ac67977fb0a895a84f4d7b151d903bb6de257a356bc563711ddb"),
    ("drawer_bowl", False, "e4201a441077d0aefaf03b99daf0526716a9c3c34ef154a582f5e4a8a967d00c"),
    ("drawer_bowl", True, "729cca753569b417ea56861999867883af7ca03c4b4d3331c75941423c195a61"),
])
def test_historical_distance_prompts_remain_byte_for_byte_unchanged(task, demo, expected):
    prompt = distance_instructions(task_key=task, demonstration=demo)
    assert hashlib.sha256(prompt.encode()).hexdigest() == expected
    assert distance_instructions(task_key=task, demonstration=demo, zero_shot_collision_free=False) == prompt


@pytest.mark.parametrize("task,expected", [
    ("bowl_plate", "2fe6ad0d0d65888d6ddc437ecad261ce56c34d477d25cdda1e15e5c3bd6b1265"),
    ("drawer_bowl", "9becc53aab90ef1cada460ba170eaa6550819c8b20bff9c00b241f0200f873e4"),
])
def test_historical_demo_direct_prompts_remain_byte_for_byte_unchanged(task, expected):
    assert hashlib.sha256(demo_direct_instructions(task_key=task).encode()).hexdigest() == expected


def test_collision_and_operational_guidance_is_exactly_shared_without_demo_advice():
    h = distance_instructions(zero_shot_collision_free=True)
    d = zero_shot_direct_instructions()
    common = shared_zero_shot_task_guidance()
    for value in common.values():
        assert h.count(value) == 1 and d.count(value) == 1
    old = shared_demo_task_guidance(task_key="bowl_plate")
    for key in ("scene_clearance", "grasp_settling", "placement", "reobservation",
                "post_release_verification", "recovery_and_stopping"):
        assert common[key] == old[key]
    for phrase in ("full robot", "held bowl", "ketchup bottle", "milk carton", "ramekin", "cabinet",
                   "stove", "table", "finger-bowl grasp", "bowl-table or bowl-plate support",
                   "swept volume", "Do not push, tip", "Occluded or uncertain space is not free space",
                   "then replan", "ground-truth object poses"):
        assert phrase in collision_free_guidance()
    assert hybrid_collision_field_guidance() in h and hybrid_collision_field_guidance() not in d
    assert "Do not return all-empty" in h and "Inflate the obstacle estimate" in h
    assert "soft repulsor does not replace a blocked solid obstacle" in h
    for prompt in (h, d):
        assert "zero-shot control: no demonstration" in prompt
        assert "external_demonstration" not in prompt and "Study its approach" not in prompt
        assert "reference stays fixed" not in prompt and "demonstration stays separate" not in prompt
        assert "no episode action or time cap" in prompt


@pytest.mark.parametrize("factory,is_direct", [(direct, True), (hybrid, False)])
def test_preview_has_only_current_observation_and_no_auth_or_demo(factory, is_direct):
    provider = Mock()
    cls = ChatGPTPlanZeroShotDirectClient if is_direct else ChatGPTPlanDistanceClient
    instance = cls(token_provider=provider, zero_shot_collision_free=True)
    payload = instance.build_payload(observation().to_dict())
    assert len(payload["input"]) == 1
    parts = payload["input"][0]["content"]
    value = json.loads(parts[0]["text"])
    assert set(value) == {"type", "observation_index", "prompt", "observation/state"}
    assert value["type"] == "observation" and value["observation_index"] == 0
    assert len([part for part in parts if part["type"] == "input_image"]) == 2
    assert "external_demonstration" not in json.dumps(payload["input"])
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
    assert payload["stream"] is True and payload["store"] is False
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert instance.demonstration is None and instance.zero_shot_collision_free is True
    assert instance.chunk_size == 5 and instance.image_history == 2
    assert not instance.stop_on_model_complete and instance.feedback_history == "block_summary"
    assert instance.history_snapshot() == [] and instance.budget_snapshot()["api_key_calls"] == 0
    provider.assert_not_called()
    if is_direct:
        assert instance.zero_shot_direct_control and not instance.demo_direct_control
        assert not instance.distance_field_control and instance.execution_mode == "fixed"
        assert instance.prompt_version == ZERO_SHOT_DIRECT_PROMPT_VERSION
        assert instance.field_representation == ZERO_SHOT_DIRECT_REPRESENTATION
        assert payload["instructions"] == zero_shot_direct_instructions()
        assert payload["text"]["format"] == {"type": "json_schema", "name": ZERO_SHOT_DIRECT_SCHEMA_NAME,
            "strict": True, "schema": zero_shot_direct_output_schema()}
    else:
        assert instance.prompt_version == ZERO_SHOT_DISTANCE_PROMPT_VERSION
        assert payload["text"]["format"] == {"type": "json_schema", "name": ZERO_SHOT_DISTANCE_SCHEMA_NAME,
            "strict": True, "schema": distance_output_schema()}


@pytest.mark.parametrize("factory", [direct, hybrid])
@pytest.mark.parametrize("extra", ["object_pose", "contacts", "reward", "success", "demonstration", "depth"])
def test_hidden_observation_inputs_fail_before_transport(factory, extra):
    transport = FakeTransport()
    instance = factory(transport)
    obs = observation().to_dict(); obs[extra] = []
    with pytest.raises(ValueError):
        instance.predict(obs)
    assert not transport.calls and not instance.history_snapshot()


@pytest.mark.parametrize("factory", [direct, hybrid])
@pytest.mark.parametrize("demo", [{}, {"type": "external_demonstration"}, object()])
def test_all_non_none_demo_inputs_are_rejected_before_auth(factory, demo):
    with pytest.raises(ValueError, match="demonstration"):
        factory(demonstration=demo)


@pytest.mark.parametrize("helper", [distance_instructions, distance_prompt_version])
def test_demo_prompt_flag_cannot_be_combined_with_zero_shot(helper):
    with pytest.raises(ValueError, match="demonstration"):
        helper(demonstration=True, zero_shot_collision_free=True)


@pytest.mark.parametrize("factory", [direct, hybrid])
@pytest.mark.parametrize("changes", [
    {"task_key": "drawer_bowl"}, {"reasoning_effort": "high"}, {"image_history": 1},
    {"image_history": 2.0}, {"feedback_history": "full"}, {"chunk_size": 10},
    {"centered_placement_guidance": False}, {"grasp_settle_guidance": False},
    {"stop_on_model_complete": True}, {"task_prompt": "another task"},
    {"config": replace(AstraConfig(), model="another-model")},
])
def test_unmatched_settings_are_rejected(factory, changes):
    transport = FakeTransport()
    with pytest.raises((ValueError, TypeError)):
        factory(transport, **changes)
    assert transport.calls == []


@pytest.mark.parametrize("value", [1, "true", None])
def test_zero_shot_flag_requires_a_real_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        distance_instructions(zero_shot_collision_free=value)
    with pytest.raises(ValueError):
        ChatGPTPlanDistanceClient(token_provider=Mock(), zero_shot_collision_free=value)
    with pytest.raises(ValueError):
        direct(zero_shot_collision_free=value)


@pytest.mark.parametrize("phase,target", [
    ("approach", "black_bowl"), ("grasp", "black_bowl"), ("transport", "plate"),
    ("align", "plate"), ("release", "plate"), ("verify", "plate"), ("recover", "black_bowl"),
])
def test_direct_preserves_five_model_actions_and_never_generates_a_field(phase, target):
    raw = raw_direct(phase, target);original = copy.deepcopy(raw)
    instance = direct(FakeTransport(response(raw)))
    result = instance.predict(observation().to_dict())
    assert set(result) == HYBRID_KEYS
    assert result["controller"] == "contact_direct" and result["field"] is None
    assert result["direct"]["actions"] == raw["actions"] == original["actions"]
    assert instance.last_compact_prediction == original and instance.last_control_context == original["control_context"]
    assert instance.last_field_expansion_s == 0 and instance.last_effective_compact_field is None
    assert instance.last_goal_grounding is None and not instance.last_task_complete
    assert not hasattr(instance, "record_control_event")
    result["direct"]["actions"][0][0] = .9
    assert instance.last_compact_prediction == original


@pytest.mark.parametrize("change", [
    lambda d: d.update(task_complete=True), lambda d: d.update(field={}),
    lambda d: d.update(object_pose=[0.,0.,0.]), lambda d: d.pop("control_context"),
    lambda d: d["actions"].pop(), lambda d: d["actions"].append([0.]*7),
    lambda d: d["actions"][0].__setitem__(0, True),
    lambda d: d["actions"][0].__setitem__(0, float("nan")),
    lambda d: d["actions"][0].__setitem__(0, 1.1),
    lambda d: d["control_context"].update(object_pose=[0.,0.,0.]),
    lambda d: d["control_context"].update(phase="handle_grasp"),
    lambda d: d.update(target_region="drawer_handle"),
    lambda d: d["control_context"].update(phase="release"),
])
def test_invalid_direct_outputs_do_not_commit_history(change):
    raw = raw_direct();change(raw)
    instance = direct(FakeTransport(response(raw)))
    with pytest.raises(SchemaError):
        instance.predict(observation().to_dict())
    assert instance.history_snapshot() == []
    assert instance.last_control_context is None and instance.last_compact_prediction is None


def test_hybrid_preserves_explicit_constraints_and_keeps_dense_grid_out_of_history():
    raw = hybrid_prediction(visual=False)
    raw["field"]["blocked_boxes"] = [{"lower_offset_m": [-.07,-.02,-.02], "upper_offset_m": [-.04,.02,.02]}]
    raw["field"]["unknown_boxes"] = [{"lower_offset_m": [.04,-.02,-.02], "upper_offset_m": [.07,.02,.02]}]
    raw["field"]["repulsors"] = [{"center_offset_m": [.03,0.,0.], "radius_m": .02, "weight": 1.}]
    transport = FakeTransport(response(raw), response(raw))
    instance = hybrid(transport)
    normalized = instance.predict(observation().to_dict())
    assert len(normalized["field"]["values"]) == 35937
    assert instance.last_compact_prediction == raw
    instance.predict(observation(1).to_dict())
    assert '"values"' not in json.dumps(transport.calls[1]["input"])
    assert "external_demonstration" not in json.dumps(transport.calls[1]["input"])
    assert collision_free_guidance() in transport.calls[1]["instructions"]


def test_live_history_compacts_five_action_feedback_without_any_reference_prefix():
    replies = [response(id=f"zero-shot-{i}") for i in range(4)]
    transport = FakeTransport(*replies);instance = direct(transport)
    blocks = []
    for chunk in range(4):
        instance.predict(observation(chunk).to_dict())
        if chunk < 3:
            records = [feedback(chunk*5+i,chunk=chunk,chunk_step=i) for i in range(5)]
            blocks.append(records)
            for record in records:instance.record_transition(record)
    for i,payload in enumerate(transport.calls):
        records = [json.loads(message["content"][0]["text"]) for message in payload["input"]
                   if message.get("role") == "user"]
        assert records[0]["type"] == "observation" and records[0]["observation_index"] == 0
        assert not any(r.get("type") == "external_demonstration" for r in records)
        assert [r for r in records if r["type"] == "execution_block_summary"] == [summarize_feedback_block(b) for b in blocks[:i]]
        assert len([c for m in payload["input"] for c in m.get("content",[]) if c.get("type")=="input_image"]) == 2*min(i+1,2)
    assert instance.history_stats()["observations"] == 4
    assert instance.history_stats()["feedback_block_summaries"] == 3
    instance.reset_history()
    assert instance.history_snapshot() == [] and instance.last_control_context is None
    assert len(instance.build_payload(observation().to_dict())["input"]) == 1


@pytest.mark.parametrize("factory,raw", [(direct, raw_direct()), (hybrid, hybrid_prediction(visual=False))])
def test_hidden_feedback_cannot_be_sent_back_to_model(factory, raw):
    instance = factory(FakeTransport(response(raw)))
    instance.predict(observation().to_dict());before = instance.history_snapshot()
    record = feedback(0);record["object_pose"] = [0.,0.,0.]
    with pytest.raises(ValueError):instance.record_transition(record)
    assert instance.history_snapshot() == before


@pytest.mark.parametrize("updates", [{"model": "another-model"}, {"reasoning": {"effort": "high"}}])
def test_direct_server_model_and_effort_mismatch_rejected(updates):
    instance = direct(FakeTransport(response(**updates)))
    with pytest.raises(APIError):instance.predict(observation().to_dict())
    assert instance.history_snapshot() == [] and instance.last_compact_prediction is None


def test_direct_schema_contains_five_actions_and_no_completion_or_hidden_fields():
    schema = zero_shot_direct_output_schema()
    assert set(schema["required"]) == set(schema["properties"]) == ZERO_SHOT_DIRECT_KEYS
    assert schema["properties"]["actions"]["minItems"] == schema["properties"]["actions"]["maxItems"] == 5
    assert not schema["additionalProperties"]
    assert distance_schema_name(zero_shot_collision_free=True) == ZERO_SHOT_DISTANCE_SCHEMA_NAME
