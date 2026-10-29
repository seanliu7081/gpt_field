"""Original task guidance and unchanged Direct5, tested without live inference."""
import copy
from dataclasses import replace
import hashlib
import json
from unittest.mock import Mock

import pytest

from gpt_field.astra import APIError, AstraConfig, SchemaError
from gpt_field.block_feedback_history import summarize_feedback_block
from gpt_field.demo_direct_subscription import demo_direct_instructions
from gpt_field.distance_subscription import collision_free_guidance, distance_instructions
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.historical_zero_shot_direct import (
    ChatGPTPlanHistoricalZeroShotDirectClient, HISTORICAL_ZERO_SHOT_DIRECT_KEYS,
    historical_zero_shot_direct_instructions, historical_zero_shot_direct_output_schema,
    historical_zero_shot_direct_prompt_version, historical_zero_shot_direct_representation,
    historical_zero_shot_direct_schema_name, shared_historical_zero_shot_task_guidance,
)
from gpt_field.hybrid_policy import HybridFieldDirectPolicy
from gpt_field.hybrid_subscription import HYBRID_KEYS
from test_block_feedback_history import feedback, text_records
from test_drawer_subscription import drawer_observation
from test_field_history import FakeTransport, envelope, observation


TASKS = ("bowl_plate", "drawer_bowl")


def obs(task, index=0):
    return drawer_observation(index) if task == "drawer_bowl" else observation(index)


def prediction(task, phase=None, target=None):
    return {"actions": [[.03+i*.01, -.01, .005, -.005, .01, 0., [-1., 1., 0., 0., -1.][i]]
                        for i in range(5)],
            "target_region": target or ("drawer_handle" if task == "drawer_bowl" else "black_bowl"),
            "routing_reason": "The current images support the next measured-pose action block.",
            "control_context": {"phase": phase or ("drawer_open" if task == "drawer_bowl" else "approach"),
                                "visual_reference": None}}


def response(task, raw=None, **updates):
    result = envelope(prediction(task) if raw is None else raw, model="gpt-6-astra", reasoning={"effort": "medium"})
    result.update(updates)
    return result


def client(task="bowl_plate", transport=None, **kwargs):
    return ChatGPTPlanHistoricalZeroShotDirectClient(task_key=task, transport=transport,
        token_provider=lambda: "offline-historical-oauth-fixture", **kwargs)


@pytest.mark.parametrize("task", TASKS)
def test_preview_has_exact_historical_task_guidance_and_no_demo_or_new_collision_prompt(task):
    provider = Mock()
    instance = ChatGPTPlanHistoricalZeroShotDirectClient(task_key=task, token_provider=provider)
    payload = instance.build_payload(obs(task).to_dict())
    original = distance_instructions(task_key=task, demonstration=False, zero_shot_collision_free=False)
    assert payload["instructions"] == historical_zero_shot_direct_instructions(task_key=task)
    for value in shared_historical_zero_shot_task_guidance(task_key=task).values():
        assert value in original and value in payload["instructions"]
    assert collision_free_guidance() not in payload["instructions"]
    for forbidden in ("external reference demonstration", "demonstration stays separate", "immutable external",
                      "The first input message", "Study its approach", "J(q)", "tool_point_uv",
                      "For coarse_field, output stage", "full robot, including the arm"):
        assert forbidden not in payload["instructions"]
    assert "demonstrations or future images" in payload["instructions"]
    assert "no episode action or time cap" in payload["instructions"]
    assert "no local calibration or correction is performed" in payload["instructions"]
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
    assert payload["store"] is False and payload["stream"] is True
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert len(payload["input"]) == 1
    assert len([p for p in payload["input"][0]["content"] if p["type"] == "input_image"]) == 2
    record = json.loads(payload["input"][0]["content"][0]["text"])
    assert set(record) == {"type", "observation_index", "prompt", "observation/state"}
    assert record["prompt"] == resolve_distance_task(task).prompt
    assert instance.demonstration is None and not instance.demo_direct_control
    assert instance.historical_zero_shot_direct_control and not instance.zero_shot_collision_free
    assert not instance.distance_field_control and not instance.field_improvements
    assert instance.execution_mode == "fixed" and instance.feedback_history == "block_summary"
    assert instance.chunk_size == 5 and not instance.stop_on_model_complete
    assert instance.prompt_version == historical_zero_shot_direct_prompt_version(task)
    assert instance.field_representation == historical_zero_shot_direct_representation(task)
    assert instance.history_snapshot() == []
    assert instance.budget_snapshot()["api_key_calls"] == 0
    provider.assert_not_called()


def test_drawer_progression_and_bowl_placement_remain_distinct():
    drawer = historical_zero_shot_direct_instructions(task_key="drawer_bowl")
    bowl = historical_zero_shot_direct_instructions(task_key="bowl_plate")
    assert "does NOT call for a vertical test lift" in drawer
    assert "Closing the drawer is not required" in drawer
    assert "Choose any reobservation or recovery movement explicitly" in drawer
    assert "Place the bowl centered within the plate rim" not in drawer
    assert "Place the bowl centered within the plate rim" in bowl
    assert "drawer_handle" not in bowl


@pytest.mark.parametrize("task", TASKS)
def test_exact_direct_schema_contains_no_field_demo_completion_or_joint_inputs(task):
    schema = historical_zero_shot_direct_output_schema(task_key=task)
    assert set(schema["properties"]) == set(schema["required"]) == HISTORICAL_ZERO_SHOT_DIRECT_KEYS
    assert schema["additionalProperties"] is False
    actions = schema["properties"]["actions"]
    assert actions["minItems"] == actions["maxItems"] == 5
    assert actions["items"]["minItems"] == actions["items"]["maxItems"] == 7
    assert schema["properties"]["target_region"]["enum"] == list(resolve_distance_task(task).target_regions)
    assert schema["properties"]["control_context"]["properties"]["phase"]["enum"] == list(resolve_distance_task(task).control_phases)
    payload = client(task).build_payload(obs(task).to_dict())
    assert payload["text"]["format"] == {"type": "json_schema", "strict": True,
        "name": historical_zero_shot_direct_schema_name(task), "schema": schema}


@pytest.mark.parametrize("task,phase,target", [
    ("bowl_plate", "approach", "black_bowl"), ("bowl_plate", "grasp", "black_bowl"),
    ("bowl_plate", "release", "plate"), ("bowl_plate", "recover", "plate"),
    ("drawer_bowl", "handle_grasp", "drawer_handle"), ("drawer_bowl", "drawer_open", "drawer_handle"),
    ("drawer_bowl", "handle_release", "drawer_handle"), ("drawer_bowl", "grasp", "black_bowl"),
    ("drawer_bowl", "release", "drawer_interior"), ("drawer_bowl", "verify", "drawer_interior"),
])
def test_mock_responses_preserve_all_five_actions_and_raw_history(task, phase, target):
    raw = prediction(task, phase, target)
    original = copy.deepcopy(raw)
    reply = response(task, raw)
    instance = client(task, FakeTransport(reply))
    normalized = instance.predict(obs(task).to_dict())
    assert set(normalized) == HYBRID_KEYS and normalized["field"] is None
    assert normalized["controller"] == "contact_direct" and normalized["near_target"] is True
    assert normalized["direct"]["actions"] == original["actions"]
    assert instance.last_compact_prediction == original
    assert instance.last_control_context == original["control_context"]
    assert instance.last_effective_compact_field is None and instance.last_field_expansion_s == 0
    assert instance.last_goal_grounding is None and not instance.last_task_complete
    assert instance.history_snapshot()[-2:] == reply["output"]
    normalized["direct"]["actions"][0][0] = .99
    assert instance.last_compact_prediction == original


@pytest.mark.parametrize("task", TASKS)
def test_real_parent_policy_executes_unchanged_actions_without_local_planning(task):
    raw = prediction(task)
    transport = FakeTransport(response(task, raw))
    instance = client(task, transport)
    policy = HybridFieldDirectPolicy(instance)
    policy.planner.plan = Mock(side_effect=AssertionError("Direct must not plan"))
    before = obs(task)
    for i in range(5):
        decision = policy.act(before)
        assert list(decision.action) == raw["actions"][i]
        assert decision.diagnostics["controller"] == "contact_direct"
        assert "plan" not in decision.diagnostics
        after = obs(task, i+1)
        policy.observe_transition(before, decision, after)
        before = after
    assert len(transport.calls) == 1
    assert instance.history_stats()["feedback_records"] == 5
    policy.planner.plan.assert_not_called()


@pytest.mark.parametrize("task", TASKS)
def test_feedback_summaries_image_retention_and_failed_request_are_transactional(task):
    replies = [response(task), response(task), APIError("offline subscription failure"), response(task)]
    transport = FakeTransport(*replies)
    instance = client(task, transport)
    instance.predict(obs(task).to_dict())
    block = [feedback(i) for i in range(5)]
    for record in block[:4]:
        instance.record_transition(record)
    partial = instance.build_payload(obs(task, 1).to_dict())
    assert [r for r in text_records(partial["input"]) if r["type"] == "execution_feedback"] == block[:4]
    instance.record_transition(block[-1])
    instance.predict(obs(task, 1).to_dict())
    records = text_records(transport.calls[1]["input"])
    assert [r for r in records if r["type"] == "execution_block_summary"] == [summarize_feedback_block(block)]
    saved = instance.history_snapshot()
    with pytest.raises(APIError):
        instance.predict(obs(task, 2).to_dict())
    assert instance.history_snapshot() == saved
    instance.predict(obs(task, 2).to_dict())
    assert transport.calls[2] == transport.calls[3]
    assert len([p for m in transport.calls[-1]["input"] for p in m.get("content", []) if p.get("type") == "input_image"]) == 4
    assert not any("external_demonstration" in json.dumps(m) for m in instance.history_snapshot())
    instance.reset_history()
    assert instance.history_snapshot() == [] and instance.last_control_context is None


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("change", [
    lambda d: d.update(task_complete=True), lambda d: d.update(field={}),
    lambda d: d["actions"].pop(), lambda d: d["actions"].append([0]*7),
    lambda d: d["actions"][0].__setitem__(0, 1.1),
    lambda d: d["control_context"].update(object_pose=[0, 0, 0]),
    lambda d: d.update(target_region="hidden_object"), lambda d: d["control_context"].update(phase="completed"),
])
def test_invalid_output_never_commits_history(task, change):
    raw = prediction(task); change(raw)
    instance = client(task, FakeTransport(response(task, raw)))
    with pytest.raises(SchemaError):
        instance.predict(obs(task).to_dict())
    assert instance.history_snapshot() == [] and instance.last_control_context is None


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("key", ["object_pose", "contacts", "reward", "task_success", "robot_joint_positions"])
def test_historical_input_still_rejects_new_or_hidden_observation_fields(task, key):
    value = obs(task).to_dict(); value[key] = []
    with pytest.raises(ValueError):
        client(task).build_payload(value)


@pytest.mark.parametrize("changes", [
    {"demonstration": {}}, {"demonstration": "not allowed"}, {"zero_shot_collision_free": True},
    {"historical_zero_shot_direct_control": False}, {"controller_mode": "hybrid"},
    {"reasoning_effort": "high"}, {"field_improvements": True}, {"execution_mode": "tracking"},
    {"distance_field_control": True}, {"feedback_history": "full"}, {"image_history": 1},
    {"image_history": 2.0}, {"centered_placement_guidance": False}, {"grasp_settle_guidance": False},
    {"stop_on_model_complete": True}, {"task_prompt": "different task"},
])
def test_mismatched_conditions_are_rejected(changes):
    with pytest.raises(ValueError):
        client(**changes)


def test_non_astra_and_server_effort_or_model_changes_are_rejected():
    with pytest.raises(ValueError):
        client(config=replace(AstraConfig(), model="gpt-other"))
    for changes in ({"model": "gpt-other"}, {"reasoning": {"effort": "high"}}):
        instance = client(transport=FakeTransport(response("bowl_plate", **changes)))
        with pytest.raises(APIError):
            instance.predict(obs("bowl_plate").to_dict())
        assert instance.history_snapshot() == []


@pytest.mark.parametrize("task,demo,digest", [
    ("bowl_plate", False, "22983492324f16da877d257e196e2b0913ab2de88c357699e1b1d253f913e810"),
    ("drawer_bowl", False, "e4201a441077d0aefaf03b99daf0526716a9c3c34ef154a582f5e4a8a967d00c"),
    ("bowl_plate", True, "ca6e403a73e1ac67977fb0a895a84f4d7b151d903bb6de257a356bc563711ddb"),
    ("drawer_bowl", True, "729cca753569b417ea56861999867883af7ca03c4b4d3331c75941423c195a61"),
])
def test_existing_hybrid_prompts_remain_frozen(task, demo, digest):
    assert hashlib.sha256(distance_instructions(task_key=task, demonstration=demo).encode()).hexdigest() == digest


@pytest.mark.parametrize("task,digest", [
    ("bowl_plate", "2fe6ad0d0d65888d6ddc437ecad261ce56c34d477d25cdda1e15e5c3bd6b1265"),
    ("drawer_bowl", "9becc53aab90ef1cada460ba170eaa6550819c8b20bff9c00b241f0200f873e4"),
])
def test_existing_demo_direct_prompts_remain_frozen(task, digest):
    assert hashlib.sha256(demo_direct_instructions(task_key=task).encode()).hexdigest() == digest
