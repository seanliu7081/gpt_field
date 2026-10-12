"""Versioned compressed feedback and subscription Direct completion contracts."""

import copy
import json

import pytest

from gpt_field.astra import APIError, SchemaError, _canonical
from gpt_field.block_feedback_history import (
    BLOCK_FEEDBACK_PROTOCOL, compact_feedback_history, summarize_feedback_block,
)
from gpt_field.compact_subscription import compact_instructions, compact_prompt_version
from gpt_field.hybrid_policy import HybridFieldDirectPolicy, ModelTaskComplete
from gpt_field.hybrid_subscription import HYBRID_GRID
from test_compact_subscription import client, direct, hybrid
from test_field_history import FakeTransport, envelope, observation


def feedback(step, chunk=0, chunk_step=None):
    before = [.1 + step * .001, .2, 1., 0., 0., 0., .04, -.04]
    after = copy.deepcopy(before)
    after[0] += .001
    return {
        "type": "execution_feedback", "step": step, "chunk_index": chunk,
        "chunk_step": step if chunk_step is None else chunk_step,
        "before_state": before, "after_state": after,
        "normalized_action": [.2, 0., 0., 0., 0., 0., -1.],
        "desired_translation_delta_m": [.01, 0., 0.],
        "desired_world_rotation_delta_rad": [0., 0., 0.],
        "measured_translation_delta_m": [.001, 0., 0.],
        "measured_translation_norm_m": .001, "commanded_translation_norm_m": .01,
        "translation_tracking_error_m": .009, "gripper_joint_delta": [0., 0.],
        "gripper_command": -1.,
        "rolling_progress": {"window_size": 5, "translation_command_steps": 5,
                             "low_motion_translation_steps": 0, "commanded_translation_path_m": .05,
                             "measured_translation_path_m": .005, "measured_net_translation_m": .005,
                             "gripper_close_steps": 0},
    }


def message(record):
    return {"role": "user", "content": [{"type": "input_text", "text": _canonical(record)}]}


def text_records(messages):
    result = []
    for item in messages:
        if item.get("role") == "user" and len(item.get("content", [])) == 1:
            try:
                result.append(json.loads(item["content"][0]["text"]))
            except (KeyError, ValueError):
                pass
    return result


def test_summary_captures_measured_motion_and_ordered_gripper_changes_without_hidden_state():
    records = [feedback(i) for i in range(5)]
    for i in (2, 3):
        records[i]["gripper_command"] = 1.
    result = summarize_feedback_block(records)
    assert result["protocol"] == BLOCK_FEEDBACK_PROTOCOL
    assert result["executed_actions"] == 5
    assert (result["first_step"], result["last_step"]) == (0, 4)
    assert result["before_state"] == records[0]["before_state"]
    assert result["after_state"] == records[-1]["after_state"]
    assert result["commanded_translation_path_m"] == pytest.approx(.05)
    assert result["measured_translation_path_m"] == pytest.approx(.005)
    assert result["measured_net_translation_m"] == pytest.approx(.005)
    assert result["commanded_translation_sum_m"] == pytest.approx([.05, 0., 0.])
    assert result["translation_tracking_error_max_m"] == pytest.approx(.009)
    assert result["gripper_close_steps"] == 2
    assert result["gripper_command_runs"] == [
        {"command": -1., "actions": 2, "first_step": 0, "last_step": 1},
        {"command": 1., "actions": 2, "first_step": 2, "last_step": 3},
        {"command": -1., "actions": 1, "first_step": 4, "last_step": 4},
    ]
    assert not {"success", "contact", "object_pose"}.intersection(result)
    result["before_state"][0] = 999
    assert records[0]["before_state"][0] != 999


@pytest.mark.parametrize("mode,count", [("field", 30), ("direct", 5)])
def test_only_complete_blocks_are_replaced_and_nonfeedback_items_are_preserved(mode, count):
    raw = hybrid(mode == "direct")
    output = envelope(raw)["output"]
    records = [feedback(i) for i in range(count)]
    source = output + [message(record) for record in records]
    original = copy.deepcopy(source)
    compacted = compact_feedback_history(source)
    assert source == original
    assert compacted[:2] == output
    assert len(compacted) == 3
    assert text_records(compacted)[0] == summarize_feedback_block(records)
    assert compact_feedback_history(compacted) == compacted
    assert compact_feedback_history(source[:-1]) == source[:-1]
    assert compact_feedback_history(source[2:]) == source[2:]


def test_five_action_partial_field_is_not_treated_as_a_direct_block():
    source = envelope(hybrid())["output"] + [message(feedback(i)) for i in range(5)]
    assert compact_feedback_history(source) == source


@pytest.mark.parametrize("defect", ["duplicate_step", "wrong_chunk_step", "mixed_chunk", "missing_measurement"])
def test_noncontiguous_feedback_is_preserved(defect):
    records = [feedback(i) for i in range(5)]
    if defect == "missing_measurement":
        del records[2]["translation_tracking_error_m"]
    else:
        records[2][{"duplicate_step": "step", "wrong_chunk_step": "chunk_step",
                    "mixed_chunk": "chunk_index"}[defect]] = 0 if defect != "mixed_chunk" else 1
    source = envelope(direct())["output"] + [message(record) for record in records]
    assert compact_feedback_history(source) == source


def test_compact_client_commits_summaries_only_after_valid_response_and_keeps_opaque_output(tmp_path):
    responses = [envelope(direct(), id=f"response-{i}") for i in range(3)]
    fake = FakeTransport(responses[0], APIError("offline failure"), responses[1], responses[2])
    instance = client(fake, controller_mode="direct", feedback_history="block_summary", log_dir=tmp_path)
    instance.predict(observation().to_dict())
    records = [feedback(i) for i in range(5)]
    for record in records:
        instance.record_transition(record)
    uncommitted = instance.history_snapshot()
    preview = instance.build_payload(observation(1).to_dict())
    assert instance.history_snapshot() == uncommitted
    assert [x for x in text_records(preview["input"]) if x["type"] == "execution_block_summary"] == [
        summarize_feedback_block(records)]
    with pytest.raises(APIError):
        instance.predict(observation(1).to_dict())
    assert instance.history_snapshot() == uncommitted
    instance.predict(observation(1).to_dict())
    assert responses[0]["output"][0] in instance.history_snapshot()
    assert responses[0]["output"][1] in instance.history_snapshot()
    assert not any(x["type"] == "execution_feedback" for x in text_records(instance.history_snapshot()))
    for i in range(5, 10):
        instance.record_transition(feedback(i, chunk=1, chunk_step=i - 5))
    instance.predict(observation(2).to_dict())
    stats = instance.history_stats()
    assert stats["protocol"] == BLOCK_FEEDBACK_PROTOCOL
    assert stats["feedback_records"] == 10 and stats["feedback_block_summaries"] == 2
    images = [part for item in fake.calls[-1]["input"] for part in item.get("content", [])
              if part.get("type") == "input_image"]
    assert len(images) == 4
    assert instance.budget_snapshot()["api_key_calls"] == 0
    assert len(fake.calls) == 4  # Only explicit predictions; summaries add no inference.
    request = json.loads(sorted(tmp_path.glob("*.request.json"))[-1].read_text())
    assert request["history_protocol"] == BLOCK_FEEDBACK_PROTOCOL
    assert request["history_stats_before_request"]["protocol"] == BLOCK_FEEDBACK_PROTOCOL
    instance.reset_history()
    assert instance.history_stats()["feedback_block_summaries"] == 0


def test_new_options_are_versioned_and_the_old_prompt_is_unchanged():
    old = client()
    new = client(execution_mode="tracking", feedback_history="block_summary", reasoning_effort="medium",
                 stop_on_model_complete=True)
    assert old.execution_mode == "fixed" and old.feedback_history == "full"
    payload = new.build_payload(observation().to_dict())
    assert new.prompt_version == "astra-compact-field-direct-history-hybrid-tracking-block_summary-completion-v3"
    assert payload["reasoning"] == {"effort": "medium"}
    assert "3 cm" in payload["instructions"] and "5 cm" in payload["instructions"]
    assert "each actual measured pose" in payload["instructions"]
    assert "execution_block_summary" in payload["instructions"]
    assert "ten geometric segments are not ten fixed" in payload["instructions"]
    assert "neither regenerated nor corrected" not in payload["instructions"]
    assert "neither regenerated nor corrected" in compact_instructions()
    assert "0, 10, 20" in payload["instructions"]
    assert new.request_key(payload, HYBRID_GRID, "hybrid") != old.request_key(payload, HYBRID_GRID, "hybrid")
    assert new.history_snapshot() == []


@pytest.mark.parametrize("kwargs", [{"execution_mode": "other"}, {"feedback_history": "other"}])
def test_invalid_optimized_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        client(**kwargs)


def direct_completion(complete=True):
    return {"actions": [[0., 0., 0., 0., 0., 0., -1.] for _ in range(5)] if complete else direct()["actions"],
            "task_complete": complete,
            "completion_reason": "The released bowl is resting on the plate and the gripper is clear." if complete else ""}


def test_forced_direct_completion_has_matching_visual_rule_and_normalizes_for_policy():
    raw = direct_completion()
    fake = FakeTransport(envelope(raw, reasoning={"effort": "medium"}))
    instance = client(fake, controller_mode="direct", stop_on_model_complete=True,
                      reasoning_effort="medium", feedback_history="block_summary")
    normalized = instance.predict(observation().to_dict())
    assert normalized["controller"] == "contact_direct"
    assert normalized["target_region"] == "plate"
    assert normalized["direct"] == {"actions": raw["actions"]}
    assert instance.last_task_complete is True
    assert instance.last_compact_prediction == raw
    assert instance.routing_is_model_prediction is False
    schema = fake.calls[0]["text"]["format"]["schema"]
    assert set(schema["required"]) == {"actions", "task_complete", "completion_reason"}
    assert schema["additionalProperties"] is False
    prompt = fake.calls[0]["instructions"]
    assert "CURRENT visual evidence" in prompt and "before any actions from that response" in prompt
    assert "Output an object containing only actions" not in prompt
    assert "repulsors" not in prompt and "routing wrapper" in prompt
    assert compact_prompt_version("direct", stop_on_model_complete=True).endswith("direct-completion-v2")


@pytest.mark.parametrize("defect", ["moving", "close", "missing", "blank_reason", "bool"])
def test_forced_direct_terminal_invalid_evidence_cannot_set_completion(defect):
    raw = direct_completion()
    if defect == "moving": raw["actions"][0][0] = .1
    if defect == "close": raw["actions"][0][6] = 1.
    if defect == "missing": del raw["task_complete"]
    if defect == "blank_reason": raw["completion_reason"] = " "
    if defect == "bool": raw["task_complete"] = 1
    instance = client(FakeTransport(envelope(raw)), controller_mode="direct", stop_on_model_complete=True)
    with pytest.raises(SchemaError):
        instance.predict(observation().to_dict())
    assert instance.last_task_complete is False and instance.history_snapshot() == []


def test_nonterminal_direct_keeps_five_original_actions():
    raw = direct_completion(False)
    instance = client(FakeTransport(envelope(raw)), controller_mode="direct", stop_on_model_complete=True)
    result = instance.predict(observation().to_dict())
    assert result["direct"] == direct()
    assert not instance.last_task_complete


def test_forced_direct_completion_stops_policy_before_any_action_and_repeated_calls():
    fake = FakeTransport(envelope(direct_completion(), reasoning={"effort": "medium"}))
    instance = client(fake, controller_mode="direct", stop_on_model_complete=True,
                      reasoning_effort="medium", feedback_history="block_summary")
    policy = HybridFieldDirectPolicy(instance)
    with pytest.raises(ModelTaskComplete) as stopped:
        policy.act(observation())
    assert stopped.value.details["controller"] == "contact_direct"
    assert stopped.value.details["routing_is_model_prediction"] is False
    assert policy._pending is None and policy.step_index == 0
    with pytest.raises(ModelTaskComplete):
        policy.act(observation())
    assert len(fake.calls) == 1
    assert instance.history_stats()["feedback_records"] == 0
