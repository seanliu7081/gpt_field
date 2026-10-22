"""Bounded wire history retains current evidence and full local provenance."""

import copy
import json

import pytest

from gpt_field.astra import APIError, SchemaError
from gpt_field.block_feedback_history import _text_object, summarize_feedback_block
from gpt_field.direct_history import _trim_images
from gpt_field.speed_first_history import (
    BOUNDED_HISTORY_INSTRUCTIONS, SpeedFirstHistoryMixin, bounded_history_input,
)
from gpt_field.speed_first_hybrid import ChatGPTPlanSpeedFirstClient, SpeedFirstHybridPolicy
from test_block_feedback_history import feedback, message
from test_distance_policy import intent, tick
from test_field_history import FakeTransport, envelope, observation


class BoundedClient(SpeedFirstHistoryMixin, ChatGPTPlanSpeedFirstClient):
    pass


def observed(index):
    value = message({"type": "observation", "observation_index": index,
                     "observation/state": [.1, .2, 1., 0., 0., 0., .04, -.04]})
    value["content"].extend([
        {"type": "input_image", "image_url": f"external-{index}"},
        {"type": "input_image", "image_url": f"wrist-{index}"},
    ])
    return value


def transcript(count=30):
    records = []
    for index in range(count):
        raw = intent(direct=True, phase="recover" if index == 2 else "grasp")
        raw["routing_reason"] = ("Failed to lift; reopen and move to the other rim." if index == 2
                                 else f"Observed intent {index}")
        records.extend([observed(index), *envelope(raw, id=f"turn-{index}")["output"],
                        message(summarize_feedback_block([feedback(index * 5 + j, chunk=index, chunk_step=j)
                                                           for j in range(5)])),
                        message({"type": "distance_control_event", "step": index * 5 + 5,
                                 "chunk_index": index, "reason": "field_budget_exhausted"})])
    return _trim_images(records + [observed(count)], 2)


def input_objects(items):
    return [obj for item in items if (obj := _text_object(item)) is not None]


def test_current_observation_and_two_image_pairs_remain_exact_and_source_unchanged():
    source = transcript()
    original = copy.deepcopy(source)
    result, stats = bounded_history_input(source)
    assert source == original
    assert result[-1] == source[-1]
    images = lambda values: [part for item in values for part in item.get("content", [])
                             if part.get("type") == "input_image"]
    assert images(result) == images(source)
    assert len(images(result)) == 4
    assert stats["retained_recent_observations"] == 4
    assert stats["older_observations"] == 27
    assert result[-4:] == source[-4:]
    assert result[0]["role"] == "user"
    digest = _text_object(result[0])
    assert digest["historical_not_current"] is True
    assert digest["older_observations"] == 27
    assert digest["omitted_older_observations"] == 15


def test_recent_recovery_reason_and_changed_strategy_survive_old_history_eviction():
    result, _ = bounded_history_input(transcript())
    rows = _text_object(result[0])["records"]
    recovery = next(row for row in rows if row["observation_index"] == 2)
    assert recovery["model_phase"] == "recover"
    assert recovery["model_routing_reason"] == "Failed to lift; reopen and move to the other rim."
    assert recovery["measured_block"]["first_step"] == 10
    assert recovery["measured_block"]["last_step"] == 14
    assert recovery["intent_switch"] is True
    assert next(row for row in rows if row["observation_index"] == 3)["intent_switch"] is True
    assert "grasp_success" not in json.dumps(rows)


def test_digest_is_bounded_deterministic_allowlisted_and_never_decodes_opaque_reasoning():
    source = transcript(150)
    # Only past visible model statements and measured robot fields enter digest.
    old_summary = json.loads(source[3]["content"][0]["text"])
    old_summary["oracle_object_pose"] = "secret-hidden-fixture"
    source[3] = message(old_summary)
    for item in source:
        if item.get("type") == "reasoning":
            item["encrypted_content"] = "opaque-uninterpretable-" * 2000
    left, stats = bounded_history_input(source)
    right, _ = bounded_history_input(copy.deepcopy(source))
    assert left == right
    digest = _text_object(left[0])
    assert len(digest["records"]) <= 12
    assert stats["digest_chars"] <= 16000
    assert stats["opaque_reasoning_chars"] <= 16000
    assert stats["recent_reasoning_items_omitted"] == 3
    assert "secret-hidden-fixture" not in json.dumps(left)
    assert len(left) < 20


def test_small_histories_and_recent_model_actions_are_not_rewritten():
    source = transcript(2)
    result, stats = bounded_history_input(source)
    assert result == source
    assert stats["older_observations"] == 0
    assert stats["digest_chars"] == 0
    assert "historical commands" in BOUNDED_HISTORY_INSTRUCTIONS


def operation_transcript(count=20):
    raw = intent(direct=True, phase="grasp")
    raw["direct"] = None
    raw["gripper_operation"] = {"command": "close"}
    records = [feedback(i, chunk=0, chunk_step=i) for i in range(count)]
    for record in records:
        record["gripper_command"] = 1.
    return [observed(0), *envelope(raw)["output"], *[message(record) for record in records],
            message({"type": "distance_control_event", "step": count,
                     "chunk_index": 0, "reason": "field_budget_exhausted"}),
            message({"type": "local_gripper_result", "step": count, "chunk_index": 0,
                     "command": "close", "stop_reason": "settled", "done": True,
                     "actions_executed": count, "fingers_stable": True,
                     "finger_motion_observed": True, "pose_error_m": .0002,
                     "grasp_success_certified": False, "task_success_certified": False}), observed(1)], records


def test_twenty_action_operation_compacts_only_with_complete_observable_result():
    source, records = operation_transcript()
    original = copy.deepcopy(source)
    result, _ = bounded_history_input(source)
    summaries = [obj for obj in input_objects(result) if obj.get("type") == "execution_block_summary"]
    assert summaries == [summarize_feedback_block(records)]
    assert result[-2] == source[-2]
    assert result[-1] == source[-1]
    assert source == original


@pytest.mark.parametrize("defect", ["no_result", "wrong_step", "wrong_chunk", "missing_feedback", "duplicate_step",
                                    "result_wrong_step", "result_wrong_chunk", "result_wrong_count"])
def test_incomplete_operation_feedback_is_preserved_without_inventing_measurements(defect):
    source, _ = operation_transcript()
    if defect == "no_result":
        source.pop(-2)
    elif defect == "missing_feedback":
        source.pop(6)
    elif defect == "duplicate_step":
        source[5] = copy.deepcopy(source[4])
    elif defect.startswith("result_"):
        record = _text_object(source[-2])
        record[{"result_wrong_step": "step", "result_wrong_chunk": "chunk_index",
                "result_wrong_count": "actions_executed"}[defect]] += 1
        source[-2] = message(record)
    else:
        event = _text_object(source[-3])
        event["step" if defect == "wrong_step" else "chunk_index"] += 1
        source[-3] = message(event)
    result, _ = bounded_history_input(source)
    assert result == source


def test_gripper_command_and_result_enter_older_digest_without_grasp_claim():
    old, _ = operation_transcript()
    tail = transcript(6)
    source = _trim_images(old[:-1] + tail, 2)
    result, _ = bounded_history_input(source)
    digest = _text_object(result[0])
    operation = next(row for row in digest["records"] if "gripper_operation" in row)
    assert operation["gripper_operation"] == {"command": "close"}
    assert operation["local_gripper_result"]["actions_executed"] == 20
    assert operation["local_gripper_result"]["grasp_success_certified"] is False
    assert operation["local_gripper_result"]["task_success_certified"] is False
    assert operation["measured_block"]["executed_actions"] == 20


def test_latest_gripper_result_retained_even_when_old_and_many_newer_intent_switches_exist():
    old, _ = operation_transcript()
    tail = transcript(60)
    for item in tail:
        value = _text_object(item)
        if item.get("role") == "assistant" and value is not None:
            value["control_context"]["phase"] = "recover" if len(value["routing_reason"]) % 2 else "grasp"
            item["content"][0]["text"] = json.dumps(value)
    result, stats = bounded_history_input(_trim_images(old[:-1] + tail, 2), max_digest_chars=2000)
    rows = _text_object(result[0])["records"]
    assert any(row.get("local_gripper_result", {}).get("stop_reason") == "settled" for row in rows)
    assert stats["digest_chars"] <= 2000


def test_successful_commits_keep_complete_local_history_and_feedback_without_duplicates(tmp_path):
    accepted = envelope(intent(direct=True, phase="grasp"), model="gpt-6-astra", reasoning={"effort": "medium"})
    transport = FakeTransport(*[accepted] * 12)
    client = BoundedClient(token_provider=lambda: "offline", transport=transport, log_dir=tmp_path)
    policy = SpeedFirstHybridPolicy(client)
    obs = observation()
    for _ in range(60):
        obs, _ = tick(policy, obs)
    snapshot = client.history_snapshot()
    assert len([item for item in snapshot if item.get("role") == "assistant"]) == 12
    assert len([item for item in snapshot if item.get("type") == "reasoning"]) == 12
    assert all(obj.get("type") != "older_history_digest" for obj in input_objects(snapshot))
    local_indexes = [json.loads(item["content"][0]["text"])["observation_index"] for item in snapshot
                     if item.get("role") == "user" and item.get("content")
                     and json.loads(item["content"][0]["text"]).get("type") == "observation"]
    assert local_indexes == list(range(12))
    assert client.history_stats()["feedback_records"] == 60
    assert client.history_stats()["observations"] == 12
    assert len(list(tmp_path.glob("*.request.json"))) == 12
    assert len(list(tmp_path.glob("*.response.json"))) == 12
    assert _text_object(transport.calls[-1]["input"][0])["type"] == "older_history_digest"
    assert len([item for item in transport.calls[-1]["input"] if item.get("role") == "assistant"]) == 3
    client.reset_history()
    assert client.history_snapshot() == []
    assert client.last_history_presentation is None


def test_failed_request_does_not_commit_digest_or_current_observation():
    transport = FakeTransport(APIError("offline failure"))
    client = BoundedClient(token_provider=lambda: "offline", transport=transport)
    client._history = transcript(20)[:-1]
    client._observation_count = 20
    original = client.history_snapshot()
    with pytest.raises(APIError):
        client.predict(observation(20).to_dict())
    assert client.history_snapshot() == original
    assert client.history_stats()["observations"] == 20


def test_overload_retry_payload_identical_and_complete_history_committed_once(monkeypatch):
    import gpt_field.speed_first_hybrid as module
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module.random, "uniform", lambda *args: 0.)
    error = APIError("offline overload")
    error.error_code = "server_is_overloaded"
    accepted = envelope(intent(direct=True, phase="recover"), model="gpt-6-astra", reasoning={"effort": "medium"})
    transport = FakeTransport(error, accepted)
    client = BoundedClient(token_provider=lambda: "offline", transport=transport)
    client._history = transcript(20)[:-1]
    client._observation_count = 20
    client.predict(observation(20).to_dict())
    assert transport.calls[0] == transport.calls[1]
    assert client.history_stats()["observations"] == 21
    assert len([item for item in client.history_snapshot() if item.get("role") == "assistant"]) == 21


@pytest.mark.parametrize("kwargs", [{"recent_observations": 1}, {"digest_entries": 0},
                                    {"max_digest_chars": 1}, {"max_reasoning_chars": -1}])
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        bounded_history_input(transcript(), **kwargs)
