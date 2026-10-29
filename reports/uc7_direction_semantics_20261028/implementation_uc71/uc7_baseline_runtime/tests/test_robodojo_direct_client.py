"""Offline tests of native robot tools, feedback pairing and OAuth boundaries."""
import copy
import json

import numpy as np
import pytest

from gpt_field.astra import APIError, AstraConfig, SchemaError
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS, read_completed_response
from gpt_field.observation import image_to_data_url
from gpt_field.robodojo_direct_client import (
    MOVE_KEYS, POSE_KEYS, TOOL_NAMESPACE, RoboDojoDirectClient, instructions, validate_command,
)


TOKEN = "offline-robodojo-oauth-fixture"
IMAGE = image_to_data_url(np.zeros((4, 4, 3), np.uint8), size=4)


def pose(**updates):
    return {"x": 0.1, "y": 0.0, "z": 1.1, "pitch_deg": 0., "roll_deg": 0.,
            "yaw_deg": 0., "gripper": 1., **updates}


def observation(step=0, **updates):
    return {"external_image": IMAGE, "wrist_image": IMAGE, "robot_state": pose(),
            "step": step, "joint_positions": [0.] * 7, **updates}


def result(**updates):
    return {"status": "reached", "requested_target": pose(), "measured_state": pose(),
            "position_error_m": 0., "orientation_error_deg": 0., "executed_steps": 10,
            "detail": "Arm reached target.", **updates}


def response(name="move_eef", args=None, ident=1, **updates):
    return {"id": f"resp_{ident}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": "medium"},
            "usage": {"input_tokens": 100, "output_tokens": 25},
            "output": [
                {"type": "reasoning", "id": f"rs_{ident}", "summary": [],
                 "encrypted_content": f"opaque-{ident}"},
                {"type": "function_call", "id": f"fc_{ident}", "call_id": f"call_{ident}",
                 "status": "completed", "name": name, "namespace": TOOL_NAMESPACE,
                 "arguments": json.dumps(args if args is not None else {"x": .2})}], **updates}


class Transport:
    def __init__(self, replies=None):
        self.requests = []
        self.replies = list(replies or [])

    def __call__(self, endpoint, payload, headers, timeout):
        self.requests.append(copy.deepcopy(payload))
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        if self.replies:
            reply = self.replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return reply
        return response(ident=len(self.requests))


def client(transport=None, **kwargs):
    return RoboDojoDirectClient(token_provider=lambda: TOKEN,
                               transport=transport or Transport(), **kwargs)


def test_preview_is_native_strict_tools_medium_and_does_not_authenticate():
    def no_auth():
        raise AssertionError("Preview must not authenticate")
    c = RoboDojoDirectClient(token_provider=no_auth)
    payload = c.build_payload(observation())
    assert payload["model"] == "gpt-6-astra"
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["store"] is False and payload["stream"] is True
    assert payload["tool_choice"] == "required" and payload["parallel_tool_calls"] is False
    assert not UNSUPPORTED_FIELDS.intersection(payload)
    assert "text" not in payload
    assert len(payload["tools"]) == 1
    namespace = payload["tools"][0]
    assert namespace["type"] == "namespace" and namespace["name"] == TOOL_NAMESPACE
    move, give_up = namespace["tools"]
    assert move["name"] == "move_eef" and give_up["name"] == "give_up"
    assert move["strict"] and set(move["parameters"]["required"]) == set(MOVE_KEYS)
    assert move["parameters"]["additionalProperties"] is False
    assert c.history_snapshot() == [] and c.budget_snapshot()["api_calls"] == 0
    obs = json.loads(payload["input"][0]["content"][0]["text"])
    assert set(obs["robot_state"]) == set(POSE_KEYS)
    assert obs["joint_positions"] == [0.] * 7


def test_prompt_has_correct_nonstandard_angles_and_arrival_semantics():
    text = instructions("drawer_bowl")
    for expected in ("pitch_deg rotates about fixed world +x", "roll_deg about +y",
                     "Rz(yaw) Ry(roll) Rx(pitch) Rx(pi)", "20 Hz", "not cuRobo",
                     "new opening only after arm arrival", "unsuccessful", "Task-stage recipe"):
        assert expected in text
    assert "drawer" in text and "rim_pinch" not in text and "frontal_bar_pinch" not in text


def test_native_tool_call_and_matched_feedback_preserve_history_but_trim_images(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "forbidden-api-key-fixture")
    transport = Transport()
    c = client(transport, log_dir=tmp_path)
    for step in range(3):
        assert c.predict(observation(step)) == {"name": "move_eef", "arguments": {"x": .2}}
        with pytest.raises(ValueError, match="feedback"):
            c.build_payload(observation(step + 1))
        c.record_result(result())
    history = c.history_snapshot()
    assert sum(item.get("type") == "function_call" for item in history) == 3
    assert all(item["namespace"] == TOOL_NAMESPACE for item in history
               if item.get("type") == "function_call")
    assert sum(item.get("type") == "function_call_output" for item in history) == 3
    outputs = [item for item in history if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in outputs] == ["call_1", "call_2", "call_3"]
    assert c.history_stats()["retained_image_pairs"] == 2
    assert c.history_stats()["observations"] == 3
    assert c.history_stats()["feedback_records"] == 3
    assert [item["encrypted_content"] for item in history if item.get("type") == "reasoning"] == [
        "opaque-1", "opaque-2", "opaque-3"]
    assert len([item for item in history if item.get("role") == "user"]) == 3
    assert "omitted" in json.dumps(history[0])
    assert c.budget_snapshot()["subscription_inference_calls"] == 3
    assert c.budget_snapshot()["api_key_calls"] == 0
    assert c.budget_snapshot()["input_tokens"] == 300
    for path in tmp_path.glob("*.json"):
        assert TOKEN not in path.read_text() and "forbidden-api-key-fixture" not in path.read_text()


@pytest.mark.parametrize("bad", [
    {"name": "move_eef", "arguments": {}},
    {"name": "move_eef", "arguments": {"note": "Nothing"}},
    {"name": "move_eef", "arguments": {"x": None}},
    {"name": "move_eef", "arguments": {"x": True}},
    {"name": "move_eef", "arguments": {"x": float("nan")}},
    {"name": "move_eef", "arguments": {"x": 10 ** 500}},
    {"name": "move_eef", "arguments": {"gripper": 2}},
    {"name": "move_eef", "arguments": {"x": 0, "object_pose": [1, 2, 3]}},
    {"name": "give_up", "arguments": {"reason": ""}},
    {"name": "declare_success", "arguments": {}},
])
def test_invalid_commands_are_not_executable(bad):
    with pytest.raises(SchemaError):
        validate_command(bad)


def test_nullable_partial_move_keeps_zero_values_and_gripper_only_commands():
    assert validate_command({"name": "move_eef", "arguments": {
        **{key: None for key in MOVE_KEYS}, "x": 0, "gripper": 0, "note": "close"}}) == {
            "name": "move_eef", "arguments": {"x": 0., "gripper": 0., "note": "close"}}
    assert validate_command({"name": "move_eef", "arguments": {"gripper": 1}})["arguments"] == {
        "gripper": 1.}


@pytest.mark.parametrize("changes", [
    {"object_positions": []}, {"reward": 1}, {"success": True},
    {"step": -1}, {"step": True}, {"joint_positions": [0] * 6},
    {"robot_state": {**pose(), "object": [1, 2, 3]}},
    {"external_image": "https://example.org/image.png"},
])
def test_observation_boundary_rejects_extra_or_invalid_data_before_inference(changes):
    transport = Transport()
    c = client(transport)
    with pytest.raises(ValueError):
        c.predict(observation(**changes))
    assert not transport.requests and c.budget_snapshot()["api_calls"] == 0


def test_feedback_validation_is_atomic_and_forbids_evaluator_data():
    c = client()
    c.predict(observation())
    previous = c.history_snapshot()
    for bad in (result(success=True), result(status="success"), result(executed_steps=-1),
                result(gripper_applied="yes"), result(position_error_m=float("nan"))):
        with pytest.raises(ValueError):
            c.record_result(bad)
        assert c.history_snapshot() == previous
    c.record_result(result(status="incomplete", gripper_applied=False, gripper_error=.2))
    with pytest.raises(ValueError):
        c.record_result(result())


@pytest.mark.parametrize("reply", [
    response(status="incomplete"),
    response(model="gpt-6-sol"),
    response(reasoning={"effort": "high"}),
    response(output=response()["output"] * 2),
    response(name="shell", args={"cmd": "noop"}),
    response(args={"x": float("inf")}),
    APIError("fixture transport error"),
])
def test_failed_responses_do_not_commit_history_or_retry(reply):
    transport = Transport([reply])
    c = client(transport)
    with pytest.raises((APIError, SchemaError)):
        c.predict(observation())
    assert len(transport.requests) == 1 and c.history_snapshot() == []


def test_duplicate_json_fields_are_rejected():
    reply = response()
    reply["output"][-1]["arguments"] = '{"x": 0.1, "x": 0.2}'
    with pytest.raises(SchemaError):
        client(Transport([reply])).predict(observation())


@pytest.mark.parametrize("namespace", ["other", "", None, [], False])
def test_foreign_or_invalid_namespaces_never_commit_or_execute(namespace):
    reply = response()
    reply["output"][-1]["namespace"] = namespace
    transport = Transport([reply])
    c = client(transport)
    with pytest.raises(SchemaError, match="namespace"):
        c.predict(observation())
    assert len(transport.requests) == 1 and c.history_snapshot() == []


def test_namespace_is_optional_but_qualified_name_is_not_reinterpreted():
    reply = response()
    del reply["output"][-1]["namespace"]
    assert client(Transport([reply])).predict(observation())["name"] == "move_eef"
    reply = response(name="robot.move_eef")
    with pytest.raises(SchemaError):
        client(Transport([reply])).predict(observation())


def test_give_up_is_a_terminal_unsuccessful_command_not_success():
    transport = Transport([response("give_up", {"reason": "Cannot continue."})])
    c = client(transport)
    assert c.predict(observation())["name"] == "give_up"
    with pytest.raises(ValueError, match="ended"):
        c.predict(observation())
    assert len(transport.requests) == 1
    c.reset_history()
    assert c.history_snapshot() == []


def test_shared_sse_parser_preserves_native_function_call_items():
    reply = response()
    event = {"type": "response.completed", "response": reply}
    parsed = read_completed_response([f"data: {json.dumps(event)}\n", "\n"])
    assert parsed["output"] == reply["output"]
    assert RoboDojoDirectClient._extract(parsed) == {"name": "move_eef", "arguments": {"x": .2}}


@pytest.mark.parametrize("kwargs", [
    {"config": AstraConfig(model="gpt-6-sol")}, {"reasoning_effort": "high"},
    {"image_history": 1}, {"config": AstraConfig(max_api_calls=3)},
    {"config": AstraConfig(max_wall_time_s=60)},
    {"config": AstraConfig(max_cost_usd=1)},
])
def test_configuration_cannot_change_model_billing_effort_or_add_episode_caps(kwargs):
    with pytest.raises(ValueError):
        client(**kwargs)
