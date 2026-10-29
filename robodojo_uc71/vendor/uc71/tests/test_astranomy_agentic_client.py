"""Offline protocol tests; no inference, credentials, or simulator actions."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.astra import APIError, AstraConfig, SchemaError
from gpt_field.astranomy_agentic_client import (
    AstranomyAgenticClient, TOOL_NAME, TOOL_NAMESPACE, validate_command, validate_feedback,
)
from gpt_field.astranomy_agentic_geometry import extract_geometry_metadata
from gpt_field.astranomy_agentic_motion import (
    ShortTargetExecutor, TargetLimitError, measured_pose, rejected_feedback,
)
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS, read_completed_response
from gpt_field.observation import image_to_data_url


TOKEN = "offline-agentic-subscription-fixture"
IMAGE = image_to_data_url(np.zeros((4, 4, 3), np.uint8), size=4, rotate=False)
STATE8 = [.1, 0., 1.1, 0., 0., 0., .04, -.04]


def geometry():
    model = SimpleNamespace(camera_name2id=lambda name: 0 if name == "agentview" else 1,
                            cam_fovy=[45., 45.])
    data = SimpleNamespace(cam_xpos=np.array([[0., 0., 2.], [.1, 0., 1.2]]),
                           cam_xmat=np.tile(np.eye(3).reshape(1, 9), (2, 1)))
    return extract_geometry_metadata(SimpleNamespace(sim=SimpleNamespace(model=model, data=data)), 4)


def pose(grip=0.):
    return [.1, 0., 1.1, 1., 0., 0., 0., grip]


def observation(step=0, **updates):
    return {"external_image": IMAGE, "wrist_image": IMAGE, "robot_state": pose(),
            "joint_positions": [0.] * 7, "geometry": geometry(), "step": step, **updates}


def feedback(**updates):
    return {"status": "segment_complete", "requested_target": pose(), "measured_pose": pose(),
            "position_error_m": 0., "orientation_error_rad": 0., "requested_steps": 3,
            "executed_steps": 3, "simulation_time_s": .15, "control_hz": 20,
            "gripper_command": 0, "detail": "Selected steps executed; arrival is not guaranteed.", **updates}


def response(name=TOOL_NAME, args=None, ident=1, **updates):
    return {"id": f"resp_{ident}", "status": "completed", "model": "gpt-6-astra",
            "reasoning": {"effort": "medium"}, "usage": {"input_tokens": 100, "output_tokens": 25},
            "output": [
                {"type": "reasoning", "id": f"rs_{ident}", "summary": [],
                 "encrypted_content": f"opaque-{ident}"},
                {"type": "message", "role": "assistant", "id": f"msg_{ident}",
                 "content": [{"type": "output_text", "text": "Approach the object, then inspect."}]},
                {"type": "function_call", "id": f"fc_{ident}", "call_id": f"call_{ident}",
                 "status": "completed", "name": name, "namespace": TOOL_NAMESPACE,
                 "arguments": json.dumps(args if args is not None else {"target": pose(), "steps": 3})}],
            **updates}


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
    return AstranomyAgenticClient(token_provider=lambda: TOKEN,
                                  transport=transport or Transport(), **kwargs)


def test_preview_uses_only_one_native_namespace_tool_and_never_authenticates():
    def no_auth():
        raise AssertionError("Payload preview must not authenticate")
    c = AstranomyAgenticClient(token_provider=no_auth)
    payload = c.build_payload(observation())
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
    assert payload["store"] is False and payload["stream"] is True
    assert payload["tool_choice"] == "required" and payload["parallel_tool_calls"] is False
    assert not UNSUPPORTED_FIELDS.intersection(payload)
    assert "text" not in payload
    assert len(payload["tools"]) == 1
    namespace = payload["tools"][0]
    assert namespace["type"] == "namespace" and namespace["name"] == TOOL_NAMESPACE
    assert len(namespace["tools"]) == 1
    tool = namespace["tools"][0]
    assert tool["name"] == TOOL_NAME and tool["strict"]
    assert tool["parameters"]["additionalProperties"] is False
    assert set(tool["parameters"]["required"]) == {"target", "steps"}
    assert tool["parameters"]["properties"]["target"]["minItems"] == 8
    assert tool["parameters"]["properties"]["steps"]["maximum"] == 5
    obs = json.loads(payload["input"][0]["content"][0]["text"])
    assert obs["robot_state"] == pose() and obs["joint_positions"] == [0.] * 7
    assert obs["geometry"] == geometry()
    assert c.history_snapshot() == [] and c.budget_snapshot()["subscription_inference_calls"] == 0


def test_all_visual_and_text_history_native_plans_and_feedback_are_retained(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "forbidden-paid-api-fixture")
    transport = Transport()
    c = client(transport, log_dir=tmp_path)
    for step in range(4):
        assert c.predict(observation(step)) == {"name": TOOL_NAME,
                                                "arguments": {"target": pose(), "steps": 3}}
        with pytest.raises(ValueError, match="feedback"):
            c.build_payload(observation(step + 1))
        c.record_result(feedback())
    history = c.history_snapshot()
    assert sum(item.get("role") == "user" for item in history) == 4
    assert sum(item.get("type") == "message" for item in history) == 4
    assert sum(item.get("type") == "function_call" for item in history) == 4
    results = [item for item in history if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in results] == [f"call_{i}" for i in range(1, 5)]
    assert [item["encrypted_content"] for item in history if item.get("type") == "reasoning"] == [
        f"opaque-{i}" for i in range(1, 5)]
    assert sum(part.get("type") == "input_image" for item in history
               for part in item.get("content", [])) == 8
    assert "omitted" not in json.dumps(history)
    assert c.history_stats()["retained_image_pairs"] == 4
    assert c.history_stats()["image_history_limit"] is None
    assert c.history_stats()["observations"] == c.history_stats()["feedback_records"] == 4
    assert c.budget_snapshot()["subscription_inference_calls"] == 4
    assert c.budget_snapshot()["api_key_calls"] == 0
    assert c.budget_snapshot()["input_tokens"] == 400
    next_payload = c.build_payload(observation(5))
    next_payload["input"][0]["content"].clear()
    assert c.history_snapshot() == history  # Preview never exposes mutable history.
    for path in tmp_path.glob("*.json"):
        assert TOKEN not in path.read_text() and "forbidden-paid-api-fixture" not in path.read_text()


@pytest.mark.parametrize("args", [
    {}, {"target": pose()}, {"target": pose(), "steps": 3, "note": "unexpected"},
    {"target": pose()[:-1], "steps": 3}, {"target": pose() + [0], "steps": 3},
    {"target": {"x": 0}, "steps": 3}, {"target": [0.] * 8, "steps": 3},
    {"target": pose(.5), "steps": 3}, {"target": pose(-1), "steps": 3},
    {"target": pose(True), "steps": 3}, {"target": [True] + pose()[1:], "steps": 3},
    {"target": [float("nan")] + pose()[1:], "steps": 3},
    {"target": [10 ** 500] + pose()[1:], "steps": 3},
    {"target": pose(), "steps": 0}, {"target": pose(), "steps": 6},
    {"target": pose(), "steps": 3.0}, {"target": pose(), "steps": True},
])
def test_bad_argument_representations_are_never_executable(args):
    with pytest.raises(SchemaError):
        validate_command({"name": TOOL_NAME, "arguments": args})


@pytest.mark.parametrize("name", ["give_up", "done", "shell", "robot.execute_eef_target", None])
def test_other_commands_are_unavailable(name):
    with pytest.raises(SchemaError):
        validate_command({"name": name, "arguments": {"target": pose(), "steps": 1}})


def test_large_well_formed_target_is_rejected_by_motion_and_returns_a_paired_tool_result():
    target = pose()
    target[0] += .1
    target[3] = 2.  # Normalization belongs to motion, not schema validation.
    c = client(Transport([response(args={"target": target, "steps": 3})]))
    command = c.predict(observation())
    assert command["arguments"]["target"] == target
    with pytest.raises(TargetLimitError) as exc:
        ShortTargetExecutor(STATE8, **command["arguments"])
    rejected = rejected_feedback(STATE8, **command["arguments"], detail=str(exc.value))
    c.record_result(rejected)
    output = json.loads(c.history_snapshot()[-1]["output"])
    assert output["status"] == "rejected" and output["executed_steps"] == 0
    assert output["requested_target"][3] == 1
    assert c.build_payload(observation(1))["input"][-2]["type"] == "function_call_output"


def test_successful_and_interrupted_segment_feedback_match_the_motion_adapter():
    for interrupted in (False, True):
        c = client()
        command = c.predict(observation())
        executor = ShortTargetExecutor(STATE8, **command["arguments"])
        for _ in range(1 if interrupted else 3):
            action = executor.next_action(STATE8)
            executor.observe_transition(STATE8, action, STATE8)
        if interrupted:
            executor.interrupt()
        c.record_result(executor.feedback())
        assert c.history_stats()["feedback_records"] == 1


@pytest.mark.parametrize("changes", [
    {"object_positions": []}, {"reward": 1}, {"success": True}, {"prompt": "alternate task"},
    {"step": -1}, {"step": True}, {"joint_positions": [0] * 6},
    {"joint_positions": [float("nan")] * 7}, {"robot_state": {"object": [1, 2, 3]}},
    {"external_image": "https://example.org/image.png"},
    {"wrist_image": image_to_data_url(np.zeros((8, 8, 3), np.uint8), size=8)},
])
def test_observation_boundary_rejects_invalid_input_before_authentication(changes):
    transport = Transport()
    c = client(transport)
    with pytest.raises(ValueError):
        c.predict(observation(**changes))
    assert not transport.requests and c.budget_snapshot()["api_calls"] == 0


def test_observation_requires_joints_and_rejects_nested_ground_truth_geometry():
    c = client()
    missing = observation()
    del missing["joint_positions"]
    with pytest.raises(ValueError):
        c.build_payload(missing)
    for where in ((), ("cameras", "wrist"), ("gripper",)):
        data = observation()
        target = data["geometry"]
        for part in where:
            target = target[part]
        target["object_pose"] = [0, 0, 0]
        with pytest.raises(ValueError):
            c.build_payload(data)


@pytest.mark.parametrize("changes", [
    {"success": True}, {"status": "success"}, {"status": "running"},
    {"executed_steps": -1}, {"executed_steps": 2}, {"requested_steps": 6},
    {"status": "rejected"}, {"control_hz": 15}, {"control_hz": 20.0},
    {"simulation_time_s": .3}, {"position_error_m": float("nan")},
    {"gripper_command": True}, {"gripper_command": 1}, {"measured_pose": pose(2)},
])
def test_invalid_feedback_is_atomic_and_cannot_inject_evaluator_data(changes):
    c = client()
    c.predict(observation())
    previous = c.history_snapshot()
    with pytest.raises(ValueError):
        c.record_result(feedback(**changes))
    assert c.history_snapshot() == previous
    c.record_result(feedback())
    with pytest.raises(ValueError):
        c.record_result(feedback())


def test_feedback_requested_steps_must_match_the_accepted_call():
    c = client()
    c.predict(observation())
    with pytest.raises(ValueError, match="pending command"):
        c.record_result(feedback(requested_steps=2, executed_steps=2, simulation_time_s=.1))


@pytest.mark.parametrize("reply", [
    response(status="incomplete"), response(model="gpt-6-sol"),
    response(reasoning={"effort": "high"}), response(output=response()["output"] * 2),
    response(name="shell", args={"cmd": "noop"}), response(output=[]),
    response(args={"target": [float("inf")] + pose()[1:], "steps": 1}),
    APIError("fixture transport error"),
])
def test_failed_responses_do_not_commit_history_or_retry(reply):
    transport = Transport([reply])
    c = client(transport)
    with pytest.raises((APIError, SchemaError)):
        c.predict(observation())
    assert len(transport.requests) == 1 and c.history_snapshot() == []
    assert c.history_stats()["pending_tool_result"] is False


@pytest.mark.parametrize("raw", ['{"target":[],"steps":1,"steps":2}', '{"target":[NaN],"steps":1}', 'not json'])
def test_strict_json_arguments_reject_duplicate_keys_nonfinite_constants_and_bad_json(raw):
    reply = response()
    reply["output"][-1]["arguments"] = raw
    with pytest.raises(SchemaError):
        client(Transport([reply])).predict(observation())


@pytest.mark.parametrize("namespace", ["other", "", None, [], False])
def test_other_namespaces_are_rejected(namespace):
    reply = response()
    reply["output"][-1]["namespace"] = namespace
    c = client(Transport([reply]))
    with pytest.raises(SchemaError, match="namespace"):
        c.predict(observation())
    assert c.history_snapshot() == []


def test_namespace_is_optional_and_sse_preserves_native_messages_and_calls():
    reply = response()
    del reply["output"][-1]["namespace"]
    event = {"type": "response.completed", "response": reply}
    parsed = read_completed_response([f"data: {json.dumps(event)}\n", "\n"])
    assert parsed["output"] == reply["output"]
    assert client(Transport([parsed])).predict(observation())["name"] == TOOL_NAME


@pytest.mark.parametrize("kwargs", [
    {"config": AstraConfig(model="gpt-6-sol")}, {"reasoning_effort": "high"},
    {"image_history": 2}, {"image_history": 0}, {"config": AstraConfig(max_api_calls=3)},
    {"config": AstraConfig(max_wall_time_s=60)}, {"config": AstraConfig(max_cost_usd=1)},
])
def test_configuration_preserves_subscription_medium_astra_full_history_and_no_episode_cap(kwargs):
    with pytest.raises(ValueError):
        client(**kwargs)


def test_reset_clears_pending_conversation_but_preserves_lifetime_usage():
    c = client()
    c.predict(observation())
    c.reset_history()
    assert c.history_snapshot() == []
    assert c.history_stats()["pending_tool_result"] is False
    assert c.budget_snapshot()["subscription_inference_calls"] == 1
    c.build_payload(observation())


def test_transport_error_redacts_the_subscription_token(tmp_path):
    c = client(Transport([APIError(f"failure {TOKEN}")]), log_dir=tmp_path)
    with pytest.raises(APIError) as exc:
        c.predict(observation())
    assert TOKEN not in str(exc.value)
    for path in tmp_path.glob("*.json"):
        assert TOKEN not in path.read_text()
