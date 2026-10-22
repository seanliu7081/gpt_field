"""Provider boundary tests, including preserving full visual history."""
import copy

import pytest

from .transport import robot_only_payload, validate_output_item, validate_completed_response


def fixture():
    return {"model": "gpt-6-astra", "reasoning": {"effort": "medium", "context": "all_turns"},
            "store": False, "stream": True, "parallel_tool_calls": True,
            "input": [{"type": "additional_tools", "tools": [{"name": "skills"}]},
                      {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,fake"}]},
                      {"type": "function_call_output", "call_id": "previous", "output": "actual result"}],
            "tools": [{"name": "exec"}]}


def test_only_robot_tools_and_full_history_preserved():
    payload = fixture()
    before = copy.deepcopy(payload)
    result = robot_only_payload(payload)
    assert payload == before
    assert result["input"] == payload["input"][1:]
    assert result["tools"][0]["name"] == "robot"
    assert result["tools"][0]["tools"][0]["name"] == "execute_eef_target"
    assert len(result["tools"]) == 1
    assert result["parallel_tool_calls"] is False
    assert result["reasoning"] == payload["reasoning"]


@pytest.mark.parametrize("field,value", [("model", "other"), ("reasoning", {"effort": "high"}),
                                           ("store", True), ("stream", False)])
def test_protocol_mismatch_is_not_forwarded(field, value):
    payload = fixture()
    payload[field] = value
    with pytest.raises(ValueError):
        robot_only_payload(payload)


@pytest.mark.parametrize("item", [
    {"type": "custom_tool_call", "name": "exec"},
    {"type": "function_call", "namespace": "skills", "name": "read"},
    {"type": "function_call", "namespace": "robot", "name": "unknown"},
    {"type": "web_search_call"}, {"type": "compaction"}, {"type": "local_shell_call"},
])
def test_native_tool_output_blocked_before_codex(item):
    with pytest.raises(ValueError):
        validate_output_item(item)


def test_robot_proposal_and_text_are_allowed():
    for item in ({"type": "function_call", "namespace": "robot", "name": "execute_eef_target"},
                 {"type": "message"}, {"type": "reasoning"}):
        validate_output_item(item)


@pytest.mark.parametrize("response", [{"model": "other", "output": []},
    {"model": "gpt-6-astra", "reasoning": {"effort": "high"}, "output": []}])
def test_response_model_or_effort_change_rejected(response):
    with pytest.raises(ValueError):
        validate_completed_response(response)
