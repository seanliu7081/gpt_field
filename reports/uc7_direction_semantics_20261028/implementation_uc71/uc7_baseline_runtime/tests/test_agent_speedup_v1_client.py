"""Offline contract tests for the stateless agent_SpeedUp_v1 client; no network."""

import copy
from email.message import Message
import io
import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

import gpt_field.agent_speedup_v1_client as module
from gpt_field.agent_speedup_v1_client import (
    AGENT_PROMPT_VERSION, AGENT_SPEEDUP_V1_VERSION, APPROACHES, HISTORY_PROTOCOL, MAX_WAYPOINT_M,
    PARAM_KEYS, SIDES, SKILLS, STRATEGIES, TASK_TARGETS, AgentSpeedUpV1Client, ScriptedAgentTransport,
    decision_instructions, decision_schema, default_smoke_script, pointing_instructions, pointing_schema,
)
from gpt_field.astra import APIError, AstraConfig, ENDPOINT, SchemaError, _canonical
from gpt_field.chatgpt_plan import UNSUPPORTED_FIELDS, _protocol_error
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.ordered_transport import ORDERED_TRANSPORT_VERSION, ordered_dumps, ordered_sse_transport
from gpt_field.speed_first_hybrid import OVERLOAD_RETRY_POLICY
from agent_speedup_fakes import png_url
from test_chatgpt_plan import sse
from test_field_history import FakeTransport, envelope


TOKEN = "offline-agent-oauth-fixture"
EXTERNAL = png_url(np.full((8, 8, 3), 120, np.uint8))
WRIST = png_url(np.full((8, 8, 3), 60, np.uint8))
MARKED = png_url(np.full((8, 8, 3), 200, np.uint8))


def decision(skill="reach", situation="Bowl ahead, left of the fingers.", **params):
    return {"situation": situation, "skill": skill,
            "params": {**{key: None for key in PARAM_KEYS}, **params}}


def summary():
    return {"eef_xyz_m": [0.1, 0.02, 1.0], "finger_gap_mm": 77.4,
            "tool_axes_world": {"wrist_image_right": [0., -1., 0.], "wrist_image_up": [1., 0., 0.],
                                "approach": [0., 0., -1.]},
            "ledger": [], "last_outcome": None}


def decision_request(**updates):
    return {"state_summary": summary(), "external_image": EXTERNAL, "wrist_image": WRIST, **updates}


def pointing_request(stage="coarse", **updates):
    return {"stage": stage, "image": MARKED, "target_phrase": "left rim of the black bowl",
            "context": {"strategy": "rim_pinch", "side": "wrist_left"}, **updates}


def reply(data, **updates):
    return envelope(data, **{"model": "gpt-6-astra", "reasoning": {"effort": "medium"}, **updates})


def make(transport=None, **kwargs):
    return AgentSpeedUpV1Client(token_provider=lambda: TOKEN, transport=transport, **kwargs)


def records(directory, suffix):
    return [json.loads(path.read_text()) for path in sorted(directory.glob(f"*{suffix}"))]


def walk(schema):
    yield schema
    if isinstance(schema, dict):
        for value in schema.values():
            yield from walk(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from walk(value)


class Clock:
    def __init__(self):
        self.now = 100.
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr(module.time, "sleep", value.sleep)
    monkeypatch.setattr(module.random, "uniform", lambda low, high: .25)
    return value


def overloaded(*, status=None, code="server_is_overloaded"):
    return _protocol_error("provider details " + TOKEN, http_status=status, error_code=code)


# schemas and instructions ----------------------------------------------------
@pytest.mark.parametrize("task_key", sorted(TASK_TARGETS))
def test_decision_schema_is_flat_strict_ordered_and_nullable_only_through_anyof(task_key):
    schema = decision_schema(task_key)
    assert list(schema["properties"]) == ["situation", "skill", "params"]
    params = schema["properties"]["params"]
    assert list(params["properties"]) == list(PARAM_KEYS)
    for node in walk(schema):
        if isinstance(node, dict):
            assert not isinstance(node.get("type"), list) and "const" not in node
            assert not {"pattern", "format", "exclusiveMaximum"} & set(node)
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert node["required"] == list(node["properties"])
    assert schema["properties"]["situation"] == {"type": "string", "minLength": 1, "maxLength": 280}
    assert schema["properties"]["skill"]["enum"] == list(SKILLS)
    props = params["properties"]
    for name, values in (("target", TASK_TARGETS[task_key]), ("strategy", STRATEGIES),
                         ("side", SIDES), ("approach", APPROACHES)):
        assert props[name] == {"anyOf": [{"type": "string", "enum": list(values)}, {"type": "null"}]}
    vector = props["waypoint_offset_m"]["anyOf"][0]
    assert (vector["minItems"], vector["maxItems"]) == (3, 3)
    assert vector["items"] == {"type": "number", "minimum": -MAX_WAYPOINT_M, "maximum": MAX_WAYPOINT_M}
    assert props["max_travel_m"]["anyOf"] == [{"type": "number", "minimum": .02, "maximum": .16},
                                              {"type": "null"}]
    assert decision_schema(task_key) == schema and decision_schema(task_key) is not schema


def test_pointing_schemas_and_unknown_arguments():
    assert pointing_schema("coarse") == {
        "type": "object", "additionalProperties": False, "required": ["cell_id"], "properties": {
            "cell_id": {"type": "integer", "minimum": 0, "maximum": 64}}}
    assert list(pointing_schema("fine")["properties"]) == ["point_id"]
    for call in (lambda: decision_schema("kitchen"), lambda: pointing_schema("medium"),
                 lambda: decision_instructions("kitchen"), lambda: pointing_instructions(None),
                 lambda: default_smoke_script("kitchen"), lambda: make(task_key="kitchen")):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("task_key", sorted(TASK_TARGETS))
def test_instructions_cover_the_supervisor_contract_compactly(task_key):
    text = decision_instructions(task_key)
    # prompt-2 (live run 001 fixes) adds the external-axes rule, reason glossary and held-object rules;
    # input size barely moves latency (r = 0.07 on the live run), output tokens do.
    # prompt-3 adds the drawer fixes and the R-5 reach facts (~100 tokens; input size barely moves latency)
    # prompt-4 adds multi-waypoint reach, chained handle skills and the hand footprint; drawer live 002 measured no
    # latency cost from input size (-0.00015 s per input token), output tokens drive latency
    # prompt-5 adds the drawer 002 bowl facts: overhang-aware hover, pull length, blocked_above_rim (~60 tokens)
    assert len(text) < 12000 and text.startswith(AGENT_PROMPT_VERSION)
    for phrase in ("more_waypoints_m", "hand body is about 20 cm", "grasp and then pull", "blocked_above_rim",
                   "anything overhanging it", "out from under anything overhanging it"):
        assert phrase in text, phrase
    for phrase in ("robot_base_direction_world", "frontal_arm_reach", "hover_beyond_frontal_reach",
                   "world_up_in_wrist_image", "pointed_not_a_bar", "finger_contact"):
        assert phrase in text, phrase
    for phrase in ("tool_axes_world apply ONLY to the WRIST image", "never convert external-image directions",
                   "external_image_axes_world", "REASONS", "reach_limit", "held_object_contact",
                   "lost_in_transport", "left_planning_grid", "held_height_above_pick_m", "target_history",
                   "loop_warning", "relative to the held object in the EXTERNAL image"):
        assert phrase in text
    assert resolve_distance_task(task_key).prompt in text
    for phrase in ("UPRIGHT", "tool_axes_world", "79 mm", "wrist_left", "wrist_right", "world_-x",
                   "+/-0.075", "CURRENT position", "only after its own sensor checks",
                   "Never centre the hand over a container wider", "never declare completion",
                   "situation first", *SKILLS, *STRATEGIES, *TASK_TARGETS[task_key]):
        assert phrase in text, phrase
    described = decision_instructions(task_key, "  a matte black bowl  ")
    assert described.startswith(text) and '"a matte black bowl"' in described
    for stage, key in (("coarse", "cell_id"), ("fine", "point_id")):
        pointing = pointing_instructions(stage)
        # prompt-3 (drawer live run 001): the view may be turned world-up; orientation comes from the context
        assert len(pointing) < 1500 and "image_orientation" in pointing and "1-64" in pointing
        assert "not mirrored" in pointing and "previous_answer_rejected" in pointing and "UPRIGHT" not in pointing
        assert f'{{"{key}": 0}}' in pointing and "single" in pointing


# payloads ---------------------------------------------------------------------
def test_payloads_are_one_user_message_without_history_or_authentication():
    fake = FakeTransport()
    provider = Mock(return_value=TOKEN)
    client = AgentSpeedUpV1Client(token_provider=provider, transport=fake, task_key="drawer_bowl")
    assert (client.reasoning_effort, client.prompt_version) == ("medium", AGENT_PROMPT_VERSION)
    assert client.task_prompt == resolve_distance_task("drawer_bowl").prompt and client.image_history == 1
    client._history = [{"role": "user", "content": [{"type": "input_text", "text": "stale"}]}]
    payload = client.build_payload(decision_request())
    assert list(payload) == ["model", "service_tier", "store", "instructions", "input", "text",
                             "stream", "reasoning", "include"]
    assert (payload["model"], payload["service_tier"], payload["store"], payload["stream"]) == (
        "gpt-6-astra", "default", False, True)
    assert payload["reasoning"] == {"effort": "medium"}
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert not UNSUPPORTED_FIELDS.intersection(payload)
    assert payload["instructions"] == decision_instructions("drawer_bowl")
    assert len(payload["input"]) == 1 and payload["input"][0]["role"] == "user"
    content = payload["input"][0]["content"]
    assert [part["type"] for part in content] == [
        "input_text", "input_text", "input_image", "input_text", "input_image"]
    assert [part["image_url"] for part in content if part["type"] == "input_image"] == [EXTERNAL, WRIST]
    assert all(part["detail"] == "high" for part in content if part["type"] == "input_image")
    assert "upright" in content[1]["text"] and "upright" in content[3]["text"]
    assert json.loads(content[0]["text"]) == {
        "request": "decision", "task": client.task_prompt, "state_summary": summary()}
    assert payload["text"]["format"] == {"type": "json_schema", "name": "agent_speedup_v1_decision",
                                         "strict": True, "schema": decision_schema("drawer_bowl")}
    for stage in ("coarse", "fine"):
        pointing = client.build_payload(pointing_request(stage), mode=f"pointing_{stage}")
        assert list(pointing) == list(payload) and len(pointing["input"]) == 1
        parts = pointing["input"][0]["content"]
        assert [part["type"] for part in parts] == ["input_text", "input_text", "input_image"]
        assert parts[2] == {"type": "input_image", "detail": "high", "image_url": MARKED}
        assert json.loads(parts[0]["text"])["target_phrase"] == "left rim of the black bowl"
        assert pointing["text"]["format"]["schema"] == pointing_schema(stage)
        assert pointing["instructions"] == pointing_instructions(stage)
    assert not any(item.get("type") == "reasoning" for item in payload["input"])
    provider.assert_not_called()
    assert fake.calls == [] and client.budget_snapshot()["api_calls"] == 0
    assert make().transport is ordered_sse_transport


@pytest.mark.parametrize("mode,request_value", [
    ("decision", {"state_summary": summary(), "external_image": EXTERNAL}),
    ("decision", decision_request(ledger=[])),
    ("decision", decision_request(state_summary={"gap": float("nan")})),
    ("decision", decision_request(state_summary={"gap": np.int64(3)})),
    ("decision", decision_request(state_summary=[1, 2])),
    ("decision", decision_request(wrist_image="https://example.invalid/wrist.png")),
    ("decision", decision_request(external_image="data:image/png;base64,@@@")),
    ("decision", decision_request(external_image="data:image/png;base64,aGVsbG8=")),
    ("pointing_coarse", pointing_request("fine")),
    ("pointing_fine", pointing_request("fine", target_phrase="  ")),
    ("pointing_fine", pointing_request("fine", context=[1])),
    ("pointing_coarse", decision_request()),
    ("hybrid", decision_request()),
])
def test_invalid_requests_fail_before_any_attempt(mode, request_value):
    fake = FakeTransport(reply(decision()))
    provider = Mock(return_value=TOKEN)
    client = AgentSpeedUpV1Client(token_provider=provider, transport=fake)
    with pytest.raises(ValueError):
        client.build_payload(request_value, mode=mode)
    with pytest.raises(ValueError):
        client.predict(request_value, mode=mode)
    provider.assert_not_called()
    assert fake.calls == [] and client.budget_snapshot()["api_calls"] == 0
    with pytest.raises(ValueError):
        client.point({"stage": "medium"})
    with pytest.raises(ValueError):
        client.point(None)


# wire order -------------------------------------------------------------------
def test_ordered_wire_sends_situation_before_skill_and_params(tmp_path):
    captured = []

    def fake_open(request, timeout_s):
        captured.append(request)
        stream = io.BytesIO(sse({"type": "response.completed",
                                 "response": reply(decision(waypoint_offset_m=[0, 0, -.02]))}).encode())
        stream.headers = Message()
        stream.headers["Content-Type"] = "text/event-stream; charset=utf-8"
        return stream

    client = make(log_dir=tmp_path)
    with patch("gpt_field.chatgpt_plan._open_public", side_effect=fake_open):
        result = client.decide(decision_request())
    assert result == decision(waypoint_offset_m=[0., 0., -.02])
    assert type(result["params"]["waypoint_offset_m"][0]) is float
    request = captured[0]
    assert request.full_url == ENDPOINT and request.get_header("Authorization") == f"Bearer {TOKEN}"
    body = request.data.decode("utf-8")
    assert body == ordered_dumps(client.build_payload(decision_request()))
    schema = body[body.index('"schema":'):]
    assert schema.startswith('"schema":{"type":"object","additionalProperties":false,"properties":{"situation":')
    assert schema.index('"situation":') < schema.index('"skill":') < schema.index('"params":')
    order = [schema.index(f'"{key}":{{"anyOf"') for key in PARAM_KEYS]
    assert order == sorted(order)
    assert body.index('"model":') < body.index('"instructions":') < body.index('"input":') < body.index('"text":')
    assert '"properties":{"params"' in _canonical(decision_schema("bowl_plate"))  # legacy wire would sort
    assert client.last_output_key_order == ["situation", "skill", "params"]
    assert records(tmp_path, ".request.json")[0]["wire_transport_version"] == ORDERED_TRANSPORT_VERSION
    assert TOKEN not in "".join(path.read_text() for path in tmp_path.iterdir())


def test_output_key_order_is_recorded_not_rejected(tmp_path):
    data = decision()
    reordered = {"skill": data["skill"], "params": data["params"], "situation": data["situation"]}
    client = make(FakeTransport(reply(reordered)), log_dir=tmp_path)
    assert client.decide(decision_request()) == data
    assert client.last_output_key_order == ["skill", "params", "situation"]
    assert records(tmp_path, ".result.json")[0]["output_key_order"] == ["skill", "params", "situation"]


# statelessness, validation and accounting ----------------------------------------
def test_requests_are_stateless_and_counted_by_kind():
    first = decision("grasp_with", target="black_bowl", strategy="rim_pinch", side="wrist_left")
    fake = FakeTransport(reply(first), reply({"cell_id": 12}), reply({"point_id": None}), reply(first))
    client = make(fake)
    assert client.decide(decision_request()) == first
    assert client._history == [] and client.history_snapshot() == []
    assert client.point(pointing_request("coarse")) == {"cell_id": 12}
    assert client.point(pointing_request("fine")) == {"point_id": None}
    assert client.decide(decision_request()) == first
    assert fake.calls[3] == fake.calls[0]
    assert client.request_key(fake.calls[3], None, "decision") == client.request_key(fake.calls[0], None, "decision")
    assert client.request_key(fake.calls[0], None, "decision") != client.request_key(fake.calls[0], None, "pointing_coarse")
    for payload in fake.calls:
        assert len(payload["input"]) == 1 and payload["input"][0]["role"] == "user"
    stats = client.history_stats()
    assert stats == {"protocol": HISTORY_PROTOCOL, "accepted_requests": 4, "messages": 0,
                     "retained_image_pairs": 0,
                     "calls_by_kind": {"decision": 2, "pointing_coarse": 1, "pointing_fine": 1},
                     "accepted_by_kind": {"decision": 2, "pointing_coarse": 1, "pointing_fine": 1},
                     "overload_retry_policy": OVERLOAD_RETRY_POLICY, "overload_retry_requests": 0,
                     "overload_retry_wait_s": 0.}
    json.dumps(stats, allow_nan=False)
    budget = client.budget_snapshot()
    assert budget["api_calls"] == 4 and budget["input_tokens"] == 400
    client.reset_history()
    assert client.history_stats()["calls_by_kind"] == {"decision": 0, "pointing_coarse": 0, "pointing_fine": 0}
    assert client.budget_snapshot()["api_calls"] == 4


BAD_DECISIONS = [
    {"situation": "x", "skill": "reach"},
    {**decision(), "close": True},
    decision("close"),
    decision(situation="x" * 281),
    decision(target="top_drawer_handle"),
    decision(strategy="hook_and_pull"),
    decision(side="left"),
    decision(approach="world_+z"),
    decision(waypoint_offset_m=[0., 0.]),
    decision(waypoint_offset_m=[0., 0., .08]),
    decision(waypoint_offset_m=[True, 0., 0.]),
    decision(waypoint_offset_m=[0., "0", 0.]),
    decision(max_travel_m=.01),
    decision(max_travel_m=.2),
    {**decision(), "params": {**decision()["params"], "gripper": "close"}},
    {**decision(), "params": None},
    [decision()],
]


@pytest.mark.parametrize("bad", BAD_DECISIONS)
def test_bad_decision_structure_raises_schema_error(bad):
    client = make(FakeTransport(reply(bad)))
    with pytest.raises(SchemaError):
        client.decide(decision_request())
    assert client._history == [] and client.history_stats()["accepted_requests"] == 0
    assert client.history_stats()["calls_by_kind"]["decision"] == 1


@pytest.mark.parametrize("stage,bad", [
    ("coarse", {"cell_id": -1}), ("coarse", {"cell_id": 65}), ("coarse", {"cell_id": 3.0}),
    ("coarse", {"cell_id": True}), ("coarse", {"cell_id": "3"}), ("coarse", {"point_id": 3}),
    ("fine", {"cell_id": 3}), ("fine", {"point_id": 3, "cell_id": 2}), ("fine", None),
])
def test_bad_pointing_structure_raises_schema_error(stage, bad):
    client = make(FakeTransport(reply(bad)))
    with pytest.raises(SchemaError):
        client.point(pointing_request(stage))
    assert client._history == []


def test_valid_bounds_unparseable_text_and_direct_validation():
    edge = decision(waypoint_offset_m=[MAX_WAYPOINT_M, -MAX_WAYPOINT_M, 0], max_travel_m=.16)
    garbled = reply(decision())
    garbled["output"][1]["content"][0]["text"] = '{"situation": "x", "skill":'
    nonfinite = reply(decision())
    nonfinite["output"][1]["content"][0]["text"] = ordered_dumps(decision()).replace('"max_travel_m":null', '"max_travel_m":NaN')
    client = make(FakeTransport(reply(edge), garbled, nonfinite))
    assert client.decide(decision_request())["params"]["waypoint_offset_m"] == [.075, -.075, 0.]
    for _ in range(2):
        with pytest.raises(SchemaError):
            client.decide(decision_request())
    with pytest.raises(SchemaError):
        client.validate_response(object(), None)
    assert client.history_stats()["accepted_requests"] == 1


def test_model_and_effort_mismatch_are_rejected_without_retry(clock):
    fake = FakeTransport(reply(decision(), model="gpt-6-astra-mini"), reply(decision(), reasoning={"effort": "high"}))
    client = make(fake, clock=clock)
    with pytest.raises(APIError) as model_error:
        client.decide(decision_request())
    assert model_error.value.error_code == "response_model_mismatch"
    with pytest.raises(APIError) as effort_error:
        client.decide(decision_request())
    assert effort_error.value.error_code == "response_reasoning_effort_mismatch"
    assert len(fake.calls) == 2 and clock.waits == [] and client._history == []
    scripted = make(ScriptedAgentTransport([decision()], effort="low"))
    with pytest.raises(APIError):
        scripted.decide(decision_request())
    with pytest.raises(ValueError):
        make(config=AstraConfig(model="gpt-5"))
    with pytest.raises(ValueError):
        make(config=AstraConfig(max_cost_usd=1.))


# overload retry ----------------------------------------------------------------
@pytest.mark.parametrize("status", [None, 503])
def test_overload_retries_the_identical_request_once(status, clock, tmp_path):
    fake = FakeTransport(overloaded(status=status), reply(decision()))
    client = make(fake, clock=clock, log_dir=tmp_path)
    assert client.decide(decision_request()) == decision()
    assert fake.calls[0] == fake.calls[1] and clock.waits == [2.25]
    stats = client.history_stats()
    assert stats["overload_retry_requests"] == 1 and stats["overload_retry_wait_s"] == pytest.approx(2.25)
    assert stats["calls_by_kind"]["decision"] == 2 and stats["accepted_by_kind"]["decision"] == 1
    assert stats["accepted_requests"] == 1 and client._history == []
    budget = client.budget_snapshot()
    assert budget["api_calls"] == 2 and budget["unknown_usage_calls"] == 1
    requests = records(tmp_path, ".request.json")
    assert [item["attempt"] for item in requests] == [1, 2]
    assert requests[0]["request_key"] == requests[1]["request_key"]
    assert all(item["overload_retry_policy"] == OVERLOAD_RETRY_POLICY for item in requests)
    retries = records(tmp_path, ".overload_retry.json")
    assert len(retries) == 1 and retries[0]["status"] == "bounded_retry_scheduled"
    assert (retries[0]["attempt"], retries[0]["next_attempt"], retries[0]["request_kind"]) == (1, 2, "decision")
    errors = [item for item in records(tmp_path, ".result.json") if item["status"] == "error"]
    assert errors[0]["error_code"] == "server_is_overloaded" and errors[0]["request_kind"] == "decision"
    assert TOKEN not in "".join(path.read_text() for path in tmp_path.iterdir())


def test_overload_retry_is_bounded_and_specific(clock):
    fake = FakeTransport(overloaded(), overloaded(), overloaded())
    client = make(fake, clock=clock)
    with pytest.raises(APIError) as caught:
        client.decide(decision_request())
    assert caught.value.error_code == "server_is_overloaded" and TOKEN not in str(caught.value)
    assert len(fake.calls) == 3 and clock.waits == [2.25, 4.25] and client._history == []
    for error in (overloaded(status=503, code="unknown_error"), overloaded(status=429)):
        fake = FakeTransport(error, reply(decision()))
        with pytest.raises(APIError):
            make(fake, clock=clock).decide(decision_request())
        assert len(fake.calls) == 1
    fake = FakeTransport(overloaded(), reply({"point_id": 7}))
    client = make(fake, clock=clock)
    assert client.point(pointing_request("fine")) == {"point_id": 7}
    assert fake.calls[0] == fake.calls[1]
    assert client.history_stats()["calls_by_kind"] == {"decision": 0, "pointing_coarse": 0, "pointing_fine": 2}


# logs and offline fixture --------------------------------------------------------
def test_logs_record_request_kinds_versions_and_pre_request_stats(tmp_path):
    fake = FakeTransport(reply(decision()), reply({"cell_id": 36}), reply({"point_id": 28}))
    client = make(fake, log_dir=tmp_path)
    client.decide(decision_request())
    client.point(pointing_request("coarse"))
    client.point(pointing_request("fine"))
    assert len(list(tmp_path.iterdir())) == 9
    requests = records(tmp_path, ".request.json")
    assert [item["mode"] for item in requests] == ["decision", "pointing_coarse", "pointing_fine"]
    for index, item in enumerate(requests):
        assert item["wire_transport_version"] == ORDERED_TRANSPORT_VERSION
        assert item["prompt_version"] == AGENT_PROMPT_VERSION
        assert item["agent_speedup_v1_version"] == AGENT_SPEEDUP_V1_VERSION
        assert (item["history_protocol"], item["image_history"], item["attempt"]) == (HISTORY_PROTOCOL, 1, 1)
        assert item["request_ordinal"] == index + 1 and "offline_mock" not in item
        assert sum(item["history_stats_before_request"]["calls_by_kind"].values()) == index
        assert item["request_key"] == client.request_key(item["payload"], None, item["mode"])
    results = records(tmp_path, ".result.json")
    assert [item["request_kind"] for item in results] == ["decision", "pointing_coarse", "pointing_fine"]
    assert all(item["status"] == "valid" for item in results)
    assert [item["output_key_order"] for item in results] == [["situation", "skill", "params"],
                                                              ["cell_id"], ["point_id"]]
    assert TOKEN not in "".join(path.read_text() for path in tmp_path.iterdir())


def test_scripted_transport_pops_decisions_cycles_pointing_and_labels_mock(tmp_path):
    for task_key in TASK_TARGETS:
        for item in default_smoke_script(task_key):
            assert module._validate_decision(copy.deepcopy(item), task_key) == item
    script = default_smoke_script("bowl_plate")
    transport = ScriptedAgentTransport(script, coarse=(36, None), fine=(28,))
    client = make(transport, log_dir=tmp_path)
    first = client.decide(decision_request())
    assert first["skill"] == "reach" and first["params"]["waypoint_offset_m"] == [0., 0., -.02]
    assert first["params"]["strategy"] is None
    assert client.decide(decision_request())["skill"] == "retreat"
    hold = client.decide(decision_request())
    assert hold == {"situation": "scripted smoke hold", "skill": "retreat",
                    "params": {key: None for key in PARAM_KEYS}}
    assert client.decide(decision_request()) == hold
    assert [client.point(pointing_request("coarse"))["cell_id"] for _ in range(3)] == [36, None, 36]
    assert client.point(pointing_request("fine")) == {"point_id": 28}
    assert transport.kinds == ["decision"] * 4 + ["pointing_coarse"] * 3 + ["pointing_fine"]
    assert transport.calls[0] == client.build_payload(decision_request())
    transport.calls[0]["model"] = "mutated"
    assert script[0]["skill"] == "reach" and len(script) == 2
    assert client.budget_snapshot()["billing"] == "offline_mock"
    assert client.history_stats()["accepted_by_kind"] == {"decision": 4, "pointing_coarse": 3, "pointing_fine": 1}
    assert all(item.get("offline_mock") is True for item in records(tmp_path, ".json"))
    empty = make(ScriptedAgentTransport([], coarse=(), fine=()))
    assert empty.point(pointing_request("coarse")) == {"cell_id": None}
    with pytest.raises(ValueError):
        transport(ENDPOINT, {**client.build_payload(decision_request()), "max_output_tokens": 10}, {}, 1)
    payload = client.build_payload(decision_request())
    payload["text"]["format"]["name"] = "other"
    with pytest.raises(ValueError):
        transport(ENDPOINT, payload, {}, 1)
    for bad in ({"skill": "reach"}, [1]):
        with pytest.raises(ValueError):
            ScriptedAgentTransport(bad)
    with pytest.raises(ValueError):
        ScriptedAgentTransport([], coarse=(1.5,))
    with pytest.raises(SchemaError):
        make(ScriptedAgentTransport([decision("close")])).decide(decision_request())


def test_blank_situation_is_accepted_and_zero_mark_means_not_visible():
    client = make(FakeTransport(reply(decision(situation="   "))))
    assert client.decide(decision_request())["situation"] == "   "
    client = make(FakeTransport(reply({"cell_id": 0})))
    assert client.point(pointing_request("coarse")) == {"cell_id": None}
