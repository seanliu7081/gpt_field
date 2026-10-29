"""V7 retries are bounded, charged, observable and preserve the input/action boundary."""
import copy
from dataclasses import replace
import json

import pytest

from gpt_field.astra import APIError, AstraConfig, BudgetExceeded, MissingCredentials, SchemaError
from gpt_field.chatgpt_plan import read_completed_response
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v6 import output_schema as schema_v6
from gpt_field.visual_field_subscription_v7 import (
    HISTORY_VERSION, VERSION, VisualFieldSubscriptionClientV7, output_schema, shared_guidance,
)
from gpt_field.visual_reference_overlay import LABEL
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation, input_texts
from test_visual_field_v6_protocol import prediction


def stream_error(*, code="stream_missing_completion", status=None):
    error = APIError("deliberately withheld arbitrary provider body")
    error.error_code, error.http_status = code, status
    return error


def client(*responses, **kwargs):
    transport = FakeTransport(*(value if isinstance(value, BaseException) else envelope(
        value, model="gpt-6-astra", reasoning={"effort": "medium"}, id=f"v7_{i}")
        for i, value in enumerate(responses)))
    instance = VisualFieldSubscriptionClientV7(token_provider=lambda: "offline-oauth-fixture",
                                               transport=transport, **kwargs)
    return instance, transport


def invalid():
    value = prediction(controller="visual_field", phase="release")
    assert value["field_intent"]["motion_axes"] == "xyz"
    return value


def current_record(payload):
    return [part for part in input_texts(payload) if part["type"] == "observation"][-1]


def current_images(payload):
    message = [message for message in payload["input"] if message.get("role") == "user"
               and message.get("content") and '"type":"observation"' in message["content"][0].get("text", "")][-1]
    return [part["image_url"] for part in message["content"] if part["type"] == "input_image"]


@pytest.mark.parametrize("order", ["stream_then_schema", "schema_then_stream"])
def test_mixed_retries_share_one_observation_and_never_exceed_three_requests(order, tmp_path):
    first = [stream_error(), invalid()] if order == "stream_then_schema" else [invalid(), stream_error()]
    accepted = prediction(controller="visual_field")
    instance, transport = client(*first, accepted, prediction(), log_dir=tmp_path)
    context = {"refresh_reason": "arrival", "visual_feedback": {"fixture": "observed only"}}
    instance.observable_context = context
    obs = observation().to_dict(); before = copy.deepcopy(obs)
    result = instance.predict(obs)
    assert result["controller"] == "visual_field"
    assert "orientation_command" not in result["field_intent"]
    assert result["field_intent"]["orientation_target_xyzw"] == [0., 0., 0., 1.]
    assert len(transport.calls) == 3 and obs == before
    assert instance.observable_context is context
    records = [current_record(payload) for payload in transport.calls]
    images = [current_images(payload) for payload in transport.calls]
    assert all(pair == images[0] for pair in images) and len(images[0]) == 2
    for record in records:
        assert record["observation_index"] == 0
        assert record["observation/state"] == list(obs["observation/state"])
        assert record["observation/joint_position"] == list(obs["observation/joint_position"])
    final_context = records[-1]["observable_context"]
    assert final_context["transport_retry"]["error_code"] == "stream_missing_completion"
    assert "arbitrary provider" not in json.dumps(final_context)
    assert final_context["output_repair"]["rejected_model_output_json"] == json.dumps(
        invalid(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert instance.history_stats()["observations"] == 1
    assert instance.history_stats()["semantic_repair_requests"] == 1
    assert instance.history_stats()["stream_retry_requests"] == 1
    budget = instance.budget_snapshot()
    assert budget["api_calls"] == 3 and budget["api_key_calls"] == 0
    assert budget["unknown_usage_calls"] == 1
    logs = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("*.result.json"))]
    assert [item["status"] for item in logs] == ["error", "error", "valid"]
    assert len(list(tmp_path.glob("*.request.json"))) == 3
    assert len(list(tmp_path.glob("*.response.json"))) == 2
    retries = [json.loads(path.read_text()) for path in tmp_path.glob("*.v7_retry.json")]
    assert len(retries) == 1 and retries[0]["actions_executed_from_failed_response"] == 0
    # Accepted raw wire, not normalized quaternion/boxes or failed output, is
    # committed. No robot execute method can be reached inside this client.
    instance.predict(observation(1).to_dict())
    previous = [json.loads(part["text"]) for message in transport.calls[-1]["input"]
                if message.get("role") == "assistant" for part in message.get("content", [])
                if part.get("type") == "output_text"]
    assert previous == [accepted]
    assert current_record(transport.calls[-1])["observable_context"] == context


@pytest.mark.parametrize("failures,expected_calls,error_type", [
    ([stream_error(), stream_error()], 2, APIError),
    ([invalid(), invalid()], 2, SchemaError),
    ([stream_error(), invalid(), stream_error()], 3, APIError),
    ([invalid(), stream_error(), invalid()], 3, SchemaError),
])
def test_second_same_failure_is_terminal_and_commits_no_action_or_history(failures, expected_calls, error_type):
    instance, transport = client(*failures, prediction())
    original = instance.observable_context
    with pytest.raises(error_type): instance.predict(observation().to_dict())
    assert len(transport.calls) == expected_calls
    assert instance.history_snapshot() == [] and instance.history_stats()["observations"] == 0
    assert instance.observable_context is original and instance._v5_repair_record is None
    assert len(transport.responses) == 1  # No unbounded recovery or hidden fallback.


@pytest.mark.parametrize("error", [
    stream_error(code="invalid_api_key", status=401), stream_error(code="insufficient_quota", status=429),
    stream_error(code="rate_limit_exceeded", status=429), stream_error(code="subscription_stream_interrupted"),
    stream_error(code="subscription_network_error"), stream_error(code="stream_invalid_json"),
    stream_error(code="stream_missing_completion", status=503), APIError("untyped transport error"),
    TimeoutError("timeout"), BudgetExceeded("budget exhausted"), MissingCredentials("no authorization"),
])
def test_other_failures_never_retry_or_fallback(error):
    instance, transport = client(error, prediction())
    with pytest.raises((APIError, BudgetExceeded)): instance.predict(observation().to_dict())
    assert len(transport.calls) == 1
    assert instance.budget_snapshot()["api_key_calls"] == 0
    assert instance.history_stats()["stream_retry_requests"] == 0


def test_malformed_output_extraction_has_no_semantic_repair_record():
    instance, transport = client(prediction(), prediction())
    transport.responses[0]["output"][1]["content"][0]["text"] = "incomplete {"
    with pytest.raises(SchemaError): instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and instance.history_stats()["semantic_repair_requests"] == 0


def test_real_missing_completed_parser_error_retries_without_using_partial_text():
    instance, transport = client(prediction(), prediction())
    incomplete = [b'data: {"type":"response.output_text.delta","delta":"PARTIAL-NOT-ACTION"}\n',
                  b'\n', b'data: [DONE]\n', b'\n']
    try:
        read_completed_response(incomplete)
    except APIError as error:
        transport.responses[0] = error  # Raw local parser exception has no .error_code yet.
    assert instance.predict(observation().to_dict()) == prediction()
    assert len(transport.calls) == 2
    assert "PARTIAL-NOT-ACTION" not in json.dumps(transport.calls[-1])


def test_retry_reserves_budget_again_and_all_attempt_waits_are_counted():
    class Clock:
        now = 0.
        def __call__(self): return self.now
    clock = Clock()
    instance, transport = client(stream_error(), invalid(), prediction(), clock=clock)
    def timed_transport(*args):
        clock.now += 7.
        return transport(*args)
    instance.transport = timed_transport
    instance.predict(observation().to_dict())
    assert clock.now == 21.
    assert instance.budget_snapshot()["elapsed_s"] == 21.
    assert instance.budget_snapshot()["api_calls"] == 3
    config = replace(instance.config, max_api_calls=1)
    limited, transport = client(stream_error(), prediction(), config=config)
    with pytest.raises(BudgetExceeded): limited.predict(observation().to_dict())
    assert len(transport.calls) == 1 and limited.budget_snapshot()["api_calls"] == 1


def test_repair_annotation_keeps_original_images_and_is_absent_when_healthy():
    instance, _ = client()
    obs = observation().to_dict()
    plain = instance.build_payload(obs)
    instance.observable_context = {"visual_feedback": {"visual": {
        "reference_health": {"repair_required": True, "status": "unresponsive"},
        "tracking": {"moving": {"tracked_point_uv": [.2, .3], "reliable": True},
                     "target": {"tracked_point_uv": [.7, .6], "reliable": True}},
    }}}
    annotated = instance.build_payload(obs)
    assert current_images(annotated)[:2] == current_images(plain)
    assert len(current_images(annotated)) == 3 and len(current_images(plain)) == 2
    assert sum(part.get("text") == LABEL for message in annotated["input"]
               for part in message.get("content", [])) == 1
    assert instance.history_snapshot() == []


@pytest.mark.parametrize("task", ["bowl_plate", "drawer_bowl"])
def test_prompt_keeps_generic_constraints_and_subscription_contract(task):
    instance, transport = client(*(prediction() for _ in range(4)), task_key=task)
    for i in range(4): instance.predict(observation(i, task=task).to_dict())
    payload = transport.calls[-1]
    assert output_schema() == schema_v6()
    assert payload["model"] == "gpt-6-astra" and payload["reasoning"] == {"effort": "medium"}
    assert payload["text"]["format"]["schema"] == output_schema()
    assert payload["store"] is False and payload["stream"] is True
    assert instance.prompt_version == VERSION+"-hybrid"
    assert next(x for x in input_texts(payload) if x["type"] == "visual_history_summary")["protocol"] == HISTORY_VERSION
    text = shared_guidance(resolve_distance_task(task))
    assert len(text.split()) <= 950
    for phrase in ("reference_health", "fixed robot base", "not verified object identities",
                   "parallax", "stalled descent", "graspable", "Field release/z", "zero XYZ",
                   "positive half_extents_m", "CURRENT measured", "exactly five",
                   "tiny/noisy test is inconclusive", "not insufficient samples"):
        assert phrase in text
    instance.reset_history()
    assert instance.history_stats()["stream_retry_requests"] == 0
    assert instance.history_stats()["semantic_repair_requests"] == 0
    assert instance.budget_snapshot()["api_calls"] == 4  # Reset never erases usage.


@pytest.mark.parametrize("name", ["depth", "object_poses", "contacts", "camera_calibration", "libero_success"])
def test_privileged_context_still_fails_before_request_or_diagnostic(name):
    instance, transport = client(prediction())
    obs = observation().to_dict(); obs[name] = "not-policy-input"
    with pytest.raises(ValueError): instance.predict(obs)
    instance.observable_context = {"visual_feedback": {name: "not-policy-input"}}
    with pytest.raises(ValueError): instance.predict(observation().to_dict())
    assert not transport.calls
