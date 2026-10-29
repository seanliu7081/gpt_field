"""No-network contract tests for ChatGPT subscription field inference."""

import copy
from dataclasses import replace
from email.message import Message
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from gpt_field.astra import APIError, AstraConfig, BudgetExceeded, ENDPOINT, MissingCredentials, SchemaError
from gpt_field.chatgpt_plan import (
    ChatGPTPlanFieldClient, MODELS_ENDPOINT, UNSUPPORTED_FIELDS, _NoRedirect,
    list_plan_models, read_completed_response, subscription_sse_transport,
)
from gpt_field.field_history import FieldHistoryAstraClient, FieldHistoryPolicy
from gpt_field.schema import GridSpec
from test_field_history import envelope, execute, observation, prediction


TOKEN = "oauth-access-fixture-one"


def sse(event, *, name=None, ending="\n\n"):
    return (("event: " + name + "\n") if name else "") + "data: " + json.dumps(event) + ending


def completed(data=None, **updates):
    response = envelope(data, model="gpt-6-astra", **updates)
    return {"type": "response.completed", "response": response}


def streamed_items(data=None):
    final = completed(data)
    items = final["response"]["output"]
    final["response"]["output"] = []
    events = [{"type": "response.output_item.done", "output_index": index, "item": item}
              for index, item in enumerate(items)]
    return events, final


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.headers = []

    def __call__(self, endpoint, payload, headers, timeout_s):
        self.calls.append(copy.deepcopy(payload))
        self.headers.append(dict(headers))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)


def client(fake=None, **kwargs):
    return ChatGPTPlanFieldClient(token_provider=lambda: TOKEN, transport=fake, **kwargs)


class StreamTests(unittest.TestCase):
    def test_assemble_empty_terminal_from_complete_items_preserves_usage_and_opaque_data(self):
        events, final = streamed_items()
        final["response"]["usage"] = {"input_tokens": 1099, "output_tokens": 3285}
        events[1]["item"]["status"] = "completed"
        response_id = final["response"]["id"]
        wire = sse({"type": "response.created", "response": {"id": response_id}})
        for event in reversed(events):  # Output index, not arrival order, defines ordering.
            wire += sse({**event, "response_id": response_id})
        wire += sse(final)
        result = read_completed_response(io.StringIO(wire))
        self.assertEqual(result["output"], [event["item"] for event in events])
        self.assertEqual(result["output"][0]["encrypted_content"], "opaque-fixture")
        self.assertEqual(result["output"][1]["phase"], "final_answer")
        self.assertEqual(result["usage"], {"input_tokens": 1099, "output_tokens": 3285})
        self.assertEqual(result["_subscription_transport"], {
            "terminal_output_count": 0, "completed_item_count": 2,
            "output_assembly_source": "response.output_item.done", "sse_completion_verified": True})
        self.assertEqual(final["response"]["output"], [])

    def test_assembly_can_follow_added_and_delta_but_never_uses_delta_text(self):
        events, final = streamed_items()
        message = events[1]["item"]
        partial = {**message, "content": [], "status": "in_progress"}
        prefix = [{"type": "response.output_item.added", "output_index": 1, "item": partial},
                  {"type": "response.output_text.delta", "output_index": 1,
                   "item_id": message["id"], "delta": "invalid partial JSON is ignored"}]
        wire = "".join(sse(event) for event in prefix + events + [final])
        self.assertEqual(read_completed_response(io.StringIO(wire))["output"][1], message)

    def test_assembly_rejects_missing_done_gaps_partial_trailing_items_or_unindexed_delta(self):
        events, final = streamed_items()
        trailing = {"type": "response.output_text.delta", "output_index": 2,
                    "item_id": "msg_unfinished", "delta": "partial"}
        unindexed = [{"type": kind, "delta": "partial"} for kind in (
            "response.output_text.delta", "response.content_part.added", "response.reasoning_text.delta")]
        unindexed.append({"type": "response.some_future_content.done", "item_id": "msg_missing_index"})
        trailing_added = {"type": "response.output_item.added", "output_index": 2,
                          "item": {"type": "message", "id": "msg_unfinished", "status": "in_progress"}}
        prefixes = [[], events[1:], events + [trailing], events + [trailing_added]]
        prefixes.extend(events + [event] for event in unindexed)
        for prefix in prefixes:
            with self.subTest(prefix=prefix), self.assertRaises(APIError):
                read_completed_response(io.StringIO("".join(sse(event) for event in prefix + [final])))

    def test_assembly_rejects_duplicate_done_invalid_index_and_item_conflicts(self):
        events, final = streamed_items()
        invalid_indexes = [-1, True, "0", None]
        prefixes = [events + [events[0]]]
        prefixes += [[{**events[0], "output_index": index}] + events[1:] for index in invalid_indexes]
        prefixes += [[{**events[0], "item_id": "rs_other"}] + events[1:]]
        prefixes += [[events[0], {**events[1], "item": {**events[1]["item"], "id": events[0]["item"]["id"]}}]]
        no_index = copy.deepcopy(events[0])
        del no_index["output_index"]
        prefixes.append([no_index] + events[1:])
        for prefix in prefixes:
            with self.subTest(prefix=prefix), self.assertRaises(APIError):
                read_completed_response(io.StringIO("".join(sse(event) for event in prefix + [final])))

    def test_assembly_rejects_incomplete_done_items(self):
        events, final = streamed_items()
        message = events[1]["item"]
        items = [{**message, "status": "in_progress"}, {**message, "status": "incomplete"},
                 {**message, "content": []}, {**message, "content": [{"type": "output_text"}]},
                 {**message, "role": "user"}, {**message, "id": ""}, None]
        for item in items:
            prefix = [events[0], {**events[1], "item": item}, final]
            with self.subTest(item=item), self.assertRaises(APIError):
                read_completed_response(io.StringIO("".join(sse(event) for event in prefix)))

    def test_assembly_checks_response_identity_on_created_done_and_completed(self):
        events, final = streamed_items()
        prefixes = [
            [{"type": "response.created", "response": {"id": "resp_different"}}] + events,
            [{**events[0], "response_id": "resp_different"}, events[1]],
            [{**events[0], "response_id": 123}, events[1]],
        ]
        for prefix in prefixes:
            with self.subTest(prefix=prefix), self.assertRaises(APIError):
                read_completed_response(io.StringIO("".join(sse(event) for event in prefix + [final])))

    def test_complete_terminal_must_agree_with_streamed_done_items(self):
        events, _ = streamed_items()
        full = completed()
        wire = "".join(sse(event) for event in events + [full])
        self.assertEqual(read_completed_response(io.StringIO(wire)), full["response"])
        changed = copy.deepcopy(events)
        changed[1]["item"]["phase"] = "commentary"
        for prefix in (changed, events + [{"type": "response.output_text.delta", "output_index": 2,
                                          "item_id": "msg_extra", "delta": "partial"}]):
            with self.subTest(prefix=prefix), self.assertRaises(APIError):
                read_completed_response(io.StringIO("".join(sse(event) for event in prefix + [full])))

    def test_done_items_do_not_authorize_execution_before_completion_or_after_failure(self):
        events, final = streamed_items()
        failures = [[], [{"type": "response.failed", "response": {"error": {"code": "usage_limit"}}}, final],
                    [{"type": "response.incomplete", "response": {}}, final]]
        for suffix in failures:
            with self.subTest(suffix=suffix), self.assertRaises(APIError):
                read_completed_response(io.StringIO("".join(sse(event) for event in events + suffix)))

    def test_assembled_refusal_is_never_committed_or_executed(self):
        events, final = streamed_items()
        events[1]["item"]["content"] = [{"type": "refusal", "refusal": "Cannot comply"}]
        response = read_completed_response(io.StringIO("".join(sse(event) for event in events + [final])))
        instance = client(FakeTransport(response))
        with self.assertRaises(APIError):
            instance.predict(observation().to_dict())
        self.assertEqual(instance.history_snapshot(), [])

    def test_assembled_field_commits_exact_items_and_mime_metadata_is_merged(self):
        events, final = streamed_items()
        wire = "".join(sse(event) for event in events + [final])
        stream = io.BytesIO(wire.encode())
        stream.headers = Message()
        stream.headers["Content-Type"] = "text/plain"
        with patch("gpt_field.chatgpt_plan._open_public", return_value=stream):
            response = subscription_sse_transport(ENDPOINT, {"store": False, "stream": True}, {}, 30)
        self.assertEqual(response["_subscription_transport"]["response_media_type"], "text/plain")
        self.assertEqual(response["_subscription_transport"]["output_assembly_source"], "response.output_item.done")
        instance = client(FakeTransport(response))
        self.assertEqual(instance.predict(observation().to_dict()), prediction())
        self.assertEqual(instance.history_snapshot()[1:], [event["item"] for event in events])
        self.assertNotIn("_subscription_transport", json.dumps(instance.history_snapshot()))

    def test_complete_stream_with_deltas_comments_and_crlf(self):
        wire = ": keepalive\r\n\r\n" + sse({"type": "response.output_text.delta", "delta": "ignored"})
        wire += sse(completed(), name="response.completed").replace("\n", "\r\n")
        self.assertEqual(read_completed_response(io.BytesIO(wire.encode())), completed()["response"])

    def test_multiline_json_event(self):
        data = json.dumps(completed(), indent=2)
        wire = "\n".join("data: " + line for line in data.splitlines()) + "\n\n"
        self.assertEqual(read_completed_response(io.StringIO(wire)), completed()["response"])

    def test_partial_output_never_authorizes_execution(self):
        delta = sse({"type": "response.output_text.delta", "delta": json.dumps(prediction())})
        endings = ["", "data: [DONE]\n\n", sse(completed(), ending="\n")]
        for ending in endings:
            with self.subTest(ending=ending[:40]), self.assertRaises(APIError):
                read_completed_response(io.StringIO(delta + ending))

    def test_failure_incomplete_and_usage_limit_after_delta_are_terminal(self):
        for kind in ("response.failed", "response.incomplete", "error", "response.error"):
            error = {"code": "subscription_sharing_usage_limit_exceeded", "message": TOKEN}
            event = {"type": kind, "error": error, "response": {"error": error}}
            wire = sse({"type": "response.output_text.delta", "delta": "partial"}) + sse(event) + sse(completed())
            with self.subTest(kind=kind), self.assertRaises(APIError) as caught:
                read_completed_response(io.StringIO(wire))
            self.assertIn("subscription_sharing_usage_limit_exceeded", str(caught.exception))
            self.assertNotIn(TOKEN, str(caught.exception))

    def test_malformed_or_contradictory_stream_rejected(self):
        wires = ["data: nope\n\n", "data: []\n\n", "data: {}\n\n",
                 'data: {"type":"a","type":"response.completed"}\n\n',
                 sse(completed(), name="response.failed"),
                 sse({"type": "response.completed", "response": {"status": "incomplete", "output": []}}),
                 sse({"type": "response.completed", "response": {"status": "completed", "output": [], "error": {"code": "x"}}}),
                 "HTTP/1.1 200 OK\n\n", "data: \udcff\n\n"]
        for wire in wires:
            with self.subTest(wire=wire[:40]), self.assertRaises(APIError):
                read_completed_response(io.StringIO(wire))
        with self.assertRaises(APIError):
            read_completed_response(io.BytesIO(b"data: \xff\n\n"))

    def test_sse_http_content_type_and_public_endpoint_required(self):
        stream = io.BytesIO(sse(completed()).encode())
        stream.headers = Message()
        stream.headers["Content-Type"] = "text/event-stream; charset=utf-8"
        with patch("gpt_field.chatgpt_plan._open_public", return_value=stream) as opened:
            result = subscription_sse_transport(ENDPOINT, {"store": False, "stream": True}, {}, 30)
        self.assertEqual(result, completed()["response"])
        self.assertEqual(opened.call_args.args[0].full_url, ENDPOINT)
        for endpoint, payload in (("https://chatgpt.com/backend-api/responses", {"store": False, "stream": True}),
                                  (ENDPOINT, {"store": False}),
                                  (ENDPOINT, {"store": False, "stream": True, "max_output_tokens": 12})):
            with self.subTest(endpoint=endpoint, payload=payload), self.assertRaises(ValueError):
                subscription_sse_transport(endpoint, payload, {}, 30)
        stream = io.BytesIO(b"{}")
        stream.status = 200
        stream.headers = Message()
        stream.headers["Content-Type"] = "application/json"
        with patch("gpt_field.chatgpt_plan._open_public", return_value=stream), self.assertRaises(APIError) as caught:
            subscription_sse_transport(ENDPOINT, {"store": False, "stream": True}, {}, 30)
        self.assertEqual(caught.exception.http_status, 200)
        self.assertEqual(caught.exception.error_code, "unexpected_response_content_type")
        self.assertEqual(caught.exception.response_media_type, "application/json")

    def test_http_redirect_never_forwards_bearer(self):
        with self.assertRaises(APIError):
            _NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://elsewhere.invalid")

    def test_plain_text_mime_still_requires_a_real_completed_sse_event(self):
        stream = io.BytesIO(sse(completed()).encode())
        stream.headers = Message()
        stream.headers["Content-Type"] = "text/plain"
        with patch("gpt_field.chatgpt_plan._open_public", return_value=stream):
            result = subscription_sse_transport(ENDPOINT, {"store": False, "stream": True}, {}, 30)
        self.assertEqual(result["output"], completed()["response"]["output"])
        self.assertEqual(result["_subscription_transport"], {
            "response_media_type": "text/plain", "sse_completion_verified": True})
        for wire in (json.dumps(completed()["response"]), "<html>error</html>",
                     sse({"type": "response.output_text.delta", "delta": "partial"})):
            stream = io.BytesIO(wire.encode())
            stream.headers = Message()
            stream.headers["Content-Type"] = "text/plain"
            with self.subTest(wire=wire[:40]), patch("gpt_field.chatgpt_plan._open_public", return_value=stream), self.assertRaises(APIError):
                subscription_sse_transport(ENDPOINT, {"store": False, "stream": True}, {}, 30)

    def test_http_error_retains_only_status_and_machine_code(self):
        body = io.BytesIO(json.dumps({"error": {"code": "invalid_grant", "message": TOKEN}}).encode())
        error = HTTPError(ENDPOINT, 401, "unauthorized", {}, body)
        opener = Mock()
        opener.open.side_effect = error
        with patch("gpt_field.chatgpt_plan.build_opener", return_value=opener), self.assertRaises(APIError) as caught:
            subscription_sse_transport(ENDPOINT, {"store": False, "stream": True}, {}, 30)
        self.assertEqual(caught.exception.http_status, 401)
        self.assertEqual(caught.exception.error_code, "invalid_grant")
        self.assertNotIn(TOKEN, str(caught.exception))


class ClientTests(unittest.TestCase):
    def test_exact_original_prompt_schema_and_model_without_unsupported_fields(self):
        instance = client()
        expected = FieldHistoryAstraClient(api_key="unused").build_payload(observation().to_dict())
        actual = instance.build_payload(observation().to_dict())
        self.assertEqual(actual["instructions"], expected["instructions"])
        self.assertEqual(actual["input"], expected["input"])
        self.assertEqual(actual["text"], expected["text"])
        self.assertEqual(actual["include"], expected["include"])
        self.assertEqual(actual["model"], "gpt-6-astra")
        self.assertEqual(actual["service_tier"], "default")
        self.assertTrue(actual["stream"])
        self.assertFalse(actual["store"])
        self.assertFalse(UNSUPPORTED_FIELDS.intersection(actual))

    def test_environment_api_key_never_used_and_access_provider_invoked_once(self):
        provider = Mock(return_value=TOKEN)
        fake = FakeTransport(envelope(model="gpt-6-astra"))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-environment-never-use"}):
            instance = ChatGPTPlanFieldClient(token_provider=provider, transport=fake)
            instance.preflight()
            instance.build_payload(observation().to_dict())
            provider.assert_not_called()
            instance.predict(observation().to_dict())
        provider.assert_called_once_with()
        self.assertEqual(fake.headers[0]["Authorization"], f"Bearer {TOKEN}")
        self.assertEqual(instance._api_key, "")
        self.assertEqual(instance.budget_snapshot()["api_key_calls"], 0)

    def test_api_keys_absent_token_and_provider_errors_do_not_dispatch(self):
        providers = [lambda: "sk-proj-forbidden", lambda: "", lambda: None,
                     lambda: " whitespace ", Mock(side_effect=RuntimeError("refresh_token_secret"))]
        for provider in providers:
            fake = FakeTransport()
            instance = ChatGPTPlanFieldClient(token_provider=provider, transport=fake)
            with self.subTest(provider=provider), self.assertRaises(MissingCredentials) as caught:
                instance.predict(observation().to_dict())
            self.assertNotIn("refresh_token_secret", str(caught.exception))
            self.assertEqual(fake.calls, [])
            self.assertEqual(instance.budget_snapshot()["api_calls"], 0)

    def test_no_token_price_or_output_reservation_guard(self):
        fake = FakeTransport(envelope(usage={"input_tokens": 2_000_000, "output_tokens": 20_000}))
        instance = client(fake, config=AstraConfig(max_output_tokens=1))
        instance.predict(observation().to_dict())
        snapshot = instance.budget_snapshot()
        self.assertEqual(snapshot["output_tokens"], 20_000)
        self.assertEqual(snapshot["input_tokens"], 2_000_000)
        self.assertEqual(snapshot["billing"], "chatgpt_subscription")
        self.assertIsNone(snapshot["cost_usd_upper_bound"])
        self.assertIsNone(snapshot["subscription_cost_usd"])
        self.assertIsNone(snapshot["output_token_limit_sent"])
        for name in ("max_cost_usd", "input_price_per_million", "cached_input_price_per_million",
                     "cache_write_price_per_million", "output_price_per_million"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                client(config=replace(AstraConfig(), **{name: 1}))

    def test_actual_history_feedback_and_exact_opaque_output_items_retained(self):
        fake = FakeTransport(envelope(), envelope(), envelope())
        instance = client(fake)
        policy = FieldHistoryPolicy(instance)
        obs = observation()
        first_images = (obs.image, obs.wrist_image)
        for step in range(11):
            obs, _, _ = execute(policy, obs, step)
        third = fake.calls[2]
        images = [part["image_url"] for msg in third["input"] if isinstance(msg.get("content"), list)
                  for part in msg["content"] if part["type"] == "input_image"]
        self.assertEqual(len(images), 4)
        self.assertTrue(all(value not in images for value in first_images))
        self.assertEqual(third["input"][1:3], envelope()["output"])
        records = [json.loads(msg["content"][0]["text"]) for msg in third["input"]
                   if msg.get("role") == "user"]
        feedback = [record for record in records if record["type"] == "execution_feedback"]
        self.assertEqual([record["step"] for record in feedback], list(range(10)))
        self.assertEqual(instance.history_stats()["feedback_records"], 11)
        self.assertEqual(instance.budget_snapshot()["subscription_inference_calls"], 3)

    def test_schema_error_never_retries_or_commits(self):
        fake = FakeTransport(envelope(), envelope({}), envelope())
        instance = client(fake)
        instance.predict(observation().to_dict())
        before = instance.history_snapshot()
        with self.assertRaises(SchemaError):
            instance.predict(observation(1).to_dict())
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(instance.history_snapshot(), before)

    def test_wrong_returned_model_is_rejected_without_fallback(self):
        fake = FakeTransport(envelope(model="gpt-other"))
        instance = client(fake)
        with self.assertRaises(APIError):
            instance.predict(observation().to_dict())
        self.assertEqual(fake.calls[0]["model"], "gpt-6-astra")
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(instance.history_snapshot(), [])

    def test_call_guard_is_before_auth_and_after_one_attempt(self):
        provider = Mock(return_value=TOKEN)
        fake = FakeTransport(envelope())
        instance = ChatGPTPlanFieldClient(AstraConfig(max_api_calls=1), token_provider=provider, transport=fake)
        instance.predict(observation().to_dict())
        before = instance.history_snapshot()
        with self.assertRaises(BudgetExceeded):
            instance.predict(observation(1).to_dict())
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(instance.history_snapshot(), before)

    def test_transport_failure_and_interrupt_keep_history_and_usage_uncertainty(self):
        for error in (APIError("failed"), RuntimeError("sensitive-network-detail"), KeyboardInterrupt()):
            instance = client(FakeTransport(error))
            with self.subTest(error=error), self.assertRaises((APIError, KeyboardInterrupt)) as caught:
                instance.predict(observation().to_dict())
            self.assertNotIn("sensitive-network-detail", str(caught.exception))
            self.assertEqual(instance.history_snapshot(), [])
            self.assertEqual(instance.budget_snapshot()["api_calls"], 1)
            self.assertEqual(instance.budget_snapshot()["unknown_usage_calls"], 1)

    def test_safe_failure_codes_survive_client_and_are_logged(self):
        wire = sse({"type": "response.failed", "response": {
            "error": {"code": "subscription_sharing_usage_unavailable", "message": "sensitive-server-message"}}})
        try:
            read_completed_response(io.StringIO(wire))
        except APIError as caught:
            error = caught
        with tempfile.TemporaryDirectory() as directory:
            instance = client(FakeTransport(error), log_dir=directory)
            with self.assertRaises(APIError) as caught:
                instance.predict(observation().to_dict())
            self.assertEqual(caught.exception.error_code, "subscription_sharing_usage_unavailable")
            result_path = next(Path(directory).glob("*.result.json"))
            result = json.loads(result_path.read_text())
            self.assertEqual(result["error_code"], "subscription_sharing_usage_unavailable")
            self.assertNotIn("sensitive-server-message", result_path.read_text())

    def test_logs_redact_all_used_tokens_and_exclude_arbitrary_exception_text(self):
        other_token = "oauth-access-fixture-two"
        first = envelope()
        first["debug_echo"] = TOKEN
        second = envelope()
        second["debug_echo"] = [TOKEN, other_token]
        provider = Mock(side_effect=[TOKEN, other_token, other_token])
        with tempfile.TemporaryDirectory() as directory:
            instance = ChatGPTPlanFieldClient(token_provider=provider, log_dir=directory,
                transport=FakeTransport(first, second, RuntimeError("unseen-refresh-secret")))
            instance.predict(observation().to_dict())
            instance.predict(observation(1).to_dict())
            with self.assertRaises(APIError):
                instance.predict(observation(2).to_dict())
            logs = "".join(path.read_text() for path in Path(directory).glob("*.json"))
            for secret in (TOKEN, other_token, "unseen-refresh-secret"):
                self.assertNotIn(secret, logs)
            request = json.loads(next(Path(directory).glob("*.request.json")).read_text())
            self.assertEqual(request["billing"], "chatgpt_subscription")
            self.assertEqual(request["prompt_version"], "astra-field-history-v1")
            self.assertEqual(request["payload"]["model"], "gpt-6-astra")

    def test_oracle_or_wrong_grid_inputs_rejected_before_auth(self):
        provider = Mock(return_value=TOKEN)
        instance = ChatGPTPlanFieldClient(token_provider=provider)
        with self.assertRaises(ValueError):
            instance.predict({**observation().to_dict(), "reward": 1})
        with self.assertRaises(ValueError):
            instance.predict(observation().to_dict(), GridSpec(size=3))
        provider.assert_not_called()


class CatalogTests(unittest.TestCase):
    def test_catalog_uses_same_explicit_oauth_and_sanitizes_fields(self):
        wire = {"models": [{"slug": "gpt-6-astra", "display_name": "Astra", "visibility": "list",
                            "private_ignored": TOKEN}, {"slug": "gpt-other", "visibility": "hidden"}]}
        with patch("gpt_field.chatgpt_plan._open_public", return_value=io.BytesIO(json.dumps(wire).encode())) as opened:
            models = list_plan_models(lambda: TOKEN)
        self.assertEqual(models, [{"slug": "gpt-6-astra", "display_name": "Astra", "visibility": "list"},
                                  {"slug": "gpt-other", "visibility": "hidden"}])
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, MODELS_ENDPOINT)
        self.assertEqual(request.get_header("Authorization"), f"Bearer {TOKEN}")
        self.assertEqual(request.get_method(), "GET")

    def test_malformed_catalog_does_not_guess_models(self):
        for wire in ({"data": []}, {"models": [None]}, {"models": [{"slug": TOKEN}]}):
            with self.subTest(wire=wire), patch("gpt_field.chatgpt_plan._open_public", return_value=io.BytesIO(json.dumps(wire).encode())), self.assertRaises(APIError):
                list_plan_models(lambda: TOKEN)


if __name__ == "__main__":
    unittest.main()
