"""Offline request bounds, safe telemetry, and review/CLI binding; no provider."""
from email.message import Message
import json
import signal
import socket
import threading
import time

import pytest

from gpt_field import agent_speedup_v2_user_check_transport as transport
from gpt_field.agent_speedup_v2_client import TRANSIENT_ERROR_CODES
from gpt_field.agent_speedup_v2_user_check import ReviewError
from gpt_field.agent_speedup_v2_user_check_agent_client import AgentFirstV2Client
from gpt_field.agent_speedup_v2_user_check_client import PlanLock, UserCheckV2Client
from gpt_field.astra import APIError, ENDPOINT
from gpt_field.ordered_transport import ordered_dumps, ordered_sse_transport
from agent_speedup_v2_fakes import FakeV2Sim
from test_agent_speedup_v2_user_check_runner import PLAN, TASK, runner, settings


SECRET = "SENSITIVE_SENTINEL_DO_NOT_LOG"
PAYLOAD = {"model": "gpt-6-astra", "store": False, "stream": True,
           "reasoning": {"effort": "high"}, "input": SECRET}


def completed_lines(value=None):
    final = {"type": "response.completed", "response": {
        "id": "resp_fixture", "status": "completed", "model": "gpt-6-astra",
        "reasoning": {"effort": "high"}, "output": [
            {"type": "reasoning", "id": "rs_fixture", "encrypted_content": SECRET},
            {"type": "message", "id": "msg_fixture", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": json.dumps(PLAN if value is None else value)}]}]}}
    return [b": heartbeat\r\n", b"\r\n", b'data: {"type":"heartbeat"}\n', b"\n",
            ("data: " + json.dumps(final) + "\n").encode(), b"\n"]


def delta_lines(text, *, output_index=1, content_index=0):
    event = {"type": "response.output_text.delta", "output_index": output_index,
             "content_index": content_index, "item_id": "msg_fixture", "delta": text}
    return [("data: " + json.dumps(event) + "\n").encode(), b"\n"]


def plan_payload(name="agent_speedup_v2_plan"):
    return {**PAYLOAD, "text": {"format": {"type": "json_schema", "name": name,
                                           "strict": True, "schema": {"type": "object"}}}}


class Response:
    def __init__(self, lines=None, sock=None):
        self.socket, self.closed = sock, False
        self.lines = sock.makefile("rb") if sock is not None else iter(lines)
        self.status, self.headers = 200, Message()
        self.headers["Content-Type"] = "text/plain"
        self.headers["x-request-id"] = "req_fixture"
        self.headers["Authorization"] = SECRET
    def __iter__(self):
        return iter(self.lines)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.closed = True
        if self.socket is not None:
            self.lines.close()
            self.socket.close()


def observed(tmp_path, deadline=.1):
    return transport.UserCheckSubscriptionTransport(
        request_deadline_s=deadline, stream_progress=True,
        request_path=lambda: tmp_path / "000001-fixture.request.json")


def report(tmp_path):
    return json.loads((tmp_path / "000001-fixture.stream.json").read_text())


def test_completion_before_eof_preserves_wire_and_never_logs_content(tmp_path, monkeypatch):
    def lines():
        yield from completed_lines()
        raise AssertionError("Completion must return without waiting for EOF")
    response = Response(lines())
    def opener(request, timeout):
        assert request.data == ordered_dumps(PAYLOAD).encode()
        assert request.get_header("Authorization") == "Bearer " + SECRET
        return response
    monkeypatch.setattr(transport.protocol, "_open_public", opener)
    prior = signal.getsignal(signal.SIGALRM)
    result = observed(tmp_path)(ENDPOINT, PAYLOAD, {"Authorization": "Bearer " + SECRET}, 5.)
    stats = report(tmp_path)
    assert result["status"] == "completed" and stats["parser_verified_completion"]
    assert stats["phase"] == "completed" and not stats["stream_exhausted"]
    assert stats["comment_lines"] == 1 and stats["heartbeat_events"] == 1
    assert stats["event_types"] == {"heartbeat": 1, "response.completed": 1}
    assert stats["bytes_read"] == sum(map(len, completed_lines()))
    assert stats["provider_request_id"] == "req_fixture"
    assert SECRET not in (tmp_path / "000001-fixture.stream.json").read_text()
    assert response.closed and signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    assert signal.getsignal(signal.SIGALRM) is prior


@pytest.mark.parametrize("heartbeat", [False, True])
def test_silent_and_heartbeat_socket_streams_have_hard_deadline(tmp_path, monkeypatch, heartbeat):
    reader, writer = socket.socketpair()
    response = Response(sock=reader)
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: response)
    stop = threading.Event()
    def send():
        while not stop.is_set():
            try:
                writer.sendall(b": heartbeat\n\n")
            except OSError:
                return
            stop.wait(.003)
    thread = threading.Thread(target=send, daemon=True) if heartbeat else None
    if thread:
        thread.start()
    prior, start = signal.getsignal(signal.SIGALRM), time.monotonic()
    try:
        with pytest.raises(APIError) as failure:
            observed(tmp_path, .07)(ENDPOINT, PAYLOAD, {}, 5.)
        assert failure.value.error_code == "request_deadline_exceeded"
        assert time.monotonic() - start < .6
    finally:
        stop.set()
        writer.close()
        if thread:
            thread.join(timeout=1)
    stats = report(tmp_path)
    assert stats["phase"] == "failed" and not stats["parser_verified_completion"]
    assert stats["error_code"] == "request_deadline_exceeded"
    assert bool(stats["comment_lines"]) is heartbeat
    assert response.closed and signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    assert signal.getsignal(signal.SIGALRM) is prior


def test_deadline_covers_http_open_and_preserves_active_timer(tmp_path, monkeypatch):
    calls = []
    def opener(*args):
        calls.append(True)
        time.sleep(1)
    monkeypatch.setattr(transport.protocol, "_open_public", opener)
    with pytest.raises(APIError) as failure:
        observed(tmp_path, .03)(ENDPOINT, PAYLOAD, {}, 5.)
    assert failure.value.error_code == "request_deadline_exceeded"
    assert "headers_received_s" not in report(tmp_path)
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)
    prior = signal.getsignal(signal.SIGALRM)
    signal.setitimer(signal.ITIMER_REAL, 5.)
    try:
        with pytest.raises(APIError) as failure:
            observed(tmp_path / "prior_timer", .03)(ENDPOINT, PAYLOAD, {}, 5.)
        assert failure.value.error_code == "request_deadline_unavailable"
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 4.
        assert signal.getsignal(signal.SIGALRM) is prior and len(calls) == 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)


def test_missing_completion_cannot_succeed_and_closes_response(tmp_path, monkeypatch):
    response = Response([b'data: {"type":"response.in_progress"}\n', b"\n"])
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: response)
    with pytest.raises(APIError, match="without response.completed"):
        observed(tmp_path)(ENDPOINT, PAYLOAD, {}, 5.)
    assert report(tmp_path)["error_code"] == "stream_missing_completion"
    assert not report(tmp_path)["parser_verified_completion"] and response.closed


def test_malformed_nonhashable_event_type_reaches_unchanged_parser(tmp_path, monkeypatch):
    lines = [('data: ' + json.dumps({"type": [SECRET]}) + '\n').encode(), b'\n']
    with pytest.raises(APIError) as expected:
        transport.protocol.read_completed_response(lines)
    response = Response(lines)
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: response)
    with pytest.raises(APIError) as actual:
        observed(tmp_path)(ENDPOINT, PAYLOAD, {}, 5.)
    assert str(actual.value) == str(expected.value)
    assert response.closed and SECRET not in (tmp_path / "000001-fixture.stream.json").read_text()


def test_unknown_provider_code_and_exception_type_are_not_logged(tmp_path, monkeypatch):
    error = type(SECRET, (APIError,), {})(SECRET)
    error.error_code = SECRET.lower()
    def opener(*args):
        raise error
    monkeypatch.setattr(transport.protocol, "_open_public", opener)
    with pytest.raises(APIError):
        observed(tmp_path)(ENDPOINT, PAYLOAD, {}, 5.)
    stats = report(tmp_path)
    assert stats["error_type"] == "OtherError" and stats["error_code"] == "other_error"
    assert SECRET.lower() not in (tmp_path / "000001-fixture.stream.json").read_text().lower()


def test_progress_is_atomic_and_updates_while_socket_is_silent(tmp_path, monkeypatch):
    reader, writer = socket.socketpair()
    response = Response(sock=reader)
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: response)
    observed_progress = []
    def inspect_waiting():
        time.sleep(1.05)
        observed_progress.append(report(tmp_path))
    observer = threading.Thread(target=inspect_waiting, daemon=True)
    observer.start()
    try:
        with pytest.raises(APIError) as failure:
            observed(tmp_path, 1.2)(ENDPOINT, PAYLOAD, {}, 5.)
        assert failure.value.error_code == "request_deadline_exceeded"
    finally:
        writer.close()
        observer.join(timeout=1)
    assert observed_progress[0]["phase"] == "reading_stream"
    assert observed_progress[0]["elapsed_s"] >= 1.
    assert observed_progress[0]["lines_read"] == 0
    assert report(tmp_path)["phase"] == "failed"


@pytest.mark.parametrize("executive", [False, True])
def test_deadline_is_not_retried_by_either_real_client(tmp_path, monkeypatch, executive):
    calls = []
    def opener(*args):
        calls.append(True)
        time.sleep(1)
    monkeypatch.setattr(transport.protocol, "_open_public", opener)
    kwargs = dict(task_prompt=TASK.prompt, token_provider=lambda: "offline", log_dir=tmp_path)
    client = (AgentFirstV2Client(**kwargs, approved_plan=PLAN, lock=PlanLock(PLAN, task_prompt=TASK.prompt))
              if executive else UserCheckV2Client(**kwargs))
    transport.configure_client_transport(client, request_deadline_s=.03, stream_progress=True)
    with pytest.raises(APIError) as failure:
        client.predict({"fields": {}, "images": []}, mode="agent" if executive else "plan")
    assert failure.value.error_code == "request_deadline_exceeded"
    assert "request_deadline_exceeded" not in TRANSIENT_ERROR_CODES
    assert len(calls) == 1 and client.budget_snapshot()["api_calls"] == 1
    requests = list(tmp_path.glob("*.request.json"))
    assert len(requests) == 1 and len(list(tmp_path.glob("*.stream.json"))) == 1
    assert requests[0].with_name(requests[0].name.replace(".request.json", ".stream.json")).exists()
    assert not list(tmp_path.glob("*.transient_retry.json"))


def prepare_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: Response(completed_lines()))
    config, sim_config, monitor = settings()
    fake = FakeV2Sim(prompt=TASK.prompt)
    store = runner.prepare_review(
        tmp_path / "review", config=config, sim_config=sim_config, task_key=TASK.key,
        episode_index=0, monitor_options=monitor, video=False, token_provider=lambda: "offline",
        sim_factory=lambda _: fake, request_deadline_s=1., stream_progress=True)
    return store, fake, config, sim_config, monitor


def test_runner_draft_and_executive_share_selected_transport(tmp_path, monkeypatch):
    store, fake, config, sim_config, monitor = prepare_stream(tmp_path, monkeypatch)
    assert fake.actions == [] and len(list((store.path / "requests").glob("*.stream.json"))) == 1
    execution = store.load()["preparation"]["execution"]
    assert execution["request_deadline_s"] == 1. and execution["stream_progress"]
    assert execution["subscription_transport"]["deadline_scope"].endswith("after authentication")
    store.approve(expected_revision=0, reviewed_steps=[0])
    def execute(output, **kwargs):
        client = kwargs["client_factory"](config, task_prompt=TASK.prompt, token_provider=lambda: "offline")
        assert isinstance(client.transport, transport.UserCheckSubscriptionTransport)
        assert client.transport.settings == execution["subscription_transport"]
        return {"fake": True}
    def normalize(output, summary, review, lock, **kwargs):
        assert kwargs["client_metadata"]["subscription_transport"] == execution["subscription_transport"]
        return summary
    monkeypatch.setattr(runner.base, "execute", execute)
    monkeypatch.setattr(runner, "_normalize_artifacts", normalize)
    result = runner.execute_reviewed(
        tmp_path / "run", review_dir=store.path, config=config, sim_config=sim_config,
        task_key=TASK.key, episode_index=0, monitor_options=monitor, video=False,
        token_provider=lambda: "offline", request_deadline_s=1., stream_progress=True)
    assert result == {"fake": True}


def test_settings_change_invalidates_approval_and_run_cli_restores_them(tmp_path, monkeypatch):
    import gpt_field.chatgpt_auth as auth
    store, _, config, sim_config, monitor = prepare_stream(tmp_path, monkeypatch)
    store.approve(expected_revision=0, reviewed_steps=[0])
    with pytest.raises(ReviewError, match="Runtime code changed"):
        runner.execute_reviewed(
            tmp_path / "run", review_dir=store.path, config=config, sim_config=sim_config,
            task_key=TASK.key, episode_index=0, monitor_options=monitor, video=False,
            token_provider=lambda: "offline", request_deadline_s=2., stream_progress=True)
    assert store.load()["status"] == "approved" and not (tmp_path / "run").exists()
    monkeypatch.setattr(runner, "_live_gate", lambda *args: None)
    class NoCredentials:
        def __init__(self, **kwargs):
            pass
        def access_token(self):
            pytest.fail("CLI restoration must not read credentials")
    monkeypatch.setattr(auth, "SubscriptionCredentials", NoCredentials)
    calls = []
    def execute(output, **kwargs):
        calls.append(kwargs)
        return {"status": "fake", "output": str(output), "evaluation_completed": True,
                "user_check": {"review_id": store.load()["id"]}}
    monkeypatch.setattr(runner, "execute_reviewed", execute)
    assert runner.main(["run", "--review-dir", str(store.path), "--output", str(tmp_path / "run")]) == 0
    assert calls[0]["request_deadline_s"] == 1. and calls[0]["stream_progress"]


def test_default_unchanged_and_preflight_advertises_opt_in_settings():
    client = UserCheckV2Client(task_prompt=TASK.prompt)
    assert not transport.configure_client_transport(client)["enabled"]
    assert client.transport is ordered_sse_transport
    config, sim_config, monitor = settings()
    default = runner.runtime_inventory(config, sim_config, monitor_options=monitor)
    result = runner.preflight(config, sim_config, task_key=TASK.key, episode_index=0,
                              monitor_options=monitor, request_deadline_s=180., stream_progress=True)
    assert result["subscription_transport"]["request_deadline_s"] == 180.
    assert "--request-deadline-s 180.0" in result["live_command"] and "--stream-progress" in result["live_command"]
    assert result["live_command_first_run"] == result["live_command"]
    assert not result["simulator_started"] and result["api_calls"] == 0
    assert result["runtime_sha256"]["@configuration"] != default["@configuration"]


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_invalid_deadline_rejected_before_preparation(tmp_path, value):
    config, sim_config, monitor = settings()
    with pytest.raises(ValueError, match="finite and positive"):
        runner.prepare_review(tmp_path / "review", config=config, sim_config=sim_config,
                              task_key=TASK.key, episode_index=0, monitor_options=monitor,
                              request_deadline_s=value)
    assert not (tmp_path / "review").exists()


def test_visible_delta_counts_shape_and_unicode_are_content_free_and_reset(tmp_path, monkeypatch):
    value = {"text": SECRET + ' café \\" quote', "values": [1, 2]}
    visible = json.dumps(value, ensure_ascii=False, indent=2)
    chunks = [visible[start:start + 3] for start in range(0, len(visible), 3)]
    lines = [line for chunk in chunks for line in delta_lines(chunk)] + completed_lines(value)
    response = Response(lines)
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: response)
    current = [tmp_path / "000001-fixture.request.json"]
    client_transport = transport.UserCheckSubscriptionTransport(
        request_deadline_s=1., stream_progress=True, request_path=lambda: current[0])
    baseline = transport.protocol.read_completed_response(lines)
    result = client_transport(ENDPOINT, plan_payload(), {}, 5.)
    assert result["output"] == baseline["output"]
    stats = report(tmp_path)
    assert stats["visible_delta_events"] == len(chunks)
    assert stats["visible_characters"] == len(visible)
    assert stats["visible_utf8_bytes"] == len(visible.encode("utf-8"))
    assert stats["visible_whitespace_characters"] == sum(char in " \t\r\n" for char in visible)
    assert stats["longest_json_whitespace_run"] == 5
    assert stats["first_visible_output_s"] <= stats["last_visible_output_s"] <= stats["completion_s"]
    shape = stats["visible_output_parts"]["1:0"]
    assert shape["root_closed_candidate"] and shape["container_depth"] == 0
    assert not shape["inside_string"] and not shape["lexical_anomaly"]
    assert SECRET not in json.dumps(stats)
    current[0] = tmp_path / "000002-fixture.request.json"
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: Response(completed_lines(value)))
    client_transport(ENDPOINT, plan_payload(), {}, 5.)
    second = json.loads((tmp_path / "000002-fixture.stream.json").read_text())
    assert second["visible_characters"] == 0 and not second["visible_output_parts"]
    assert "first_visible_output_s" not in second


def test_lexical_state_is_bounded_and_mismatched_brackets_are_not_closed_candidates():
    progress = transport._Progress(None, transport.transport_settings(), time.monotonic)
    output = transport._VisibleOutput(progress, guard=False)
    for index in range(transport.MAX_TEXT_PARTS + 2):
        output.delta({"delta": "{", "output_index": index, "content_index": 0})
    assert len(output.parts) == len(progress.data["visible_output_parts"]) == transport.MAX_TEXT_PARTS
    assert progress.data["visible_untracked_delta_events"] == 2
    output.delta({"delta": "]", "output_index": 0, "content_index": 0})
    assert progress.data["visible_output_parts"]["0:0"]["lexical_anomaly"]
    assert not progress.data["visible_output_parts"]["0:0"]["root_closed_candidate"]
    output.delta({"delta": "[" * 200 + "]" * 200 + "}", "output_index": 1, "content_index": 0})
    shape = progress.data["visible_output_parts"]["1:0"]
    assert shape["depth_tracking_limited"] and not shape["root_closed_candidate"]
    assert len(output.parts["1:0"].stack) <= transport.MAX_LEXICAL_DEPTH


def test_long_whitespace_inside_strings_passes_plan_guard(tmp_path, monkeypatch):
    value = {"text": SECRET + " " * (transport.MAX_PLAN_JSON_WHITESPACE_RUN + 100)}
    visible = json.dumps(value)
    lines = delta_lines(visible[:30]) + delta_lines(visible[30:]) + completed_lines(value)
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: Response(lines))
    result = observed(tmp_path, 1.)(ENDPOINT, plan_payload(), {}, 5.)
    stats = report(tmp_path)
    assert result["status"] == "completed" and stats["parser_verified_completion"]
    assert stats["longest_visible_whitespace_run"] > transport.MAX_PLAN_JSON_WHITESPACE_RUN
    assert stats["longest_json_whitespace_run"] == 1
    assert SECRET not in json.dumps(stats)


@pytest.mark.parametrize("mode", ["plan", "replan"])
def test_cross_delta_structural_whitespace_guard_is_not_retried_by_real_planner(tmp_path, monkeypatch, mode):
    limit = transport.MAX_PLAN_JSON_WHITESPACE_RUN
    lines = delta_lines("{") + delta_lines(" " * (limit // 2)) + delta_lines("\n" * (limit // 2 + 1))
    calls = []
    response = Response(lines)
    def opener(*args):
        calls.append(True)
        return response
    monkeypatch.setattr(transport.protocol, "_open_public", opener)
    client = UserCheckV2Client(task_prompt=TASK.prompt, token_provider=lambda: "offline", log_dir=tmp_path)
    transport.configure_client_transport(client, request_deadline_s=1., stream_progress=True)
    with pytest.raises(APIError) as failure:
        client.predict({"fields": {}, "images": []}, mode=mode)
    assert failure.value.error_code == "plan_output_whitespace_limit"
    assert failure.value.error_code not in TRANSIENT_ERROR_CODES
    assert len(calls) == client.budget_snapshot()["api_calls"] == 1 and response.closed
    stats = json.loads(next(tmp_path.glob("*.stream.json")).read_text())
    assert stats["longest_json_whitespace_run"] == limit + 1
    assert stats["plan_output_guard_applied"] and not stats["parser_verified_completion"]
    assert not list(tmp_path.glob("*.transient_retry.json"))
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_visible_character_guard_and_executive_scope(tmp_path, monkeypatch):
    lines = delta_lines("x" * (transport.MAX_PLAN_VISIBLE_CHARACTERS + 1))
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: Response(lines))
    with pytest.raises(APIError) as failure:
        observed(tmp_path / "plan", 1.)(ENDPOINT, plan_payload(), {}, 5.)
    assert failure.value.error_code == "plan_visible_output_limit"
    assert failure.value.error_code not in TRANSIENT_ERROR_CODES
    # The executive keeps its previous behavior; the new early guards are plan-only.
    executive_lines = delta_lines(" " * (transport.MAX_PLAN_JSON_WHITESPACE_RUN + 1)) + completed_lines()
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: Response(executive_lines))
    result = observed(tmp_path / "executive", 1.)(ENDPOINT, plan_payload("user_check_agent_first"), {}, 5.)
    assert result["status"] == "completed"
    assert not report(tmp_path / "executive")["plan_output_guard_applied"]


def test_completed_only_text_is_guarded_without_double_counting_deltas(tmp_path, monkeypatch):
    lines = completed_lines()
    final = json.loads(lines[-2].decode()[len("data: "):])
    final["response"]["output"][-1]["content"][0]["text"] = "{" + " " * (transport.MAX_PLAN_JSON_WHITESPACE_RUN + 1) + '"ok":true}'
    lines[-2] = ("data: " + json.dumps(final) + "\n").encode()
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: Response(lines))
    with pytest.raises(APIError) as failure:
        observed(tmp_path, 1.)(ENDPOINT, plan_payload(), {}, 5.)
    assert failure.value.error_code == "plan_output_whitespace_limit"
    assert report(tmp_path)["visible_delta_events"] == 0


def test_malformed_delta_value_does_not_change_completed_parser_or_leak(tmp_path, monkeypatch):
    lines = delta_lines([SECRET]) + completed_lines()
    expected = transport.protocol.read_completed_response(lines)
    monkeypatch.setattr(transport.protocol, "_open_public", lambda *args: Response(lines))
    result = observed(tmp_path, 1.)(ENDPOINT, plan_payload(), {}, 5.)
    assert result["output"] == expected["output"]
    assert report(tmp_path)["visible_invalid_delta_events"] == 1
    assert SECRET not in json.dumps(report(tmp_path))
