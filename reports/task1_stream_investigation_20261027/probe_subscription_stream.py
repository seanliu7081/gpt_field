"""One explicitly enabled, bounded subscription probe; metadata only, no simulator.

Run --self-test for offline fixtures. --live --output PATH makes exactly one
tiny model request through the production ordered serializer, urllib opener,
and strict SSE reader. No transport retries and no raw stream/content logging.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import sys
import tempfile
import threading
import time
from urllib.request import Request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gpt_field import chatgpt_plan as plan
from gpt_field.astra import APIError, ENDPOINT
from gpt_field.chatgpt_auth import SubscriptionCredentials
from gpt_field.ordered_transport import ordered_dumps

DEADLINE_S = 60.0
EVENT_TYPES = frozenset({
    "response.created", "response.queued", "response.in_progress",
    "response.completed", "response.failed", "response.incomplete", "response.error", "error",
    "response.output_item.added", "response.output_item.done",
    "response.content_part.added", "response.content_part.done",
    "response.output_text.delta", "response.output_text.done",
    "response.refusal.delta", "response.refusal.done",
    "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
    "response.reasoning_text.delta", "response.reasoning_text.done",
    "ping", "heartbeat",
})


class ProbeDeadline(BaseException):
    """BaseException keeps credential exception sanitizers from masking the deadline."""


class Report:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.open("x").close()
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.data = {
            "version": "isolated-subscription-stream-probe-1",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "phase": "created", "success": False, "hard_deadline_s": DEADLINE_S,
            "model": "gpt-6-astra", "reasoning_effort": "high",
            "model_requests_attempted": 0, "transport_retries": 0,
            "policy_actions": 0, "simulator_started": False,
            "bytes_read": 0, "lines_read": 0, "comment_lines": 0,
            "heartbeat_events": 0, "events_seen": 0, "event_types": {},
            "stream_exhausted": False, "parser_verified_completion": False,
        }
        self.flush()

    def elapsed(self):
        return round(time.monotonic() - self.started, 6)

    def flush(self):
        with self.lock:
            self.data["elapsed_s"] = self.elapsed()
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(json.dumps(self.data, indent=2) + "\n")
            os.replace(temporary, self.path)

    def update(self, **values):
        with self.lock:
            self.data.update(values)
            self.flush()

    def start_progress(self):
        def progress():
            while not self.stop.wait(1):
                self.flush()
        self.thread = threading.Thread(target=progress, daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        self.flush()


def safe_id(value, prefix):
    return value if isinstance(value, str) and re.fullmatch(prefix + r"[A-Za-z0-9_-]{1,150}", value) else None


class ObservedLines:
    """Observe counters while passing the original line objects through unchanged."""
    def __init__(self, lines, report):
        self.lines, self.report = iter(lines), report
        self.event_name, self.data_lines = None, []
        self.first = True

    def __iter__(self):
        return self

    def __next__(self):
        try:
            raw = next(self.lines)
        except StopIteration:
            self.report.update(stream_exhausted=True)
            raise
        with self.report.lock:
            stats = self.report.data
            stats.setdefault("first_line_s", self.report.elapsed())
            stats["last_line_s"] = self.report.elapsed()
            stats["lines_read"] += 1
            stats["bytes_read"] += len(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
            try:
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if self.first:
                    line = line.removeprefix("\ufeff")
                    self.first = False
                line = line.rstrip("\r\n")
                if not line:
                    if self.data_lines:
                        self.observe_event("\n".join(self.data_lines))
                    self.event_name, self.data_lines = None, []
                elif line.startswith(":"):
                    stats["comment_lines"] += 1
                else:
                    field, _, value = line.partition(":")
                    value = value[1:] if value.startswith(" ") else value
                    if field == "event":
                        self.event_name = value
                    elif field == "data":
                        self.data_lines.append(value)
            except (UnicodeError, ValueError, TypeError):
                # Production parser owns all validation; telemetry cannot repair data.
                stats["telemetry_decode_errors"] = stats.get("telemetry_decode_errors", 0) + 1
            self.report.flush()
        return raw

    def observe_event(self, raw):
        stats = self.report.data
        stats["events_seen"] += 1
        stats.setdefault("first_event_s", self.report.elapsed())
        stats["last_event_s"] = self.report.elapsed()
        try:
            event = json.loads(raw)
            kind = event.get("type") if isinstance(event, dict) else None
        except (ValueError, TypeError):
            kind = "done_marker" if raw == "[DONE]" else "unparsed"
        label = kind if kind in EVENT_TYPES or kind in ("done_marker", "unparsed") else "other"
        stats["event_types"][label] = stats["event_types"].get(label, 0) + 1
        if kind in ("ping", "heartbeat") or self.event_name in ("ping", "heartbeat"):
            stats["heartbeat_events"] += 1


def read_stream(lines, report):
    report.update(phase="reading_stream")
    response = plan.read_completed_response(ObservedLines(lines, report))
    effort = (response.get("reasoning") or {}).get("effort")
    report.update(
        parser_verified_completion=True, completion_s=report.elapsed(),
        response_id=safe_id(response.get("id"), "resp_"),
        response_status="completed" if response.get("status") == "completed" else "other",
        response_model="gpt-6-astra" if response.get("model") == "gpt-6-astra" else "other",
        response_effort=effort if effort in ("high", "xhigh", "medium", "low") else None,
    )
    if response.get("model") != "gpt-6-astra" or effort != "high":
        error = APIError("Probe metadata mismatch")
        error.error_code = "probe_response_metadata_mismatch"
        raise error
    texts = [part.get("text", "") for item in response["output"] if item.get("type") == "message"
             for part in item.get("content", []) if part.get("type") == "output_text"]
    try:
        value = json.loads("".join(texts))
    except (ValueError, TypeError):
        value = None
    if not isinstance(value, dict) or set(value) != {"ok"} or value["ok"] is not True:
        error = APIError("Probe output check failed")
        error.error_code = "probe_output_check_failed"
        raise error
    report.update(phase="completed", success=True, output_check_passed=True)


def record_error(report, error):
    code = "probe_wall_deadline" if isinstance(error, ProbeDeadline) else plan._failure_metadata(error).get("error_code")
    report.update(phase="failed", success=False, error_type=type(error).__name__, error_code=code)


def run_live(output):
    report = Report(output)
    previous_handler = signal.getsignal(signal.SIGALRM)
    def deadline(_signal, _frame):
        raise ProbeDeadline()
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, DEADLINE_S)
    try:
        report.start_progress()
        report.update(phase="auth", auth_started_s=report.elapsed())
        token = plan._access_token(SubscriptionCredentials().access_token)
        report.update(phase="opening_http", auth_completed_s=report.elapsed())
        payload = {
            "model": "gpt-6-astra", "service_tier": "default", "store": False,
            "instructions": "Return ok true.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": 'Return {"ok": true}.'}]}],
            "text": {"format": {"type": "json_schema", "name": "subscription_stream_probe", "strict": True,
                "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                           "required": ["ok"], "additionalProperties": False}}},
            "stream": True, "reasoning": {"effort": "high"}, "include": ["reasoning.encrypted_content"],
        }
        request = Request(ENDPOINT, data=ordered_dumps(payload).encode("utf-8"), method="POST", headers={
            "Authorization": "Bearer " + token, "Content-Type": "application/json", "Accept": "text/event-stream"})
        report.update(model_requests_attempted=1, http_open_started_s=report.elapsed())
        with plan._open_public(request, DEADLINE_S) as response:
            media_type = response.headers.get_content_type()
            status = getattr(response, "status", None)
            request_id = safe_id(response.headers.get("x-request-id"), "req_")
            if request_id is None:
                request_id = safe_id(response.headers.get("request-id"), "req_")
            report.update(headers_received_s=report.elapsed(), http_status=status if type(status) is int else None,
                          response_media_type=media_type if media_type in ("text/plain", "text/event-stream") else "other",
                          request_id=request_id)
            if media_type not in ("text/plain", "text/event-stream"):
                error = APIError("Unexpected content type")
                error.error_code = "unexpected_response_content_type"
                raise error
            read_stream(response, report)
    except BaseException as error:
        record_error(report, error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        report.close()
    return 0 if report.data["success"] else 1


def self_test():
    from email.message import Message
    from unittest.mock import patch
    sentinel = "SENSITIVE_SENTINEL_CONTENT_NEVER_LOG"
    response = {"id": "resp_fixture", "status": "completed", "model": "gpt-6-astra",
                "reasoning": {"effort": "high"}, "output": [
        {"id": "rs_fixture", "type": "reasoning", "encrypted_content": sentinel},
        {"id": "msg_fixture", "type": "message", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": '{"ok":true}'}]}]}
    with tempfile.TemporaryDirectory(prefix="subscription-probe-selftest-") as tmp:
        report = Report(Path(tmp) / "completed.json")
        def complete_before_eof():
            yield b": heartbeat\r\n"
            yield b"\r\n"
            yield b'event: heartbeat\n'
            yield b'data: {"type":"heartbeat"}\n'
            yield b'\n'
            yield b"event: response.completed\n"
            yield b'data: {"type":"response.completed",\n'
            yield ("data: \"response\":" + json.dumps(response) + "}\n").encode()
            yield b"\n"
            raise AssertionError("Parser must return at completion without waiting for EOF")
        read_stream(complete_before_eof(), report)
        report.close()
        assert report.data["success"] and report.data["parser_verified_completion"]
        assert not report.data["stream_exhausted"]
        assert report.data["comment_lines"] == 1 and report.data["heartbeat_events"] == 1
        assert report.data["events_seen"] == 2
        assert sentinel not in report.path.read_text()
        incomplete = Report(Path(tmp) / "incomplete.json")
        try:
            read_stream([b": heartbeat\n", b"\n", b'data: {"type":"response.in_progress"}\n', b"\n"], incomplete)
        except APIError as error:
            record_error(incomplete, error)
        finally:
            incomplete.close()
        assert not incomplete.data["success"] and not incomplete.data["parser_verified_completion"]
        assert incomplete.data["error_code"] == "stream_missing_completion"
        assert incomplete.data["stream_exhausted"]
        # All providers/openers are replaced before run_live: these three cases
        # exercise only local sleeps, proving the deadline covers each phase.
        for blocked_phase in ("auth", "headers", "stream"):
            class FakeCredentials:
                def access_token(self):
                    if blocked_phase == "auth":
                        time.sleep(1)
                    return sentinel
            class FakeResponse:
                status = 200
                headers = Message()
                headers["Content-Type"] = "text/event-stream"
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    return False
                def __iter__(self):
                    time.sleep(1)
                    yield b"\n"
            def offline_open(*args):
                if blocked_phase == "headers":
                    time.sleep(1)
                return FakeResponse()
            destination = Path(tmp) / (blocked_phase + "_deadline.json")
            with patch.dict(globals(), DEADLINE_S=.03, SubscriptionCredentials=FakeCredentials), \
                    patch.object(plan, "_open_public", offline_open):
                assert run_live(destination) == 1
            recorded = json.loads(destination.read_text())
            assert recorded["error_code"] == "probe_wall_deadline" and not recorded["success"]
            assert recorded["elapsed_s"] < .5 and sentinel not in destination.read_text()
    print("Offline self-test passed: completion before EOF, multiline SSE, heartbeat counts, content redaction, missing completion, auth/header/stream deadlines.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Explicitly allow one bounded model-only request")
    mode.add_argument("--self-test", action="store_true", help="Run offline fixtures only")
    parser.add_argument("--output", type=Path, help="Fresh path for atomic metadata-only progress and final report")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.output is None:
        parser.error("--live requires --output")
    return run_live(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
