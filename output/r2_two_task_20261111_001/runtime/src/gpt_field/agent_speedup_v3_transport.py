"""V3-only bounded subscription stream; no partial answer can authorize action."""
from __future__ import annotations

from contextlib import contextmanager
import math
import signal
import threading
import time
from urllib.error import URLError
from urllib.request import Request

from . import chatgpt_plan as protocol
from .astra import APIError, ENDPOINT
from .ordered_transport import ordered_dumps

TRANSPORT_VERSION = "agent-speedup-v3-bounded-subscription-1"
DEFAULT_REQUEST_DEADLINE_S = 180.0
MAX_VISIBLE_CHARACTERS = 131072
MAX_JSON_WHITESPACE_RUN = 2048
MAX_STREAM_BYTES = 64 * 1024 * 1024
MAX_EVENT_BYTES = 8 * 1024 * 1024
READ_CHUNK_BYTES = 16384


class SubscriptionRequestDeadline(APIError):
    """A request exceeded its total wall-clock budget, even with active traffic."""


class SubscriptionOutputGuard(APIError):
    """A stream exceeded bounded output resources; its answer was not accepted."""


class SubscriptionDeadlineUnavailable(APIError):
    """The CLI hard deadline cannot be installed without changing another timer."""


def transport_settings():
    return {"version": TRANSPORT_VERSION, "default_request_deadline_s": DEFAULT_REQUEST_DEADLINE_S,
            "deadline_source": "configured timeout_s, limited by remaining client wall-clock budget",
            "deadline_scope": "entire synchronous transport call including serialization, connect and active stream",
            "deadline_mechanism": "main-thread SIGALRM plus monotonic checks; prior handler restored",
            "deadline_precondition": "POSIX main thread and no active ITIMER_REAL timer; otherwise reject before network",
            "max_visible_characters": MAX_VISIBLE_CHARACTERS,
            "max_json_whitespace_run": MAX_JSON_WHITESPACE_RUN,
            "max_stream_bytes": MAX_STREAM_BYTES, "max_event_bytes": MAX_EVENT_BYTES,
            "partial_output_executable": False, "guard_failures_retried": False}


def _error(cls, code, message):
    error = cls(message)
    error.error_code = code
    return error


def check_deadline_capability():
    if (threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL")):
        raise _error(SubscriptionDeadlineUnavailable, "subscription_deadline_unavailable",
                     "Bounded subscription transport requires a POSIX main thread")
    timer = signal.getitimer(signal.ITIMER_REAL)
    if timer[0] or timer[1]:
        raise _error(SubscriptionDeadlineUnavailable, "subscription_deadline_unavailable",
                     "Bounded subscription transport cannot replace an active process timer")
    return timer


@contextmanager
def _request_deadline(seconds, clock, diagnostics):
    previous_timer = check_deadline_capability()
    previous_handler = signal.getsignal(signal.SIGALRM)
    expires = clock() + seconds

    def expired(*_args):
        diagnostics["deadline_exceeded"] = True
        raise _error(SubscriptionRequestDeadline, "subscription_request_deadline",
                     "Subscription request exceeded its total deadline; no partial answer accepted")

    def check():
        if clock() >= expires:
            expired()

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield check
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_handler)
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)


class _JsonTextGuard:
    """JSON whitespace outside quoted strings, retaining escape state across deltas."""
    def __init__(self):
        self.characters = 0
        self.inside_string = False
        self.escaped = False
        self.whitespace_run = 0
        self.longest_whitespace_run = 0

    def feed(self, text):
        for char in text:
            self.characters += 1
            if self.inside_string:
                self.whitespace_run = 0
                if self.escaped:
                    self.escaped = False
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.inside_string = False
            elif char in " \t\r\n":
                self.whitespace_run += 1
                self.longest_whitespace_run = max(self.longest_whitespace_run, self.whitespace_run)
                if self.whitespace_run > MAX_JSON_WHITESPACE_RUN:
                    raise _error(SubscriptionOutputGuard, "subscription_output_whitespace_limit",
                                 "Subscription JSON contains excessive consecutive syntax whitespace")
            else:
                self.whitespace_run = 0
                if char == '"':
                    self.inside_string = True
            if self.characters > MAX_VISIBLE_CHARACTERS:
                raise _error(SubscriptionOutputGuard, "subscription_visible_output_limit",
                             "Subscription visible output exceeded its character budget")


class _VisibleOutputGuard:
    def __init__(self, diagnostics):
        self.streams = {}
        self.maximum_lengths = {}
        self.diagnostics = diagnostics

    def text(self, key, text, *, complete=False):
        if not isinstance(text, str):
            return  # The unchanged protocol parser rejects malformed output.
        guard = _JsonTextGuard() if complete else self.streams.setdefault(key, _JsonTextGuard())
        try:
            guard.feed(text)
        finally:
            self.maximum_lengths[key] = max(self.maximum_lengths.get(key, 0), guard.characters)
            self.diagnostics["visible_characters"] = sum(self.maximum_lengths.values())
            self.diagnostics["longest_json_whitespace_run"] = max(
                self.diagnostics["longest_json_whitespace_run"], guard.longest_whitespace_run)
        if self.diagnostics["visible_characters"] > MAX_VISIBLE_CHARACTERS:
            raise _error(SubscriptionOutputGuard, "subscription_visible_output_limit",
                         "Subscription visible output exceeded its aggregate character budget")

    def item(self, index, item):
        if not isinstance(item, dict) or item.get("type") != "message":
            return  # Never inspect or modify opaque reasoning.
        for part_index, part in enumerate(item.get("content") or []):
            if isinstance(part, dict) and part.get("type") == "output_text":
                self.text((index, part_index), part.get("text"), complete=True)

    def event(self, event):
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "response.output_text.delta":
            self.text((event.get("output_index", 0), event.get("content_index", 0)), event.get("delta"))
        elif kind == "response.output_text.done":
            self.text((event.get("output_index", 0), event.get("content_index", 0)), event.get("text"), complete=True)
        elif kind == "response.output_item.done":
            self.item(event.get("output_index", 0), event.get("item"))
        elif kind == "response.completed":
            response = event.get("response")
            if isinstance(response, dict):
                for index, item in enumerate(response.get("output") or []):
                    self.item(index, item)


def _bounded_lines(response, check, diagnostics):
    """Bound raw chunks and records even when a provider never sends a newline."""
    pending = bytearray()
    record_bytes = 0
    read = getattr(response, "read1", None) or response.read
    while True:
        check()
        chunk = read(READ_CHUNK_BYTES)
        check()
        if not chunk:
            if pending:
                yield bytes(pending)
            return
        diagnostics["stream_bytes"] += len(chunk)
        diagnostics["chunks"] += 1
        if diagnostics["stream_bytes"] > MAX_STREAM_BYTES:
            raise _error(SubscriptionOutputGuard, "subscription_stream_size_limit", "Subscription stream byte budget exceeded")
        pending.extend(chunk)
        while b"\n" in pending:
            end = pending.index(b"\n") + 1
            line = bytes(pending[:end])
            del pending[:end]
            record_bytes += len(line)
            if record_bytes > MAX_EVENT_BYTES:
                raise _error(SubscriptionOutputGuard, "subscription_event_size_limit", "Subscription event byte budget exceeded")
            if line.rstrip(b"\r\n") == b"":
                record_bytes = 0
            yield line
        if record_bytes + len(pending) > MAX_EVENT_BYTES:
            raise _error(SubscriptionOutputGuard, "subscription_event_size_limit", "Subscription event byte budget exceeded")


def _checked_records(response, check, diagnostics):
    guard = _VisibleOutputGuard(diagnostics)
    for event_name, raw in protocol._event_records(_bounded_lines(response, check, diagnostics)):
        check()
        diagnostics["events"] += 1
        if raw != "[DONE]":
            try:
                event = protocol._strict_loads(raw)
            except (ValueError, TypeError):
                raise APIError("Subscription stream event is not valid JSON") from None
            guard.event(event)
        # Reframe the same event data for the established strict completion
        # parser, preserving all identity, done-item and error checks.
        if event_name is not None:
            yield "event: " + event_name + "\n"
        for line in raw.split("\n"):
            yield "data: " + line + "\n"
        yield "\n"


class V3BoundedSubscriptionTransport:
    """One synchronous subscription attempt with bounded time and output."""
    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.last_diagnostics = None

    def preflight(self):
        check_deadline_capability()

    def __call__(self, endpoint, payload, headers, timeout_s):
        if endpoint != ENDPOINT:
            raise ValueError("Only the public OpenAI Responses endpoint is supported")
        if payload.get("store") is not False or payload.get("stream") is not True:
            raise ValueError("Subscription requests require store=false and stream=true")
        if protocol.UNSUPPORTED_FIELDS.intersection(payload):
            raise ValueError("Subscription request contains unsupported parameters")
        seconds = float(timeout_s)
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("A finite positive total request deadline is required")
        started = self.clock()
        diagnostics = {"version": TRANSPORT_VERSION, "deadline_s": seconds, "deadline_exceeded": False,
                       "stream_bytes": 0, "chunks": 0, "events": 0, "visible_characters": 0,
                       "longest_json_whitespace_run": 0, "partial_output_accepted": False}
        self.last_diagnostics = diagnostics
        try:
            with _request_deadline(seconds, self.clock, diagnostics) as check:
                request = Request(endpoint, data=ordered_dumps(payload).encode(), headers=headers, method="POST")
                check()
                with protocol._open_public(request, max(.001, seconds-(self.clock()-started))) as response:
                    media_type = response.headers.get_content_type()
                    if media_type not in ("text/event-stream", "text/plain"):
                        raise protocol._protocol_error("Subscription endpoint did not return an SSE stream",
                            http_status=getattr(response, "status", None), error_code="unexpected_response_content_type")
                    completed = protocol.read_completed_response(_checked_records(response, check, diagnostics))
                    check()
                    diagnostics.update(status="completed", elapsed_s=max(0., self.clock()-started))
                    return {**completed, "_subscription_transport": {
                        **completed.get("_subscription_transport", {}), "response_media_type": media_type,
                        "sse_completion_verified": True, "v3_bounds": dict(diagnostics)}}
        except (URLError, TimeoutError, OSError) as exc:
            diagnostics.update(status="error", error_code="subscription_stream_interrupted")
            raise protocol._protocol_error(f"Subscription stream interrupted ({type(exc).__name__})",
                                           error_code="subscription_stream_interrupted") from None
        except BaseException as exc:
            diagnostics.update(status="error", error_code=getattr(exc, "error_code", None), error_type=type(exc).__name__)
            raise
        finally:
            diagnostics["elapsed_s"] = max(0., self.clock()-started)
