"""Opt-in user-check request deadlines and content-free stream diagnostics.

Authentication remains in the existing client before this transport starts.
The deadline covers serialization, HTTP opening, and the entire SSE stream;
it is not an episode budget. The production ordered serializer and strict
completion parser remain authoritative and receive unchanged request/line data.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import threading
import time
from urllib.error import URLError
from urllib.request import Request

from . import chatgpt_plan as protocol
from .astra import APIError, ENDPOINT
from .ordered_transport import ordered_dumps, ordered_sse_transport

TRANSPORT_VERSION = "user-check-stream-diagnostics-2"
MAX_TEXT_PARTS = 16
MAX_LEXICAL_DEPTH = 128
PLAN_FORMAT_NAMES = frozenset({"agent_speedup_v2_plan", "agent_speedup_v2_replan"})
MAX_PLAN_VISIBLE_CHARACTERS = 131072
MAX_PLAN_JSON_WHITESPACE_RUN = 2048
JSON_WHITESPACE = frozenset(" \t\r\n")
EVENT_TYPES = frozenset({
    "response.created", "response.queued", "response.in_progress", "response.completed",
    "response.failed", "response.incomplete", "response.error", "error",
    "response.output_item.added", "response.output_item.done",
    "response.content_part.added", "response.content_part.done",
    "response.output_text.delta", "response.output_text.done",
    "response.refusal.delta", "response.refusal.done",
    "response.reasoning_summary_part.added", "response.reasoning_summary_part.done",
    "response.reasoning_summary_text.delta", "response.reasoning_summary_text.done",
    "response.reasoning_text.delta", "response.reasoning_text.done", "ping", "heartbeat",
})
SAFE_ERROR_CODES = frozenset(protocol._LOCAL_FAILURE_CODES.values()) | frozenset({
    "request_deadline_exceeded", "request_deadline_unavailable", "subscription_network_error",
    "subscription_stream_interrupted", "response_reasoning_effort_mismatch",
    "stream_invalid_output_item", "stream_response_id_mismatch", "stream_output_id_mismatch",
    "stream_invalid_output_index", "stream_duplicate_output_item", "stream_terminal_output_mismatch",
    "stream_missing_completed_output_items", "server_is_overloaded", "rate_limit_exceeded",
    "insufficient_quota", "invalid_api_key", "invalid_request_error", "server_error", "unknown_error",
    "plan_output_whitespace_limit", "plan_visible_output_limit",
})
SAFE_ERROR_TYPES = frozenset({
    "APIError", "TimeoutError", "URLError", "HTTPError", "OSError", "ConnectionError",
    "ConnectionResetError", "BrokenPipeError", "ValueError", "TypeError", "UnicodeError",
    "KeyboardInterrupt", "SystemExit", "MissingCredentials", "BudgetExceeded",
})


def _safe_failure(error):
    code = protocol._failure_metadata(error).get("error_code")
    name = type(error).__name__
    return {"error_type": name if name in SAFE_ERROR_TYPES else "OtherError",
            "error_code": code if code in SAFE_ERROR_CODES or code is None else "other_error"}


def transport_settings(request_deadline_s=None, stream_progress=False):
    if request_deadline_s is not None:
        if (type(request_deadline_s) not in (int, float)
                or not math.isfinite(request_deadline_s) or request_deadline_s <= 0):
            raise ValueError("request_deadline_s must be finite and positive")
        request_deadline_s = float(request_deadline_s)
    if type(stream_progress) is not bool:
        raise ValueError("stream_progress must be a boolean")
    return {
        "version": TRANSPORT_VERSION,
        "enabled": request_deadline_s is not None or stream_progress,
        "request_deadline_s": request_deadline_s, "stream_progress": stream_progress,
        "deadline_scope": "HTTP open and complete SSE stream; starts after authentication",
        "deadline_mechanism": "POSIX main-thread SIGALRM plus monotonic checks; no active prior timer",
        "deadline_error_code": "request_deadline_exceeded", "deadline_retried": False,
        "progress_file": "requests/<request UUID>.stream.json" if stream_progress else None,
        "logged_content": "phase, timing, counts, allowlisted event types and response metadata only",
        "visible_output_diagnostics": "delta character/UTF-8/whitespace counts and bounded lexical state; no text",
        "max_tracked_output_parts": MAX_TEXT_PARTS, "max_lexical_depth": MAX_LEXICAL_DEPTH,
        "plan_output_guard": {
            "scope": sorted(PLAN_FORMAT_NAMES), "requires_strict_json_schema": True,
            "max_visible_characters": MAX_PLAN_VISIBLE_CHARACTERS,
            "max_consecutive_json_syntax_whitespace": MAX_PLAN_JSON_WHITESPACE_RUN,
            "whitespace_inside_strings_excluded": True, "guard_failures_retried": False,
        },
        "partial_output_executable": False,
    }


def _error(code, message):
    error = APIError(message)
    error.error_code = code
    return error


@contextmanager
def _request_deadline(seconds, clock):
    if seconds is None:
        yield lambda: None
        return
    if (threading.current_thread() is not threading.main_thread()
            or not hasattr(signal, "setitimer") or not hasattr(signal, "ITIMER_REAL")):
        raise _error("request_deadline_unavailable", "Request deadline requires a POSIX main thread")
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if any(previous_timer):
        raise _error("request_deadline_unavailable", "Request deadline cannot replace an active process timer")
    previous_handler = signal.getsignal(signal.SIGALRM)
    expires = clock() + seconds
    def expired(*_args):
        raise _error("request_deadline_exceeded", "Subscription request exceeded its hard deadline")
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


def _safe_id(value, prefix):
    return value if isinstance(value, str) and re.fullmatch(prefix + r"[A-Za-z0-9_-]{1,150}", value) else None


class _Progress:
    def __init__(self, path, settings, clock):
        self.path, self.clock, self.started = path, clock, clock()
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.thread = None
        self.data = {
            "version": TRANSPORT_VERSION, "started_at": datetime.now(timezone.utc).isoformat(),
            "phase": "transport_started", "settings": settings,
            "authentication": "completed before transport invocation",
            "bytes_read": 0, "lines_read": 0, "comment_lines": 0,
            "heartbeat_events": 0, "events_seen": 0, "event_types": {},
            "stream_exhausted": False, "parser_verified_completion": False,
            "partial_output_accepted": False,
        }
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.open("x").close()
            self.flush()
            self.thread = threading.Thread(target=self._ticker, daemon=True)
            self.thread.start()

    def elapsed(self):
        return round(max(0., self.clock() - self.started), 6)

    def flush(self):
        with self.lock:
            self.data["elapsed_s"] = self.elapsed()
            if self.path is not None:
                temporary = self.path.with_name(self.path.name + ".tmp")
                temporary.write_text(json.dumps(self.data, indent=2) + "\n")
                os.replace(temporary, self.path)

    def update(self, **values):
        with self.lock:
            self.data.update(values)
            self.flush()

    def _ticker(self):
        while not self.stop.wait(1.):
            try:
                self.flush()
            except OSError:
                # The main-thread final flush still reports a persistent I/O failure.
                with self.lock:
                    self.data["progress_write_error"] = "OSError"

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
        self.flush()


class _TextPartShape:
    """Bounded lexical state only: root closure is never a JSON-validity claim."""
    def __init__(self):
        self.stack = []
        self.depth = self.syntax_whitespace = self.syntax_run = self.max_syntax_run = 0
        self.whitespace_run = self.max_whitespace_run = 0
        self.in_string = self.escaped = self.depth_limited = self.anomaly = False
        self.root_kind = None
        self.root_closed = self.trailing = False

    def feed(self, text):
        for char in text:
            self.whitespace_run = self.whitespace_run + 1 if char in JSON_WHITESPACE else 0
            self.max_whitespace_run = max(self.max_whitespace_run, self.whitespace_run)
            if self.in_string:
                self.syntax_run = 0
                if ord(char) < 32:
                    self.anomaly = True
                if self.escaped:
                    self.escaped = False
                    if char not in '"\\/bfnrtu':
                        self.anomaly = True
                elif char == "\\":
                    self.escaped = True
                elif char == '"':
                    self.in_string = False
                continue
            if char in JSON_WHITESPACE:
                self.syntax_whitespace += 1
                self.syntax_run += 1
                self.max_syntax_run = max(self.max_syntax_run, self.syntax_run)
                continue
            self.syntax_run = 0
            if self.root_closed:
                self.trailing = True
            if self.root_kind is None:
                self.root_kind = "object" if char == "{" else "array" if char == "[" else "other"
            if char == '"':
                self.in_string = True
            elif char in "{[":
                self.depth += 1
                if self.depth <= MAX_LEXICAL_DEPTH:
                    self.stack.append(char)
                else:
                    self.depth_limited = True
            elif char in "}]":
                if self.depth == 0:
                    self.anomaly = True
                else:
                    if self.depth <= MAX_LEXICAL_DEPTH:
                        if self.stack.pop() != ("{" if char == "}" else "["):
                            self.anomaly = True
                    self.depth -= 1
                    if self.depth == 0 and self.root_kind in ("object", "array"):
                        self.root_closed = True

    def snapshot(self):
        return {"container_depth": self.depth, "inside_string": self.in_string,
                "escape_pending": self.escaped, "root_kind": self.root_kind,
                "root_closed_candidate": self.root_closed and not (self.anomaly or self.trailing or self.depth_limited),
                "lexical_anomaly": self.anomaly, "trailing_non_whitespace": self.trailing,
                "depth_tracking_limited": self.depth_limited}


class _VisibleOutput:
    def __init__(self, progress, guard):
        self.progress, self.guard, self.parts = progress, guard, {}
        progress.data.update(
            visible_delta_events=0, visible_characters=0, visible_utf8_bytes=0,
            visible_whitespace_characters=0, visible_syntax_whitespace_characters=0,
            longest_visible_whitespace_run=0, longest_json_whitespace_run=0,
            visible_invalid_delta_events=0, visible_untracked_delta_events=0,
            visible_invalid_unicode_events=0, visible_output_parts={}, plan_output_guard_applied=guard,
            visible_shape_note="Lexical root closure is a candidate only; strict completed-response parsing remains authoritative.")

    def _check(self, shape=None, characters=None):
        if not self.guard:
            return
        if characters is not None and characters > MAX_PLAN_VISIBLE_CHARACTERS:
            raise _error("plan_visible_output_limit", "Plan visible output exceeded its character bound")
        if shape is not None and shape.max_syntax_run > MAX_PLAN_JSON_WHITESPACE_RUN:
            raise _error("plan_output_whitespace_limit", "Plan output exceeded its JSON syntax whitespace bound")

    def delta(self, event):
        text = event.get("delta")
        stats = self.progress.data
        if not isinstance(text, str):
            stats["visible_invalid_delta_events"] += 1
            return
        stats["visible_delta_events"] += 1
        stats["visible_characters"] += len(text)
        stats["visible_whitespace_characters"] += sum(char in JSON_WHITESPACE for char in text)
        try:
            stats["visible_utf8_bytes"] += len(text.encode("utf-8"))
        except UnicodeError:
            stats["visible_invalid_unicode_events"] += 1
        if text:
            stats.setdefault("first_visible_output_s", self.progress.elapsed())
            stats["last_visible_output_s"] = self.progress.elapsed()
        self._check(characters=stats["visible_characters"])
        indexes = (event.get("output_index"), event.get("content_index"))
        if any(type(index) is not int or not 0 <= index <= 4095 for index in indexes):
            stats["visible_untracked_delta_events"] += 1
            return
        key = f"{indexes[0]}:{indexes[1]}"
        if key not in self.parts:
            if len(self.parts) >= MAX_TEXT_PARTS:
                stats["visible_untracked_delta_events"] += 1
                return
            self.parts[key] = _TextPartShape()
        part = self.parts[key]
        before = part.syntax_whitespace
        part.feed(text)
        stats["visible_syntax_whitespace_characters"] += part.syntax_whitespace - before
        stats["longest_visible_whitespace_run"] = max(stats["longest_visible_whitespace_run"], part.max_whitespace_run)
        stats["longest_json_whitespace_run"] = max(stats["longest_json_whitespace_run"], part.max_syntax_run)
        stats["visible_output_parts"][key] = part.snapshot()
        self._check(part)

    def completed_text(self, text):
        if self.guard and isinstance(text, str):
            self._check(characters=len(text))
            part = _TextPartShape()
            part.feed(text)
            self._check(part)

    def event(self, event, kind):
        if kind == "response.output_text.delta":
            self.delta(event)
        elif self.guard and kind == "response.output_text.done":
            self.completed_text(event.get("text"))
        elif self.guard and kind in ("response.output_item.done", "response.completed"):
            response = event.get("response")
            items = response.get("output", []) if kind == "response.completed" and isinstance(response, dict) else [event.get("item")]
            if not isinstance(items, list):
                return
            total = 0
            for item in items:
                if not isinstance(item, dict) or item.get("type") != "message" or not isinstance(item.get("content"), list):
                    continue
                for part in item["content"]:
                    if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                        total += len(part["text"])
                        self._check(characters=total)
                        self.completed_text(part["text"])


class _ObservedLines:
    def __init__(self, lines, progress, check, guard=False):
        self.lines, self.progress, self.check = iter(lines), progress, check
        self.event_name, self.data_lines, self.first = None, [], True
        self.visible = _VisibleOutput(progress, guard)

    def __iter__(self):
        return self

    def __next__(self):
        self.check()
        try:
            raw = next(self.lines)
        except StopIteration:
            self.progress.update(stream_exhausted=True)
            raise
        self.check()
        with self.progress.lock:
            stats = self.progress.data
            stats.setdefault("first_line_s", self.progress.elapsed())
            stats["last_line_s"] = self.progress.elapsed()
            stats["lines_read"] += 1
            stats["bytes_read"] += (len(raw) if isinstance(raw, bytes) else
                                    len(raw.encode("utf-8")) if isinstance(raw, str) else 0)
            try:
                line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                if not isinstance(line, str):
                    raise TypeError("Non-text line is left to the production parser")
                if self.first:
                    line, self.first = line.removeprefix("\ufeff"), False
                line = line.rstrip("\r\n")
                if not line:
                    if self.data_lines:
                        self._event("\n".join(self.data_lines))
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
                stats["telemetry_decode_errors"] = stats.get("telemetry_decode_errors", 0) + 1
        return raw

    def _event(self, raw):
        stats = self.progress.data
        stats["events_seen"] += 1
        stats.setdefault("first_event_s", self.progress.elapsed())
        stats["last_event_s"] = self.progress.elapsed()
        try:
            event = json.loads(raw)
            kind = event.get("type") if isinstance(event, dict) else None
        except (ValueError, TypeError):
            event, kind = None, "done_marker" if raw == "[DONE]" else "unparsed"
        label = kind if kind in EVENT_TYPES or kind in ("done_marker", "unparsed") else "other"
        stats["event_types"][label] = stats["event_types"].get(label, 0) + 1
        if kind in ("ping", "heartbeat") or self.event_name in ("ping", "heartbeat"):
            stats["heartbeat_events"] += 1
        if isinstance(event, dict):
            self.visible.event(event, kind)
        response = event.get("response") if isinstance(event, dict) else None
        if isinstance(response, dict):
            stats["response_id"] = _safe_id(response.get("id"), "resp_")
            status = response.get("status")
            stats["response_status"] = status if status in ("queued", "in_progress", "completed", "failed", "incomplete") else "other"


class UserCheckSubscriptionTransport:
    """One ordered HTTP/SSE attempt, optionally bounded and observed; no retry."""
    def __init__(self, *, request_deadline_s=None, stream_progress=False,
                 request_path=None, clock=time.monotonic):
        self.settings = transport_settings(request_deadline_s, stream_progress)
        self.request_path, self.clock, self.last_diagnostics = request_path, clock, None

    def __call__(self, endpoint, payload, headers, timeout_s):
        if endpoint != ENDPOINT:
            raise ValueError("Only the public OpenAI Responses endpoint is supported")
        if payload.get("store") is not False or payload.get("stream") is not True:
            raise ValueError("Subscription requests require store=false and stream=true")
        if protocol.UNSUPPORTED_FIELDS.intersection(payload):
            raise ValueError("Subscription request contains unsupported parameters")
        path = None
        if self.settings["stream_progress"]:
            current = self.request_path() if self.request_path is not None else None
            if not current or not str(current).endswith(".request.json"):
                raise ValueError("Stream progress requires the current logged request path")
            path = Path(str(current)[:-len(".request.json")] + ".stream.json")
        progress = _Progress(path, self.settings, self.clock)
        self.last_diagnostics = progress.data
        seconds = self.settings["request_deadline_s"]
        try:
            with _request_deadline(seconds, self.clock) as check:
                request = Request(endpoint, data=ordered_dumps(payload).encode("utf-8"), headers=headers, method="POST")
                check()
                progress.update(phase="opening_http", http_open_started_s=progress.elapsed())
                # Preserve the existing socket idle timeout. SIGALRM separately
                # owns the whole-request deadline and its nonretryable error.
                with protocol._open_public(request, timeout_s) as response:
                    media_type = response.headers.get_content_type()
                    status = getattr(response, "status", None)
                    progress.update(
                        phase="reading_stream", headers_received_s=progress.elapsed(),
                        http_status=status if type(status) is int else None,
                        response_media_type=media_type if media_type in ("text/plain", "text/event-stream") else "other",
                        provider_request_id=_safe_id(response.headers.get("x-request-id"), "req_"),
                    )
                    if media_type not in ("text/plain", "text/event-stream"):
                        raise protocol._protocol_error("Subscription endpoint did not return an SSE stream",
                            http_status=status, error_code="unexpected_response_content_type")
                    text_options = payload.get("text")
                    text_format = text_options.get("format") if isinstance(text_options, dict) else None
                    guard = (isinstance(text_format, dict) and text_format.get("type") == "json_schema"
                             and text_format.get("strict") is True and isinstance(text_format.get("name"), str)
                             and text_format["name"] in PLAN_FORMAT_NAMES)
                    completed = protocol.read_completed_response(_ObservedLines(response, progress, check, guard))
                    check()
                    reasoning = completed.get("reasoning")
                    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
                    progress.update(
                        phase="completed", completion_s=progress.elapsed(), parser_verified_completion=True,
                        response_id=_safe_id(completed.get("id"), "resp_"), response_status="completed",
                        response_model="gpt-6-astra" if completed.get("model") == "gpt-6-astra" else "other",
                        response_effort=effort if effort in ("low", "medium", "high", "xhigh") else None,
                    )
                    if media_type == "text/plain":
                        completed = {**completed, "_subscription_transport": {
                            **completed.get("_subscription_transport", {}),
                            "response_media_type": media_type, "sse_completion_verified": True}}
                    return completed
        except (URLError, TimeoutError, OSError) as exc:
            error = protocol._protocol_error(f"Subscription stream interrupted ({type(exc).__name__}); no retry",
                                             error_code="subscription_stream_interrupted")
            progress.update(phase="failed", **_safe_failure(error))
            raise error from None
        except BaseException as exc:
            progress.update(phase="failed", **_safe_failure(exc))
            raise
        finally:
            progress.close()


def configure_client_transport(client, *, request_deadline_s=None, stream_progress=False):
    settings = transport_settings(request_deadline_s, stream_progress)
    if settings["enabled"]:
        if client.transport is not ordered_sse_transport:
            raise ValueError("Stream options require the production ordered subscription transport")
        client.transport = UserCheckSubscriptionTransport(
            request_deadline_s=request_deadline_s, stream_progress=stream_progress,
            request_path=lambda: client.last_log_path)
    return settings
