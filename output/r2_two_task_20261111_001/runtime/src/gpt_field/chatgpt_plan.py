"""Subscription-only, streaming Responses transport for the unchanged field policy.

Uses the public Sign in with ChatGPT route. This module never reads an API key,
never retries a request, and never substitutes a model. OAuth credential storage
and refresh belong to the caller's token provider. Subscription usage is counted
in calls and server-reported tokens; it is not priced using API-key token rates.

Protocol references:
https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference
https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from email.utils import parsedate_to_datetime
import math
import re
import time
from typing import Any, Callable, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

from .astra import (
    APIError, AstraConfig, BudgetExceeded, ENDPOINT, MissingCredentials, SchemaError,
    _canonical, _strict_loads, validate_prediction,
)
from .field_history import FieldHistoryAstraClient
from .direct_history import TASK
from .schema import GridSpec


MODELS_ENDPOINT = "https://api.openai.com/v1/models"
TRANSPORT_VERSION = "chatgpt-subscription-http-sse-v1"
UNSUPPORTED_FIELDS = frozenset({
    "background", "conversation", "max_output_tokens", "max_tool_calls", "metadata",
    "moderation", "multi_agent", "prompt", "prompt_cache_retention", "safety_identifier",
    "temperature", "top_logprobs", "top_p", "truncation", "user", "previous_response_id",
})
_PRICE_FIELDS = (
    "max_cost_usd", "input_price_per_million", "output_price_per_million",
    "cached_input_price_per_million", "cache_write_price_per_million",
)
_LOCAL_FAILURE_CODES = {
    "Subscription endpoint redirected; request stopped without retry": "endpoint_redirected",
    "Subscription stream contains invalid UTF-8": "stream_invalid_utf8",
    "Subscription stream contains an invalid line": "stream_invalid_line",
    "Subscription stream contains a malformed SSE field": "stream_malformed_sse_field",
    "Subscription stream ended within an SSE event": "stream_truncated_event",
    "Subscription stream ended before response.completed": "stream_missing_completion",
    "Subscription stream event is not valid JSON": "stream_invalid_json",
    "Subscription stream event has no valid type": "stream_missing_event_type",
    "Subscription SSE event name and JSON type disagree": "stream_event_type_mismatch",
    "Subscription completed event has an invalid response envelope": "stream_invalid_completion",
    "Subscription stream ended without response.completed": "stream_missing_completion",
    "Subscription endpoint did not return an SSE stream": "unexpected_response_content_type",
    "Subscription response model differs from requested model; output rejected": "response_model_mismatch",
}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward an OAuth bearer credential through a redirect, even if
        # the destination looks like another OpenAI endpoint.
        raise APIError("Subscription endpoint redirected; request stopped without retry")


def _safe_error_code(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("code")
    if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", value):
        return value
    return "unknown_error"


def _failure_metadata(error: BaseException) -> dict[str, Any]:
    """Only allowlisted machine diagnostics may be persisted or displayed."""
    result = {}
    status = getattr(error, "http_status", None)
    if type(status) is int and 100 <= status <= 599:
        result["http_status"] = status
    code = getattr(error, "error_code", None)
    if code is None:
        code = _LOCAL_FAILURE_CODES.get(str(error))
    if code is not None:
        result["error_code"] = _safe_error_code(code)
    retry_after = getattr(error, "retry_after_s", None)
    if (type(retry_after) in (int, float) and math.isfinite(retry_after)
            and retry_after >= 0):
        result["retry_after_s"] = retry_after
    media_type = getattr(error, "response_media_type", None)
    if isinstance(media_type, str) and re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", media_type):
        result["response_media_type"] = media_type
    return result


def _retry_after_seconds(value: Any) -> float | None:
    """Parse HTTP Retry-After without retaining arbitrary header contents."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    try:
        if re.fullmatch(r"[0-9]{1,12}", value):
            return float(value)
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            return None
        seconds = date.timestamp() - time.time()
        return max(0., seconds) if math.isfinite(seconds) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _protocol_error(message: str, *, http_status=None, error_code=None) -> APIError:
    result = APIError(message)
    result.http_status = http_status
    result.error_code = error_code
    return result


def _access_token(provider: Callable[[], str]) -> str:
    try:
        token = provider()
    except Exception:
        # Auth exceptions may contain HTTP bodies or token material. The auth
        # UI/provider must present its own sanitized recovery instructions.
        raise MissingCredentials("ChatGPT subscription authorization is unavailable") from None
    if (not isinstance(token, str) or not token or token != token.strip()
            or token.startswith("sk-") or any(char.isspace() for char in token)):
        raise MissingCredentials("A ChatGPT OAuth access token is required; API keys are not accepted")
    return token


def _open_public(request: Request, timeout_s: float):
    """One HTTP attempt at an exact public endpoint; no redirect or retry."""
    if request.full_url not in (ENDPOINT, MODELS_ENDPOINT):
        raise ValueError("Only the public Responses and models endpoints are allowed")
    try:
        return build_opener(_NoRedirect()).open(request, timeout=timeout_s)
    except HTTPError as exc:
        try:
            parsed = _strict_loads(exc.read().decode("utf-8"))
            code = _safe_error_code(parsed.get("error")) if isinstance(parsed, dict) else "unknown_error"
        except (ValueError, UnicodeError, OSError):
            code = "unknown_error"
        # Do not retain arbitrary server error text, URLs or bearer headers.
        error = _protocol_error(
            f"ChatGPT subscription HTTP {exc.code}: {code}; no retry or API-key fallback",
            http_status=exc.code, error_code=code,
        )
        error.retry_after_s = _retry_after_seconds(exc.headers.get("Retry-After") if exc.headers else None)
        raise error from None
    except (URLError, TimeoutError, OSError) as exc:
        raise _protocol_error(f"Subscription transport failure ({type(exc).__name__}); no retry",
                              error_code="subscription_network_error") from None


def _event_records(lines: Iterable[bytes | str]):
    """Parse UTF-8 SSE records, supporting comments, CRLF and multiline data."""
    event_name = None
    data: list[str] = []
    first = True
    for raw in lines:
        try:
            line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        except UnicodeError:
            raise APIError("Subscription stream contains invalid UTF-8") from None
        if not isinstance(line, str):
            raise APIError("Subscription stream contains an invalid line")
        if first:
            line = line.removeprefix("\ufeff")
            first = False
        line = line.rstrip("\r\n")
        if not line:
            if data:
                yield event_name, "\n".join(data)
            event_name, data = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event_name = value
        elif field == "data":
            data.append(value)
        elif field not in ("id", "retry"):
            raise APIError("Subscription stream contains a malformed SSE field")
    # A terminal event missing its record delimiter is an interrupted stream.
    if data:
        raise APIError("Subscription stream ended within an SSE event")


def _completed_item(item: Any) -> dict[str, Any]:
    """Check complete item structure without inspecting opaque reasoning data."""
    if (not isinstance(item, dict) or not isinstance(item.get("type"), str) or not item["type"]
            or not isinstance(item.get("id"), str) or not item["id"]
            or ("status" in item and item["status"] != "completed")):
        raise _protocol_error("Subscription stream has an invalid completed output item",
                              error_code="stream_invalid_output_item")
    if item["type"] == "message":
        content = item.get("content")
        if item.get("role") != "assistant" or not isinstance(content, list) or not content:
            raise _protocol_error("Subscription stream has an incomplete assistant message",
                                  error_code="stream_invalid_output_item")
        for part in content:
            if (not isinstance(part, dict) or part.get("type") not in ("output_text", "refusal")
                    or not isinstance(part.get("text" if part["type"] == "output_text" else "refusal"), str)):
                raise _protocol_error("Subscription stream has incomplete assistant content",
                                      error_code="stream_invalid_output_item")
    return item


class _CompletedItems:
    """Retain complete item.done events until the response itself completes.

    Deltas and item.added events only establish expected indexes/identities; they
    never supply executable text. Some subscription streams omit terminal output
    while delivering each complete item in its own authoritative done event.
    """

    def __init__(self):
        self.response_id: str | None = None
        self.done: dict[int, dict[str, Any]] = {}
        self.seen: set[int] = set()
        self.added: set[int] = set()
        self.item_ids: dict[int, str] = {}
        self.unindexed_output = False

    def _response_identity(self, value: Any) -> None:
        if (not isinstance(value, str) or not value
                or (self.response_id is not None and self.response_id != value)):
            raise _protocol_error("Subscription stream response identities disagree",
                                  error_code="stream_response_id_mismatch")
        self.response_id = value

    def _item_identity(self, index: int, value: Any) -> None:
        if (not isinstance(value, str) or not value
                or (index in self.item_ids and self.item_ids[index] != value)
                or any(other != index and ident == value for other, ident in self.item_ids.items())):
            raise _protocol_error("Subscription stream output item identities disagree",
                                  error_code="stream_output_id_mismatch")
        self.item_ids[index] = value

    def observe(self, event: Mapping[str, Any]) -> None:
        if "response_id" in event:
            self._response_identity(event["response_id"])
        response = event.get("response")
        if isinstance(response, dict) and "id" in response:
            self._response_identity(response["id"])
        kind = event["type"]
        is_item = kind in ("response.output_item.added", "response.output_item.done")
        if "output_index" not in event:
            if is_item:
                raise _protocol_error("Subscription output item has no index",
                                      error_code="stream_invalid_output_index")
            if (kind.startswith(("response.output_", "response.content_part.", "response.reasoning_"))
                    or "item_id" in event):
                self.unindexed_output = True
            return
        index = event["output_index"]
        if type(index) is not int or index < 0:
            raise _protocol_error("Subscription output item has an invalid index",
                                  error_code="stream_invalid_output_index")
        self.seen.add(index)
        if "item_id" in event:
            self._item_identity(index, event["item_id"])
        if not is_item:
            return
        item = event.get("item")
        if not isinstance(item, dict):
            raise _protocol_error("Subscription stream has a malformed output item",
                                  error_code="stream_invalid_output_item")
        self._item_identity(index, item.get("id"))
        if kind == "response.output_item.added":
            if index in self.added or index in self.done:
                raise _protocol_error("Subscription stream repeats an output item",
                                      error_code="stream_duplicate_output_item")
            self.added.add(index)
        else:
            if index in self.done:
                raise _protocol_error("Subscription stream repeats a completed output item",
                                      error_code="stream_duplicate_output_item")
            self.done[index] = copy.deepcopy(_completed_item(item))

    def finish(self, response: dict[str, Any]) -> dict[str, Any]:
        output = response["output"]
        if output:
            for index, item in enumerate(output):
                _completed_item(item)
                self._item_identity(index, item["id"])
            if any(index >= len(output) for index in self.seen):
                raise _protocol_error("Subscription terminal output omits a streamed item",
                                      error_code="stream_terminal_output_mismatch")
            if any(item != output[index] for index, item in self.done.items()):
                raise _protocol_error("Subscription terminal output contradicts a completed item",
                                      error_code="stream_terminal_output_mismatch")
            return response
        if (not self.done or self.unindexed_output or self.seen != set(self.done)
                or sorted(self.done) != list(range(len(self.done)))):
            raise _protocol_error("Subscription terminal output lacks complete contiguous output items",
                                  error_code="stream_missing_completed_output_items")
        return {**response, "output": [self.done[index] for index in range(len(self.done))],
                "_subscription_transport": {
                    **response.get("_subscription_transport", {}),
                    "terminal_output_count": 0,
                    "completed_item_count": len(self.done),
                    "output_assembly_source": "response.output_item.done",
                    "sse_completion_verified": True,
                }}


def read_completed_response(lines: Iterable[bytes | str]) -> dict[str, Any]:
    """Require response.completed; assemble only explicit complete output items."""
    items = _CompletedItems()
    for event_name, raw in _event_records(lines):
        if raw == "[DONE]":
            raise APIError("Subscription stream ended before response.completed")
        try:
            event = _strict_loads(raw)
        except (ValueError, TypeError):
            raise APIError("Subscription stream event is not valid JSON") from None
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise APIError("Subscription stream event has no valid type")
        kind = event["type"]
        if event_name not in (None, "message", kind):
            raise APIError("Subscription SSE event name and JSON type disagree")
        if kind in ("response.failed", "response.incomplete", "error", "response.error"):
            response = event.get("response")
            error = response.get("error") if isinstance(response, dict) else event.get("error", event)
            code = _safe_error_code(error)
            raise _protocol_error(f"Subscription stream {kind}: {code}; no retry or fallback", error_code=code)
        items.observe(event)
        if kind == "response.completed":
            response = event.get("response")
            if (not isinstance(response, dict) or response.get("status") != "completed"
                    or response.get("error") or not isinstance(response.get("output"), list)):
                raise APIError("Subscription completed event has an invalid response envelope")
            return items.finish(response)
    raise APIError("Subscription stream ended without response.completed")


def subscription_sse_transport(endpoint: str, payload: dict[str, Any],
                               headers: dict[str, str], timeout_s: float) -> Mapping[str, Any]:
    if endpoint != ENDPOINT:
        raise ValueError("Only the public OpenAI Responses endpoint is supported")
    if payload.get("store") is not False or payload.get("stream") is not True:
        raise ValueError("ChatGPT plan requests require store=false and stream=true")
    if UNSUPPORTED_FIELDS.intersection(payload):
        raise ValueError("Subscription request contains unsupported parameters")
    request = Request(endpoint, data=_canonical(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with _open_public(request, timeout_s) as response:
            media_type = response.headers.get_content_type()
            if media_type not in ("text/event-stream", "text/plain"):
                error = _protocol_error("Subscription endpoint did not return an SSE stream",
                    http_status=getattr(response, "status", None), error_code="unexpected_response_content_type")
                error.response_media_type = media_type
                raise error
            # Some delivery paths label SSE as text/plain. Do not infer success
            # from HTTP 200 or the MIME type: the same strict event parser must
            # still receive response.completed with a complete response object.
            completed = read_completed_response(response)
            if media_type == "text/plain":
                completed = {**completed, "_subscription_transport": {
                    **completed.get("_subscription_transport", {}),
                    "response_media_type": media_type, "sse_completion_verified": True}}
            return completed
    except (URLError, TimeoutError, OSError) as exc:
        raise _protocol_error(f"Subscription stream interrupted ({type(exc).__name__}); no retry",
                              error_code="subscription_stream_interrupted") from None


def list_plan_models(token_provider: Callable[[], str], timeout_s: float = 30) -> list[dict[str, str]]:
    """Fetch sanitized account model choices; catalog presence is not grant proof."""
    token = _access_token(token_provider)
    request = Request(MODELS_ENDPOINT, headers={"Authorization": f"Bearer {token}"}, method="GET")
    try:
        with _open_public(request, timeout_s) as response:
            parsed = _strict_loads(response.read().decode("utf-8"))
    except (ValueError, UnicodeError, URLError, TimeoutError, OSError):
        raise APIError("ChatGPT model catalog could not be read") from None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("models"), list):
        raise APIError("ChatGPT model catalog has an invalid envelope")
    models = []
    for item in parsed["models"]:
        if (not isinstance(item, dict) or not isinstance(item.get("slug"), str)
                or not item["slug"] or token in item["slug"]):
            raise APIError("ChatGPT model catalog has an invalid model entry")
        model = {"slug": item["slug"]}
        for name in ("display_name", "visibility"):
            if name in item:
                if not isinstance(item[name], str):
                    raise APIError("ChatGPT model catalog has an invalid display field")
                model[name] = item[name].replace(token, "[REDACTED]")
        models.append(model)
    return models


class ChatGPTPlanFieldClient(FieldHistoryAstraClient):
    """The original field prompt/history with explicit subscription OAuth only.

    ``max_api_calls`` can bound a connection pilot. It does not impose a robot
    action horizon. Dollar caps and API token prices are rejected, since they do
    not describe ChatGPT plan usage. The inherited config's max_output_tokens is
    omitted from HTTP and never used as a usage/accounting guard on this route.
    """

    def __init__(self, config: AstraConfig | None = None, *, token_provider: Callable[[], str],
                 image_history: int = 2, transport=None, log_dir=None, clock=time.monotonic,
                 task_prompt: str = TASK):
        config = config or AstraConfig()
        if not callable(token_provider):
            raise TypeError("token_provider must be callable")
        if any(getattr(config, name) is not None for name in _PRICE_FIELDS):
            raise ValueError("API-key dollar caps/prices do not apply to ChatGPT subscription usage")
        self.token_provider = token_provider
        self._used_access_tokens: set[str] = set()
        super().__init__(config, image_history=image_history, task_prompt=task_prompt, api_key="",
                         transport=transport or subscription_sse_transport, log_dir=log_dir, clock=clock)

    def preflight(self) -> None:
        # No authentication or inference merely to inspect a payload.
        if self.config.endpoint != ENDPOINT:
            raise ValueError("Only the public OpenAI Responses endpoint is supported")

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                      mode: str = "field") -> dict[str, Any]:
        payload = super().build_payload(observation, grid, mode)
        for name in UNSUPPORTED_FIELDS:
            payload.pop(name, None)
        payload["store"] = False
        payload["stream"] = True
        return payload

    def budget_snapshot(self) -> dict[str, Any]:
        return {
            "snapshot_version": TRANSPORT_VERSION,
            "configuration": asdict(self.config),
            "billing": "chatgpt_subscription",
            "api_calls": self._api_calls,
            "subscription_inference_calls": self._api_calls,
            "api_key_calls": 0,
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "cached_input_tokens": self._cached_tokens,
            "reasoning_tokens": self._reasoning_tokens,
            "unknown_usage_calls": self._unknown_usage_calls,
            "subscription_cost_usd": None,
            "cost_usd_upper_bound": None,
            "reservation_per_call_usd": None,
            "max_api_calls": self.config.max_api_calls,
            "max_cost_usd": None,
            "max_wall_time_s": self.config.max_wall_time_s,
            "elapsed_s": max(0.0, self.clock() - self._started),
            "output_token_limit_sent": None,
            "usage_accounting": "Server-reported tokens; account subscription limits apply; no API-key billing",
        }

    def load_budget_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        raise ValueError("Subscription pilot resumption is unsupported; use a fresh run")

    def reconcile_request_logs(self) -> int:
        raise ValueError("Subscription pilot resumption is unsupported; use a fresh run")

    def _record_usage(self, response: Mapping[str, Any], reservation=None) -> None:
        self._history_request_settled = True
        usage = response.get("usage")
        if not isinstance(usage, dict):
            self._unknown_usage_calls += 1
            return
        inp, out = usage.get("input_tokens"), usage.get("output_tokens")
        if type(inp) is not int or type(out) is not int or inp < 0 or out < 0:
            self._unknown_usage_calls += 1
            return
        self._input_tokens += inp
        self._output_tokens += out
        for section, field, attr in (("input_tokens_details", "cached_tokens", "_cached_tokens"),
                                     ("output_tokens_details", "reasoning_tokens", "_reasoning_tokens")):
            details = usage.get(section)
            value = details.get(field) if isinstance(details, dict) else None
            if type(value) is int and value >= 0:
                setattr(self, attr, getattr(self, attr) + value)

    def _write_log(self, path, data: Any) -> None:
        def redact(value):
            if isinstance(value, str):
                for token in self._used_access_tokens:
                    value = value.replace(token, "[REDACTED]")
                return value
            if isinstance(value, dict):
                return {redact(key): redact(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [redact(item) for item in value]
            return value
        if isinstance(data, dict):
            data = {**data, "billing": "chatgpt_subscription", "transport_version": TRANSPORT_VERSION}
        super()._write_log(path, redact(data))

    def predict(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                mode: str = "field") -> dict[str, Any]:
        grid = grid or GridSpec()
        with self._history_lock, self._lock:
            self.preflight()
            payload = self.build_payload(observation, grid, mode)
            _, timeout = self._reserve()
            token = _access_token(self.token_provider)
            self._used_access_tokens.add(token)
            self.last_request_key = self.request_key(payload, grid, mode)
            self.last_response = None
            ident = f"{self._api_calls + 1:06d}-{uuid4().hex}"
            path = self.log_dir / ident if self.log_dir is not None else None
            request_path = path.with_suffix(".request.json") if path is not None else None
            self._write_log(request_path, {
                "request_key": self.last_request_key, "attempt": 1,
                "grid": asdict(grid), "mode": mode, "payload": payload,
                "configuration": asdict(self.config), "request_ordinal": self._api_calls + 1,
                "budget_elapsed_s_before": max(0.0, self.clock() - self._started), "timeout_s": timeout,
            })
            self.last_log_path = str(request_path) if request_path is not None else None
            self._api_calls += 1
            settled = False
            try:
                response = self.transport(self.config.endpoint, payload, {
                    "Authorization": f"Bearer {token}", "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                }, timeout)
                if not isinstance(response, Mapping):
                    raise APIError("Subscription transport did not return a completed response")
                self.last_response = dict(response)
                self._write_log(path.with_suffix(".response.json") if path is not None else None, response)
                self._record_usage(response)
                settled = True
                if "model" in response and response["model"] != self.config.model:
                    raise APIError("Subscription response model differs from requested model; output rejected")
                if (self.config.max_wall_time_s is not None
                        and self.clock() - self._started >= self.config.max_wall_time_s):
                    raise BudgetExceeded("max_wall_time_s reached while waiting for subscription response")
                result = validate_prediction(self._extract(response), grid, mode)
                self._write_log(path.with_suffix(".result.json") if path is not None else None,
                                {"status": "valid", "budget": self.budget_snapshot()})
                self._history = payload["input"] + copy.deepcopy(response["output"])
                self._observation_count += 1
                return copy.deepcopy(result)
            except BaseException as exc:
                if not settled:
                    self._unknown_usage_calls += 1
                self._history_request_settled = True
                # Deliberately exclude str(exc) and raw HTTP bodies: credentials
                # from auth/network libraries must never reach run artifacts.
                self._write_log(path.with_suffix(".result.json") if path is not None else None,
                                {"status": "error", "error_type": type(exc).__name__,
                                 "error": "Subscription request failed; no retry or API-key fallback",
                                 **_failure_metadata(exc),
                                 "budget": self.budget_snapshot()})
                if not isinstance(exc, Exception):
                    raise
                if isinstance(exc, (APIError, SchemaError, BudgetExceeded)):
                    # Our public APIError strings are safe; remove any provider-
                    # supplied secret if a custom transport raised this error.
                    message = str(exc)
                    for used in self._used_access_tokens:
                        message = message.replace(used, "[REDACTED]")
                    sanitized = type(exc)(message)
                    for key, value in _failure_metadata(exc).items():
                        setattr(sanitized, key, value)
                    raise sanitized from None
                raise APIError(f"Subscription request failed ({type(exc).__name__}); no retry") from None
