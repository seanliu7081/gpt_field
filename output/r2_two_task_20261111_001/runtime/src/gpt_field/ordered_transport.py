"""Subscription SSE transport that preserves payload key order on the wire.

The legacy transport serializes with ``sort_keys=True`` (astra._canonical), so
structured-output schema properties reach the server alphabetized. This copy of
``chatgpt_plan.subscription_sse_transport`` differs only in the serializer: the
request body keeps insertion order, so a schema can deliberately put its
geometry/rationale field before the action fields. All endpoint, store/stream,
unsupported-parameter, media-type and strict SSE completion checks are reused.
"""

from __future__ import annotations

import json
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import Request

from . import chatgpt_plan as _plan
from .astra import ENDPOINT

ORDERED_TRANSPORT_VERSION = "chatgpt-subscription-http-sse-ordered-v1"


def ordered_dumps(payload: Any) -> str:
    """Compact JSON in insertion order; NaN/Infinity are rejected."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def ordered_sse_transport(endpoint: str, payload: dict[str, Any],
                          headers: dict[str, str], timeout_s: float) -> Mapping[str, Any]:
    if endpoint != ENDPOINT:
        raise ValueError("Only the public OpenAI Responses endpoint is supported")
    if payload.get("store") is not False or payload.get("stream") is not True:
        raise ValueError("ChatGPT plan requests require store=false and stream=true")
    if _plan.UNSUPPORTED_FIELDS.intersection(payload):
        raise ValueError("Subscription request contains unsupported parameters")
    request = Request(endpoint, data=ordered_dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with _plan._open_public(request, timeout_s) as response:
            media_type = response.headers.get_content_type()
            if media_type not in ("text/event-stream", "text/plain"):
                error = _plan._protocol_error("Subscription endpoint did not return an SSE stream",
                    http_status=getattr(response, "status", None), error_code="unexpected_response_content_type")
                error.response_media_type = media_type
                raise error
            completed = _plan.read_completed_response(response)
            if media_type == "text/plain":
                completed = {**completed, "_subscription_transport": {
                    **completed.get("_subscription_transport", {}),
                    "response_media_type": media_type, "sse_completion_verified": True}}
            return completed
    except (URLError, TimeoutError, OSError) as exc:
        raise _plan._protocol_error(f"Subscription stream interrupted ({type(exc).__name__}); no retry",
                                    error_code="subscription_stream_interrupted") from None
