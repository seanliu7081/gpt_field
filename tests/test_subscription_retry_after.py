"""Retry hints survive sanitized HTTP errors without changing transport attempts."""
import io
import json
from urllib.error import HTTPError
from unittest.mock import Mock, patch

import pytest

from gpt_field.astra import APIError, ENDPOINT
from gpt_field.chatgpt_plan import (
    _failure_metadata, _retry_after_seconds, subscription_sse_transport,
)


@pytest.mark.parametrize("value, expected", [
    ("7", 7.), (" 0 ", 0.), (None, None), ("NaN", None), ("-4", None),
    ("1.5", None), ("untrusted-secret", None),
])
def test_retry_after_numeric_or_invalid(value, expected):
    assert _retry_after_seconds(value) == expected


def test_retry_after_http_date():
    with patch("gpt_field.chatgpt_plan.time.time", return_value=1445412470):
        assert _retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") == 10.
        assert _retry_after_seconds("Wed, 21 Oct 2015 07:27:00 GMT") == 0.


def test_http_overload_preserves_safe_retry_hint_and_does_not_retry_transport():
    body = io.BytesIO(json.dumps({"error": {"code": "server_is_overloaded",
                                          "message": "untrusted-secret"}}).encode())
    error = HTTPError(ENDPOINT, 503, "untrusted-secret", {"Retry-After": "8"}, body)
    opener = Mock()
    opener.open.side_effect = error
    with patch("gpt_field.chatgpt_plan.build_opener", return_value=opener):
        with pytest.raises(APIError) as caught:
            subscription_sse_transport(ENDPOINT, {"store": False, "stream": True}, {}, 30)
    assert _failure_metadata(caught.value) == {
        "http_status": 503, "error_code": "server_is_overloaded", "retry_after_s": 8.,
    }
    assert "untrusted-secret" not in str(caught.value)
    assert opener.open.call_count == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "secret"])
def test_metadata_drops_invalid_retry_hint(value):
    error = APIError("failure")
    error.retry_after_s = value
    assert "retry_after_s" not in _failure_metadata(error)
