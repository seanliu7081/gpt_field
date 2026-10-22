"""The frozen five-action history interface through ChatGPT subscription OAuth.

Only the transport changes relative to ``DirectHistoryAstraClient``: this class
uses the same subscription-only streaming/validation/accounting implementation
as the field client. It does not change models, retry, read API keys, execute
actions, or introduce a grasp controller.
"""

from __future__ import annotations

from typing import Any, Mapping

from .chatgpt_plan import ChatGPTPlanFieldClient, TRANSPORT_VERSION, UNSUPPORTED_FIELDS
from .direct_history import DirectHistoryAstraClient
from .schema import GridSpec


class ChatGPTPlanDirectClient(ChatGPTPlanFieldClient):
    """Return exactly five normalized 7D actions using the original direct prompt.

    History, feedback validation, snapshot copying, and image retention remain
    inherited. Mode defaults to direct and cannot silently become field mode.
    The inherited subscription client owns inference attempts, token accounting,
    response completion, model checking, and history commit-on-valid semantics.
    """

    _instructions = staticmethod(DirectHistoryAstraClient._instructions)
    request_key = DirectHistoryAstraClient.request_key

    def build_payload(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                      mode: str = "direct") -> dict[str, Any]:
        # The direct implementation's lexical super resolves to AstraClient,
        # preserving the exact frozen direct prompt/schema/history behavior.
        payload = DirectHistoryAstraClient.build_payload(self, observation, grid, mode)
        for name in UNSUPPORTED_FIELDS:
            payload.pop(name, None)
        payload["store"] = False
        payload["stream"] = True
        return payload

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
        # Bypass the field-specific logger, which would stamp a field prompt
        # version onto a direct-action request. Keep full direct provenance.
        DirectHistoryAstraClient._write_log(self, path, redact(data))

    def predict(self, observation: Mapping[str, Any], grid: GridSpec | None = None,
                mode: str = "direct") -> dict[str, Any]:
        return super().predict(observation, grid, mode)
