"""Strict request/response JSON lines on a dedicated inherited Unix socket.

The model remains on the parent's main thread. This module does not invoke a
model, read credentials, start processes, or use stdin/stdout as the protocol.
A socketpair passed with pass_fds remains independent of a worker's dup2 logs.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
import os
import re
import socket
import time


class ProtocolError(RuntimeError):
    pass


class RemoteError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


_WORKER_ENV_KEYS = frozenset({
    "PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM", "HOME", "USER", "LOGNAME",
    "PYTHONPATH", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE",
    "VIRTUAL_ENV", "LD_LIBRARY_PATH", "CUDA_HOME", "CUDA_PATH", "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
    "OMNI_KIT_ACCEPT_EULA", "ACCEPT_EULA", "DISPLAY", "XAUTHORITY", "TMPDIR",
    "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_RUNTIME_DIR",
})


def worker_environment(environment: Mapping[str, str], *, overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Allowlist runtime settings; never copy auth/token/key environment fields.

    License keys here are acceptance switches, not authorization to accept them.
    Caller must already have user consent before supplying such a switch.
    """
    result = {key: value for key, value in environment.items() if key in _WORKER_ENV_KEYS}
    for key, value in (overrides or {}).items():
        if key not in _WORKER_ENV_KEYS:
            raise ProtocolError("Worker environment override is not allowlisted")
        result[key] = value
    if any(not isinstance(value, str) for value in result.values()):
        raise ProtocolError("Worker environment values must be strings")
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _constant(_value):
    raise ValueError("Non-finite JSON constant")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON number")
    return number


class JsonLineChannel:
    VERSION = 1

    def __init__(self, connection: socket.socket, *, role: str,
                 max_bytes: int = 16 * 1024 * 1024, timeout_s: float = 30.0):
        if role not in {"parent", "worker"}:
            raise ValueError("role must be parent or worker")
        if type(max_bytes) is not int or max_bytes < 128:
            raise ValueError("max_bytes must be at least 128")
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")
        if connection.family != socket.AF_UNIX or connection.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM:
            raise ValueError("A dedicated Unix stream socket is required")
        self.socket = connection
        self.role = role
        self.max_bytes = max_bytes
        self.timeout_s = float(timeout_s)
        self.state = "IDLE"
        self.next_sequence = 1
        self._pending_sequence: int | None = None
        self._buffer = bytearray()

    @classmethod
    def from_fd(cls, fd: int, **kwargs):
        """Duplicate the inherited fd so channel ownership is explicit."""
        return cls(socket.socket(fileno=os.dup(fd)), **kwargs)

    def close(self) -> None:
        if self.state != "CLOSED":
            self.state = "CLOSED"
            self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def _fail(self, message: str):
        self.close()
        raise ProtocolError(message)

    def _deadline(self, timeout_s: float | None) -> float:
        duration = self.timeout_s if timeout_s is None else timeout_s
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Deadline duration must be positive and finite")
        return time.monotonic() + duration

    def _set_remaining(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            self._fail("IPC deadline exceeded")
        self.socket.settimeout(remaining)

    def _write(self, message: dict, deadline: float) -> None:
        try:
            raw = json.dumps(message, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False).encode("utf-8") + b"\n"
        except (ValueError, TypeError, RecursionError):
            self._fail("IPC payload is not finite JSON")
        if len(raw) > self.max_bytes:
            self._fail("IPC frame exceeds byte limit")
        try:
            self._set_remaining(deadline)
            self.socket.sendall(raw)
        except (TimeoutError, OSError):
            self._fail("IPC write failed or timed out")

    def _read(self, deadline: float | None) -> dict:
        # None permits an idle worker to wait through model/approval latency.
        # Once any frame byte arrives, normal bounded completion resumes.
        if deadline is None and self._buffer:
            deadline = time.monotonic() + self.timeout_s
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                if newline + 1 > self.max_bytes:
                    self._fail("IPC frame exceeds byte limit")
                raw = bytes(self._buffer[:newline])
                del self._buffer[:newline + 1]
                break
            if len(self._buffer) >= self.max_bytes:
                self._fail("IPC frame exceeds byte limit")
            try:
                if deadline is None:
                    self.socket.settimeout(None)
                else:
                    self._set_remaining(deadline)
                part = self.socket.recv(min(65536, self.max_bytes - len(self._buffer)))
            except (TimeoutError, OSError):
                self._fail("IPC read failed or timed out")
            if not part:
                self._fail("IPC peer closed before a complete frame")
            if deadline is None:
                deadline = time.monotonic() + self.timeout_s
            self._buffer.extend(part)
        try:
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant, parse_float=_float)
        except (ValueError, UnicodeDecodeError, RecursionError):
            self._fail("Malformed IPC JSON frame")
        if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != self.VERSION:
            self._fail("Unsupported IPC envelope")
        if type(value.get("seq")) is not int or value["seq"] != self.next_sequence:
            self._fail("IPC sequence mismatch")
        return value

    def call(self, method: str, payload: Mapping | None = None, *, timeout_s: float | None = None) -> dict:
        if self.role != "parent" or self.state != "IDLE":
            raise ProtocolError("Only an idle parent may send a request")
        if not isinstance(method, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", method) is None:
            raise ProtocolError("Invalid IPC method")
        if payload is not None and not isinstance(payload, Mapping):
            raise ProtocolError("Request payload must be an object")
        deadline = self._deadline(timeout_s)
        self.state = "WAITING_RESPONSE"
        self._write({"version": self.VERSION, "kind": "request", "seq": self.next_sequence,
                     "method": method, "payload": dict(payload or {})}, deadline)
        response = self._read(deadline)
        if set(response) != {"version", "kind", "seq", "ok", "payload", "error"} or response["kind"] != "response" or type(response["ok"]) is not bool:
            self._fail("Invalid IPC response envelope")
        if response["ok"]:
            if not isinstance(response["payload"], dict) or response["error"] is not None:
                self._fail("Invalid IPC success response")
        elif response["payload"] is not None or not isinstance(response["error"], str) or re.fullmatch(r"[a-z][a-z0-9_]{0,95}", response["error"]) is None:
            self._fail("Invalid IPC error response")
        self.next_sequence += 1
        self.state = "IDLE"
        if not response["ok"]:
            raise RemoteError(response["error"])
        return response["payload"]

    def receive_request(self, *, timeout_s: float | None = None) -> dict:
        """Wait indefinitely while idle by default; bound partial frames.

        An explicit timeout bounds the entire wait. This is only socket
        waiting: it does not heartbeat Kit or advance simulator physics.
        """
        if self.role != "worker" or self.state != "IDLE":
            raise ProtocolError("Only an idle worker may receive a request")
        self.state = "READING_REQUEST"
        request = self._read(None if timeout_s is None else self._deadline(timeout_s))
        if (set(request) != {"version", "kind", "seq", "method", "payload"}
                or request["kind"] != "request" or not isinstance(request["payload"], dict)
                or not isinstance(request["method"], str)
                or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", request["method"]) is None):
            self._fail("Invalid IPC request envelope")
        self._pending_sequence = request["seq"]
        self.state = "HANDLING_REQUEST"
        return request

    def respond(self, sequence: int, payload: Mapping | None = None, *, error_code: str | None = None,
                timeout_s: float | None = None) -> None:
        if self.role != "worker" or self.state != "HANDLING_REQUEST" or type(sequence) is not int or sequence != self._pending_sequence:
            raise ProtocolError("Response must match the pending worker request")
        if payload is not None and not isinstance(payload, Mapping):
            raise ProtocolError("Response payload must be an object")
        if error_code is not None and (not isinstance(error_code, str) or re.fullmatch(r"[a-z][a-z0-9_]{0,95}", error_code) is None or payload is not None):
            raise ProtocolError("Errors carry only a sanitized code")
        self._write({"version": self.VERSION, "kind": "response", "seq": sequence,
                     "ok": error_code is None, "payload": dict(payload or {}) if error_code is None else None,
                     "error": error_code}, self._deadline(timeout_s))
        self._pending_sequence = None
        self.next_sequence += 1
        self.state = "IDLE"
