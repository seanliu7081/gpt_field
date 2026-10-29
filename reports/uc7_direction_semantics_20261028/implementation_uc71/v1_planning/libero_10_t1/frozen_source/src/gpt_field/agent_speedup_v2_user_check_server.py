"""Loopback-only, selection-based review UI for the separate user-check runner.

HTTP handlers only read/edit/approve one ReviewStore. They never import a runner,
request model inference, create a simulator, or execute an action. The owning
command decides what to do after approval.
"""

from __future__ import annotations

import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from typing import Any
from urllib.parse import urlsplit

from .agent_speedup_v2_plan import ALLOWED_UNTIL
from .agent_speedup_v2_user_check_plan import variant_plan_schema as plan_schema
from .agent_speedup_v2_types import SLICE_DO_KINDS


MAX_BODY_BYTES = 131_072
IMPLEMENTED_DO = SLICE_DO_KINDS
HTML_PATH = Path(__file__).with_name("agent_speedup_v2_user_check.html")


def review_schema() -> dict:
    """Use the actual v2 schema; no independent enum or array-limit copies."""
    return {
        "plan_schema": plan_schema(),
        "implemented_do": list(IMPLEMENTED_DO),
        "allowed_until": {kind: list(values) for kind, values in ALLOWED_UNTIL.items()},
    }


def _strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object keys are not accepted.")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("JSON numbers must be finite.")


class _BadRequest(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class ReviewServer:
    """One isolated review session.

    ``url`` contains a random bearer token in its fragment (never an HTTP request
    target or referrer). Keep this URL private. IPv4 loopback is mandatory; use
    SSH local forwarding for access from another machine.
    """

    def __init__(self, store, *, host: str = "127.0.0.1", port: int = 8765):
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("The review server must bind to 127.0.0.1 or localhost.")
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("port must be an integer in 0..65535.")
        self.store = store
        self.token = secrets.token_urlsafe(32)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._serving = threading.Event()
        # Read a single known asset, never a path received in a request.
        self._html = HTML_PATH.read_bytes()
        self._http = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self._http.daemon_threads = True
        self._http.block_on_close = False
        self.port = self._http.server_address[1]
        self._hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        self._origins = {f"http://{host}" for host in self._hosts}
        self.url = f"http://127.0.0.1:{self.port}/#token={self.token}"

    def _handler(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            # HTTP/1.0 closes every connection, avoiding unconsumed error bodies.
            protocol_version = "HTTP/1.0"
            server_version = "UserCheck"
            sys_version = ""

            def setup(self):
                super().setup()
                self.connection.settimeout(10)

            def log_message(self, _format, *_args):
                # Logs must not include Authorization, plan text, or URL secrets.
                return

            def _reply(self, status: int, data: Any, *, html: bool = False):
                nonce = secrets.token_urlsafe(24)
                if html:
                    body = data.replace(b"__CSP_NONCE__", nonce.encode("ascii"))
                    media = "text/html; charset=utf-8"
                else:
                    body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    media = "application/json; charset=utf-8"
                self.send_response(status)
                self.send_header("Content-Type", media)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; "
                    f"script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
                    "img-src data:; connect-src 'self'; "
                    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
                )
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _check_request(self, *, api: bool) -> str:
                hosts = self.headers.get_all("Host", [])
                if len(hosts) != 1 or hosts[0].lower() not in service._hosts:
                    raise _BadRequest(403, "Only the local review host is accepted.")
                origins = self.headers.get_all("Origin", [])
                if len(origins) > 1 or (origins and origins[0] not in service._origins):
                    raise _BadRequest(403, "The request origin is not this review UI.")
                if self.headers.get("Sec-Fetch-Site", "") in {"cross-site", "same-site"}:
                    raise _BadRequest(403, "Cross-origin review requests are not accepted.")
                target = urlsplit(self.path)
                if target.scheme or target.netloc or target.query or target.fragment:
                    raise _BadRequest(404, "Unknown review route.")
                if api:
                    values = self.headers.get_all("Authorization", [])
                    expected = f"Bearer {service.token}"
                    if len(values) != 1 or not hmac.compare_digest(
                        values[0].encode("utf-8"), expected.encode("utf-8")
                    ):
                        raise _BadRequest(401, "Open the private review URL from the command output.")
                return target.path

            def _body(self) -> dict:
                if self.headers.get_all("Transfer-Encoding"):
                    raise _BadRequest(400, "Transfer-Encoding is not supported.")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1:
                    raise _BadRequest(411, "Exactly one Content-Length is required.")
                try:
                    length = int(lengths[0])
                except ValueError:
                    raise _BadRequest(400, "Invalid Content-Length.") from None
                if length < 0:
                    raise _BadRequest(400, "Invalid Content-Length.")
                if length > MAX_BODY_BYTES:
                    raise _BadRequest(413, "Review request is too large.")
                types = self.headers.get_all("Content-Type", [])
                if len(types) != 1 or types[0].split(";", 1)[0].strip().lower() != "application/json":
                    raise _BadRequest(415, "Content-Type must be application/json.")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise _BadRequest(400, "Incomplete JSON body.")
                try:
                    value = json.loads(
                        raw.decode("utf-8"), object_pairs_hook=_strict_pairs, parse_constant=_reject_constant
                    )
                except (UnicodeDecodeError, ValueError, RecursionError):
                    raise _BadRequest(400, "The request must contain a valid JSON object.") from None
                if not isinstance(value, dict):
                    raise _BadRequest(400, "The request must contain a JSON object.")
                return value

            def _failure(self, exc: Exception):
                if isinstance(exc, _BadRequest):
                    self._reply(exc.status, {"error": str(exc)})
                    return
                if isinstance(exc, ValueError):
                    message = str(exc)
                    code = str(getattr(exc, "code", ""))
                    conflict = any(
                        word in (code + " " + message).lower()
                        for word in ("revision", "stale", "consumed", "already approved", "not editable")
                    )
                    result = {"error": message}
                    issues = getattr(exc, "issues", None)
                    if issues is not None:
                        result["issues"] = issues
                    self._reply(409 if conflict else 400, result)
                    return
                # Avoid exposing file paths, stack traces, or backend internals.
                self._reply(500, {"error": "Unable to read or save the review session."})

            def do_GET(self):
                try:
                    path = self._check_request(api=self.path != "/")
                    if path == "/":
                        self._reply(200, service._html, html=True)
                    elif path == "/api/schema":
                        self._reply(200, review_schema())
                    elif path == "/api/review":
                        with service._lock:
                            result = service.store.load()
                        self._reply(200, result)
                    else:
                        raise _BadRequest(404, "Unknown review route.")
                except Exception as exc:
                    self._failure(exc)

            def do_POST(self):
                try:
                    path = self._check_request(api=True)
                    if path not in {"/api/update", "/api/approve"}:
                        raise _BadRequest(404, "Unknown review route.")
                    body = self._body()
                    expected = {"expected_revision", "reviewed_steps"}
                    if path == "/api/update":
                        expected.add("plan")
                    if set(body) != expected:
                        raise _BadRequest(400, "Unexpected or missing review request fields.")
                    if type(body["expected_revision"]) is not int or body["expected_revision"] < 0:
                        raise _BadRequest(400, "expected_revision must be a nonnegative integer.")
                    reviewed = body["reviewed_steps"]
                    if (
                        not isinstance(reviewed, list)
                        or any(type(index) is not int or index < 0 for index in reviewed)
                        or len(set(reviewed)) != len(reviewed)
                    ):
                        raise _BadRequest(400, "reviewed_steps must be unique nonnegative integer indices.")
                    with service._lock:
                        if path == "/api/update":
                            if not isinstance(body["plan"], dict):
                                raise _BadRequest(400, "plan must be a JSON object.")
                            result = service.store.update(
                                body["plan"],
                                expected_revision=body["expected_revision"],
                                reviewed_steps=reviewed,
                            )
                        else:
                            result = service.store.approve(
                                expected_revision=body["expected_revision"], reviewed_steps=reviewed
                            )
                    self._reply(200, result)
                except Exception as exc:
                    self._failure(exc)

            def do_OPTIONS(self):
                self._reply(405, {"error": "This review server does not allow cross-origin access."})

            def do_PUT(self):
                self._reply(405, {"error": "Unsupported method."})

            do_DELETE = do_PUT
            do_PATCH = do_PUT
            do_HEAD = do_PUT

        return Handler

    def serve_forever(self):
        if self._closed:
            raise RuntimeError("Review server has closed.")
        self._serving.set()
        try:
            self._http.serve_forever(poll_interval=0.1)
        finally:
            self._serving.clear()

    def start_background(self):
        if self._closed:
            raise RuntimeError("Review server has closed.")
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self.serve_forever, name="v2-user-check-ui", daemon=True)
        self._thread.start()
        self._serving.wait(timeout=2)
        return self

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._serving.is_set():
            self._http.shutdown()
        self._http.server_close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)

    def __enter__(self):
        return self.start_background()

    def __exit__(self, *_args):
        self.close()


def serve(store, host="127.0.0.1", port=8765, ready_callback=None, stop_event=None):
    """Run a foreground service until Ctrl+C or ``stop_event`` is set.

    No approval watcher lives here: a standalone review server remains available
    after approval, and an integrated runner controls its own execution boundary.
    """
    with ReviewServer(store, host=host, port=port) as server:
        if ready_callback is not None:
            ready_callback(server.url)
        if stop_event is None:
            stop_event = threading.Event()
        try:
            while not stop_event.wait(0.25):
                pass
        except KeyboardInterrupt:
            pass
