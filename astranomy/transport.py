"""Loopback gateway enforcing the robot-only tool surface for native Codex.

Some Codex builds always advertise skill tools. Filter tool declarations at the
provider boundary and reject unapproved returned calls before Codex sees them.
Conversation messages/images and model reasoning settings are not summarized.
"""
from __future__ import annotations

import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
import urllib.error
import urllib.request

from gpt_field.astranomy_agentic_client import tool_definitions
from .policy import MODEL, EFFORT


def robot_only_payload(payload):
    if payload.get("model") != MODEL or payload.get("reasoning", {}).get("effort") != EFFORT:
        raise ValueError("Requested model/effort differs from the approved policy")
    if payload.get("store") is not False or payload.get("stream") is not True:
        raise ValueError("Expected non-stored streaming subscription inference")
    result = copy.deepcopy(payload)
    result["input"] = [item for item in result["input"] if item.get("type") != "additional_tools"]
    tools = tool_definitions()
    tools[0]["description"] = "Propose one complete absolute EEF target held for 1-5 steps. No motion occurs until the host validates the completed turn."
    result["tools"] = [{"type": "namespace", "name": "robot",
        "description": "Propose one short EEF target; host execution occurs after the completed turn.",
        "tools": tools}]
    result["tool_choice"] = "auto"  # Codex must finish the turn after acknowledgement.
    result["parallel_tool_calls"] = False
    return result


def validate_output_item(item):
    kind = item.get("type")
    if kind == "function_call":
        if item.get("namespace") != "robot" or item.get("name") != "execute_eef_target":
            raise ValueError("Non-robot function call rejected before reaching Codex")
    elif kind not in ("message", "reasoning"):
        raise ValueError("Non-policy output item rejected before reaching Codex")


def validate_completed_response(response):
    if "model" in response and response["model"] != MODEL:
        raise ValueError("Response changed the requested model")
    reasoning = response.get("reasoning") or {}
    if reasoning.get("effort") not in (None, EFFORT):
        raise ValueError("Response changed the requested reasoning effort")
    for item in response.get("output", []):
        validate_output_item(item)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class PolicyGateway:
    def __init__(self, *, upstream, token, log_dir):
        if upstream != "https://api.openai.com/v1" and not upstream.startswith("http://127.0.0.1:"):
            raise ValueError("Only the official subscription route or loopback test provider is supported")
        self.token = token
        self.capability = secrets.token_urlsafe(32)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.request_count = 0
        self.active = []
        self.lock = threading.Lock()
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                # Local catalog only; this path never contacts a remote service.
                body = Path(__file__).with_name("model_catalog.json").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                if self.path != "/v1/responses" or self.headers.get("Authorization") != "Bearer " + gateway.capability:
                    self.send_error(403)
                    return
                sent_headers = False
                response = None
                started = time.monotonic()
                record = {"forwarded": False}
                try:
                    if self.headers.get("Content-Encoding"):
                        raise ValueError("Compressed requests are disabled for the audited transport")
                    payload = robot_only_payload(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                    with gateway.lock:
                        gateway.request_count += 1
                        index = gateway.request_count
                    prefix = gateway.log_dir / f"request_{index:05d}"
                    prefix.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=True))
                    record["request"] = index
                    headers = {"Authorization": "Bearer " + gateway.token, "Content-Type": "application/json",
                               "Accept": "text/event-stream", "User-Agent": "gpt-field-astranomy/1"}
                    for key in ("originator", "session_id", "version", "x-codex-turn-metadata"):
                        if self.headers.get(key):
                            headers[key] = self.headers[key]
                    request = urllib.request.Request(upstream + "/responses", data=json.dumps(payload).encode(), headers=headers)
                    record["forwarded"] = True
                    response = urllib.request.build_opener(NoRedirect()).open(request, timeout=180)
                    with gateway.lock:
                        gateway.active.append(response)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    sent_headers = True
                    with prefix.with_suffix(".sse").open("wb") as archive:
                        while True:
                            frame = []
                            while True:
                                line = response.readline()
                                if not line:
                                    break
                                frame.append(line)
                                if line in (b"\n", b"\r\n"):
                                    break
                            if not frame:
                                break
                            raw = b"".join(frame)
                            data = b"\n".join(line[5:].strip() for line in frame if line.startswith(b"data:"))
                            if data and data != b"[DONE]":
                                event = json.loads(data)
                                if event.get("type") in ("response.output_item.added", "response.output_item.done"):
                                    validate_output_item(event["item"])
                                if event.get("type") == "response.completed":
                                    validate_completed_response(event.get("response", {}))
                            archive.write(raw)
                            self.wfile.write(raw)
                            self.wfile.flush()
                    record["stream_finished"] = True
                except Exception as exc:
                    # Never expose an OAuth token, arbitrary upstream body, or URL.
                    record.update(error_type=type(exc).__name__, http_status=getattr(exc, "code", None))
                    if not sent_headers:
                        self.send_error(502, "Subscription transport or robot-policy protocol failed")
                    self.close_connection = True
                finally:
                    if response is not None:
                        response.close()
                        with gateway.lock:
                            if response in gateway.active:
                                gateway.active.remove(response)
                    record["elapsed_s"] = time.monotonic() - started
                    with gateway.lock:
                        with (gateway.log_dir / "transport.jsonl").open("a") as stream:
                            stream.write(json.dumps(record) + "\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        with self.lock:
            active = list(self.active)
        for response in active:
            response.close()
        self.server.server_close()
        self.thread.join(timeout=2)
