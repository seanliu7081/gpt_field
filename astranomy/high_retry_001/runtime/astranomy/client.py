"""Persistent Codex app-server client. No inference occurs at import/preflight.

Robot actions execute only after a completed turn containing exactly one valid
proposal. The dynamic-tool acknowledgement is deliberately not motion feedback.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib

from gpt_field.astra import AstraConfig
from gpt_field.astranomy_agentic_client import (
    AstranomyAgenticClient, validate_command, validate_feedback,
)
from .policy import MODEL, EFFORT, PROVIDER, TASK, config_overrides, dynamic_tools, policy_instructions
from .transport import PolicyGateway


class ProtocolError(RuntimeError):
    pass


def _toml(value):
    # JSON scalar/array strings are TOML-compatible; only the empty table is used.
    return "{}" if value == {} else json.dumps(value, ensure_ascii=True)


def server_command(binary, *, base_url="https://api.openai.com/v1", log_dir=None):
    settings = config_overrides()
    settings["model_catalog_json"] = str(Path(__file__).with_name("model_catalog.json"))
    settings.update({f"model_providers.{PROVIDER}.{key}": value for key, value in {
        "name": "Astranomy ChatGPT subscription", "base_url": base_url,
        "env_key": "ASTRANOMY_SUBSCRIPTION_TOKEN", "wire_api": "responses",
        "requires_openai_auth": False, "supports_websockets": False,
        "request_max_retries": 0, "stream_max_retries": 0,
        "stream_idle_timeout_ms": 180000,
    }.items()})
    # CLI table overrides merge with config tables. Disable configured MCP
    # servers individually too, rather than assuming {} deletes existing ones.
    config_file = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "config.toml"
    if config_file.is_file():
        config = tomllib.loads(config_file.read_text())
        for name in config.get("mcp_servers", {}):
            settings[f"mcp_servers.{json.dumps(name)}.enabled"] = False
    if log_dir is not None:
        settings["log_dir"] = str(Path(log_dir).resolve())
    command = [str(binary), "app-server", "--listen", "stdio://"]
    for key, value in settings.items():
        command.extend(["-c", key + "=" + _toml(value)])
    return command


class StdioRPC:
    """Small synchronous RPC driver, with a reader thread so signals remain live."""

    def __init__(self, command, *, token, log_dir, check_stop=lambda: None):
        self.check_stop = check_stop
        self.counter = 0
        self.messages = queue.Queue()
        self.events = []
        self.closed = False
        self.token = token
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.wire = (self.log_dir / "rpc.jsonl").open("a")
        # Use the existing home for Codex's own thread store; never alter its
        # configuration/auth files or repurpose HOME/CODEX_HOME.
        self.scratch = tempfile.TemporaryDirectory(prefix="astranomy-policy-")
        command = list(command)
        key = f"model_providers.{PROVIDER}.base_url="
        location = next(i for i, part in enumerate(command) if part.startswith(key))
        upstream = json.loads(command[location][len(key):])
        self.gateway = PolicyGateway(upstream=upstream, token=token,
                                     log_dir=self.log_dir / f"wire_{time.time_ns()}")
        command[location] = key + json.dumps(self.gateway.url)
        env = {k: v for k, v in os.environ.items()
               if k in {"PATH", "HOME", "CODEX_HOME", "LANG", "LC_ALL", "TZ",
                        "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY"}}
        env["ASTRANOMY_SUBSCRIPTION_TOKEN"] = self.gateway.capability
        # Nothing inherits OPENAI_API_KEY, cloud credentials, simulator state,
        # parent session identifiers, or arbitrary Codex feature overrides.
        try:
            self.process = subprocess.Popen(command, cwd=self.scratch.name, env=env,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1, start_new_session=True)
        except BaseException:
            self.gateway.close()
            self.wire.close()
            self.scratch.cleanup()
            raise
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.stderr_reader = threading.Thread(target=self._stderr, daemon=True)
        self.reader.start()
        self.stderr_reader.start()

    def _record(self, direction, message):
        raw = json.dumps({"direction": direction, "message": message}, allow_nan=False)
        self.wire.write(raw.replace(self.token, "[REDACTED]") + "\n")
        self.wire.flush()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.messages.put(json.loads(line))
        except Exception as exc:
            self.messages.put(ProtocolError(f"app-server stream failed: {type(exc).__name__}"))
        finally:
            self.messages.put(ProtocolError("app-server exited"))

    def _stderr(self):
        with (self.log_dir / "app_server.stderr.log").open("a") as stream:
            for line in self.process.stderr:
                stream.write(line.replace(self.token, "[REDACTED]"))
                stream.flush()

    def send(self, message):
        self._record("send", message)
        self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
        self.process.stdin.flush()

    def receive(self, *, timeout=240):
        deadline = time.monotonic() + timeout
        while True:
            self.check_stop()
            try:
                message = self.messages.get(timeout=min(.25, max(.001, deadline - time.monotonic())))
                if isinstance(message, Exception):
                    raise message
                self._record("receive", message)
                return message
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise ProtocolError("app-server RPC/response inactivity timeout; episode incomplete")

    def call(self, method, params):
        self.counter += 1
        ident = self.counter
        self.send({"id": ident, "method": method, "params": params})
        while True:
            message = self.receive()
            if message.get("id") == ident and "method" not in message:
                if "error" in message:
                    raise ProtocolError(f"{method} failed: {message['error']}")
                return message["result"]
            self.events.append(message)

    def next_event(self):
        return self.events.pop(0) if self.events else self.receive()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            self.reader.join(timeout=2)
            self.stderr_reader.join(timeout=2)
            self.wire.close()
            self.gateway.close()
            self.scratch.cleanup()


class CodexPolicyClient:
    def __init__(self, config=None, *, token_provider, task_key=TASK, log_dir,
                 rpc_factory=StdioRPC, binary=None, check_stop=lambda: None):
        config = config or AstraConfig(model=MODEL, max_api_calls=None, max_wall_time_s=None, max_cost_usd=None)
        if task_key != TASK or config.model != MODEL:
            raise ValueError("This isolated retry is pinned to drawer_bowl and gpt-6-astra")
        # Reuse existing strict sensor-schema validation, never its transport.
        self.validator = AstranomyAgenticClient(config, token_provider=token_provider, task_key=TASK)
        self.token_provider, self.rpc_factory = token_provider, rpc_factory
        self.binary = binary or shutil.which("codex")
        if not self.binary:
            raise FileNotFoundError("codex is not on PATH")
        self.log_dir = Path(log_dir)
        self.check_stop = check_stop
        self.rpc = None
        self.thread_id = None
        self.turns = self.proposals = self.token_rotations = 0
        self.pending_feedback = None
        self.awaiting_result = False
        self._used_access_tokens = set()
        self.current_token = None
        self.usage = None
        self.compactions = 0
        self.previous_requests = 0

    def _connect(self):
        token = self.token_provider()
        if not isinstance(token, str) or not token or token.startswith("sk-"):
            raise ValueError("A subscription OAuth token is required; API keys are refused")
        if self.rpc and token == self.current_token:
            return
        self._used_access_tokens.add(token)
        if self.rpc:
            self.previous_requests += getattr(getattr(self.rpc, "gateway", None), "request_count", 0)
            self.rpc.close()
            self.token_rotations += 1
        self.rpc = self.rpc_factory(server_command(self.binary, log_dir=self.log_dir / "codex_logs"),
            token=token, log_dir=self.log_dir, check_stop=self.check_stop)
        self.current_token = token
        self.rpc.call("initialize", {"clientInfo": {"name": "gpt_field_astranomy", "title": "Astranomy Robot Policy", "version": "1.0"},
                                    "capabilities": {"experimentalApi": True}})
        self.rpc.send({"method": "initialized", "params": {}})
        params = {"model": MODEL, "modelProvider": PROVIDER, "approvalPolicy": "never",
                  "sandbox": "read-only", "baseInstructions": policy_instructions(),
                  "developerInstructions": "", "config": config_overrides()}
        if self.thread_id:
            params["threadId"] = self.thread_id
            result = self.rpc.call("thread/resume", params)
        else:
            params.update(dynamicTools=dynamic_tools(), environments=[],
                          allowProviderModelFallback=False, experimentalRawEvents=False)
            result = self.rpc.call("thread/start", params)
        if (result.get("model") != MODEL or result.get("modelProvider") != PROVIDER
                or result.get("reasoningEffort") != EFFORT):
            raise ProtocolError("Codex changed the requested model, provider, or reasoning effort")
        ident = result["thread"]["id"]
        if self.thread_id and ident != self.thread_id:
            raise ProtocolError("Token renewal changed the persistent thread")
        if result.get("instructionSources"):
            raise ProtocolError("Unexpected repository/skill instructions in robot thread")
        self.thread_id = ident

    def predict(self, request):
        if self.awaiting_result:
            raise ProtocolError("Record actual execution feedback before another observation")
        payload = self.validator.build_payload(request)
        content = payload["input"][-1]["content"]
        inputs = []
        if self.pending_feedback is not None:
            inputs.append({"type": "text", "text": json.dumps({"type": "actual_execution_feedback", **self.pending_feedback})})
        for item in content:
            if item["type"] == "input_text":
                inputs.append({"type": "text", "text": item["text"]})
            else:
                inputs.append({"type": "image", "url": item["image_url"], "detail": "high"})
        self._connect()
        self.check_stop()
        self.turns += 1
        started = self.rpc.call("turn/start", {"threadId": self.thread_id, "input": inputs,
            "model": MODEL, "effort": EFFORT, "environments": []})
        turn_id = started["turn"]["id"]
        proposal = None
        while True:
            event = self.rpc.next_event()
            method, params = event.get("method", ""), event.get("params", {})
            if method == "thread/tokenUsage/updated":
                self.usage = params.get("tokenUsage")
            if method == "item/tool/call":
                if (params.get("threadId") != self.thread_id or params.get("turnId") != turn_id
                        or params.get("tool") != "execute_eef_target" or params.get("namespace") != "robot"
                        or proposal is not None):
                    self.rpc.send({"id": event["id"], "result": {"success": False, "contentItems": [
                        {"type": "inputText", "text": "Invalid/duplicate proposal; no robot action executed."}]}})
                    raise ProtocolError("Unexpected or duplicate policy tool; entire turn discarded")
                proposal = validate_command({"name": params["tool"], "arguments": params["arguments"]})
                self.rpc.send({"id": event["id"], "result": {"success": True, "contentItems": [
                    {"type": "inputText", "text": "Proposal recorded. No robot action has executed. End this turn; actual feedback follows in the next observation."}]}})
            elif "id" in event and "method" in event:
                self.rpc.send({"id": event["id"], "error": {"code": -32601, "message": "Optional tools/approvals are disabled for robot policy"}})
                raise ProtocolError("Unexpected server request; no action executed")
            elif method in ("thread/compacted", "thread/contextCompacted") or (
                    method == "item/started" and params.get("item", {}).get("type") == "contextCompaction"):
                self.compactions += 1
                raise ProtocolError("History compaction changed full-history protocol; no action executed")
            elif method == "item/started" and params.get("item", {}).get("type") in (
                    "commandExecution", "fileChange", "mcpToolCall", "webSearch", "imageGeneration", "collabAgentToolCall"):
                raise ProtocolError("Optional native tool attempted; no robot action executed")
            elif method == "turn/completed" and params.get("turn", {}).get("id") == turn_id:
                if params["turn"].get("status") != "completed" or proposal is None:
                    raise ProtocolError("Turn failed, interrupted, or supplied no target; no action executed")
                self.awaiting_result = True
                self.pending_feedback = None
                self.proposals += 1
                return copy.deepcopy(proposal)

    def record_result(self, feedback):
        if not self.awaiting_result:
            raise ProtocolError("No completed proposal awaits execution feedback")
        self.pending_feedback = validate_feedback(feedback)
        self.awaiting_result = False

    def budget_snapshot(self):
        gateway = getattr(self.rpc, "gateway", None)
        count = self.previous_requests + gateway.request_count if gateway is not None else None
        return {"billing": "chatgpt_subscription", "api_key_calls": 0,
                "api_calls": count, "subscription_inference_calls": count,
                "inference_request_count_available": count is not None,
                "counting_note": "Counted at the provider gateway; Codex turns are reported separately",
                "codex_turns_started": self.turns, "accepted_proposals": self.proposals,
                "thread_id": self.thread_id, "token_usage": self.usage,
                "token_rotations": self.token_rotations, "compactions": self.compactions}

    def close(self):
        if self.rpc:
            self.rpc.close()
