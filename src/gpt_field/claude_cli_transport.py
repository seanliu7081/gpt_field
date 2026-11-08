"""Claude backend for the UC7.1 executive: one OpenAI-Responses payload -> one `claude -p` call.

The UC7.1 clients build OpenAI Responses payloads (instructions, input items with
input_text/input_image, a strict json_schema output format and a reasoning effort)
and expect a completed Responses-shaped dict back. This transport keeps every
prompt, schema, image and validator unchanged and swaps only the model call.

Calls go through the locally installed Claude Code CLI in print mode, so they use
the user's own Claude login (no API key). The CLI runs with --safe-mode (no hooks,
CLAUDE.md, plugins or MCP), no tools, no session persistence, the payload
instructions as the entire system prompt and the payload schema as --json-schema.

Differences from the GPT transport that a reader of the results must know:
- The CLI accepts one user turn, so earlier turns (observations, retained images,
  and the model's own accepted JSON decisions) are replayed in chronological order
  inside that turn under explicit labels. GPT's opaque encrypted reasoning items
  cannot be replayed to Claude and are dropped (Claude never produces them).
- Structured output is validated by the CLI against the same schema; the
  harness's own validators still run afterwards, unchanged.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
import uuid

from .astra import APIError

TRANSPORT_VERSION = "claude-cli-responses-adapter-1"
DEFAULT_MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _data_url_block(url: str) -> dict:
    if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
        raise ValueError("Only inline base64 data-URL images are supported")
    header, data = url.split(",", 1)
    media_type = header[len("data:"):].split(";", 1)[0]
    if media_type not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        raise ValueError(f"Unsupported image media type {media_type!r}")
    base64.b64decode(data[:64] + "=" * (-len(data[:64]) % 4))  # cheap sanity check
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _single_line_strings(value, counter: list):
    """GPT's strict decoder emits single-line strings; Claude often writes markdown
    lists with newlines. The harness rejects control/zero-width characters when it
    echoes model text into later requests, so: control chars -> space, other
    harness-unsafe code points dropped, whitespace runs collapsed. Counted."""
    from .agent_speedup_v2_plan import unsafe_characters
    import unicodedata
    if isinstance(value, dict):
        return {key: _single_line_strings(item, counter) for key, item in value.items()}
    if isinstance(value, list):
        return [_single_line_strings(item, counter) for item in value]
    if isinstance(value, str) and (unsafe_characters(value) or "  " in value.strip()):
        cleaned = "".join(" " if unicodedata.category(ch) == "Cc" else ch for ch in value)
        cleaned = "".join(ch for ch in cleaned if not unsafe_characters(ch))
        cleaned = " ".join(cleaned.split())
        if cleaned != value:
            counter.append(1)
        return cleaned or value
    return value


def _user_parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    blocks = []
    for part in content:
        kind = part.get("type")
        if kind == "input_text":
            blocks.append({"type": "text", "text": part["text"]})
        elif kind == "input_image":
            blocks.append(_data_url_block(part["image_url"]))
        else:
            raise ValueError(f"Unsupported user content part {kind!r}")
    return blocks


def _assistant_text(item) -> str:
    texts = [part.get("text", "") for part in item.get("content", [])
             if isinstance(part, dict) and part.get("type") == "output_text"]
    return "\n".join(texts)


def responses_input_to_claude_content(items: list) -> tuple[list[dict], dict]:
    """Flatten Responses input items into one Claude user turn, oldest first."""
    turns = []  # ("user", blocks) | ("assistant", text)
    dropped_reasoning = 0
    for item in items:
        if item.get("type") == "reasoning":
            dropped_reasoning += 1
            continue
        role = item.get("role")
        if role == "user":
            turns.append(("user", _user_parts(item.get("content", []))))
        elif role == "assistant" or item.get("type") == "message":
            turns.append(("assistant", _assistant_text(item)))
        else:
            raise ValueError(f"Unsupported input item {item.get('type')!r}/{role!r}")
    if not turns or turns[-1][0] != "user":
        raise ValueError("The final input item must be the current user observation")
    *history, (_, current) = turns
    blocks: list[dict] = []
    if history:
        blocks.append({"type": "text", "text": (
            "EPISODE HISTORY (chronological). Earlier observations sent to you and the JSON "
            "decisions you returned for them follow, oldest first. Older images may have been "
            "dropped to bound context; text is complete.")})
        obs = 0
        for role, value in history:
            if role == "user":
                obs += 1
                blocks.append({"type": "text", "text": f"--- Earlier observation #{obs} ---"})
                blocks.extend(value)
            else:
                blocks.append({"type": "text", "text": f"--- Your accepted JSON decision for observation #{obs} ---\n{value}"})
        blocks.append({"type": "text", "text": (
            "--- CURRENT observation (decide the next action for THIS one) ---")})
    blocks.extend(current)
    meta = {"history_turns": len(history), "dropped_gpt_reasoning_items": dropped_reasoning,
            "images": sum(b["type"] == "image" for b in blocks)}
    return blocks, meta


class ClaudeCliTransport:
    """Drop-in replacement for the subscription transport callable."""

    def __init__(self, *, model: str = DEFAULT_MODEL, effort: str | None = None,
                 claude_bin: str = "claude", request_path=None, timeout_floor_s: float = 600.0):
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}")
        self.model, self.effort, self.claude_bin = model, effort, claude_bin
        self.request_path, self.timeout_floor_s = request_path, timeout_floor_s
        self.last_diagnostics = None
        self.calls = []

    def describe(self) -> dict:
        return {"transport_version": TRANSPORT_VERSION, "backend": "claude_code_cli_print_mode",
                "model": self.model, "effort_override": self.effort,
                "auth": "user's Claude login via local Claude Code CLI; no API key",
                "cli_flags": ["--safe-mode", "--tools ''", "--no-session-persistence",
                              "--system-prompt <payload.instructions>", "--json-schema <payload schema>"]}

    def __call__(self, endpoint, payload, headers, timeout_s):
        fmt = (payload.get("text") or {}).get("format") or {}
        if fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
            raise ValueError("Claude transport requires a json_schema output format")
        effort = self.effort or (payload.get("reasoning") or {}).get("effort") or "high"
        if effort not in EFFORTS:  # e.g. "minimal" has no Claude equivalent
            effort = "low"
        blocks, meta = responses_input_to_claude_content(payload["input"])
        line = json.dumps({"type": "user", "session_id": "",
                           "message": {"role": "user", "content": blocks},
                           "parent_tool_use_id": None})
        cmd = [self.claude_bin, "-p", "--safe-mode", "--tools", "", "--no-session-persistence",
               "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
               "--model", self.model, "--effort", effort,
               "--system-prompt", payload["instructions"],
               "--json-schema", json.dumps(fmt["schema"], separators=(",", ":"))]
        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CLAUDECODE")}
        started = time.monotonic()
        deadline = max(float(timeout_s or 0), self.timeout_floor_s)
        try:
            # Empty throwaway working directory so the CLI sees no project files; removed after the call.
            with tempfile.TemporaryDirectory(prefix="claude_cli_transport_") as workdir:
                proc = subprocess.run(cmd, input=line + "\n", capture_output=True, text=True,
                                      timeout=deadline, cwd=workdir, env=env)
        except subprocess.TimeoutExpired:
            err = APIError(f"Claude CLI exceeded {deadline:.0f}s; no output executed")
            err.error_code = "subscription_stream_interrupted"
            raise err from None
        wall = time.monotonic() - started
        result, assistant_models = None, set()
        for raw in proc.stdout.splitlines():
            try:
                event = json.loads(raw)
            except ValueError:
                continue
            if event.get("type") == "result":
                result = event
            elif event.get("type") == "assistant":
                model = (event.get("message") or {}).get("model")
                if model:
                    assistant_models.add(model)
        diag = {"wall_s": round(wall, 3), "returncode": proc.returncode, **meta, "effort": effort,
                "requested_model": self.model}
        self.last_diagnostics = diag
        if result is None or result.get("is_error") or "structured_output" not in result:
            api_error = (result or {}).get("api_error") or (result or {}).get("subtype") or "no_result"
            diag["error"] = api_error
            self.calls.append(diag)
            err = APIError(f"Claude CLI call failed ({api_error}); stderr tail: {proc.stderr[-300:]!r}")
            err.error_code = ("server_is_overloaded" if "overload" in str(api_error)
                              else "subscription_network_error" if "connect" in str(api_error)
                              else "claude_cli_error")
            raise err
        structured = result["structured_output"]
        try:  # prefer the model's own key order when it matches the validated object
            raw = json.loads(result.get("result"))
            if raw == structured:
                structured = raw
        except (TypeError, ValueError):
            pass
        normalized: list = []
        structured = _single_line_strings(structured, normalized)
        diag["strings_normalized"] = len(normalized)
        text = json.dumps(structured, ensure_ascii=False, separators=(",", ":"))
        usage = result.get("usage") or {}
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_write = int(usage.get("cache_creation_input_tokens") or 0)
        input_tokens = int(usage.get("input_tokens") or 0) + cache_read + cache_write
        thinking = int(((usage.get("output_tokens_details") or {}).get("thinking_tokens")) or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        models = sorted(set(result.get("modelUsage", {})) | assistant_models)
        diag.update(duration_ms=result.get("duration_ms"), duration_api_ms=result.get("duration_api_ms"),
                    ttft_ms=result.get("ttft_ms"), num_turns=result.get("num_turns"),
                    input_tokens=input_tokens, cache_read_tokens=cache_read, output_tokens=output_tokens,
                    thinking_tokens=thinking, models=models,
                    list_cost_usd_estimate=result.get("total_cost_usd"))
        self.calls.append(diag)
        return {
            "id": f"claude_{uuid.uuid4().hex}", "object": "response", "status": "completed",
            "error": None, "incomplete_details": None,
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                      "input_tokens_details": {"cached_tokens": cache_read},
                      "output_tokens_details": {"reasoning_tokens": min(thinking, output_tokens)}},
            "_claude_transport": {**self.describe(), **diag},
        }
