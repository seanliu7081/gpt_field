"""Offline tests for the Claude CLI transport (a fake `claude` binary; no network)."""
import importlib.util
import json
import stat
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from gpt_field.astra import APIError
from gpt_field.claude_cli_transport import ClaudeCliTransport, responses_input_to_claude_content

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
SCHEMA = {"type": "object", "properties": {"belief": {"type": "string"}, "n": {"type": "integer"}},
          "required": ["belief", "n"], "additionalProperties": False}


def _payload(history=()):
    current = {"role": "user", "content": [{"type": "input_text", "text": "obs now"},
                                           {"type": "input_image", "detail": "high", "image_url": PNG}]}
    return {"model": "gpt-6-astra", "instructions": "SYSTEM", "input": [*history, current],
            "text": {"format": {"type": "json_schema", "name": "x", "strict": True, "schema": SCHEMA}},
            "reasoning": {"effort": "high"}, "stream": True, "store": False}


def _fake_cli(tmp_path, result):
    script = tmp_path / "claude"
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import json, sys
        argv = sys.argv[1:]
        line = json.loads(sys.stdin.readline())
        json.dump({{"argv": argv, "message": line}}, open({str(tmp_path / 'seen.json')!r}, "w"))
        print(json.dumps({{"type": "assistant", "message": {{"model": "claude-opus-5-5"}}}}))
        print({json.dumps(json.dumps(result))})
        """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_flattens_history_and_drops_gpt_reasoning():
    history = [
        {"role": "user", "content": [{"type": "input_text", "text": "obs 1"},
                                     {"type": "input_image", "image_url": PNG}]},
        {"type": "reasoning", "encrypted_content": "opaque"},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": '{"n":1}'}]},
    ]
    blocks, meta = responses_input_to_claude_content(_payload(history)["input"])
    texts = [b["text"] for b in blocks if b["type"] == "text"]
    assert meta == {"history_turns": 2, "dropped_gpt_reasoning_items": 1, "images": 2}
    assert texts[-1] == "obs now" and any('{"n":1}' in t for t in texts)
    assert [b["source"]["media_type"] for b in blocks if b["type"] == "image"] == ["image/png"] * 2


def test_round_trip_normalizes_newlines_and_reports_usage(tmp_path):
    result = {"type": "result", "is_error": False, "num_turns": 2, "duration_ms": 10,
              "result": '{"belief":"a\\n- b","n":3}', "structured_output": {"belief": "a\n- b", "n": 3},
              "usage": {"input_tokens": 5, "cache_read_input_tokens": 7, "cache_creation_input_tokens": 0,
                        "output_tokens": 20, "output_tokens_details": {"thinking_tokens": 12}},
              "modelUsage": {"claude-opus-5-5": {}}}
    transport = ClaudeCliTransport(claude_bin=_fake_cli(tmp_path, result))
    response = transport("ignored", _payload(), {}, 5)
    assert json.loads(response["output"][0]["content"][0]["text"]) == {"belief": "a - b", "n": 3}
    assert response["usage"]["input_tokens"] == 12 and response["usage"]["output_tokens"] == 20
    assert "model" not in response and response["status"] == "completed"
    seen = json.loads((tmp_path / "seen.json").read_text())
    argv = seen["argv"]
    assert argv[argv.index("--system-prompt") + 1] == "SYSTEM"
    assert json.loads(argv[argv.index("--json-schema") + 1]) == SCHEMA
    assert argv[argv.index("--effort") + 1] == "high" and "--safe-mode" in argv
    assert transport.last_diagnostics["strings_normalized"] == 1


def test_cli_error_raises_without_output(tmp_path):
    result = {"type": "result", "is_error": True, "api_error": "connection_failed"}
    transport = ClaudeCliTransport(claude_bin=_fake_cli(tmp_path, result))
    with pytest.raises(APIError) as info:
        transport("ignored", _payload(), {}, 5)
    assert info.value.error_code == "subscription_network_error"


def test_call_leaves_no_temp_directory(tmp_path, monkeypatch):
    result = {"type": "result", "is_error": False, "result": '{"belief":"a","n":1}',
              "structured_output": {"belief": "a", "n": 1}, "usage": {}}
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    transport = ClaudeCliTransport(claude_bin=_fake_cli(tmp_path, result))
    transport("ignored", _payload(), {}, 5)
    assert list(scratch.iterdir()) == []


def _wrapper():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_uc71_claude.py"
    spec = importlib.util.spec_from_file_location("evaluate_uc71_claude_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapper_labels_only_run_summaries():
    wrapper = _wrapper()
    label = wrapper.backend_label("claude-opus-5-5", "low")
    stock = {"billing": "chatgpt_subscription", "reasoning_effort": "high", "status": "success"}
    summary = wrapper.label_summary("out/trial/summary.json", stock, label)
    assert summary["model_backend"]["model"] == "claude-opus-5-5"
    assert summary["model_backend"]["effort"] == "low"
    assert summary["billing"] == wrapper.BILLING
    assert summary["reasoning_effort"] == "high" and summary["status"] == "success"
    assert stock["billing"] == "chatgpt_subscription"  # input not mutated
    assert wrapper.label_summary("out/trial/manifest.json", stock, label) is stock
