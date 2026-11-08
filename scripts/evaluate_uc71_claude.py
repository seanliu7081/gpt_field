"""Run the unchanged UC7.1 runner with Claude (via the local Claude Code CLI) as the model.

Everything except the model call is the stock runner
(scripts/evaluate_agent_speedup_v2_user_check.py): prompts, schemas, validators,
simulator, local controller, evaluator, timing and artifacts. This wrapper only:
  * installs ClaudeCliTransport as every client's transport (plan + executive calls),
  * supplies a placeholder in place of the ChatGPT OAuth token (no OpenAI call is made),
  * relabels the "billing" field in logs/budgets/summaries so artifacts do not claim ChatGPT,
  * adds a top-level "model_backend" block to summary.json / runner_summary.json,
  * writes <output>/claude_backend.json with per-call Claude diagnostics.

The config/manifests still carry the runner's fixed model name ("gpt-6-astra") and
its "reasoning_effort" because the stock validators require them. They describe the
stock GPT configuration, not the model that answered: read summary["model_backend"],
claude_backend.json, or each *.response.json "_claude_transport" for that.

Usage (same subcommands/flags as the stock runner, plus):
  --claude-model claude-opus-5-5   --claude-effort high
Stream-progress / request-deadline options are OpenAI-SSE-only and are rejected.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpt_field import astra, chatgpt_auth, chatgpt_plan  # noqa: E402
from gpt_field.claude_cli_transport import ClaudeCliTransport, DEFAULT_MODEL  # noqa: E402

BILLING = "claude_subscription_via_claude_code_cli"
PLACEHOLDER_TOKEN = "claude-cli-transport-placeholder-no-openai-call"


def _load_runner():
    path = ROOT / "scripts/evaluate_agent_speedup_v2_user_check.py"
    spec = importlib.util.spec_from_file_location("uc71_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["uc71_runner"] = module
    spec.loader.exec_module(module)
    return module


SUMMARY_FILES = ("summary.json", "runner_summary.json")


def backend_label(model, effort):
    return {
        "backend": "claude_code_cli", "model": model,
        "effort": effort or "same as reasoning_effort (mapped to the nearest Claude level)",
        "billing": BILLING,
        "note": ("Top-level model/reasoning_effort fields and configuration blocks are the stock "
                 "GPT settings kept for validator compatibility; Claude answered every model call. "
                 "Per-call details: claude_backend.json."),
    }


def label_summary(path, value, label):
    """Mark run summaries with the backend that actually answered. Other files unchanged."""
    if Path(path).name not in SUMMARY_FILES or not isinstance(value, dict):
        return value
    value = {**value, "model_backend": label}
    if value.get("billing") == "chatgpt_subscription":
        value["billing"] = BILLING
    return value


def _pop_option(argv, name, default):
    if name in argv:
        i = argv.index(name)
        value = argv[i + 1]
        del argv[i:i + 2]
        return value
    return default


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    model = _pop_option(argv, "--claude-model", DEFAULT_MODEL)
    effort = _pop_option(argv, "--claude-effort", None)
    if any(flag in argv for flag in ("--stream-progress", "--request-deadline-s")):
        raise SystemExit("--stream-progress/--request-deadline-s apply to the OpenAI SSE transport only")
    output = Path(argv[argv.index("--output") + 1]).resolve() if "--output" in argv else None
    review = Path(argv[argv.index("--review-dir") + 1]).resolve() if "--review-dir" in argv else None

    runner = _load_runner()
    transports = []
    original_configure = runner.configure_client_transport

    def configure(client, *, request_deadline_s=None, stream_progress=False):
        settings = original_configure(client, request_deadline_s=None, stream_progress=False)
        transport = ClaudeCliTransport(model=model, effort=effort)
        client.transport = transport
        transports.append((type(client).__qualname__, transport))
        return {**settings, "model_backend": transport.describe()}

    runner.configure_client_transport = configure

    class PlaceholderCredentials:
        def __init__(self, *args, **kwargs):
            pass

        def access_token(self):
            return PLACEHOLDER_TOKEN

    chatgpt_auth.SubscriptionCredentials = PlaceholderCredentials

    original_write = astra.AstraClient._write_log

    def write_log(self, path, data):
        if isinstance(data, dict) and data.get("billing") == "chatgpt_subscription":
            data = {**data, "billing": BILLING, "model_backend": model}
        return original_write(self, path, data)

    astra.AstraClient._write_log = write_log

    label = backend_label(model, effort)
    original_write_json = runner.base.write_json

    def write_json(path, value):
        return original_write_json(path, label_summary(path, value, label))

    runner.base.write_json = write_json
    for cls in vars(chatgpt_plan).values():
        if isinstance(cls, type) and "budget_snapshot" in vars(cls):
            def snapshot(self, _orig=vars(cls)["budget_snapshot"]):
                value = _orig(self)
                if isinstance(value, dict) and value.get("billing") == "chatgpt_subscription":
                    value = {**value, "billing": BILLING, "model_backend": model}
                return value
            cls.budget_snapshot = snapshot

    try:
        return runner.main(argv)
    finally:
        record = {"backend": "claude", "model": model, "effort_override": effort,
                  "note": "Runner config names gpt-6-astra for validator compatibility; Claude answered every call.",
                  "clients": [{"client": name, **t.describe(), "calls": t.calls} for name, t in transports]}
        for target, label in ((output, "claude_backend.json"), (review, "claude_backend_prepare.json")):
            if target is not None and target.exists():
                (target / label).write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    raise SystemExit(main())
