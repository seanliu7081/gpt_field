# UC7.1 with Claude as the model

`scripts/evaluate_uc71_claude.py` runs the stock UC7.1 runner
(`scripts/evaluate_agent_speedup_v2_user_check.py`) with Claude answering every
model call: the initial plan and every execution request. Prompts, schemas,
validators, simulator, controller, evaluator, timing and artifacts are the stock
code. The normal GPT path is unchanged: no tracked runner or library code imports
these files, and the wrapper only patches objects inside its own process.

| File | Role |
|---|---|
| `src/gpt_field/claude_cli_transport.py` | Turns one OpenAI Responses payload into one `claude -p` call and returns a Responses-shaped reply |
| `scripts/evaluate_uc71_claude.py` | Wrapper around the stock runner; same subcommands and flags |
| `scripts/replay_uc71_requests_claude.py` | Optional: re-sends recorded GPT requests to Claude to compare latency and validity (no simulator, no execution) |
| `tests/test_claude_cli_transport.py` | Offline tests with a fake `claude` binary |

## Running

Requirements: the Claude Code CLI (`claude`) installed and logged in. Calls use that
login; no `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is read. Inside a network sandbox
the CLI needs `api.anthropic.com`.

```bash
env -u OPENAI_API_KEY MUJOCO_GL=cgl PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -u scripts/evaluate_uc71_claude.py start \
  --review-dir runs/NEW_RUN/review --output runs/NEW_RUN/trial \
  --task libero_10_t0 --episode-index 0 \
  --expected-initial-observation-sha256 <reset sha for this machine> \
  --sim-python .cache/venv-libero/bin/python --libero-root .cache/LIBERO \
  --planning-video --auto-approve \
  --claude-model claude-opus-5-5 --claude-effort high
```

- `--claude-model` defaults to `claude-opus-5-5`.
- `--claude-effort` accepts `low|medium|high|xhigh|max`. Without it, the runner's
  `reasoning_effort` is passed through (an unknown level maps to `low`).
- `--stream-progress` and `--request-deadline-s` are OpenAI-SSE options and are
  rejected. Each CLI call has its own timeout of at least 600 s.

## Reading the artifacts

The runner's validators require the GPT configuration, so `configuration.model`
stays `gpt-6-astra` and `reasoning_effort` keeps the stock value, even for a Claude
low-effort run. Those fields describe the stock settings, not the model that answered.
The real backend is recorded in:

- `summary.json` / `runner_summary.json` → top-level `model_backend` (model, effort, billing);
- `<output>/claude_backend.json` and `<review>/claude_backend_prepare.json` → per-call wall time, tokens, effort and model;
- each `requests/*.response.json` → `_claude_transport`.

`billing` reads `claude_subscription_via_claude_code_cli` in request logs, budgets and summaries.

## Differences from the GPT transport

- **History in one turn.** The CLI takes one user turn, so earlier observations,
  retained images and the model's own accepted JSON decisions are replayed oldest
  first inside that turn under explicit labels. GPT's encrypted reasoning items cannot
  be passed to Claude and are dropped.
- **Single-line strings.** The harness rejects control characters in model text that it
  echoes back into later requests, and Claude often writes multi-line text. String
  values are collapsed to single-line text (control characters to spaces, whitespace
  runs collapsed); the count is logged as `strings_normalized`.
- **Errors.** CLI timeouts map to `subscription_stream_interrupted` and overload to
  `server_is_overloaded`, both retried by the stock client. Other CLI errors end the run.
- **Cost.** History is re-sent on every call, so input tokens grow with the number of
  calls; a long failing episode gets expensive.
