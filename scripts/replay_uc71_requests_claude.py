"""Open-loop replay: send each recorded UC7.1 GPT request, byte-identical, to Claude.

Measures Claude latency/tokens on exactly the inputs GPT saw and checks that each
answer passes the harness's own validator. Decisions are not executed; request k
still contains GPT's earlier decisions as history. This is a latency/format check,
not a task-success measurement.

  PYTHONPATH=src .venv/bin/python scripts/replay_uc71_requests_claude.py \
      runs/uc71_mac_t0_ep0_001/trial runs/uc71_claude_replay_t0_001 [--effort high]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpt_field.claude_cli_transport import ClaudeCliTransport, DEFAULT_MODEL  # noqa: E402
from gpt_field.agent_speedup_v2_user_check_agent_client import validate_agent_decision  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trial", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=None)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    transport = ClaudeCliTransport(model=args.model, effort=args.effort)
    rows = []
    for req_path in sorted(glob.glob(str(args.trial / "requests/*.request.json"))):
        req = json.loads(Path(req_path).read_text())
        if req.get("mode") != "agent":
            continue
        gpt = json.loads(Path(req_path.replace(".request.", ".response.")).read_text())
        stream = Path(req_path.replace(".request.json", ".stream.json"))
        gpt_latency = json.loads(stream.read_text()).get("completion_s") if stream.exists() else None
        gpt_text = [p["text"] for it in gpt["output"] if it.get("type") == "message" for p in it["content"]][0]
        row = {"ordinal": req["request_ordinal"], "gpt_latency_s": gpt_latency,
               "gpt_reasoning_tokens": gpt["usage"]["output_tokens_details"].get("reasoning_tokens"),
               "gpt_decision": json.loads(gpt_text)}
        try:
            resp = transport("replay", req["payload"], {}, 600)
            decision = json.loads(resp["output"][0]["content"][0]["text"])
            try:
                validate_agent_decision(decision)
                row["validator"] = "pass"
            except Exception as exc:  # harness validator verdict, recorded
                row["validator"] = f"fail: {exc}"
            row.update(claude=resp["_claude_transport"], claude_decision=decision)
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}", claude=transport.last_diagnostics)
        rows.append(row)
        print(json.dumps({k: row.get(k) for k in ("ordinal", "gpt_latency_s", "validator", "error")}
                         | {"claude_wall_s": (row.get("claude") or {}).get("wall_s")}), flush=True)
        (args.output / "replay.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
