"""Launch-only handoff; starts delegated review clock, never approves a plan."""
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
args = sys.argv[1:]
if args[0] != "prepare":
    raise ValueError("prepare only")
reservation = json.loads(Path(args[args.index("--reservation-file") + 1]).read_text())
if not (reservation["stage"] == "P5" and reservation["task"] in {"libero_10_t1", "libero_10_t5", "libero_10_t7"} and reservation["episode_index"] in {1, 2, 3} and reservation["seed"] == 7):
    raise ValueError("outside explicitly delegated P5 review scope")
if not (ROOT / "delegated_review_authorization.json").exists():
    raise ValueError("authorization evidence missing")
runner = runpy.run_path(str(ROOT / "baseline/scripts/evaluate_speedup_evidence.py"))
result = runner["cli"](args)
if result == 0:
    result = runner["cli"](["review-begin", "--session-dir", args[args.index("--session-dir") + 1], "--authorization-note", "Explicit user delegation for frozen t1/t5/t7 seed7 P5 ep1-3 matrix; begin actual Codex review immediately after preparation. No plan approval is performed by this launcher."])
raise SystemExit(result)
