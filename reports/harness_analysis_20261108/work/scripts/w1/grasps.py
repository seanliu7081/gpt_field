"""Count close commands (open->close transitions) in executed actions = grasp attempts."""
import json, os, sys
for run in sys.argv[1:]:
    path = os.path.join(run, "trial", "trace.jsonl")
    if not os.path.exists(path):
        continue
    prev = -1.0
    closes, opens = [], []
    n = 0
    for line in open(path):
        if '"event": "action"' not in line[:40]:
            continue
        d = json.loads(line)
        a = (d.get("decision") or {}).get("action") or d.get("action")
        if isinstance(a, dict):
            a = a.get("action")
        if not isinstance(a, list) or len(a) != 7:
            continue
        g = a[6]
        if g > 0 and prev <= 0:
            closes.append(n)
        if g < 0 and prev > 0:
            opens.append(n)
        prev = g if g != 0 else prev
        n += 1
    print(os.path.basename(run)[22:], "actions", n, "close_cmds", len(closes), closes, "open_cmds", len(opens))
