"""Measure executive request text sizes (fields JSON) in recorded first agent requests; read-only."""
import json, sys, glob, os

def size(v):
    return len(json.dumps(v, ensure_ascii=True, separators=(",", ":")))

def first_text(path):
    data = json.load(open(path))
    payload = data.get("payload", data)
    inp = payload.get("input") or []
    # last user message content's first input_text is the request JSON
    for msg in reversed(inp):
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "input_text":
                    return part["text"]
    return None

for run in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(run, "**", "requests", "*.request.json"), recursive=True))
    rows = []
    for f in files:
        try:
            t = first_text(f)
        except Exception as e:
            rows.append((os.path.basename(f), "ERR " + type(e).__name__)); continue
        if t is None:
            continue
        try:
            obj = json.loads(t)
        except Exception:
            rows.append((os.path.basename(f), "nonjson", len(t))); continue
        if obj.get("request") != "agent":
            continue
        fields = {k: v for k, v in obj.items() if k not in ("request", "task")}
        rg = fields.get("reviewed_goal_contract")
        rows.append((os.path.basename(f)[:6], size(fields), "goal" if rg else "-",
                     "compacted" if "context_detail_compaction" in fields else ""))
    mx = max((r[1] for r in rows if isinstance(r[1], int)), default=None)
    first = rows[0] if rows else None
    print(run.split("/runs/")[-1], "n=", len(rows), "first=", first, "max=", mx,
          "compacted=", sum(1 for r in rows if len(r) > 3 and r[3]))
