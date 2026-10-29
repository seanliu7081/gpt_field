import json, sys, os, glob
for run in sys.argv[1:]:
    seq = []
    counts = {}
    for path in sorted(glob.glob(os.path.join(run, "requests", "*.request.json"))):
        req = json.load(open(path))
        users = [i for i in req["payload"]["input"] if i.get("role") == "user"]
        f = {}
        for part in users[-1]["content"]:
            if part.get("type") == "input_text":
                try:
                    g = json.loads(part["text"])
                    if isinstance(g, dict) and "context" in g: f = g
                except Exception: pass
        # Prefer unprojected original context if logged (r10+)
        ctx = req.get("original_context") or f.get("context") or {}
        lf = ctx.get("last_feedback") or {}
        if isinstance(lf, dict) and "summary" in lf and isinstance(lf["summary"], dict):
            lf = lf["summary"]
        reason = None if not lf else (lf.get("action"), lf.get("reason"))
        seq.append((f.get("step"), reason))
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    print("==", run)
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("   %3d  %s" % (v, k))
