"""Common prefix (in input items) between consecutive requests; where image parts are removed; macro usage."""
import json, glob, sys, hashlib
run = sys.argv[1]
files = sorted(glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/*.request.json"))
prev = None
for k, f in enumerate(files):
    d = json.load(open(f)); p = d["payload"]
    items = [hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest() for x in p["input"]]
    nimg = [sum(1 for c in (x.get("content") or []) if isinstance(c, dict) and c.get("type") == "input_image") if isinstance(x, dict) and isinstance(x.get("content"), list) else 0 for x in p["input"]]
    if prev is not None:
        cp = 0
        for a, b in zip(prev, items):
            if a != b: break
            cp += 1
        print(k + 1, "items", len(items), "common_prefix_with_prev", cp, "of", len(prev), "| images per item (nonzero idx):", [(i, n) for i, n in enumerate(nimg) if n])
    else:
        print(k + 1, "items", len(items), "| images:", [(i, n) for i, n in enumerate(nimg) if n])
    prev = items
