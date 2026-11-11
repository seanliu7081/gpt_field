"""Share of each executive request's JSON text taken by native-macro/legacy fields."""
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from calls import load, request_fields
for run in sys.argv[1:]:
    tot = mac = 0
    n = 0
    for rp in sorted(glob.glob(os.path.join(run, "trial", "requests", "*.request.json"))):
        r = load(rp)
        if r.get("mode") != "agent":
            continue
        f = request_fields(r["payload"]) or {}
        ctx = f.get("context", {})
        size = len(json.dumps(f, separators=(",", ":")))
        legacy = sum(len(json.dumps(ctx.get(k), separators=(",", ":"))) for k in ("macro", "native_evidence"))
        legacy += len(json.dumps((ctx.get("capabilities") or {}).get("native_macros"), separators=(",", ":")))
        if "reviewed_goal_contract" in f:
            legacy += len(json.dumps(f["reviewed_goal_contract"], separators=(",", ":")))
        tot += size; mac += legacy; n += 1
    if n:
        print(os.path.basename(run)[22:], "requests", n, "mean chars", tot // n, "legacy/macro share", round(mac / tot, 3))
