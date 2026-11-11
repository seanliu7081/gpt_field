"""Max executive request field size vs the 45k compaction and 56k hard-error bounds."""
import json, glob, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from calls import load, request_fields
for run in sys.argv[1:]:
    sizes = []; compacted = 0
    for rp in sorted(glob.glob(os.path.join(run, "trial", "requests", "*.request.json"))):
        r = load(rp)
        if r.get("mode") != "agent":
            continue
        f = request_fields(r["payload"]) or {}
        sizes.append(len(json.dumps(f, ensure_ascii=False, separators=(",", ":"))))
        compacted += "context_detail_compaction" in f
    if sizes:
        print(os.path.basename(run)[22:], "max", max(sizes), "compacted", compacted, "of", len(sizes))
