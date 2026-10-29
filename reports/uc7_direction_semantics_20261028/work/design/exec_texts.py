"""Print executive belief/plan texts containing direction words from a run's requests/*.response.json."""
import json, glob, re, os, sys
run = sys.argv[1]
pat = re.compile(r"(\+x|\+y|-y|-x|right of|right side|to the right|left of|world [xy])", re.I)
for f in sorted(glob.glob(os.path.join(run, "*.response.json"))):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    texts = []
    for item in d.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") in ("output_text",):
                texts.append(c.get("text", ""))
    for t in texts:
        try:
            j = json.loads(t)
        except Exception:
            j = {"raw": t}
        for key in ("plan", "belief", "why", "raw"):
            v = j.get(key) if isinstance(j, dict) else None
            if isinstance(v, str) and pat.search(v):
                m = pat.search(v)
                s = max(0, m.start() - 160)
                print(os.path.basename(f)[:6], key, "|", v[s:m.end() + 160].replace("\n", " "))
                break
