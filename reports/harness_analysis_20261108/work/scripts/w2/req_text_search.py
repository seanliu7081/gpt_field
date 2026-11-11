import json, sys, glob, os, re
RUNS = "/workspace/gpt_field/runs"
runs = sys.argv[1].split(",")
terms = sys.argv[2].split(",")
def strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)
    elif isinstance(o, str):
        if o.startswith("data:") or (len(o) > 2000 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", o[:2000] or "")):
            return
        yield o
for r in runs:
    files = sorted(glob.glob(f"{RUNS}/{r}/trial/requests/*.request.json"))
    hits = {t: [] for t in terms}
    for f in files:
        d = json.load(open(f))
        text = "\n".join(strings(d))
        for t in terms:
            for m in re.finditer(re.escape(t), text, flags=re.I):
                hits[t].append((os.path.basename(f)[:6], text[max(0, m.start()-100):m.end()+100].replace("\n", " ")))
    print("==", r, "requests", len(files))
    for t in terms:
        print("  term", t, "files", len({h[0] for h in hits[t]}), "hits", len(hits[t]))
        for h in hits[t][:3]:
            print("     ", h[0], "|", h[1])
