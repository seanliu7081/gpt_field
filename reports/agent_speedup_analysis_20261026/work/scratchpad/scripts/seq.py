import json, sys, os, glob
for run in sys.argv[1:]:
    rows = []
    for path in sorted(glob.glob(os.path.join(run, "requests", "*.response.json"))):
        r = json.load(open(path))
        d = None
        for item in r.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        try: d = json.loads(part["text"])
                        except Exception: d = {}
        tr = (r.get("_subscription_transport") or {}).get("v3_bounds") or {}
        u = r.get("usage") or {}
        req = json.load(open(path.replace(".response.json", ".request.json")))
        step = None
        users = [i for i in req["payload"]["input"] if i.get("role") == "user"]
        for part in users[-1]["content"]:
            if part.get("type") == "input_text":
                try:
                    f = json.loads(part["text"]); step = f.get("step", step)
                except Exception: pass
        rows.append(((d or {}).get("type"), step, round(tr.get("elapsed_s") or 0, 1), u.get("input_tokens"), u.get("output_tokens")))
    tot = sum(r[2] for r in rows)
    types = {}
    for r in rows: types[r[0]] = types.get(r[0], 0) + 1
    print("==", run, "n=%d sum_transport_s=%.1f" % (len(rows), tot), types)
    print("  ", [(t, s, e) for t, s, e, i, o in rows])
