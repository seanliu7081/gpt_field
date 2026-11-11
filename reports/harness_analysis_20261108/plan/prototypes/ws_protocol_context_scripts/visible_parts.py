"""Composition of executive visible output by field (offline, read-only)."""
import json, glob, sys, os, statistics as st
def text_of(resp):
    for o in resp.get("output", []):
        if o.get("type") == "message":
            for c in o.get("content", []):
                if c.get("type") == "output_text":
                    return c["text"]
rows = []
for run in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(run, "trial", "requests", "*.response.json"))):
        r = json.load(open(f))
        t = text_of(r)
        if not t: continue
        d = json.loads(t)
        if "belief" not in d: continue
        u = r["usage"]; vis = u["output_tokens"] - u["output_tokens_details"]["reasoning_tokens"]
        tot = len(t)
        def L(k): return len(json.dumps(d.get(k), separators=(",", ":")))
        segs = (d.get("execution") or {}).get("segments") or []
        rows.append({"run": os.path.basename(run)[:40], "vis_tok": vis, "chars": tot,
                     "belief": len(d["belief"]), "plan": len(d["plan"]), "why": len(d["why"]),
                     "execution": L("execution") if d.get("execution") else 0,
                     "motion": L("motion") if d.get("motion") else 0,
                     "obstacle_map": L("obstacle_map") if d.get("obstacle_map") else 0,
                     "nseg": len(segs), "type": d["type"],
                     "route_nonnull": sum(1 for s in segs if s.get("route") is not None),
                     "quat_nonnull": sum(1 for s in segs if s.get("quaternion_wxyz") is not None)})
print("n responses", len(rows))
for k in ("vis_tok", "chars", "belief", "plan", "why", "execution", "obstacle_map", "nseg"):
    v = [r[k] for r in rows]
    print(f"{k:13s} mean {st.mean(v):7.1f} median {st.median(v):7.1f} p90 {sorted(v)[int(.9*len(v))]:7.1f} max {max(v)}")
prose = sum(r["belief"] + r["plan"] + r["why"] for r in rows); allc = sum(r["chars"] for r in rows)
print("prose share of chars %.2f" % (prose/allc), "| execution share %.2f" % (sum(r["execution"] for r in rows)/allc),
      "| obstacle_map share %.2f" % (sum(r["obstacle_map"] for r in rows)/allc))
print("chars per visible token %.2f" % (allc/sum(r["vis_tok"] for r in rows)))
print("types", {t: sum(1 for r in rows if r["type"] == t) for t in set(r["type"] for r in rows)})
print("segments with route override", sum(r["route_nonnull"] for r in rows), "of", sum(r["nseg"] for r in rows))
print("responses with obstacle_map non-null", sum(1 for r in rows if r["obstacle_map"]))
print("over 600 plan chars", sum(1 for r in rows if r["plan"] > 600), "| over 200 why", sum(1 for r in rows if r["why"] > 200), "| over 300 why", sum(1 for r in rows if r["why"] > 300))
