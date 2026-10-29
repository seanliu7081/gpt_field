import json,glob,sys
R=sys.argv[1]
reqs=sorted(glob.glob(R+"/trial/requests/*.request.json"))
for p in reqs:
    d=json.load(open(p))
    inp=d["payload"]["input"]
    last=inp[-1]
    txt=[c["text"] for c in last["content"] if c["type"]=="input_text"][0]
    try:
        f=json.loads(txt)
    except Exception as e:
        print(p, "nonjson", txt[:100]); continue
    ctx=f.get("context",{})
    ne=ctx.get("native_evidence",{})
    parts=ne.get("parts",{})
    macro=ctx.get("macro",{})
    res=p.replace(".request.json",".result.json")
    try:
        r=json.load(open(res))
    except Exception:
        r={}
    out=r.get("result") or r.get("data") or r
    typ=None
    if isinstance(out,dict):
        typ=out.get("type")
        if typ is None:
            for k,v in out.items():
                if isinstance(v,dict) and "type" in v: typ=v.get("type"); break
    print(p.split("/")[-1][:6], "step", f.get("step"), "keys", sorted(f.keys()), "parts", sorted(parts.keys()) if isinstance(parts,dict) else type(parts), "macro_active", macro.get("active"), "step_index", macro.get("step_index"), "type", typ)
