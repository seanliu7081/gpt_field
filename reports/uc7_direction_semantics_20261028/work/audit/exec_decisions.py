import json,glob,sys,re
R=sys.argv[1]
for p in sorted(glob.glob(R+"/trial/requests/*.response.json")):
    d=json.load(open(p))
    text=None
    for o in d.get("output",[]):
        if o.get("type")=="message":
            for c in o.get("content",[]):
                if c.get("type") in ("output_text","text"):
                    text=c.get("text")
    if text is None:
        print(p.split("/")[-1][:6], "no text"); continue
    try:
        a=json.loads(text)
    except Exception:
        print(p.split("/")[-1][:6], "nonjson"); continue
    mot=a.get("motion"); ex=a.get("execution")
    det=""
    if mot: det="motion "+json.dumps({k:mot.get(k) for k in ("position_mode","xyz_m","gripper","steps")})
    if ex: det="execute "+json.dumps([[s.get("skill"),s.get("xyz_m"),s.get("gripper")] for s in ex.get("segments",[])])
    hits=[]
    for key in ("belief","plan","why"):
        for m in re.finditer(r"([^.;]{0,90}(?:\+x|-x|\+y|-y|right|left|front|behind|image)[^.;]{0,90})", a.get(key,"")):
            hits.append(key+": "+m.group(1).strip())
    print(p.split("/")[-1][:6], a.get("type"), det[:300])
    for h in hits[:4]:
        print("    ", h[:260])
