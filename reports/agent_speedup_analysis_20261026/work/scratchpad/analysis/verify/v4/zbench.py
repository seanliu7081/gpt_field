import json,base64,struct,zlib,time,statistics as st,sys
p=sys.argv[1]
urls=None
with open(p) as f:
    n=0
    for line in f:
        if line.startswith('{"event": "executed"'):
            n+=1
            if n==200:
                r=json.loads(line); urls=[r["observation"]["observation/image"],r["observation"]["observation/wrist_image"]]; break
def idat(raw):
    i=8; data=b""; hdr=None
    while i<len(raw):
        ln=struct.unpack(">I",raw[i:i+4])[0]; typ=raw[i+4:i+8]; body=raw[i+8:i+8+ln]
        if typ==b"IHDR": hdr=struct.unpack(">IIBBBBB",body)
        if typ==b"IDAT": data+=body
        i+=12+ln
    return hdr,data
res={}
for name,u in zip(("agentview","wrist"),urls):
    raw=base64.b64decode(u.split(",",1)[1])
    hdr,comp=idat(raw)
    filt=zlib.decompress(comp)
    def t(fn,reps=30):
        xs=[]
        for _ in range(reps):
            a=time.perf_counter(); fn(); xs.append(time.perf_counter()-a)
        return st.median(xs)*1000
    res[name]=dict(png_bytes=len(raw),hdr=hdr,raw_filtered=len(filt),
        inflate_ms=t(lambda: zlib.decompress(comp)),
        deflate6_ms=t(lambda: zlib.compress(filt,6),10),
        b64dec_ms=t(lambda: base64.b64decode(u.split(",",1)[1])),
        b64enc_ms=t(lambda: base64.b64encode(raw)),
        json_dumps_ms=t(lambda: json.dumps({"x":u})),
        json_loads_ms=t(lambda: json.loads(json.dumps({"x":u}))))
for k,v in res.items(): print(k,{a:(round(b,2) if isinstance(b,float) else b) for a,b in v.items()})
enc=sum(v["deflate6_ms"] for v in res.values()); dec=sum(v["inflate_ms"] for v in res.values())
print("per step proxy: 2 encodes (deflate6) %.1f ms + 6 decodes (inflate only) %.1f ms = %.1f ms (excludes PNG filtering/unfiltering, rendering, IPC)"%(enc,3*dec,enc+3*dec))
