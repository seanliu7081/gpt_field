import json,sys,collections
p=sys.argv[1]
kinds=collections.Counter()
examples={}
def skel(o,depth=0,maxd=3):
    if isinstance(o,dict):
        if depth>=maxd: return "{...%d keys}"%len(o)
        return {k:skel(v,depth+1,maxd) for k,v in o.items()}
    if isinstance(o,list):
        if len(o)>0 and depth<maxd: return ["list[%d]"%len(o), skel(o[0],depth+1,maxd)]
        return "list[%d]"%len(o)
    if isinstance(o,str):
        return "str[%d]:%s"%(len(o),o[:40])
    return o
with open(p) as f:
    for i,line in enumerate(f):
        r=json.loads(line)
        k=r.get("kind") or r.get("type") or r.get("event") or "?"
        kinds[k]+=1
        if k not in examples: examples[k]=(i,skel(r))
print(kinds)
for k,(i,s) in examples.items():
    print("=====",k,"line",i)
    print(json.dumps(s,indent=1)[:4000])
