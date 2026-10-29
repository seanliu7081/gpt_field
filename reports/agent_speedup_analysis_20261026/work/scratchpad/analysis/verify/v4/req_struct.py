import json,sys,glob
def skel(o,depth=0,maxd=4):
    if isinstance(o,dict):
        if depth>=maxd: return "{...%d keys}"%len(o)
        return {k:skel(v,depth+1,maxd) for k,v in o.items()}
    if isinstance(o,list):
        if len(o)>0 and depth<maxd: return ["list[%d]"%len(o), skel(o[0],depth+1,maxd)]
        return "list[%d]"%len(o)
    if isinstance(o,str):
        return "str[%d]:%s"%(len(o),o[:60])
    return o
d=sys.argv[1]
fs=sorted(glob.glob(d+"/requests/000002-*"))
for f in fs:
    print("=====",f)
    print(json.dumps(skel(json.load(open(f))),indent=1)[:3500])
