import json, sys
path = sys.argv[1]; target_ev = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv)>3 else 1
def trunc(o, depth=0):
    if isinstance(o, str):
        return o if len(o) < 300 else o[:300]+'...<%d chars>'%len(o)
    if isinstance(o, dict):
        return {k: trunc(v, depth+1) for k,v in o.items()}
    if isinstance(o, list):
        if len(o) > 20: return [trunc(x, depth+1) for x in o[:20]] + ['...<%d items>'%len(o)]
        return [trunc(x, depth+1) for x in o]
    return o
cnt=0
with open(path) as f:
    for line in f:
        r=json.loads(line)
        if r.get('event')==target_ev:
            print(json.dumps(trunc(r), indent=1)[:12000])
            cnt+=1
            if cnt>=n: break
