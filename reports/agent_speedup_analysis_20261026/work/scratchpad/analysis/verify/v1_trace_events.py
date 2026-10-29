import json, sys, collections
path = sys.argv[1]
ev = collections.Counter()
keys_by_ev = collections.defaultdict(collections.Counter)
with open(path) as f:
    for i, line in enumerate(f):
        line=line.strip()
        if not line: continue
        r = json.loads(line)
        e = r.get('event')
        ev[e]+=1
        for k in r.keys():
            keys_by_ev[e][k]+=1
for e,c in ev.most_common():
    print(e, c, dict(keys_by_ev[e]))
