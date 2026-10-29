import json, sys, collections
for path in sys.argv[1:]:
    by=collections.defaultdict(float); n=collections.Counter(); tot=0; calls=0
    with open(path) as f:
        for line in f:
            r=json.loads(line)
            if r.get('event')=='action':
                dg=r['diagnostics']
                for c in (dg.get('api_calls') or []):
                    sk=(dg.get('routing') or {}).get('skill')
                    by[sk]+=c.get('latency_s') or 0; n[sk]+=1; tot+=c.get('latency_s') or 0; calls+=1
    print(path, 'calls', calls, 'total api latency', round(tot,3), {k:(round(v,1),n[k]) for k,v in by.items()}, 'per call', round(tot/max(calls,1),2))
