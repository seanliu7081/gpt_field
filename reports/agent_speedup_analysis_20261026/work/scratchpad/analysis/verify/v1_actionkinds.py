import json, sys, collections
for path in sys.argv[1:]:
    ek=collections.Counter(); lat=[]; refreshed=0; rskill=collections.Counter()
    with open(path) as f:
        for line in f:
            r=json.loads(line)
            if r.get('event')=='action':
                dg=r.get('diagnostics') or {}
                ek[dg.get('execution_kind')]+=1
                if dg.get('refreshed'):
                    refreshed+=1
                    lat.append(r.get('decision_latency_s'))
                    rt=(dg.get('routing') or {})
                    rskill[(rt.get('kind'), rt.get('skill'))]+=1
    print(path)
    print(' execution_kind', ek.most_common())
    print(' refreshed rows', refreshed, 'sum decision_latency_s on refreshed', round(sum(x or 0 for x in lat),2))
    print(' routing (kind,skill) on refreshed', rskill.most_common())
