#!/usr/bin/env python3
"""Read-only OLS of per-call latency on token counts using calls.json (stdlib only)."""
import json, sys
d = json.load(open(sys.argv[1]))
def solve(X, y):
    n = len(X[0]); A = [[sum(x[i]*x[j] for x in X) for j in range(n)] for i in range(n)]
    b = [sum(x[i]*yy for x, yy in zip(X, y)) for i in range(n)]
    M = [A[i]+[b[i]] for i in range(n)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(M[r][i])); M[i], M[p] = M[p], M[i]
        for r in range(n):
            if r != i:
                f = M[r][i]/M[i][i]; M[r] = [a-f*c for a, c in zip(M[r], M[i])]
    return [M[i][n]/M[i][i] for i in range(n)]
def fit(rows, label):
    X = [[1, r['in']/1000, r['vis'], r['reas']] for r in rows]; y = [r['lat'] for r in rows]
    co = solve(X, y); pred = [sum(a*b for a, b in zip(x, co)) for x in X]
    my = sum(y)/len(y); r2 = 1-sum((a-b)**2 for a, b in zip(y, pred))/sum((a-my)**2 for a in y)
    print('%s n=%d: lat = %.3f + %.5f*in_ktok + %.5f*visible_tok + %.5f*reasoning_tok  R2=%.3f' % (label, len(rows), *co, r2))
    return co
rows = []
for run, v in d.items():
    for i, c in enumerate(v['calls']):
        if c.get('result_status') != 'valid' or c.get('input_tokens') is None: continue
        lat = c.get('transport_elapsed_s') if c.get('transport_elapsed_s') is not None else c.get('budget_interval_s')
        rows.append({'run': run, 'i': i, 'lat': lat, 'in': c['input_tokens'], 'vis': c['output_tokens']-(c['reasoning_tokens'] or 0), 'reas': c['reasoning_tokens'] or 0})
co = fit(rows, 'all calls')
fit([r for r in rows if r['i'] > 0], 'excluding first calls')
fit([r for r in rows if r['run'] != 'agent_speedup_v2_user_check_libero_10_t1_001'], 'v3 only')
def means(run):
    rr = [r for r in rows if r['run'] == run]; n = len(rr)
    return {k: sum(r[k] for r in rr)/n for k in ('lat', 'in', 'vis', 'reas')}
a = means('agent_speedup_v2_user_check_libero_10_t1_001'); b = means('agent_speedup_v3_t1_008')
print('uc7 means', {k: round(v, 1) for k, v in a.items()}); print('t1_008 means', {k: round(v, 1) for k, v in b.items()})
din = co[1]*(b['in']-a['in'])/1000; dvis = co[2]*(b['vis']-a['vis']); drea = co[3]*(b['reas']-a['reas'])
print('attribution of per-call delta %.2f s: input %.2f, visible %.2f, reasoning %.2f, unexplained %.2f' % (b['lat']-a['lat'], din, dvis, drea, b['lat']-a['lat']-din-dvis-drea))
