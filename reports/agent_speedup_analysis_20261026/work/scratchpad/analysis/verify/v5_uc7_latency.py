import json, os, glob, math
root = '/workspace/gpt_field/runs'
keys = ['t0_001', 't1_001', 't2_002', 't3_003', 't3_005', 't4_001', 't5_001', 't5_002', 't6_009', 't6_011', 't7_001', 't8_001', 't9_001']
uc6 = ['t6_007', 't3_002']
def pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)
def rank(v):
    s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]: j += 1
        for k in range(i, j + 1): r[s[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def collect(keys):
    calls = []
    for k in keys:
        d = os.path.join(root, 'agent_speedup_v2_user_check_libero_10_' + k)
        summ = json.load(open(os.path.join(d, 'summary.json')))
        ver = (summ.get('user_check') or {}).get('version')
        multi = 0; n_act_calls = 0
        for line in open(os.path.join(d, 'trace.jsonl')):
            r = json.loads(line)
            if r.get('event') != 'action':
                continue
            diag = r.get('diagnostics') or {}
            ac = diag.get('api_calls') or []
            if not ac:
                continue
            if len(ac) > 1:
                multi += 1
            for i, c in enumerate(ac):
                n_act_calls += 1
                log = diag.get('api_request_log')
                base = log[:-len('.request.json')] if log else None
                resp = json.load(open(base + '.response.json')) if base and os.path.exists(base + '.response.json') else None
                req = json.load(open(log)) if log else None
                res = json.load(open(base + '.result.json')) if base and os.path.exists(base + '.result.json') else None
                u = (resp or {}).get('usage') or {}
                calls.append(dict(run=k, ver=ver, step=r.get('step'), kind=c.get('kind'), ok=c.get('ok'), lat=c.get('latency_s'),
                                  api_latency_s=diag.get('api_latency_s'), n_in_act=len(ac),
                                  inp=u.get('input_tokens'), out=u.get('output_tokens'),
                                  reas=(u.get('output_tokens_details') or {}).get('reasoning_tokens'),
                                  cached=(u.get('input_tokens_details') or {}).get('cached_tokens'),
                                  server=(resp['completed_at'] - resp['created_at']) if resp else None,
                                  client_interval=((res or {}).get('budget') or {}).get('elapsed_s', 0) - (req or {}).get('budget_elapsed_s_before', 0) if res and req else None,
                                  log=log))
        # also count request files in the dir
        nreq = len(glob.glob(os.path.join(d, 'requests', '*.request.json')))
        print(k, ver, 'act-call rows', n_act_calls, 'multi-call acts', multi, 'request files', nreq)
    return calls
for label, ks in (('uc7', keys), ('uc6', uc6)):
    calls = collect(ks)
    good = [c for c in calls if c['lat'] is not None and c['inp'] is not None and c['reas'] is not None]
    print(label, 'calls', len(calls), 'with usage', len(good), 'ok', sum(1 for c in calls if c['ok']))
    lat = [c['lat'] for c in good]; inp = [c['inp'] for c in good]; out = [c['out'] for c in good]; reas = [c['reas'] for c in good]
    vis = [o - r for o, r in zip(out, reas)]
    print(' latency mean %.2f median %.2f max %.2f' % (sum(lat) / len(lat), sorted(lat)[len(lat) // 2], max(lat)))
    print(' input tokens min %d max %d mean %.0f' % (min(inp), max(inp), sum(inp) / len(inp)))
    print(' r(lat,input) %.3f  r(lat,reasoning) %.3f  r(lat,output) %.3f  r(lat,visible) %.3f' % (pearson(lat, inp), pearson(lat, reas), pearson(lat, out), pearson(lat, vis)))
    print(' spearman: input %.3f reasoning %.3f output %.3f' % (pearson(rank(lat), rank(inp)), pearson(rank(lat), rank(reas)), pearson(rank(lat), rank(out))))
    print(' r(reasoning,input) %.3f' % pearson(reas, inp))
    srv = [c['server'] for c in good]
    print(' r(server,input) %.3f r(server,output) %.3f' % (pearson(srv, inp), pearson(srv, out)))
    ci = [c['client_interval'] for c in good if c['client_interval'] is not None]
    print(' mean |client_interval - lat| %.3f' % (sum(abs(c['client_interval'] - c['lat']) for c in good if c['client_interval'] is not None) / len(ci)))
    # per-run first/last input tokens
    from collections import defaultdict
    byrun = defaultdict(list)
    for c in good: byrun[c['run']].append(c)
    for k, cs in byrun.items():
        print('   ', k, 'n', len(cs), 'input first %d last %d max %d' % (cs[0]['inp'], cs[-1]['inp'], max(x['inp'] for x in cs)))
    json.dump(calls, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v5_%s_calls.json' % label, 'w'))
