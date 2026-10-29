import json, sys, os, glob, collections
d = sys.argv[1]
verbose = len(sys.argv) > 2 and sys.argv[2] == '-v'
files = sorted(glob.glob(os.path.join(d, 'requests', '*.request.json')))
rows = []
for f in files:
    base = f[:-len('.request.json')]
    req = json.load(open(f))
    inp = req['payload']['input']
    users = [m for m in inp if m.get('role') == 'user']
    last = users[-1]
    txt = None
    for c in last['content']:
        if c.get('type') == 'input_text':
            txt = c['text']; break
    try:
        ctx = json.loads(txt)
    except Exception as e:
        ctx = {'_parse_error': str(e), '_text': txt[:200]}
    context = ctx.get('context') if isinstance(ctx.get('context'), dict) else {}
    lf = context.get('last_feedback') if context else None
    if lf is None: lf = ctx.get('last_feedback')
    reason = status = action = None
    if isinstance(lf, dict):
        s = lf.get('summary') if isinstance(lf.get('summary'), dict) else lf
        reason = s.get('reason'); status = s.get('status'); action = s.get('action')
    cd = ctx.get('current_decision') if isinstance(ctx.get('current_decision'), dict) else {}
    rag = cd.get('repeated_action_guard')
    blocker = cd.get('blocker')
    ea = cd.get('evidence_acquisition')
    # response
    rtype = None; resp_ok = os.path.exists(base + '.response.json')
    if resp_ok:
        resp = json.load(open(base + '.response.json'))
        for o in resp.get('output', []):
            if o.get('type') == 'message':
                for c in o.get('content', []):
                    if c.get('type') == 'output_text':
                        try:
                            rtype = json.loads(c['text']).get('type')
                        except Exception as e:
                            rtype = 'PARSE_ERR:' + c['text'][:80]
    res = json.load(open(base + '.result.json')) if os.path.exists(base + '.result.json') else {}
    el = (res.get('v3_transport_diagnostics') or {}).get('elapsed_s')
    rows.append(dict(file=os.path.basename(f), ordinal=req.get('request_ordinal'), attempt=req.get('attempt'), step=ctx.get('step'),
                     reason=reason, status=status, action=action, rtype=rtype, elapsed=el, result_status=res.get('status'),
                     rag=rag, blocker=blocker, ea=ea, cd_keys=list(cd.keys())))
print('n requests', len(rows))
for r in rows:
    print(r['ordinal'], r['attempt'], 'step', r['step'], '| fb:', r['action'], r['status'], r['reason'], '| resp type:', r['rtype'], '| el %.1f' % (r['elapsed'] or -1), '|', r['result_status'])
c = collections.Counter(r['reason'] for r in rows)
print('reason counter', c.most_common())
print('rtype counter', collections.Counter(r['rtype'] for r in rows).most_common())
tot = sum(r['elapsed'] or 0 for r in rows)
print('sum transport elapsed', tot)
by = collections.defaultdict(float); byn=collections.Counter()
for r in rows:
    by[r['rtype']] += (r['elapsed'] or 0); byn[r['rtype']]+=1
print('elapsed by rtype', {k: (round(v,1), byn[k]) for k, v in by.items()})
json.dump([{k: v for k, v in r.items() if k not in ('ea','rag','blocker')} for r in rows], open(os.path.join('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify', os.path.basename(d.rstrip('/')) + '_requests.json'), 'w'), indent=1)
