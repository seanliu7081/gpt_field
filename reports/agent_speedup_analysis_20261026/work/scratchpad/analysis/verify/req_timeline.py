import json, os, sys, glob, datetime
run = sys.argv[1]
rq = sorted(glob.glob(os.path.join(run, 'requests', '*.request.json')))
for f in rq:
    base = f[:-len('.request.json')]
    try:
        r = json.load(open(f))
    except Exception as e:
        print('ERR', f, e); continue
    meta = r.get('request_meta', {})
    resp = None
    if os.path.exists(base + '.response.json'):
        try:
            resp = json.load(open(base + '.response.json'))
        except Exception:
            resp = None
    res = None
    if os.path.exists(base + '.result.json'):
        res = json.load(open(base + '.result.json'))
    ca = resp.get('created_at') if resp else None
    co = resp.get('completed_at') if resp else None
    fmt = lambda t: datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%H:%M:%S') if t else None
    el = res.get('budget', {}).get('elapsed_s') if res else None
    # parse type of output
    typ = None
    if resp:
        for o in resp.get('output', []):
            if o.get('type') == 'message':
                for c in o.get('content', []):
                    t = c.get('text')
                    if t:
                        try:
                            typ = json.loads(t).get('type')
                        except Exception:
                            typ = 'unparsed'
    print(os.path.basename(base)[:6], meta.get('kind'), 'step', meta.get('step'), 'created', fmt(ca), 'completed', fmt(co), 'budget_elapsed', None if el is None else round(el, 1), 'type', typ, 'status', res.get('status') if res else None)
