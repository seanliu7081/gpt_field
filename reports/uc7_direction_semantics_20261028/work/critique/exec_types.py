import json, glob, os, sys, collections
def decisions(reqdir):
    out = []
    for f in sorted(glob.glob(os.path.join(reqdir, '*.response.json'))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get('text', {}).get('format', {}).get('name') != 'user_check_agent_first':
            continue
        txt = None
        for item in d.get('output', []):
            if item.get('type') == 'message':
                for c in item.get('content', []):
                    if c.get('type') == 'output_text':
                        txt = c.get('text')
        if txt is None:
            continue
        try:
            j = json.loads(txt)
        except Exception:
            continue
        out.append((os.path.basename(f)[:6], j))
    return out
for r in sys.argv[1:]:
    req = os.path.join(r, 'requests')
    ds = decisions(req)
    c = collections.Counter(j['type'] for _, j in ds)
    macros = [(n, j['macro']) for n, j in ds if j['type'] == 'macro']
    print('==', r.replace('/workspace/gpt_field/runs/', ''), dict(c), 'macros:', macros[:6])
