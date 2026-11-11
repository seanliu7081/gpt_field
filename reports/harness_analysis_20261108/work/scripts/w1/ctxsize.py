import json, glob, sys, os, hashlib
run = sys.argv[1]
files = sorted(glob.glob(os.path.join(run, 'trial', 'requests', '*.request.json')))
prev = {}
tot = {}
static_total = 0; dyn_total = 0
for f in files:
    d = json.load(open(f))
    inp = d['payload']['input']
    last = inp[-1]
    txt = [c['text'] for c in last['content'] if c['type']=='input_text'][0]
    fields = json.loads(txt)
    parts = {}
    for k, v in fields.items():
        if k == 'context':
            for kk, vv in v.items():
                parts['context.'+kk] = json.dumps(vv, separators=(',',':'))
        else:
            parts[k] = json.dumps(v, separators=(',',':'))
    same = {k: (prev.get(k) == s) for k, s in parts.items()}
    sizes = {k: len(s) for k, s in parts.items()}
    st = sum(sizes[k] for k in sizes if same[k]); dy = sum(sizes[k] for k in sizes if not same[k])
    if prev:
        static_total += st; dyn_total += dy
    print(os.path.basename(f)[:6], 'total', len(txt), 'unchanged-vs-prev chars', st, 'changed', dy)
    for k in sorted(sizes, key=lambda k: -sizes[k])[:14]:
        print('    %-32s %6d %s' % (k, sizes[k], 'same' if same[k] else 'CHANGED'))
    prev = parts
    if os.path.basename(f).startswith('000003'): break
