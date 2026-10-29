import json, sys, os, glob, collections
tot_ran = tot_cand = tot_req = 0
tot_rej = collections.Counter()
for d in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(d, 'requests', '*.request.json')))
    ran = cand = 0; n=0; rej=collections.Counter(); steps_with=[]
    for f in files:
        r = json.load(open(f))
        users = [m for m in r['payload']['input'] if m.get('role') == 'user']
        txt = next(c['text'] for c in users[-1]['content'] if c.get('type') == 'input_text')
        try:
            ctx = json.loads(txt)
        except Exception:
            print('  parse fail', f); continue
        n += 1
        cd = ctx.get('current_decision') or {}
        ea = cd.get('evidence_acquisition') if isinstance(cd, dict) else None
        if isinstance(ea, dict):
            ran += 1
            c = ea.get('candidates') or []
            if len(c) >= 1:
                cand += 1; steps_with.append((ctx.get('step'), len(c)))
            for x in (ea.get('rejected') or []):
                if isinstance(x, dict):
                    rej[x.get('reason') or x.get('stage') or str(sorted(x.keys()))] += 1
                else:
                    rej[str(x)[:60]] += 1
    tot_ran += ran; tot_cand += cand; tot_req += n
    tot_rej.update(rej)
    print(os.path.relpath(d, '/workspace/gpt_field'), 'requests', n, 'ea present', ran, 'with>=1 cand', cand, steps_with, 'rejections', sum(rej.values()))
print('TOTAL requests', tot_req, 'ea present', tot_ran, 'with cand', tot_cand, 'rejections', sum(tot_rej.values()))
print('rejection reasons', tot_rej.most_common(12))
