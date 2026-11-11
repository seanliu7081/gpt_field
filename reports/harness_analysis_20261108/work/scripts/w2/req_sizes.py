import json, glob, os, sys, statistics
RUNS = '/workspace/gpt_field/runs'
out = {}
tot = 0; comp = 0; maxlen = 0; shares = []
for run in sorted(glob.glob(RUNS + '/agent_speedup_v2_uc71_*')):
    rows = []
    for p in sorted(glob.glob(run + '/trial/requests/*.request.json')):
        d = json.load(open(p))
        last = d['payload']['input'][-1]
        txt = [c['text'] for c in last['content'] if c.get('type') == 'input_text'][0]
        ctx = json.loads(txt)
        n = len(txt)
        c = ctx['context']
        macro_chars = len(json.dumps(c.get('macro'))) + len(json.dumps(c.get('native_evidence')))
        cap = c.get('capabilities') or {}
        macro_chars += len(json.dumps(cap.get('native_macros')))
        rgc = ctx.get('reviewed_goal_contract')
        rows.append({'file': os.path.basename(p), 'chars': n, 'compacted': 'context_detail_compaction' in ctx,
                     'macro_chars': macro_chars, 'share': macro_chars / n,
                     'input_tokens': None})
        tot += 1; comp += 'context_detail_compaction' in ctx; maxlen = max(maxlen, n); shares.append(macro_chars / n)
    out[os.path.basename(run)] = rows
for k, rows in out.items():
    if rows:
        print(k[23:], len(rows), 'compacted', sum(r['compacted'] for r in rows), 'chars min/med/max', min(r['chars'] for r in rows),
              int(statistics.median(r['chars'] for r in rows)), max(r['chars'] for r in rows),
              'macro share min/max', round(min(r['share'] for r in rows), 3), round(max(r['share'] for r in rows), 3),
              'macro chars med', int(statistics.median(r['macro_chars'] for r in rows)))
print('TOTAL', tot, 'compacted', comp, 'max chars', maxlen, 'share range', round(min(shares), 3), round(max(shares), 3), 'median', round(statistics.median(shares), 3))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
