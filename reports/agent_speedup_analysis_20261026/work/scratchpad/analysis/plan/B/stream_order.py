"""Plan B: how much visible output precedes the action in uc7 executive responses. Read-only.

For each uc7 execution response: visible tokens (output - reasoning), the share of the
JSON text taken by belief/plan/why (emitted before type/motion/macro/execution in the
schema order), and an estimate of seconds that action-first ordering plus early
dispatch could remove: prose_share * visible_tokens * 0.0212 s/token (the uc7 OLS
visible-token coefficient, model_io.md section 1.2). Estimate only.
"""
import json, glob, os, statistics

R = '/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_'
RUNS = ['t0_001', 't1_001', 't2_002', 't3_003', 't3_005', 't4_001', 't5_001', 't5_002',
        't6_009', 't6_011', 't7_001', 't8_001', 't9_001']
COEF = 0.0212
rows = []
for name in RUNS:
    for f in sorted(glob.glob(R + name + '/requests/*.response.json')):
        r = json.load(open(f))
        text = None
        for item in r.get('output', []):
            if item.get('type') == 'message':
                text = item['content'][0]['text']
        if text is None:
            continue
        try:
            obj = json.loads(text)
        except Exception:
            continue
        if not isinstance(obj, dict) or 'belief' not in obj:
            continue
        u = r.get('usage') or {}
        out = u.get('output_tokens') or 0
        reas = (u.get('output_tokens_details') or {}).get('reasoning_tokens') or 0
        vis = out - reas
        prose = sum(len(json.dumps(obj[k])) + len(k) + 4 for k in ('belief', 'plan', 'why'))
        share = prose / max(1, len(text))
        rows.append({'run': name, 'chars': len(text), 'prose_chars': prose, 'share': share, 'visible': vis,
                     'reasoning': reas, 'save_s': share * vis * COEF, 'type': obj.get('type')})
print('responses', len(rows))
print('visible tokens mean', round(statistics.mean(r['visible'] for r in rows), 1))
print('prose share of output text: mean', round(statistics.mean(r['share'] for r in rows), 3),
      'median', round(statistics.median(r['share'] for r in rows), 3))
print('estimated per-call saving (s): mean', round(statistics.mean(r['save_s'] for r in rows), 2),
      'median', round(statistics.median(r['save_s'] for r in rows), 2))
raw = {'t0_001', 't1_001', 't2_002', 't3_003', 't3_005', 't4_001', 't5_001', 't5_002', 't7_001', 't8_001', 't9_001'}
per = {}
for r in rows:
    per.setdefault(r['run'], 0.0)
    per[r['run']] += r['save_s']
print('per-run estimated saving (s):', {k: round(v, 1) for k, v in per.items()})
print('raw-11 mean per-run saving (s):', round(statistics.mean(per[k] for k in raw), 1))
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stream_order.json'), 'w'))
