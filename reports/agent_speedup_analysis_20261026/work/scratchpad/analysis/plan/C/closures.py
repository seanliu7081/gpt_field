"""Plan C: classify every close episode in uc7 runs by settled finger gap (read-only, stdlib).
settled gap = gap 15 executed steps after the close command started (min-close is 12 steps).
empty: settled gap < 3 mm (fingers met: nothing between the pads)."""
import json
ROOT = '/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_'
RUNS = ['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t6_009','t6_011','t7_001','t8_001','t9_001']
tot = {'closes': 0, 'empty': 0}
out = {}
for key in RUNS:
    rows = []
    with open(ROOT + key + '/episode.events.jsonl') as f:
        for line in f:
            d = json.loads(line)
            if d.get('event') == 'executed':
                rows.append(d)
    s = json.load(open(ROOT + key + '/summary.json'))
    cmd = 'open'; start = None; res = []
    for i, d in enumerate(rows):
        a = d.get('action') or []; st = d.get('state') or []
        if len(a) < 7 or len(st) < 8: continue
        new = 'close' if a[6] > 0 else 'open' if a[6] < 0 else cmd
        if new != cmd:
            cmd = new
            if cmd == 'close': start = i; res.append({'step': d.get('step'), 'settled_gap_mm': None})
        if cmd == 'close' and start is not None and i - start == 15:
            res[-1]['settled_gap_mm'] = round((st[6] - st[7]) * 1000, 1)
    empty = [r for r in res if r['settled_gap_mm'] is not None and r['settled_gap_mm'] < 3.0]
    out[key] = {'raw': s.get('task_success'), 'closes': len(res), 'empty': len(empty), 'detail': res}
    tot['closes'] += len(res); tot['empty'] += len(empty)
    print(key, 'raw', s.get('task_success'), 'closes', len(res), 'empty', len(empty), [r['settled_gap_mm'] for r in res])
print(tot)
json.dump(out, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/C/closures.json', 'w'), indent=1)
