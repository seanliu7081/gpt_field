import json
from collections import Counter
D = '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/data/'
trig = json.load(open(D + 'triggers.json'))
out = {}
for key in ['t04_t100', 't08_t100', 't08_stoveoff', 'drawer_001', 'drawer_003']:
    d = json.load(open(D + key + '.json'))
    rows = d['rows']; api = sorted(int(k) for k in d['api_steps'])
    calls = trig[key]
    by_step = {r['step']: r for r in rows}
    cls = Counter(); details = []
    for ci, c in enumerate(calls):
        if c['trigger'] != 'stall': continue
        # the stalled fragment ran from previous call step to this call step
        start = calls[ci-1]['step']; end = c['step']
        prev_resp = calls[ci-1]
        frag_rows = [by_step[s] for s in range(start, end) if s in by_step]
        last = frag_rows[-1]
        si = last['seg_index']; nseg = prev_resp['nseg']
        seg_rows = [r for r in frag_rows if r['seg_index'] == si]
        errs = [r['pos_err'] for r in seg_rows if r['pos_err'] is not None]
        mn = min(errs); en = errs[-1]
        imin = errs.index(mn)
        disp_last = [r['disp'] for r in seg_rows[-4:]]
        skill, prof = last['skill'], last['profile']
        if prof == 'transit' and mn < 0.010 and en > mn + 0.002:
            kind = 'transit_overshoot_false_stall'
        elif prof == 'transit':
            kind = 'transit_stall_far' if en > 0.03 else 'transit_stall_near'
        elif skill == 'approach' or prof == 'contact':
            kind = 'contact_stall'
        else:
            kind = 'other'
        last_seg = (si == nseg - 1)
        cls[(kind, 'last_segment' if last_seg else 'pending_discarded')] += 1
        details.append(dict(call=c['i'], step=end, kind=kind, skill=skill, profile=prof, seg=f'{si+1}/{nseg}', min_err_mm=round(mn*1000,1), end_err_mm=round(en*1000,1), steps=len(seg_rows), last4_disp_mm=[round(x*1000,1) for x in disp_last], gap=round(last['gap_mm'] or 0,1)))
    out[key] = dict(counts={f'{a}|{b}': v for (a, b), v in cls.items()}, details=details)
    print('=====', key, dict(cls))
    for x in details: print('   ', x)
json.dump(out, open(D + 'stall_class.json', 'w'), indent=1)
