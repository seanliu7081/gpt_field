import json, sys, numpy as np
from collections import Counter, defaultdict
D = '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/data/'
summary = {}
for key in ['t04_t100', 't08_t100', 't08_stoveoff', 'drawer_001', 'drawer_003']:
    d = json.load(open(D + key + '.json'))
    rows = d['rows']; api = set(int(k) for k in d['api_steps'])
    # segment boundaries: new segment when api call at this step, or seg_index changes
    segs = []; cur = None
    for r in rows:
        newfrag = r['step'] in api
        if cur is None or newfrag or r['seg_index'] != cur['seg_index'] or r['primitive'] != cur['primitive']:
            cur = {'start': r['step'], 'seg_index': r['seg_index'], 'primitive': r['primitive'], 'skill': r['skill'], 'profile': r['profile'], 'rows': []}
            segs.append(cur)
        cur['rows'].append(r)
    stats = defaultdict(lambda: Counter())
    tail = defaultdict(list)
    for s in segs:
        rs = s['rows']; k = (s['skill'], s['profile'])
        n = len(rs)
        errs = [r['pos_err'] for r in rs if r['pos_err'] is not None]
        rot = [r['rot_err'] for r in rs if r['rot_err'] is not None]
        end = rs[-1]
        stats[k]['segments'] += 1; stats[k]['steps'] += n
        if s['skill'] in ('move_to_pose', 'move_relative', 'approach') and errs:
            # steps after position error first < 10 mm
            idx10 = next((i for i, e in enumerate(errs) if e < 0.010), None)
            idx5 = next((i for i, e in enumerate(errs) if e < 0.005), None)
            t10 = n - 1 - idx10 if idx10 is not None else 0
            stats[k]['tail_steps_err_lt10mm'] += t10
            zero_cmd = sum(1 for r in rs if r['cmd_norm_norm'] < 1e-9)
            stats[k]['zero_translation_cmd_steps'] += zero_cmd
            tail[k].append((s['start'], n, round(errs[0]*1000,1), round(errs[-1]*1000,1), round(rot[0],2) if rot else None, round(rot[-1],3) if rot else None, t10, zero_cmd, end['frag_reason'] or end['frag_status']))
    summary[key] = {str(k): dict(v) for k, v in stats.items()}
    print('=====', key)
    for k, v in stats.items():
        print('  ', k, dict(v))
    for k in tail:
        if k[1] == 'transit':
            print('   transit segments (start, steps, err0 mm, err_end mm, rot0, rot_end, steps after err<10mm, zero-cmd steps, end):')
            for t in tail[k]: print('     ', t)
json.dump(summary, open(D + 'segment_summary.json', 'w'), indent=1)
