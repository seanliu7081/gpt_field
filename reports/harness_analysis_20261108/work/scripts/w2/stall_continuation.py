"""For each stall-triggered call, compare the discarded pending segments of the stalled fragment with the next call's segments."""
import json, sys
d = json.load(open(sys.argv[1]))
def sig(s):
    return (s.get('skill'), s.get('gripper'), s.get('profile'))
res = {'stalls': 0, 'with_pending': 0, 'next_starts_with_pending_skills': 0, 'rows': []}
for run, rows in d.items():
    for i, r in enumerate(rows):
        if r['trigger_reason'] != 'segment_stall':
            continue
        res['stalls'] += 1
        prev = rows[i - 1]
        prev_segs = (prev['execution'] or {}).get('segments') or []
        # stalled segment index from stage outcomes
        so = r['trigger_stage_outcomes'] or []
        idx = None
        for s in so:
            if s.get('reason') == 'segment_stall' and s.get('path', '').startswith('execution.evidence.segments['):
                idx = int(s['path'].split('[')[1].split(']')[0])
        ev = r['trigger_evidence'] or {}
        if idx is None:
            idx = ev.get('current_segment_index')
        pending = prev_segs[idx + 1:] if idx is not None else []
        nxt = (r['execution'] or {}).get('segments') or ([] if r['motion'] is None else [{'skill': 'motion'}])
        stalled = prev_segs[idx] if idx is not None and idx < len(prev_segs) else None
        cont = bool(pending) and [sig(s) for s in nxt[:len(pending)]] == [sig(s) for s in pending]
        res['rows'].append({'run': run[23:], 'call': r['ordinal'], 'stalled_skill': (stalled or {}).get('skill'),
                            'stalled_profile': (stalled or {}).get('profile'), 'pending': [sig(s) for s in pending],
                            'next': [sig(s) for s in nxt], 'next_continues_pending': cont})
        if pending:
            res['with_pending'] += 1
            if cont:
                res['next_starts_with_pending_skills'] += 1
json.dump(res, open(sys.argv[2], 'w'), indent=1)
print({k: v for k, v in res.items() if k != 'rows'})
for row in res['rows']:
    print(row['run'][:40], row['call'], row['stalled_skill'], row['stalled_profile'], 'pending', row['pending'], '-> next', row['next'][:3], row['next_continues_pending'])
