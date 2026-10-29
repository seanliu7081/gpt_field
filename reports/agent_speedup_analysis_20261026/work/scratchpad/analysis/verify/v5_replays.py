import json, sys
from collections import Counter, defaultdict
path = '/workspace/gpt_field/runs/speedup_next_20261026T101438Z/request_diagnostics.jsonl'
rows = [json.loads(l) for l in open(path)]
by = defaultdict(list)
for r in rows:
    by[r['request_id']].append(r)
for rid, rs in sorted(by.items()):
    print('=====', rid, len(rs))
    begin = [r for r in rs if r['event'] == 'request_begin'][0]
    print('begin', {k: begin[k] for k in begin if k not in ('monotonic_s',)})
    for r in rs:
        if r['event'] not in ('stream_event',) and 'public_output_text_chars' not in r or r['event'] in ('request_end', 'request_summary', 'summary'):
            pass
    evs = Counter(r['event'] for r in rs)
    print('events', evs)
    etypes = Counter(r.get('event_type') for r in rs if 'event_type' in r)
    print('event_types', etypes)
    # non-stream rows
    for r in rs:
        if 'public_output_text_chars' in r and r['event'] != 'request_end' and 'actual_http_attempts' not in r:
            continue
        if r['event'] in ('request_begin',):
            continue
        d = {k: v for k, v in r.items() if k not in ('monotonic_s', 'utc', 'diagnostic_source_sha256', 'payload_ordered_wire_sha256')}
        print('  ', json.dumps(d)[:900])
    # delta analysis
    deltas = [r for r in rs if r.get('event_type') == 'response.output_text.delta']
    prev = 0
    incs = []
    for r in deltas:
        c = r['public_output_text_chars']
        incs.append((r['elapsed_s'], c - prev, c))
        prev = c
    if not incs:
        continue
    print('n deltas', len(incs), 'final chars', incs[-1][2], 'max inc', max(i[1] for i in incs), 'first delta t', round(incs[0][0], 3), 'last delta t', round(incs[-1][0], 3))
    json.dump(incs, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v5_incs_%s.json' % rid, 'w'))
