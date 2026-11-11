import json, glob, sys, os
run = sys.argv[1]
req = os.path.join(run, 'trial', 'requests')
prev = None
rows = []
for f in sorted(glob.glob(os.path.join(req, '*.result.json'))):
    d = json.load(open(f))
    b = d.get('budget', {})
    sf = f.replace('.result.json', '.stream.json')
    s = json.load(open(sf)) if os.path.exists(sf) else {}
    rq = f.replace('.result.json', '.request.json')
    rsz = os.path.getsize(rq) if os.path.exists(rq) else None
    cur = dict(calls=b.get('api_calls'), inp=b.get('input_tokens'), out=b.get('output_tokens'), cached=b.get('cached_input_tokens'), reas=b.get('reasoning_tokens'), el=b.get('elapsed_s'))
    if prev is None:
        delta = cur
    else:
        delta = {k: (cur[k] - prev[k]) if (cur[k] is not None and prev[k] is not None) else None for k in cur}
    prev = cur
    rows.append((os.path.basename(f)[:6], d.get('request_kind'), d.get('status'), delta, rsz, s.get('elapsed_s'), s.get('headers_received_s'), s.get('first_visible_output_s'), s.get('visible_characters'), s.get('phase')))
print('idx kind status d_inp d_out d_cached d_reas d_el req_bytes stream_el hdr_s first_vis_s vis_chars phase')
for r in rows:
    d = r[3]
    print(r[0], r[1], r[2], d['inp'], d['out'], d['cached'], d['reas'], None if d['el'] is None else round(d['el'],2), r[4], None if r[5] is None else round(r[5],2), r[6], r[7], r[8], r[9])
