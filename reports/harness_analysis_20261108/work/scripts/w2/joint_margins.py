import json, sys
LIM = [(-2.8973, 2.8973), (-1.7628, 1.7628), (-2.8973, 2.8973), (-3.0718, -0.0698), (-2.8973, 2.8973), (-0.0175, 3.7525), (-2.8973, 2.8973)]
d = json.load(open(sys.argv[1]))
out = {}
for run, rows in d.items():
    hits = []
    for r in rows:
        q = r['joints']
        if not q: continue
        for j, (lo, hi) in enumerate(LIM):
            m = min(q[j] - lo, hi - q[j])
            if m < 0.02:
                hits.append((r['ordinal'], 'joint%d' % (j + 1), round(m, 4), 'upper' if hi - q[j] < q[j] - lo else 'lower'))
    out[run] = hits
    print(run[23:], len(set(h[0] for h in hits)), 'requests within 0.02 rad;', hits)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
