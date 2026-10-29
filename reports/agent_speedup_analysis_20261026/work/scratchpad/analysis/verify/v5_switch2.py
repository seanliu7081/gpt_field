import json
from collections import Counter
base = '/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/'
I = {rid: json.load(open(base + 'v5_incs_%s.json' % rid)) for rid in ('prep_001', 'prep_002', 'prep_003', 'prep_004')}
# compare prefix increments
a, b = [x[1] for x in I['prep_001']], [x[1] for x in I['prep_003']]
c, d = [x[1] for x in I['prep_002']], [x[1] for x in I['prep_004']]
def first_diff(x, y):
    for i, (u, v) in enumerate(zip(x, y)):
        if u != v: return i
    return min(len(x), len(y))
print('prep_001 vs prep_003 first differing increment index', first_diff(a, b))
print('prep_002 vs prep_004 first differing increment index', first_diff(c, d))
print('prep_001 vs prep_002 first diff', first_diff(a, c))
# cumulative-char sets
ca = set(x[2] for x in I['prep_001']); cb = set(x[2] for x in I['prep_003'])
print('max normal inc first 279: p1', max(a[:279]), 'p3', max(b[:279]))
for rid, sw in (('prep_001', 279), ('prep_003', 279)):
    incs = I[rid]
    pre = incs[:sw]; post = incs[sw:]
    t0, t1 = pre[0][0], pre[-1][0]
    print('====', rid)
    print(' pre: deltas', len(pre), 'chars', pre[-1][2], 'first t %.3f last t %.3f' % (t0, t1), 'rate %.1f d/s %.1f c/s' % (len(pre) / (t1 - t0), pre[-1][2] / (t1 - t0)), 'max inc', max(x[1] for x in pre))
    ts, te = post[0][0], post[-1][0]
    pc = post[-1][2] - pre[-1][2]
    print(' post: deltas', len(post), 'chars', pc, 'start t %.3f end t %.3f' % (ts, te), 'rate %.2f d/s %.1f c/s' % (len(post) / (te - ts), pc / (te - ts)))
    cnt = Counter(x[1] for x in post)
    print(' post inc distribution (top 15):', cnt.most_common(15))
    big = [x[1] for x in post if x[1] >= 16]
    print(' post incs >=16:', len(big), 'sum', sum(big), 'share of post chars %.4f' % (sum(big) / pc), 'share of total chars %.4f' % (sum(big) / post[-1][2]))
    four = [x[1] for x in post if x[1] == 4]
    print(' post incs ==4:', len(four))
    # minute-level stability: deltas per 10-s window
    w = Counter(int((x[0] - ts) // 10) for x in post)
    print(' deltas per 10 s window:', [w[k] for k in sorted(w)])
    wc = Counter()
    for x in post: wc[int((x[0] - ts) // 10)] += x[1]
    print(' chars per 10 s window:', [wc[k] for k in sorted(wc)])
# normal completions
for rid in ('prep_002', 'prep_004'):
    incs = I[rid]
    t0, t1 = incs[0][0], incs[-1][0]
    print(rid, 'deltas', len(incs), 'chars', incs[-1][2], 'rate %.1f d/s %.1f c/s' % (len(incs) / (t1 - t0), incs[-1][2] / (t1 - t0)), 'max inc', max(x[1] for x in incs))
# where the analysts' "830" and "873" fall
for rid, target in (('prep_001', 830), ('prep_003', 873)):
    for i, x in enumerate(I[rid]):
        if x[2] >= target:
            print(rid, 'cum reaches', target, 'at index', i, 'count', i + 1, 't %.3f' % x[0]); break
