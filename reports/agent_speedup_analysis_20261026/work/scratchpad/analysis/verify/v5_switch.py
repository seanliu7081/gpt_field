import json, sys
from collections import Counter
base = '/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/'
for rid in ('prep_001', 'prep_003', 'prep_002', 'prep_004'):
    incs = json.load(open(base + 'v5_incs_%s.json' % rid))
    print('=====', rid, 'n', len(incs), 'final', incs[-1][2])
    # per-second bins
    bins = Counter(); cb = Counter()
    for t, inc, c in incs:
        bins[int(t)] += 1; cb[int(t)] += inc
    secs = sorted(bins)
    line = ' '.join('%d:%d/%d' % (s, bins[s], cb[s]) for s in secs[:40])
    print('per-second deltas/chars (first 40 s):', line)
    if rid in ('prep_001', 'prep_003'):
        # print increments sequence around the switch
        for i, (t, inc, c) in enumerate(incs):
            if 230 <= i <= 300:
                print('  i=%d t=%.3f inc=%d cum=%d' % (i, t, inc, c))
