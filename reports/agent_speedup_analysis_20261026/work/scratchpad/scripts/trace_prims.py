#!/usr/bin/env python3
"""Read-only: count executed primitives / gripper commands / step ranges in trace.jsonl.
Usage: python3 -I trace_prims.py <run_dir> [...]
"""
import collections, json, sys
for run in sys.argv[1:]:
    prim = collections.Counter(); ranges = {}; first_succ = None; segs = []
    grip = collections.Counter(); last = None
    with open(run + '/trace.jsonl') as f:
        for line in f:
            e = json.loads(line)
            if e.get('event') == 'executed':
                p = e.get('primitive'); prim[p] += 1; s = e.get('step')
                ranges.setdefault(p, [s, s])[1] = s
                if last is None or last[0] != p:
                    segs.append([p, s, s])
                else:
                    segs[-1][2] = s
                last = (p,)
                if e.get('success') and first_succ is None:
                    first_succ = s
                a = e.get('action')
                if isinstance(a, list) and len(a) == 7:
                    grip['close' if a[6] > 0 else ('open' if a[6] < 0 else 'neutral')] += 1
    print('==', run)
    print('   executed by primitive:', dict(prim), '| first raw success step:', first_succ, '| gripper:', dict(grip))
    print('   contiguous primitive blocks:', [(p, a, b) for p, a, b in segs][:40])
