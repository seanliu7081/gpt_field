#!/usr/bin/env python3
"""Read-only: per-step measured end-effector translation from trace executed events (state -> next_state)."""
import json, math, sys, statistics
for run in sys.argv[1:]:
    prev = None; d = []
    with open(run + '/trace.jsonl') as f:
        for line in f:
            e = json.loads(line)
            if e.get('event') == 'action':
                prev = e.get('state')
            elif e.get('event') == 'executed' and prev is not None:
                ns = e.get('next_state')
                if isinstance(ns, list) and isinstance(prev, list):
                    d.append(math.dist(prev[:3], ns[:3]) * 1000)
    if not d:
        print(run, 'no data'); continue
    lt1 = sum(1 for x in d if x < 1.0); lt05 = sum(1 for x in d if x < 0.5)
    print('%s steps=%d median_mm=%.2f mean_mm=%.2f p90_mm=%.2f  <1mm: %d (%.0f%%)  <0.5mm: %d (%.0f%%) total_path_m=%.2f' % (
        run, len(d), statistics.median(d), statistics.mean(d), sorted(d)[int(.9 * (len(d) - 1))], lt1, 100 * lt1 / len(d), lt05, 100 * lt05 / len(d), sum(d) / 1000))
