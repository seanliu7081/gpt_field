"""Plan B: hedged duplicate requests under an independence assumption (estimate only). Read-only.

For each uc7 raw-success run, E[min(L1, L2)] over all ordered pairs of that run's non-first
call latencies (episode.events.jsonl skill_decision api_latency_s, via B/decisions.json),
and the same over first calls across runs.
"""
import json, os, statistics, itertools
B = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(B, 'decisions.json')))
RAW = ['t0_001', 't1_001', 't2_002', 't3_003', 't3_005', 't4_001', 't5_001', 't5_002', 't7_001', 't8_001', 't9_001']
tot_mean = tot_min = n = 0
for name in RAW:
    L = [r['lat'] for r in D[name]['rows'][1:]]
    tot_mean += statistics.mean(L) * len(L)
    tot_min += statistics.mean(min(a, b) for a, b in itertools.product(L, L)) * len(L)
    n += len(L)
print('non-first calls', n, 'mean', round(tot_mean / n, 2), 'E[min of 2 iid]', round(tot_min / n, 2),
      'saving per call', round((tot_mean - tot_min) / n, 2))
first = [D[x]['rows'][0]['lat'] for x in RAW]
print('first calls mean', round(statistics.mean(first), 1), 'E[min of 2]',
      round(statistics.mean(min(a, b) for a, b in itertools.product(first, first)), 1))
