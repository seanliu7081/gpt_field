import json, sys
import numpy as np
from pathlib import Path
for run in sys.argv[1:]:
    ev = Path(run)/'trial'/'episode.events.jsonl'
    ex = {}
    with open(ev) as f:
        for line in f:
            d = json.loads(line)
            if d.get('event') == 'executed':
                ex[d['step']] = (d['wall_elapsed_s'], d.get('decision_latency_s') or 0.0)
    steps = sorted(ex)
    dts = []; dec_total = 0.0
    for a, b in zip(steps[:-1], steps[1:]):
        dt = ex[b][0] - ex[a][0]; dl = ex[b][1]
        if dl > 0.05:
            dec_total += dl
            dts.append(dt - dl)
        else:
            dts.append(dt)
    dts = np.array(dts)
    print(f"{run[:62]:62s} steps={len(steps)} span={ex[steps[-1]][0]-ex[steps[0]][0]:.1f}s decision_latency(sum,>0.05s)={dec_total:.1f}s non-decision per-step mean={dts.mean()*1000:.0f}ms median={np.median(dts)*1000:.0f}ms p95={np.percentile(dts,95)*1000:.0f}ms sum={dts.sum():.1f}s")
