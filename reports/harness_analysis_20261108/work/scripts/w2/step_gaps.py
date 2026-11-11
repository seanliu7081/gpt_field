"""Per-step wall gap between consecutive 'executed' video events, excluding steps whose decision made a model call.
Also time from an executed event to the next agent_action event (decision+render) etc. Reports quintile means."""
import json, sys
import numpy as np
path = sys.argv[1]
ex, calls = {}, set()
dec = {}
with open(path) as f:
    for line in f:
        r = json.loads(line)
        ev, st = r.get("event"), r.get("step")
        if ev == "executed":
            ex[st] = r["wall_elapsed_s"]
        elif ev in ("agent_action", "skill_decision"):
            dec[st] = r["wall_elapsed_s"]
            if r.get("api_calls_in_act"):
                calls.add(st)
steps = sorted(ex)
gaps = []
for a, b in zip(steps, steps[1:]):
    if b == a + 1 and b not in calls:   # step b's decision made no model call
        gaps.append((b, ex[b] - ex[a]))
g = np.array([v for _, v in gaps])
idx = np.array([s for s, _ in gaps])
q = np.array_split(np.arange(len(g)), 5)
out = {"n_executed": len(steps), "n_call_steps": len(calls), "n_gaps": len(g),
       "mean": float(g.mean()), "median": float(np.median(g)),
       "quintile_means": [round(float(g[i].mean()), 4) for i in q],
       "quintile_medians": [round(float(np.median(g[i])), 4) for i in q]}
print(json.dumps(out))
