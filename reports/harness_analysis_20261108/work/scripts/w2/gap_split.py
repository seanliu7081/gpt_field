"""Split per-step wall into gap1 = executed(k) -> agent_action(k+1) and gap2 = agent_action(k+1) -> executed(k+1),
for steps whose decision made no model call; also event-record byte size by quintile."""
import json, sys
import numpy as np
ex, dec, calls, size_ex, size_dec = {}, {}, set(), {}, {}
with open(sys.argv[1]) as f:
    for line in f:
        r = json.loads(line)
        ev, st = r.get("event"), r.get("step")
        if ev == "executed":
            ex[st] = r["wall_elapsed_s"]; size_ex[st] = len(line)
        elif ev in ("agent_action", "skill_decision"):
            dec[st] = r["wall_elapsed_s"]; size_dec[st] = len(line)
            if r.get("api_calls_in_act"):
                calls.add(st)
# events: decision at step k is emitted with step=k (pre-step); executed with step=k
rows = []
for k in sorted(ex):
    if k - 1 in ex and k in dec and k not in calls:
        rows.append((k, dec[k] - ex[k - 1], ex[k] - dec[k], size_ex[k] + size_dec[k]))
a = np.array(rows)
q = np.array_split(np.arange(len(a)), 5)
print(json.dumps({"n": len(a),
  "gap1_q": [round(float(a[i, 1].mean()), 4) for i in q],
  "gap2_q": [round(float(a[i, 2].mean()), 4) for i in q],
  "event_bytes_per_step_q": [int(a[i, 3].mean()) for i in q]}))
