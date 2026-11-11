"""Attribute per-step increments of forbidden contact points (sim_status.collision) to the active fragment skill."""
import json, sys
from collections import Counter, defaultdict
trace, steps_json = sys.argv[1], sys.argv[2]
skill = {r["step"]: (r["skill"], r["profile"]) for r in json.load(open(steps_json))}
prev = 0; inc = Counter(); nsteps = Counter(); keys_seen = None
with open(trace) as f:
    for line in f:
        if not line.startswith('{"event": "executed"'):
            continue
        r = json.loads(line)
        col = (r.get("sim_status") or {}).get("collision") or {}
        cnt = (col.get("counts") or {})
        if keys_seen is None: keys_seen = list(col.keys())[:20]
        cur = cnt.get("forbidden_contact_points", 0) or 0
        d = cur - prev; prev = cur
        if d > 0:
            inc[skill.get(r["step"])] += d; nsteps[skill.get(r["step"])] += 1
print("collision keys", keys_seen)
print("forbidden points by skill:", dict(inc), "| steps with new forbidden points:", dict(nsteps), "| total", sum(inc.values()))
