"""Approximate per-step cost of deepcopy(self._feedback_history) (agent.py:258) as history grows,
using the run's final last_feedback record as a representative entry."""
import json, sys, time
from copy import deepcopy
d = json.load(open(sys.argv[1]))
entry = d["last_feedback"]
print("entry json chars", len(json.dumps(entry)))
for n in (5, 13, 21, 42):
    hist = [deepcopy(entry) for _ in range(n)]
    t = time.perf_counter(); reps = 20
    for _ in range(reps): deepcopy(hist)
    print(n, "entries -> %.1f ms per deepcopy" % ((time.perf_counter() - t) / reps * 1000))
