"""Recompute review context hash and approval binding (read-only) to confirm preparation is bound."""
import json, hashlib, sys, copy
def cs(v): return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
F = ("schema_version", "version", "id", "created_at", "task_key", "episode_index", "task_prompt",
     "initial_observation_sha256", "runtime_sha256", "draft_plan", "preparation")
for path in sys.argv[1:]:
    d = json.load(open(path))
    ctx = {f: d[f] for f in F}
    ok = cs(ctx) == d["context_sha256"]
    b = cs({"context_sha256": d["context_sha256"], "plan": d["plan"], "revision": d["revision"], "reviewed_steps": d["reviewed_steps"]})
    print(path.split("runs/")[-1], "context ok", ok, "binding ok", b == (d.get("approved") or {}).get("binding_sha256"))
    d2 = copy.deepcopy(ctx); d2["preparation"]["quarter_turns"] = 2
    print("   edited preparation changes context hash:", cs(d2) != d["context_sha256"])
