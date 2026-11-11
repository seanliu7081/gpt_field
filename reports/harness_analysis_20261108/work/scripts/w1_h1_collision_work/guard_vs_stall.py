"""For each segment_stall handoff: steps between the first guarded-motion fire inside the stalled segment and
the recorded stall handoff (steps the guard would have saved)."""
import json
import numpy as np
W = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/h1_collision_work"
RUNS = {
 "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001": (0.10, 0.158),
 "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001": (0.05, 0.172),
 "agent_speedup_v2_uc71_default_routing_t04_20261103_001": (0.05, 0.175),
 "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001": (0.10, 0.155),
 "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001": (0.10, 0.179),
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001": (0.10, 0.198),
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003": (0.10, 0.178),
}
K, CMIN, PMAX = 3, 0.002, 0.0003
saved_all = []; nofire = 0
for run, (scale, per_act) in RUNS.items():
    rows = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    fb = {r["step"]: r["feedback"] for r in rows if r["event"] == "feedback"}
    fires = set(); streak = 0
    for step in sorted(fb):
        f = fb[step]; na, md = f.get("normalized_action"), f.get("measured_translation_delta_m")
        if na is None or md is None: streak = 0; continue
        c = np.asarray(na[:3]) * scale; cn = float(np.linalg.norm(c))
        if cn < CMIN: streak = 0; continue
        prog = float(np.dot(np.asarray(md), c / cn))
        streak = streak + 1 if prog < PMAX else 0
        if streak >= K: fires.add(step)
    seen = set()
    for r in rows:
        if r["event"] != "action": continue
        f = r["diagnostics"].get("feedback") or {}
        if f.get("action") != "execute" or f.get("reason") != "segment_stall": continue
        k = f.get("observed_at_step")
        if k in seen: continue
        seen.add(k)
        segs = (f.get("evidence") or {}).get("segment_trace") or []
        seg = segs[-1]; n = seg.get("actions", 0)
        start = k - n
        first = next((s for s in range(start, k) if s in fires), None)
        if first is None: nofire += 1; continue
        saved_all.append((k - first, per_act))
print("stalls with a guard fire inside the stalled segment:", len(saved_all), "without:", nofire)
print("steps saved per stall: median", float(np.median([s for s, _ in saved_all])), "total", sum(s for s, _ in saved_all),
      "~seconds", round(sum(s * p for s, p in saved_all), 1))
