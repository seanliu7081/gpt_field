"""Offline: per-step forbidden contact increments joined with fragment stage info.
Usage: python contact_phases.py RUN_DIR [OUT_JSON]
"""
import json, sys
from collections import defaultdict, Counter

run = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None
rows = [json.loads(l) for l in open(f"{run}/trial/trace.jsonl")]
actions = {r["step"]: r for r in rows if r["event"] == "action"}
executed = {r["step"]: r for r in rows if r["event"] == "executed"}
feedback = {r["step"]: r for r in rows if r["event"] == "feedback"}
CLASSES = ["robot_scene", "robot_nonfinger_manipulated_object", "robot_object", "manipulated_object_scene",
           "finger_object", "manipulated_object_support", "finger_articulated_part"]
prev = {c: 0 for c in CLASSES}
prev_pairs = {}
table = []
for step in sorted(executed):
    ex = executed[step]
    col = ex["sim_status"].get("collision") or {}
    counts = col.get("counts", {})
    inc = {c: counts.get("points_" + c, 0) - prev[c] for c in CLASSES}
    for c in CLASSES:
        prev[c] = counts.get("points_" + c, 0)
    # pair increments
    pinc = {}
    for cls, plist in (col.get("pairs_by_class") or {}).items():
        if cls in ("finger_object", "manipulated_object_support"):
            continue
        for p in plist:
            key = (cls, tuple(p["pair"]))
            d = p["points"] - prev_pairs.get(key, 0)
            prev_pairs[key] = p["points"]
            if d:
                pinc["|".join([cls, *p["pair"]])] = d
    fb = feedback.get(step, {}).get("feedback", {})
    pf = fb.get("primitive_feedback") or {}
    frag = pf.get("fragment") or {}
    seg_i = frag.get("current_segment_index")
    segs = frag.get("segments") or []
    seg = next((s for s in segs if s.get("index") == seg_i), None)
    req = (seg or {}).get("request") or {}
    ac = actions.get(step, {})
    diag = ac.get("diagnostics", {})
    params = (diag.get("routing") or {}).get("params") or {}
    table.append({
        "step": step,
        "kind": diag.get("execution_kind"),
        "decision_type": params.get("type"),
        "refreshed": diag.get("refreshed"),
        "seg_index": seg_i, "skill": frag.get("current_skill") or (diag.get("primitive")),
        "profile": req.get("profile"), "route_used": bool((seg or {}).get("routing")),
        "gripper_command": fb.get("gripper_command"), "gap_mm": fb.get("finger_gap_mm"),
        "eef_after": (fb.get("after_state") or [None]*3)[:3],
        "inc": {k: v for k, v in inc.items() if v},
        "pair_inc": pinc,
        "macro": diag.get("macro"), "primitive": diag.get("primitive"),
    })

forb = ["robot_scene", "robot_nonfinger_manipulated_object", "robot_object", "manipulated_object_scene"]
# contiguous episodes per class (consecutive control steps with nonzero increments)
episodes = []
for cls in forb:
    cur = None
    for row in table:
        v = row["inc"].get(cls, 0)
        if v:
            if cur and row["step"] == cur["end"] + 1:
                cur["end"] = row["step"]; cur["samples"] += v; cur["steps"] += 1
                cur["skills"][f"{row['skill']}/{row['profile']}"] += 1
                for k, n in row["pair_inc"].items():
                    if k.startswith(cls + "|"): cur["pairs"][k[len(cls)+1:]] += n
                cur["grip"].add(row["gripper_command"])
            else:
                if cur: episodes.append(cur)
                cur = {"class": cls, "start": row["step"], "end": row["step"], "samples": v, "steps": 1,
                       "skills": Counter({f"{row['skill']}/{row['profile']}": 1}),
                       "pairs": Counter({k[len(cls)+1:]: n for k, n in row["pair_inc"].items() if k.startswith(cls + "|")}),
                       "grip": {row["gripper_command"]}, "eef_start": row["eef_after"]}
    if cur: episodes.append(cur)
for e in episodes:
    e["skills"] = dict(e["skills"]); e["pairs"] = dict(e["pairs"]); e["grip"] = sorted(map(str, e["grip"]))
episodes.sort(key=lambda e: e["start"])
summary = {"run": run, "steps": len(table),
           "totals": {c: sum(r["inc"].get(c, 0) for r in table) for c in CLASSES},
           "episodes_per_class": Counter(e["class"] for e in episodes),
           "samples_by_skill_profile": {}}
for cls in forb:
    agg = Counter()
    for r in table:
        if r["inc"].get(cls):
            agg[f"{r['skill']}/{r['profile']}/grip={r['gripper_command']}"] += r["inc"][cls]
    summary["samples_by_skill_profile"][cls] = dict(agg)
print(json.dumps(summary, indent=1, default=str))
for e in episodes:
    print(json.dumps(e, default=str))
if out:
    json.dump({"summary": summary, "episodes": episodes, "table": table}, open(out, "w"), default=str)
