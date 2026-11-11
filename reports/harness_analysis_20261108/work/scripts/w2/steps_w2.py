"""Compact per-step fragment trace from trace.jsonl feedback rows (skips heavy executed rows by prefix)."""
import json, sys
trace, out = sys.argv[1], sys.argv[2]
rows = []
with open(trace) as f:
    for line in f:
        if not line.startswith('{"event": "feedback"'):
            continue
        r = json.loads(line)
        fb = r["feedback"]; pf = fb.get("primitive_feedback") or {}; fr = pf.get("fragment") or {}
        segs = fr.get("segments") or []
        ci = fr.get("current_segment_index")
        seg = segs[ci] if isinstance(ci, int) and ci < len(segs) else {}
        req = seg.get("request") or {}
        rows.append({"step": r["step"], "status": fr.get("status"), "reason": fr.get("reason"),
                     "seg": ci, "skill": fr.get("current_skill"), "profile": req.get("profile"),
                     "seg_status": fr.get("current_segment_status"), "seg_reason": seg.get("reason"),
                     "seg_actions": fr.get("current_segment_actions"),
                     "pos_err": fr.get("position_error_m"), "ori_err": fr.get("orientation_error_rad"),
                     "route_rem": seg.get("route_remaining_m"), "has_route": "routing" in seg,
                     "dxyz": fb.get("measured_translation_delta_m"), "gap": fb.get("finger_gap_mm"),
                     "act": fb.get("normalized_action"), "n_segs": len(segs),
                     "pending": fr.get("pending_segment_indices"), "ptype": pf.get("type"), "prim": pf.get("primitive"),
                     "stall_evidence": seg.get("stall_evidence")})
json.dump(rows, open(out, "w"))
print(len(rows), "feedback rows")
