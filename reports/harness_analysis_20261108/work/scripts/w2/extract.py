"""Independent per-step extraction from runs/*/trial/trace.jsonl (verifier W2)."""
import json, sys, os, pickle
RUNS = "/workspace/gpt_field/runs"
RUNLIST = {
 "T04_t100": "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
 "T04_20mm": "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001",
 "T04_default": "agent_speedup_v2_uc71_default_routing_t04_20261103_001",
 "T08_t100": "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
 "T08_stoveoff": "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001",
 "drawer001": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001",
 "drawer003": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003",
 "H3_T04": "foundation_vision_t04_can_obstacle_20261105_001",
 "H3_T08": "foundation_vision_t08_bottle_obstacle_20261105_001",
}
FORB = ("robot_scene", "robot_nonfinger_manipulated_object", "robot_object", "manipulated_object_scene")
OUT = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
os.makedirs(OUT, exist_ok=True)

def pairs_flat(coll):
    out = {}
    for cls, rows in (coll.get("pairs_by_class") or {}).items():
        for r in rows:
            out[(cls, tuple(r["pair"]))] = r["points"]
    return out

def run(label):
    path = f"{RUNS}/{RUNLIST[label]}/trial/trace.jsonl"
    steps = {}
    prev_counts, prev_pairs = None, {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            ev = d.get("event")
            if ev == "reset":
                c = d["sim_status"]["collision"]
                prev_counts = c["counts"]; prev_pairs = pairs_flat(c)
            elif ev == "action":
                s = d["step"]
                diag = d.get("diagnostics") or {}
                st = steps.setdefault(s, {})
                st["state"] = d.get("state")
                st["action"] = d.get("action")
                st["primitive"] = diag.get("primitive")
                st["macro"] = diag.get("macro")
                st["api_calls"] = len(diag.get("api_calls") or [])
                st["api_latency_s"] = diag.get("api_latency_s")
                r = diag.get("routing") or {}
                st["decision_type"] = (r.get("params") or {}).get("type") if isinstance(r.get("params"), dict) else None
                st["has_new_decision"] = bool(diag.get("api_calls"))
                if st["has_new_decision"]:
                    st["params"] = r.get("params")
            elif ev == "executed":
                s = d["step"]
                c = d["sim_status"]["collision"]
                counts = c["counts"]; pairs = pairs_flat(c)
                st = steps.setdefault(s, {})
                st["next_state"] = d.get("next_state")
                st["inc"] = {k: counts.get(f"points_{k}", 0) - prev_counts.get(f"points_{k}", 0) for k in FORB}
                st["inc_allowed"] = {k: counts.get(f"points_{k}", 0) - prev_counts.get(f"points_{k}", 0)
                                     for k in ("finger_object", "finger_articulated_part", "manipulated_object_support")}
                st["forb_inc"] = counts["forbidden_contact_points"] - prev_counts["forbidden_contact_points"]
                st["substeps"] = counts["physics_substeps"] - prev_counts["physics_substeps"]
                dp = {}
                for k, v in pairs.items():
                    dv = v - prev_pairs.get(k, 0)
                    if dv:
                        dp[k] = dv
                st["pair_inc"] = dp
                st["success"] = d.get("success")
                prev_counts, prev_pairs = counts, pairs
            elif ev == "feedback":
                s = d["step"]
                fb = d["feedback"]
                st = steps.setdefault(s, {})
                st["fb_primitive"] = fb.get("primitive")
                st["measured_delta"] = fb.get("measured_translation_delta_m")
                st["before_state"] = fb.get("before_state"); st["after_state"] = fb.get("after_state")
                st["gripper_command"] = fb.get("gripper_command")
                st["finger_gap_mm"] = fb.get("finger_gap_mm")
                st["norm_action"] = fb.get("normalized_action")
                pf = fb.get("primitive_feedback") or {}
                frag = pf.get("fragment")
                if frag:
                    idx = frag.get("current_segment_index")
                    segs = frag.get("segments") or []
                    cur = segs[idx] if idx is not None and idx < len(segs) else None
                    st["frag"] = {
                        "status": frag.get("status"), "reason": frag.get("reason"),
                        "seg_index": idx, "skill": frag.get("current_skill"),
                        "seg_status": frag.get("current_segment_status"),
                        "seg_actions": frag.get("current_segment_actions"),
                        "target": frag.get("target_xyz_m"), "target_q": frag.get("target_quaternion_wxyz"),
                        "pos_err": frag.get("position_error_m"), "ori_err": frag.get("orientation_error_rad"),
                        "gripper": frag.get("gripper_command"),
                        "profile": (cur or {}).get("request", {}).get("profile"),
                        "routed": bool((cur or {}).get("routing")),
                        "routing": (cur or {}).get("routing"),
                        "route_tracking": (cur or {}).get("route_tracking"),
                        "seg_reason": (cur or {}).get("reason"),
                        "stall": (cur or {}).get("stall_evidence"),
                        "entry_step": (cur or {}).get("entry_at_fragment_step"),
                        "request": (cur or {}).get("request"),
                        "n_segs": len(segs),
                    }
                    if frag.get("status") in ("failed", "succeeded") :
                        st["frag_end"] = {"segments": [{k: v for k, v in sg.items() if k != "measurements"} for sg in segs]}
                lm = pf.get("local_motion")
                if lm:
                    st["local_motion"] = {k: lm.get(k) for k in ("status", "requested_target", "position_error_m", "orientation_error_rad", "requested_steps", "executed_steps")}
                st["fb_status"] = pf.get("status"); st["fb_reason"] = pf.get("reason")
    return steps

if __name__ == "__main__":
    labels = sys.argv[1:] or list(RUNLIST)
    for lab in labels:
        st = run(lab)
        pickle.dump(st, open(f"{OUT}/{lab}_steps.pkl", "wb"))
        tot = sum(v.get("forb_inc", 0) for v in st.values())
        print(lab, "steps", len(st), "forbidden_total", tot)
