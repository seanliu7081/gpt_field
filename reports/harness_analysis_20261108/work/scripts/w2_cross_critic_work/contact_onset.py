"""For every forbidden-contact episode onset (h1_collision / cross_eval spans), the active segment's profile and the
remaining distance from the measured grip site to that segment's commanded endpoint.
A 'fast until the last 2 cm' approach would meet any onset with remaining distance > 0.02 m at fast speed."""
import json
import numpy as np
SPANS = {  # run dir -> list of (onset_action, label)  from contact_spans.md
 "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001": [(114,"palm-mug"),(261,"palm-mug"),(420,"palm-mug")],
 "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001": [(120,"palm-mug"),(141,"mug-can")],
 "agent_speedup_v2_uc71_default_routing_t04_20261103_001": [(137,"palm-mug")],
 "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001": [(37,"finger-bottle"),(223,"pot-bottle"),(510,"pot-burner"),(540,"pot-burner")],
 "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001": [(548,"pot-burner"),(626,"pot-burner")],
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001": [(148,"robot-cabinet"),(826,"robot-cabinet"),(841,"bowl-cabinet"),(1159,"robot-cabinet"),(1227,"robot-cabinet"),(2075,"palm-moka"),(2217,"palm-moka")],
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003": [(223,"robot-cabinet"),(667,"bowl-cabinet"),(1732,"palm-moka"),(1813,"palm-moka"),(1960,"palm-moka")],
 "foundation_vision_t08_bottle_obstacle_20261105_001": [(277,"pot-burner"),(299,"pot-burner")],
}
rows = []
for run, spans in SPANS.items():
    lines = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    fb = {l["step"]: l["feedback"] for l in lines if l.get("event") == "feedback"}
    for onset, label in spans:
        # contact spans are 1-based action indices in contact_spans.md; feedback 'step' is 0-based -> use onset-1
        f = fb.get(onset - 1) or fb.get(onset)
        fr = (f.get("primitive_feedback") or {}).get("fragment") if f else None
        if not fr:
            rows.append({"run": run[:40], "onset": onset, "label": label, "segment": None}); continue
        i = fr.get("current_segment_index")
        seg = next((s for s in fr.get("segments", []) if s.get("index") == i), {})
        req = seg.get("request") or {}
        pos = np.array(fr["measured_pose_wxyz"][:3]); tgt = seg.get("target_xyz_m") or fr.get("target_xyz_m")
        rem = float(np.linalg.norm(np.array(tgt) - pos)) if tgt else None
        entry = seg.get("entry_pose_wxyz")
        L = float(np.linalg.norm(np.array(tgt) - np.array(entry[:3]))) if (tgt and entry) else None
        rows.append({"run": run.replace("agent_speedup_v2_uc71_","")[:34], "onset": onset, "label": label, "skill": fr.get("current_skill"),
                     "profile": req.get("profile"), "gripper": fr.get("gripper_command"), "segment_len_cm": round(100*L,1) if L else None,
                     "remaining_to_target_cm": round(100*rem,1) if rem is not None else None})
for r in rows: print(r)
appr = [r for r in rows if r.get("profile") == "contact" and r.get("skill") in ("approach",)]
far = [r for r in appr if (r["remaining_to_target_cm"] or 0) > 2.0]
print(f"\nonsets in contact-profile approach: {len(appr)}; with >2 cm still to go (would be at fast speed under 'fast until last 2 cm'): {len(far)}")
json.dump(rows, open("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/cross_critic_work/data/contact_onset.json","w"), indent=1)
