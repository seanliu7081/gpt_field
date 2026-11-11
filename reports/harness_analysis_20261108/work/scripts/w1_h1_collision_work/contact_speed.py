"""Contact-profile steps far (>20 mm) from their own stage target: how many, and how many would a
transit-speed guarded approach need for the same path (at each run's measured transit mm/act)?"""
import json, math
from collections import defaultdict
W = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/h1_collision_work"
RUNS = {
 "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001": 0.158,
 "agent_speedup_v2_uc71_can_obstacle_t04_20261103_001": 0.172,
 "agent_speedup_v2_uc71_default_routing_t04_20261103_001": 0.175,
 "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001": 0.155,
 "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001": 0.179,
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001": 0.198,
 "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003": 0.178,
}
FORB = ["robot_scene", "robot_nonfinger_manipulated_object", "robot_object", "manipulated_object_scene"]
for run, per_act in RUNS.items():
    rows = [json.loads(l) for l in open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl")]
    fb = {r["step"]: r["feedback"] for r in rows if r["event"] == "feedback"}
    t = json.load(open(f"{W}/data/{run}_contacts.json"))["table"]
    tr_steps = tr_path = 0.0
    far_steps = far_path = 0
    far_forb = 0
    prev = None
    for r in t:
        p = r["eef_after"]
        d = math.dist(p, prev) if prev else 0.0
        prev = p
        if r["profile"] == "transit" and r["skill"] in ("move_to_pose", "move_relative"):
            tr_steps += 1; tr_path += d
        if r["profile"] != "contact" or r["skill"] not in ("approach", "move_to_pose", "move_relative"): continue
        frag = ((fb.get(r["step"]) or {}).get("primitive_feedback") or {}).get("fragment") or {}
        err = frag.get("position_error_m")
        if err is not None and err > 0.02:
            far_steps += 1; far_path += d
            if any(r["inc"].get(c) for c in FORB): far_forb += 1
    transit_speed = tr_path / max(tr_steps, 1)
    need = far_path / transit_speed if transit_speed else 0
    print(f"{run[26:66]} contact-profile steps >20mm from target: {far_steps} (path {far_path:.2f} m, forbidden-contact steps {far_forb}); "
          f"at measured transit {1000*transit_speed:.1f} mm/act they need ~{need:.0f} -> save ~{far_steps-need:.0f} act = {per_act*(far_steps-need):.0f} s")
