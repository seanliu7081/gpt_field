"""Routing sub-reason of every route_unavailable fragment end, and the T08 first-plan envelope numbers, from trace feedback rows."""
import json, sys
for name, run in [("T04", "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001"),
                  ("T08", "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001"),
                  ("stove", "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001"),
                  ("d001", "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001")]:
    with open(f"/workspace/gpt_field/runs/{run}/trial/trace.jsonl") as f:
        for line in f:
            if not line.startswith('{"event": "feedback"'):
                continue
            r = json.loads(line)
            fr = ((r["feedback"].get("primitive_feedback") or {}).get("fragment")) or {}
            if fr.get("reason") in ("route_unavailable", "target_envelope_rejected"):
                segs = fr.get("segments") or []
                ci = fr.get("current_segment_index")
                seg = segs[ci] if isinstance(ci, int) and ci < len(segs) else {}
                rt = seg.get("routing") or {}
                print(name, "step", r["step"], fr.get("reason"), "| routing.reason:", rt.get("reason"),
                      "| pos_err %.3f ori_err %.3f" % (fr.get("position_error_m") or -1, fr.get("orientation_error_rad") or -1),
                      "| clearance", rt.get("clearance_m"), "| boxes", rt.get("obstacle_count"), "| map_source", rt.get("map_source"))
