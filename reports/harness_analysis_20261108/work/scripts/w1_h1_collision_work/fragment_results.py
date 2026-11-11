"""List fragment results (per execute decision): segment outcomes, routing status/reason, expanded boxes."""
import json, sys
run = sys.argv[1]
verbose = len(sys.argv) > 2
rows = [json.loads(l) for l in open(f"{run}/trial/trace.jsonl")]
seen = set()
for r in rows:
    if r["event"] != "action":
        continue
    fb = (r["diagnostics"].get("feedback") or {})
    if fb.get("action") != "execute":
        continue
    key = (fb.get("observed_at_step"), fb.get("reason"))
    if key in seen:
        continue
    seen.add(key)
    ev = fb.get("evidence") or {}
    print(f"=== feedback observed_at_step={fb.get('observed_at_step')} status={fb.get('status')} reason={fb.get('reason')} steps={fb.get('executed_steps')}")
    for s in ev.get("segment_trace") or ev.get("segments") or []:
        ro = s.get("routing") or {}
        rt = s.get("route_tracking") or {}
        line = f"   seg{s['index']} {s['skill']} {s['request']['profile']} status={s['status']} reason={s['reason']} actions={s['actions']} pos_err={s.get('position_error_m')}"
        if ro:
            line += f" | route status={ro.get('status')} reason={ro.get('reason')} straight={ro.get('straight_path_valid')} method={ro.get('method')} clearance={ro.get('clearance_m')} n_boxes={ro.get('obstacle_count')} intersect={ro.get('intersecting_box_count')} planner_calls={(ro.get('planner') or {}).get('calls')}"
        if rt.get("status") == "unavailable":
            line += f" | tracking unavailable reason={rt.get('reason')}"
        print(line)
        if s.get("detail"):
            print("      detail:", s["detail"][:200])
        if verbose and ro:
            print("      expanded:", [ (b['lower_xyz_m'], b['upper_xyz_m'], b['intersects_local_map']) for b in ro.get('expanded_boxes', [])])
            print("      start/end:", s.get('entry_pose_wxyz', [None]*3)[:3], s.get('target_xyz_m'))
