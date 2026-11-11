"""Print model decisions (segments + obstacle_map) at refreshed steps from trace.jsonl."""
import json, sys
run = sys.argv[1]
rows = [json.loads(l) for l in open(f"{run}/trial/trace.jsonl")]
last = None
for r in rows:
    if r["event"] != "action":
        continue
    d = r["diagnostics"]
    if not d.get("refreshed"):
        continue
    p = (d.get("routing") or {}).get("params") or {}
    key = json.dumps(p, sort_keys=True)
    if key == last:
        continue
    last = key
    print(f"--- step {r['step']} type={p.get('type')} latency={d.get('api_latency_s')}")
    print("  why:", (p.get("why") or "")[:300])
    om = p.get("obstacle_map")
    if om is not None:
        print("  obstacle_map: clearance", om["clearance_m"])
        for b in om["avoid_boxes"]:
            print("     box", [round(x, 3) for x in b["lower_xyz_m"]], [round(x, 3) for x in b["upper_xyz_m"]])
    else:
        print("  obstacle_map: null (reuse)")
    ex = p.get("execution")
    if ex:
        for s in ex["segments"]:
            print("   seg", s["skill"], s["profile"], s["gripper"], s["xyz_m"], "q=", s["quaternion_wxyz"] and [round(x, 3) for x in s["quaternion_wxyz"]], "max", s["max_steps"], "obs", s["observation_after"], "route", s.get("route"))
    mo = p.get("motion")
    if mo:
        print("   motion", mo)
    ma = p.get("macro")
    if ma:
        print("   macro", ma)
