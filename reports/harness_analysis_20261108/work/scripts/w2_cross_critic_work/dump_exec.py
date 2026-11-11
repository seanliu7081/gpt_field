"""Print each executive answer of a uc7.1-family run: type, segments (skill, xyz, profile, steps), obstacle map."""
import json, glob, os, sys
run = sys.argv[1]
base = f"/workspace/gpt_field/runs/{run}/trial/requests"
for p in sorted(glob.glob(base + "/*.response.json")):
    r = json.load(open(p))
    txt = None
    for o in r.get("output", []):
        if o.get("type") == "message":
            for c in o.get("content", []):
                if c.get("type") in ("output_text", "text"):
                    txt = c.get("text")
    if txt is None:
        print(os.path.basename(p), "no text"); continue
    try:
        a = json.loads(txt)
    except Exception as e:
        print(os.path.basename(p), "parse fail", e); continue
    idx = os.path.basename(p)[:6]
    print(f"--- {idx} type={a.get('type')}")
    print("  belief:", (a.get("belief") or "")[:300].replace("\n"," "))
    ex = a.get("execution") or {}
    for s in ex.get("segments", []) or []:
        print("   seg", s.get("skill"), s.get("xyz_m"), s.get("profile"), "max", s.get("max_steps"), "obs", s.get("observation_after"), "q", s.get("quaternion_wxyz"), "grip", s.get("gripper"), "route", bool(s.get("route")))
    om = a.get("obstacle_map")
    if om:
        print("   map clearance", om.get("clearance_m"), [ (b["lower_xyz_m"], b["upper_xyz_m"]) for b in om.get("avoid_boxes", [])])
    if a.get("motion"):
        print("   motion", a["motion"])
