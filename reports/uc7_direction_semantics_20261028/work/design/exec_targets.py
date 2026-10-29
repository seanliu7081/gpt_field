"""Print executive decisions' absolute xyz targets and gripper-open stages for a run (responses only)."""
import json, glob, os, sys
run = sys.argv[1]; lo, hi = int(sys.argv[2]), int(sys.argv[3])
for f in sorted(glob.glob(os.path.join(run, "*.response.json"))):
    n = int(os.path.basename(f)[:6])
    if not lo <= n <= hi: continue
    d = json.load(open(f))
    for item in d.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") != "output_text": continue
            try: j = json.loads(c["text"])
            except Exception: continue
            segs = (j.get("execution") or {}).get("segments") or []
            mo = j.get("motion")
            desc = [(s["skill"], s.get("xyz_m"), s.get("gripper")) for s in segs]
            if mo: desc.append(("motion", mo.get("position_mode"), mo.get("xyz_m"), mo.get("gripper")))
            print(n, j.get("type"), desc, "|", j.get("plan", "")[:110].replace("\n", " "))
