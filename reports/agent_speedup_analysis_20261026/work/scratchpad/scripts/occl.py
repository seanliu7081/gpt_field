import json, sys, os, glob, math
def rot(q):
    w,x,y,z = q; n = math.sqrt(w*w+x*x+y*y+z*z); w,x,y,z = w/n,x/n,y/n,z/n
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
def mm(a,b): return [[sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def mv(a,v): return [sum(a[i][k]*v[k] for k in range(3)) for i in range(3)]
def tmv(a,v): return [sum(a[k][i]*v[k] for k in range(3)) for i in range(3)]
run = sys.argv[1]
req = json.load(open(sorted(glob.glob(os.path.join(run, "requests", "*.request.json")))[-1]))
users = [i for i in req["payload"]["input"] if i.get("role") == "user"]
f = {}
for part in users[-1]["content"]:
    if part.get("type") == "input_text":
        try:
            g = json.loads(part["text"])
            if isinstance(g, dict) and "current_decision" in g: f = g
        except Exception: pass
ea = f["current_decision"].get("evidence_acquisition") or {}
p = ea.get("witness_world_point_m")
print(run, "| view reason:", ea.get("reason"), "| rejected:", ea.get("rejected"), "| witness:", p)
if p is None: sys.exit()
ents = json.load(open(os.path.join(run, "agent_stats.json")))["shared_geometry"]["entities"]
for ident, e in ents.items():
    if e["role"] in ("free_region", "destination_region") or e["attachment"] == "held":
        continue
    R = rot(e["quaternion_wxyz"]); pos = e["position_m"]
    unc = e["uncertainty_m"] + (0.02 if e.get("stale") else 0.)
    for i, b in enumerate(e["boxes"]):
        c = [pos[k] + mv(R, b["center_local_m"])[k] for k in range(3)]
        Rb = mm(R, rot(b["quaternion_wxyz"]))
        local = tmv(Rb, [p[k]-c[k] for k in range(3)])
        half = [h+unc for h in b["half_extents_m"]]
        margin = min(half[k]-abs(local[k]) for k in range(3))
        if margin >= -0.01:
            print("   %-28s box%d role=%-15s inside_padded=%s margin_m=%.4f (unc=%.3f stale=%s)" % (ident, i, e["role"], margin >= 0, margin, unc, e.get("stale")))
