"""Action/path shares by profile, speeds, rotation-wait steps, realized rotation rate (T04 first fragment)."""
import json, sys
import numpy as np
for name in sys.argv[1:]:
    rows = json.load(open(f"data/steps_{name}.json"))
    n = len(rows)
    sp = np.array([np.linalg.norm(r["dxyz"]) * 1000 if r["dxyz"] else 0 for r in rows])
    cmd = np.array([np.linalg.norm(r["act"][:3]) * 100 if r["act"] else 0 for r in rows])  # normalized*0.1m -> mm
    def cls(r):
        if r["skill"] == "approach": return "approach"
        if r["skill"] in ("move_to_pose", "move_relative") and r["profile"] == "transit": return "transit"
        if r["skill"] == "set_gripper": return "gripper"
        return "other:" + str(r["skill"] or r["prim"])
    c = [cls(r) for r in rows]
    path = sp.sum()
    out = {}
    for k in sorted(set(c)):
        m = np.array([x == k for x in c])
        out[k] = {"actions_share": round(m.mean(), 3), "path_share": round(sp[m].sum() / path, 3), "mean_mm": round(sp[m].mean(), 2)}
    tm = np.array([x == "transit" for x in c])
    out["transit_p90_max"] = [round(float(np.percentile(sp[tm], 90)), 1), round(float(sp[tm].max()), 1)]
    out["transit_zero_cmd_steps"] = int(((cmd < 1e-6) & tm).sum())
    out["n"] = n
    print(name, json.dumps(out))
rows = json.load(open("data/steps_T04.json"))
print("T04 steps 1-70 (seg, pos_err mm, ori_err rad, cmd_mm):")
print([(r["step"], r["seg"], round(r["pos_err"] * 1000, 1), round(r["ori_err"], 3), round(np.linalg.norm(r["act"][:3]) * 100, 1)) for r in rows[1:71:3]])
