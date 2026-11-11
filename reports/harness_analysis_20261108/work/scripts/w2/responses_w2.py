"""Parse every execution response: type, segment count, observation_after placement, requested steps, prose chars."""
import json, glob, sys
import numpy as np
RUNS = {"T04": "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001",
        "T08": "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001",
        "stove": "agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001",
        "d001": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001",
        "d003": "agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003"}
allrows = {}
for name, run in RUNS.items():
    rows = []
    for f in sorted(glob.glob(f"/workspace/gpt_field/runs/{run}/trial/requests/*.response.json")):
        d = json.load(open(f))
        msg = [o for o in d["output"] if o.get("type") == "message"][0]
        txt = [c["text"] for c in msg["content"] if c.get("type") == "output_text"][0]
        o = json.loads(txt)
        segs = ((o.get("execution") or {}).get("segments")) or []
        oa = [bool(s.get("observation_after")) for s in segs]
        rows.append({"file": f.split("/")[-1][:6], "type": o.get("type"), "n_segs": len(segs), "obs_after": oa,
                     "last_obs_after": oa[-1] if oa else None, "mid_obs_after": any(oa[:-1]) if oa else None,
                     "req_steps": sum(int(s.get("max_steps") or 0) for s in segs),
                     "skills": [(s.get("skill"), s.get("profile"), s.get("gripper")) for s in segs],
                     "chars": len(txt), "prose_chars": sum(len(o.get(k) or "") for k in ("belief", "plan", "why")),
                     "has_map": o.get("obstacle_map") is not None, "motion": o.get("motion")})
    allrows[name] = rows
json.dump(allrows, open(sys.argv[1], "w"), indent=1)
ex = [r for rows in allrows.values() for r in rows if r["type"] == "execute"]
print("types", {t: sum(1 for rows in allrows.values() for r in rows if r["type"] == t) for t in set(r["type"] for rows in allrows.values() for r in rows)})
print("execute n", len(ex), "last obs_after true", sum(r["last_obs_after"] for r in ex), "any mid obs_after", sum(r["mid_obs_after"] for r in ex))
print("mean segs %.2f" % np.mean([r["n_segs"] for r in ex]), "max segs", max(r["n_segs"] for r in ex), "mean req steps %.0f" % np.mean([r["req_steps"] for r in ex]))
allr = [r for rows in allrows.values() for r in rows]
print("mean chars %.0f, prose share %.3f" % (np.mean([r["chars"] for r in allr]), sum(r["prose_chars"] for r in allr) / sum(r["chars"] for r in allr)))
