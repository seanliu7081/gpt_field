import json, glob, re
R = "/workspace/gpt_field/runs/"
pat = re.compile(r"vision|foundation|\bSAM\b|DA3|segment(ation|ed)?\b|mask|candidate|bbox|bounding box|supported point|centroid|scene point|depth", re.I)
for r in ["foundation_vision_t04_can_obstacle_20261105_001", "foundation_vision_t08_bottle_obstacle_20261105_001",
          "agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001", "agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001"]:
    print("=====", r)
    for i, rq in enumerate(sorted(glob.glob(R + r + "/trial/requests/*.response.json")), 1):
        resp = json.load(open(rq))
        texts = [c["text"] for o in resp.get("output", []) if o.get("type") == "message" for c in o.get("content", []) if c.get("type") == "output_text"]
        if not texts: print(i, "no text"); continue
        out = json.loads(texts[-1])
        blob = " ".join(str(out.get(k, "")) for k in ("belief", "plan", "why"))
        hits = sorted({m.group(0).lower() for m in pat.finditer(blob)})
        om = out.get("obstacle_map")
        nb = len(om.get("avoid_boxes", [])) if isinstance(om, dict) else None
        if hits or nb:
            print(i, "hits", hits, "| obstacle boxes:", nb)
            for m in pat.finditer(blob):
                s = max(0, m.start() - 90); print("    ...", blob[s:m.end() + 90].replace("\n", " "))
                break
