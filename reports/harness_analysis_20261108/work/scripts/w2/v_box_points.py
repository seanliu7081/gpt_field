import json, glob, numpy as np
R = "/workspace/gpt_field/runs/"
for run in ("foundation_vision_t04_can_obstacle_20261105_001", "foundation_vision_t08_bottle_obstacle_20261105_001"):
    pls = [json.loads(l) for l in open(R + run + "/trial/foundation_vision/payloads.jsonl")]
    resps = sorted(glob.glob(R + run + "/trial/requests/*.response.json"))
    for i, rq in enumerate(resps):
        t = [c["text"] for o in json.load(open(rq))["output"] if o.get("type") == "message" for c in o["content"] if c.get("type") == "output_text"][-1]
        om = json.loads(t).get("obstacle_map")
        if not om or not om.get("avoid_boxes"): continue
        sc = pls[i]["summary"]["scene"]; P = np.array(sc["surface_points_world"]); sup = np.array(sc["sample_crossview_support_counts"])
        counts = []
        for b in om["avoid_boxes"]:
            lo, hi = np.array(b["lower_xyz_m"]), np.array(b["upper_xyz_m"])
            inside = np.all((P >= lo) & (P <= hi), axis=1)
            counts.append((int(inside.sum()), int((inside & (sup >= 2)).sum())))
        print(run[:30], "call", i + 1, "points in each box (all, supported):", counts)
