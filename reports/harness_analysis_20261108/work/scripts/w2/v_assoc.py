import json, glob
RUNS = {"T04": "/workspace/gpt_field/runs/foundation_vision_t04_can_obstacle_20261105_001",
        "T08": "/workspace/gpt_field/runs/foundation_vision_t08_bottle_obstacle_20261105_001"}
for tag, run in RUNS.items():
    for s in sorted(glob.glob(run + "/trial/foundation_vision/frame_*_snapshot.json")):
        d = json.load(open(s))
        g = d["geometry"]
        ext = g["views"]["external"]; wr = g["views"]["wrist"]
        print(tag, d["frame_id"], "ext sup/valid %.1f%% contra %.1f%% | wrist sup %.1f%% contra %.1f%%" % (
            100*ext["crossview_supported_points"]/ext["valid_depth_pixels"], 100*ext["contradicted_points"]/ext["valid_depth_pixels"],
            100*wr["crossview_supported_points"]/wr["valid_depth_pixels"], 100*wr["contradicted_points"]/wr["valid_depth_pixels"]),
            "age %.2f" % d["age_s"], "cap %.2f" % d["captured_at_s"])
        for o in d["objects"]:
            ev = o["geometry"].get("crossview_mask_evidence") or []
            if ev:
                print("   ", o["key"], o["text"][:40], [(e["source_view"][0], e["comparable_points"], e["mask_matches"],
                       None if e["mask_agreement_fraction"] is None else round(e["mask_agreement_fraction"], 3),
                       e["contradicted_points"], e["occluded_points"]) for e in ev], o["geometry"]["surface_hypothesis_count"], o["geometry"]["supported_point_count"])
