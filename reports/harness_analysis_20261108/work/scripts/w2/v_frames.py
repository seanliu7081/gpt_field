"""Independent recount of per-target-frame vision outcomes from raw snapshots (verify_h3)."""
import json, glob, os, sys, collections
RUNS = {"T04": "/workspace/gpt_field/runs/foundation_vision_t04_can_obstacle_20261105_001",
        "T08": "/workspace/gpt_field/runs/foundation_vision_t08_bottle_obstacle_20261105_001"}
out = {}
for tag, run in RUNS.items():
    snaps = sorted(glob.glob(run + "/trial/foundation_vision/frame_*_snapshot.json"))
    cats = collections.Counter()
    per_target = collections.defaultdict(lambda: collections.Counter())
    texts = {}
    uniq_ext = 0; any_wrist = 0; uniq_both = 0; usable = 0
    timing = []
    frames = []
    for s in snaps:
        d = json.load(open(s))
        frames.append(d["frame_id"])
        timing.append(d["timing_s"])
        for o in d["objects"]:
            k = o["key"]; texts[k] = o["text"]
            cc = o["candidate_counts"]; e, w = cc.get("external", 0), cc.get("wrist", 0)
            assoc = o["geometry"].get("association_status")
            if o["centroid_world_m"] is not None: usable += 1
            if e == 1: uniq_ext += 1
            if w >= 1: any_wrist += 1
            if e == 1 and w == 1: uniq_both += 1
            if e == 0 and w == 0: c = "none"
            elif e > 1 or w > 1: c = "ambiguous"
            elif (e == 1) != (w == 1): c = "one_view"
            elif assoc == "rejected_mixed_instance": c = "assoc_rejected"
            else: c = "other:" + str(assoc)
            cats[c] += 1; per_target[k][c] += 1
    import statistics
    tot = [t["total"] for t in timing]; seg = [t["segmentation"] for t in timing]
    dep = [t["depth"] for t in timing]; geo = [t["geometry"] for t in timing]
    out[tag] = {"frames": frames, "n_frames": len(snaps), "target_frames": sum(cats.values()),
                "categories": dict(cats), "uniq_ext": uniq_ext, "any_wrist": any_wrist,
                "uniq_both": uniq_both, "usable_centroids": usable,
                "per_target": {k: dict(v) for k, v in per_target.items()}, "texts": texts,
                "timing_median_total": statistics.median(tot), "first_total": tot[0],
                "sum_total": sum(tot), "median_seg": statistics.median(seg),
                "median_depth": statistics.median(dep), "median_geo": statistics.median(geo),
                "first_seg": seg[0]}
print(json.dumps(out, indent=1))
