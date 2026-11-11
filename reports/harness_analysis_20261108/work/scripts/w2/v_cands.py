"""Raw SAM candidate audit: per-prompt per-view counts, scores, cross-prompt IoU duplicates."""
import json, glob, collections, itertools
import numpy as np
RUNS = {"T04": "/workspace/gpt_field/runs/foundation_vision_t04_can_obstacle_20261105_001",
        "T08": "/workspace/gpt_field/runs/foundation_vision_t08_bottle_obstacle_20261105_001"}
for tag, run in RUNS.items():
    print("=====", tag)
    files = sorted(glob.glob(run + "/trial/foundation_vision/frame_*_sam_candidates.json"))
    raw_counts = collections.defaultdict(collections.Counter)
    scores = collections.defaultdict(list)
    dup = collections.Counter()
    dup_frames = collections.defaultdict(list)
    for f in files:
        meta = json.load(open(f))
        fid = meta["frame_id"]
        arr = np.load(f.replace(".json", ".npz"))
        rows = meta["candidates"]
        for r in rows:
            raw_counts[(r["target_key"], r["view"])][fid] += 1
            scores[(r["target_key"], r["view"])].append(round(r["score"], 3))
        for a, b in itertools.combinations(rows, 2):
            if a["view"] != b["view"] or a["target_key"] == b["target_key"]: continue
            ma, mb = arr[a["array_key"]], arr[b["array_key"]]
            iou = (ma & mb).sum() / max(1, (ma | mb).sum())
            if iou > 0.8:
                key = (tuple(sorted((a["target_key"], b["target_key"]))), a["view"])
                dup[key] += 1; dup_frames[key].append((fid, round(float(iou), 3)))
    keys = sorted({k for k in raw_counts})
    for k in keys:
        frames_with = sorted(raw_counts[k].keys())
        print(k, "frames_with_any:", len(frames_with), "counts:", dict(sorted(raw_counts[k].items())), "scores range:", min(scores[k]), max(scores[k]))
    print("duplicates IoU>0.8 (pair, view): count, frames")
    for k, v in dup.items():
        print(" ", k, v, dup_frames[k])
