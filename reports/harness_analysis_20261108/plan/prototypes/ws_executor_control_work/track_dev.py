"""Measured grip-site deviation from the planned transit polyline (route_tracking.distance_to_path_m),
and along-path overshoot past the endpoint, on routed transit steps (100 mm runs)."""
import pickle
import numpy as np
BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
dev, over = [], []
for name in ("T04_t100", "T08_t100", "T08_stoveoff", "drawer001", "drawer003", "H3_T04", "H3_T08"):
    d = pickle.load(open(BASE + name + "_steps.pkl", "rb"))
    for k in sorted(d):
        frag = d[k].get("frag") or {}
        rt = frag.get("route_tracking") or {}
        if frag.get("profile") == "transit" and "distance_to_path_m" in rt:
            dev.append(rt["distance_to_path_m"])
dev = np.asarray(dev)
print(f"routed transit steps n={len(dev)}: lateral deviation median {np.median(dev)*1000:.2f} mm p95 {np.percentile(dev,95)*1000:.2f} p99 {np.percentile(dev,99)*1000:.2f} max {dev.max()*1000:.2f} mm")
