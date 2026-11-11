"""Open-finger inward deflection as a contact channel: free vs contact steps (diagnostic labels only)."""
import pickle
import numpy as np
BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
free, contact, per_ep = [], [], []
for name in ("T04_t100", "T08_t100", "T08_stoveoff", "drawer001", "drawer003", "T04_20mm", "T04_default", "H3_T04", "H3_T08"):
    d = pickle.load(open(BASE + name + "_steps.pkl", "rb")); ks = sorted(d)
    prev_grip = None
    for i, k in enumerate(ks):
        x = d[k]; b, a = np.asarray(x["before_state"]), np.asarray(x["after_state"]); act = np.asarray(x["norm_action"])
        grip = act[6]
        steady = prev_grip is not None and grip == prev_grip == -1.0 and b[6] - b[7] > 0.075  # open and settled
        prev_grip = grip
        if not steady: continue
        inward = max((b[6] - a[6]) * 1000, (a[7] - b[7]) * 1000)
        c = (x.get("forb_inc", 0) or 0) + sum((x.get("inc_allowed") or {}).values())
        robot_obj = sum(v for kk, v in (x.get("inc") or {}).items() if kk in ("robot_object", "robot_scene"))
        if c == 0: free.append(inward)
        else: contact.append(inward)
free, contact = np.asarray(free), np.asarray(contact)
print(f"open settled steps: free n={len(free)} inward p99={np.percentile(free,99):.4f} mm p99.9={np.percentile(free,99.9):.4f} max={free.max():.4f}; std={free.std():.4f}")
print(f"contact n={len(contact)} inward median={np.median(contact):.4f} p75={np.percentile(contact,75):.4f} p90={np.percentile(contact,90):.4f}; frac>0.10mm={np.mean(contact>0.10):.3f}; free frac>0.10mm={np.mean(free>0.10):.4f}")
