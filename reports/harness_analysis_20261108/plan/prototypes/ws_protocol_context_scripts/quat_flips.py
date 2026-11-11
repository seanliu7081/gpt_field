"""Quaternion sign behaviour of robot_pose_wxyz across executive requests (offline, read-only)."""
import json, glob, sys, os, math
def load(run):
    out = []
    for f in sorted(glob.glob(os.path.join(run, "trial", "requests", "*.request.json"))):
        p = json.load(open(f))["payload"]
        t = json.loads(p["input"][-1]["content"][0]["text"])
        out.append(t["robot_pose_wxyz"][3:7])
    return out
def dot(a, b): return sum(x*y for x, y in zip(a, b))
for run in sys.argv[1:]:
    qs = load(run)
    raw_flips = sum(1 for a, b in zip(qs, qs[1:]) if dot(a, b) < 0)
    w = [q[0] for q in qs]
    # rule 1: w >= 0
    c1 = [q if q[0] >= 0 else [-v for v in q] for q in qs]
    flips_w = sum(1 for a, b in zip(c1, c1[1:]) if dot(a, b) < 0)
    # rule 2: hemisphere of the first measured orientation
    ref = qs[0]
    c2 = [q if dot(q, ref) >= 0 else [-v for v in q] for q in qs]
    flips_ref = sum(1 for a, b in zip(c2, c2[1:]) if dot(a, b) < 0)
    min_ref_dot = min(abs(dot(q, ref)) for q in qs)
    print(os.path.basename(run)[:70], "n", len(qs), "raw flips", raw_flips, "| w>=0 flips", flips_w,
          "| ref-hemisphere flips", flips_ref, "| |w| min/max %.4f/%.4f" % (min(abs(x) for x in w), max(abs(x) for x in w)),
          "| min |q.qref| %.3f" % min_ref_dot)
