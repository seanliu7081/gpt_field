import sys, json
sys.path.insert(0, "/workspace/gpt_field/src")
exec(open("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts/v_geom_repro.py").read().split("g = build_geometry(frame, depths(S0, S0), {})")[0].split("for se, sw in")[0])
from collections import Counter
lm = json.load(open(R + "foundation_vision_t04_can_obstacle_20261105_001/inputs/layout_manifest.json"))
poses = {k: np.array(v["body_xyz_m"]) for k, v in lm["custom_after_passive_hold_object_poses"].items()}
rng = np.random.default_rng(1)
P = np.column_stack([rng.uniform(-0.30, 0.15, 20000), rng.uniform(-0.40, 0.40, 20000), rng.uniform(0.0, 1.0, 20000)])
far = np.ones(len(P), bool)
for k, X in poses.items(): far &= np.linalg.norm(P[:, :2] - X[:2], axis=1) > (0.15 if "plate" in k else 0.10)
for zlo, zhi in ((0.457, 0.477), (0.56, 0.64)):
    Q = P[far].copy(); Q[:, 2] = zlo + (zhi - zlo) * Q[:, 2]; Q = Q[:3000]
    for se, sw in ((S0, S0), (0.84, 0.72)):
        gg = build_geometry(frame, depths(se, sw), {})
        c = Counter(x["state"] for x in gg.query(Q, frame.captured_at_s, 1e9))
        print(f"free air z {zlo:.3f}-{zhi:.3f} scales ({se:.3f},{sw:.3f}): ", {k: round(v / len(Q), 3) for k, v in c.items()})
