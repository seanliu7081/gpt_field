import sys, json
sys.path.insert(0, "/workspace/gpt_field/src")
exec(open("/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts/v_geom_repro.py").read().split("for se, sw in")[0])
from gpt_field.foundation_vision.geometry import _mask_on_depth, _sample, _mask_at_world, _independent
fv = R + "foundation_vision_t04_can_obstacle_20261105_001/trial/foundation_vision/"
meta = json.load(open(fv + "frame_000000_sam_candidates.json")); arr = np.load(fv + "frame_000000_sam_candidates.npz")
masks = {r["view"]: arr[r["array_key"]] for r in meta["candidates"] if r["target_key"] == "extra_0"}
lm = json.load(open(R + "foundation_vision_t04_can_obstacle_20261105_001/inputs/layout_manifest.json")); b = lm["can_after_hold_bounds"]
tlo, thi = np.array(b["lower_xyz_m"]), np.array(b["upper_xyz_m"]); print("true size", (thi - tlo).round(3).tolist(), "centre", ((tlo + thi) / 2).round(4).tolist())
for se, sw in ((S0, S0), (0.84, 0.72)):
    g = build_geometry(frame, depths(se, sw), {})
    cfg = g.config; views = g._views
    allpts = []
    for name, mask in masks.items():
        view = views[name]; sel = _mask_on_depth(mask, view); P = view.points_world[sel]
        sup = np.ones(len(P), int)
        for on, om in masks.items():
            if on == name: continue
            other = views[on]
            _, _, surf, occ, front, _ = _sample(P, other, cfg)
            sup += surf & _mask_at_world(P, om, other)
        elig = (sup >= 2) & (view.contradictions[sel] == 0)
        Q = P[elig]; allpts.append(Q)
        print(f"  scales({se:.3f},{sw:.3f}) {name}: mask pts {len(P)}, eligible {len(Q)}; raw extent {(P.max(0)-P.min(0)).round(3).tolist()}; eligible extent {(Q.max(0)-Q.min(0)).round(3).tolist() if len(Q) else None}")
    Q = np.concatenate(allpts)
    if len(Q):
        ext = Q.max(0) - Q.min(0); c = Q.mean(0)
        print(f"  => engine-style bounds (if association accepted): extent {ext.round(3).tolist()} ratio {(ext/(thi-tlo)).round(2).tolist()} centroid {c.round(4).tolist()} centroid err xy {np.linalg.norm(c[:2]-((tlo+thi)/2)[:2])*100:.1f} cm z {(c[2]-((tlo+thi)/2)[2])*100:+.1f} cm")
