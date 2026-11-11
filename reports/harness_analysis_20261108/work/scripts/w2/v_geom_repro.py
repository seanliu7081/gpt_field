import sys, json
sys.path.insert(0, "/workspace/gpt_field/src")
import numpy as np
from gpt_field.foundation_vision.bridge import load_manifest
from gpt_field.foundation_vision.backends import depth_on_original_rays
from gpt_field.foundation_vision.types import DepthEstimate
from gpt_field.foundation_vision.geometry import build_geometry, GeometryConfig, _project
R = "/workspace/gpt_field/runs/"
m = load_manifest(R + "foundation_vision_preparation_001/input/manifest.json"); frame = m.frames[0]
z = np.load(R + "foundation_vision_preparation_001/da3_real_003/cold_native_prediction.npz")
cams = {c.name: c for c in frame.cameras}; names = [c.name for c in frame.cameras]
raw = z["raw_depth_model_units"]; K = z["native_intrinsics"]; conf = z["raw_score_map"]; S0 = 0.8780772385418903
def depths(se, sw):
    out = {}
    for i, n in enumerate(names):
        d, sc = depth_on_original_rays(raw[i] * (se if n == "external" else sw), K[i], cams[n], conf[i])
        out[n] = DepthEstimate(n, d, cams[n].intrinsics, cams[n].world_to_camera, sc, source="da3")
    return out
for se, sw in ((S0, S0), (0.84, 0.72), (0.848, 0.744)):
    g = build_geometry(frame, depths(se, sw), {})
    v = g.to_dict()["views"]
    print(f"scales ext {se:.3f} wrist {sw:.3f}:", {n: (v[n]["crossview_supported_points"], v[n]["contradicted_points"], round(100 * v[n]["crossview_supported_points"] / v[n]["valid_depth_pixels"], 1)) for n in v})
g = build_geometry(frame, depths(S0, S0), {})
# F5: 10 cm straight descent from the grip site; who blocks?
state = np.array(json.load(open(R + "foundation_vision_preparation_001/input/observation.json"))["state"]) if "state" in json.load(open(R + "foundation_vision_preparation_001/input/observation.json")) else None
grip = np.array(frame.hand_world[:3, 3]) + frame.hand_world[:3, :3] @ np.array([0, 0, 0.097])
print("grip site", grip.round(4))
path = [grip.tolist(), (grip - [0, 0, 0.10]).tolist()]
res = g.assess_path(path, 0.01, frame.captured_at_s, 1e9)
print("assess_path 10 cm descent r=1cm:", {k: res[k] for k in ("status", "reason") if k in res}, res.get("obstacle_view"), np.round(res.get("obstacle_point_world", [np.nan]*3), 4).tolist(), res.get("surface_distance_m"))
# where is that obstacle point in the wrist image?
wr = cams["wrist"]
if "obstacle_point_world" in res:
    uv, zz = _project(np.array([res["obstacle_point_world"]]), wr.intrinsics, wr.world_to_camera)
    print("   obstacle point projects in wrist at", uv.round(1).tolist(), "depth", zz.round(3).tolist())
# count retained hypotheses near the grip site
allp = np.concatenate([vv.points_world for vv in g._views.values()])
dd = np.linalg.norm(allp - grip, axis=1)
print("retained hypotheses within 5 cm:", int((dd < 0.05).sum()), "within 10 cm:", int((dd < 0.10).sum()))
# probes along the descent classified by query
probes = np.linspace(grip, grip - [0, 0, 0.10], 11)
q = g.query(probes, frame.captured_at_s, 1e9)
print("query states along descent:", [x["state"] for x in q], [x["surface_views"] for x in q])
print("---- radius sweep for the 10 cm descent")
for rad in (0.0, 0.01, 0.015, 0.02, 0.03, 0.04):
    res = g.assess_path(path, rad, frame.captured_at_s, 1e9)
    extra = ""
    if res.get("obstacle_point_world"):
        uv, zz = _project(np.array([res["obstacle_point_world"]]), cams[res["obstacle_view"]].intrinsics, cams[res["obstacle_view"]].world_to_camera)
        extra = f" first blocking point view={res['obstacle_view']} px={uv.round(0).tolist()} depth={zz.round(3).tolist()} dist={res['surface_distance_m']:.4f}"
    print(f"  radius {rad:.3f}: {res['status']} ({res['reason']}){extra}")
# which retained points lie within 2.5 cm of the descent axis, by view and wrist row
for n, vv in g._views.items():
    P = vv.points_world
    a = grip; b = grip - np.array([0, 0, 0.10]); dlt = b - a
    fr = np.clip((P - a) @ dlt / (dlt @ dlt), 0, 1); dist = np.linalg.norm(P - (a + fr[:, None] * dlt), axis=1)
    sel = dist <= 0.025
    if sel.any():
        rows = vv.rows[sel]
        print(f"  {n}: {sel.sum()} retained points within 2.5 cm of descent axis; depth-raster rows {rows.min()}-{rows.max()}")
