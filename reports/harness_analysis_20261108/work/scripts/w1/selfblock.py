import sys, json
sys.path.insert(0, '/workspace/gpt_field/src')
import numpy as np
from collections import Counter
from gpt_field.foundation_vision.bridge import load_manifest
from gpt_field.foundation_vision.backends import depth_on_original_rays
from gpt_field.foundation_vision.types import DepthEstimate
from gpt_field.foundation_vision.geometry import build_geometry, _project
R = '/workspace/gpt_field/runs/'
m = load_manifest(R + 'foundation_vision_preparation_001/input/manifest.json')
frame = m.frames[0]; cams = {c.name: c for c in frame.cameras}; names = [c.name for c in frame.cameras]
z = np.load(R + 'foundation_vision_preparation_001/da3_real_003/cold_native_prediction.npz')
raw = z['raw_depth_model_units']; K = z['native_intrinsics']; conf = z['raw_score_map']
S0 = 0.8780772385418903
D = {}
for i, n in enumerate(names):
    d, sc = depth_on_original_rays(raw[i]*S0, K[i], cams[n], conf[i])
    D[n] = DepthEstimate(n, d, cams[n].intrinsics, cams[n].world_to_camera, sc, source='da3')
g = build_geometry(frame, D, {})
hand = frame.hand_world; grip = hand[:3, 3] + hand[:3, :3] @ np.array([0, 0, 0.097])
path = np.array([grip, grip + [0, 0, -0.10]])
centers = np.linspace(path[0], path[1], 11)
dirs = np.array([[x, y, zz] for x in (-1, 0, 1) for y in (-1, 0, 1) for zz in (-1, 0, 1) if x or y or zz], float)
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
probes = (centers[:, None, :] + np.vstack((np.zeros((1, 3)), 0.02*dirs))[None]).reshape(-1, 3)
res = g.query(probes, frame.captured_at_s, 5.0)
surf = [i for i, r in enumerate(res) if r['state'] == 'surface']
print('probes', len(probes), 'surface', len(surf), Counter(r['state'] for r in res))
byview = Counter(v for i in surf for v in res[i]['surface_views'])
print('surface views', byview)
for v in ('wrist', 'external'):
    idx = [i for i in surf if v in res[i]['surface_views']]
    if not idx: continue
    uv, zc = _project(probes[idx], cams[v].intrinsics, cams[v].world_to_camera)
    dep = D[v].depth_m[np.clip(np.round(uv[:, 1]).astype(int), 0, 511), np.clip(np.round(uv[:, 0]).astype(int), 0, 511)]
    print(v, 'surface-probe pixel v range %.0f-%.0f u range %.0f-%.0f; probe depth %.3f-%.3f; observed depth there %.3f-%.3f' % (uv[:, 1].min(), uv[:, 1].max(), uv[:, 0].min(), uv[:, 0].max(), zc.min(), zc.max(), np.nanmin(dep), np.nanmax(dep)))
# wrist image: what is at the bottom rows (fingers)? depth of the bottom-centre region
dw = D['wrist'].depth_m
print('wrist depth at bottom-centre rows 430-511, cols 150-360: median %.3f; centre region rows 200-300: median %.3f' % (np.nanmedian(dw[430:, 150:360]), np.nanmedian(dw[200:300, 200:300])))
