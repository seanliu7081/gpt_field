import sys, json
sys.path.insert(0, '/workspace/gpt_field/src')
import numpy as np
from gpt_field.foundation_vision.bridge import load_manifest
from gpt_field.foundation_vision.backends import depth_on_original_rays
from gpt_field.foundation_vision.types import DepthEstimate
from gpt_field.foundation_vision.geometry import build_geometry, _mask_on_depth
from scipy import ndimage
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
mk = np.load(R + 'foundation_vision_sam_validation_001/sam_inference_002/all_candidate_masks.npz')
masks = {'external': mk['mask_000'], 'wrist': mk['mask_001']}
g = build_geometry(frame, D, {})
print('true can size xyz (m): [0.062, 0.062, 0.076]')
for n in names:
    v = g._views[n]
    for tag, msk in (('raw mask', masks[n]), ('mask eroded 2px', ndimage.binary_erosion(masks[n], iterations=2))):
        sel = _mask_on_depth(msk, v); P = v.points_world[sel]
        ext = P.max(0) - P.min(0); rob = np.percentile(P, 98, axis=0) - np.percentile(P, 2, axis=0)
        print(n, tag, 'points', len(P), 'min/max extent', ext.round(3).tolist(), 'p2-p98 extent', rob.round(3).tolist())
# depth-edge (flying pixel) prevalence inside masks: pixels whose depth differs from 3x3 median by >3 cm
for n in names:
    dep = D[n].depth_m; med = ndimage.median_filter(np.nan_to_num(dep, nan=0), size=5)
    fly = np.abs(dep - med) > 0.03
    print(n, 'mask pixels with |depth - 5x5 median| > 3 cm: %.3f; whole image: %.3f' % (fly[masks[n]].mean(), np.nanmean(fly)))
