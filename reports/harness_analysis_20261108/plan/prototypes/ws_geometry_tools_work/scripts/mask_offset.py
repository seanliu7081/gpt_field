"""Offline: pixel offset between a saved SAM mask's centre column and the truth centre's projection."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from lift_eval import cams_by_step

def project(cam, X):
    R = np.asarray(cam['rotation_world_from_camera'], float)
    l = R.T @ (np.asarray(X) - np.asarray(cam['position_world_m'], float))
    fx, fy = cam['focal_length_px']; cx, cy = cam['principal_point_px']
    return cx + fx * l[0] / -l[2], cy - fy * l[1] / -l[2]

run, key, box = sys.argv[1], sys.argv[2], sys.argv[3]
lo, hi = [np.array([float(v) for v in s.split(',')]) for s in box.split(':')]
cams = cams_by_step(run)
fv = Path(run) / 'trial' / 'foundation_vision'
meta = sorted(fv.glob('frame_*_sam_candidates.json'))[0]
m = json.load(open(meta)); step = m['frame_id']; masks = np.load(fv / m['masks']['path'])
r = [r for r in m['candidates'] if r['target_key'] == key and r['view'] == 'external'][0]
ys, xs = np.nonzero(masks[r['array_key']])
cam = cams[step]['external']
corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
uv = np.array([project(cam, c) for c in corners])
uc, vc = project(cam, (lo + hi) / 2)
print(f"{Path(run).name} step {step}: mask u [{xs.min()},{xs.max()}] median {np.median(xs):.1f} | truth-box u [{uv[:,0].min():.1f},{uv[:,0].max():.1f}] centre {uc:.1f}"
      f" | mask v [{ys.min()},{ys.max()}] | truth v [{uv[:,1].min():.1f},{uv[:,1].max():.1f}] | mask px {len(xs)} score {r['score']:.2f}")
