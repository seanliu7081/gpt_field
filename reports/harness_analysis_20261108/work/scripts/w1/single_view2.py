"""Offline single-view estimate v2: bottom-centre ray ∩ support plane, silhouette width at that range, top-edge height.
Plane height is the TRUE table height (upper bound); deployment must estimate it (e.g. DA3 external plane fit)."""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/scripts')
from triangulate import cams_for, ray

run, key, plane_z = sys.argv[1], sys.argv[2], float(sys.argv[3])
tlo = np.array([float(x) for x in sys.argv[4].split(',')]); thi = np.array([float(x) for x in sys.argv[5].split(',')])
fv = Path(run)/'trial'/'foundation_vision'
for meta in sorted(fv.glob('frame_*_sam_candidates.json'))[:2]:
    m = json.load(open(meta)); step = m['frame_id']
    ext = [r for r in m['candidates'] if r['target_key'] == key and r['view'] == 'external']
    if len(ext) != 1: continue
    mk = np.load(fv/m['masks']['path'])[ext[0]['array_key']]
    cam = cams_for(run, step)['external']; fx = cam['focal_length_px'][0]
    ys, xs = np.nonzero(mk)
    ub = xs[ys == ys.max()].mean(); vb = ys.max() + 0.5
    p, d = ray(cam, ub, vb); s = (plane_z - p[2]) / d[2]; front = p + s * d
    rng = np.linalg.norm(front - p)
    vm = int((ys.min() + ys.max()) / 2); row = xs[ys == vm]
    width = (row.max() - row.min() + 1) * rng / fx
    hd = d[:2] / np.linalg.norm(d[:2]); centre = front[:2] + hd * width / 2
    vt = ys.min(); ut = xs[ys == vt].mean(); pt, dt = ray(cam, ut, vt)
    k = np.dot(centre - pt[:2], dt[:2]) / np.dot(dt[:2], dt[:2]); top = (pt + k * dt)[2]
    tc = (tlo + thi) / 2
    print(f"step {step} {key}: centre={np.round(centre,4).tolist()} width={width:.3f} top={top:.3f} | truth centre={np.round(tc[:2],4).tolist()} width={thi[1]-tlo[1]:.3f} top={thi[2]:.3f} | xy err={np.linalg.norm(centre-tc[:2])*100:.1f} cm top err={(top-thi[2])*100:+.1f} cm")
