"""Offline: single external-view obstacle localisation = SAM mask + support-plane back-projection.
Uses the TRUE table height as the plane (upper bound on accuracy; in deployment the plane must be estimated)."""
import json, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/scripts')
from triangulate import cams_for, ray

def plane_hit(p, d, z):
    s = (z - p[2]) / d[2]; return p + s * d

run, key, plane_z = sys.argv[1], sys.argv[2], float(sys.argv[3])
tlo = np.array([float(x) for x in sys.argv[4].split(',')]); thi = np.array([float(x) for x in sys.argv[5].split(',')])
fv = Path(run)/'trial'/'foundation_vision'
for meta in sorted(fv.glob('frame_*_sam_candidates.json'))[:3]:
    m = json.load(open(meta)); step = m['frame_id']
    ext = [r for r in m['candidates'] if r['target_key'] == key and r['view'] == 'external']
    if len(ext) != 1: continue
    mk = np.load(fv/m['masks']['path'])[ext[0]['array_key']]
    cams = cams_for(run, step); cam = cams['external']
    ys, xs = np.nonzero(mk)
    # bottom contour: for each column, the lowest mask pixel -> ground contact points (front edge)
    pts = []
    for u in np.unique(xs):
        v = ys[xs == u].max()
        p, d = ray(cam, u, v + 0.5)
        pts.append(plane_hit(p, d, plane_z))
    pts = np.array(pts)
    front = pts.mean(axis=0)
    width_y = pts[:, 1].max() - pts[:, 1].min()
    # assume roughly round footprint: centre is half a width further along the horizontal view direction
    p0, dc = ray(cam, xs.mean(), ys.max())
    hd = dc[:2] / np.linalg.norm(dc[:2])
    centre_xy = front[:2] + hd * width_y / 2
    # top height: top mask pixel ray at the centre's horizontal distance
    vt = ys.min(); ut = xs[ys == vt].mean()
    pt, dt = ray(cam, ut, vt)
    s = np.dot(centre_xy - pt[:2], dt[:2]) / np.dot(dt[:2], dt[:2])
    top_z = (pt + s * dt)[2]
    tc = (tlo + thi) / 2
    print(f"step {step} {key}: est centre xy={np.round(centre_xy,4).tolist()} width={width_y:.3f} top_z={top_z:.3f} | truth centre xy={np.round(tc[:2],4).tolist()} width_y={thi[1]-tlo[1]:.3f} top_z={thi[2]:.3f} | xy err={np.linalg.norm(centre_xy-tc[:2]):.3f} m, top err={top_z-thi[2]:+.3f} m")
