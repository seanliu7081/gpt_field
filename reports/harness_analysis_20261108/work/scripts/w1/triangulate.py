"""Offline: triangulate SAM mask centroids across the two calibrated views (no DA3), compare with layout truth."""
import json, sys
import numpy as np
from pathlib import Path

def cams_for(run, step):
    # find request with that step
    reqdir = Path(run)/'trial'/'requests'
    for req in sorted(reqdir.glob('*.request.json')):
        d = json.load(open(req))
        last = d['payload']['input'][-1]
        for c in last['content']:
            if c.get('type') == 'input_text' and c['text'].startswith('{'):
                t = json.loads(c['text'])
                if t.get('step') == step:
                    return t['geometry']['cameras']
    return None

def ray(c, u, v):
    R = np.array(c['rotation_world_from_camera']); fx, fy = c['focal_length_px']; cx, cy = c['principal_point_px']
    dcv = np.array([(u-cx)/fx, (v-cy)/fy, 1.0]); w = R @ np.diag([1, -1, -1]) @ dcv
    return np.array(c['position_world_m']), w/np.linalg.norm(w)

def closest(p1, d1, p2, d2):
    # midpoint of closest approach
    w0 = p1 - p2; a = d1@d1; b = d1@d2; c = d2@d2; d = d1@w0; e = d2@w0
    den = a*c - b*b
    s = (b*e - c*d)/den; t = (a*e - b*d)/den
    q1 = p1 + s*d1; q2 = p2 + t*d2
    return (q1+q2)/2, np.linalg.norm(q1-q2)

if __name__ == '__main__':
    run = sys.argv[1]; truth = np.array([float(x) for x in sys.argv[2].split(',')]); key = sys.argv[3]
    fv = Path(run)/'trial'/'foundation_vision'
    for meta in sorted(fv.glob('frame_*_sam_candidates.json')):
        m = json.load(open(meta)); step = m['frame_id']
        rows = [r for r in m['candidates'] if r['target_key'] == key]
        ext = [r for r in rows if r['view'] == 'external']; wr = [r for r in rows if r['view'] == 'wrist']
        if len(ext) != 1 or len(wr) != 1:
            print(step, key, 'ext', len(ext), 'wrist', len(wr), '-> no pair'); continue
        masks = np.load(fv/m['masks']['path'])
        cams = cams_for(run, step)
        out = []
        pts = {}
        for r in (ext[0], wr[0]):
            mk = masks[r['array_key']]; yy, xx = np.nonzero(mk)
            pts[r['view']] = (xx.mean(), yy.mean(), r['score'])
        p1, d1 = ray(cams['external'], *pts['external'][:2]); p2, d2 = ray(cams['wrist'], *pts['wrist'][:2])
        X, gap = closest(p1, d1, p2, d2)
        print(step, key, 'scores ext/wrist', round(pts['external'][2], 3), round(pts['wrist'][2], 3), 'triangulated', X.round(4), 'ray gap m', round(gap, 4), 'err vs truth xy m', round(np.linalg.norm(X[:2]-truth[:2]), 4), 'z err', round(X[2]-truth[2], 4))
