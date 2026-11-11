"""Offline prototype of the proposed box lifting with uncertainty (no sim, no model, no GPU).

Inputs: H3's saved SAM masks (trial/foundation_vision/frame_*_sam_candidates.{json,npz}) and the
synchronized calibration in the saved executive requests. Truth: layout collision bounds of the
added obstacle (diagnostic only).

Methods (per frame, per obstacle mask):
  one_view  : external mask only + support plane z_s (+/- sigma_s):
              bottom-centre pixel ray -> plane gives the FRONT-bottom point; the box depth along the
              horizontal ray direction is unknown, so it is set equal to the measured width with a
              sigma of half the width; top = top-centre ray at the box centre's horizontal range.
  two_view  : external + wrist mask centroids -> midpoint triangulation; gap reported.
Prints centre / top error against truth and whether truth lies inside the reported 2-sigma box.

usage: lift_eval.py RUN_DIR TARGET_KEY 'lo_x,lo_y,lo_z:hi_x,hi_y,hi_z' SUPPORT_Z SUPPORT_SIGMA [PLANE_BIAS]
"""
import json, math, sys
from pathlib import Path
import numpy as np

MASK_SIGMA_PX = 1.5   # assumed silhouette-boundary sigma; to be measured in the benchmark


def cams_by_step(run):
    out = {}
    for req in sorted((Path(run) / 'trial' / 'requests').glob('*.request.json')):
        d = json.load(open(req))
        for c in d['payload']['input'][-1]['content']:
            if c.get('type') == 'input_text' and c['text'].startswith('{'):
                t = json.loads(c['text'])
                if 'geometry' in t and 'step' in t:
                    out[t['step']] = t['geometry']['cameras']
    return out


def ray(cam, u, v):
    R = np.asarray(cam['rotation_world_from_camera'], float)
    fx, fy = cam['focal_length_px']; cx, cy = cam['principal_point_px']
    d = R @ np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
    return np.asarray(cam['position_world_m'], float), d / np.linalg.norm(d)


def plane_hit(o, d, z):
    s = (z - o[2]) / d[2]
    return o + s * d, s


def one_view(cam, mask, zs, sig_s):
    ys, xs = np.nonzero(mask)
    u_mid = float(np.median(xs))
    v_bot = float(ys.max()) + 0.5; v_top = float(ys.min()) - 0.5
    o, d_bot = ray(cam, u_mid, v_bot)
    front, s_bot = plane_hit(o, d_bot, zs)                  # front-bottom contact point on the support
    horiz = np.array([d_bot[0], d_bot[1], 0.0]); horiz /= np.linalg.norm(horiz)
    # width across the ray at the front point's range, from the mask's horizontal extent at its bottom rows
    rows = ys >= np.percentile(ys, 70)
    ul, ur = float(xs[rows].min()) - 0.5, float(xs[rows].max()) + 0.5
    pl, _ = plane_hit(*ray(cam, ul, v_bot), zs); pr, _ = plane_hit(*ray(cam, ur, v_bot), zs)
    width = float(np.linalg.norm(pr - pl))
    centre_xy = front[:2] + horiz[:2] * width / 2            # depth assumed = width (round/boxy objects)
    # top: intersect the top-centre ray with the vertical plane through the centre, normal = horiz
    o_t, d_t = ray(cam, u_mid, v_top)
    n = horiz; p0 = np.r_[centre_xy + horiz[:2] * width / 2, zs]   # far edge of the top face
    st = float((p0 - o_t) @ n / (d_t @ n))
    top = float((o_t + st * d_t)[2])
    # sigma model: pixel sigma at range, plane sigma amplified along the ray, unknown depth = width/2
    rng = float(s_bot); fx = cam['focal_length_px'][0]
    px_m = MASK_SIGMA_PX * rng / fx
    elev = math.asin(-d_bot[2])
    along = math.hypot(sig_s / math.tan(elev), width / 2 / 2)   # depth half-width ~ width/4 as 1-sigma
    sig = np.array([along, px_m + 0.0, math.hypot(sig_s, px_m)])
    centre = np.r_[centre_xy, (zs + top) / 2]
    half = np.array([width / 2, width / 2, (top - zs) / 2])
    return centre, half, top, sig


def main():
    run, key, box, zs, sig_s = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
    bias = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    lo, hi = [np.array([float(v) for v in s.split(',')]) for s in box.split(':')]
    truth_c = (lo + hi) / 2; truth_top = hi[2]
    cams = cams_by_step(run)
    fv = Path(run) / 'trial' / 'foundation_vision'
    rows = []
    for meta in sorted(fv.glob('frame_*_sam_candidates.json')):
        m = json.load(open(meta)); step = m['frame_id']
        if step not in cams:
            continue
        masks = np.load(fv / m['masks']['path'])
        ext = [r for r in m['candidates'] if r['target_key'] == key and r['view'] == 'external']
        wr = [r for r in m['candidates'] if r['target_key'] == key and r['view'] == 'wrist']
        if len(ext) != 1:
            continue
        mk = masks[ext[0]['array_key']]
        c, half, top, sig = one_view(cams[step]['external'], mk, zs + bias, sig_s)
        err = c - truth_c
        inside = bool(np.all(np.abs(err[:2]) <= 2 * np.maximum(sig[:2], 1e-6)))
        line = (f"step {step:4d} one_view centre err xy {100*np.linalg.norm(err[:2]):.1f} cm "
                f"(dx {100*err[0]:+.1f} dy {100*err[1]:+.1f}) top err {100*(top-truth_top):+.1f} cm "
                f"sigma_xy ({100*sig[0]:.1f},{100*sig[1]:.1f}) cm  truth_in_2sigma_xy={inside}")
        if len(wr) == 1:
            ys, xs = np.nonzero(mk); pe = ray(cams[step]['external'], xs.mean(), ys.mean())
            wk = masks[wr[0]['array_key']]; ys2, xs2 = np.nonzero(wk); pw = ray(cams[step]['wrist'], xs2.mean(), ys2.mean())
            (p1, d1), (p2, d2) = pe, pw
            w0 = p1 - p2; a, b, cc, dd, e = d1 @ d1, d1 @ d2, d2 @ d2, d1 @ w0, d2 @ w0
            den = a * cc - b * b; s = (b * e - cc * dd) / den; t = (a * e - b * dd) / den
            X = (p1 + s * d1 + p2 + t * d2) / 2; gap = np.linalg.norm(p1 + s * d1 - p2 - t * d2)
            line += f" | two_view err xy {100*np.linalg.norm(X[:2]-truth_c[:2]):.1f} cm gap {100*gap:.1f} cm"
        rows.append((np.linalg.norm(err[:2]), inside))
        print(line)
    if rows:
        e = np.array([r[0] for r in rows])
        print(f"SUMMARY {Path(run).name} key={key} bias={bias:+.3f} n={len(rows)} median xy err {100*np.median(e):.1f} cm "
              f"max {100*e.max():.1f} cm truth_in_2sigma {sum(r[1] for r in rows)}/{len(rows)}")


if __name__ == '__main__':
    main()
