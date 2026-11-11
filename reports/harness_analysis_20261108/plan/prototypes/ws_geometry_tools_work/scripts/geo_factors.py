"""Offline: camera-geometry factors for the uncertainty model (no sim, no model calls).

For each saved request frame given on the command line (run dir, request ordinal), and a
truth point (the added obstacle's box centre / top), print:
  - external range and elevation of the ray to the point,
  - one-view ray-plane amplification (horizontal error per unit plane-height error),
  - metric size of 1 px and of the plan-cell quantisation sigma (64 px / sqrt(12)) at that range,
  - wrist-external ray angle (triangulation conditioning) for the same point.
"""
import json, math, sys
from pathlib import Path
import numpy as np


def request_fields(run, ordinal):
    reqs = sorted((Path(run) / 'trial' / 'requests').glob('*.request.json'))
    d = json.load(open(reqs[ordinal - 1]))
    for c in d['payload']['input'][-1]['content']:
        if c.get('type') == 'input_text' and c['text'].startswith('{'):
            return json.loads(c['text'])
    raise SystemExit('no fields')


def ray_to(cam, X):
    o = np.array(cam['position_world_m']); d = np.asarray(X) - o
    return o, d / np.linalg.norm(d), float(np.linalg.norm(np.asarray(X) - o))


def main():
    run, ordinal, box = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    lo, hi = [np.array([float(v) for v in s.split(',')]) for s in box.split(':')]
    f = request_fields(run, ordinal)
    cams = f['geometry']['cameras']
    for name, X in (('centre', (lo + hi) / 2), ('top', np.r_[(lo[:2] + hi[:2]) / 2, hi[2]]),
                    ('base', np.r_[(lo[:2] + hi[:2]) / 2, lo[2]])):
        oe, de, re = ray_to(cams['external'], X)
        ow, dw, rw = ray_to(cams['wrist'], X)
        elev = math.degrees(math.asin(-de[2]))
        fx_e = cams['external']['focal_length_px'][0]; fx_w = cams['wrist']['focal_length_px'][0]
        cell_sigma_px = 64 / math.sqrt(12)
        angle = math.degrees(math.acos(abs(float(de @ dw))))
        print(f"{Path(run).name[-40:]} req{ordinal} {name:6s} ext range {re:.3f} m elev {elev:5.1f} deg "
              f"| ray-plane horiz/plane-z amplification {1/math.tan(math.radians(elev)):.2f} "
              f"| ext 1px={re/fx_e*1000:.2f} mm cell-sigma={cell_sigma_px*re/fx_e*100:.2f} cm "
              f"| wrist range {rw:.3f} m 1px={rw/fx_w*1000:.2f} mm | ray angle ext-wrist {angle:5.1f} deg")


if __name__ == '__main__':
    main()
