"""Overlay projected current-robot collision hulls on saved RGB (offline)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
src = (HERE / 'cf_v2.py').read_text().split('out = {}')[0]
ns = {'__file__': str(HERE / 'cf_v2.py')}
exec(compile(src, 'cf_v2_head', 'exec'), ns)
restore, step_inputs, m2, kin = ns['restore'], ns['step_inputs'], ns['m2'], ns['kin']
outdir = Path(sys.argv[1]); outdir.mkdir(parents=True, exist_ok=True)
step = int(sys.argv[2]) if len(sys.argv) > 2 else 18
rec, state, q = step_inputs(step)
scene = restore(rec)
bound = kin.bind(state, q)
planner = m2.CandidatePlanner(kin)
shapes = planner.robot.shapes(bound, q, state[6:8])
report = {}
for camera in scene.frame.cameras:
    rgb = np.ascontiguousarray(camera.rgb.copy())
    K, E = camera.intrinsics, camera.world_to_camera
    overlay = rgb.copy()
    colours = {}
    for i, shape in enumerate(shapes):
        cam = shape.surface_points() @ E[:3, :3].T + E[:3, 3]
        front = cam[:, 2] > 1e-3
        if front.sum() < 3:
            continue
        uv = cam[front] @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        colour = tuple(int(c) for c in np.random.default_rng(i).integers(60, 255, 3))
        cv2.polylines(overlay, [hull], True, colour, 1)
        colours[shape.name] = {'colour': colour, 'uv_mean': uv.mean(0).round(1).tolist(),
                               'behind_camera_vertices': int((~front).sum())}
    # grip site and EEF axes
    p = np.asarray(state[:3])
    for axis, col in ((np.array([.05, 0, 0]), (255, 0, 0)), (np.array([0, .05, 0]), (0, 255, 0)), (np.array([0, 0, .05]), (0, 0, 255))):
        a = np.array([p, p + axis]) @ E[:3, :3].T + E[:3, 3]
        if np.all(a[:, 2] > 0):
            uv = a @ K.T; uv = uv[:, :2] / uv[:, 2:3]
            cv2.line(overlay, tuple(np.round(uv[0]).astype(int)), tuple(np.round(uv[1]).astype(int)), col, 2)
    big = cv2.resize(overlay, (overlay.shape[1] * 2, overlay.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(outdir / f'overlay_step{step}_{camera.name}.png'), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    report[camera.name] = colours
(outdir / f'overlay_step{step}.json').write_text(json.dumps(report, indent=1))
print(json.dumps({k: {n: v['uv_mean'] for n, v in d.items()} for k, d in report.items()}, indent=1))
