"""Pixel-region provenance: what happens to robot-finger pixels and to the background wall in the external view."""
import json, sys
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
from k2_scene_audit import replica
step = 13
rec = record_for_step(step); scene, _ = restore(rec); state, q = step_state(step)
planner = CandidatePlanner(kin)
prov = replica(planner, scene, state, q)
P = prov['points']; keep = prov['keep']; view = prov['view']; r = prov['row']; c = prov['col']
cam = {x.name: x for x in scene.frame.cameras}
out = {}
def region(v, r0, r1, c0, c1, label):
    m = (view == v) & (r >= r0) & (r < r1) & (c >= c0) & (c < c1)
    k = m & keep
    gaps = prov['front_gap'][k]
    fin = gaps[np.isfinite(gaps)]
    d = {'pixels': int(m.sum()), 'kept': int(k.sum()), 'removed': int((m & ~keep).sum()),
         'kept_strong': int((k & prov['strong']).sum()), 'kept_in_silhouette': int((k & prov['in_sil']).sum()),
         'kept_front_gap_m_p10_p50_p90': np.percentile(fin, [10, 50, 90]).round(3).tolist() if len(fin) else None,
         'kept_xyz_median': np.median(P[k], 0).round(3).tolist() if k.sum() else None,
         'removed_by': {s: int(n) for s, n in zip(*np.unique(prov['removed_by'][m & ~keep], return_counts=True))}}
    out[label] = d
# external view regions (upright raster coordinates, rows=v, cols=u)
region('external', 120, 195, 200, 300, 'external_finger_block_rows120_195_cols200_300')
region('external', 0, 30, 175, 335, 'external_top_robot_rows0_30_cols175_335')
region('external', 0, 120, 0, 170, 'external_background_wall_rows0_120_cols0_170')
region('external', 60, 120, 340, 512, 'external_background_right_rows60_120_cols340_512')
# wrist view: bottom rows are the fingers/hand per H3 F5 (rows >= ~430)
region('wrist', 430, 512, 0, 512, 'wrist_rows430_512')
region('wrist', 0, 430, 0, 512, 'wrist_rows0_430')
# depth values along a column through the robot top edge, for inspection
d_ext = np.load(V / rec['arrays'])['camera_0_depth_m']
out['external_depth_rows0_30_col190'] = d_ext[0:30, 190].round(3).tolist()
out['external_depth_rows0_30_col260'] = d_ext[0:30, 260].round(3).tolist()
out['external_depth_rows0_30_col100_wall'] = d_ext[0:30, 100].round(3).tolist()
out['external_depth_rows120_195_col250_fingers'] = d_ext[120:195:5, 250].round(3).tolist()
print(json.dumps(out, indent=1))
(W2 / 'k2_region_probe.json').write_text(json.dumps(out, indent=1))
# wrist overlay with strong/removed
import cv2
for v in ('wrist',):
    m = view == v; o = cam[v].rgb.copy()
    rem = m & ~keep; o[r[rem], c[rem]] = (0.4*o[r[rem], c[rem]] + [150, 0, 0]).astype(np.uint8)
    st = m & keep & prov['strong']; o[r[st], c[st]] = (0.4*o[r[st], c[st]] + [0, 150, 0]).astype(np.uint8)
    cv2.imwrite(str(W2 / f'{v}_selfremoved_red_strong_green_frame14.png'), cv2.cvtColor(np.concatenate([cam[v].rgb, o], 1), cv2.COLOR_RGB2BGR))
