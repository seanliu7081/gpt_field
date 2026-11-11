"""Trace the 'uncertain' link5/link6 intersection points of the prep check back to source pixels."""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
from k2_scene_audit import replica
step = 13
rec = record_for_step(step); scene, _ = restore(rec); state, q = step_state(step)
planner = CandidatePlanner(kin)
prov = replica(planner, scene, state, q)
P, keep = prov['points'], prov['keep']
tree = cKDTree(P)
probe = {'prep_uncertain_link5': [-0.1958522958433731, -0.07793056549096058, 0.813619557361292],
         'prep_uncertain_link6': [-0.20798223585454417, -0.08472090360947211, 0.8219281108280816],
         'debug_v2_goal_link5': [-0.19807643480266826, -0.07931306335804075, 0.8143761101081948],
         'debug_v2_direct_link5': [-0.19261332651999774, -0.07220139988081462, 0.8245332186357051],
         'initial_debug_link6': [-0.11267635940894379, -0.007094041496755563, 0.839106912161268],
         'previous_debug_link5': [-0.1195094431674877, -0.021499016638038944, 0.8369246950951655]}
out = {}
# current robot link positions (FK) for context
shapes = planner.robot.shapes(planner.bound, q, state[6:8])
out['current_robot_shape_centers'] = {s.name: np.round(s.center, 3).tolist() for s in shapes}
for name, p in probe.items():
    d, i = tree.query(p)
    rec_ = {'nearest_dist_m': float(d), 'view': prov['view'][i], 'row': int(prov['row'][i]), 'col': int(prov['col'][i]),
            'support': int(prov['support'][i]), 'contradictions': int(prov['contra'][i]), 'strong': bool(prov['strong'][i]),
            'kept': bool(keep[i]), 'owner': prov['owners'][i], 'in_robot_silhouette': bool(prov['in_sil'][i]),
            'front_gap_m': None if not np.isfinite(prov['front_gap'][i]) else float(prov['front_gap'][i])}
    cam = next(c for c in scene.frame.cameras if c.name == prov['view'][i])
    rec_['rgb'] = cam.rgb[int(prov['row'][i]), int(prov['col'][i])].tolist()
    # all kept points within 3 cm: their views/rows
    near = tree.query_ball_point(p, 0.03)
    near = [j for j in near if keep[j]]
    rec_['kept_within_3cm'] = len(near)
    if near:
        vs, cnt = np.unique(prov['view'][near], return_counts=True)
        rec_['kept_within_3cm_views'] = dict(zip(vs.tolist(), cnt.tolist()))
        rr = prov['row'][near]; cc = prov['col'][near]
        rec_['kept_within_3cm_pixel_box'] = [int(cc.min()), int(rr.min()), int(cc.max()), int(rr.max())]
        rec_['kept_within_3cm_strong'] = int(prov['strong'][near].sum())
    out[name] = rec_
# where are all kept external points above z 0.6 over the table, in image space?
ext = (prov['view'] == 'external') & keep & (P[:, 2] > .6) & (P[:, 0] > -.45) & (P[:, 0] < .19) & (P[:, 1] > -.6) & (P[:, 1] < .7)
rr, cc = prov['row'][ext], prov['col'][ext]
out['external_kept_above_z060'] = {'n': int(ext.sum()), 'row_p5_p50_p95': np.percentile(rr, [5, 50, 95]).tolist(),
                                   'col_p5_p50_p95': np.percentile(cc, [5, 50, 95]).tolist(),
                                   'xyz_p5': np.percentile(P[ext], 5, axis=0).round(3).tolist(),
                                   'xyz_p95': np.percentile(P[ext], 95, axis=0).round(3).tolist(),
                                   'in_robot_silhouette': int(prov['in_sil'][ext].sum())}
# image of these pixels for inspection
cam = next(c for c in scene.frame.cameras if c.name == 'external')
img = cam.rgb.copy()
over = img.copy(); over[rr, cc] = [255, 0, 255]
wristcam = next(c for c in scene.frame.cameras if c.name == 'wrist')
import cv2
cv2.imwrite(str(W2 / 'external_kept_above_z060_frame14.png'), cv2.cvtColor(np.concatenate([img, over], 1), cv2.COLOR_RGB2BGR))
# self-removed pixels overlay (both views) for inspection
for v, c in (('external', cam), ('wrist', wristcam)):
    m = (prov['view'] == v)
    o = c.rgb.copy()
    rem = m & ~keep
    o[prov['row'][rem], prov['col'][rem]] = (0.4*o[prov['row'][rem], prov['col'][rem]] + [150, 0, 0]).astype(np.uint8)
    st = m & keep & prov['strong']
    o[prov['row'][st], prov['col'][st]] = (0.4*o[prov['row'][st], prov['col'][st]] + [0, 150, 0]).astype(np.uint8)
    cv2.imwrite(str(W2 / f'{v}_selfremoved_red_strong_green_frame14.png'), cv2.cvtColor(np.concatenate([c.rgb, o], 1), cv2.COLOR_RGB2BGR))
print(json.dumps(out, indent=1))
(W2 / 'k2_phantom_trace.json').write_text(json.dumps(out, indent=1))
