"""Evaluate query()/assess_path() states vs evaluation-only truth on the saved T04 frame-0 geometry."""
import sys, json, time
sys.path.insert(0, '/workspace/gpt_field/src')
import numpy as np
from collections import Counter
from gpt_field.foundation_vision.bridge import load_manifest
from gpt_field.foundation_vision.backends import depth_on_original_rays
from gpt_field.foundation_vision.types import DepthEstimate
from gpt_field.foundation_vision.geometry import build_geometry
R = '/workspace/gpt_field/runs/'
m = load_manifest(R + 'foundation_vision_preparation_001/input/manifest.json')
frame = m.frames[0]; cams = {c.name: c for c in frame.cameras}; names = [c.name for c in frame.cameras]
z = np.load(R + 'foundation_vision_preparation_001/da3_real_003/cold_native_prediction.npz')
raw = z['raw_depth_model_units']; K = z['native_intrinsics']; conf = z['raw_score_map']
def geom(se, sw):
    D = {}
    for i, n in enumerate(names):
        s = se if n == 'external' else sw
        d, sc = depth_on_original_rays(raw[i]*s, K[i], cams[n], conf[i])
        D[n] = DepthEstimate(n, d, cams[n].intrinsics, cams[n].world_to_camera, sc, source='da3')
    return build_geometry(frame, D, {})
TABLE_Z = 0.43676
CAN_LO = np.array([-0.054474949094583634, -0.24536935556741868, 0.43696]); CAN_HI = np.array([0.0077168271458434145, -0.18373144547662518, 0.51315])
AX = (CAN_LO[:2] + CAN_HI[:2]) / 2; RAD = float(np.mean((CAN_HI - CAN_LO)[:2]) / 2)
rng = np.random.default_rng(0)
# (1) points inside the true can volume
n = 2000
th = rng.uniform(0, 2*np.pi, n); rr = RAD*np.sqrt(rng.uniform(0, 1, n)) * 0.9
inside_can = np.column_stack((AX[0] + rr*np.cos(th), AX[1] + rr*np.sin(th), rng.uniform(CAN_LO[2] + 0.005, CAN_HI[2] - 0.005, n)))
# (2) points just under the table top (inside table slab)
in_table = np.column_stack((rng.uniform(-0.3, 0.15, n), rng.uniform(-0.45, 0.45, n), rng.uniform(TABLE_Z - 0.02, TABLE_Z - 0.003, n)))
# (3) points in true free air: 2-4 cm above table, away from known objects (mugs/plates/can xy from initial_state qpos)
objs = np.array([[-0.0233, -0.2178], [-0.0875, -0.1559], [-0.2024, 0.0183], [-0.0276, 0.0779], [-0.0045, -0.3232], [0.0054, 0.3109]])
def free_band(z0, z1):
    P = np.column_stack((rng.uniform(-0.3, 0.15, 4*n), rng.uniform(-0.45, 0.45, 4*n), rng.uniform(z0, z1, 4*n)))
    d = np.min(np.linalg.norm(P[:, None, :2] - objs[None], axis=2), axis=1)
    return P[d > 0.13][:n]
low_air = free_band(TABLE_Z + 0.02, TABLE_Z + 0.04)
mid_air = free_band(0.56, 0.64)
for se, sw, tag in ((0.8780772385418903, 0.8780772385418903, 'baseline_ratio'), (0.84, 0.72, 'selfcal')):
    g = geom(se, sw)
    out = {}
    for name, P in (('inside_true_can', inside_can), ('inside_table_slab', in_table), ('free_air_2to4cm_above_table', low_air), ('free_air_z0.56-0.64', mid_air)):
        res = g.query(P, frame.captured_at_s, 5.0)
        c = Counter(r['state'] for r in res)
        out[name] = {k: round(v / len(P), 3) for k, v in c.items()}
    print(tag, json.dumps(out))
g = geom(0.8780772385418903, 0.8780772385418903)
hand = frame.hand_world; grip = hand[:3, 3] + hand[:3, :3] @ np.array([0, 0, 0.097])
print('grip site', grip.round(4))
paths = {
 'descend_10cm_from_grip_site': [grip, grip + [0, 0, -0.10]],
 'free_air_segment_z0.60_15cm': [np.array([-0.15, 0.25, 0.60]), np.array([0.0, 0.25, 0.60])],
 'through_true_can': [np.array([-0.023, -0.30, 0.48]), np.array([-0.023, -0.13, 0.48])],
 'over_can_8cm_clear': [np.array([-0.023, -0.30, 0.59]), np.array([-0.023, -0.13, 0.59])],
}
for name, pts in paths.items():
    t = time.perf_counter()
    r = g.assess_path(np.array(pts), 0.02, frame.captured_at_s, 5.0)
    dt = time.perf_counter() - t
    print(name, 'status', r['status'], r['reason'], 'obstacle', r.get('obstacle_view'), np.round(r.get('obstacle_point_world', []), 3).tolist() if r.get('obstacle_point_world') else None, 'dist', r.get('surface_distance_m'), 'time %.3f s' % dt)
# how many retained surface points lie within 2.5 cm of the open gripper volume (robot self-surface)?
allp = np.concatenate([v.points_world for v in g._views.values()])
d = np.linalg.norm(allp - grip, axis=1)
print('retained surface hypotheses within 5 cm of grip site:', int((d < 0.05).sum()), 'within 10 cm:', int((d < 0.10).sum()), 'of', len(allp))
# fraction of hypotheses with support>=2 & no contradictions (usable for free-space) per view
for nme, v in g._views.items():
    print(nme, 'free-space-eligible pixels (support>=2 & 0 contra):', round(float(((v.support >= 2) & (v.contradictions == 0)).mean()), 4))
