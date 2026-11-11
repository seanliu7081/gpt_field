"""A15: where does each view put the (flat, z=0.437 m) table? Retained planner cloud, frame 14."""
import json
import numpy as np
from scipy.spatial import cKDTree
from k2common import WORK, scene_at, kin, truth_boxes
from gpt_field.foundation_harness.motion import CandidatePlanner
scene, state, q, record, err = scene_at(13)
pl = CandidatePlanner(kin); pl._prepare(scene, state, q)
cloud, strong = pl._cloud, pl._strong
allp = np.asarray(scene.scene_points, float); valid = np.isfinite(allp).all(1); src = np.flatnonzero(valid)
view_of = np.full(len(allp), '', object)
for n, (a, b) in scene.evidence['scene_point_view_slices'].items():
    view_of[a:b] = n
views = view_of[src[cKDTree(allp[valid]).query(cloud)[1]]]
T = truth_boxes(); tz = T['table_top_z']
out = {'table_top_z_truth': tz}
for v in ('external', 'wrist'):
    z = cloud[views == v, 2]
    # points away from all objects: outside the xy boxes of mugs/can/plates (truth, scoring only)
    out[v] = {'n': int(len(z)), 'z_quantiles_(1,5,25,50,75,95)': np.quantile(z, [.01, .05, .25, .5, .75, .95]).round(3).tolist(),
              'frac_below_table_minus_2cm': round(float((z < tz - 0.02).mean()), 3),
              'frac_within_table_+-1cm': round(float((np.abs(z - tz) <= 0.01).mean()), 3),
              'strong_frac': round(float(strong[views == v].mean()), 3)}
print(json.dumps(out, indent=1))
(WORK / 'data/a15_wrist_table_depth.json').write_text(json.dumps(out, indent=1))
