"""A1: what the keypose-2 planner treats as blocking (strong) vs ignorable (uncertain).

Offline on saved run-001 public inputs (frame for observation index 13 by default).
Scores the planner's post-self-mask cloud against evaluator layout truth (can box,
table plane) -- truth is used ONLY for scoring here, never by the policy.
"""
import json
import sys
from time import perf_counter

import numpy as np

from k2common import WORK as W2, scene_at, kin, truth_boxes
from gpt_field.foundation_harness.motion import CandidatePlanner

step = int(sys.argv[1]) if len(sys.argv) > 1 else 13
scene, state, q, record, err = scene_at(step)
planner = CandidatePlanner(kin)
t0 = perf_counter()
planner._prepare(scene, state, q)
prepare_s = perf_counter() - t0
cloud, strong, owners = planner._cloud, planner._strong, planner._owners
T = truth_boxes()
lo, hi = T['can']
table_z = T['table_top_z']

# view of each retained point (scene_points are concatenated per view in order)
slices = scene.evidence['scene_point_view_slices']
all_pts = np.asarray(scene.scene_points, float)
valid = np.isfinite(all_pts).all(1)
src_idx = np.flatnonzero(valid)
view_of = np.full(len(all_pts), '', object)
for name, (a, b) in slices.items():
    view_of[a:b] = name
# reconstruct the keep mask by matching retained points back to source indices
from scipy.spatial import cKDTree
tree = cKDTree(all_pts[valid])
d, idx = tree.query(cloud)
assert d.max() < 1e-9
views = view_of[src_idx[idx]]


def summary(mask, label):
    m = np.asarray(mask, bool)
    out = {'region': label, 'points': int(m.sum()), 'strong': int((m & strong).sum()),
           'uncertain': int((m & ~strong).sum())}
    for v in ('external', 'wrist'):
        mv = m & (views == v)
        out[f'{v}_points'] = int(mv.sum())
        out[f'{v}_strong'] = int((mv & strong).sum())
    return out

regions = []
for margin in (0.0, 0.01):
    inside = np.all((cloud >= lo - margin) & (cloud <= hi + margin), axis=1)
    regions.append(summary(inside, f'can_box_dilated_{int(margin*1000)}mm'))
regions.append(summary(np.abs(cloud[:, 2] - table_z) <= 0.01, 'table_plane_+-1cm'))
regions.append(summary(cloud[:, 2] > 0.60, 'above_z_0.60 (no scene object; robot/phantom only)'))
regions.append(summary((cloud[:, 2] > 0.75) & (cloud[:, 2] < 0.90), 'z_0.75_0.90 (debug-hit band)'))
regions.append(summary(owners != '', 'owned_by_targets'))
regions.append(summary(np.ones(len(cloud), bool), 'all_retained'))

# Robot pose and link positions at the measured start
shapes = planner.robot.shapes(planner.bound, q, state[6:8])
links = {s.name: np.round(s.center, 3).tolist() for s in shapes}

out = {'step': step, 'frame_id': scene.frame_id, 'arrays': record['arrays'], 'prepare_s': prepare_s,
       'self_filter': planner._self_filter_evidence, 'regions': regions,
       'targets_in_snapshot': {k: {'query': o.query, 'status': o.status, 'points': int(len(o.points))}
                               for k, o in scene.objects.items()},
       'robot_link_centers_at_start': links, 'eef_start': state[:3].tolist()}
print(json.dumps(out, indent=1))
(W2 / 'data').mkdir(exist_ok=True)
(W2 / f'data/a1_scene_strength_step{step}.json').write_text(json.dumps(out, indent=1))
