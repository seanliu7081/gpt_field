"""A8: would a keypose-2 payload be blocked by its OWN unowned strong points?

For every saved run-001 snapshot (44 perception revisions at 20 frames) and
every target object: (a) is the keypose-2 planning surface (the payload model
built at close, motion.py:536-544) empty? (b) if not, how many STRONG scene points
that are NOT owned by that object lie inside its convex hull dilated by the
6 mm collision margin (motion.py:267-293 -> payload intersection)? Those points
would block every non-contact keypose of the held object (transport/preplace)
because only owned points are exempt (motion.py:280-282). Offline only.
"""
import json
from collections import defaultdict

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from k2common import WORK, records, restore, kin, events, truth_boxes  # noqa: F401
from gpt_field.foundation_harness.motion import CandidatePlanner
from gpt_field.foundation_harness.surface import planning_surface

rows = []
seen = set()
for rec in records:
    fid = rec['snapshot']['frame_id']
    key = rec['arrays']
    scene, err = restore(rec)  # noqa: F821
    step = rec['snapshot']['evidence']['observation_index']
    fb = next((e for e in events if e['event'] == 'feedback' and e['step'] == max(step, 1)), None)
    state = np.asarray(fb['measured_state']); q = np.asarray(fb['joint_position'])
    pl = CandidatePlanner(kin)
    pl._prepare(scene, state, q)
    cloud, strong, owners = pl._cloud, pl._strong, pl._owners
    for oid, obj in scene.objects.items():
        pts, ev = planning_surface(obj)
        row = {'frame': fid, 'revision': key, 'object': oid, 'query': obj.query, 'status': obj.status,
               'raw_points': int(len(obj.points)), 'planning_points': int(len(pts)),
               'planning_status': ev.get('status'), 'planning_reason': ev.get('reason')}
        if len(pts) >= 4:
            hull = ConvexHull(pts)
            lo, hi = pts.min(0) - 0.01, pts.max(0) + 0.01
            cand = np.flatnonzero(np.all((cloud >= lo) & (cloud <= hi), axis=1))
            inside = cand[np.all(cloud[cand] @ hull.equations[:, :3].T + hull.equations[:, 3] <= 0.006, axis=1)]
            other = inside[owners[inside] != oid]
            T = truth_boxes(); clo, chi = T['can']
            in_can = np.all((cloud[other] >= clo - 0.005) & (cloud[other] <= chi + 0.005), axis=1)
            above = cloud[other, 2] > T['table_top_z'] + 0.015
            own_pts = cloud[owners == oid]
            d_own = cKDTree(own_pts).query(cloud[other])[0] if len(own_pts) else np.full(len(other), np.inf)
            mug_self = other[above & ~in_can & (d_own <= 0.01)]
            row.update({'inside_not_owned_strong_above_table_not_can_within_1cm_of_owned': int(strong[mug_self].sum()),
                        'inside_not_owned_strong_in_can_box': int(strong[other[in_can]].sum()),
                        'inside_not_owned_strong_table_band': int(strong[other[~above]].sum()),'hull_extent_m': np.ptp(pts, axis=0).round(3).tolist(),
                        'inside_hull_points': int(len(inside)),
                        'inside_not_owned': int(len(other)),
                        'inside_not_owned_strong': int(strong[other].sum()),
                        'inside_owned_by_other_target_strong': int((strong[other] & (owners[other] != '')).sum())})
        rows.append(row)
summary = defaultdict(lambda: {'n': 0, 'empty_planning_surface': 0, 'blocked_by_own_unowned_strong': 0})
for r in rows:
    s = summary[r['object']]
    s['n'] += 1
    s['empty_planning_surface'] += int(r['planning_points'] < 4)
    s['blocked_by_own_unowned_strong'] += int(r.get('inside_not_owned_strong', 0) > 0)
    s['self_block_mug_surface_only'] = s.get('self_block_mug_surface_only', 0) + int(r.get('inside_not_owned_strong_above_table_not_can_within_1cm_of_owned', 0) > 0)
    s.setdefault('self_block_counts', []).append(r.get('inside_not_owned_strong_above_table_not_can_within_1cm_of_owned'))
out = {'rows': rows, 'summary': summary}
for r in rows:
    print(json.dumps({k: r[k] for k in r if k not in ('revision',)}))
print(json.dumps(summary, indent=1))
(WORK / 'data/a8_payload_self_block.json').write_text(json.dumps(out, indent=1))
