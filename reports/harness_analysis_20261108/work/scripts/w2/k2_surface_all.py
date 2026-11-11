"""surface.planning_surface on every saved target of every saved revision; score extents vs evaluation-only AABBs."""
import json, sys
from time import perf_counter
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
from gpt_field.foundation_harness import surface as S_
match = {'white mug': 'porcelain_mug_1', 'textured ceramic mug': 'porcelain_mug_1', 'yellow and white mug': 'white_yellow_mug_1',
         'left plate': 'plate_1', 'right plate': 'plate_2'}
rows = []
for rec in records:
    data = np.load(V / rec['arrays'])
    for i, obj in enumerate(rec['snapshot']['objects']):
        o = ObjectGeometry(obj['id'], obj['query'], data[f'object_{i}_points'], data[f'object_{i}_normals'], obj['status'], evidence=obj['evidence'])
        S_._CACHE.clear(); S_._CACHE_POINTS = 0
        t0 = perf_counter(); pts, ev = planning_surface(o); dt = perf_counter() - t0
        raw = np.asarray(o.points, float)
        row = {'arrays': rec['arrays'], 'frame': rec['snapshot']['frame_id'], 'id': obj['id'], 'query': obj['query'], 'status': obj['status'],
               'raw': len(raw), 'planning': len(pts), 'surface_status': ev.get('status'), 'reason': ev.get('reason'),
               'components': ev.get('component_count'), 'mass_fraction': ev.get('selected_mass_fraction'),
               'second_ratio': ev.get('second_component_mass_ratio'), 'extract_s': dt,
               'views': obj['evidence'].get('accepted_points_by_view')}
        if len(raw):
            row['raw_extent'] = np.ptp(raw, 0).round(3).tolist()
        if len(pts):
            row['planning_extent'] = np.ptp(pts, 0).round(3).tolist()
            row['planning_bounds'] = [pts.min(0).round(3).tolist(), pts.max(0).round(3).tolist()]
        t = match.get(obj['query'])
        if t and len(pts):
            lo, hi = np.array(truth[t]['lower']), np.array(truth[t]['upper'])
            row['truth'] = t; row['truth_extent'] = (hi - lo).round(3).tolist()
            row['bound_overshoot_lo_hi_m'] = [(lo - pts.min(0)).round(3).tolist(), (pts.max(0) - hi).round(3).tolist()]
            inside = np.all((pts >= lo - .01) & (pts <= hi + .01), axis=1)
            row['planning_points_within_truth_box_1cm'] = float(inside.mean())
        rows.append(row)
(W2 / 'k2_surface_all.json').write_text(json.dumps(rows, indent=1))
from collections import Counter
print('n objects-revisions', len(rows))
print(Counter((r['query'], r['surface_status'], r['reason']) for r in rows))
for r in rows:
    if r['query'] in match:
        print(r['frame'], r['id'], r['status'], r['raw'], r['planning'], r['surface_status'], r.get('components'), round(r.get('mass_fraction') or 0, 3),
              r.get('raw_extent'), r.get('planning_extent'), r.get('truth_extent'), r.get('bound_overshoot_lo_hi_m'), r.get('planning_points_within_truth_box_1cm'), round(r['extract_s'], 3))
