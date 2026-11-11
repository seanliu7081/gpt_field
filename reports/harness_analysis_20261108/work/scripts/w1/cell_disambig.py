"""Offline: would the plan's own external-view cell (8x8 grid, 1..64) disambiguate SAM candidates?"""
import json, sys
import numpy as np
from pathlib import Path

run = Path(sys.argv[1])
plan = json.load(open(run/'trial'/'approved_plan.json'))
cells = {p['id']: (p['x'], p['w']) for p in plan['parts']}
fv = run/'trial'/'foundation_vision'
H = W = 512; S = 64
def cell_box(c):
    r, k = (c-1)//8, (c-1)%8
    return k*S, r*S, (k+1)*S, (r+1)*S
stats = {'ambiguous_frames': 0, 'resolved_by_cell': 0, 'single_ext_candidates': 0, 'single_consistent_with_cell': 0}
for meta in sorted(fv.glob('frame_*_sam_candidates.json')):
    m = json.load(open(meta)); masks = np.load(fv/m['masks']['path'])
    for key, (xc, wc) in cells.items():
        ext = [r for r in m['candidates'] if r['target_key'] == key and r['view'] == 'external']
        if xc == 0 or not ext: continue
        u0, v0, u1, v1 = cell_box(xc)
        hits = []
        for r in ext:
            mk = masks[r['array_key']]
            hits.append(int(mk[v0:v1, u0:u1].sum()))
        overl = [h > 0 for h in hits]
        if len(ext) > 1:
            stats['ambiguous_frames'] += 1
            if sum(overl) == 1: stats['resolved_by_cell'] += 1
            print(m['frame_id'], key, 'cell', xc, 'cands', len(ext), 'overlap px', hits)
        else:
            stats['single_ext_candidates'] += 1
            stats['single_consistent_with_cell'] += int(overl[0])
            if not overl[0]:
                print(m['frame_id'], key, 'cell', xc, 'single candidate NOT in plan cell; bbox', ext[0]['bbox_xyxy_exclusive'])
print(stats)
