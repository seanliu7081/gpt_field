"""Per true object (evaluation-only boxes): how its observed points are owned / strong / uncertain / self-removed."""
import json, sys
import numpy as np
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
from k2_scene_audit import replica
steps = [int(x) for x in sys.argv[1:]] or [0, 4, 13, 15, 19]
rows = []
for step in steps:
    rec = record_for_step(step); scene, _ = restore(rec); state, q = step_state(step)
    prov = replica(CandidatePlanner(kin), scene, state, q)
    P, keep, S, O, view = prov['points'], prov['keep'], prov['strong'], prov['owners'], prov['view']
    res = {'step': step, 'frame': scene.frame_id, 'targets': {k: o.query for k, o in scene.objects.items()}}
    for name in ('can', 'porcelain_mug_1', 'white_yellow_mug_1', 'red_coffee_mug_1', 'plate_1', 'plate_2'):
        box = in_box(P, truth[name]['lower'], truth[name]['upper'], .005)
        # exclude the table slab below plates/mugs bottoms: only points at least 3 mm above table top
        box &= P[:, 2] > truth['table_top_z'] + .003
        d = {}
        for v in ('external', 'wrist'):
            m = box & (view == v)
            d[v] = {'n': int(m.sum()), 'removed': int((m & ~keep).sum()),
                    'owned': {k: int((m & keep & (O == k)).sum()) for k in scene.objects if (m & keep & (O == k)).sum()},
                    'unowned_strong': int((m & keep & (O == '') & S).sum()),
                    'unowned_uncertain': int((m & keep & (O == '') & ~S).sum())}
        res[name] = d
    rows.append(res); print(json.dumps(res), flush=True)
(W2 / 'k2_object_strength.json').write_text(json.dumps(rows, indent=1))
