"""Run-001 link0 blocking points under the keypose-2 self-filter, and whether link0-region points are still in the cloud."""
import json, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa
from k2_scene_audit import replica
rr = json.load(open(RUN / 'analysis/reconstructed_rejections.json'))['results']
out = []
for r in rr:
    step = r['step']; hit = r['initial_configuration']['point_world_m']
    rec = record_for_step(step); scene, _ = restore(rec); state, q = step_state(step)
    planner = CandidatePlanner(kin)
    prov = replica(planner, scene, state, q)
    P = prov['points']; d, i = cKDTree(P).query(hit)
    shapes = planner.robot.shapes(planner.kinematics.bind(state, q), q, state[6:8])
    link0 = next(s for s in shapes if s.name == 'link0_collision')
    inside = link0.contains(P, .006) & prov['keep']
    out.append({'step': step, 'run001_link0_hit_point': [round(v, 4) for v in hit], 'match_dist_m': float(d),
                'keypose2_kept': bool(prov['keep'][i]), 'removed_by': prov['removed_by'][i], 'view': prov['view'][i],
                'strong': bool(prov['strong'][i]),
                'kept_points_inside_link0_hull_6mm': int(inside.sum()), 'of_which_strong': int((inside & prov['strong']).sum())})
    print(json.dumps(out[-1]), flush=True)
    if len(out) >= 4: break
(W2 / 'k2_link0_points.json').write_text(json.dumps(out, indent=1))
