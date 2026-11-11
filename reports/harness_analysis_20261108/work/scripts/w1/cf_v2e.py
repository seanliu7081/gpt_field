"""First failing sample on the straight edge for the 6 cm pregrasp after masking."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
src = (HERE / 'cf_v2d.py').read_text().split('\nout = {}\nfor step')[0]
ns = {'__file__': str(HERE / 'cf_v2d.py')}
exec(compile(src, 'cf_v2d_head', 'exec'), ns)
restore, step_inputs, m2, kin, planning_surface, masked_scene, residual = (ns[k] for k in (
    'restore', 'step_inputs', 'm2', 'kin', 'planning_surface', 'masked_scene', 'residual'))
rec, state, q = step_inputs(18)
scene, removed, fg = masked_scene(restore(rec), state, q)
pts, _ = planning_surface(scene.objects['white_mug'])
center, top = pts.mean(0), float(pts[:, 2].max())
R = m2._pose(state)[:3, :3]
out = {}
for label, dz in (('pregrasp_6cm', .06), ('pregrasp_12cm', .12)):
    planner = m2.CandidatePlanner(kin)
    planner._prepare(scene, state, q)
    planner._deadline = time.perf_counter() + 60
    planner._contact_parts = ()
    ik = planner.bound.solve(np.array([center[0], center[1], top + dz]), R, q, max_iterations=80)
    goal = np.asarray(ik.joints)
    n = max(1, int(np.ceil(np.max(np.abs(goal - q)) / planner.config.max_joint_step_rad)))
    rows = []
    for i, qq in enumerate(np.linspace(q, goal, n + 1)):
        r = planner._configuration(qq, scene, state[6:8], None, ())
        if not r['safe']:
            p = np.asarray([r['point_world_m']]) if r.get('point_world_m') else None
            rows.append({'sample': i, 'reason': r['reason'], 'part': r.get('part'), 'owner': r.get('point_object_id'),
                         'point': r.get('point_world_m'),
                         'residual_to_current_robot_m': None if p is None else float(min(residual(s, p)[0] for s in planner._initial_shapes))})
            break
    # goal configuration alone
    g = planner._configuration(goal, scene, state[6:8], None, ())
    out[label] = {'ik_ok': bool(ik.ok), 'samples': n + 1, 'first_failure': rows,
                  'goal_config': {k: g.get(k) for k in ('safe', 'reason', 'part', 'point_object_id', 'point_world_m')}}
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
