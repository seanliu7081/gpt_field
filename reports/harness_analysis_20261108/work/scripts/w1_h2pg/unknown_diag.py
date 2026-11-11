"""Why is swept space 'unknown'? And what blocks a 5 cm descent? Offline only."""
import sys, json
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from motion_unblock import restore_masked, prepare, kin, record_by_arrays, feedback_state
from gpt_field.foundation_harness.motion import CandidatePlanner, MotionConfig, _pose
OUT = Path(__file__).parent
arrays, step = 'frame_000010_revision_000022.npz', 9
record = record_by_arrays(arrays); state, q = feedback_state(step)
res = {}
for variant, dil, margin in [('consistent_6.5mm', None, .0065), ('image_silhouette_4px', 4, .003)]:
    scene = restore_masked(record, state, q, dil)
    planner = CandidatePlanner(kin, MotionConfig(allow_unknown_space=True))
    prepare(planner, scene, state, q, margin)
    out = {}
    for name, disp in {'up_5cm': [0, 0, .05], 'down_5cm': [0, 0, -.05]}.items():
        start = _pose(state); goal = start.copy(); goal[:3, 3] += disp
        ik = planner.bound.solve(goal[:3, 3], goal[:3, :3], q, max_iterations=80)
        qg = np.asarray(ik.joints)
        edge = planner._edge(np.asarray(q, float), qg, scene, np.asarray(state[6:8]), None, ())
        # probe classification at the goal configuration
        shapes = planner.robot.shapes(planner.bound, qg, state[6:8])
        probes = np.concatenate([s.surface_points() for s in shapes])
        kinds = np.concatenate([[s.name] * len(s.surface_points()) for s in shapes])
        sel = np.linspace(0, len(probes) - 1, planner.config.free_probe_limit, dtype=int)
        probes, kinds = probes[sel], kinds[sel]
        known = planner._known_start_volume(probes)
        rows = scene.geometry.query(probes[~known], now_s=0., max_age_s=5.)
        reasons = Counter((r['state'], r['reason']) for r in rows)
        by_part = Counter(k for k, r in zip(kinds[~known], rows) if r['state'] != 'estimated_free')
        free_views = Counter(len(r['free_evidence_views']) for r in rows)
        out[name] = {'ik_ok': ik.ok, 'direct_edge': {k: edge.get(k) for k in ('safe', 'reason', 'part', 'point_world_m', 'point_object_id', 'unknown_samples')},
                     'goal_probe_known_in_start_volume': int(known.sum()), 'goal_probe_states': {f'{a}:{b}': c for (a, b), c in reasons.items()},
                     'non_free_probes_by_part': dict(by_part), 'free_view_count_hist': dict(free_views)}
        if edge.get('point_world_m'):
            p = np.array(edge['point_world_m']); hand = next(s for s in planner._initial_shapes if s.name == 'hand_collision')
            out[name]['direct_edge']['blocking_point_minus_start_hand_center_m'] = (p - hand.center).round(3).tolist()
    res[variant] = out
    print(variant, json.dumps(out, indent=1), flush=True)
(OUT / 'unknown_diag_results.json').write_text(json.dumps(res, indent=1))
