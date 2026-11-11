"""How many unknown probes would single-view uncontradicted ray evidence certify? Offline only."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from motion_unblock import restore_masked, prepare, kin, record_by_arrays, feedback_state
from gpt_field.foundation_harness.motion import CandidatePlanner, MotionConfig, _pose
from gpt_field.foundation_vision.geometry import _sample
OUT = Path(__file__).parent
arrays, step = 'frame_000010_revision_000022.npz', 9
record = record_by_arrays(arrays); state, q = feedback_state(step)
scene = restore_masked(record, state, q, 4)
planner = CandidatePlanner(kin, MotionConfig(allow_unknown_space=False)); prepare(planner, scene, state, q, .003)
res = {}
geo = scene.geometry
for v in geo._views.values():
    pass
sup = {n: int(((v.support >= 2) & (v.contradictions == 0)).sum()) for n, v in geo._views.items()}
tot = {n: int(len(v.points_world)) for n, v in geo._views.items()}
res['crossview_supported_depth_pixels'] = {n: [sup[n], tot[n], round(sup[n] / tot[n], 3)] for n in sup}
for name, disp in {'up_5cm': [0, 0, .05], 'down_5cm': [0, 0, -.05], 'astra_step9': [0, -.15, .05]}.items():
    start = _pose(state); goal = start.copy(); goal[:3, 3] += disp
    qg = np.asarray(planner.bound.solve(goal[:3, 3], goal[:3, :3], q, max_iterations=80).joints)
    worst = None
    for qq in np.linspace(np.asarray(q, float), qg, 6)[1:]:
        shapes = planner.robot.shapes(planner.bound, qq, state[6:8])
        probes = np.concatenate([s.surface_points() for s in shapes])
        probes = probes[np.linspace(0, len(probes) - 1, 160, dtype=int)]
        probes = probes[~planner._known_start_volume(probes)]
        rows = geo.query(probes, now_s=0., max_age_s=5.)
        nonfree = np.array([r['state'] != 'estimated_free' for r in rows])
        single = np.zeros(len(probes), bool); anyview = np.zeros(len(probes), int)
        for vname, view in geo._views.items():
            _, _, surface, behind, front, valid = _sample(probes, view, geo.config)
            single |= front & (view.contradictions_map_dummy if False else True)
            anyview += front.astype(int)
        cnt = {'probes_outside_start_volume': int(len(probes)), 'nonfree_runtime_rule': int(nonfree.sum()),
               'nonfree_with_single_view_front_evidence': int((nonfree & (anyview >= 1)).sum()),
               'nonfree_without_any_view_evidence': int((nonfree & (anyview == 0)).sum())}
        if worst is None or cnt['nonfree_runtime_rule'] > worst['nonfree_runtime_rule']:
            worst = cnt
    res[name] = worst
print(json.dumps(res, indent=1))
(OUT / 'free_relax_results.json').write_text(json.dumps(res, indent=1))
