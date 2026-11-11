import sys, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / 'pkg_cur'))
sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location('csf', HERE / 'current_selffilter.py')
src = (HERE / 'current_selffilter.py').read_text().split('out = []')[0]
ns = {'__file__': str(HERE / 'current_selffilter.py')}
exec(compile(src, 'csf_head', 'exec'), ns)
from gpt_field.foundation_harness.motion import CandidatePlanner
from gpt_field.foundation_harness.surface import planning_surface, _CACHE
res = []
for arrays, step in [('frame_000005_revision_000011.npz', 4), ('frame_000019_revision_000041.npz', 18)]:
    record = next(r for r in ns['RECORDS'] if r['arrays'] == arrays)
    fb = next(x for x in ns['EVENTS'] if x['event'] == 'feedback' and x['step'] == step)
    state, q = np.array(fb['measured_state']), np.array(fb['joint_position'])
    scene = ns['restore'](record)
    planner = CandidatePlanner(ns['kin'])
    _CACHE.clear()
    t0 = time.perf_counter(); planner._prepare(scene, state, q); t1 = time.perf_counter()
    surf = {}
    _CACHE.clear()
    for oid, obj in scene.objects.items():
        s0 = time.perf_counter(); pts, ev = planning_surface(obj); surf[oid] = {'s': round(time.perf_counter() - s0, 3), 'raw': len(obj.points), 'kept': len(pts),
             'xy_extent_m': np.ptp(pts, 0)[:2].round(4).tolist() if len(pts) else None, 'z_range': [round(float(pts[:,2].min()),3), round(float(pts[:,2].max()),3)] if len(pts) else None, 'status': ev.get('status'), 'reason': ev.get('reason')}
    row = {'arrays': arrays, 'prepare_s_including_planning_surface': round(t1 - t0, 3), 'planning_surface': surf}
    res.append(row); print(json.dumps(row, indent=1), flush=True)
(HERE / 'time_current_results.json').write_text(json.dumps(res, indent=1))
