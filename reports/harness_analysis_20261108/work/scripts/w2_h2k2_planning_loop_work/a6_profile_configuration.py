"""A6: where does one keypose-2 _configuration() check spend its time? (cProfile)"""
import cProfile
import io
import json
import pstats
from time import perf_counter

import numpy as np

from k2common import WORK, scene_at, kin
from gpt_field.foundation_harness.motion import CandidatePlanner, MotionConfig

scene, state, q, record, err = scene_at(13)
pl = CandidatePlanner(kin)
pl._prepare(scene, state, q)
pl._deadline = float('inf')
qs = [q + 0.02 * np.sin(np.arange(7) + i) for i in range(20)]
pr = cProfile.Profile()
pr.enable()
for qq in qs:
    pl._configuration(qq, scene, state[6:8], None, ())
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(18)
print(s.getvalue()[:6000])
# split: self-collision only, free-check off, free-check on
shapes = pl.robot.shapes(pl.bound, q, state[6:8])
t0 = perf_counter()
for _ in range(20):
    pl.robot.self_collision(shapes)
sc = (perf_counter() - t0) / 20
t0 = perf_counter()
for qq in qs:
    pl._configuration(qq, scene, state[6:8], None, (), free_check=False)
nf = (perf_counter() - t0) / len(qs)
t0 = perf_counter()
for qq in qs:
    pl._configuration(qq, scene, state[6:8], None, ())
wf = (perf_counter() - t0) / len(qs)
res = {'self_collision_ms': round(sc * 1000, 1), 'configuration_no_free_check_ms': round(nf * 1000, 1),
       'configuration_with_free_check_ms': round(wf * 1000, 1),
       'rrt_budget_configs_at_12s': int(12 / wf), 'note': 'single-thread CPU on a loaded 256-core host; indicative'}
print(json.dumps(res))
(WORK / 'data/a6_profile_configuration.json').write_text(json.dumps({**res, 'profile_top': s.getvalue()[:6000]}, indent=1))
