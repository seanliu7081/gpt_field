import json, sys
sys.path.insert(0, '/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/src')
from gpt_field.agent_speedup_v2_user_check import validate_review_plan, plan_validation, ReviewError
base = '/workspace/gpt_field/reports/uc7_direction_semantics_20261028/implementation_uc71/v1_planning/'
import glob, os
res = {}
for d in sorted(glob.glob(base + '*/review.json')):
    name = d.split('/')[-2]
    r = json.load(open(d))
    plan = r.get('draft_plan') or r.get('plan')
    steps = [(s['do'], s['obj'], s['sense'], s.get('ref')) for s in plan['steps']]
    try:
        validate_review_plan(plan)
        res[name] = ('valid', steps)
    except ReviewError as e:
        res[name] = ('INVALID: ' + str(e), steps)
    except Exception as e:
        res[name] = ('EXC: %s %s' % (type(e).__name__, e), steps)
# also uc7.1 approved plans in runs
for p in sorted(glob.glob('/workspace/gpt_field/runs/agent_speedup_v2_uc71_*/trial/approved_plan.json')):
    name = p.split('/')[-3]
    r = json.load(open(p))
    plan = r.get('plan') if 'plan' in r else r
    try:
        validate_review_plan(plan)
        res[name] = ('valid', len(plan['steps']))
    except Exception as e:
        res[name] = ('INVALID/EXC %s: %s' % (type(e).__name__, e), None)
for k, v in res.items():
    print(k, v)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
