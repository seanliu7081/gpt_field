"""Plan B helper: print each uc7 decision compactly (read-only)."""
import json, sys, os
R = '/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_'
name = sys.argv[1]
h = json.load(open(R + name + '/user_check_plan_history.json'))
for d in sorted(h['model_decisions'], key=lambda x: x['decision']):
    a = d['action']
    segs = []
    if a['type'] == 'execute':
        for s in a['execution']['segments']:
            xyz = s['xyz_m']
            segs.append(f"{s['skill']}/{s['profile'][0]}/{s['gripper']}/{s['max_steps']}/{'OA' if s['observation_after'] else '--'}" + ('' if xyz is None else '@' + ','.join(f'{v:.3f}' for v in xyz)) + ('' if s['quaternion_wxyz'] is None else 'q'))
    elif a['type'] == 'motion':
        m = a['motion']; segs.append(f"motion {m['position_mode']} {m['xyz_m']} {m['gripper']} {m['steps']}")
    print(f"#{d['decision']:02d} step {d['step']:4d} {a['type']}: " + ' | '.join(segs))
    print('    PLAN:', a['plan'][:400].replace('\n', ' '))
    print('    WHY :', a['why'][:300].replace('\n', ' '))
