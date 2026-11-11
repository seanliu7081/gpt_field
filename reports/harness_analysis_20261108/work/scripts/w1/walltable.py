import json, os
base = '/workspace/gpt_field/runs'
runs = {
 'T04 transit100': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
 'T08 transit100': 'agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001',
 'T08 stove-off': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
 'drawer 001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
 'drawer 003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}
rows = []
for name, run in runs.items():
    rs = json.load(open(os.path.join(base, run, 'result_summary.json')))
    s = json.load(open(os.path.join(base, run, 'trial', 'summary.json')))
    t = rs['timing']; tc = s['timing_components']
    wall = t['total_wrapper_wall_s']; prep = t['initial_preparation_elapsed_s']; plan = t['initial_plan_model_s']
    startup = s['startup_to_initial_observation_s']; ctrl = s['control_elapsed_s']; el = s['elapsed_s']
    model = tc['model_pipeline_s']; sim = tc['simulator_step_s']; vid = tc['video_render_s']; rsd = tc['run_state_s']; obs = tc['observe_transition_s']; pol = tc['policy_compute_s']
    resid = ctrl - model - pol - sim - vid - obs
    fin = el - startup - ctrl
    outer = wall - prep - el
    steps = s['steps']
    row = dict(run=name, wall=wall, prep_sim=prep - plan, plan=plan, exec_startup=startup, model=model, sim=sim, video=vid, run_state=rsd, observe_policy=obs + pol, other_ctrl=resid - rsd, finalize=fin, outer=outer, steps=steps, calls=None)
    rows.append(row)
hdr = ['run','wall','prep_sim','plan','exec_startup','model','sim','video','run_state','observe_policy','other_ctrl','finalize','outer','steps']
print(' | '.join(hdr))
for r in rows:
    print(' | '.join(r['run'] if k == 'run' else ('%d' % r[k] if k == 'steps' else '%.1f' % r[k]) for k in hdr))
print()
for r in rows:
    w = r['wall']
    print('%-15s model %.0f%%  video %.0f%%  sim %.0f%%  run_state+observe+other %.0f%%  plan %.0f%%  startup(prep sim+exec sim) %.0f%%  finalize+outer %.0f%%' % (
        r['run'], r['model']/w*100, r['video']/w*100, r['sim']/w*100, (r['run_state']+r['observe_policy']+r['other_ctrl'])/w*100, r['plan']/w*100, (r['prep_sim']+r['exec_startup'])/w*100, (r['finalize']+r['outer'])/w*100))
json.dump(rows, open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/data/walltable.json', 'w'), indent=1)
