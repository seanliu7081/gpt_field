import json, sys, os
base = '/workspace/gpt_field/runs'
D = '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/data/'
runs = {
 't04_t100': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
 't08_t100': 'agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001',
 't08_stoveoff': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
 'drawer_001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
 'drawer_003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}
res = {}
for key, run in runs.items():
    summ = json.load(open(os.path.join(base, run, 'trial', 'summary.json')))
    tc = summ['timing_components']; steps = summ['steps']
    per_step_novideo = (summ['control_without_video_s'] - tc['model_pipeline_s']) / steps
    per_step_video = (summ['control_elapsed_s'] - tc['model_pipeline_s']) / steps
    d = json.load(open(D + key + '.json'))
    api = {int(k): v for k, v in d['api_steps'].items()}
    calls = sorted(api)
    # execution steps after each call until next call (or episode end)
    ends = calls[1:] + [steps]
    lat = [api[s] for s in calls]
    exec_steps = [e - s for s, e in zip(calls, ends)]
    # upper bound: overlap call k (k>=1, i.e. not first) with the execution of fragment k-1?  Use: the call k could run during fragment k-1 (speculative/async).
    ub_video = sum(min(lat[k], exec_steps[k-1] * per_step_video) for k in range(1, len(calls)))
    ub_novideo = sum(min(lat[k], exec_steps[k-1] * per_step_novideo) for k in range(1, len(calls)))
    ub_realtime = sum(min(lat[k], exec_steps[k-1] * 0.05) for k in range(1, len(calls)))
    res[key] = dict(calls=len(calls), model_s=round(sum(lat),1), mean_exec_steps_per_call=round(sum(exec_steps)/len(calls),1),
                    per_step_video=round(per_step_video,4), per_step_novideo=round(per_step_novideo,4),
                    overlap_ub_with_current_overhead_s=round(ub_video,1), overlap_ub_no_video_s=round(ub_novideo,1), overlap_ub_if_realtime_sim_s=round(ub_realtime,1))
    print(key, res[key])
json.dump(res, open(D + 'overlap.json', 'w'), indent=1)
