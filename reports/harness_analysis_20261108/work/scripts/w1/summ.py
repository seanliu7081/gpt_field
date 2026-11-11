import json, sys, os
for run in sys.argv[1:]:
    p = os.path.join(run, 'trial', 'summary.json')
    if not os.path.exists(p):
        print(run, 'NO trial/summary.json'); continue
    d = json.load(open(p))
    tc = d.get('timing_components') or {}
    uc = d.get('user_check') or {}
    st = d.get('steps')
    print('==', run)
    print('  status', d.get('status'), 'libero', d.get('libero_success'), 'v2', d.get('v2_success'), 'steps', st)
    print('  control_elapsed %.1f without_video %.1f elapsed %.1f startup_to_obs %s' % (d.get('control_elapsed_s') or 0, d.get('control_without_video_s') or 0, d.get('elapsed_s') or 0, d.get('startup_to_initial_observation_s')))
    print('  timing', {k: round(v,2) for k,v in tc.items() if isinstance(v,(int,float))})
    if st:
        print('  per-step: sim %.4f video %.4f run_state %.4f observe %.4f policy %.4f' % tuple((tc.get(k,0) or 0)/st for k in ('simulator_step_s','video_render_s','run_state_s','observe_transition_s','policy_compute_s')))
    print('  prep elapsed', uc.get('preparation_elapsed_s'), 'prep model', uc.get('preparation_model_s'))
    mc = d.get('model_calls') or {}
    print('  model_calls', {k: mc.get(k) for k in ('requests_sent','by_kind','failed') if k in mc})
    col = (d.get('collision') or {}).get('counts') or {}
    print('  forbidden', col.get('forbidden_contact_points'), {k:v for k,v in col.items() if k.startswith('points_') and v and 'passive' not in k and 'finger_object' not in k})
