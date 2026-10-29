import json, os, sys
root = '/workspace/gpt_field/runs'
runs = ['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t6_009','t6_011','t7_001','t8_001','t9_001','t6_007','t3_002']
for r in runs:
    d = os.path.join(root, 'agent_speedup_v2_user_check_libero_10_' + r)
    s = json.load(open(os.path.join(d, 'summary.json')))
    rv = json.load(open(os.path.join(d, 'user_check_review.json')))
    prep = rv.get('preparation', {})
    tc = s.get('timing_components', {})
    print(r, rv.get('version')[-1], 'status', s.get('status'), 'steps', s.get('steps'),
          'control_elapsed_s', round(s.get('control_elapsed_s') or -1, 1),
          'control_wo_video', round(s.get('control_without_video_s') or -1, 1),
          'elapsed_s', round(s.get('elapsed_s') or -1, 1),
          'video_render_s', round(tc.get('video_render_s') or -1, 1),
          'prep_elapsed_s', round(prep.get('elapsed_s') or -1, 1), 'approval', prep.get('approval_mode'),
          'startup_to_init_obs', s.get('startup_to_initial_observation_s'),
          'created', rv.get('created_at'), 'approved', (rv.get('approved') or {}).get('at'))
