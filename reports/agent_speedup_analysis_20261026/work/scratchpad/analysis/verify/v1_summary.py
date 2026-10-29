import json, sys, os
for d in sys.argv[1:]:
    p=os.path.join(d,'summary.json')
    if not os.path.exists(p):
        print(d, 'no summary'); continue
    s=json.load(open(p))
    tc=s.get('timing_components') or {}
    mc=s.get('model_calls') or {}
    print(os.path.basename(d.rstrip('/')), '| status', s.get('status'), '| libero', s.get('libero_success'), '| v2', s.get('v2_success'), '| v3', s.get('v3_success'),
          '| api_calls', s.get('api_calls'), '| steps', s.get('steps'),
          '| control_elapsed_s', s.get('control_elapsed_s'), '| control_wo_video', s.get('control_without_video_s'),
          '| elapsed_s', s.get('elapsed_s'),
          '| model_pipeline_s', tc.get('model_pipeline_s'), '| decision_s', tc.get('decision_s'), '| observe', tc.get('observe_transition_s'), '| sim', tc.get('simulator_step_s'), '| video', tc.get('video_render_s'), '| policy', tc.get('policy_compute_s'), '| run_state', tc.get('run_state_s'),
          '| latency_by_kind', mc.get('latency_s_by_kind'), '| prompt_version', s.get('prompt_version'))
