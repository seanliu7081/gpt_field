import json, os, sys, glob
out = {}
for d in sorted(glob.glob('/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_t*_0??') + glob.glob('/workspace/gpt_field/runs/agent_speedup_v3_*')):
    p = os.path.join(d, 'summary.json')
    if not os.path.exists(p) or d.endswith(('_launch', '_review')):
        continue
    s = json.load(open(p))
    uc = s.get('user_check') or {}
    tc = s.get('timing_components') or {}
    b = s.get('budget') or {}
    perf = None
    pp = os.path.join(d, 'performance.json')
    if os.path.exists(pp):
        perf = json.load(open(pp))
    out[os.path.basename(d)] = dict(
        version=uc.get('version'), status=s.get('status'), steps=s.get('steps'), api_calls=s.get('api_calls'),
        elapsed_s=s.get('elapsed_s'), control_elapsed_s=s.get('control_elapsed_s'),
        control_without_video_s=s.get('control_without_video_s'), libero_success=s.get('libero_success'),
        v2_success=s.get('v2_success'), timing=tc,
        budget=dict((k, b.get(k)) for k in ('api_calls', 'input_tokens', 'cached_input_tokens', 'output_tokens', 'reasoning_tokens', 'elapsed_s')),
        perf_completion=(perf or {}).get('completion_target'),
    )
json.dump(out, open(sys.argv[1], 'w'), indent=1)
for k, v in out.items():
    t = v['timing']
    print(k, v['version'], v['status'], v['steps'], v['api_calls'], round(v['elapsed_s'] or 0, 1),
          'model', round(t.get('model_pipeline_s') or 0, 1), 'ctrl', round(v['control_elapsed_s'] or 0, 1),
          'in_tok', v['budget']['input_tokens'], 'cached', v['budget']['cached_input_tokens'], 'out', v['budget']['output_tokens'], 'reas', v['budget']['reasoning_tokens'])
