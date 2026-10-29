import json, os, sys, glob
root = '/workspace/gpt_field/runs'
dirs = sorted(d for d in glob.glob(os.path.join(root, 'agent_speedup_v2_user_check_libero_10_*')) if not d.endswith(('_launch', '_review')))
for d in dirs:
    name = os.path.basename(d).replace('agent_speedup_v2_user_check_libero_10_', '')
    pt = os.path.join(d, 'physical_truth.json')
    sm = os.path.join(d, 'summary.json')
    rv = os.path.join(d, 'user_check_review.json')
    ver = None
    if os.path.exists(rv):
        try:
            ver = json.load(open(rv)).get('version')
        except Exception as e:
            ver = 'ERR'
    s = json.load(open(sm)) if os.path.exists(sm) else {}
    if not os.path.exists(pt):
        print(name, ver, 'status', s.get('status'), 'NO physical_truth'); continue
    p = json.load(open(pt))
    su = p.get('summary', {})
    print(name, ver, 'status', s.get('status'), 'steps', s.get('steps'), 'libero', s.get('libero_success'), 'v2', s.get('v2_success'),
          '| raw', su.get('raw_libero_success'), 'raw_after', su.get('raw_libero_success_after_settle'),
          'strict_term', su.get('physical_success_at_terminal'), 'strict_after', su.get('physical_success_after_settle'),
          'verdict', su.get('verdict'), 'holdonly', su.get('terminal_failure_is_hold_only'))
    print('    fail', json.dumps(su.get('terminal_failing_checks')))
