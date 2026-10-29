import json, statistics as st
rows = json.load(open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v5_plan_rows.json'))
def is_prep_dir(p):
    return ('_review' in p) or ('speedup_evidence' in p) or ('v3_completion_baseline' in p)
prep = [r for r in rows if is_prep_dir(r['path'])]
inrun = [r for r in rows if not is_prep_dir(r['path'])]
robo = [r for r in prep if 'robocasa' in r['path']]
core = [r for r in prep if 'robocasa' not in r['path']]
print('plan/replan requests total', len(rows), 'prep-dir', len(prep), '(robocasa', len(robo), ') in-run', len(inrun))
def tail(r):
    d = r.get('client_interval_s')
    return (r.get('status') != 'valid') or (d is not None and d >= 120)
# replays from request_diagnostics: two censored at 120 s (prep_001, prep_003), two completed 17.31 / 11.21 s
replays = [dict(path='speedup_next prep_001', compact_in_payload=False, prompt_version='agent-speedup-v2-user-check-prompt-5', status='deadline_exceeded', client_interval_s=120.004),
           dict(path='speedup_next prep_002', compact_in_payload=False, prompt_version='agent-speedup-v2-user-check-prompt-5', status='valid', client_interval_s=17.308),
           dict(path='speedup_next prep_003', compact_in_payload=False, prompt_version='agent-speedup-v2-user-check-prompt-5', status='deadline_exceeded', client_interval_s=120.003),
           dict(path='speedup_next prep_004', compact_in_payload=False, prompt_version='agent-speedup-v2-user-check-prompt-5', status='valid', client_interval_s=11.215)]
def summarize(label, rs):
    t = [r for r in rs if tail(r)]
    ds = [r['client_interval_s'] for r in rs if r.get('client_interval_s') is not None]
    comp = [r['client_interval_s'] for r in rs if r.get('status') == 'valid' and r.get('client_interval_s') is not None and r['client_interval_s'] < 120]
    print('%-62s n=%2d tails=%2d  mean %.1f median %.1f  max completed<120: %.2f' % (label, len(rs), len(t), st.mean(ds) if ds else float('nan'), st.median(ds) if ds else float('nan'), max(comp) if comp else float('nan')))
    return t
compact = [r for r in core if r['compact_in_payload']]
noncompact = [r for r in core if not r['compact_in_payload']]
p5 = [r for r in noncompact if r['prompt_version'] == 'agent-speedup-v2-user-check-prompt-5']
other = [r for r in noncompact if r['prompt_version'] != 'agent-speedup-v2-user-check-prompt-5']
summarize('compact (core prep dirs)', compact)
t = summarize('non-compact prompt-5 (core prep dirs), logged', p5)
summarize('non-compact prompt-5 logged + 4 replays', p5 + replays)
summarize('non-compact other prompts (core prep dirs)', other)
for r in other: print('     other:', r['path'][:70], r['prompt_version'], r['effort'], round(r['client_interval_s'], 2))
summarize('ALL non-compact core prep logged + replays', noncompact + replays)
summarize('ALL non-compact prep incl robocasa logged + replays', noncompact + robo + replays)
summarize('ALL non-compact plan/replan incl in-run + robocasa + replays', noncompact + robo + inrun + replays)
print('tails:')
for r in t + [x for x in replays if tail(x)]:
    print('   ', r['path'][:80], r.get('status'), r.get('error_type'), r.get('error_code'), round(r['client_interval_s'], 2) if r.get('client_interval_s') is not None else None)
print('robocasa rows:', [(r['path'][:45], r.get('status'), r.get('client_interval_s')) for r in robo])
print('compact all valid:', all(r['status'] == 'valid' for r in compact), 'durations', sorted(round(r['client_interval_s'], 2) for r in compact))
print('any payload with max_output_tokens:', any(r['max_output_tokens_in_payload'] for r in rows))
print('output_token_limit_sent values:', set(str(r.get('output_token_limit_sent')) for r in rows))
print('resp max_output_tokens values:', set(str(r.get('resp_max_output_tokens')) for r in rows if 'resp_max_output_tokens' in r))
print('timeouts:', set(r['timeout_s'] for r in rows))
uc7 = ['t0_001', 't1_001', 't2_002', 't3_003', 't3_005', 't4_001', 't5_001', 't5_002', 't6_009', 't6_011', 't7_001', 't8_001', 't9_001']
for k in uc7:
    rr = [r for r in rows if r['path'].startswith('runs/agent_speedup_v2_user_check_libero_10_%s_review/' % k)]
    print('uc7', k, [(r['prompt_version'], r['compact_in_payload'], r['max_output_tokens_in_payload'], r['wire']) for r in rr])
# pretty-printed among completed non-compact core
pp = [r for r in p5 if r.get('status') == 'valid' and r.get('out_ws', {}).get('ws_outside', 0) > 0]
print('completed prompt-5 non-compact:', sum(1 for r in p5 if r.get('status') == 'valid'), 'with outside-string whitespace:', len(pp), 'range', min(r['out_ws']['ws_outside'] for r in pp if r['out_ws']['ws_outside'] < 10000), max(r['out_ws']['ws_outside'] for r in pp if r['out_ws']['ws_outside'] < 10000))
print('compact outputs outside-string whitespace:', [r['out_ws']['ws_outside'] for r in compact])
