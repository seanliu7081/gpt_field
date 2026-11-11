import json, sys, os, statistics as st
run = sys.argv[1]
ev = os.path.join(run, 'trial', 'episode.events.jsonl')
rows = []
with open(ev) as f:
    for line in f:
        d = json.loads(line)
        rows.append({k: d.get(k) for k in ('event','step','wall_elapsed_s','decision_latency_s','api_latency_s','api_calls_in_act','execution_kind','frame_count','model_calls','macro','primitive','status','success','video_time_s')})
decision_rows = [r for r in rows if r['event'] in ('agent_action','skill_decision')]
exec_rows = [r for r in rows if r['event']=='executed']
print('events', len(rows), 'decisions', len(decision_rows), 'executed', len(exec_rows))
api_steps = {}
for r in decision_rows:
    if r['api_calls_in_act']:
        api_steps[r['step']] = (r['api_latency_s'], r['decision_latency_s'], r['api_calls_in_act'])
print('decisions with API calls:', len(api_steps))
for s,(a,dl,k) in sorted(api_steps.items()):
    print('  step', s, 'api', round(a,2) if a else a, 'decision_latency', round(dl,2) if dl else dl, k)
walls = [r['wall_elapsed_s'] for r in exec_rows]
nonapi = []; apig = []
for a,b in zip(exec_rows, exec_rows[1:]):
    g = b['wall_elapsed_s'] - a['wall_elapsed_s']
    if b['step'] in api_steps: apig.append((b['step'], g, api_steps[b['step']][0]))
    else: nonapi.append(g)
nonapi_sorted = sorted(nonapi)
print('first exec wall', round(walls[0],2), 'last exec wall', round(walls[-1],2))
print('non-api step gaps: n', len(nonapi), 'mean %.4f median %.4f p10 %.4f p90 %.4f max %.3f sum %.2f' % (st.mean(nonapi), st.median(nonapi), nonapi_sorted[int(0.1*len(nonapi))], nonapi_sorted[int(0.9*len(nonapi))], max(nonapi), sum(nonapi)))
print('api-step gaps: n', len(apig), 'sum gap %.2f sum api %.2f residual %.2f' % (sum(g for s,g,l in apig), sum(l for s,g,l in apig), sum(g-l for s,g,l in apig)))
dls_nonapi = [r['decision_latency_s'] for r in decision_rows if r['step'] not in api_steps and r['decision_latency_s'] is not None]
print('non-api decision_latency: n', len(dls_nonapi), 'sum %.2f mean %.4f' % (sum(dls_nonapi), st.mean(dls_nonapi)))
term = [r for r in rows if r['event']=='terminal']
print('terminal wall', term[0]['wall_elapsed_s'] if term else None, 'video_time', term[0]['video_time_s'] if term else None)
print('first observation wall', rows[0]['wall_elapsed_s'])
# per-decision -> executed (sim step) duration
dec_by_step = {r['step']: r for r in decision_rows}
simd = []
for r in exec_rows:
    d = dec_by_step.get(r['step'])
    if d: simd.append(r['wall_elapsed_s'] - d['wall_elapsed_s'])
print('decision->executed: mean %.4f median %.4f sum %.2f' % (st.mean(simd), st.median(simd), sum(simd)))
# executed->next decision
ex_by_step = {r['step']: r for r in exec_rows}
post = []
for r in decision_rows:
    p = ex_by_step.get(r['step']-1)
    if p and r['step'] not in api_steps: post.append(r['wall_elapsed_s'] - p['wall_elapsed_s'])
print('executed(prev)->decision (non-api): mean %.4f median %.4f sum %.2f' % (st.mean(post), st.median(post), sum(post)))
