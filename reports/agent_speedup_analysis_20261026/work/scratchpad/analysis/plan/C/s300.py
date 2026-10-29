"""Plan C S300 arithmetic (read-only, stdlib). Inputs: uc7 summary.json timing_components + runner.log latencies,
C1 trial_003 summary (compact I/O rates). All counterfactuals are ESTIMATES (arithmetic on measured components)."""
import json, os, statistics as st
ROOT = '/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_'
RUNS = ['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t7_001','t8_001','t9_001','t6_009','t6_011']
T3 = '/workspace/gpt_field/results/speedup_evidence_20261026T055123Z/evaluations/trial_003/trial/summary.json'
t3 = json.load(open(T3)); tc3 = t3['timing_components']
VID = tc3['video_render_s'] / t3['steps']; RS = tc3['run_state_s'] / t3['steps']
print('trial_003 compact rates: video %.4f s/step, run_state %.6f s/step, steps %d' % (VID, RS, t3['steps']))
def lat(key):
    out = []
    p = ROOT + key + '_launch/runner.log'
    for line in open(p, errors='replace'):
        line = line.strip()
        if line.startswith('{'):
            try: ev = json.loads(line)
            except Exception: continue
            if ev.get('event') == 'model_call': out.append(ev['latency_s'])
    return out
VETO_REPLACED = {'t4_001': [1], 't8_001': [1], 't9_001': [1], 't6_011': [1, 9]}   # V4b: call index of the replacement
rows = []
for k in RUNS:
    s = json.load(open(ROOT + k + '/summary.json')); tc = s['timing_components']; uc = s['user_check']
    steps = s['steps']; L = lat(k)
    base = uc['preparation_elapsed_s'] + (uc.get('approval_to_claim_s') or 0) + s['elapsed_s']
    compact = base - tc['video_render_s'] - tc['run_state_s'] + (VID + RS) * steps
    veto = sum(L[i] for i in VETO_REPLACED.get(k, []))
    noveto = compact - veto - len(VETO_REPLACED.get(k, [])) * (compact - tc['model_pipeline_s'] - uc['preparation_elapsed_s']) / steps
    model = tc['model_pipeline_s'] - veto
    rows.append(dict(run=k, raw=s['task_success'], steps=steps, calls=len(L), model=round(tc['model_pipeline_s'], 1),
                     base=round(base, 1), compact=round(compact, 1), plan_c=round(noveto, 1),
                     nonmodel_plan_c=round(noveto - model, 1), model_left=round(300 - (noveto - model), 1)))
for r in rows: print(r)
succ = [r for r in rows if r['raw']]
for col in ('base', 'compact', 'plan_c'):
    v = [r[col] for r in succ]
    print(col, 'mean %.1f median %.1f  <=300: %d/11 %s  (+5 s start-up: %d/11)' % (
        st.mean(v), st.median(v), sum(x <= 300 for x in v), [r['run'] for r in succ if r[col] <= 300], sum(x + 5 <= 300 for x in v)))
ml = [r['model_left'] for r in succ]
print('model budget left under 300 s after plan C non-model costs: mean %.1f, range %.1f..%.1f s' % (st.mean(ml), min(ml), max(ml)))
print('  = calls at 15.1 s/call: %.1f ; at 21.5 s/call: %.1f ; actual calls mean %.1f' % (st.mean(ml) / 15.1, st.mean(ml) / 21.5, st.mean(r['calls'] for r in succ)))
json.dump(rows, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/C/s300.json', 'w'), indent=1)
