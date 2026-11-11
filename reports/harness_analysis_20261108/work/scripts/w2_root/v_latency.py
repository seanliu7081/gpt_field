"""Independent per-call latency regression for the uc7.1 family (186 executive calls), from raw artifacts.

Tokens: differences of cumulative budget counters in trial/requests/*.result.json (file order = call order).
Latency: (a) trace.jsonl action.diagnostics.api_calls[].latency_s; (b) result budget elapsed differences.
Fits: reasoning+output; input only; reasoning+visible+input (h1_speed_attempts form); +run fixed effects; +call index.
"""
import json, glob
from pathlib import Path
import numpy as np

RUNS = Path('/workspace/gpt_field/runs')
UC71 = {
    'H1_T04': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
    'H1_T08': 'agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001',
    'H3_T04': 'foundation_vision_t04_can_obstacle_20261105_001',
    'H3_T08': 'foundation_vision_t08_bottle_obstacle_20261105_001',
    'T20_T04': 'agent_speedup_v2_uc71_can_obstacle_t04_20261103_001',
    'DEF_T04': 'agent_speedup_v2_uc71_default_routing_t04_20261103_001',
    'SOFF_T08': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
    'DRW_001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
    'DRW_003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}
VISION_S = {'H3_T04': 14.73370562470518 / 11, 'H3_T08': 13.208119614282623 / 11}


def ols(X, y, names):
    X = np.column_stack([np.ones(len(y))] + X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    n, k = X.shape
    sigma2 = ss_res / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {'r2': r2, 'n': n, 'coef': {nm: (float(b), float(s)) for nm, b, s in zip(['const'] + names, beta, se)}}


def main():
    rows = []
    for key, run in UC71.items():
        T = RUNS / run / 'trial'
        trace = [json.loads(l) for l in open(T / 'trace.jsonl')]
        lats = []
        for r in trace:
            if r.get('event') == 'action':
                for c in (r.get('diagnostics') or {}).get('api_calls') or []:
                    lats.append((r['step'], c['latency_s']))
        files = sorted(glob.glob(str(T / 'requests' / '*.result.json')))
        assert len(files) == len(lats), (key, len(files), len(lats))
        prev = {'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0, 'cached_input_tokens': 0, 'elapsed_s': 0.0}
        for i, (f, (step, lat)) in enumerate(zip(files, lats)):
            b = json.load(open(f))['budget']
            d = {k: b[k] - prev[k] for k in prev}
            prev = {k: b[k] for k in prev}
            rows.append({'run': key, 'idx': i, 'step': step, 'lat': lat, 'lat_novision': lat - VISION_S.get(key, 0.0),
                         'inp': d['input_tokens'], 'out': d['output_tokens'], 'rsn': d['reasoning_tokens'],
                         'vis': d['output_tokens'] - d['reasoning_tokens'], 'cached': d['cached_input_tokens'],
                         'budget_elapsed': d['elapsed_s']})
    json.dump(rows, open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/v_percall.json', 'w'), indent=0)
    y = np.array([r['lat'] for r in rows])
    yv = np.array([r['lat_novision'] for r in rows])
    inp = np.array([r['inp'] for r in rows]) / 1000.0
    out = np.array([r['out'] for r in rows], float)
    rsn = np.array([r['rsn'] for r in rows], float)
    vis = np.array([r['vis'] for r in rows], float)
    idx = np.array([r['idx'] for r in rows], float)
    first = (idx == 0).astype(float)
    print('n calls', len(rows))
    print('sum latency', y.sum(), 'sum budget elapsed', sum(r['budget_elapsed'] for r in rows))
    print('corr(inp, rsn)=%.3f corr(inp,idx)=%.3f corr(lat,inp)=%.3f corr(lat,rsn)=%.3f' % (
        np.corrcoef(inp, rsn)[0, 1], np.corrcoef(inp, idx)[0, 1], np.corrcoef(y, inp)[0, 1], np.corrcoef(y, rsn)[0, 1]))
    fits = {}
    fits['A_rsn_out'] = ols([rsn, out], y, ['rsn', 'out'])
    fits['B_inp_only'] = ols([inp], y, ['inp_k'])
    fits['C_rsn_vis_inp'] = ols([inp, rsn, vis], y, ['inp_k', 'rsn', 'vis'])
    fits['D_rsn_out_inp'] = ols([rsn, out, inp], y, ['rsn', 'out', 'inp_k'])
    fits['E_C_plus_first'] = ols([inp, rsn, vis, first], y, ['inp_k', 'rsn', 'vis', 'first_call'])
    # H1-style subset: the 5 runs used in h1_speed_attempts (T04 t100, T08 t100, stove-off, drawer 001/003)
    sub = np.array([r['run'] in ('H1_T04', 'H1_T08', 'SOFF_T08', 'DRW_001', 'DRW_003') for r in rows])
    fits['F_C_on_h1_5runs'] = ols([inp[sub], rsn[sub], vis[sub]], y[sub], ['inp_k', 'rsn', 'vis'])
    fits['G_C_novision'] = ols([inp, rsn, vis], yv, ['inp_k', 'rsn', 'vis'])
    for k, f in fits.items():
        cs = ', '.join(f"{nm}={b:.4f}±{s:.4f}" for nm, (b, s) in f['coef'].items())
        print(f"{k:18s} n={f['n']} R2={f['r2']:.3f}  {cs}")
    # decomposition of total model time under fit C
    c = fits['C_rsn_vis_inp']['coef']
    tot = y.sum()
    parts = {'fixed': c['const'][0] * len(y), 'input': c['inp_k'][0] * inp.sum(), 'reasoning': c['rsn'][0] * rsn.sum(), 'visible': c['vis'][0] * vis.sum()}
    print('decomp under C (share of sum latency):', {k: round(v / tot * 100, 1) for k, v in parts.items()})
    # per-run means
    for key in UC71:
        rr = [r for r in rows if r['run'] == key]
        print(f"{key:9s} n={len(rr):2d} mean_lat={np.mean([r['lat'] for r in rr]):6.2f} mean_rsn={np.mean([r['rsn'] for r in rr]):6.1f} mean_vis={np.mean([r['vis'] for r in rr]):6.1f} "
              f"inp_first={rr[0]['inp']} inp_last={rr[-1]['inp']} cached_share={sum(r['cached'] for r in rr)/max(1,sum(r['inp'] for r in rr))*100:.1f}%")
    json.dump(fits, open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/v_latency_fits.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
