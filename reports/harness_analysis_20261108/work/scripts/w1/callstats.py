import json, glob, sys, os
import numpy as np
runs = {
 't04_t100': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
 't08_t100': 'agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001',
 't08_stoveoff': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
 'drawer_001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
 'drawer_003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}
base = '/workspace/gpt_field/runs'
allrows = []
out = {}
for key, run in runs.items():
    files = sorted(glob.glob(os.path.join(base, run, 'trial', 'requests', '*.result.json')))
    prev = None; rows = []
    for f in files:
        d = json.load(open(f)); b = d['budget']
        s = json.load(open(f.replace('.result.json', '.stream.json')))
        cur = (b['input_tokens'], b['output_tokens'], b['cached_input_tokens'], b['reasoning_tokens'])
        dl = cur if prev is None else tuple(c - p for c, p in zip(cur, prev)); prev = cur
        inp, outp, cached, reas = dl
        lat = s['elapsed_s']; hdr = s['headers_received_s']; fv = s['first_visible_output_s']; comp = s['completion_s']
        rq = json.load(open(f.replace('.result.json', '.request.json')))
        nimg = sum(1 for it in rq['payload']['input'] for c in (it.get('content') or []) if isinstance(c, dict) and c.get('type') == 'input_image')
        rows.append(dict(inp=inp, out=outp, cached=cached, reas=reas, vis=outp - reas, lat=lat, hdr=hdr, think=fv - hdr, stream=comp - fv, nimg=nimg, req_mb=os.path.getsize(f.replace('.result.json', '.request.json'))/1e6))
    allrows += [dict(r, run=key) for r in rows]
    a = lambda k: np.array([r[k] for r in rows], float)
    out[key] = dict(calls=len(rows), lat_sum=a('lat').sum(), lat_mean=a('lat').mean(), lat_median=float(np.median(a('lat'))), lat_p90=float(np.percentile(a('lat'), 90)), lat_max=a('lat').max(),
        first_call=rows[0]['lat'], inp_total=a('inp').sum(), inp_last=rows[-1]['inp'], cached_total=a('cached').sum(), cached_share=a('cached').sum()/a('inp').sum(),
        reas_mean=a('reas').mean(), vis_mean=a('vis').mean(), hdr_sum=a('hdr').sum(), think_sum=a('think').sum(), stream_sum=a('stream').sum(), imgs_mean=a('nimg').mean(), req_mb_mean=a('req_mb').mean(),
        zero_reasoning_calls=int((a('reas') == 0).sum()))
for k, v in out.items():
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
# regression over all calls
X = np.array([[1, r['inp']/1000, r['reas'], r['vis']] for r in allrows])
y = np.array([r['lat'] for r in allrows])
coef, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
r2 = 1 - ((y - pred)**2).sum() / ((y - y.mean())**2).sum()
print('n', len(y), 'coef [intercept, s/ktok input, s/reasoning tok, s/visible tok]', np.round(coef, 4), 'R2', round(r2, 3))
comp = {'intercept': coef[0]*len(y), 'input': (X[:,1]*coef[1]).sum(), 'reasoning': (X[:,2]*coef[2]).sum(), 'visible': (X[:,3]*coef[3]).sum()}
tot = sum(comp.values())
print('decomposition', {k: (round(v,1), round(v/tot*100,1)) for k, v in comp.items()}, 'sum', round(tot,1), 'measured', round(y.sum(),1))
# phase decomposition from stream timings
hdr = sum(r['hdr'] for r in allrows); think = sum(r['think'] for r in allrows); stream = sum(r['stream'] for r in allrows)
print('stream phases: headers %.1f (%.1f%%), pre-visible %.1f (%.1f%%), visible streaming %.1f (%.1f%%), total %.1f' % (hdr, hdr/y.sum()*100, think, think/y.sum()*100, stream, stream/y.sum()*100, y.sum()))
vis = np.array([r['vis'] for r in allrows]); st = np.array([r['stream'] for r in allrows])
print('visible tokens/s during streaming (median)', round(float(np.median(vis/np.maximum(st,1e-3))),1))
reas = np.array([r['reas'] for r in allrows]); th = np.array([r['think'] for r in allrows])
A = np.c_[np.ones(len(reas)), reas, X[:,1]]
c2 = np.linalg.lstsq(A, th, rcond=None)[0]
print('pre-visible time ~ %.2f + %.4f*reas + %.4f*input_k' % tuple(c2))
json.dump({'per_run': out, 'rows': allrows}, open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/data/callstats.json','w'), indent=1, default=float)
