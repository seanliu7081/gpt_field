import json, sys, os, math
import numpy as np
S='/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis'
info=json.load(open(f'{S}/runinfo.json'))
def rows(fn):
    return [json.loads(l) for l in open(f'{S}/{fn}')]
uc=rows('uc_rows.jsonl'); v3=rows('v3_rows.jsonl')
def ver(r):
    return info.get(os.path.basename(r['run']),{}).get('version')
def lat(r):
    return r.get('trace_latency') if r.get('trace_latency') is not None else r.get('client_interval_s')
def prep(rs):
    out=[]
    for r in rs:
        if r.get('res_status')!='valid' or r.get('input_tokens') is None: continue
        r=dict(r)
        r['lat']=lat(r)
        r['visible']=(r['output_tokens'] or 0)-(r['reasoning_tokens'] or 0)
        r['uncached']=(r['input_tokens'] or 0)-(r['cached_tokens'] or 0)
        out.append(r)
    return out
def corr(x,y):
    x=np.array(x,float); y=np.array(y,float)
    if x.std()==0 or y.std()==0: return float('nan')
    return float(np.corrcoef(x,y)[0,1])
def spearman(x,y):
    import numpy as np
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
    return corr(rx,ry)
def ols(rs, feats, ykey='lat'):
    X=np.array([[1.0]+[float(r[f]) for f in feats] for r in rs])
    y=np.array([float(r[ykey]) for r in rs])
    beta,res,rank,sv=np.linalg.lstsq(X,y,rcond=None)
    pred=X@beta; ss_res=((y-pred)**2).sum(); ss_tot=((y-y.mean())**2).sum()
    n,k=X.shape
    sigma2=ss_res/(n-k) if n>k else float('nan')
    try:
        cov=sigma2*np.linalg.inv(X.T@X); se=np.sqrt(np.diag(cov))
    except Exception:
        se=[float('nan')]*k
    return dict(zip(['intercept']+feats,[ (round(float(b),6), round(float(s),6)) for b,s in zip(beta,se)])), round(1-ss_res/ss_tot,3), n
def describe(name, rs):
    L=np.array([r['lat'] for r in rs])
    print(f'### {name}: n={len(rs)} runs={len(set(r["run"] for r in rs))}')
    print(f'  latency s: mean {L.mean():.2f} median {np.median(L):.2f} p90 {np.percentile(L,90):.2f} max {L.max():.2f} min {L.min():.2f} sum {L.sum():.1f}')
    feats=['input_tokens','uncached','cached_tokens','wire_bytes','text_chars','n_img','img_b64_chars','output_tokens','reasoning_tokens','visible','out_text_chars','ordinal']
    for f in feats:
        vals=[r[f] for r in rs]
        a=np.array(vals,float)
        print(f'  {f:16s} mean {a.mean():12.1f} median {np.median(a):12.1f} min {a.min():10.0f} max {a.max():12.0f}  r={corr(vals,L):+.3f} rho={spearman(vals,L):+.3f}')
    for fs in (['input_tokens'],['reasoning_tokens'],['visible'],['output_tokens'],['input_tokens','reasoning_tokens','visible'],['n_img','text_chars','reasoning_tokens','visible']):
        rs2=[dict(r, input_tokens=r['input_tokens']/1000.0, text_chars=r['text_chars']/1000.0) for r in rs]
        b,r2,n=ols(rs2,fs)
        print('  OLS', fs, 'R2', r2, b)
    # server timestamps
    srv=[r['server_s'] for r in rs if r.get('server_s') is not None]
    if srv:
        d=[r['lat']-r['server_s'] for r in rs if r.get('server_s') is not None]
        print(f'  server created->completed s: mean {np.mean(srv):.2f} median {np.median(srv):.2f}; lat - server: mean {np.mean(d):.2f} median {np.median(d):.2f}')
    ci=[(r['lat'], r['client_interval_s']) for r in rs if r.get('client_interval_s') is not None]
    if ci:
        dd=[a-b for a,b in ci]
        print(f'  trace_latency - client_interval: mean {np.mean(dd):.3f} median {np.median(dd):.3f} max {np.max(dd):.3f}')
uc7=[r for r in prep(uc) if ver(r)=='agent-speedup-v2-user-check-7' and r.get('kind')=='agent']
uc6=[r for r in prep(uc) if ver(r)=='agent-speedup-v2-user-check-6' and r.get('kind')=='agent']
v3e=[r for r in prep(v3) if r.get('kind')=='agent']
describe('uc7 execution (agent) calls', uc7)
describe('uc6 execution (agent) calls', uc6)
describe('v3 execution (agent) calls, all revisions', v3e)
v3late=[r for r in v3e if (ver(r) or '') not in ('agent-speedup-v3-1','agent-speedup-v3-2','agent-speedup-v3-3')]
describe('v3 execution calls r4+', v3late)
json.dump(dict(uc7=uc7,uc6=uc6,v3=v3e), open(f'{S}/exec_calls.json','w'))
