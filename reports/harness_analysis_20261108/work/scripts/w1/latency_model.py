import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from req_stats2 import per_call

runs = sys.argv[1:]
allrows = []
for run in runs:
    rows = per_call(run)
    for r in rows:
        r['run'] = run
    allrows += rows
    a = np.array([[r['in_tok'], r['reas'], r['out'], r['ttfv'], r['done'], r['hdr']] for r in rows], float)
    # time per reasoning token approximations
    gen = a[:, 4] - a[:, 3]
    vis_out = a[:, 2] - a[:, 1]
    print(f"{run[:70]:70s} n={len(rows):2d} in/call={a[:,0].mean():7.0f} reas/call={a[:,1].mean():5.0f} out/call={a[:,2].mean():5.0f} ttfv/call={a[:,3].mean():5.1f} done/call={a[:,4].mean():5.1f} gen_s/call={gen.mean():4.1f} vis_tok/call={vis_out.mean():4.0f} vis_tok_per_gen_s={vis_out.sum()/gen.sum():5.1f}")

X = np.array([[1.0, r['in_tok']/1000, r['reas']] for r in allrows])
y = np.array([r['ttfv'] for r in allrows])
coef, res, rk, sv = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
print('ttfv ~ a + b*in_ktok + c*reas_tok :', [round(c, 4) for c in coef], 'R2=', round(1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum(), 3))
# per-run residual mean
for run in runs:
    idx = [i for i, r in enumerate(allrows) if r['run'] == run]
    print(f"  resid mean {run[:60]:60s} {np.mean(y[idx]-pred[idx]):6.2f}  n={len(idx)}")
# add H3 indicator
ind = np.array([1.0 if 'foundation_vision' in r['run'] else 0.0 for r in allrows])
X2 = np.column_stack([X, ind])
coef2, *_ = np.linalg.lstsq(X2, y, rcond=None)
pred2 = X2 @ coef2
print('with H3 indicator:', [round(c, 4) for c in coef2], 'R2=', round(1 - ((y-pred2)**2).sum()/((y-y.mean())**2).sum(), 3))
# reasoning tokens ~ first call / call index
