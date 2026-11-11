"""Robustness of the input-token latency coefficient: run fixed effects, call-index control, steady-window subset,
and the implied cost of history growth per run."""
import json, sys
import numpy as np
rows = [r for r in json.load(open(sys.argv[1])) if r['phase'] == 'trial']
runs = sorted(set(r['run'] for r in rows))
idx = {}
for r in rows:
    idx.setdefault(r['run'], 0); r['k'] = idx[r['run']]; idx[r['run']] += 1
def fit(sel, extra_fe=False, with_k=False):
    rr = [r for r in rows if sel(r)]
    y = np.array([r['latency_s'] for r in rr])
    cols = [np.ones(len(rr)), np.array([r['input'] for r in rr]) / 1000, np.array([r['reasoning'] for r in rr], float),
            np.array([r['visible'] for r in rr], float)]
    if with_k: cols.append(np.array([r['k'] for r in rr], float))
    if extra_fe:
        for run in runs[1:]: cols.append(np.array([1.0 if r['run'] == run else 0.0 for r in rr]))
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y - X @ b
    XtXi = np.linalg.inv(X.T @ X); hc = XtXi @ (X.T * e**2) @ X @ XtXi * len(y) / (len(y) - X.shape[1])
    return {"n": len(rr), "input_coef": round(b[1], 5), "input_se": round(float(np.sqrt(hc[1, 1])), 5), "reason_coef": round(b[2], 5),
            "r2": round(1 - e @ e / ((y - y.mean()) @ (y - y.mean())), 4)}
out = {"all": fit(lambda r: True), "run_FE": fit(lambda r: True, extra_fe=True), "run_FE+call_index": fit(lambda r: True, True, True),
       "calls_k>=6": fit(lambda r: r['k'] >= 6), "calls_k>=6_FE": fit(lambda r: r['k'] >= 6, True)}
c = 0.01794
out["history_growth_cost_s_per_run"] = {run: round(c * sum((r['input'] - rows_first) / 1000 for r in rows if r['run'] == run), 1)
    for run in runs for rows_first in [next(r['input'] for r in rows if r['run'] == run)]}
out["last_call_input_cost_s"] = {run: round(c * [r['input'] for r in rows if r['run'] == run][-1] / 1000, 2) for run in runs}
print(json.dumps(out, indent=1))
