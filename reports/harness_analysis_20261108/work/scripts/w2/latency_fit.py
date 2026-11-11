"""OLS latency fits on the 128 H1 execution calls (raw usage + stream timing), with HC-robust SEs, bootstrap CIs,
partial R^2 for input tokens, and a phase-level fit (pre-visible phase vs input/reasoning)."""
import json, sys
import numpy as np
rows = [r for r in json.load(open(sys.argv[1])) if r['phase'] == 'trial']
y = np.array([r['latency_s'] for r in rows])
inp = np.array([r['input'] for r in rows]) / 1000.0
rea = np.array([r['reasoning'] for r in rows], float)
vis = np.array([r['visible'] for r in rows], float)
hdr = np.array([r['headers_s'] for r in rows])
fv = np.array([r['first_visible_s'] for r in rows])
lv = np.array([r['last_visible_s'] for r in rows])
mb = np.array([r['request_bytes'] for r in rows]) / 1e6
def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + list(X))
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    XtXi = np.linalg.inv(X.T @ X)
    hc = XtXi @ (X.T * e**2) @ X @ XtXi * len(y) / (len(y) - X.shape[1])   # HC1
    return b, np.sqrt(np.diag(hc)), r2
def boot(Xs, y, idx, n=4000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        s = rng.integers(0, len(y), len(y))
        b, _, _ = ols([x[s] for x in Xs], y[s]); out.append(b[idx])
    return np.percentile(out, [2.5, 97.5])
res = {}
for name, Xs in [("input_only", [inp]), ("reason_visible", [rea, vis]), ("input_reason_visible", [inp, rea, vis])]:
    b, se, r2 = ols(Xs, y)
    res[name] = {"coef": b.round(5).tolist(), "hc1_se": se.round(5).tolist(), "r2": round(r2, 4)}
res["input_ci95_boot_full"] = boot([inp, rea, vis], y, 1).round(5).tolist()
res["partial_r2_input"] = round(1 - (1 - res["input_reason_visible"]["r2"]) / (1 - res["reason_visible"]["r2"]), 4)
res["corr_input_reasoning"] = round(float(np.corrcoef(inp, rea)[0, 1]), 3)
res["corr_input_latency"] = round(float(np.corrcoef(inp, y)[0, 1]), 3)
# phases
pre = fv - hdr; stream = lv - fv
for name, yy, Xs in [("headers_vs_input_mb", hdr, [inp, mb]), ("previsible_vs_input_reason", pre, [inp, rea]),
                     ("visible_phase_vs_visible", stream, [vis])]:
    b, se, r2 = ols(Xs, yy)
    res[name] = {"coef": b.round(5).tolist(), "hc1_se": se.round(5).tolist(), "r2": round(r2, 4)}
tot = y.sum()
res["phase_shares"] = {"headers": round(hdr.sum() / tot, 3), "previsible": round(pre.sum() / tot, 3), "visible": round(stream.sum() / tot, 3),
                       "tail": round((y - lv).sum() / tot, 3)}
b = res["input_reason_visible"]["coef"]
res["attribution_s"] = {"fixed": round(b[0] * len(y), 1), "input": round(b[1] * inp.sum(), 1), "reasoning": round(b[2] * rea.sum(), 1),
                        "visible": round(b[3] * vis.sum(), 1), "total": round(tot, 1)}
res["visible_tok_per_s_median"] = round(float(np.median(vis / np.maximum(stream, 1e-6))), 1)
# zero-reasoning calls: does latency rise with input?
z = rea == 0
if z.sum() >= 3:
    bz, sez, r2z = ols([inp[z], vis[z]], y[z]); res["zero_reasoning_calls"] = {"n": int(z.sum()), "coef": bz.round(4).tolist(), "se": sez.round(4).tolist(), "r2": round(r2z, 3)}
print(json.dumps(res, indent=1))
