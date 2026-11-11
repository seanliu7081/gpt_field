"""Per-run contact-free translation plant fits (a, b, rho) with bootstrap CI of pooled a."""
import pickle
import numpy as np
BASE = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/"
FILES = {"T04_t100": 0.10, "T08_t100": 0.10, "T08_stoveoff": 0.10, "drawer001": 0.10, "drawer003": 0.10,
         "T04_20mm": 0.05, "T04_default": 0.05, "H3_T04": 0.10, "H3_T08": 0.10}
allX, allY, groups = [], [], []
for gi, (name, scale) in enumerate(FILES.items()):
    d = pickle.load(open(BASE + name + "_steps.pkl", "rb")); ks = sorted(d); rows = []
    for k in ks:
        x = d[k]; b, a = np.asarray(x["before_state"]), np.asarray(x["after_state"]); act = np.asarray(x["norm_action"])
        rows.append(dict(c=act[:3] * scale, cr=act[3:6] * 0.5, s=a[:3] - b[:3], f=(x.get("forb_inc", 0) or 0) + sum((x.get("inc_allowed") or {}).values())))
    X, Y = [], []
    for i in range(1, len(rows)):
        if rows[i]["f"] or rows[i - 1]["f"]: continue
        for k in range(3):
            X.append([rows[i - 1]["s"][k], rows[i]["c"][k]]); Y.append(rows[i]["s"][k])
    X, Y = np.asarray(X), np.asarray(Y)
    (a_, b_), *_ = np.linalg.lstsq(X, Y, rcond=None)
    print(f"{name:12s} contact-free a={a_:.3f} b={b_:.4f} rho={b_/(1-a_):.3f} n={len(Y)//3}")
    allX.append(X); allY.append(Y)
rng = np.random.default_rng(1); est = []
for _ in range(300):
    idx = rng.integers(0, len(allX), len(allX))
    X = np.vstack([allX[i] for i in idx]); Y = np.concatenate([allY[i] for i in idx])
    (a_, b_), *_ = np.linalg.lstsq(X, Y, rcond=None); est.append(a_)
print(f"run-bootstrap pooled a: median {np.median(est):.3f} 95% [{np.percentile(est,2.5):.3f}, {np.percentile(est,97.5):.3f}]")
