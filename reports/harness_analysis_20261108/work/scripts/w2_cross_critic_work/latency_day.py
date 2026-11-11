"""Day/throughput confound check and multivariate latency fits on all calls.

Reads data/calls_all.json (from calls_all.py). numpy only.
"""
import json, os, statistics as stt
import numpy as np

D = os.path.join(os.path.dirname(__file__), "..", "data")
rows = json.load(open(os.path.join(D, "calls_all.json")))

def med(xs):
    xs = [x for x in xs if x is not None]
    return round(stt.median(xs), 2) if xs else None

def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None

print("== per run summary (exec + plan) ==")
runs = []
for r in rows:
    if r["run"] not in runs:
        runs.append(r["run"])
out = {"per_run": {}, "fits": {}}
for run in runs:
    for kind in ("plan", "exec"):
        rs = [r for r in rows if r["run"] == run and r["kind"] == kind]
        if not rs:
            continue
        lat = [r.get("elapsed_s") or r.get("server_s") for r in rs]
        rec = {
            "n": len(rs),
            "utc_first": min(r["utc"] for r in rs if r.get("utc")),
            "utc_last": max(r["utc"] for r in rs if r.get("utc")),
            "lat_mean": mean(lat), "lat_med": med(lat),
            "server_s_mean": mean([r.get("server_s") for r in rs]),
            "reason_mean": mean([r.get("reasoning_tokens") for r in rs]),
            "visible_mean": mean([r.get("visible_tokens") for r in rs]),
            "input_k_mean": mean([(r.get("input_tokens") or 0) / 1000 for r in rs]),
            "ttfv_med": med([r.get("first_visible_s") for r in rs]),
            "headers_med": med([r.get("headers_s") for r in rs]),
            "vis_rate_med": med([r.get("visible_rate") for r in rs]),
            "cached_share": round(sum(r.get("cached_tokens") or 0 for r in rs) / max(1, sum(r.get("input_tokens") or 0 for r in rs)), 4),
            "service_tier": sorted(set(str(r.get("service_tier")) for r in rs)),
            "effort": sorted(set(str(r.get("effort")) for r in rs)),
        }
        # rate of "pre-visible" phase: (first_visible - headers) per reasoning token for calls with reasoning>50
        pv = [((r["first_visible_s"] - r["headers_s"]) / r["reasoning_tokens"]) for r in rs
              if r.get("first_visible_s") and r.get("headers_s") and (r.get("reasoning_tokens") or 0) >= 50]
        rec["pre_visible_s_per_reason_tok_med"] = med(pv)
        out["per_run"][f"{run}/{kind}"] = rec
        print(f"{run:18s} {kind:4s} n={rec['n']:3d} {rec['utc_first'][:16]}..{rec['utc_last'][11:16]} "
              f"lat {rec['lat_mean']}/{rec['lat_med']} reas {rec['reason_mean']} vis {rec['visible_mean']} "
              f"in_k {rec['input_k_mean']} ttfv {rec['ttfv_med']} hdr {rec['headers_med']} "
              f"vrate {rec['vis_rate_med']} pv/rt {rec['pre_visible_s_per_reason_tok_med']} cache {rec['cached_share']} tier {rec['service_tier']}")

# Regressions --------------------------------------------------------------
def fit(rs, cols, ycol="elapsed_s", name=""):
    X, y = [], []
    for r in rs:
        yy = r.get(ycol)
        if yy is None:
            continue
        xs = []
        ok = True
        for c in cols:
            if callable(c):
                v = c(r)
            else:
                v = r.get(c)
            if v is None:
                ok = False
                break
            xs.append(float(v))
        if ok:
            X.append([1.0] + xs)
            y.append(float(yy))
    X = np.array(X); y = np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # standard errors
    n, p = X.shape
    s2 = (res ** 2).sum() / max(1, n - p)
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"name": name, "n": int(n), "r2": round(float(r2), 3),
            "coef": [round(float(b), 5) for b in beta], "se": [round(float(s), 5) for s in se]}

uc71 = [r for r in rows if r["run"].startswith(("H1_", "T20_", "H3_")) and r["kind"] == "exec"]
h1day = [r for r in uc71 if not r["run"].startswith("H3_")]
inputk = lambda r: (r.get("input_tokens") or 0) / 1000
d1105 = lambda r: 1.0 if r.get("utc_date") == "2026-11-05" else 0.0
fits = [
    fit(uc71, [inputk], name="uc71+H3 exec: lat ~ input_k (single var)"),
    fit(uc71, ["reasoning_tokens", "visible_tokens"], name="uc71+H3 exec: lat ~ reasoning + visible"),
    fit(uc71, ["reasoning_tokens", "visible_tokens", inputk], name="uc71+H3 exec: lat ~ reasoning + visible + input_k"),
    fit(uc71, ["reasoning_tokens", "visible_tokens", inputk, d1105], name="uc71+H3 exec: + day(11-05) dummy"),
    fit(h1day, ["reasoning_tokens", "visible_tokens", inputk], name="uc71 only (11-02/03): lat ~ reasoning + visible + input_k"),
    fit(h1day, [inputk], name="uc71 only: lat ~ input_k"),
]
# correlation of input_k with reasoning across uc71 calls (confound check)
ik = np.array([inputk(r) for r in uc71]); rt = np.array([r["reasoning_tokens"] for r in uc71])
idx = np.array([r["idx"] for r in uc71])
print("\ncorr(input_k, reasoning) uc71+H3 exec:", round(float(np.corrcoef(ik, rt)[0, 1]), 3))
print("corr(call index, reasoning):", round(float(np.corrcoef(idx, rt)[0, 1]), 3))
# first call vs later calls
first = [r for r in uc71 if r["idx"] == 1]
later = [r for r in uc71 if r["idx"] > 1]
print("first-call mean reasoning", mean([r["reasoning_tokens"] for r in first]), "later", mean([r["reasoning_tokens"] for r in later]))

# ttfv regression (pre-visible phase) with day dummy
ttfv = [r for r in uc71 if r.get("first_visible_s")]
fits.append(fit(ttfv, ["reasoning_tokens", inputk], ycol="first_visible_s", name="TTFV ~ reasoning + input_k (uc71+H3)"))
fits.append(fit(ttfv, ["reasoning_tokens", inputk, d1105], ycol="first_visible_s", name="TTFV ~ reasoning + input_k + day(11-05)"))

# astranomy: per-day floor (tiny calls) and fit
ast = [r for r in rows if r["run"].startswith("AST_")]
for day in sorted(set(r["utc_date"] for r in ast)):
    rs = [r for r in ast if r["utc_date"] == day]
    tiny = [r for r in rs if (r.get("reasoning_tokens") or 0) == 0 and (r.get("output_tokens") or 0) < 80]
    print(f"AST {day}: n={len(rs)} lat mean {mean([r['elapsed_s'] for r in rs])} med {med([r['elapsed_s'] for r in rs])}; "
          f"tiny(0 reasoning,<80 out) n={len(tiny)} lat med {med([r['elapsed_s'] for r in tiny])} server_s med {med([r['server_s'] for r in tiny])}; "
          f"reas mean {mean([r['reasoning_tokens'] for r in rs])}; in_k mean {mean([inputk(r) for r in rs])}")
fits.append(fit(ast, ["reasoning_tokens", "visible_tokens", inputk], name="astranomy: lat ~ reasoning + visible + input_k"))
fits.append(fit(ast, ["reasoning_tokens", "visible_tokens", inputk, d1105], name="astranomy: + day(11-05)"))

# H2
h2 = [r for r in rows if r["run"] == "H2_T04"]
print("H2 utc", h2[0]["utc"], h2[-1]["utc"])
fits.append(fit(h2, ["output_tokens"], name="H2: lat ~ output_tokens"))
fits.append(fit(h2, ["output_tokens", inputk], name="H2: lat ~ output + input_k"))

for f_ in fits:
    print(f_)
out["fits"] = fits

# plan calls: identical requests compare
print("\n== plan calls ==")
for r in rows:
    if r["kind"] == "plan":
        print(r["run"], r["utc"][:16], "lat", r.get("elapsed_s"), "reas", r["reasoning_tokens"], "vis", r.get("visible_tokens"),
              "ttfv", r.get("first_visible_s"), "hdr", r.get("headers_s"), "vrate", round(r.get("visible_rate") or 0, 1), "in", r["input_tokens"])

json.dump(out, open(os.path.join(D, "latency_day.json"), "w"), indent=1)
