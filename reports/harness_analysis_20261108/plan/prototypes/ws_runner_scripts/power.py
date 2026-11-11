"""Paired McNemar power for an A/B sweep (offline Monte Carlo; numpy + stdlib only)."""
import math, sys
import numpy as np

def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial(n, 0.5)
    s = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * s)

def logit(p): return math.log(p / (1 - p))
def expit(x): return 1 / (1 + np.exp(-x))

def solve_mu(target, sigma, rng):
    # find mu so that E[expit(mu + sigma*z)] = target
    z = rng.standard_normal(200000)
    lo, hi = -10, 10
    for _ in range(60):
        mid = (lo + hi) / 2
        if expit(mid + sigma * z).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

def power(n_scenes, repeats, pa, pb, sigma=1.5, sims=4000, seed=0):
    rng = np.random.default_rng(seed)
    mu_a = solve_mu(pa, sigma, rng)
    mu_b = solve_mu(pb, sigma, rng)
    hits = 0
    for _ in range(sims):
        z = rng.standard_normal(n_scenes)
        qa = expit(mu_a + sigma * z)[:, None]
        qb = expit(mu_b + sigma * z)[:, None]
        a = rng.random((n_scenes, repeats)) < qa
        b = rng.random((n_scenes, repeats)) < qb
        bb = int(np.sum(a & ~b)); cc = int(np.sum(~a & b))
        if mcnemar_exact_p(bb, cc) < 0.05:
            hits += 1
    return hits / sims

if __name__ == "__main__":
    for sigma in (1.0, 2.0):
        print(f"scene heterogeneity sigma (logit) = {sigma}")
        for n in (14, 34):
            for r in (1, 2, 3, 5):
                row = []
                for pa, pb in ((0.5, 0.6), (0.5, 0.7), (0.5, 0.8)):
                    row.append(f"{int(round((pb-pa)*100))}pp:{power(n, r, pa, pb, sigma=sigma, sims=2000):.2f}")
                print(f"  scenes={n:2d} repeats={r} episodes/arm={n*r:3d}  " + "  ".join(row))
