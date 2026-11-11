"""Power of the proposed primary rule (paired McNemar exact test, alpha=0.05 two-sided) and Wilson half-width."""
import math, random
from math import comb
def wilson_half(p, n, z=1.96):
    return z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
def mcnemar_reject(b, c, alpha=0.05):
    n = b + c
    if n == 0: return False
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(0, k+1)) / 2**n
    return min(1.0, p) < alpha
def power(p1, p2, n_pairs, rho_scene=0.0, n_scenes=14, sims=20000, seed=1):
    rnd = random.Random(seed)
    rej = 0
    reps = n_pairs // n_scenes
    for _ in range(sims):
        b = c = 0
        for s in range(n_scenes):
            # scene difficulty shared by both harnesses (logit shift) to model within-scene correlation
            shift = rnd.gauss(0, rho_scene)
            q1 = 1/(1+math.exp(-(math.log(p1/(1-p1)) + shift)))
            q2 = 1/(1+math.exp(-(math.log(p2/(1-p2)) + shift)))
            for r in range(reps):
                a1 = rnd.random() < q1; a2 = rnd.random() < q2
                if a1 and not a2: b += 1
                elif a2 and not a1: c += 1
        rej += mcnemar_reject(b, c)
    return rej / sims
print('Wilson half-width p=0.7 n=42: %.3f' % wilson_half(0.7, 42))
for (p1, p2) in [(0.7, 0.5), (0.7, 0.45), (0.8, 0.5), (0.9, 0.6)]:
    print(f'p1={p1} p2={p2}: power(n=42, no scene effect)={power(p1,p2,42):.2f}  power(scene sd=1.5 logit)={power(p1,p2,42,1.5):.2f}  power(n=14 one repeat)={power(p1,p2,14):.2f}')
