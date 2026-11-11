"""Rotation: capped-proportional with a plant-derived active-brake switch; measure post-arrival overshoot."""
import math
import numpy as np
cap = 0.1
def law_cmd(r, s, a, b, e, kind):
    if kind == "current":
        return math.copysign(min(abs(r), cap), r)
    # switch to braking when the zero-command coast a*s/(1-a) would pass the target band
    coast = a * s / (1 - a)
    if (r > 0 and coast >= r - e) or (r < 0 and coast <= r + e):
        tgt = (1 - a) * (r - math.copysign(e, r))
        return max(-cap, min(cap, (tgt - a * s) / b))
    return math.copysign(min(abs(r), cap), r)
def run(theta0, kind, a=0.90, b=0.0271, tol=0.06, noise=0.002, seed=0):
    rng = np.random.default_rng(seed); x = s = 0.0; arrived = 0; exit_step = None; post = 0.0
    for step in range(1, 400):
        r = theta0 - x
        c = law_cmd(r, s, a, b, tol / 2, kind)
        s = a * s + b * c + noise * rng.normal(); x += s
        if exit_step is None:
            arrived = arrived + 1 if abs(theta0 - x) <= tol else 0
            if arrived >= 2: exit_step = step
        else:
            post = max(post, abs(theta0 - x))
            if step - exit_step > 40: break
    return exit_step, post
for theta in (0.3, 0.785, 1.57):
    for kind in ("current", "switch"):
        res = [run(theta, kind, seed=k) for k in range(100)]
        print(f"theta={theta:.2f} {kind:8s}: median exit step {np.median([r[0] for r in res]):.0f}, p90 post-exit max error {np.percentile([r[1] for r in res],90):.3f} rad")
