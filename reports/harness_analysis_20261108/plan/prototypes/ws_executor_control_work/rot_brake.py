"""Rotation plant (a_r=0.90, b_r=0.027): current capped-proportional law vs braking law; 1.57 rad turn."""
import math
import numpy as np
cap = 0.1
def run(theta0, law, a=0.90, b=0.0271, tol=0.06, a_brake=0.90, noise=0.002, seed=0, exit_tol=None):
    rng = np.random.default_rng(seed); x = s = 0.0; arrived = 0; mo = 0.0
    exit_tol = exit_tol or tol
    for step in range(1, 400):
        r = theta0 - x
        if law == "current":
            c = math.copysign(min(abs(r), cap), r)
        else:
            e = exit_tol / 2
            tgt = (1 - a_brake) * (r - math.copysign(e, r)) if abs(r) > e else 0.0
            c = max(-cap, min(cap, (tgt - a_brake * s) / (b * (1 - a_brake) / (1 - a) if False else b)))
        s = a * s + b * c + noise * rng.normal(); x += s; mo = max(mo, x - theta0)
        arrived = arrived + 1 if abs(theta0 - x) <= exit_tol else 0
        if arrived >= 2: return step, mo
    return 400, mo
for theta in (0.3, 0.785, 1.57):
    for law in ("current", "braking"):
        for et in (0.06, 0.02):
            res = [run(theta, law, seed=k, exit_tol=et) for k in range(100)]
            print(f"theta={theta:.2f} {law:8s} exit_tol={et}: median steps {np.median([r[0] for r in res]):.0f}, p90 overshoot {np.percentile([r[1] for r in res],90):.3f} rad")
