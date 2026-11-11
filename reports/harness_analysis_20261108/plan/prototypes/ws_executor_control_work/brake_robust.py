"""Braking-law robustness: true plant a in [0.53,0.77] (rho 0.25) vs a_brake used by the executor."""
import math
import numpy as np
cap, tol, rho = 0.10, 0.004, 0.255
def run(a_true, a_brake, law, d0=0.23, noise=0.0006, seed=0):
    b_true = rho * (1 - a_true); b_brake = rho * (1 - a_brake)
    rng = np.random.default_rng(seed); x = s = 0.0; arrived = 0; mo = 0.0
    for step in range(1, 300):
        r = d0 - x
        if law == "current":
            c = max(-cap, min(cap, r)) if abs(r) > tol else 0.0
        else:
            e = tol / 2
            tgt = (1 - a_brake) * (r - math.copysign(e, r)) if abs(r) > e else 0.0
            c = max(-cap, min(cap, (tgt - a_brake * s) / b_brake))
        s = a_true * s + b_true * c + noise * rng.normal(); x += s; mo = max(mo, x - d0)
        arrived = arrived + 1 if abs(d0 - x) <= tol else 0
        if arrived >= 2: return step, mo
    return 300, mo
for a_true in (0.53, 0.63, 0.71, 0.77):
    for law, ab in (("current", None), ("braking", 0.71), ("braking", 0.77)):
        res = [run(a_true, ab if ab else 0.7, law, seed=k) for k in range(100)]
        st = np.median([r[0] for r in res]); ov = np.percentile([r[1] for r in res], 90) * 1000
        print(f"a_true={a_true:.2f} {law:8s} a_brake={ab}: median steps {st:.0f}, p90 overshoot {ov:.1f} mm")
