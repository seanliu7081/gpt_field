"""Valid elevation sweep (camera stays above the table, looks at the plate); checks A1 right/away vs local image axes
at the plate pixel and over the tabletop."""
import numpy as np
F, C0 = 618.0386719675123, 255.5
ANCHOR = np.array([0.149, 0.006, 0.44])
def unit(v): return v / np.linalg.norm(v)
def look_at(P, target):
    z = unit(P - target)                       # camera +Z points backward (OpenGL)
    x = unit(np.cross([0, 0, 1.0], z)) if abs(z[2]) < 1 - 1e-12 else np.array([0, 1.0, 0])
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
def ray(R, P, uv, zt):
    d = R @ np.array([(uv[0] - C0) / F, -(uv[1] - C0) / F, -1.0])
    if d[2] >= 0: return None
    return P + (zt - P[2]) / d[2] * d
def project(R, P, x):
    c = R.T @ (x - P); return np.array([C0 + F * c[0] / -c[2], C0 - F * c[1] / -c[2]])
dist = np.linalg.norm(np.array([0.6066, 0.0, 0.96]) - ANCHOR)
for elev in (8, 20, 32, 48, 60, 75, 88):
    e = np.radians(elev)
    P = ANCHOR + dist * np.array([np.cos(e), 0.0, np.sin(e)])
    R = look_at(P, ANCHOR)
    right = unit(R[:2, 0]); away = np.array([-right[1], right[0]])
    uv = project(R, P, ANCHOR); p = ray(R, P, uv, ANCHOR[2])
    tr = unit((ray(R, P, uv + [2, 0], ANCHOR[2]) - p)[:2]); tu = unit((ray(R, P, uv + [0, -2], ANCHOR[2]) - p)[:2])
    ang = lambda a, b: np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
    worst_u = 0
    for u in np.linspace(5, 506, 25):
        for v in np.linspace(5, 506, 25):
            q = ray(R, P, (u, v), ANCHOR[2])
            if q is None or not (-0.45 <= q[0] <= 0.45 and -0.55 <= q[1] <= 0.55): continue
            q2 = ray(R, P, (u, v - 2), ANCHOR[2])
            worst_u = max(worst_u, ang(unit((q2 - q)[:2]), away))
    print(f"elev {elev:2d}: R[2,1]={R[2,1]:+.3f} right={np.round(right,3)} away={np.round(away,3)} "
          f"at plate: right_err={ang(tr,right):.2f} behind_err={ang(tu,away):.2f}; tabletop max behind_err={worst_u:.1f} deg")
