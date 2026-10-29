"""Does 'roll <= 89 deg classifies correctly' hold over the whole image, not only at the plate pixel?
Truth = table direction of a small image-right / image-up step at each table pixel (as in rotation_check.py)."""
import numpy as np
F, C0, SIZE = 618.0386719675123, 255.5, 512
R0 = np.array([[0.0, -0.5288, 0.8488], [1.0, 0.0, 0.0], [0.0, 0.8488, 0.5288]])
P0 = np.array([0.6066, 0.0, 0.96]); Z = 0.44
def rot(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
def ray(R, uv):
    d = R @ np.array([(uv[0] - C0) / F, -(uv[1] - C0) / F, -1.0])
    if d[2] >= 0: return None
    return P0 + (Z - P0[2]) / d[2] * d
def unit(v): return v / np.linalg.norm(v)
for deg in (0, 15, 30, 45, 60, 75, 85, 89):
    R = R0 @ rot([0, 0, 1], deg)
    right = unit(R[:2, 0]); away = np.array([-right[1], right[0]])
    basis = {"right": right, "left": -right, "behind": away, "front": -away}
    total = wrong = 0; worst_r = worst_b = 0
    for u in np.linspace(5, 506, 40):
        for v in np.linspace(5, 506, 40):
            p = ray(R, (u, v))
            if p is None or not (-0.45 <= p[0] <= 0.45 and -0.55 <= p[1] <= 0.55): continue   # LIBERO tabletop extent only
            tr = unit((ray(R, (u + 2, v)) - p)[:2]); tu = unit((ray(R, (u, v - 2)) - p)[:2])
            labels = {"right": tr, "left": -tr, "behind": tu, "front": -tu}
            total += 1
            bad = [k for k, vec in labels.items() if max(basis, key=lambda b: basis[b] @ vec) != k]
            wrong += bool(bad)
            worst_r = max(worst_r, np.degrees(np.arccos(np.clip(tr @ right, -1, 1))))
            worst_b = max(worst_b, np.degrees(np.arccos(np.clip(tu @ away, -1, 1))))
    print(f"roll {deg:2d}: table pixels {total:4d}, pixels with any word misclassified {wrong:4d}, "
          f"max right err {worst_r:5.1f} deg, max behind err {worst_b:5.1f} deg")
