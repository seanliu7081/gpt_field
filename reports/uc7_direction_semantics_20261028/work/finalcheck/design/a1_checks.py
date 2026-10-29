"""A1 check properties (pure numpy; no simulator, no model calls).
1. identity: away . (-R[:2,2]) == R[2,1] / |R[:2,0]|  (so the away-sign check is sign(R[2,1]); no 'view not vertical' pre-test needed)
2. 'roll = asin(R[2,0])' vs true roll atan2(R[2,0], R[2,1]) on the LIBERO agentview
3. behind/front: global away vs local image-up at off-centre table points (zero roll), and vs radial 'farther from camera'
"""
import numpy as np
rng = np.random.default_rng(0)

def rand_rot():
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

worst = 0.0
for _ in range(100000):
    R = rand_rot()
    r = R[:2, 0]; n = np.linalg.norm(r)
    if n < 1e-6: continue
    right = r / n; away = np.array([-right[1], right[0]])
    lhs = away @ (-R[:2, 2]); rhs = R[2, 1] / n
    worst = max(worst, abs(lhs - rhs))
print(f"1. max |away.(-R[:2,2]) - R[2,1]/|R[:2,0]|| over 1e5 random rotations = {worst:.2e}")

def rot(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K

R0 = np.array([[0.0, -0.5288, 0.8488], [1.0, 0.0, 0.0], [0.0, 0.8488, 0.5288]])
print("2. roll label (LIBERO agentview rolled about its own view axis)")
for deg in (0, 15, 30, 60, 89, 91, 120, 180):
    R = R0 @ rot([0, 0, 1], deg)
    asin_label = np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1)))
    true_roll = np.degrees(np.arctan2(R[2, 0], R[2, 1]))
    r = R[:2, 0] / np.linalg.norm(R[:2, 0]); away = np.array([-r[1], r[0]])
    fires = away @ (-R[:2, 2]) < 0
    print(f"   roll {deg:4d}: asin(R[2,0])={asin_label:+6.1f}  atan2(R[2,0],R[2,1])={true_roll:+7.1f}  "
          f"R[2,1]={R[2,1]:+.3f}  away-sign check fires={fires}")

# 3. off-centre anchors, unrolled LIBERO camera
F, C0, SIZE = 618.0386719675123, 255.5, 512
P0 = np.array([0.6066, 0.0, 0.96]); Z = 0.44
def ray(R, uv):
    d = R @ np.array([(uv[0] - C0) / F, -(uv[1] - C0) / F, -1.0]); return P0 + (Z - P0[2]) / d[2] * d
right = R0[:2, 0] / np.linalg.norm(R0[:2, 0]); away = np.array([-right[1], right[0]])
worst_up, worst_rad, worst_right = 0, 0, 0
for u in np.linspace(5, 506, 30):
    for v in np.linspace(5, 506, 30):
        p = ray(R0, (u, v))
        if p[2] != p[2]: continue
        up = ray(R0, (u, v - 2)) - p; up = up[:2] / np.linalg.norm(up[:2])
        rt = ray(R0, (u + 2, v)) - p; rt = rt[:2] / np.linalg.norm(rt[:2])
        rad = p[:2] - P0[:2]; rad = rad / np.linalg.norm(rad)
        ang = lambda a, b: np.degrees(np.arccos(np.clip(a @ b, -1, 1)))
        worst_up = max(worst_up, ang(up, away)); worst_rad = max(worst_rad, ang(rad, away)); worst_right = max(worst_right, ang(rt, right))
print(f"3. zero roll, whole 512x512 image on the table plane: max angle(local image-right, global right)={worst_right:.1f} deg; "
      f"max angle(local image-up, global away)={worst_up:.1f} deg; max angle(radial farther-from-camera, global away)={worst_rad:.1f} deg")
