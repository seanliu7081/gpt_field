"""Does the calibration-derived direction frame survive camera rotations?

Camera model = the request's convention: columns of R are camera X (image-right),
Y (image-up), Z (backward) in world; u = cx + f*X/(-Z), v = cy - f*Y/(-Z).
Global frame (S1): right = unit(R[:2,0]); away = up x right = (-right_y, right_x),
sign-checked against the horizontal view direction -R[:2,2].
Truth: what "right/left/behind/front of the plate" means AS SEEN IN THE IMAGE at the
plate's pixel, i.e. the table direction of a small image-right / image-up step there.
"""
import numpy as np

F, C0, SIZE = 618.0386719675123, 255.5, 512
R0 = np.array([[0.0, -0.5288, 0.8488], [1.0, 0.0, 0.0], [0.0, 0.8488, 0.5288]])  # LIBERO living-room agentview
P0 = np.array([0.6066, 0.0, 0.96])
ANCHOR = np.array([0.149, 0.006, 0.44])   # plate centre on the table plane


def rot(axis, deg):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a); t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def project(R, P, x):
    c = R.T @ (x - P)
    return np.array([C0 + F * c[0] / -c[2], C0 - F * c[1] / -c[2]]), c[2] < 0


def ray_to_table(R, P, uv, z):
    d = R @ np.array([(uv[0] - C0) / F, -(uv[1] - C0) / F, -1.0])
    return P + (z - P[2]) / d[2] * d


def unit(v):
    n = np.linalg.norm(v)
    return None if n < 1e-9 else v / n


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))


def check(name, R, P):
    uv, front = project(R, P, ANCHOR)
    inside = front and 0 <= uv[0] < SIZE and 0 <= uv[1] < SIZE
    p = ray_to_table(R, P, uv, ANCHOR[2])
    true_right = unit((ray_to_table(R, P, uv + [3, 0], ANCHOR[2]) - p)[:2])
    true_up = unit((ray_to_table(R, P, uv + [0, -3], ANCHOR[2]) - p)[:2])      # image-up = "behind"
    right = unit(R[:2, 0])
    if right is None:
        print(f"{name:34s} global frame UNAVAILABLE (camera X is vertical)"); return
    away = np.array([-right[1], right[0]])
    view_h = unit(-R[:2, 2])
    sign_note = ""
    if view_h is not None and away @ view_h < 0:
        sign_note = "away points TOWARD camera (sign check fires)"
    # classify the four true image directions with the global frame
    labels = {"right": true_right, "left": -true_right, "behind": true_up, "front": -true_up}
    basis = {"right": right, "left": -right, "behind": away, "front": -away}
    wrong = [k for k, v in labels.items() if max(basis, key=lambda b: basis[b] @ v) != k]
    roll = float(np.degrees(np.arcsin(np.clip(R[2, 0], -1, 1))))
    print(f"{name:34s} in_image={str(inside):5s} cam_x_tilt={roll:+6.1f}deg  "
          f"right_err={angle(right, true_right):5.1f}deg  behind_err={angle(away, true_up):5.1f}deg  "
          f"words_wrong={wrong if wrong else 'none'} {sign_note}")


print("== original LIBERO agentview"); check("original", R0, P0)
print("== yaw: camera orbits the plate (moved to another side of the table)")
for deg in (30, 90, 180, 250):
    Rz = rot([0, 0, 1], deg); check(f"yaw {deg}", Rz @ R0, ANCHOR + Rz @ (P0 - ANCHOR))
print("== pitch: camera raised/lowered around the plate (more top-down / more horizontal)")
for deg in (-15, 20, 40, 57):
    Rx = rot(R0[:, 0], deg); check(f"pitch {deg:+d} (orbit about camera X)", Rx @ R0, ANCHOR + Rx @ (P0 - ANCHOR))
print("== top-down camera directly above the plate")
Rtd = np.column_stack([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).astype(float); check("top-down", Rtd, ANCHOR + [0, 0, 0.6])
print("== roll: camera turned about its own viewing axis (image appears tilted)")
for deg in (5, 15, 30, 60, 89, 180):
    Rr = R0 @ rot([0, 0, 1], deg)   # rotate camera frame about its own Z
    check(f"roll {deg}", Rr, P0)
