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
    roll = float(np.degrees(np.arctan2(R[2, 0], R[2, 1])))          # roll about the viewing axis
    print(f"{name:34s} in_image={str(inside):5s} roll={roll:+7.1f}deg  "
          f"right_err={angle(right, true_right):5.1f}deg  behind_err={angle(away, true_up):5.1f}deg  "
          f"words_wrong={wrong if wrong else 'none'} {sign_note}")


print("== original LIBERO agentview"); check("original", R0, P0)
print("== yaw: camera orbits the plate (moved to another side of the table)")
for deg in (30, 90, 180, 250):
    Rz = rot([0, 0, 1], deg); check(f"yaw {deg}", Rz @ R0, ANCHOR + Rz @ (P0 - ANCHOR))
print("== elevation: camera above the table on the +x side, looking at the plate (no roll)")
def look_at(C, A):
    f = (A - C) / np.linalg.norm(A - C)
    X = np.cross(f, [0, 0, 1.0]); X /= np.linalg.norm(X)
    Z = -f; Y = np.cross(Z, X)
    return np.column_stack([X, Y, Z])
D = np.linalg.norm(P0 - ANCHOR)
for e in (8, 20, 35, 49, 65, 80, 88):
    C = ANCHOR + D * np.array([np.cos(np.radians(e)), 0, np.sin(np.radians(e))])
    assert C[2] > ANCHOR[2], "camera must stay above the support plane"
    check(f"elevation {e}", look_at(C, ANCHOR), C)
print("== top-down camera directly above the plate")
Rtd = np.column_stack([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).astype(float); check("top-down", Rtd, ANCHOR + [0, 0, 0.6])
print("== roll: camera turned about its own viewing axis (image appears tilted)")
for deg in (5, 15, 30, 60, 89, 180):
    Rr = R0 @ rot([0, 0, 1], deg)   # rotate camera frame about its own Z
    check(f"roll {deg}", Rr, P0)

print("== roll over the whole visible table (not only the plate pixel): share of anchors with >=1 word wrong")
def words_wrong_at(R, P, a):
    uv, front = project(R, P, a)
    if not (front and 0 <= uv[0] < SIZE and 0 <= uv[1] < SIZE):
        return None
    p = ray_to_table(R, P, uv, a[2])
    tr = unit((ray_to_table(R, P, uv + [3, 0], a[2]) - p)[:2]); tu = unit((ray_to_table(R, P, uv + [0, -3], a[2]) - p)[:2])
    r = unit(R[:2, 0]); w = np.array([-r[1], r[0]])
    labels = {"right": tr, "left": -tr, "behind": tu, "front": -tu}
    basis = {"right": r, "left": -r, "behind": w, "front": -w}
    return any(max(basis, key=lambda b: basis[b] @ v) != k for k, v in labels.items())
xs, ys = np.linspace(-0.45, 0.45, 31), np.linspace(-0.55, 0.55, 37)
for deg in (0, 5, 15, 30, 45, 60):
    Rr = R0 @ rot([0, 0, 1], deg)
    res = [words_wrong_at(Rr, P0, np.array([x, y, ANCHOR[2]])) for x in xs for y in ys]
    res = [v for v in res if v is not None]
    print(f"roll {deg:2d}: {sum(res)}/{len(res)} visible table anchors have >=1 word wrong ({100*sum(res)/len(res):.1f}%)")
