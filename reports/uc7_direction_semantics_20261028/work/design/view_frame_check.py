"""Offline check of the proposed view-frame math on the recorded calibration (no simulator, no model).
1) view frame from external R; 2) ray-cast grid-cell centres onto a table plane; 3) image-space agreement test."""
import json, sys
import numpy as np
REQ = sys.argv[1]
d = json.load(open(REQ))
txt = d["payload"]["input"][0]["content"][0]["text"]
fields = json.loads(txt[txt.find("{"):])
cams = fields["geometry"]["cameras"]
def view_frame(cam):
    R = np.asarray(cam["rotation_world_from_camera"], float)
    right = R[:2, 0]
    n = np.linalg.norm(right)
    right = right / n
    away = np.array([-right[1], right[0]])          # up x right, horizontal
    fwd = -R[:2, 2]; fwd = fwd / np.linalg.norm(fwd)
    return right, away, float(n), float(away @ fwd)
for label in ("external", "wrist"):
    r, a, n, agree = view_frame(cams[label])
    print(f"{label:8s} right_xy={np.round(r,4)} away_xy={np.round(a,4)} |horiz(R[:,0])|={n:.4f} away.forward={agree:+.3f}")
ext = cams["external"]
R = np.asarray(ext["rotation_world_from_camera"], float); C = np.asarray(ext["position_world_m"], float)
fx, fy = ext["focal_length_px"]; cx, cy = ext["principal_point_px"]
def cell_uv(k, size=512, n=8):
    row, col = (k - 1) // n, (k - 1) % n
    s = size / n
    return col * s + s / 2, row * s + s / 2
def raycast(u, v, z):
    dcam = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
    dw = R @ dcam
    t = (z - C[2]) / dw[2]
    return C + t * dw
Z = 0.437
for k, name in [(55, "region E (relayout plan)"), (61, "plate C (relayout plan)"), (35, "pudding D"), (38, "mug A"),
                (54, "t6_007 E"), (43, "t6_006 E"), (53, "t6_007 plate C"), (60, "t6_006 plate C")]:
    u, v = cell_uv(k)
    p = raycast(u, v, Z)
    print(f"cell {k:2d} ({u:.0f},{v:.0f}) -> world ({p[0]:+.3f},{p[1]:+.3f}) at z={Z}  [{name}]")
# image-space agreement: region cell vs anchor cell
def agreement(region_cell, anchor_cell, relation):
    ue, ve = cell_uv(region_cell); ua, va = cell_uv(anchor_cell)
    du, dv = ue - ua, ve - va
    expected = {"right_of": (1, 0), "left_of": (-1, 0), "behind": (0, -1), "front_of": (0, 1)}[relation]
    score = du * expected[0] + dv * expected[1]
    return du, dv, score
for rc, ac, rel in [(55, 61, "right_of"), (54, 53, "right_of"), (43, 60, "right_of")]:
    du, dv, s = agreement(rc, ac, rel)
    print(f"pointing check region cell {rc} vs anchor cell {ac} for {rel} (external_view): du={du:+.0f} dv={dv:+.0f} projection={s:+.0f} -> {'agrees' if s > 0 else 'disagrees' if s < 0 else 'indeterminate'}")
# legacy world semantics right_of=+x: image direction of world +x at the anchor
pa = raycast(*cell_uv(61), Z)
def project(p):
    q = R.T @ (p - C); return cx + fx * q[0] / -q[2], cy - fy * q[1] / -q[2]
u0, v0 = project(pa); u1, v1 = project(pa + np.array([0.05, 0, 0]))
print(f"legacy right_of=+x at plate: image displacement for +5 cm x = ({u1-u0:+.1f},{v1-v0:+.1f}) px (down = toward camera)")
u1, v1 = project(pa + np.array([0, 0.05, 0]))
print(f"external_view right_of=+y at plate: image displacement for +5 cm y = ({u1-u0:+.1f},{v1-v0:+.1f}) px")
# Threshold-free pointing evidence in the support plane (external camera only):
# image displacement anchor->region mapped through the camera's horizontal axes, projected on the relation direction.
def plane_disp(region_cell, anchor_cell):
    ue, ve = cell_uv(region_cell); ua, va = cell_uv(anchor_cell)
    du, dv = ue - ua, ve - va
    return du * R[:2, 0] - dv * R[:2, 1]     # v grows downward
for rc, ac, label in [(55, 61, "relayout E vs C"), (54, 53, "t6_007 E vs C"), (43, 60, "t6_006 E vs C")]:
    w = plane_disp(rc, ac)
    for name, d in [("new right_of=external-view right (0,1)", np.array([0.0, 1.0])), ("old right_of=world +x", np.array([1.0, 0.0]))]:
        s = float(w @ d)
        verdict = "agrees" if s > 1e-9 else "disagrees" if s < -1e-9 else "indeterminate"
        print(f"{label:16s} {name:40s} score={s:+7.1f} -> {verdict}")
