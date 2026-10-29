"""Project relayout-t6 settled object positions into the planner's 8x8 external grid.

Read-only: reads run artifacts, writes only into this folder.
"""
import json, sys, base64, io
import numpy as np

RUN = "/workspace/gpt_field/runs/agent_speedup_v2_uc7_relayout_t6_t7_20261028_001/task_6"
OUT = "/tmp/claude-0/-workspace/10fe2541-a89e-49c2-8e8c-939d1a05ad07/scratchpad/dirsem/finalcheck/refs"

req = json.load(open(RUN + "/trial/requests/000001-8f98c6e98dac4dd189fb1e11d6126df8.request.json"))
geo = json.loads(req["payload"]["input"][0]["content"][0]["text"])["geometry"]
cam = geo["cameras"]["external"]
R = np.array(cam["rotation_world_from_camera"])
P = np.array(cam["position_world_m"])
fx, fy = cam["focal_length_px"]
cx, cy = cam["principal_point_px"]
print("R[:,0] (image-right in world) =", np.round(R[:, 0], 6))
print("R[:,1] (image-up in world)    =", np.round(R[:, 1], 6))
print("-R[:,2] (forward)             =", np.round(-R[:, 2], 6))
right = R[:2, 0] / np.linalg.norm(R[:2, 0])
away = np.array([-right[1], right[0]])
print("right_xy", np.round(right, 6), "away_xy", np.round(away, 6),
      "away.(-R[:2,2])", float(away @ (-R[:2, 2])), "roll_deg(asin R[2,0])", float(np.degrees(np.arcsin(R[2, 0]))))


def project(p):
    pc = R.T @ (np.asarray(p, float) - P)
    X, Y, Z = pc
    u = cx + fx * X / (-Z)
    v = cy - fy * Y / (-Z)
    return u, v


def cell(u, v, width=512, n=8):
    c = width // n
    col, row = int(u // c), int(v // c)
    if not (0 <= col < n and 0 <= row < n):
        return None
    return row * n + col + 1


def ray_cell_centre(cell_id, z=0.44, width=512, n=8):
    row, col = divmod(cell_id - 1, n)
    c = width / n
    u, v = (col + 0.5) * c, (row + 0.5) * c
    X = (u - cx) / fx
    Y = -(v - cy) / fy
    d = R @ np.array([X, Y, -1.0])
    t = (z - P[2]) / d[2]
    return P + t * d


pv = json.load(open(RUN + "/physical_validation.json"))
settled = {o["object"]: np.array(o["settled_xyz_m"]) for o in pv["objects"]}
plan = json.load(open(RUN + "/trial/approved_plan.json"))
pointed = {p["id"]: (p["text"], p["x"], p["w"]) for p in plan["parts"]}
print("planner x cells:", {k: v[1] for k, v in pointed.items()})
rows = {}
for name, xyz in settled.items():
    out = []
    for dz in (-0.005, 0.0, 0.02, 0.04, 0.06):
        u, v = project(xyz + np.array([0, 0, dz]))
        out.append((round(dz, 3), round(u, 1), round(v, 1), cell(u, v)))
    rows[name] = out
    print(name, np.round(xyz, 4), out)
for cid in (35, 38, 39, 55, 54, 61):
    print("cell", cid, "centre ray-cast to z=0.44 ->", np.round(ray_cell_centre(cid), 3))
json.dump({"right_xy": right.tolist(), "away_xy": away.tolist(), "projections": rows,
           "pointed": pointed}, open(OUT + "/project_cells.json", "w"), indent=1)

# Draw the projected points on the displayed (upright, de-mirrored) image for a visual check.
try:
    from PIL import Image, ImageDraw
    obs = json.load(open(RUN + "/review/initial_observation.json"))
    url = obs["observation/image"]
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
    shown = np.ascontiguousarray(img[:, ::-1])  # contact_skills.py:283
    frame = Image.fromarray(shown)
    d = ImageDraw.Draw(frame)
    for k in range(1, 8):
        d.line([(k * 64, 0), (k * 64, 511)], fill=(255, 255, 0))
        d.line([(0, k * 64), (511, k * 64)], fill=(255, 255, 0))
    for i in range(64):
        r, c = divmod(i, 8)
        d.text((c * 64 + 2, r * 64 + 2), str(i + 1), fill=(255, 255, 0))
    colors = {"porcelain_mug_1": (255, 0, 0), "chocolate_pudding_1": (0, 255, 0), "plate_1": (0, 0, 255),
              "red_coffee_mug_1": (255, 0, 255)}
    for name, xyz in settled.items():
        u, v = project(xyz + np.array([0, 0, 0.02]))
        d.ellipse([u - 5, v - 5, u + 5, v + 5], outline=colors[name], width=3)
        d.text((u + 6, v), name.split("_")[0], fill=colors[name])
    frame.save(OUT + "/relayout_t6_projected.png")
    # same for the raw-mirrored variant (double flip) to show what a mismatch would look like
    print("image shape", shown.shape)
except Exception as exc:  # pragma: no cover
    print("image overlay failed:", exc)
