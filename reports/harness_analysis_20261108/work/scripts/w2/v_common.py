import json, glob, base64, io
import numpy as np
RUNS = {"T04": "/workspace/gpt_field/runs/foundation_vision_t04_can_obstacle_20261105_001",
        "T08": "/workspace/gpt_field/runs/foundation_vision_t08_bottle_obstacle_20261105_001"}
FLIP = np.diag([1., -1., -1.])

def requests(run):
    return sorted(glob.glob(run + "/trial/requests/*.request.json"))

def last_user_fields(req):
    inp = req["payload"]["input"]
    users = [m for m in inp if m.get("role") == "user"]
    return json.loads(users[-1]["content"][0]["text"])

def cam_from_geometry(geom, name):
    c = geom["cameras"][name]
    R = np.asarray(c["rotation_world_from_camera"], float)
    p = np.asarray(c["position_world_m"], float)
    rot = FLIP @ R.T
    E = np.eye(4); E[:3, :3] = rot; E[:3, 3] = -rot @ p
    fx, fy = c["focal_length_px"]; cx, cy = c["principal_point_px"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    return K, E, p

def project(K, E, X):
    X = np.atleast_2d(X)
    c = X @ E[:3, :3].T + E[:3, 3]
    uv = (c @ K.T)[:, :2] / c[:, 2:3]
    return uv, c[:, 2]

def ray(K, E, uv):
    d_cam = np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    R = E[:3, :3]
    d = R.T @ d_cam
    o = -R.T @ E[:3, 3]
    return o, d / np.linalg.norm(d)

def closest(o1, d1, o2, d2):
    w = o1 - o2; a = d1 @ d1; b = d1 @ d2; c = d2 @ d2; d = d1 @ w; e = d2 @ w
    den = a * c - b * b
    s = (b * e - c * d) / den; t = (a * e - b * d) / den
    p1 = o1 + s * d1; p2 = o2 + t * d2
    return (p1 + p2) / 2, np.linalg.norm(p1 - p2)

def decode_images(req):
    from PIL import Image
    inp = req["payload"]["input"]
    users = [m for m in inp if m.get("role") == "user"]
    imgs = [c for c in users[-1]["content"] if c["type"] == "input_image"]
    out = []
    for c in imgs:
        b = base64.b64decode(c["image_url"].split(",", 1)[1])
        out.append(np.array(Image.open(io.BytesIO(b)).convert("RGB")))
    return out
