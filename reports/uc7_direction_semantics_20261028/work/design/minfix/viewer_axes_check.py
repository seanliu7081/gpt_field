"""Offline prototype (no sim, no model): viewer axes from saved calibration, and
old-vs-new half-plane masks for LIBERO t6 (proposed resolver semantics)."""
import json, glob, sys
import numpy as np

RUN = "/workspace/gpt_field/runs/agent_speedup_v2_uc7_relayout_t6_t7_20261028_001/task_6"
ROBOCASA = "/workspace/gpt_field/runs/robocasa_coffee_user_check_001/trial/initial_sensors.json"

def viewer_axes(geometry):
    R = np.asarray(geometry["cameras"]["external"]["rotation_world_from_camera"], float)
    r = R[:2, 0]
    n = float(np.hypot(*r))
    if not (R.shape == (3, 3) and np.isfinite(R).all() and n > 0):
        return None
    r = r / n
    f = np.array([r[1], -r[0]])          # r x z_up = toward the camera (front)
    return r, f

def libero_geometry():
    p = sorted(glob.glob(RUN + "/trial/requests/000001-*.request.json"))[0]
    d = json.load(open(p))
    txt = [c["text"] for c in d["payload"]["input"][0]["content"] if c["type"] == "input_text"][0]
    return json.loads(txt)["geometry"]

for name, geo in (("LIBERO agentview (t6 relayout req 1)", libero_geometry()),
                  ("RoboCasa robot0_agentview_left", json.load(open(ROBOCASA))["geometry"])):
    r, f = viewer_axes(geo)
    cam = np.asarray(geo["cameras"]["external"]["position_world_m"])
    print(f"{name}: right={np.round(r,4).tolist()} front(toward camera)={np.round(f,4).tolist()} cam_xy={np.round(cam[:2],3).tolist()}")
    print("   old constant: right_of=+x left_of=-x behind=+y front_of=-y")
    for word, vec in (("right_of", r), ("left_of", -r), ("front_of", f), ("behind", -f)):
        ang = np.degrees(np.arctan2(vec[1], vec[0]))
        print(f"   new {word:9s} -> world xy {np.round(vec,3).tolist()}  (heading {ang:+.1f} deg)")

# old vs new masks on a synthetic 5 mm grid around the t6 plate (radius 0.069 m at (0.149,0.006))
r, f = viewer_axes(libero_geometry())
cell = 0.005
xs, ys = np.meshgrid(np.arange(-0.4, 0.4, cell) + cell/2, np.arange(-0.4, 0.4, cell) + cell/2)  # half-cell offset: no exact ties
th = np.linspace(0, 2*np.pi, 64, endpoint=False)
poly = np.c_[0.149 + 0.069*np.cos(th), 0.006 + 0.069*np.sin(th)]
low, high = poly.min(0), poly.max(0)
old = {"right_of": xs > high[0], "left_of": xs < low[0], "behind": ys > high[1], "front_of": ys < low[1]}
def new(word):
    axis = r if word in ("right_of", "left_of") else f
    sign = 1.0 if word in ("right_of", "front_of") else -1.0
    s = sign * (xs*axis[0] + ys*axis[1])
    return s > (sign * (poly @ axis)).max()
perm = {"right_of": "behind", "left_of": "front_of", "front_of": "right_of", "behind": "left_of"}
for word in ("right_of", "left_of", "front_of", "behind"):
    m = new(word)
    diff = int((m != old[perm[word]]).sum())
    cen = (xs[m].mean(), ys[m].mean())
    print(f"new {word:9s} == old {perm[word]:9s}? differing cells={diff}  half-plane centre ~({cen[0]:+.3f},{cen[1]:+.3f}) on +/-0.4 m synthetic grid")
# pudding start (official) and goal box membership under new right_of
for label, p in (("official pudding start", (-0.0397, 0.1170)), ("relayout release", (0.2897, -0.0201)),
                 ("t6_007 final", (0.145, 0.146)), ("LIBERO goal box centre", (0.15, 0.10))):
    s = p[0]*r[0] + p[1]*r[1]
    print(f"{label:24s} {p}: in new right_of half-plane: {bool(s > (poly @ r).max())}; in LIBERO box: {0.10<=p[0]<=0.20 and 0.05<=p[1]<=0.15}")
