"""Re-derive the plan-time wrist quarter turns that frame_alignment_turns would return if given the exact external
calibration (adapter exposing image_right_world_xy = unit(R_ext[:2,0])) instead of None. Mirrors geometry.py:1525-1540
for a top-down hand (no import of project code)."""
import json, sys
import numpy as np
W = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])   # hand_from_camera (astranomy_agentic_geometry.py:27)
for path in sys.argv[1:]:
    d = json.load(open(path))
    txt = d["payload"]["input"][0]["content"][0]["text"]
    cams = json.loads(txt[txt.find("{"):])["geometry"]["cameras"]
    Re = np.asarray(cams["external"]["rotation_world_from_camera"])
    Rw = np.asarray(cams["wrist"]["rotation_world_from_camera"])
    Rh = Rw @ W                      # R_cam = R_hand @ W  ->  R_hand = R_cam @ W (W symmetric, W@W = I)
    right = Re[:2, 0] / np.linalg.norm(Re[:2, 0])
    cands = [Rh[:2, 1], -Rh[:2, 0], -Rh[:2, 1], Rh[:2, 0]]
    scores = [float(c @ right) for c in cands]
    print(path.split("/runs/")[1][:70], "k_with_calibration =", int(np.argmax(scores)), "scores", np.round(scores, 3),
          "| hand approach z", round(float(Rh[2, 2]), 3))
