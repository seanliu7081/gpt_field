"""Ray-cast plan-time wrist cells (quarter_turns=0, i.e. the unturned upright wrist image) with the step-0 wrist calibration."""
import json, sys
import numpy as np
d = json.load(open(sys.argv[1]))
txt = d["payload"]["input"][0]["content"][0]["text"]
cam = json.loads(txt[txt.find("{"):])["geometry"]["cameras"]["wrist"]
R = np.asarray(cam["rotation_world_from_camera"]); C = np.asarray(cam["position_world_m"])
f = cam["focal_length_px"][0]; c = cam["principal_point_px"][0]
def cell_uv(k):
    r, q = (k - 1) // 8, (k - 1) % 8
    return q * 64 + 32, r * 64 + 32
for k, name in [(12, "plate C"), (13, "plate C alt"), (29, "region E"), (49, "white mug A"), (55, "pudding D"), (24, "red mug (visible right)")]:
    u, v = cell_uv(k)
    dw = R @ np.array([(u - c) / f, -(v - c) / f, -1.0])
    t = (0.437 - C[2]) / dw[2]
    p = C + t * dw
    print(f"wrist cell {k:2d} (u={u},v={v}) -> world ({p[0]:+.3f},{p[1]:+.3f})  [{name}]")
