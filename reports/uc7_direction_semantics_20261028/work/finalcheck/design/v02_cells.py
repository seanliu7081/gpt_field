"""V0.2 check: do the relayout ground-truth object centres project into the planner's pointed cells (35, 38, 61)?
Uses the recorded agentview calibration of the relayout t6 trial request 1 (displayed image = upright, unmirrored)."""
import json, glob, sys
import numpy as np
f = sorted(glob.glob(sys.argv[1] + '/requests/000001-*.request.json'))[0]
text = json.load(open(f))['payload']['input'][0]['content'][0]['text']
geo = json.loads(text)['geometry']['cameras']['external']
R = np.asarray(geo['rotation_world_from_camera']); P = np.asarray(geo['position_world_m'])
fx = geo['focal_length_px'][0]; cx, cy = geo['principal_point_px']
def cell(x):
    c = R.T @ (np.asarray(x, float) - P)
    u, v = cx + fx * c[0] / -c[2], cy - fx * c[1] / -c[2]
    col, row = int(u // 64), int(v // 64)
    return round(u, 1), round(v, 1), row * 8 + col + 1
# relayout layout_changes.json custom_xyz (object origins at z=0.48 before settling); plate pose preserved (official init)
for name, xyz, pointed in (("pudding", (-0.16, -0.14, 0.46), 35), ("white mug", (-0.12, 0.18, 0.48), 38),
                           ("plate (rotation_check ANCHOR)", (0.149, 0.006, 0.44), 61)):
    u, v, c = cell(xyz)
    print(f"{name:30s} -> pixel ({u},{v}) cell {c}; planner pointed {pointed}; same={c == pointed}")
