import sys, json
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent))
from common import restore, record_by_arrays, feedback_state, kinematics
from gpt_field.foundation_harness.robot_geometry import RobotCollisionModel
OUT = Path(__file__).parent
arrays, step = 'frame_000005_revision_000011.npz', 4
record = record_by_arrays(arrays)
scene, data, _ = restore(record)
state, q = feedback_state(step)
kin = kinematics(); model = RobotCollisionModel(kin)
bound = kin.bind(state, q)
shapes = model.shapes(bound, q, state[6:8])
info = {}
for cam in scene.frame.cameras:
    K, E = cam.intrinsics, cam.world_to_camera
    center = -E[:3,:3].T @ E[:3,3]
    info[cam.name] = {'camera_center_world': center.tolist(), 'K': K.tolist(), 'forward_world': (E[:3,:3].T @ [0,0,1]).tolist()}
    img = cam.rgb.copy()
    over = img.copy()
    for shape in shapes:
        v = shape.surface_points()
        c = v @ E[:3,:3].T + E[:3,3]
        c = c[c[:,2] > 1e-3]
        if len(c) < 3: continue
        uv = (c @ K.T)[:, :2] / c[:, 2:3]
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        color = (255,0,0) if 'link' in shape.name else (0,255,0)
        cv2.polylines(over, [hull], True, color, 1)
    cv2.imwrite(str(OUT / f'overlay_{cam.name}.png'), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    # depth visualisation
    d = scene.geometry._views[cam.name].estimate.depth_m
    dn = np.nan_to_num((d - np.nanmin(d)) / (np.nanmax(d) - np.nanmin(d)))
    cv2.imwrite(str(OUT / f'depth_{cam.name}.png'), (dn*255).astype(np.uint8))
    info[cam.name]['depth_range'] = [float(np.nanmin(d)), float(np.nanmax(d))]
pose, _ = bound.forward(q)
info['eef_fk'] = pose[:3,3].tolist(); info['state_pos'] = state[:3].tolist()
info['base_origin_world'] = bound.world_from_base[:3,3].tolist()
print(json.dumps(info, indent=1))
