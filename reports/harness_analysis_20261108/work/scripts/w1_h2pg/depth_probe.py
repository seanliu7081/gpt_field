import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import restore, record_by_arrays
record = record_by_arrays('frame_000005_revision_000011.npz')
scene, data, _ = restore(record)
for cam in scene.frame.cameras:
    view = scene.geometry._views[cam.name]
    d = view.estimate.depth_m
    K, E = cam.intrinsics, cam.world_to_camera
    def world(u, v):
        z = d[v, u]
        p = np.linalg.inv(K) @ [u, v, 1] * z
        return z, (p - E[:3, 3]) @ E[:3, :3]
    probes = {'external': {'hand_center': (250, 90), 'base_column': (250, 165), 'yellow_mug_body': (335, 300), 'white_mug_body': (125, 270),
                           'table_front': (256, 470), 'table_mid': (256, 400), 'table_far': (420, 180), 'left_plate': (20, 340), 'right_plate': (485, 345),
                           'can': (85, 320), 'red_mug': (280, 230), 'back_wall': (60, 60)},
              'wrist': {'hand_bar': (256, 470), 'finger_left': (100, 440), 'table_top': (256, 60), 'table_mid': (300, 180), 'yellow_mug_rim': (60, 270),
                        'yellow_mug_inside': (110, 280), 'yellow_handle': (210, 272), 'white_mug_body': (450, 360)}}[cam.name]
    print('==', cam.name)
    for k, (u, v) in probes.items():
        z, w = world(u, v)
        print(f'  {k:18s} uv=({u},{v}) depth={z:.3f}  world={np.round(w,3)}')
    valid = np.isfinite(d)
    # table plane check: world z of all points; histogram
    pts = view.points_world
    print('  world z percentiles', np.round(np.percentile(pts[:,2],[1,5,25,50,75,95,99]),3))
