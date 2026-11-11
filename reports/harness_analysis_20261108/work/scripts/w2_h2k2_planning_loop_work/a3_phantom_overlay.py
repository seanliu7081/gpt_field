"""A3: overlay retained high (z>0.60 m) points and table-plane strength on the RGB.

Red: retained uncertain points above z=0.60 (nothing real is that high except the
robot). Green: robot silhouette outline (current pose, visual envelopes).
Blue: strong (cross-view/owned) table-plane points; yellow: uncertain table-plane points.
"""
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

from k2common import WORK as W2, scene_at, kin, truth_boxes
from gpt_field.foundation_harness.motion import CandidatePlanner

step = int(sys.argv[1]) if len(sys.argv) > 1 else 13
scene, state, q, record, err = scene_at(step)
planner = CandidatePlanner(kin)
planner._prepare(scene, state, q)
cloud, strong = planner._cloud, planner._strong
T = truth_boxes()
all_pts = np.asarray(scene.scene_points, float)
valid = np.isfinite(all_pts).all(1)
src = np.flatnonzero(valid)
view_of = np.full(len(all_pts), '', object)
for name, (a, b) in scene.evidence['scene_point_view_slices'].items():
    view_of[a:b] = name
d, idx = cKDTree(all_pts[valid]).query(cloud)
views = view_of[src[idx]]
vis_shapes = planner.visual_robot.shapes(planner.bound, q, state[6:8])
for cam in scene.frame.cameras:
    E, K = cam.world_to_camera, cam.intrinsics
    origin = -E[:3, :3].T @ E[:3, 3]
    img = cam.rgb.copy()
    h, w = img.shape[:2]
    sil = np.zeros((h, w), np.uint8)
    for s in vis_shapes:
        if s.contains(origin[None], 1e-7)[0]:
            continue
        v = s.surface_points() @ E[:3, :3].T + E[:3, 3]
        if np.all(v[:, 2] > 1e-7):
            pv = v @ K.T
            poly = cv2.convexHull(np.float32(pv[:, :2] / pv[:, 2:3]))
            cv2.fillConvexPoly(sil, np.int32(np.rint(np.clip(poly, -1e6, 1e6))), 1)
    def draw(mask, color):
        pts = cloud[mask]
        cp = pts @ E[:3, :3].T + E[:3, 3]
        uv = (cp @ K.T)[:, :2] / cp[:, 2:3]
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        px = np.int32(uv[ok])
        img[px[:, 1], px[:, 0]] = color
    table = (views == cam.name) & (np.abs(cloud[:, 2] - T['table_top_z']) <= 0.01)
    draw(table & ~strong, (255, 220, 0))
    draw(table & strong, (0, 80, 255))
    draw((views == cam.name) & (cloud[:, 2] > 0.60), (255, 0, 0))
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    big = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(W2 / f'data/a3_overlay_{cam.name}_step{step}.png'), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    print(cam.name, img.shape)
