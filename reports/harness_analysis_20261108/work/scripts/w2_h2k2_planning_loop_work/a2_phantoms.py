"""A2: identify the retained z>0.6 m 'obstacles' and the table-plane strength.

For every retained (post self-mask) point above z=0.60 m (nothing in the T04
scene except the robot is that high: table top 0.437 m, can top 0.513 m, mug
tops ~0.54 m), test whether it projects inside the CURRENT robot silhouette in
its source camera and how far in front of the robot's first ray entry it lies.
Also histogram wrist/external table-plane heights. Offline; truth only scores.
"""
import json
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
slices = scene.evidence['scene_point_view_slices']
view_of = np.full(len(all_pts), '', object)
for name, (a, b) in slices.items():
    view_of[a:b] = name
d, idx = cKDTree(all_pts[valid]).query(cloud)
views = view_of[src[idx]]
cams = {c.name: c for c in scene.frame.cameras}
vis_shapes = planner.visual_robot.shapes(planner.bound, q, state[6:8])
col_shapes = planner.robot.shapes(planner.bound, q, state[6:8])

out = {'step': step}
for vname in ('external', 'wrist'):
    cam = cams[vname]
    E, K = cam.world_to_camera, cam.intrinsics
    origin = -E[:3, :3].T @ E[:3, 3]
    h, w = cam.rgb.shape[:2]
    sil = np.zeros((h, w), np.uint8)
    for s in vis_shapes:
        if s.contains(origin[None], 1e-7)[0]:
            continue
        v = s.surface_points() @ E[:3, :3].T + E[:3, 3]
        if np.all(v[:, 2] > 1e-7):
            pv = v @ K.T
            poly = cv2.convexHull(np.float32(pv[:, :2] / pv[:, 2:3]))
            cv2.fillConvexPoly(sil, np.int32(np.rint(np.clip(poly, -1e6, 1e6))), 1)
    sel = (views == vname) & (cloud[:, 2] > 0.60)
    pts = cloud[sel]
    res = {'points_above_0.60': int(sel.sum())}
    if len(pts):
        cp = pts @ E[:3, :3].T + E[:3, 3]
        uv = (cp @ K.T)[:, :2] / cp[:, 2:3]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
        px = np.rint(np.where(inside[:, None], uv, 0)).astype(int)
        in_sil = inside & sil[px[:, 1], px[:, 0]].astype(bool)
        dist = np.linalg.norm(pts - origin, axis=1)
        entry = np.full(len(pts), np.inf)
        for s in vis_shapes:
            if s.contains(origin[None], 1e-7)[0]:
                continue
            entry = np.minimum(entry, s.ray_entry_distances(origin, pts))
        ahead = entry - dist  # >0: predicted in front of first robot surface
        finite = np.isfinite(entry)
        # nearest collision-shape surface distance proxy: max plane residual (<=0 inside)
        res.update({
            'fraction_in_current_robot_silhouette': float(in_sil.mean()),
            'fraction_with_finite_robot_ray_entry': float(finite.mean()),
            'in_front_of_robot_surface_m_quantiles_(5,25,50,75,95)': np.quantile(ahead[finite], [.05, .25, .5, .75, .95]).round(4).tolist() if finite.any() else None,
            'z_quantiles': np.quantile(pts[:, 2], [.05, .5, .95]).round(3).tolist(),
            'x_range': [float(pts[:, 0].min()), float(pts[:, 0].max())],
            'y_range': [float(pts[:, 1].min()), float(pts[:, 1].max())],
            'pixel_rgb_mean_in_silhouette': cam.rgb[px[in_sil, 1], px[in_sil, 0]].mean(0).round(1).tolist() if in_sil.any() else None,
        })
    # table plane height distribution for this view (inside table xy, away from objects)
    tsel = (views == vname) & (cloud[:, 2] < 0.60)
    tz = cloud[tsel, 2]
    res['z_hist_below_0.60'] = {f'{a:.3f}-{b:.3f}': int(((tz >= a) & (tz < b)).sum())
                                for a, b in zip(np.arange(0.38, 0.60, 0.02), np.arange(0.40, 0.62, 0.02))}
    res['silhouette_pixels'] = int(sil.sum())
    out[vname] = res
    cv2.imwrite(str(W2 / f'data/a2_silhouette_{vname}_step{step}.png'), np.uint8(sil * 255))

# the specific debug points
for label, p in [('initial_configuration_debug.link6', [-0.11267635940894379, -0.007094041496755563, 0.839106912161268]),
                 ('previous_astra_direct_debug_v2.goal.link5', [-0.19807643480266826, -0.07931306335804075, 0.8143761101081948]),
                 ('saved_input_check.final.link6_example', [-0.20798223585454417, -0.08472090360947211, 0.8219281108280816])]:
    p = np.array(p)
    dd, ii = cKDTree(cloud).query(p)
    row = {'nearest_retained_point_m': float(dd), 'strong': bool(strong[ii]), 'view': str(views[ii])}
    for s in col_shapes:
        if s.name in ('link5_collision', 'link6_collision', 'link7_collision', 'link4_collision'):
            local = (p - s.pose[:3, 3]) @ s.pose[:3, :3]
            row[f'{s.name}_max_plane_residual_m'] = round(float(np.max(local @ s.equations[:, :3].T + s.equations[:, 3])), 4)
    cam = cams['external']; E, K = cam.world_to_camera, cam.intrinsics
    cp = E[:3, :3] @ p + E[:3, 3]; uv = (K @ cp)[:2] / cp[2]
    row['external_uv'] = uv.round(1).tolist()
    out[label] = row
print(json.dumps(out, indent=1))
(W2 / f'data/a2_phantoms_step{step}.json').write_text(json.dumps(out, indent=1))
