"""Why the can (same instance in both views) failed the 0.9 cross-view mask agreement (T04 frame 0)."""
import sys, json
sys.path.insert(0, '/workspace/gpt_field/src')
import numpy as np
from scipy import ndimage
from gpt_field.foundation_vision.bridge import load_manifest
from gpt_field.foundation_vision.backends import depth_on_original_rays
from gpt_field.foundation_vision.types import DepthEstimate
from gpt_field.foundation_vision.geometry import build_geometry, _mask_on_depth, _sample, _mask_at_world, _project, GeometryConfig
R = '/workspace/gpt_field/runs/'
m = load_manifest(R + 'foundation_vision_preparation_001/input/manifest.json')
frame = m.frames[0]; cams = {c.name: c for c in frame.cameras}; names = [c.name for c in frame.cameras]
z = np.load(R + 'foundation_vision_preparation_001/da3_real_003/cold_native_prediction.npz')
raw = z['raw_depth_model_units']; K = z['native_intrinsics']; conf = z['raw_score_map']
mk = np.load(R + 'foundation_vision_sam_validation_001/sam_inference_002/all_candidate_masks.npz')
masks = {'external': mk['mask_000'], 'wrist': mk['mask_001']}
S0 = 0.8780772385418903
D = {}
for i, n in enumerate(names):
    d, sc = depth_on_original_rays(raw[i]*S0, K[i], cams[n], conf[i])
    D[n] = DepthEstimate(n, d, cams[n].intrinsics, cams[n].world_to_camera, sc, source='da3')
g = build_geometry(frame, D, {'can': masks})
og = g.object_geometry['can']
print('reproduced pair evidence', json.dumps(og['crossview_mask_evidence']))
# calibration check: project the true can centre into each view
CAN_LO = np.array([-0.054474949094583634, -0.24536935556741868, 0.43696]); CAN_HI = np.array([0.0077168271458434145, -0.18373144547662518, 0.51315])
ctr = (CAN_LO + CAN_HI) / 2
for n in names:
    uv, zc = _project(ctr[None], cams[n].intrinsics, cams[n].world_to_camera)
    u, v = uv[0]; inside = masks[n][int(round(v)), int(round(u))]
    ys, xs = np.nonzero(masks[n])
    print(n, 'true can centre projects to (u=%.1f, v=%.1f), inside SAM mask: %s; mask centroid (u=%.1f, v=%.1f)' % (u, v, inside, xs.mean(), ys.mean()))
# where do the external->wrist comparable-but-outside points land?
vE, vW = g._views['external'], g._views['wrist']
sel = _mask_on_depth(masks['external'], vE); P = vE.points_world[sel]
rows, cols, surface, occl, front, valid = _sample(P, vW, g.config)
inm = _mask_at_world(P, masks['wrist'], vW)
bad = surface & ~inm
dist = ndimage.distance_transform_edt(~masks['wrist'])
db = dist[rows[bad], cols[bad]]
print('external->wrist comparable', int(surface.sum()), 'in-mask', int((surface & inm).sum()), 'outside', int(bad.sum()))
print('  outside points: distance to wrist mask (px) p50 %.1f p75 %.1f p90 %.1f max %.1f; within 3px %.3f, within 5px %.3f, within 10px %.3f'
      % (np.median(db), np.percentile(db, 75), np.percentile(db, 90), db.max(), (db <= 3).mean(), (db <= 5).mean(), (db <= 10).mean()))
for tol_px in (0, 2, 3, 5, 8):
    dil = ndimage.binary_dilation(masks['wrist'], iterations=tol_px) if tol_px else masks['wrist']
    inm2 = _mask_at_world(P, dil, vW)
    print('   wrist mask dilated %d px -> agreement %.3f' % (tol_px, (surface & inm2).sum() / surface.sum()))
# isolate SAM coverage from depth error: use TRUE geometry of external can pixels (ray-cast) and project to wrist
cam = cams['external']; h, w = cam.rgb.shape[:2]
vv, uu = np.nonzero(masks['external'])
rc = np.linalg.solve(cam.intrinsics, np.stack((uu.astype(float), vv.astype(float), np.ones(len(uu)))))
Rw = cam.world_to_camera[:3, :3]; t = cam.world_to_camera[:3, 3]; C = -Rw.T @ t; dw = Rw.T @ rc
AX = (CAN_LO[:2] + CAN_HI[:2]) / 2; RAD = float(np.mean((CAN_HI - CAN_LO)[:2]) / 2)
oc = C[:2, None] - AX[:, None]; dd = dw[:2]; a = (dd**2).sum(0); b = 2*(oc*dd).sum(0); c = (oc**2).sum(0) - RAD**2
disc = b*b - 4*a*c; ok = disc >= 0; l1 = (-b - np.sqrt(np.where(ok, disc, 0))) / (2*a); zs = C[2] + l1*dw[2]
side = ok & (l1 > 0) & (zs >= CAN_LO[2]) & (zs <= CAN_HI[2])
lt = (CAN_HI[2] - C[2]) / dw[2]; pt = C[:2, None] + lt*dd; top = (lt > 0) & (np.linalg.norm(pt - AX[:, None], axis=0) <= RAD)
lam = np.minimum(np.where(side, l1, np.inf), np.where(top, lt, np.inf)); hit = np.isfinite(lam)
PT = (C[:, None] + dw*lam).T[hit]
uvw, zw = _project(PT, cams['wrist'].intrinsics, cams['wrist'].world_to_camera)
insideimg = (uvw[:, 0] >= 0) & (uvw[:, 0] <= 511) & (uvw[:, 1] >= 0) & (uvw[:, 1] <= 511) & (zw > 0)
pix = np.floor(uvw[insideimg] + 0.5).astype(int)
inmask = masks['wrist'][pix[:, 1], pix[:, 0]]
# visibility in wrist of these true points: is the point the first hit along the wrist ray? approximate using cylinder ray-cast from wrist
camw = cams['wrist']; Rw2 = camw.world_to_camera[:3, :3]; t2 = camw.world_to_camera[:3, 3]; C2 = -Rw2.T @ t2
dirs = (PT[insideimg] - C2); rngs = np.linalg.norm(dirs, axis=1); dirs = (dirs / rngs[:, None]).T
oc2 = C2[:2, None] - AX[:, None]; dd2 = dirs[:2]; a2 = (dd2**2).sum(0); b2 = 2*(oc2*dd2).sum(0); c2 = (oc2**2).sum(0) - RAD**2
disc2 = b2*b2 - 4*a2*c2; ok2 = disc2 >= 0; m1 = (-b2 - np.sqrt(np.where(ok2, disc2, 0))) / (2*a2); zs2 = C2[2] + m1*dirs[2]
side2 = ok2 & (m1 > 0) & (zs2 >= CAN_LO[2]) & (zs2 <= CAN_HI[2])
lt2 = (CAN_HI[2] - C2[2]) / dirs[2]; pt2 = C2[:2, None] + lt2*dd2; top2 = (lt2 > 0) & (np.linalg.norm(pt2 - AX[:, None], axis=0) <= RAD)
first = np.minimum(np.where(side2, m1, np.inf), np.where(top2, lt2, np.inf))
visible = np.abs(first - rngs) < 0.004
print('TRUE-geometry external can points: %d hit can, %d project inside wrist image, of which visible-from-wrist %d; visible & inside wrist SAM mask %.3f; any inside mask %.3f'
      % (hit.sum(), insideimg.sum(), visible.sum(), inmask[visible].mean() if visible.any() else float('nan'), inmask.mean()))
# true silhouette of the can in the wrist view vs SAM wrist mask (coverage / IoU)
hh, ww = 512, 512; vw_, uw_ = np.indices((hh, ww), dtype=float)
rcw = np.linalg.solve(camw.intrinsics, np.stack((uw_.ravel(), vw_.ravel(), np.ones(hh*ww)))); dww = Rw2.T @ rcw
oc3 = C2[:2, None] - AX[:, None]; dd3 = dww[:2]; a3 = (dd3**2).sum(0); b3 = 2*(oc3*dd3).sum(0); c3 = (oc3**2).sum(0) - RAD**2
disc3 = b3*b3 - 4*a3*c3; ok3 = disc3 >= 0; n1 = (-b3 - np.sqrt(np.where(ok3, disc3, 0))) / (2*a3); zs3 = C2[2] + n1*dww[2]
side3 = ok3 & (n1 > 0) & (zs3 >= CAN_LO[2]) & (zs3 <= CAN_HI[2]); lt3 = (CAN_HI[2] - C2[2]) / dww[2]; pt3 = C2[:2, None] + lt3*dd3
top3 = (lt3 > 0) & (np.linalg.norm(pt3 - AX[:, None], axis=0) <= RAD)
sil = (side3 | top3).reshape(hh, ww)
sm = masks['wrist']
print('wrist: true can silhouette px %d, SAM mask px %d, IoU %.3f, SAM covers %.3f of silhouette, SAM precision %.3f'
      % (sil.sum(), sm.sum(), (sil & sm).sum() / (sil | sm).sum(), (sil & sm).sum() / sil.sum(), (sil & sm).sum() / sm.sum()))
ys, xs = np.nonzero(sil); print('   true silhouette bbox x %d-%d y %d-%d (touches image border: %s)' % (xs.min(), xs.max(), ys.min(), ys.max(), xs.max() >= 511 or ys.max() >= 511 or xs.min() <= 0 or ys.min() <= 0))
# same for external
came = cams['external']; Re = came.world_to_camera[:3, :3]; te = came.world_to_camera[:3, 3]; Ce = -Re.T @ te
rce = np.linalg.solve(came.intrinsics, np.stack((uw_.ravel(), vw_.ravel(), np.ones(hh*ww)))); dwe = Re.T @ rce
oc4 = Ce[:2, None] - AX[:, None]; dd4 = dwe[:2]; a4 = (dd4**2).sum(0); b4 = 2*(oc4*dd4).sum(0); c4 = (oc4**2).sum(0) - RAD**2
disc4 = b4*b4 - 4*a4*c4; ok4 = disc4 >= 0; n4 = (-b4 - np.sqrt(np.where(ok4, disc4, 0))) / (2*a4); zs4 = Ce[2] + n4*dwe[2]
side4 = ok4 & (n4 > 0) & (zs4 >= CAN_LO[2]) & (zs4 <= CAN_HI[2]); lt4 = (CAN_HI[2] - Ce[2]) / dwe[2]; pt4 = Ce[:2, None] + lt4*dd4
top4 = (lt4 > 0) & (np.linalg.norm(pt4 - AX[:, None], axis=0) <= RAD)
sile = (side4 | top4).reshape(hh, ww); se = masks['external']
print('external: true can silhouette px %d, SAM mask px %d, IoU %.3f' % (sile.sum(), se.sum(), (sile & se).sum() / (sile | se).sum()))
