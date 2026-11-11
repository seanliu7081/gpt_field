"""Rebuild geometry after a robust per-frame affine alignment of wrist depth to the external view
(estimated only from the two predicted depth maps; no truth used in the fix). Re-run H2's own _object
and the run-time grasp search. Offline only."""
import sys, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import record_by_arrays, kinematics, V
from gpt_field.foundation_vision.types import CameraFrame, VisionFrame, DepthEstimate
from gpt_field.foundation_vision.geometry import build_geometry
from gpt_field.foundation_vision.engine import EngineConfig
from gpt_field.foundation_harness.perception import HarnessPerception, _normals
from gpt_field.foundation_harness.motion import CandidatePlanner, PlanningFailure
planner = CandidatePlanner(kinematics())

def fit_affine(frame, depths):
    geom = build_geometry(frame, depths, {})
    ext, wr = geom._views['external'], geom._views['wrist']
    Kw, Ew = wr.camera.intrinsics, wr.camera.world_to_camera
    c = ext.points_world @ Ew[:3, :3].T + Ew[:3, 3]; c = c[c[:, 2] > .05]
    uv = (c @ Kw.T)[:, :2] / c[:, 2:3]
    ins = (uv[:, 0] >= 0) & (uv[:, 0] <= 511) & (uv[:, 1] >= 0) & (uv[:, 1] <= 511)
    zc, px = c[ins, 2], np.rint(uv[ins]).astype(int)
    order = np.argsort(zc); key = px[order, 1] * 512 + px[order, 0]
    _, first = np.unique(key, return_index=True); sel = order[first]
    zr, u, v = zc[sel], px[sel, 0], px[sel, 1]
    dw = wr.estimate.depth_m[v, u]; good = np.isfinite(dw) & (v < 380)
    r = zr[good] / dw[good]; med = np.median(r); keep = np.abs(r - med) < 3 * 1.4826 * np.median(np.abs(r - med))
    a, b = np.linalg.lstsq(np.column_stack((dw[good][keep], np.ones(keep.sum()))), zr[good][keep], rcond=None)[0]
    return float(a), float(b)

def run(arrays, targets, align):
    record = record_by_arrays(arrays); data = np.load(V / record['arrays'])
    cams, depths = [], {}
    for c in record['array_metadata']['cameras']:
        p, name = c['array_prefix'], c['view']
        cams.append(CameraFrame(name, data[p+'_rgb'], data[p+'_intrinsics'], data[p+'_world_to_camera'], 0.))
        depths[name] = DepthEstimate(name, data[p+'_depth_m'].astype(float), data[p+'_depth_intrinsics'], data[p+'_world_to_camera'])
    frame = VisionFrame(record['snapshot']['frame_id'], 0., tuple(cams))
    fit = None
    if align:
        a, b = fit_affine(frame, depths); fit = [round(a, 4), round(b, 4)]
        d = depths['wrist']; depths['wrist'] = DepthEstimate('wrist', a * d.depth_m + b, d.intrinsics, d.world_to_camera)
    geom = build_geometry(frame, depths, {})
    perc = HarnessPerception(SimpleNamespace(config=EngineConfig()))
    perc._frame, perc._geometry = frame, geom
    perc._normal_maps = {n: _normals(v) for n, v in geom._views.items()}
    snap = {o['id']: o for o in record['snapshot']['objects']}
    meta = {c['id']: c for c in record['array_metadata']['candidates']}
    out = {'wrist_affine_fit': fit, 'views': {n: {'crossview_supported_frac': round(float(((v.support >= 2) & (v.contradictions == 0)).mean()), 4),
                                                  'contradicted_frac': round(float((v.contradictions > 0).mean()), 4),
                                                  'occluded_frac': round(float((v.occlusions > 0).mean()), 4)} for n, v in geom._views.items()}}
    for tid in targets:
        ev = snap[tid]['evidence']
        masks = {view: np.unpackbits(data[meta[cid]['mask_bits_key']])[:512*512].reshape(512, 512).astype(bool)
                 for view, cid in ev['selected_candidate_ids'].items()}
        obj, _ = perc._object({'id': tid, 'query': snap[tid]['query'], 'selections': {}, 'parent_id': None}, masks, ev['selection'], ev['selected_candidate_ids'])
        P = obj.points
        row = {'points': int(len(P)), 'accepted_by_view': obj.evidence['accepted_points_by_view'],
               'single_view_by_view': obj.evidence['accepted_single_view_points_by_view'],
               'xy_extent_m': np.ptp(P, 0)[:2].round(4).tolist() if len(P) else None,
               'z_range_m': [round(float(P[:, 2].min()), 3), round(float(P[:, 2].max()), 3)] if len(P) else None}
        try:
            g = planner._grasps(obj, {}); row['grasp_proposals'] = len(g)
        except PlanningFailure as exc:
            row['grasp_proposals'] = 0; row['grasp_failure'] = exc.reason
        out[tid] = row
    return out

res = {}
for arrays, targets in [('frame_000003_revision_000007.npz', ['yellow_white_mug', 'yellow_handle']), ('frame_000005_revision_000011.npz', ['yellow_white_mug'])]:
    for align in (False, True):
        key = f'{arrays}|wrist_affine_aligned={align}'
        res[key] = run(arrays, targets, align); print(key, json.dumps(res[key]), flush=True)
(Path(__file__).parent / 'align_whatif_results.json').write_text(json.dumps(res, indent=1))
