"""What if the wrist depth scale were consistent? Re-run H2's own _object/_normals and the planner's
grasp search on saved masks + depth, with the wrist depth divided by s. Offline only."""
import sys, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from common import record_by_arrays, kinematics, V
from gpt_field.foundation_vision.types import CameraFrame, VisionFrame, DepthEstimate
from gpt_field.foundation_vision.geometry import build_geometry, GeometryConfig
from gpt_field.foundation_vision.engine import EngineConfig
from gpt_field.foundation_harness.perception import HarnessPerception, _normals
from gpt_field.foundation_harness.motion import CandidatePlanner, PlanningFailure
OUT = Path(__file__).parent
planner = CandidatePlanner(kinematics())

def build(record, wrist_scale):
    data = np.load(V / record['arrays'])
    cams, depths = [], {}
    for c in record['array_metadata']['cameras']:
        p, name = c['array_prefix'], c['view']
        cams.append(CameraFrame(name, data[p+'_rgb'], data[p+'_intrinsics'], data[p+'_world_to_camera'], 0.))
        d = data[p+'_depth_m'].astype(float) / (wrist_scale if name == 'wrist' else 1.)
        depths[name] = DepthEstimate(name, d, data[p+'_depth_intrinsics'], data[p+'_world_to_camera'])
    frame = VisionFrame(record['snapshot']['frame_id'], 0., tuple(cams))
    return frame, build_geometry(frame, depths, {}), data

def run(arrays, targets, wrist_scale):
    record = record_by_arrays(arrays)
    frame, geom, data = build(record, wrist_scale)
    perc = HarnessPerception(SimpleNamespace(config=EngineConfig()))
    perc._frame, perc._geometry = frame, geom
    perc._normal_maps = {n: _normals(v) for n, v in geom._views.items()}
    snap = {o['id']: o for o in record['snapshot']['objects']}
    meta = {c['id']: c for c in record['array_metadata']['candidates']}
    out = {}
    for tid in targets:
        ev = snap[tid]['evidence']
        masks = {view: np.unpackbits(data[meta[cid]['mask_bits_key']])[:512*512].reshape(512, 512).astype(bool)
                 for view, cid in ev['selected_candidate_ids'].items()}
        obj, accepted = perc._object({'id': tid, 'query': snap[tid]['query'], 'selections': {}, 'parent_id': None},
                                      masks, ev['selection'], ev['selected_candidate_ids'])
        P = obj.points
        row = {'status': obj.status, 'points': int(len(P)), 'accepted_by_view': obj.evidence['accepted_points_by_view'],
               'single_view_by_view': obj.evidence['accepted_single_view_points_by_view'],
               'pairs': [{k: p[k] for k in ('source_view', 'mask_matches', 'depth_contradictions', 'occluded')} for p in obj.evidence['crossview_pairs']],
               'xy_extent_m': np.ptp(P, 0)[:2].round(4).tolist() if len(P) else None,
               'z_range_m': [round(float(P[:, 2].min()), 3), round(float(P[:, 2].max()), 3)] if len(P) else None,
               'normals_up_frac': round(float((obj.normals[:, 2] > .85).mean()), 3) if len(P) else None,
               'normals_zero_frac': round(float((np.linalg.norm(obj.normals, axis=1) < .5).mean()), 3) if len(P) else None}
        try:
            g = planner._grasps(obj, {})
            row['grasp_proposals'] = len(g)
            row['best_grasp_alignment'] = round(float(g[0][0]), 3)
            row['best_grasp_width_m'] = round(float(np.linalg.norm(np.subtract(*g[0][3]))), 4)
        except PlanningFailure as exc:
            row['grasp_proposals'] = 0; row['grasp_failure'] = exc.reason
        out[tid] = row
    return out

res = {}
for arrays, targets in [('frame_000003_revision_000007.npz', ['yellow_white_mug', 'yellow_handle']),
                        ('frame_000005_revision_000011.npz', ['yellow_white_mug', 'seed_3'])]:
    for s in (1.0, 1.2):
        key = f'{arrays}|wrist_depth/{s}'
        res[key] = run(arrays, targets, s)
        print(key, json.dumps(res[key], indent=1), flush=True)
(OUT / 'scale_whatif_results.json').write_text(json.dumps(res, indent=1))
