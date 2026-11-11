"""Would robust payload geometry have produced placement candidates? Offline, saved artifacts only."""
import sys, json, math
from pathlib import Path
import numpy as np, cv2
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).parent))
from common import restore, record_by_arrays, kinematics
from gpt_field.foundation_harness.motion import CandidatePlanner, PlanningFailure
from gpt_field.robodojo_direct_sim import rotation_matrix
OUT = Path(__file__).parent
planner = CandidatePlanner(kinematics())

def best_probe(P, target):
    pts, nrm = target.points, target.normals
    sup = pts[np.abs(nrm[:, 2]) > .85]
    tree = cKDTree(sup[:, :2])
    cands = sup[np.linspace(0, len(sup) - 1, min(24, len(sup)), dtype=int)]
    local = P - P.mean(0)
    best = None
    for c in cands:
        for yaw in (0., math.pi/2, -math.pi/2, math.pi):
            R = rotation_matrix([0., 0., yaw])
            rot = local @ R.T
            cxy = (rot[:, :2].min(0) + rot[:, :2].max(0)) / 2
            fp = rot[:, :2] + (c[:2] - cxy)
            lo, hi = fp.min(0), fp.max(0)
            probes = np.array([[x, y] for x in (lo[0], (lo[0]+hi[0])/2, hi[0]) for y in (lo[1], (lo[1]+hi[1])/2, hi[1])])
            d, _ = tree.query(probes)
            if best is None or d.max() < best:
                best = float(d.max())
    return best, np.ptp(sup, 0)[:2].round(4).tolist()

rows = []
for arrays, obj_id, target_id in [('frame_000005_revision_000011.npz', 'yellow_white_mug', 'seed_3'),
                                  ('frame_000017_revision_000037.npz', 'white_mug', 'seed_1')]:
    record = record_by_arrays(arrays)
    scene, data, pvi = restore(record)
    obj, target = scene.objects[obj_id], scene.objects[target_id]
    P = obj.points
    meta = {c['id']: c for c in record['array_metadata']['candidates']}
    sel_ids = obj.evidence['selected_candidate_ids']
    interior = np.zeros(len(P), bool); xview = np.zeros(len(P), bool)
    for part, idx in pvi[obj_id]:
        view = scene.geometry._views[part['view']]
        cand = meta[sel_ids[part['view']]]
        mask = np.unpackbits(data[cand['mask_bits_key']])[:512*512].reshape(512, 512).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        d = view.estimate.depth_m
        gy, gx = np.gradient(d); grad = np.nan_to_num(np.hypot(gx, gy), nan=1.)
        r, c = view.rows[idx], view.cols[idx]
        interior[part['start']:part['end']] = (dist[r, c] > 2) & (grad[r, c] < .01)
        xview[part['start']:part['end']] = part['support'] == 'crossview_supported'
    def trim(Q):
        lo, hi = np.percentile(Q, 2, 0), np.percentile(Q, 98, 0)
        return Q[np.all((Q >= lo) & (Q <= hi), 1)]
    variants = {'V0_runtime_all_points': P, 'V1_crossview_only': P[xview], 'V2_edge_and_gradient_filtered': P[interior],
                'V3_V2_plus_2_98_trim': trim(P[interior])}
    q3 = variants['V3_V2_plus_2_98_trim']
    variants['V4_bottom_2cm_slice_of_V3'] = q3[q3[:, 2] < q3[:, 2].min() + .02]
    for name, Q in variants.items():
        held = {'points_in_gripper': (Q - Q.mean(0)).tolist()}
        try:
            poses = planner._placements(target, held, np.eye(3), {'placement_relation': 'on'})
            n = len(poses)
        except PlanningFailure as exc:
            n = 0
        best, sup_ext = best_probe(Q, target)
        rows.append({'arrays': arrays, 'object': obj_id, 'target': target_id, 'variant': name, 'points': int(len(Q)),
                     'xy_extent_m': np.ptp(Q, 0)[:2].round(4).tolist(), 'z_range_m': [float(Q[:, 2].min().round(3)), float(Q[:, 2].max().round(3))],
                     'placements_returned_by_runtime_code': n, 'best_worst_probe_gap_m': round(best, 4),
                     'allowed_gap_m': .012, 'visible_support_xy_extent_m': sup_ext})
        print(json.dumps(rows[-1]))
(OUT / 'placement_variants_results.json').write_text(json.dumps(rows, indent=1))
