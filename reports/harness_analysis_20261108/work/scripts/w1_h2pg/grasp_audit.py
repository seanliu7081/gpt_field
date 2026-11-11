"""Where do the run-time 'opposing contact' grasp proposals for the whole yellow mug come from? Offline only."""
import sys, json
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent))
from common import restore, record_by_arrays, kinematics
from gpt_field.foundation_harness.motion import CandidatePlanner
planner = CandidatePlanner(kinematics())
record = record_by_arrays('frame_000005_revision_000011.npz')
scene, data, pvi = restore(record)
obj = scene.objects['yellow_white_mug']
P = obj.points
label = np.empty(len(P), object); edge = np.zeros(len(P)); grad = np.zeros(len(P))
meta = {c['id']: c for c in record['array_metadata']['candidates']}
for part, idx in pvi['yellow_white_mug']:
    view = scene.geometry._views[part['view']]
    label[part['start']:part['end']] = part['view'] + ':' + part['support']
    mask = np.unpackbits(data[meta[obj.evidence['selected_candidate_ids'][part['view']]]['mask_bits_key']])[:512*512].reshape(512, 512).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    gy, gx = np.gradient(view.estimate.depth_m)
    edge[part['start']:part['end']] = dist[view.rows[idx], view.cols[idx]]
    grad[part['start']:part['end']] = np.nan_to_num(np.hypot(gx, gy)[view.rows[idx], view.cols[idx]], nan=1.)
g = planner._grasps(obj, {})
tree_idx = {tuple(np.round(p, 7)): i for i, p in enumerate(P)}
rows = []
for score, pose, fingers, contacts in g:
    ids = [int(np.argmin(np.linalg.norm(P - np.array(c), axis=1))) for c in contacts]
    rows.append({'alignment': round(float(score), 3), 'width_m': round(float(np.linalg.norm(np.subtract(*contacts))), 4),
                 'grasp_xyz': np.round(pose[:3, 3], 3).tolist(), 'approach': np.round(pose[:3, 2], 2).tolist(),
                 'contacts': [{'source': label[i], 'xyz': np.round(P[i], 3).tolist(), 'mask_edge_px': round(float(edge[i]), 2),
                               'depth_grad_m_per_px': round(float(grad[i]), 4), 'normal': np.round(obj.normals[i], 2).tolist()} for i in ids]})
summary = {'proposals': len(rows),
           'contacts_within_2px_of_mask_edge': int(sum(c['mask_edge_px'] <= 2 for r in rows for c in r['contacts'])),
           'contacts_with_depth_grad_gt_1cm_per_px': int(sum(c['depth_grad_m_per_px'] > .01 for r in rows for c in r['contacts'])),
           'contacts_below_true_table_z_0.4268': int(sum(c['xyz'][2] < .4268 for r in rows for c in r['contacts'])),
           'contact_sources': {s: int(sum(c['source'] == s for r in rows for c in r['contacts'])) for s in set(label)},
           'proposals': rows[:6]}
print(json.dumps(summary, indent=1))
(Path(__file__).parent / 'grasp_audit_results.json').write_text(json.dumps(summary, indent=1))
