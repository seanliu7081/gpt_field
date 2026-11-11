import sys, json
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent))
from common import restore, record_by_arrays
OUT = Path(__file__).parent
res = {}
for arrays, obj_id, truth in [('frame_000005_revision_000011.npz', 'yellow_white_mug', {'origin': [-0.0276, 0.0779, 0.4369], 'low_z': 0.4368, 'high_z': 0.54}),
                              ('frame_000017_revision_000037.npz', 'white_mug', {'origin': [-0.0875, -0.1559, 0.4344], 'low_z': 0.4368, 'high_z': 0.5428})]:
    record = record_by_arrays(arrays)
    scene, data, pvi = restore(record)
    obj = scene.objects[obj_id]
    P = obj.points
    out = {'n': len(P), 'extent_all_m': np.ptp(P, 0).round(4).tolist(), 'min': P.min(0).round(3).tolist(), 'max': P.max(0).round(3).tolist(),
           'truth': truth, 'parts': []}
    sel_ids = obj.evidence['selected_candidate_ids']
    meta = {c['id']: c for c in record['array_metadata']['candidates']}
    for part, indices in pvi[obj_id]:
        view = scene.geometry._views[part['view']]
        pts = P[part['start']:part['end']]
        rows, cols = view.rows[indices], view.cols[indices]
        cand = meta[sel_ids[part['view']]]
        mask = np.unpackbits(data[cand['mask_bits_key']])[:512*512].reshape(512, 512).astype(np.uint8)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        dmask = dist[rows, cols]
        d = view.estimate.depth_m
        gy, gx = np.gradient(d)
        grad = np.hypot(gx, gy)[rows, cols]
        below_table = pts[:, 2] < truth['low_z'] - 0.01
        r = {'view': part['view'], 'support': part['support'], 'n': int(len(pts)),
             'extent_m': np.ptp(pts, 0).round(4).tolist(),
             'p2_p98_extent_m': (np.percentile(pts, 98, 0) - np.percentile(pts, 2, 0)).round(4).tolist(),
             'z_percentiles': np.percentile(pts[:, 2], [1, 5, 50, 95, 99]).round(3).tolist(),
             'frac_below_true_table_minus_1cm': float(below_table.mean()),
             'frac_within_2px_of_mask_edge': float((dmask <= 2).mean()),
             'frac_depth_grad_gt_1cm_per_px': float((grad > .01).mean())}
        # who defines the xy extremes?
        ext = {}
        for axis, nm in ((0, 'x'), (1, 'y')):
            for fn, lab in ((np.argmin, 'min'), (np.argmax, 'max')):
                i = fn(pts[:, axis])
                ext[f'{nm}{lab}'] = {'value': float(pts[i, axis]), 'pixel_uv': [int(cols[i]), int(rows[i])], 'mask_edge_dist_px': float(dmask[i]),
                                     'depth_grad_m_per_px': float(grad[i]), 'z': float(pts[i, 2])}
        r['xy_extreme_points'] = ext
        out['parts'].append(r)
    # robust alternatives
    def extent(Q):
        return np.ptp(Q, 0).round(4).tolist() if len(Q) else None
    xv = np.concatenate([P[p['start']:p['end']] for p, _ in pvi[obj_id] if p['support'] == 'crossview_supported'])
    out['extent_crossview_only_m'] = extent(xv)
    out['p2_p98_extent_all_m'] = (np.percentile(P, 98, 0) - np.percentile(P, 2, 0)).round(4).tolist()
    out['p2_p98_extent_crossview_only_m'] = (np.percentile(xv, 98, 0) - np.percentile(xv, 2, 0)).round(4).tolist()
    # statistical outlier removal: keep points whose 16-NN mean distance < median + 2*MAD
    from scipy.spatial import cKDTree
    t = cKDTree(P); dd, _ = t.query(P, 17); m = dd[:, 1:].mean(1)
    med = np.median(m); mad = np.median(np.abs(m - med))
    keep = m < med + 3 * 1.4826 * mad
    out['sor_kept'] = int(keep.sum()); out['extent_after_SOR_m'] = extent(P[keep])
    # largest connected component at 1 cm radius
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    pairs = t.query_pairs(0.01, output_type='ndarray')
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    nc, lab = connected_components(g, directed=False)
    big = np.bincount(lab).argmax()
    out['components_1cm'] = int(nc); out['largest_component_n'] = int((lab == big).sum()); out['extent_largest_component_m'] = extent(P[lab == big])
    res[obj_id] = out
    print(json.dumps(out, indent=1))
(OUT / 'payload_points_results.json').write_text(json.dumps(res, indent=1))
