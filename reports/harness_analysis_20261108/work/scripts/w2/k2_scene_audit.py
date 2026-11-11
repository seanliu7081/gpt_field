"""Replicate CandidatePlanner._prepare (motion.py:96-208) with per-point provenance and score it.

Truth (layout manifest / asset AABBs) is used ONLY to score, never fed to the planner.
"""
import json, sys
from time import perf_counter
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts')
from k2_common import *  # noqa

def replica(planner, scene, state8, joints):
    """Line-by-line replica of motion.py:96-208 returning per-point provenance."""
    planner.bound = planner.kinematics.bind(state8, joints)
    points = np.asarray(scene.scene_points, float).reshape(-1, 3)
    owners = np.asarray(scene.scene_point_object_ids, object)
    strong = np.zeros(len(points), bool); owned_strong = np.zeros(len(points), bool)
    for object_id, obj in scene.objects.items():
        ids = np.flatnonzero((owners == object_id) & np.isfinite(points).all(axis=1))
        if not len(ids):
            continue
        surface, _ = planning_surface(obj)
        if len(surface):
            d, _ = cKDTree(surface).query(points[ids])
            strong[ids[d <= 1e-6]] = True; owned_strong[ids[d <= 1e-6]] = True
    view_label = np.empty(len(points), object); rows = np.zeros(len(points), int); cols = np.zeros(len(points), int)
    support = np.zeros(len(points), int); contra = np.zeros(len(points), int)
    off = 0
    for name, view in scene.geometry._views.items():
        n = len(view.points_world)
        strong[off:off+n] |= (view.support >= 2) & (view.contradictions == 0)
        view_label[off:off+n] = name; rows[off:off+n] = view.rows; cols[off:off+n] = view.cols
        support[off:off+n] = view.support; contra[off:off+n] = view.contradictions
        off += n
    valid = np.isfinite(points).all(axis=1)
    source_indices = np.flatnonzero(valid)
    P, O, S = points[valid], owners[valid], strong[valid]
    self_shapes = planner.visual_robot.shapes(planner.bound, joints, state8[6:8])
    margin = planner.config.collision_margin_m
    keep = np.ones(len(P), bool)
    removed_by = np.full(len(P), '', object)
    in_sil = np.zeros(len(P), bool)          # inside some envelope silhouette polygon
    front_gap = np.full(len(P), np.inf)      # entry - measured (positive: in front of the robot surface)
    cameras = {c.name: c for c in scene.frame.cameras}
    for name, interval in scene.evidence['scene_point_view_slices'].items():
        ids = np.flatnonzero((O == '') & (source_indices >= interval[0]) & (source_indices < interval[1]))
        camera = cameras[name]
        origin = -camera.world_to_camera[:3, :3].T @ camera.world_to_camera[:3, 3]
        cp = P[ids] @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
        pr = cp @ camera.intrinsics.T
        uv = pr[:, :2] / pr[:, 2:3]
        h, w = camera.rgb.shape[:2]
        rv = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
        px = np.rint(np.where(rv[:, None], uv, 0)).astype(int)
        rays = P[ids] - origin
        md = np.linalg.norm(rays, axis=1)
        rays /= np.maximum(md[:, None], 1e-12)
        for shape in self_shapes:
            if shape.contains(origin[None], 1e-7)[0]:
                continue
            vertices = shape.surface_points() @ camera.world_to_camera[:3, :3].T + camera.world_to_camera[:3, 3]
            if np.all(vertices[:, 2] <= 1e-7):
                continue
            sil = rv
            if np.all(vertices[:, 2] > 1e-7):
                import cv2
                pv = vertices @ camera.intrinsics.T
                poly = cv2.convexHull(np.float32(pv[:, :2] / pv[:, 2:3]))
                mask = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(mask, np.int32(np.rint(np.clip(poly, -1e6, 1e6))), 1)
                sil = rv & mask[px[:, 1], px[:, 0]].astype(bool)
            in_sil[ids[sil]] = True
            # exact entry distance for all silhouette points (diagnostic; the runtime adds a sphere pre-filter)
            sil_ids = np.flatnonzero(sil)
            for o in range(0, len(sil_ids), 4096):
                ch = sil_ids[o:o+4096]
                entry = shape.ray_entry_distances(origin, P[ids[ch]])
                gap = entry - md[ch]
                front_gap[ids[ch]] = np.minimum(front_gap[ids[ch]], np.where(np.isfinite(entry), gap, np.inf))
            center = shape.center - origin
            along = np.einsum('ij,j->i', rays, center)
            radius = shape.radius
            broad = (keep[ids] & sil & (along+radius >= 0) & (float(center@center)-along*along <= radius*radius)
                     & (md >= np.maximum(along-radius, 0.)-margin))
            local_ids = np.flatnonzero(broad)
            for o in range(0, len(local_ids), 1024):
                ch = local_ids[o:o+1024]
                entry = shape.ray_entry_distances(origin, P[ids[ch]])
                occ = np.isfinite(entry) & (md[ch] >= entry-margin)
                keep[ids[ch[occ]]] = False
                removed_by[ids[ch[occ]]] = shape.name
    prov = dict(points=P, owners=O, strong=S, owned_strong=owned_strong[valid], keep=keep, removed_by=removed_by,
                in_sil=in_sil, front_gap=front_gap, view=view_label[valid], row=rows[valid], col=cols[valid],
                support=support[valid], contra=contra[valid], self_shapes=self_shapes)
    return prov

def stats(prov, label):
    P, keep, S, view = prov['points'], prov['keep'], prov['strong'], prov['view']
    out = {'frame': label, 'points': int(len(P)), 'self_removed': int((~keep).sum()),
           'kept': int(keep.sum()), 'strong_kept': int((S & keep).sum()), 'uncertain_kept': int((~S & keep).sum())}
    out['strong_share_of_kept'] = out['strong_kept'] / max(1, out['kept'])
    out['strong_kept_owned_coherent'] = int((prov['owned_strong'] & keep).sum())
    for v in ('external', 'wrist'):
        m = view == v
        out[f'{v}_kept'] = int((m & keep).sum()); out[f'{v}_strong_kept'] = int((m & keep & S).sum())
        out[f'{v}_support_ge2'] = int((m & (prov['support'] >= 2)).sum())
        out[f'{v}_contradicted'] = int((m & (prov['contra'] > 0)).sum())
        out[f'{v}_removed'] = int((m & ~keep).sum())
    # truth regions (evaluation only)
    tz = truth['table_top_z']
    can = in_box(P, truth['can']['lower'], truth['can']['upper'], .01)
    objs = np.zeros(len(P), bool)
    for k in ('white_yellow_mug_1', 'porcelain_mug_1', 'red_coffee_mug_1', 'plate_1', 'plate_2', 'can'):
        objs |= in_box(P, truth[k]['lower'], truth[k]['upper'], .03)
    table_xy = (P[:, 0] > -0.45) & (P[:, 0] < 0.19) & (P[:, 1] > -0.6) & (P[:, 1] < 0.7)
    table = table_xy & ~objs & (np.abs(P[:, 2] - tz) < .10)
    out['can_points_kept'] = {v: int((can & keep & (view == v)).sum()) for v in ('external', 'wrist')}
    out['can_points_strong'] = {v: int((can & keep & S & (view == v)).sum()) for v in ('external', 'wrist')}
    out['can_points_self_removed'] = int((can & ~keep).sum())
    out['table_points'] = {}
    for v in ('external', 'wrist'):
        m = table & (view == v) & keep
        if m.sum():
            out['table_points'][v] = {'n': int(m.sum()), 'median_z_minus_truth_m': float(np.median(P[m, 2]) - tz),
                                      'p10_p90_z_minus_truth_m': [float(np.percentile(P[m, 2], 10) - tz), float(np.percentile(P[m, 2], 90) - tz)],
                                      'strong_share': float((S & m).sum() / m.sum()),
                                      'contradicted_share': float(((prov['contra'] > 0) & m).sum() / m.sum())}
    # phantom candidates: kept points above any scene object (z > 0.60) over the table footprint
    high = keep & table_xy & (P[:, 2] > 0.60)
    out['kept_above_z060_over_table'] = {v: {'n': int((high & (view == v)).sum()), 'strong': int((high & S & (view == v)).sum()),
                                            'inside_robot_silhouette': int((high & prov['in_sil'] & (view == v)).sum())} for v in ('external', 'wrist')}
    # kept unowned points inside the current robot silhouette (predicted in front of the robot by > margin)
    fg = keep & prov['in_sil'] & (prov['owners'] == '')
    gaps = prov['front_gap'][fg]
    out['kept_inside_robot_silhouette'] = {v: int((fg & (view == v)).sum()) for v in ('external', 'wrist')}
    finite = gaps[np.isfinite(gaps)]
    out['kept_inside_silhouette_front_gap_m'] = ({'n_ray_hits': int(len(finite)), 'p10': float(np.percentile(finite, 10)), 'median': float(np.median(finite)),
                                                  'p90': float(np.percentile(finite, 90))} if len(finite) else None)
    out['kept_inside_silhouette_no_envelope_hit'] = int((~np.isfinite(gaps)).sum())
    below = keep & table_xy & (P[:, 2] < tz - 0.05)
    out['kept_below_table_minus_5cm_over_table'] = {v: int((below & (view == v)).sum()) for v in ('external', 'wrist')}
    return out

if __name__ == '__main__':
    steps = [int(x) for x in sys.argv[1:]] or [0, 4, 13, 19]
    allout = []
    for step in steps:
        rec = record_for_step(step)
        scene, err = restore(rec)
        state, q = step_state(step)
        planner = CandidatePlanner(kin)
        t0 = perf_counter(); planner._prepare(scene, state, q); prep_s = perf_counter() - t0
        t0 = perf_counter(); prov = replica(CandidatePlanner(kin), scene, state, q); rep_s = perf_counter() - t0
        assert int((~prov['keep']).sum()) == planner._self_removed, (int((~prov['keep']).sum()), planner._self_removed)
        assert int((prov['strong'] & prov['keep']).sum()) == int(planner._strong.sum())
        s = stats(prov, f"step{step}/frame{scene.frame_id}/{rec['arrays']}")
        s['prepare_s_cpu'] = prep_s; s['ownership_match_max_m'] = err
        s['objects'] = {k: {'status': o.status, 'raw': len(o.points), 'owned_points': int((scene.scene_point_object_ids == k).sum())} for k, o in scene.objects.items()}
        allout.append(s)
        print(json.dumps(s), flush=True)
    (W2 / 'k2_scene_audit.json').write_text(json.dumps(allout, indent=1))
