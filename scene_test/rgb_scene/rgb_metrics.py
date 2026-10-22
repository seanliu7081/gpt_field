"""Shared, independent continuous-surface metrics for the new RGB experiment."""
from pathlib import Path
import sys
import numpy as np
import trimesh
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/'libero_scene'/'_deps'), str(HERE.parent/'_deps'), str(HERE.parent)]
from reconstruct import sample_mesh, sha256, write_json, jsonable
from mesh_queries import MeshDistance

TOLERANCES = (.001, .002, .005, .010)


def mesh_distance(mesh, points):
    import point_cloud_utils as pcu
    points = np.asarray(points, np.float64)
    if len(points) == 0:
        return np.empty(0)
    if not len(mesh.faces):
        return np.full(len(points), np.inf)
    # PCU0.34 singleton bug is avoided by the validated batched path.
    query = np.repeat(points, 2, axis=0) if len(points) == 1 else points
    distance, _, _ = pcu.closest_points_on_mesh(query, np.ascontiguousarray(mesh.vertices, np.float64), np.ascontiguousarray(mesh.faces, np.int64))
    distance = np.asarray(distance).reshape(-1)[:len(points)]
    if np.any(~np.isfinite(distance)) or np.any(distance < 0):
        raise ValueError('nonfinite/negative unsigned distance')
    return distance


def validate_distance(mesh, points):
    if not len(mesh.faces) or not len(points):
        return {'status': 'empty_not_validatable'}
    points = np.asarray(points)[:12]
    fast, reference = mesh_distance(mesh, points), MeshDistance(mesh)(points)
    error = float(np.max(np.abs(fast-reference)))
    if error > 2e-7:
        raise ValueError(f'native unsigned vs independent normalized projection differs by {error} m')
    return {'passed': True, 'queries': len(points), 'max_absolute_difference_m': error, 'tolerance_m': 2e-7}


def stats(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    return {'sample_count': len(values), 'finite_count': len(finite),
            'mean_m': float(np.mean(values)) if len(values) and np.all(np.isfinite(values)) else None,
            'p50_m': float(np.quantile(values, .5)) if len(values) and np.all(np.isfinite(values)) else None,
            'p95_m': float(np.quantile(values, .95)) if len(values) and np.all(np.isfinite(values)) else None,
            'sampled_max_m': float(np.max(values)) if len(values) and np.all(np.isfinite(values)) else None}


def scores(pred_to_ref, ref_to_pred):
    a, b = np.asarray(pred_to_ref), np.asarray(ref_to_pred)
    result = {'prediction_to_reference': stats(a), 'reference_to_prediction': stats(b),
              'symmetric_mean_m': float((a.mean()+b.mean())/2) if len(a) and len(b) and np.all(np.isfinite(a)) and np.all(np.isfinite(b)) else None,
              'maximum_is_sampled_not_hausdorff_bound': True, 'thresholds': {}}
    for t in TOLERANCES:
        precision = float(np.mean(a <= t)) if len(a) else 0.
        recall = float(np.mean(b <= t)) if len(b) else 0.
        result['thresholds'][f'{t*1000:g}mm'] = {'precision': precision, 'recall': recall,
                            'f1': 2*precision*recall/(precision+recall) if precision+recall else 0.}
    return result


def compare_meshes(prediction, reference, reference_points, count, rng):
    pred_points = sample_mesh(prediction, count, rng) if len(prediction.faces) else np.empty((0, 3))
    a = mesh_distance(reference, pred_points)
    b = mesh_distance(prediction, reference_points)
    result = scores(a, b)
    result.update(prediction_faces=len(prediction.faces), reference_faces=len(reference.faces),
                  prediction_area_m2=float(prediction.area), reference_area_m2=float(reference.area),
                  prediction_surface_samples=len(pred_points), reference_surface_samples=len(reference_points),
                  distance_query_validation={'prediction_target': validate_distance(prediction, reference_points),
                                             'reference_target': validate_distance(reference, pred_points)})
    return result, {'prediction_points_m': pred_points, 'prediction_to_reference_m': a, 'reference_to_prediction_m': b}


def save_mesh(path, mesh):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, vertices_m=np.asarray(mesh.vertices, np.float64), faces=np.asarray(mesh.faces, np.int32))


def load_mesh(path):
    with np.load(path, allow_pickle=False) as data:
        return trimesh.Trimesh(vertices=data['vertices_m'], faces=data['faces'], process=False)
