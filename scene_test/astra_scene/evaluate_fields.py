"""Evaluate Astra analytic fields and saved-only decodes; GT is evaluator-only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
RGB = HERE.parent / 'rgb_scene'
sys.path.insert(0, str(RGB))
from ray_evaluation import camera_rays, raycast_depth, depth_hit_scores, independent_first_hit
from rgb_metrics import load_mesh
from field_csg import validate_field, field_bounds, analytic_first_hits

REGIONS = ('valid_depth', 'nonrobot_visible', 'target_bowl_visible', 'target_cabinet_visible')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked(root, record, consumed):
    path = (Path(root) / record['file']).resolve()
    digest = sha(path)
    if digest != record['sha256']:
        raise ValueError(f'Hash mismatch: {path.name}')
    consumed[str(path)] = digest
    return path


def verify_integrity(manifest, groups):
    integrity = manifest.get('integrity', {})
    if integrity.get('settings') is not True:
        raise ValueError('Missing or failed upstream settings integrity')
    for group in groups:
        rows = integrity.get(group)
        if not isinstance(rows, list) or not rows or any(x.get('unchanged') is not True for x in rows):
            raise ValueError(f'Missing or failed upstream integrity: {group}')


def verify_source_records(root, records, consumed):
    if not records:
        raise ValueError('No source provenance records')
    for record in records:
        checked(Path('/'), {'file': record['path'], 'sha256': record['sha256']}, consumed)
        checked(root, {'file': record['copy'], 'sha256': record['copy_sha256']}, consumed)


def verify_upstream(build, manifest, consumed):
    verify_integrity(manifest, ('sources', 'inputs', 'artifacts'))
    verify_source_records(build, manifest['sources'], consumed)
    checked(build, {'file': 'settings.json', 'sha256': manifest['settings_sha256']}, consumed)
    records = [x for x in manifest['inputs'] if x.get('copy') == 'inputs/inference_manifest.json']
    if len(records) != 1:
        raise ValueError('Missing unique frozen inference manifest')
    record = records[0]
    original = checked(Path('/'), {'file': record['path'], 'sha256': record['sha256']}, consumed)
    frozen = checked(build, {'file': record['copy'], 'sha256': record['copy_sha256']}, consumed)
    if record['sha256'] != record['copy_sha256']:
        raise ValueError('Inference manifest copy differs from original')
    inference = json.loads(frozen.read_text())
    if (inference.get('status') != 'complete' or inference.get('model') != 'gpt-6-astra'
            or inference.get('reasoning_effort') != 'medium'
            or inference.get('billing') != 'chatgpt_subscription'
            or inference.get('api_key_calls') != 0 or inference.get('subscription_inference_calls') != 3):
        raise ValueError('Inference run is not the completed authorized Astra experiment')
    verify_integrity(inference, ('sources', 'inputs', 'prepared', 'artifacts'))
    verify_source_records(original.parent, inference['sources'], consumed)
    for upstream in (manifest, inference):
        source = [x for x in upstream['sources'] if Path(x['path']).name == 'field_csg.py']
        if len(source) != 1 or source[0]['sha256'] != sha(HERE/'field_csg.py'):
            raise ValueError('Analytic evaluator and saved field use different CSG semantics')
    predictions = inference['predictions']
    if (len(predictions) != 3 or len({x['name'] for x in predictions}) != 3
            or any(x['status'] != 'valid' for x in predictions)):
        raise ValueError('Expected three independent valid Astra field predictions')
    for prediction in predictions:
        for key in ('field', 'raw_field', 'raw_response', 'rgb'):
            checked(original.parent, prediction[key], consumed)
    return {p['name']: p for p in predictions}


def six_views(bounds, side=64):
    """Six fixed external cameras derived solely from the model's root bounds."""
    low, high = (np.asarray(x, float) for x in bounds)
    center = .5 * (low + high)
    radius = .5 * float(np.linalg.norm(high - low))
    distance = 3 * radius + .1
    focal = side / (2 * np.tan(np.pi / 6))
    K = np.array([[focal, 0, (side-1)/2], [0, focal, (side-1)/2], [0, 0, 1.]])
    for axis in range(3):
        for sign in (-1, 1):
            eye = center.copy()
            eye[axis] += sign * distance
            forward = (center - eye) / distance
            down = np.array([0., 1., 0.])
            if abs(down @ forward) > .9:
                down = np.array([0., 0., 1.])
            right = np.cross(down, forward)
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            rotation = np.column_stack((right, down, forward))
            yield {'name': f'axis{axis}_{sign:+d}', 'eye': eye,
                   'rotation': rotation, 'K': K, 'shape': (side, side)}


def analytic_depth(spec, origins, directions, shape):
    values = np.asarray(analytic_first_hits(spec, origins, directions), float)
    if values.shape != (len(directions),):
        raise ValueError('Analytic intersection shape mismatch')
    if np.any(np.isfinite(values) & (values <= 0)):
        raise ValueError('Analytic intersection must be positive or NaN')
    return values.reshape(shape)


def representation_scores(decoded, original):
    result = depth_hit_scores(decoded, original, np.isfinite(original))
    extra = np.isfinite(decoded) & ~np.isfinite(original)
    result.update({'total_test_rays': int(original.size),
                   'new_hit_where_original_misses': int(extra.sum()),
                   'new_hit_fraction_all_test_rays': float(extra.mean()),
                   'scope': 'six model-bounds-derived external views; first-hit axial distance'})
    return result


def validate_virtual_hits(mesh, K, depth):
    """One fixed-seed hit and miss per rigidly transformed diagnostic view.

    The primary image already receives the helper's full independent check.
    These additional checks cover every view transform without repeating dozens
    of exhaustive triangle scans per view.
    """
    flat = depth.ravel()
    rng = np.random.default_rng(20261019)
    chosen = []
    for available in (np.flatnonzero(np.isfinite(flat)), np.flatnonzero(~np.isfinite(flat))):
        if len(available):
            chosen.append(int(rng.choice(available)))
    rays = camera_rays(depth.shape, K)
    maximum = 0.
    for index in chosen:
        expected = independent_first_hit(np.asarray(mesh.triangles), rays[index])
        actual = flat[index]
        if np.isfinite(expected) != np.isfinite(actual):
            raise ValueError('Virtual-view native/independent visibility mismatch')
        if np.isfinite(expected):
            maximum = max(maximum, abs(float(expected-actual)))
    if maximum > 2e-6:
        raise ValueError('Virtual-view native/independent depth mismatch')
    return {'passed': True, 'rays': len(chosen), 'max_absolute_error_m': maximum,
            'tolerance_m': 2e-6, 'selection': 'fixed seed one hit and one miss where available'}


def _freeze(output):
    paths = [Path(__file__), HERE/'field_csg.py', RGB/'ray_evaluation.py', RGB/'rgb_metrics.py']
    source = output/'source'
    source.mkdir()
    result = []
    for path in paths:
        target = source/path.name
        shutil.copyfile(path, target)
        result.append({'path': str(path.resolve()), 'sha256': sha(path),
                       'copy': str(target.relative_to(output)), 'copy_sha256': sha(target)})
    return result


def run(args):
    started = time.monotonic()
    build = args.fields.resolve()
    refs = args.references.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'arrays').mkdir()
    consumed = {}
    sources = _freeze(output)
    manifest = {'status': 'running', 'fields': str(build), 'references': str(refs),
                'sources': sources, 'settings': {'virtual_view_side': args.view_side,
                 'primary_evaluation': 'all exact GT camera pixels', 'oracle_alignment': False,
                 'representation': 'CSG implicit field; not previous projective TSDF'}}
    dump(output/'manifest.json', manifest)
    try:
        bm = json.loads((build/'manifest.json').read_text())
        consumed[str(build/'manifest.json')] = sha(build/'manifest.json')
        if bm['status'] != 'complete':
            raise ValueError('Field build is not complete')
        predictions = verify_upstream(build, bm, consumed)
        # All field blocks and resulting meshes are frozen upstream artifacts.
        configurations = bm['configurations']
        snapshots = bm['snapshots']
        expected_names = {'bowl002_initial', 'drawer003_closed_step60', 'drawer003_open_step552'}
        if len(snapshots) != 3 or {s['name'] for s in snapshots} != expected_names:
            raise ValueError('Expected the three fixed snapshots exactly once')
        expected_configs = {(name, h) for name in expected_names for h in (.004, .002)}
        actual_configs = {(c['snapshot'], c['spacing_m']) for c in configurations}
        if len(configurations) != 6 or actual_configs != expected_configs:
            raise ValueError('Expected exactly the six 4/2 mm configurations')
        reference = json.loads(refs.read_text())
        consumed[str(refs)] = sha(refs)
        flags = reference.get('artifacts_unchanged')
        if (reference['status'] != 'complete' or not isinstance(flags, dict)
                or not flags or not all(value is True for value in flags.values())):
            raise ValueError('Reference preparation did not pass integrity checks')
        rm = {s['name']: s for s in reference['snapshots']}
        inputs = json.loads(args.rgb_inputs.read_text())
        consumed[str(args.rgb_inputs.resolve())] = sha(args.rgb_inputs)
        im = {s['name']: s for s in inputs['inputs']}
        results = {'status': 'running', 'settings': manifest['settings'],
                   'snapshots': [], 'configurations': []}
        for snapshot in snapshots:
            name = snapshot['name']
            reference_row = rm[name]
            if im[name]['sha256'] != reference_row['input_rgb']['sha256']:
                raise ValueError('Model RGB and evaluator RGB do not match')
            if predictions[name]['rgb']['sha256'] != im[name]['sha256']:
                raise ValueError('Actual Astra input RGB differs from the evaluator image')
            if snapshot['field']['sha256'] != predictions[name]['field']['sha256']:
                raise ValueError('Built field differs from the accepted Astra field')
            input_path = Path(im[name]['rgb'])
            if not input_path.is_absolute():
                input_path = args.rgb_inputs.parent/input_path
            if sha(input_path) != im[name]['sha256']:
                raise ValueError('RGB source has changed')
            consumed[str(input_path.resolve())] = im[name]['sha256']
            checked(refs.parent, reference_row['input_rgb'], consumed)
            spec_path = checked(build, snapshot['field'], consumed)
            spec = validate_field(json.loads(spec_path.read_text()))
            gt_file = checked(refs.parent, reference_row['evaluator_reference'], consumed)
            calibration_file = checked(refs.parent, reference_row['calibration'], consumed)
            calibration = json.loads(calibration_file.read_text())
            K = np.asarray(calibration['K'], float)
            with np.load(gt_file, allow_pickle=False) as data:
                gt = data['depth_z_m'].copy()
                masks = {r: data[r].astype(bool) for r in REGIONS}
            directions = camera_rays(gt.shape, K)
            original = analytic_depth(spec, np.zeros_like(directions), directions, gt.shape)
            snapshot_result = {'name': name, 'regions': {},
                               'estimated_focal_px': spec['estimated_focal_px'],
                               'K_gt': K.tolist(), 'bounds_m': [x.tolist() for x in field_bounds(spec)],
                               'input_rgb_sha256': im[name]['sha256'],
                               'field_sha256': sha(spec_path)}
            for region in REGIONS:
                snapshot_result['regions'][region] = depth_hit_scores(original, gt, masks[region])
            path = output/'arrays'/f'{name}_reference_and_analytic.npz'
            np.savez_compressed(path, gt_depth_m=gt, analytic_depth_m=original,
                                K_gt=K, **masks)
            snapshot_result['arrays'] = {'file': str(path.relative_to(output)), 'sha256': sha(path)}
            views = list(six_views(field_bounds(spec), args.view_side))
            original_views = []
            for view in views:
                dr = camera_rays(view['shape'], view['K']) @ view['rotation'].T
                origin = np.broadcast_to(view['eye'], dr.shape).copy()
                original_views.append(analytic_depth(spec, origin, dr, view['shape']))
            original_views = np.stack(original_views)
            snapshot_result['B_views'] = [{k: v.tolist() if isinstance(v, np.ndarray) else v
                                          for k, v in view.items()} for view in views]
            results['snapshots'].append(snapshot_result)
            print(json.dumps({'analytic_evaluated': name}), flush=True)
            for config in configurations:
                if config['snapshot'] != name:
                    continue
                if config['status'] != 'ok':
                    results['configurations'].append({'name': config['name'], 'status': config['status']})
                    continue
                for record in config['fields']:
                    checked(build, record, consumed)
                mesh_file = checked(build, config['decoded_mesh'], consumed)
                mesh = load_mesh(mesh_file)
                decoded, validation = raycast_depth(mesh, K, gt.shape, validate=True)
                row = {'name': config['name'], 'snapshot': name, 'spacing_m': config['spacing_m'],
                       'status': 'ok', 'regions': {}, 'ray_validation': validation,
                       'decoded_mesh': {'file': str(mesh_file), 'sha256': sha(mesh_file)}}
                for region in REGIONS:
                    row['regions'][region] = depth_hit_scores(decoded, gt, masks[region])
                decoded_views = []
                validations = []
                for view in views:
                    transformed = trimesh.Trimesh(vertices=(mesh.vertices-view['eye'])@view['rotation'],
                                                  faces=mesh.faces, process=False)
                    depth, check = raycast_depth(transformed, view['K'], view['shape'], validate=False)
                    check['independent_validation'] = validate_virtual_hits(transformed, view['K'], depth)
                    decoded_views.append(depth)
                    validations.append(check)
                decoded_views = np.stack(decoded_views)
                row['representation_B_rays'] = representation_scores(decoded_views, original_views)
                row['B_ray_validation'] = validations
                path = output/'arrays'/f'{config["name"]}.npz'
                np.savez_compressed(path, decoded_depth_m=decoded,
                                    B_analytic_depth_m=original_views, B_decoded_depth_m=decoded_views)
                row['arrays'] = {'file': str(path.relative_to(output)), 'sha256': sha(path)}
                results['configurations'].append(row)
                dump(output/'metrics.json', results)
                print(json.dumps({'evaluated': config['name']}), flush=True)
        results['status'] = 'complete'
        dump(output/'metrics.json', results)
        source_unchanged = all(sha(x['path']) == x['sha256'] and
                               sha(output/x['copy']) == x['copy_sha256'] for x in sources)
        inputs_unchanged = all(sha(p) == h for p, h in consumed.items())
        manifest.update({'status': 'complete' if source_unchanged and inputs_unchanged else 'integrity_failed',
                         'source_unchanged': source_unchanged, 'inputs_unchanged': inputs_unchanged,
                         'consumed_sha256': consumed, 'metrics_sha256': sha(output/'metrics.json'),
                         'elapsed_s': time.monotonic()-started,
                         'artifacts_sha256': {str(p.relative_to(output)): sha(p)
                                               for p in sorted((output/'arrays').glob('*.npz'))}})
        dump(output/'manifest.json', manifest)
        if manifest['status'] != 'complete':
            raise ValueError('Evaluator source/input changed during the run')
    except Exception as exc:
        manifest.update({'status': 'failed', 'error_type': type(exc).__name__,
                         'error': str(exc), 'elapsed_s': time.monotonic()-started})
        dump(output/'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', required=True, type=Path)
    parser.add_argument('--references', type=Path, default=RGB/'references_004/manifest.json')
    parser.add_argument('--rgb-inputs', type=Path, default=RGB/'inputs_001/rgb_inputs.json')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--view-side', type=int, default=64)
    args = parser.parse_args()
    if not 8 <= args.view_side <= 256:
        parser.error('--view-side must be between 8 and 256')
    run(args)


if __name__ == '__main__':
    main()
