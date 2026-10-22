"""Sample Astra's implicit CSG field, save scalar blocks, then decode from disk.

The ``decode`` subcommand reads only saved numeric NPZ blocks. It does not import
the CSG evaluator or read a model response, RGB image, calibration, or truth.
All units are metres, coordinates camera-CV XYZ, negative values inside material.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ALIGNMENT_M = .004
BLOCK_CELLS = 32
MAX_CANDIDATES = 250_000
MAX_SAMPLED_NODES = 300_000_000
MAX_EXTENT_M = 8.
MAX_VOLUME_M3 = 64.
MAX_FACES = 20_000_000
EXPECTED_NAMES = {'bowl002_initial', 'drawer003_closed_step60', 'drawer003_open_step552'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked_file(root, record):
    path = (Path(root) / record['file']).resolve()
    if not path.is_file() or sha256(path) != record['sha256']:
        raise ValueError(f'input SHA256 mismatch: {path}')
    return path


def grid_domain(raw_lower, raw_upper, spacing, block_cells=BLOCK_CELLS):
    """Geometry-only AABB, padded by 8 mm and aligned identically at both scales."""
    lower, upper = np.asarray(raw_lower, float), np.asarray(raw_upper, float)
    h = float(spacing)
    if (lower.shape != (3,) or upper.shape != (3,) or
            not np.all(np.isfinite([lower, upper])) or np.any(upper <= lower)):
        raise ValueError('finite nonempty geometry bounds required')
    if h <= 0 or not np.isfinite(h) or not np.isclose(ALIGNMENT_M / h, round(ALIGNMENT_M / h), atol=1e-12, rtol=0):
        raise ValueError('spacing must divide the common 4 mm alignment')
    if block_cells < 1 or block_cells > 128:
        raise ValueError('block_cells must be in [1,128]')
    extent = upper - lower
    if np.any(extent > MAX_EXTENT_M) or np.prod(extent) > MAX_VOLUME_M3:
        raise ValueError('model geometry exceeds declared 8 m axis / 64 m^3 bounds budget')
    margin = 2 * ALIGNMENT_M
    lower = np.floor((lower - margin) / ALIGNMENT_M) * ALIGNMENT_M
    upper = np.ceil((upper + margin) / ALIGNMENT_M) * ALIGNMENT_M
    cells = np.rint((upper - lower) / h).astype(np.int64)
    counts = (cells + block_cells - 1) // block_cells
    candidates = int(np.prod(counts))
    if candidates > MAX_CANDIDATES:
        raise MemoryError(f'{candidates} candidate blocks exceed {MAX_CANDIDATES}; no automatic coarsening')
    return lower, cells, counts


def plan_blocks(evaluate, lower, cells, spacing, block_cells=BLOCK_CELLS):
    """Discard a block only with a 1-Lipschitz certificate that zero is absent.

    This certifies block skipping, not that vertex sampling detects every thin
    feature within a retained block. Trilinear interpolation can lose features.
    """
    counts = (np.asarray(cells) + block_cells - 1) // block_cells
    ids = np.stack(np.meshgrid(*(np.arange(int(x)) for x in counts), indexing='ij'), axis=-1).reshape(-1, 3)
    starts = ids * block_cells
    shapes = np.minimum(block_cells, np.asarray(cells) - starts)
    centers = np.asarray(lower) + (starts + shapes / 2) * spacing
    radii = np.linalg.norm(shapes * spacing / 2, axis=1)
    center_values = np.empty(len(centers), float)
    for first in range(0, len(centers), 32768):
        values = np.asarray(evaluate(centers[first:first + 32768]), float)
        if values.shape != (len(centers[first:first + 32768]),) or not np.all(np.isfinite(values)):
            raise ValueError('field evaluator returned invalid centre values')
        center_values[first:first + len(values)] = values
    tolerance = 1e-10 * np.maximum(1., np.abs(center_values) + radii)
    retained = np.abs(center_values) <= radii + tolerance
    node_counts = np.prod(shapes + 1, axis=1)
    sampled = int(node_counts[retained].sum())
    if sampled > MAX_SAMPLED_NODES:
        raise MemoryError(f'{sampled} retained vertices exceed {MAX_SAMPLED_NODES}; no automatic coarsening')
    return {'block_ids': ids, 'start_cells': starts, 'cell_shapes': shapes,
            'center_values_m': center_values, 'half_diagonal_m': radii,
            'certificate_tolerance_m': tolerance, 'retained': retained,
            'sampled_nodes': sampled}


def _save_one_block(evaluate, lower, spacing, start, cell_shape, block_id, directory):
    axes = [np.arange(int(n) + 1, dtype=np.int64) + start[k] for k, n in enumerate(cell_shape)]
    indices = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    points = lower + indices.reshape(-1, 3) * spacing
    raw_values = np.asarray(evaluate(points), np.float64)
    if raw_values.shape != (len(points),) or not np.all(np.isfinite(raw_values)):
        raise ValueError('field evaluator returned invalid vertex samples')
    values = raw_values.astype(np.float32).reshape(tuple(cell_shape + 1))
    quantization = float(np.max(np.abs(values.ravel().astype(float) - raw_values)))
    block_lower = lower + start * spacing
    path = directory / ('block_' + '_'.join(f'{int(x):04d}' for x in block_id) + '.npz')
    with path.open('xb') as stream:
        np.savez_compressed(stream, values=values, lower_m=block_lower,
                            spacing_m=np.float64(spacing), axis_order='xyz', units='m',
                            negative='inside_material', coordinate_frame='camera_cv',
                            vertex_quantization_max_m=quantization,
                            global_start_cells=start, global_lower_m=lower)
    return {'file': path.name, 'sha256': sha256(path), 'shape': list(values.shape),
            'lower_m': block_lower.tolist(), 'sampled_min_m': float(values.min()),
            'sampled_max_m': float(values.max()), 'vertex_quantization_max_m': quantization}


_WORKER_ARGS = None


def _initialize_sample_worker(core_path, spec, lower, spacing, directory):
    global _WORKER_ARGS
    core = frozen_core(core_path)
    _WORKER_ARGS = (core, spec, lower, spacing, directory)


def _sample_block_worker(item):
    core, spec, lower, spacing, directory = _WORKER_ARGS
    start, cell_shape, block_id = item
    return _save_one_block(lambda p: core.evaluate_field(spec, p), lower, spacing,
                           start, cell_shape, block_id, directory)


def save_field_blocks(evaluate, raw_lower, raw_upper, spacing, directory, *, block_cells=BLOCK_CELLS,
                      workers=1, worker_spec=None, worker_core=None):
    """Serialize actual float32 samples; no scene data are embedded in blocks."""
    if not isinstance(workers, int) or workers < 1 or workers > 16:
        raise ValueError('workers must be in [1,16]')
    if workers > 1 and (worker_spec is None or worker_core is None):
        raise ValueError('parallel sampling requires explicit frozen CSG core and model specification')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    lower, cells, counts = grid_domain(raw_lower, raw_upper, spacing, block_cells)
    plan = plan_blocks(evaluate, lower, cells, spacing, block_cells)
    with (directory / 'block_certificates.npz').open('xb') as stream:
        np.savez_compressed(stream, **plan, lower_m=lower, spacing_m=spacing, total_cells=cells)
    items = [(plan['start_cells'][i], plan['cell_shapes'][i], plan['block_ids'][i])
             for i in np.flatnonzero(plan['retained'])]
    records = []
    if workers == 1:
        for start, cell_shape, block_id in items:
            records.append(_save_one_block(evaluate, lower, spacing, start, cell_shape, block_id, directory))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_sample_worker,
                                 initargs=(str(worker_core), worker_spec, lower, spacing, directory)) as pool:
            # Ordered map results preserve identical block and mesh ordering.
            for index, record in enumerate(pool.map(_sample_block_worker, items, chunksize=1)):
                records.append(record)
                if (index + 1) % 256 == 0 or index + 1 == len(items):
                    print(json.dumps({'configuration': directory.name, 'saved_blocks': index + 1,
                                      'total_blocks': len(items)}), flush=True)
    summary = {'geometry_lower_m': np.asarray(raw_lower).tolist(), 'geometry_upper_m': np.asarray(raw_upper).tolist(),
               'grid_lower_m': lower.tolist(), 'grid_upper_m': (lower + cells * spacing).tolist(),
               'grid_cells': cells.tolist(), 'candidate_blocks': int(np.prod(counts)),
               'retained_blocks': len(records), 'skipped_blocks': int((~plan['retained']).sum()),
               'sampled_nodes': plan['sampled_nodes'], 'block_cells': block_cells,
               'workers': workers,
               'certificate_file': 'block_certificates.npz',
               'certificate_sha256': sha256(directory / 'block_certificates.npz'),
               'certificate': '|phi(center)| > Euclidean block half diagonal + floating tolerance; 1-Lipschitz field',
               'thin_feature_warning': 'retained block corner samples may miss sub-voxel material or cavities',
               'boundary_handling': '8 mm geometry-derived padding; no closing, clipping, or artificial caps'}
    return records, summary


def decode_block(path):
    """Marching cubes of one saved XYZ field, including zero at block boundaries."""
    from skimage.measure import marching_cubes
    with np.load(path, allow_pickle=False) as saved:
        if str(saved['axis_order']) != 'xyz' or str(saved['units']) != 'm':
            raise ValueError('unsupported saved field convention')
        values = np.array(saved['values'], np.float32)
        lower = np.array(saved['lower_m'], np.float64)
        h = float(saved['spacing_m'])
    if values.ndim != 3 or min(values.shape) < 2 or lower.shape != (3,) or not np.isfinite(h) or h <= 0:
        raise ValueError('invalid saved grid shape or transform')
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(lower)):
        raise ValueError('non-finite saved field')
    # Lewiner classifies zero with the nonpositive side. Including min == 0 is
    # essential when a surface lies exactly on a shared block boundary.
    if values.min() > 0 or values.max() <= 0:
        return np.empty((0, 3), float), np.empty((0, 3), np.int32)
    try:
        vertices, faces, _, _ = marching_cubes(values, level=0., spacing=(h,) * 3,
                                               gradient_direction='ascent', method='lewiner',
                                               allow_degenerate=False)
    except RuntimeError as error:
        if 'No surface found' not in str(error):
            raise
        return np.empty((0, 3), float), np.empty((0, 3), np.int32)
    return vertices.astype(np.float64) + lower, faces.astype(np.int32)


def decode_saved_fields(block_paths, output_mesh=None):
    """Only saved blocks are inputs. Return float64 vertices and int32 faces.

    Cells tile without overlap. Per-block vertex indices remain separate; this
    does not create extra surface area, but this is not a welded topology mesh.
    """
    paths = [Path(p) for p in block_paths]
    if not paths or len(set(map(str, paths))) != len(paths):
        raise ValueError('require a nonempty unique list of saved block paths')
    vertices, faces, nvertices, nfaces = [], [], 0, 0
    for path in paths:
        v, f = decode_block(path)
        if len(f):
            nfaces += len(f)
            if nfaces > MAX_FACES:
                raise MemoryError(f'decoded faces exceed {MAX_FACES}; no simplification applied')
            vertices.append(v)
            faces.append(f + nvertices)
            nvertices += len(v)
    v = np.concatenate(vertices) if vertices else np.empty((0, 3), np.float64)
    f = np.concatenate(faces) if faces else np.empty((0, 3), np.int32)
    if output_mesh is not None:
        with Path(output_mesh).open('xb') as stream:
            np.savez_compressed(stream, vertices_m=v, faces=f)
    return v, f


def mesh_summary(vertices, faces):
    area = 0.
    for first in range(0, len(faces), 250000):
        triangle = vertices[faces[first:first + 250000]]
        area += float(np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1).sum() / 2)
    return {'vertices': len(vertices), 'faces': len(faces), 'surface_area_m2': area}


def validate_inference_manifest(manifest):
    if manifest.get('status') != 'complete':
        raise ValueError('inference manifest must be complete')
    predictions = manifest.get('predictions', [])
    if len(predictions) != 3 or {x['name'] for x in predictions} != EXPECTED_NAMES:
        raise ValueError('requires exactly three unique declared RGB snapshots')
    for group in ('sources', 'inputs', 'prepared', 'artifacts'):
        flags = manifest.get('integrity', {}).get(group)
        if not isinstance(flags, list) or not flags or not all(x.get('unchanged') is True for x in flags):
            raise ValueError(f'inference integrity group {group} is absent or failed')
    if manifest.get('integrity', {}).get('settings') is not True:
        raise ValueError('inference settings integrity absent or failed')
    if any(x.get('status') != 'valid' for x in predictions):
        raise ValueError('inference includes an invalid field response')


def frozen_core(path):
    spec = importlib.util.spec_from_file_location('astra_build_frozen_core', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_run(args):
    inference = Path(args.inferences).resolve()
    inference_manifest = json.loads(inference.read_text())
    validate_inference_manifest(inference_manifest)
    core_records = [x for x in inference_manifest.get('sources', [])
                    if Path(x['path']).name == 'field_csg.py']
    if len(core_records) != 1 or core_records[0]['sha256'] != sha256(HERE / 'field_csg.py'):
        raise ValueError('CSG runtime must exactly match the source frozen before inference')
    frozen_inference_core = checked_file(inference.parent, {'file': core_records[0]['copy'],
                                                          'sha256': core_records[0]['copy_sha256']})
    if sha256(frozen_inference_core) != core_records[0]['sha256']:
        raise ValueError('inference CSG source copy differs from its declared original')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for subdir in ('source', 'inputs', 'fields', 'geometry'):
        (output / subdir).mkdir()
    settings = {'schema_version': 1, 'stage': 'astra_implicit_field_to_saved_scalar_grid_to_mesh',
                'spacings_m': args.spacings, 'grid_alignment_m': ALIGNMENT_M,
                'geometry_padding_m': 2 * ALIGNMENT_M, 'block_cells': BLOCK_CELLS,
                'workers': getattr(args, 'workers', 4),
                'limits': {'candidate_blocks': MAX_CANDIDATES, 'sampled_nodes': MAX_SAMPLED_NODES,
                           'extent_per_axis_m': MAX_EXTENT_M, 'bounds_volume_m3': MAX_VOLUME_M3, 'decoded_faces': MAX_FACES},
                'field_input': 'Astra returned CSG geometry only; no RGB/GT/depth model input in sampling',
                'decoder_input': 'disk-loaded numeric field blocks only',
                'representation': 'camera-CV metre coordinates; negative material; CSG 1-Lipschitz implicit field, not generally exact Euclidean SDF',
                'limitation': 'sampling may lose thin features; no watertightness or completeness claim',
                'billing': inference_manifest.get('billing'), 'model': inference_manifest.get('model'),
                'reasoning_effort': inference_manifest.get('reasoning_effort')}
    write_json(output / 'settings.json', settings)
    sources = []
    for path in (Path(__file__).resolve(), HERE / 'field_csg.py'):
        destination = output / 'source' / path.name
        shutil.copy2(path, destination)
        sources.append({'path': str(path), 'sha256': sha256(path), 'copy': str(destination.relative_to(output)), 'copy_sha256': sha256(destination)})
    shutil.copy2(inference, output / 'inputs' / 'inference_manifest.json')
    inputs = [{'path': str(inference), 'sha256': sha256(inference), 'copy': 'inputs/inference_manifest.json',
               'copy_sha256': sha256(output / 'inputs' / 'inference_manifest.json')},
              {'path': str(frozen_inference_core), 'sha256': sha256(frozen_inference_core),
               'role': 'CSG source frozen before inference'}]
    manifest = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'inference_manifest': str(inference), 'settings': settings, 'settings_sha256': sha256(output / 'settings.json'),
                'sources': sources, 'inputs': inputs, 'snapshots': [], 'configurations': []}
    write_json(output / 'manifest.json', manifest)
    started = time.perf_counter()
    try:
        core = frozen_core(output / 'source' / 'field_csg.py')
        for prediction in inference_manifest['predictions']:
            name = prediction['name']
            # These are hashed for provenance only. Their image/text contents
            # are never supplied to the field sampler or saved-only decoder.
            for key in ('raw_response', 'raw_field', 'rgb'):
                if key in prediction:
                    provenance_path = checked_file(inference.parent, prediction[key])
                    inputs.append({'path': str(provenance_path), 'sha256': prediction[key]['sha256'],
                                   'role': f'{name}:{key}'})
            source = checked_file(inference.parent, prediction['field'])
            destination = output / 'inputs' / f'{name}.json'
            shutil.copy2(source, destination)
            inputs.append({'path': str(source), 'sha256': sha256(source), 'copy': str(destination.relative_to(output)), 'copy_sha256': sha256(destination)})
            field = json.loads(destination.read_text())
            validated = core.validate_field(field)
            if validated is not None:
                field = validated
            lower, upper = core.field_bounds(field)
            snapshot = {'name': name, 'field': {'file': str(destination.relative_to(output)), 'sha256': sha256(destination)},
                        'geometry_lower_m': np.asarray(lower).tolist(), 'geometry_upper_m': np.asarray(upper).tolist(),
                        'model_response_provenance': {key: prediction[key] for key in ('raw_response', 'raw_field', 'rgb') if key in prediction}}
            manifest['snapshots'].append(snapshot)
            for h in args.spacings:
                config = {'name': f'{name}_h{h * 1000:g}mm', 'snapshot': name, 'spacing_m': h, 'status': 'running', 'fields': []}
                manifest['configurations'].append(config)
                write_json(output / 'manifest.json', manifest)
                config_start = time.perf_counter()
                try:
                    directory = output / 'fields' / config['name']
                    records, plan = save_field_blocks(lambda p: core.evaluate_field(field, p), lower, upper, h, directory,
                                                      workers=getattr(args, 'workers', 4), worker_spec=field,
                                                      worker_core=output / 'source' / 'field_csg.py')
                    config['allocation'] = plan
                    config['fields'] = [dict(r, file=str((directory / r['file']).relative_to(output))) for r in records]
                    geometry = output / 'geometry' / f'{config["name"]}_decoded_field.npz'
                    vertices, faces = decode_saved_fields([output / r['file'] for r in config['fields']], geometry)
                    config['decoded_mesh'] = {'file': str(geometry.relative_to(output)), 'sha256': sha256(geometry), **mesh_summary(vertices, faces)}
                    config['status'] = 'ok' if len(faces) else 'empty_surface'
                except Exception as error:
                    config.update(status='failed', error=repr(error), traceback=traceback.format_exc())
                config['elapsed_s'] = time.perf_counter() - config_start
                write_json(output / 'manifest.json', manifest)
                print(json.dumps({key: config[key] for key in ('name', 'status', 'elapsed_s')}), flush=True)
        manifest['status'] = 'complete' if all(x['status'] == 'ok' for x in manifest['configurations']) else 'failed'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        manifest['elapsed_s'] = time.perf_counter() - started
        artifacts = []
        for config in manifest['configurations']:
            artifacts.extend(config['fields'])
            if 'decoded_mesh' in config:
                artifacts.append(config['decoded_mesh'])
            if 'allocation' in config:
                plan = config['allocation']
                artifacts.append({'file': f'fields/{config["name"]}/{plan["certificate_file"]}', 'sha256': plan['certificate_sha256']})
        manifest['integrity'] = {
            'sources': [{'path': x['path'], 'unchanged': sha256(x['path']) == x['sha256'] and sha256(output / x['copy']) == x['copy_sha256']} for x in sources],
            'inputs': [{'path': x['path'], 'unchanged': sha256(x['path']) == x['sha256'] and
                        ('copy' not in x or sha256(output / x['copy']) == x['copy_sha256'])} for x in inputs],
            'artifacts': [{'path': x['file'], 'unchanged': sha256(output / x['file']) == x['sha256']} for x in artifacts],
            'settings': sha256(output / 'settings.json') == manifest['settings_sha256']}
        if not all(x['unchanged'] for group in ('sources', 'inputs', 'artifacts') for x in manifest['integrity'][group]) or not manifest['integrity']['settings']:
            manifest['status'] = 'integrity_failed'
        write_json(output / 'manifest.json', manifest)
    return 0 if manifest['status'] == 'complete' else 1


def decode_run(args):
    directory, output = Path(args.fields).resolve(), Path(args.output).resolve()
    paths = sorted(directory.glob('block_[0-9]*.npz'))
    if not paths:
        raise ValueError('no saved scalar blocks found')
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    shutil.copy2(source, output / 'build_fields_used.py')
    hashes = {str(path): sha256(path) for path in [source, *paths]}
    manifest = {'status': 'running', 'inputs': hashes, 'decoder_input': 'saved scalar blocks only',
                'model_spec_image_depth_or_ground_truth_inputs': []}
    started = time.perf_counter()
    try:
        target = output / 'decoded_surface.npz'
        vertices, faces = decode_saved_fields(paths, target)
        manifest['decoded_mesh'] = {'file': target.name, 'sha256': sha256(target), **mesh_summary(vertices, faces)}
        manifest['inputs_unchanged'] = {path: sha256(path) == digest for path, digest in hashes.items()}
        manifest['source_copy_unchanged'] = sha256(output / 'build_fields_used.py') == hashes[str(source)]
        manifest['status'] = 'complete' if all(manifest['inputs_unchanged'].values()) and manifest['source_copy_unchanged'] else 'integrity_failed'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest['elapsed_s'] = time.perf_counter() - started
        write_json(output / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'decoded_mesh': manifest['decoded_mesh']}), flush=True)
    return 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--inferences', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--spacings', nargs='+', type=float, default=[.004, .002])
    build.add_argument('--workers', type=int, default=4)
    build.set_defaults(function=build_run)
    decode = commands.add_parser('decode')
    decode.add_argument('--fields', required=True)
    decode.add_argument('--output', required=True)
    decode.set_defaults(function=decode_run)
    args = parser.parse_args()
    raise SystemExit(args.function(args))
