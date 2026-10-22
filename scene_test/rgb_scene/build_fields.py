"""Prediction-only field build, saved-only decode and representation-loss test.

No reference argument and no ground-truth geometry/depth/calibration/masks are
loaded. A/C evaluation is a separate executable after this stage completes.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import trimesh
from field_rgb import (ProjectiveField, TAU_M, JUMP_ABS_M, JUMP_REL, TILES, SLAB_M,
                       plan_blocks, block_shape, visible_mesh)
from rgb_metrics import (sha256, write_json, sample_mesh, compare_meshes, save_mesh, load_mesh)

HERE = Path(__file__).resolve().parent


def checked_file(root, item, key='file'):
    path = (root/item[key]).resolve()
    if not path.is_file() or sha256(path) != item['sha256']:
        raise ValueError(f'input hash mismatch: {path}')
    return path


def allocation_plan(blocks, spacings, max_vertices):
    result = []
    for h in spacings:
        counts = np.array([np.prod(block_shape(b, h)) for b in blocks], np.int64)
        cells = np.array([np.prod(np.array(block_shape(b, h))-1) for b in blocks], np.int64)
        result.append({'spacing_m': h, 'blocks': len(blocks), 'sum_vertices': int(counts.sum()),
                       'maximum_block_vertices': int(counts.max()) if len(counts) else 0,
                       'decompressed_field_array_bytes': int((5*counts+cells).sum()),
                       'maximum_build_array_bytes_estimate': int(counts.max()*22) if len(counts) else 0,
                       'within_per_block_budget': bool(np.all(counts <= max_vertices))})
    return result


def validate_prediction_manifest(manifest):
    if manifest.get('status') != 'complete':
        raise ValueError('prediction manifest is not complete')
    for group in ('sources_unchanged', 'model_files_unchanged', 'frozen_artifacts_unchanged', 'inputs_unchanged', 'predictions_unchanged'):
        flags = manifest.get(group)
        if not isinstance(flags, dict) or not flags or not all(value is True for value in flags.values()):
            raise ValueError(f'prediction provenance group {group} is absent/empty/failed')
    names = [item['name'] for item in manifest['predictions']]
    expected = {'bowl002_initial', 'drawer003_closed_step60', 'drawer003_open_step552'}
    if len(names) != 3 or set(names) != expected:
        raise ValueError('requires exactly the three unique declared RGB snapshots')


def build_one(depth, K, block, h, max_vertices, directory, output):
    field = ProjectiveField.build(depth, K, block, h, max_vertices)
    path = directory/f'{block["id"]}.npz'
    field.save(path)
    record = {'file': str(path.relative_to(output)), 'sha256': sha256(path), 'shape': list(field.values.shape),
              'observed_vertices': int(field.observed.sum()), 'valid_cells': int(field.cell_valid.sum()),
              'vertex_quantization_max_m': field.metadata['vertex_quantization_max_m']}
    del field
    field = ProjectiveField.load(path)
    mesh = field.decode()
    record['decoded_faces'] = len(mesh.faces)
    return record, mesh


_WORKER_ARGS = None


def initialize_worker(depth, K, h, max_vertices, directory, output):
    global _WORKER_ARGS
    depth.setflags(write=False)
    K.setflags(write=False)
    _WORKER_ARGS = (depth, K, h, max_vertices, directory, output)


def build_worker(block):
    depth, K, h, max_vertices, directory, output = _WORKER_ARGS
    return build_one(depth, K, block, h, max_vertices, directory, output)


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name in ('source', 'inputs', 'samples', 'geometry', 'fields', 'distances'):
        (out/name).mkdir()
    manifest_path = Path(args.predictions).resolve()
    prediction_manifest = json.loads(manifest_path.read_text())
    validate_prediction_manifest(prediction_manifest)
    settings = {'schema_version': 1, 'experiment': 'single_rgb_to_saved_projective_field_to_visible_geometry',
                'stage': 'prediction_only_build_decode_representation_loss',
                'spacings_m': args.spacings, 'surface_samples_per_direction': args.surface_samples,
                'tau_m': TAU_M, 'image_tiles_per_axis': TILES, 'camera_z_slab_m': SLAB_M,
                'grid_alignment_m': .004, 'depth_jump_abs_m': JUMP_ABS_M, 'depth_jump_relative': JUMP_REL,
                'max_block_vertices': args.max_block_vertices, 'workers': args.workers, 'seed': args.seed,
                'model_input': 'single RGB only; depth and focal are model predictions',
                'field_input': 'predicted depth_m and K_pred only; no GT/pose/semantics',
                'decoder_input': 'saved phi/observed/cell_valid/grid transform/core clip planes only',
                'surface_sampling': 'area-uniform on continuous triangles; same source points for all spacings',
                'unknown': 'unsupported pixels, camera-frustum exterior and behind-band remain unknown',
                'limitation': 'open visible surfaces; projective axial residual is not a Euclidean SDF or complete solid',
                'test_mode': 'preflight_only' if args.plan_only else ('formal' if args.surface_samples >= 100000 and args.spacings == [.004, .002, .001] and not args.snapshot else 'smoke')}
    write_json(out/'settings.json', settings)
    sources = [HERE/name for name in ('field_rgb.py', 'rgb_metrics.py', 'build_fields.py')]
    sources += [HERE.parent/name for name in ('reconstruct.py', 'mesh_queries.py', 'sdf_grid.py')]
    source_records = []
    for path in sources:
        copy = out/'source'/path.name
        shutil.copy2(path, copy)
        source_records.append({'path': str(path), 'sha256': sha256(path), 'copy': str(copy.relative_to(out)), 'copy_sha256': sha256(copy)})
    shutil.copy2(manifest_path, out/'inputs'/'prediction_manifest.json')
    inputs = [{'path': str(manifest_path), 'sha256': sha256(manifest_path), 'copy': 'inputs/prediction_manifest.json', 'copy_sha256': sha256(out/'inputs'/'prediction_manifest.json')}]
    manifest = {'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'prediction_manifest': str(manifest_path), 'settings_sha256': sha256(out/'settings.json'),
                'sources': source_records, 'inputs': inputs, 'configurations': []}
    metrics = {'status': 'running', 'settings': settings, 'snapshots': [], 'configurations': []}
    write_json(out/'manifest.json', manifest)
    write_json(out/'metrics.json', metrics)
    start = time.perf_counter()
    try:
        records = [r for r in prediction_manifest['predictions'] if not args.snapshot or r['name'] in args.snapshot]
        if not records:
            raise ValueError('no selected prediction')
        for scene_index, record in enumerate(records):
            name = record['name']
            source = checked_file(manifest_path.parent, record)
            if record.get('depth_units') != 'm' or record.get('depth_convention') != 'camera_z':
                raise ValueError('requires metric camera-Z predictions')
            copy = out/'inputs'/f'{name}_prediction.npz'
            shutil.copy2(source, copy)
            inputs.append({'path': str(source), 'sha256': sha256(source), 'copy': str(copy.relative_to(out)), 'copy_sha256': sha256(copy)})
            with np.load(copy, allow_pickle=False) as data:
                depth, K = data['depth_m'].copy(), data['K_pred'].copy()
            blocks = plan_blocks(depth, K)
            plans = allocation_plan(blocks, args.spacings, args.max_block_vertices)
            snapshot = {'name': name, 'prediction_sha256': sha256(copy), 'K_pred': K.tolist(),
                        'image_shape': list(depth.shape), 'blocks': blocks, 'allocation_plan': plans}
            metrics['snapshots'].append(snapshot)
            write_json(out/'metrics.json', metrics)
            print(json.dumps({'preflight': name, 'allocation_plan': plans}), flush=True)
            if not all(p['within_per_block_budget'] for p in plans):
                raise MemoryError(f'{name} allocation preflight exceeds per-block budget; no configuration skipped')
            if args.plan_only:
                continue
            pred_mesh, mesh_info = visible_mesh(depth, K)
            if not len(pred_mesh.faces):
                raise ValueError(f'{name} empty predicted visible surface')
            pred_mesh_path = out/'geometry'/f'{name}_predicted_visible.npz'
            save_mesh(pred_mesh_path, pred_mesh)
            pred_points = sample_mesh(pred_mesh, args.surface_samples, np.random.default_rng(args.seed+scene_index*100))
            samples_path = out/'samples'/f'{name}_prediction_surface.npz'
            with samples_path.open('xb') as file:
                np.savez_compressed(file, points_m=pred_points)
            snapshot.update(predicted_surface=mesh_info, source_mesh={'file': str(pred_mesh_path.relative_to(out)), 'sha256': sha256(pred_mesh_path)},
                            source_samples={'file': str(samples_path.relative_to(out)), 'sha256': sha256(samples_path)})
            for h_index, h in enumerate(args.spacings):
                config = {'name': f'{name}_h{h*1000:g}mm', 'snapshot': name, 'spacing_m': h, 'status': 'running', 'fields': []}
                metrics['configurations'].append(config)
                config_start = time.perf_counter()
                write_json(out/'metrics.json', metrics)
                try:
                    directory = out/'fields'/config['name']
                    directory.mkdir()
                    parts = []
                    # Independent blocks; deterministic map order preserves output
                    # geometry ordering. Each worker writes a distinct NEW file.
                    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
                                             initargs=(depth, K, h, args.max_block_vertices, directory, out)) as pool:
                        results = pool.map(build_worker, blocks, chunksize=1)
                        for index, (field_record, part) in enumerate(results):
                            config['fields'].append(field_record)
                            if len(part.faces):
                                parts.append(part)
                            if (index+1) % 10 == 0 or index+1 == len(blocks):
                                print(json.dumps({'configuration': config['name'], 'blocks_done': index+1, 'blocks_total': len(blocks), 'elapsed_s': time.perf_counter()-config_start}), flush=True)
                                write_json(out/'metrics.json', metrics)
                    mesh = trimesh.util.concatenate(parts) if parts else trimesh.Trimesh(vertices=np.empty((0,3)), faces=np.empty((0,3),int), process=False)
                    del parts
                    path = out/'geometry'/f'{config["name"]}_decoded_field.npz'
                    save_mesh(path, mesh)
                    del mesh
                    mesh = load_mesh(path)
                    config['decoded_mesh'] = {'file': str(path.relative_to(out)), 'sha256': sha256(path), 'faces': len(mesh.faces), 'surface_area_m2': float(mesh.area)}
                    score, distances = compare_meshes(mesh, pred_mesh, pred_points, args.surface_samples, np.random.default_rng(args.seed+scene_index*100+h_index+1))
                    config['representation_B'] = score
                    distance_path = out/'distances'/f'{config["name"]}_representation.npz'
                    with distance_path.open('xb') as file:
                        np.savez_compressed(file, **distances)
                    config['distances'] = {'file': str(distance_path.relative_to(out)), 'sha256': sha256(distance_path)}
                    config['status'] = 'ok' if len(mesh.faces) else 'empty_reconstruction'
                    del mesh
                except Exception:
                    config['status'] = 'failed'
                    config['error'] = traceback.format_exc()
                config['elapsed_s'] = time.perf_counter()-config_start
                write_json(out/'metrics.json', metrics)
                print(json.dumps({'configuration': config['name'], 'status': config['status'], 'elapsed_s': config['elapsed_s'],
                                  'representation_B': config.get('representation_B')}), flush=True)
        expected = len(records)*len(args.spacings)
        status = 'planned' if args.plan_only else ('complete' if len(metrics['configurations']) == expected and all(c['status'] == 'ok' for c in metrics['configurations']) else 'failed')
    except BaseException:
        status = 'failed'
        manifest['error'] = traceback.format_exc()
    integrity = {'sources': [sha256(Path(r['path'])) == r['sha256'] and sha256(out/r['copy']) == r['copy_sha256'] == r['sha256'] for r in source_records],
                 'inputs': [sha256(Path(r['path'])) == r['sha256'] and (not r.get('copy') or sha256(out/r['copy']) == r['copy_sha256'] == r['sha256']) for r in inputs],
                 'settings': sha256(out/'settings.json') == manifest['settings_sha256']}
    artifacts = []
    for snapshot in metrics['snapshots']:
        artifacts.extend(snapshot[key] for key in ('source_mesh', 'source_samples') if key in snapshot)
    for config in metrics['configurations']:
        artifacts.extend(config['fields'])
        artifacts.extend(config[key] for key in ('decoded_mesh', 'distances') if key in config)
    integrity['artifacts'] = [sha256(out/r['file']) == r['sha256'] for r in artifacts]
    if not all(integrity['sources']+integrity['inputs']+integrity['artifacts']+[integrity['settings']]):
        status = 'integrity_failed'
    manifest.update(status=status, elapsed_s=time.perf_counter()-start, integrity=integrity,
                    configurations=[{'name': c['name'], 'status': c['status']} for c in metrics['configurations']])
    metrics['status'] = status
    write_json(out/'metrics.json', metrics)
    manifest['metrics_sha256'] = sha256(out/'metrics.json')
    write_json(out/'manifest.json', manifest)
    return 0 if status in ('complete', 'planned') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--spacings', nargs='+', type=float, default=[.004, .002, .001])
    parser.add_argument('--surface-samples', type=int, default=100000)
    parser.add_argument('--max-block-vertices', type=int, default=50000000)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--snapshot', nargs='+')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.workers <= 0 or args.workers > 4 or args.surface_samples <= 0 or any(h not in (.004, .002, .001) for h in args.spacings):
        parser.error('positive sample count and spacings among .004/.002/.001 required')
    raise SystemExit(run(args))

if __name__ == '__main__':
    main()
