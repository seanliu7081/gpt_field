"""Reconstruct geometry using only saved field blocks and a frozen decoder.

This command has no image, depth prediction, model, or ground-truth arguments.
The --runtime file should be field_rgb.py from the build run's source folder.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import trimesh


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    directory, runtime, output = (Path(x).resolve() for x in
                                  (args.fields, args.runtime, args.output))
    blocks = sorted(directory.glob('*.npz'))
    if not blocks or not runtime.is_file():
        raise ValueError('Require saved field NPZ blocks and a frozen runtime file')
    output.mkdir(parents=True, exist_ok=False)
    runtime_copy = output / 'field_rgb_used.py'
    shutil.copy2(runtime, runtime_copy)
    source = Path(__file__).resolve()
    shutil.copy2(source, output / 'decode_saved_fields_used.py')
    hashes = {str(p): sha256(p) for p in [runtime, source, *blocks]}
    started = time.perf_counter()
    manifest = {'status': 'running', 'input_sha256': hashes,
                'decoder_inputs': 'saved field arrays, masks, grid transforms and clip planes only',
                'image_depth_model_or_gt_inputs': [], 'blocks': []}
    try:
        spec = importlib.util.spec_from_file_location('saved_field_runtime', runtime_copy)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        parts = []
        for path in blocks:
            field = module.ProjectiveField.load(path)
            mesh = field.decode()
            manifest['blocks'].append({'file': str(path), 'faces': len(mesh.faces)})
            if len(mesh.faces):
                parts.append(mesh)
        result = trimesh.util.concatenate(parts) if parts else trimesh.Trimesh(
            vertices=np.empty((0, 3)), faces=np.empty((0, 3), int), process=False)
        target = output / 'decoded_surface.npz'
        with target.open('xb') as stream:
            np.savez_compressed(stream, vertices_m=np.asarray(result.vertices, np.float64),
                                faces=np.asarray(result.faces, np.int32))
        manifest['surface'] = {'file': target.name, 'sha256': sha256(target),
                               'vertices': len(result.vertices), 'faces': len(result.faces),
                               'area_m2': float(result.area)}
        manifest['inputs_unchanged'] = {path: sha256(path) == digest
                                         for path, digest in hashes.items()}
        manifest['runtime_copy_unchanged'] = sha256(runtime_copy) == hashes[str(runtime)]
        manifest['command_copy_unchanged'] = sha256(output / 'decode_saved_fields_used.py') == hashes[str(source)]
        complete = (all(manifest['inputs_unchanged'].values())
                    and manifest['runtime_copy_unchanged'] and manifest['command_copy_unchanged'])
        manifest['status'] = 'complete' if complete else 'integrity_failed'
    except BaseException as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        manifest['elapsed_s'] = time.perf_counter() - started
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': manifest['status'], 'surface': manifest['surface'],
                      'elapsed_s': manifest['elapsed_s']}), flush=True)
    return 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', required=True)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--output', required=True)
    raise SystemExit(run(parser.parse_args()))
