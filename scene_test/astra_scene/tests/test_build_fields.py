"""Geometry and saved-only contracts, independent of any Astra request."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build_fields as build


def save_raw(path, values, lower, spacing):
    np.savez_compressed(path, values=np.asarray(values, np.float32), lower_m=np.asarray(lower, float),
                        spacing_m=spacing, axis_order='xyz', units='m')


def area(vertices, faces):
    return build.mesh_summary(vertices, faces)['surface_area_m2']


def sphere_fixture(tmp_path, block_cells=8):
    center, radius, h = np.array([.003, .001, .051]), .029, .004
    evaluate = lambda p: np.linalg.norm(p - center, axis=1) - radius
    records, summary = build.save_field_blocks(evaluate, center-radius, center+radius, h,
                                              tmp_path / 'blocks', block_cells=block_cells)
    paths = [tmp_path / 'blocks' / item['file'] for item in records]
    return paths, summary, evaluate


def test_tiled_sphere_matches_monolithic(tmp_path):
    paths, summary, evaluate = sphere_fixture(tmp_path)
    tiled_v, tiled_f = build.decode_saved_fields(paths)
    lower, cells = np.asarray(summary['grid_lower_m']), np.asarray(summary['grid_cells'])
    indices = np.stack(np.meshgrid(*(np.arange(n+1) for n in cells), indexing='ij'), axis=-1)
    values = evaluate(lower + indices.reshape(-1, 3) * .004).reshape(tuple(cells+1))
    source = tmp_path / 'monolithic.npz'
    save_raw(source, values, lower, .004)
    mono_v, mono_f = build.decode_block(source)
    assert len(tiled_f) == len(mono_f)
    assert area(tiled_v, tiled_f) == pytest.approx(area(mono_v, mono_f), rel=2e-7)
    # Matching triangle centres checks missing or double-counted seam faces.
    a, b = tiled_v[tiled_f].mean(1), mono_v[mono_f].mean(1)
    order = lambda x: np.lexsort(np.round(x, 8).T[::-1])
    np.testing.assert_allclose(a[order(a)], b[order(b)], atol=1e-8, rtol=0)
    assert summary['skipped_blocks'] > 0


def test_plane_exactly_on_block_boundary_no_hole_or_double_area(tmp_path):
    h = .004
    paths = []
    for i, z in enumerate((-.008, 0.)):
        values = np.broadcast_to(z + np.arange(3) * h, (3, 3, 3))
        path = tmp_path / f'block_{i:04d}.npz'
        save_raw(path, values, [-.004, -.004, z], h)
        paths.append(path)
    vertices, faces = build.decode_saved_fields(paths)
    assert len(faces) == 8
    np.testing.assert_allclose(vertices[:, 2], 0., atol=1e-14)
    assert area(vertices, faces) == pytest.approx(.008**2, rel=1e-6)
    all_values = np.broadcast_to(-.008 + np.arange(5)*h, (3, 3, 5))
    save_raw(tmp_path/'full.npz', all_values, [-.004, -.004, -.008], h)
    full_v, full_f = build.decode_block(tmp_path/'full.npz')
    assert len(full_f) == len(faces)
    assert area(full_v, full_f) == pytest.approx(area(vertices, faces), rel=1e-7)


def test_lipschitz_skip_keeps_all_sphere_surface_blocks(tmp_path):
    paths, summary, evaluate = sphere_fixture(tmp_path, block_cells=4)
    with np.load(tmp_path/'blocks'/'block_certificates.npz', allow_pickle=False) as saved:
        starts, shapes = saved['start_cells'], saved['cell_shapes']
        center_values, radii, retained = saved['center_values_m'], saved['half_diagonal_m'], saved['retained']
        assert np.all(np.abs(center_values[~retained]) > radii[~retained])
        rng = np.random.default_rng(11)
        unit = rng.normal(size=(5000, 3))
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        surface = np.array([.003, .001, .051]) + .029*unit
        block_ids = np.floor((surface - saved['lower_m']) / (.004*4)).astype(int)
        kept = {tuple(row) for row in saved['block_ids'][retained]}
        assert all(tuple(row) in kept for row in block_ids)
        # The numerical certificate is independently checked at random interior
        # points; every skipped block keeps a single sign.
        for index in np.flatnonzero(~retained):
            p = saved['lower_m'] + (starts[index] + rng.uniform(size=(50, 3))*shapes[index]) * .004
            assert np.all(evaluate(p) * center_values[index] > 0)
    assert paths and summary['sampled_nodes'] > 0


def test_saved_only_decoder_subprocess_no_csg_or_spec_available(tmp_path):
    paths, _, _ = sphere_fixture(tmp_path)
    expected_v, expected_f = build.decode_saved_fields(paths)
    isolated = tmp_path/'isolated'
    isolated.mkdir()
    script = isolated/'build_fields.py'
    shutil.copy2(build.__file__, script)
    completed = subprocess.run([sys.executable, '-B', str(script), 'decode',
                                '--fields', str(tmp_path/'blocks'), '--output', str(isolated/'result')],
                               cwd=isolated, text=True, capture_output=True, check=True)
    assert 'complete' in completed.stdout
    assert not (isolated/'field_csg.py').exists()
    with np.load(isolated/'result'/'decoded_surface.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['vertices_m'], expected_v)
        np.testing.assert_array_equal(saved['faces'], expected_f)
    manifest = json.loads((isolated/'result'/'manifest.json').read_text())
    assert manifest['model_spec_image_depth_or_ground_truth_inputs'] == []
    assert all(manifest['inputs_unchanged'].values())


def test_node_values_are_actual_float32_samples(tmp_path):
    paths, _, evaluate = sphere_fixture(tmp_path)
    with np.load(paths[0], allow_pickle=False) as saved:
        shape = saved['values'].shape
        indices = np.stack(np.meshgrid(*(np.arange(n) for n in shape), indexing='ij'), axis=-1)
        points = saved['lower_m'] + indices.reshape(-1, 3) * float(saved['spacing_m'])
        actual = evaluate(points).astype(np.float32).reshape(shape)
        np.testing.assert_array_equal(saved['values'], actual)
        assert saved['values'].dtype == np.float32


def test_all_positive_and_all_negative_decode_empty(tmp_path):
    for sign in (-1, 1):
        path = tmp_path/f'{sign}.npz'
        save_raw(path, np.full((3, 3, 3), sign), [0, 0, 0], .004)
        v, f = build.decode_block(path)
        assert v.shape == f.shape == (0, 3)


def test_resource_caps_and_bad_spacing_fail_explicitly(monkeypatch):
    with pytest.raises(ValueError, match='spacing'):
        build.grid_domain([0]*3, [.1]*3, .003)
    with pytest.raises(ValueError, match='bounds budget'):
        build.grid_domain([0]*3, [9, .1, .1], .004)
    monkeypatch.setattr(build, 'MAX_CANDIDATES', 1)
    with pytest.raises(MemoryError, match='no automatic coarsening'):
        build.grid_domain([0]*3, [1]*3, .004)
    monkeypatch.setattr(build, 'MAX_SAMPLED_NODES', 1)
    with pytest.raises(MemoryError, match='no automatic coarsening'):
        build.plan_blocks(lambda p: np.zeros(len(p)), np.zeros(3), np.array([2]*3), .004)


def test_nonfinite_saved_values_rejected(tmp_path):
    path = tmp_path/'bad.npz'
    save_raw(path, np.full((2, 2, 2), np.nan), [0]*3, .004)
    with pytest.raises(ValueError, match='non-finite'):
        build.decode_block(path)


def test_output_refuses_overwrite(tmp_path):
    paths, _, _ = sphere_fixture(tmp_path)
    target = tmp_path/'mesh.npz'
    build.decode_saved_fields(paths, target)
    before = build.sha256(target)
    with pytest.raises(FileExistsError):
        build.decode_saved_fields(paths, target)
    assert build.sha256(target) == before


def test_inference_manifest_requires_all_nonempty_integrity_groups():
    manifest = {'status': 'complete', 'predictions': [
        {'name': n, 'status': 'valid'} for n in sorted(build.EXPECTED_NAMES)],
        'integrity': {k: [{'unchanged': True}] for k in ('sources', 'inputs', 'prepared', 'artifacts')}}
    manifest['integrity']['settings'] = True
    build.validate_inference_manifest(manifest)
    manifest['integrity']['artifacts'] = []
    with pytest.raises(ValueError, match='artifacts'):
        build.validate_inference_manifest(manifest)


def test_full_offline_build_copies_sources_binds_inputs_and_decodes(tmp_path):
    import field_csg
    inference = tmp_path/'inference'
    inference.mkdir()
    spec = {'schema_version': 1, 'units': 'm', 'coordinate_frame': field_csg.FRAME,
            'estimated_focal_px': 500., 'root': 0, 'uncertainty_notes': 'synthetic test fixture',
            'nodes': [{'name': 'fixture', 'op': 'sphere', 'center_m': [0., 0., .12],
                       'rotation_xyzw': [0., 0., 0., 1.], 'size_m': [.012, 0., 0.],
                       'left': None, 'right': None}]}
    manifest = {'schema_version': 1, 'status': 'complete', 'billing': 'offline_mock',
                'model': 'no_model_called', 'reasoning_effort': None, 'predictions': [],
                'integrity': {k: [{'unchanged': True}] for k in ('sources', 'inputs', 'prepared', 'artifacts')}}
    manifest['integrity']['settings'] = True
    core_source = Path(field_csg.__file__)
    shutil.copy2(core_source, inference/'field_csg_frozen.py')
    manifest['sources'] = [{'path': str(core_source), 'sha256': build.sha256(core_source),
                            'copy': 'field_csg_frozen.py', 'copy_sha256': build.sha256(inference/'field_csg_frozen.py')}]
    for name in sorted(build.EXPECTED_NAMES):
        path = inference/f'{name}.json'
        path.write_text(json.dumps(spec))
        record = {'file': path.name, 'sha256': build.sha256(path)}
        manifest['predictions'].append({'name': name, 'status': 'valid', 'field': record, 'raw_field': record})
    input_manifest = inference/'manifest.json'
    input_manifest.write_text(json.dumps(manifest))
    output = tmp_path/'result'
    assert build.build_run(Namespace(inferences=input_manifest, output=output, spacings=[.004], workers=1)) == 0
    actual = json.loads((output/'manifest.json').read_text())
    assert actual['status'] == 'complete' and len(actual['configurations']) == 3
    assert actual['settings']['billing'] == 'offline_mock'
    assert actual['integrity']['settings']
    assert all(x['unchanged'] for key in ('sources', 'inputs', 'artifacts') for x in actual['integrity'][key])
    assert {p.name for p in (output/'source').iterdir()} == {'build_fields.py', 'field_csg.py'}
    for config in actual['configurations']:
        vertices, faces = build.decode_saved_fields([output/x['file'] for x in config['fields']])
        with np.load(output/config['decoded_mesh']['file'], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['vertices_m'], vertices)
            np.testing.assert_array_equal(saved['faces'], faces)


def test_workers_one_and_four_produce_bitwise_fields_and_meshes(tmp_path):
    import field_csg
    sphere = lambda name, center, radius: {'name': name, 'op': 'sphere', 'center_m': center,
                                          'rotation_xyzw': [0., 0., 0., 1.], 'size_m': [radius, 0., 0.],
                                          'left': None, 'right': None}
    operation = lambda name, op, a, b: {'name': name, 'op': op, 'center_m': None,
                                       'rotation_xyzw': None, 'size_m': None, 'left': a, 'right': b}
    spec = {'schema_version': 1, 'units': 'm', 'coordinate_frame': field_csg.FRAME,
            'estimated_focal_px': 500., 'root': 4, 'uncertainty_notes': 'synthetic parallel determinism fixture',
            'nodes': [sphere('outer', [.003, .001, .051], .029),
                      sphere('inner', [.005, .001, .051], .025),
                      operation('shell', 'difference', 0, 1),
                      sphere('extra', [.05, .003, .065], .020),
                      operation('union', 'union', 2, 3)]}
    spec = field_csg.validate_field(spec)
    lower, upper = field_csg.field_bounds(spec)
    records = []
    for workers in (1, 4):
        block_records, _ = build.save_field_blocks(lambda p: field_csg.evaluate_field(spec, p),
                                                   lower, upper, .004, tmp_path/f'w{workers}',
                                                   workers=workers, block_cells=8, worker_spec=spec,
                                                   worker_core=Path(field_csg.__file__))
        records.append(block_records)
    assert len(records[0]) == len(records[1]) > 1
    for first, second in zip(*records):
        assert first == second
        with np.load(tmp_path/'w1'/first['file'], allow_pickle=False) as a, np.load(tmp_path/'w4'/second['file'], allow_pickle=False) as b:
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
    one = build.decode_saved_fields([tmp_path/'w1'/x['file'] for x in records[0]])
    four = build.decode_saved_fields([tmp_path/'w4'/x['file'] for x in records[1]])
    for a, b in zip(one, four):
        np.testing.assert_array_equal(a, b)
