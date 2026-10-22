import json
from pathlib import Path

import numpy as np
import pytest

from scene_test.field_planning.predicted_fields import (
    AstraSavedField, DepthProSavedField, SIGN_EPS_M, _SavedAdapter, sha256,
)

SNAPSHOT = 'drawer003_closed_step60'


def make_astra(tmp_path):
    name = SNAPSHOT + '_h2mm'
    directory = tmp_path / 'fields' / name
    directory.mkdir(parents=True)
    lower, h = np.zeros(3), .002
    values = np.broadcast_to(np.array([-.001, .001])[:, None, None], (2, 2, 2)).copy().astype('float32')
    block = directory / 'block_0001_0000_0000.npz'
    np.savez(block, values=values, lower_m=np.array([.002, 0., 0.]), spacing_m=h,
             vertex_quantization_max_m=1e-10)
    cert = directory / 'block_certificates.npz'
    np.savez(cert, block_ids=np.array([[0, 0, 0], [1, 0, 0]]),
             start_cells=np.array([[0, 0, 0], [1, 0, 0]]), cell_shapes=np.ones((2, 3), int),
             center_values_m=np.array([-.002, 0.]), half_diagonal_m=np.full(2, np.sqrt(3)*.001),
             certificate_tolerance_m=np.full(2, 1e-10), retained=np.array([False, True]),
             lower_m=lower, spacing_m=h, total_cells=np.array([2, 1, 1]))
    manifest = {'status': 'complete', 'configurations': [{
        'name': name, 'status': 'ok', 'allocation': {
            'grid_lower_m': lower.tolist(), 'grid_upper_m': [.004, .002, .002],
            'block_cells': 1, 'certificate_file': cert.name, 'certificate_sha256': sha256(cert)},
        'fields': [{'file': str(block.relative_to(tmp_path)), 'sha256': sha256(block)}],
    }]}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return AstraSavedField(tmp_path, SNAPSHOT), block


def make_depthpro(tmp_path, value=.001):
    # A supported 2 mm cell within the frozen camera/tile/slab convention.
    adapter = DepthProSavedField.__new__(DepthProSavedField)
    _SavedAdapter.__init__(adapter, tmp_path, SNAPSHOT)
    adapter.K = np.array([[1., 0., 2.], [0., 1., 2.], [0., 0., 1.]])
    adapter.image_shape = (5, 5)
    adapter.slab_m = .064
    adapter.u_edges = adapter.v_edges = np.arange(5.)
    key = (2, 2, 15)
    adapter.records = {key: {'file': 'numerical_fixture'}}
    adapter.metadata = {key: {'clip_planes': [[1, 0, 0, 0], [-1, 0, 1, 0],
                                             [0, 1, 0, 0], [0, -1, 1, 0],
                                             [0, 0, 1, -.96], [0, 0, -1, 1.024]],
                              'vertex_quantization_max_m': 1e-10}}
    block = {'values': np.full((2, 2, 2), value, np.float32),
             'observed': np.ones((2, 2, 2), bool), 'cell_valid': np.ones((1, 1, 1), bool),
             'lower_m': np.array([0., 0., 1.]), 'spacing_m': np.array(.002)}
    adapter._load_block = lambda record: block
    return adapter, block


def test_astra_saved_certificate_and_corner_interpolation(tmp_path):
    adapter, _ = make_astra(tmp_path)
    points = np.array([[.001, .001, .001], [.0039, .0001, .0001],
                       [.0021, .0001, .0001], [.003, .001, .001], [3., 0., 0.]])
    result = adapter.query(points)
    assert result.state.tolist() == [-1, 1, -1, 0, 0]
    assert result.support.tolist() == [True, True, True, True, False]
    assert np.isnan(result.raw_phi_m[0])  # center certificate is not query phi
    assert np.isclose(result.raw_phi_m[1], .0009, atol=1e-10)
    assert 0 < result.conservative_clearance_m[1] <= .0001
    assert result.summary()['supported_ambiguous'] == 1


def test_astra_certified_signs_and_clearance_have_linear_field_bound(tmp_path):
    adapter, _ = make_astra(tmp_path)
    p = np.random.default_rng(185).uniform(0, [.004, .002, .002], (4000, 3))
    r = adapter.query(p)
    phi = p[:, 0] - .003
    assert np.all(phi[r.state == 1] > 0.)
    assert np.all(phi[r.state == -1] < 0.)
    assert np.all(r.phi_lower_m <= phi + 2e-10)
    assert np.all(r.phi_upper_m >= phi - 2e-10)
    assert np.all(r.conservative_clearance_m <= np.abs(phi) + 2e-10)
    assert np.all(r.conservative_clearance_m <= np.minimum(p, [.004, .002, .002] - p).min(axis=1) + 2e-10)


def test_astra_nonfinite_and_out_of_domain_are_unknown(tmp_path):
    adapter, _ = make_astra(tmp_path)
    r = adapter.query([[np.nan, 0., 0.], [np.inf, 0., 0.], [-1e-6, .001, .001]])
    assert not r.support.any()
    assert not r.state.any()
    assert not r.conservative_clearance_m.any()


def test_astra_tampered_saved_block_is_rejected(tmp_path):
    adapter, block = make_astra(tmp_path)
    with block.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        adapter.query([[.0039, .0001, .0001]])


def test_astra_provenance_rechecks_input_immutability(tmp_path):
    adapter, _ = make_astra(tmp_path)
    assert all(adapter.provenance(verify=True)['inputs_unchanged'].values())
    (tmp_path / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='inputs changed'):
        adapter.provenance(verify=True)


def test_unsupported_snapshot_and_spacing_rejected(tmp_path):
    with pytest.raises(ValueError, match='drawer003'):
        AstraSavedField(tmp_path, 'bowl002_initial')
    with pytest.raises(ValueError, match='2 mm'):
        AstraSavedField(tmp_path, SNAPSHOT, spacing_m=.004)


def test_depthpro_positive_supported_cell_and_local_clearance(tmp_path):
    adapter, _ = make_depthpro(tmp_path, .020)
    r = adapter.query([[.001, .001, 1.001]])
    assert r.state.tolist() == [1]
    assert r.support.tolist() == [True]
    assert np.isclose(r.conservative_clearance_m[0], .001)
    assert r.conservative_clearance_m[0] < r.raw_phi_m[0]


def test_depthpro_positive_placeholder_never_makes_unknown_free(tmp_path):
    adapter, block = make_depthpro(tmp_path, .020)
    block['observed'][1, 1, 1] = False
    r = adapter.query([[.001, .001, 1.001]])
    assert r.state.tolist() == [0]
    assert not r.support.any()
    assert np.isnan(r.raw_phi_m).all()


def test_depthpro_invalid_cell_overrides_observed_vertices(tmp_path):
    adapter, block = make_depthpro(tmp_path)
    block['cell_valid'][0, 0, 0] = False
    r = adapter.query([[.001, .001, 1.001]])
    assert not r.support.any()
    assert not r.state.any()


def test_depthpro_supported_negative_is_occupied(tmp_path):
    adapter, _ = make_depthpro(tmp_path, -.001)
    r = adapter.query([[.001, .001, 1.001]])
    assert r.state.tolist() == [-1]
    assert r.support.tolist() == [True]
    assert not r.conservative_clearance_m.any()


def test_depthpro_missing_owner_outside_grid_and_bad_camera_points_unknown(tmp_path):
    adapter, _ = make_depthpro(tmp_path)
    r = adapter.query([[.001, .001, .9], [.01, .01, 1.001], [0, 0, -1], [np.nan, 0, 1], [9, 9, 1]])
    assert not r.support.any()
    assert not r.state.any()


def test_depthpro_near_zero_supported_but_ambiguous(tmp_path):
    adapter, _ = make_depthpro(tmp_path, SIGN_EPS_M / 2)
    r = adapter.query([[.001, .001, 1.001]])
    assert r.support.tolist() == [True]
    assert r.state.tolist() == [0]


def test_depthpro_clearance_respects_trilinear_gradient(tmp_path):
    adapter, block = make_depthpro(tmp_path)
    block['values'][:] = np.broadcast_to(np.array([-.002, .002])[:, None, None], (2, 2, 2))
    r = adapter.query([[.0015, .001, 1.001], [.0005, .001, 1.001]])
    assert r.state.tolist() == [1, -1]
    assert .00049 < r.conservative_clearance_m[0] < .0005
    assert not r.conservative_clearance_m[1]


def test_query_empty_and_shape_validation(tmp_path):
    adapter, _ = make_astra(tmp_path)
    assert len(adapter.query(np.empty((0, 3))).state) == 0
    with pytest.raises(ValueError, match='shape'):
        adapter.query([0, 0, 0])
