"""Analytic geometry and isolation tests for NEW RGB field implementation."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from field_rgb import (ProjectiveField, backproject, block_shape, clip_mesh_planes,
                       depth_triangles, frustum_planes, plan_blocks, sample_projective_depth, visible_mesh)


def fixture_depth():
    depth = np.full((17, 17), .3)
    K = np.array([[50., 0, 8], [0, 50., 8], [0, 0, 1.]])
    return depth, K


def test_perspective_correct_depth_matches_sloped_3d_triangle():
    depth = np.array([[.30, .33], [.32, .36]])
    K = np.array([[5., 0, 0], [0, 5., 0], [0, 0, 1.]])
    vertices = backproject(depth, K).reshape(-1, 3)
    faces, _ = depth_triangles(depth)
    for face in faces.reshape(-1, 3):
        points = np.array([[.2, .3, .5], [.1, .8, .1]]) @ vertices[face]
        estimated, valid = sample_projective_depth(depth, K, points)
        assert valid.all()
        assert np.max(np.abs(estimated-points[:, 2])) < 1e-14


def test_depth_discontinuity_remains_unknown_not_occlusion_curtain():
    depth, K = fixture_depth()
    depth[:, 9:] = .8
    points = np.array([[.005, 0, .5], [0, 0, .5]])
    d, valid = sample_projective_depth(depth, K, points)
    assert not valid[0]  # pixel8.5, rejected across jump
    blocks = plan_blocks(depth, K, tiles=1)
    meshes = []
    for block in blocks:
        field = ProjectiveField.build(depth, K, block, .004)
        mesh = field.decode()
        if len(mesh.faces):
            meshes.append(mesh)
    z = np.concatenate([m.vertices[:, 2] for m in meshes])
    assert np.all((np.abs(z-.3) < 2e-7) | (np.abs(z-.8) < 2e-7))


def test_saved_only_decode_with_input_withheld_in_fresh_process(tmp_path):
    depth, K = fixture_depth()
    block = plan_blocks(depth, K, tiles=1)[0]
    field = ProjectiveField.build(depth, K, block, .002)
    field.save(tmp_path/'field.npz')
    del depth, K, field
    script = "import sys;sys.path.insert(0,sys.argv[1]);from field_rgb import ProjectiveField;f=ProjectiveField.load(sys.argv[2]);m=f.decode();assert len(m.faces)>0;assert abs(m.vertices[:,2]-.3).max()<1e-7;print(len(m.faces))"
    result = subprocess.run([sys.executable, '-B', '-c', script, str(HERE), str(tmp_path/'field.npz')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.glob('*depth*'))
    with pytest.raises(FileExistsError):
        ProjectiveField.load(tmp_path/'field.npz').save(tmp_path/'field.npz')


def test_tile_and_slab_clipping_has_no_duplicate_area_or_caps():
    depth, K = fixture_depth()
    mesh, _ = visible_mesh(depth, K)
    pieces = []
    for block in plan_blocks(depth, K, tiles=4):
        pieces.append(clip_mesh_planes(mesh, np.asarray(block['clip_planes'])))
    assert sum(m.area for m in pieces) == pytest.approx(mesh.area, abs=1e-12)
    assert all(np.allclose(m.vertices[:, 2], .3) for m in pieces)
    # Surface crossing several slab planes still conserves its original area.
    depth = np.tile(np.linspace(.27, .39, 17), (17, 1))
    mesh, _ = visible_mesh(depth, K)
    pieces = [clip_mesh_planes(mesh, np.asarray(b['clip_planes'])) for b in plan_blocks(depth, K, tiles=2)]
    assert sum(m.area for m in pieces) == pytest.approx(mesh.area, rel=2e-12)


def test_preflight_node_guard_and_fixed_alignment():
    depth, K = fixture_depth()
    block = plan_blocks(depth, K)[0]
    for h in (.004, .002, .001):
        shape = block_shape(block, h)
        assert all(s >= 2 for s in shape)
        assert np.allclose(np.array(block['lower_m'])/.004, np.rint(np.array(block['lower_m'])/.004))
    with pytest.raises(MemoryError):
        ProjectiveField.build(depth, K, block, .001, max_vertices=10)


def test_all_eight_observed_corners_required():
    depth, K = fixture_depth()
    block = plan_blocks(depth, K, tiles=1)[0]
    field = ProjectiveField.build(depth, K, block, .004)
    shape = field.values.shape
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                observed = field.observed[x:shape[0]-1+x, y:shape[1]-1+y, z:shape[2]-1+z]
                assert np.all(observed[field.cell_valid])
    field.cell_valid[:] = False
    assert len(field.decode().faces) == 0


def test_positive_area_patch_on_shared_slab_plane_owned_once():
    depth, K = fixture_depth()
    depth[:, :8] = .300
    depth[:, 8:] = .320  # exact5*64mm boundary, other faces also enter lower slab
    mesh, _ = visible_mesh(depth, K)
    parts = [clip_mesh_planes(mesh, np.asarray(block['clip_planes']), True) for block in plan_blocks(depth, K, tiles=2)]
    assert sum(part.area for part in parts) == pytest.approx(mesh.area, abs=1e-12)


def test_invalid_pixel_between_valid_corner_rays_rejects_cell():
    depth = np.full((33, 33), .5)
    depth[16, 16] = np.nan
    K = np.array([[1000., 0, 16], [0, 1000., 16], [0, 0, 1.]])
    block = {'id': 'hole', 'lower_m': [-.002, -.002, .496], 'upper_m': [.002, .002, .504],
             'clip_planes': frustum_planes([0, 32, 0, 32], K).tolist()}
    field = ProjectiveField.build(depth, K, block, .004)
    # Its corners project away from the NaN, but interior passes through it.
    assert field.observed[:, :, :2].all()
    assert not field.cell_valid.any()


def test_half_open_frustum_ownership_of_coplanar_artificial_face():
    import trimesh
    mesh = trimesh.Trimesh(vertices=[[0, -.1, .2], [0, .1, .2], [0, 0, .4]], faces=[[0, 1, 2]], process=False)
    K = np.array([[10., 0, 8], [0, 10., 8], [0, 0, 1.]])
    left = clip_mesh_planes(mesh, frustum_planes([0, 8, 0, 16], K), True)
    right = clip_mesh_planes(mesh, frustum_planes([8, 16, 0, 16], K), True)
    assert left.area == 0
    assert right.area == pytest.approx(mesh.area)
