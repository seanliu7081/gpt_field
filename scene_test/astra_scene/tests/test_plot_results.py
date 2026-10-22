"""Verify plots display saved data and do not conceal missing geometry."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("astra_plot_results", Path(__file__).resolve().parents[1]/"plot_results.py")
plot = importlib.util.module_from_spec(spec); spec.loader.exec_module(plot)


def test_slice_reads_saved_values_and_leaves_unsaved_blocks_gray(tmp_path):
    values = np.arange(27, dtype=np.float32).reshape(3, 3, 3)*.001
    path = tmp_path/"block.npz"
    np.savez_compressed(path, values=values, lower_m=np.array([-1., -.5, 0.]), spacing_m=.5)
    record = {"file": path.name, "sha256": plot.sha(path), "shape": [3, 3, 3], "lower_m": [-1., -.5, 0.]}
    config = {"spacing_m": .5, "allocation": {"grid_lower_m": [-1., -.5, 0.], "grid_cells": [4, 2, 2]}, "fields": [record]}
    snapshot = {"geometry_lower_m": [-1., -.4, 0.], "geometry_upper_m": [1., .4, 1.]}
    consumed = {}
    mosaic, metadata = plot.saved_field_slice(tmp_path, config, snapshot, consumed)
    np.testing.assert_array_equal(mosaic[:3], values[:, 1, :])
    assert np.isnan(mosaic[3:]).all()
    assert metadata["camera_y_m"] == 0
    assert "not unknown" in metadata["gray_meaning"]
    assert consumed[str(path.resolve())] == record["sha256"]


def test_dynamic_shared_projection_does_not_clip_large_or_offset_geometry():
    bounds = [np.array([-4., -.7, .1]), np.array([1.1, 2., 5.8])]
    vertices = np.array([[x, y, z] for x in (bounds[0][0], bounds[1][0])
                         for y in (bounds[0][1], bounds[1][1]) for z in (bounds[0][2], bounds[1][2])])
    projected, depth, world = plot.mesh_projection(vertices, bounds)
    assert projected.min() >= 19
    assert projected.max() <= 780
    np.testing.assert_array_equal(world, vertices[:, [0, 2, 1]]*[1, 1, -1])
    assert np.isfinite(depth).all()


def test_missed_gt_pixels_are_cyan_and_excluded_from_hit_error():
    gt = np.array([[1., 2.], [3., 4.]])
    prediction = np.array([[1.1, np.nan], [0., 8.]])
    valid = np.array([[True, True], [True, False]])
    error, overlay, hit = plot.error_image(prediction, gt, valid)
    np.testing.assert_array_equal(hit, [[True, False], [False, False]])
    np.testing.assert_allclose(error.compressed(), [.1])
    np.testing.assert_allclose(overlay[0, 1], [0, .75, .9, 1])
    np.testing.assert_allclose(overlay[1, 0], [0, .75, .9, 1])
    np.testing.assert_allclose(overlay[1, 1], [0, 0, 0, 0])


def test_artifact_hash_mismatch_is_rejected(tmp_path):
    path = tmp_path/"saved.txt"; path.write_text("before")
    record = {"file": path.name, "sha256": plot.sha(path)}
    path.write_text("after")
    with pytest.raises(ValueError, match="SHA mismatch"):
        plot.checked(tmp_path, record, {})
