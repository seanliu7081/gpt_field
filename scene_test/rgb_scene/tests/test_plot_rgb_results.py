"""Scientific display must preserve saved support and actual face occlusion."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "plot_rgb_results", Path(__file__).resolve().parents[1] / "plot_rgb_results.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def test_renderer_uses_nearest_face_without_order_dependence():
    projected = np.array([[1., 1.], [6., 1.], [1., 6.]]*2)
    view_depth = np.array([1., 1., 1., 2., 2., 2.])
    native_depth = np.array([.5, .5, .5, 1.5, 1.5, 1.5])
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]*2)
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    palette = np.repeat(np.arange(256, dtype=float)[:, None], 3, axis=1)
    first, depth, _ = plot.rasterize_triangles(
        projected, view_depth, native_depth, vertices, faces, palette, 8)
    second, _, _ = plot.rasterize_triangles(
        projected, view_depth, native_depth, vertices, faces[::-1].copy(), palette, 8)
    assert np.array_equal(first, second)
    assert depth[2, 2] == 2
    assert first[2, 2, 0] > 100
    assert not np.isfinite(depth[7, 7])


def test_slice_displays_saved_values_and_keeps_unknown_gray_domain(tmp_path):
    values = np.full((2, 2, 2), .02, dtype=np.float32)
    values[0, 0, 0] = -.007
    observed = np.zeros_like(values, dtype=bool)
    observed[0, 0, 0] = True
    path = tmp_path / "tile00_00_slab0005.npz"
    np.savez_compressed(path, values=values, observed=observed,
                        lower_m=[0., 0., .35], spacing_m=.001,
                        metadata_json=json.dumps({"clip_planes": []}))
    config = {"snapshot": "synthetic", "spacing_m": .001,
              "fields": [{"file": path.name, "sha256": plot.sha256(path)}]}
    snapshot = {"blocks": [{"id": path.stem, "lower_m": [0, 0, .35],
                             "upper_m": [.001, .001, .351]}]}
    consumed = {}
    mosaic, metadata = plot.saved_field_slice(tmp_path, config, snapshot, consumed)
    assert np.isfinite(mosaic).sum() == 1
    assert mosaic[1000, 0] == pytest.approx(-.007)
    assert str(path) in consumed
    assert len(metadata["fields"]) == 1


def test_plot_existing_output_refusal_preserves_previous_contents(tmp_path):
    output = tmp_path / "old"
    output.mkdir()
    sentinel = output / "manifest.json"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        plot.run(argparse.Namespace(output=output, build=tmp_path / "absent"))
    assert sentinel.read_text() == "keep"
