import json
import time

import numpy as np
import pytest

from gpt_field import mark_prompting as mp
from gpt_field.mark_prompting import (
    COARSE_GRID, FINE_GRID, FINE_UPSAMPLE, MARK_PROTOCOL_VERSION, coarse_cell_center,
    coarse_cell_size, coarse_overlay, encode_png_data_url, fine_crop_origin, fine_crop_size,
    fine_overlay, fine_point_pixel,
)
from gpt_field.observation import decode_image


W = 512
BAD_IDS = (0, 65, -1, True, False, 1.0, "1", None, np.float64(3.0))


def textured(width=W, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(width, width, 3), dtype=np.uint8)


def coordinate_image(width=W):
    y, x = np.mgrid[0:width, 0:width]
    return np.stack([x % 256, y % 256, 7 + 16 * (x // 256) + 64 * (y // 256)], -1).astype(np.uint8)


def decoded(url):
    assert url.startswith("data:image/png;base64,")
    return np.asarray(decode_image(url))


def readonly(array):
    array = array.copy()
    array.setflags(write=False)
    return array


def test_protocol_constants():
    assert MARK_PROTOCOL_VERSION == "agent-speedup-v1-marks-8x8-crop96x5|128x4-8x8-gravity-turn"
    assert (COARSE_GRID, FINE_GRID, FINE_UPSAMPLE) == (8, 8, 5)
    assert coarse_cell_size(512) == 64 and fine_crop_size(512) == 96
    assert coarse_cell_size(256) == 32 and fine_crop_size(256) == 48


@pytest.mark.parametrize("width", (500, 0, -8, 4, 512.0, True, "512", None))
def test_width_validation(width):
    with pytest.raises(ValueError):
        coarse_cell_size(width)
    with pytest.raises(ValueError):
        fine_crop_origin(1, width)


@pytest.mark.parametrize("cell_id, center", (
    (1, (32.0, 32.0)), (8, (480.0, 32.0)), (57, (32.0, 480.0)), (64, (480.0, 480.0)),
    (36, (224.0, 288.0)), (10, (96.0, 96.0)),
))
def test_coarse_cell_center(cell_id, center):
    value = coarse_cell_center(cell_id, W)
    assert value == center and all(type(v) is float for v in value)


def test_coarse_ids_are_row_major():
    centers = [coarse_cell_center(i, W) for i in range(1, 65)]
    for index, (u, v) in enumerate(centers):
        row, col = divmod(index, 8)
        assert (int(u) // 64, int(v) // 64) == (col, row)
    assert coarse_cell_center(np.int64(36), W) == (224.0, 288.0)


@pytest.mark.parametrize("cell_id, origin", (
    (1, (0, 0)), (2, (48, 0)), (8, (416, 0)), (7, (368, 0)), (57, (0, 416)),
    (64, (416, 416)), (36, (176, 240)), (10, (48, 48)), (16, (416, 48)),
))
def test_fine_crop_origin_clamps_at_edges(cell_id, origin):
    value = fine_crop_origin(cell_id, W)
    assert value == origin and all(type(v) is int for v in value)


def test_fine_crop_always_contains_its_cell_and_point_grid_covers_it():
    cell, crop = coarse_cell_size(W), fine_crop_size(W)
    for cell_id in range(1, 65):
        row, col = divmod(cell_id - 1, 8)
        x0, y0 = origin = fine_crop_origin(cell_id, W)
        assert 0 <= x0 <= W - crop and 0 <= y0 <= W - crop
        assert x0 <= col * cell and col * cell + cell <= x0 + crop
        assert y0 <= row * cell and row * cell + cell <= y0 + crop
        points = np.array([fine_point_pixel(origin, i, W) for i in range(1, 65)])
        xs, ys = np.unique(points[:, 0]), np.unique(points[:, 1])
        assert len(xs) == len(ys) == 8 and np.allclose(np.diff(xs), 12) and np.allclose(np.diff(ys), 12)
        cell_px = np.arange(cell) + 0.5
        # Every pixel of the cell (pixel-centre coordinates) is within 6 px of a point per axis.
        assert np.abs((col * cell + cell_px)[:, None] - xs[None]).min(1).max() <= 6.0
        assert np.abs((row * cell + cell_px)[:, None] - ys[None]).min(1).max() <= 6.0


@pytest.mark.parametrize("origin, point_id, pixel", (
    ((176, 240), 1, (182.0, 246.0)), ((176, 240), 8, (266.0, 246.0)),
    ((176, 240), 57, (182.0, 330.0)), ((176, 240), 64, (266.0, 330.0)),
    ((176, 240), 10, (194.0, 258.0)), ((416, 0), 8, (506.0, 6.0)), ((0, 416), 57, (6.0, 506.0)),
))
def test_fine_point_pixel(origin, point_id, pixel):
    value = fine_point_pixel(origin, point_id, W)
    assert value == pixel and all(type(v) is float for v in value)
    assert fine_point_pixel(list(origin), point_id, W) == pixel


def test_coarse_to_fine_round_trip_lands_in_selected_cell_neighbourhood():
    for cell_id in (1, 8, 29, 36, 57, 64):
        origin = fine_crop_origin(cell_id, W)
        u, v = fine_point_pixel(origin, 28, W)  # an interior point of the fine grid
        cu, cv = coarse_cell_center(cell_id, W)
        assert abs(u - cu) <= 64 and abs(v - cv) <= 64


@pytest.mark.parametrize("bad", BAD_IDS)
def test_id_validation(bad):
    with pytest.raises(ValueError):
        coarse_cell_center(bad, W)
    with pytest.raises(ValueError):
        fine_crop_origin(bad, W)
    with pytest.raises(ValueError):
        fine_point_pixel((0, 0), bad, W)
    with pytest.raises(ValueError):
        fine_overlay(textured(64), bad)


@pytest.mark.parametrize("origin", ((-1, 0), (417, 0), (0, 417), (0.5, 0), (True, 0), (0, 0, 0), (0,), "ab", None))
def test_fine_point_pixel_rejects_bad_origin(origin):
    with pytest.raises(ValueError):
        fine_point_pixel(origin, 1, W)


def test_encode_png_data_url_is_lossless():
    image = textured()
    for level in (0, 1, 9):
        url = encode_png_data_url(image, compress_level=level)
        assert np.array_equal(decoded(url), image)
    flipped = readonly(image)[:, ::-1]
    assert not flipped.flags.c_contiguous
    assert np.array_equal(decoded(encode_png_data_url(flipped)), image[:, ::-1])
    assert np.array_equal(decoded(encode_png_data_url(textured(30)[:, :17])), textured(30)[:, :17])


@pytest.mark.parametrize("bad", (
    np.zeros((8, 8, 3), np.float32), np.zeros((8, 8), np.uint8), np.zeros((8, 8, 4), np.uint8),
))
def test_encode_png_data_url_rejects_non_rgb(bad):
    with pytest.raises(ValueError):
        encode_png_data_url(bad)


@pytest.mark.parametrize("level", (-1, 10, 1.0, True, "1"))
def test_encode_png_data_url_rejects_bad_level(level):
    with pytest.raises(ValueError):
        encode_png_data_url(textured(8), compress_level=level)


def test_coarse_overlay_draws_grid_and_labels_without_hiding_image():
    image = readonly(textured())
    before = image.copy()
    out = decoded(coarse_overlay(image))
    assert np.array_equal(image, before)
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8
    changed = np.any(out != image, axis=-1)
    assert 0.03 < changed.mean() < 0.25
    for cell_id in range(1, 65):
        row, col = divmod(cell_id - 1, 8)
        x0, y0 = col * 64, row * 64
        u, v = (int(c) for c in coarse_cell_center(cell_id, W))
        # Cell centre and the bulk of the cell stay untouched; the label sits in the top-left corner.
        assert not changed[v - 8:v + 8, u - 8:u + 8].any()
        assert changed[y0:y0 + 64, x0:x0 + 64].mean() < 0.25
        assert changed[y0 + 4:y0 + 12, x0 + 4:x0 + 12].mean() > 0.9
    for k in range(1, 8):
        assert changed[:, k * 64].mean() > 0.95 and changed[k * 64, :].mean() > 0.95
    assert changed[:, 0].mean() < 0.1 and changed[:, -1].mean() < 0.1


def test_coarse_label_boxes_are_semi_transparent():
    dark, light = np.zeros((512, 512, 3), np.uint8), np.full((512, 512, 3), 255, np.uint8)
    a, b = decoded(coarse_overlay(dark)), decoded(coarse_overlay(light))
    box = (slice(3, 13), slice(2, 4))  # label-box padding, left of the text glyphs
    assert (a[box] < b[box]).all(), "underlying pixels must still show through the label box"


def test_coarse_overlay_labels_differ_per_cell():
    out = decoded(coarse_overlay(np.full((512, 512, 3), 128, np.uint8)))
    patches = {out[r * 64:r * 64 + 16, c * 64:c * 64 + 24].tobytes() for r in range(8) for c in range(8)}
    assert len(patches) == 64


@pytest.mark.parametrize("bad", (
    np.zeros((512, 480, 3), np.uint8), np.zeros((500, 500, 3), np.uint8),
    np.zeros((512, 512, 3), np.float64), np.zeros((512, 512), np.uint8),
))
def test_overlays_reject_bad_images(bad):
    with pytest.raises(ValueError):
        coarse_overlay(bad)
    with pytest.raises(ValueError):
        fine_overlay(bad, 1)


def test_fine_overlay_nearest_maps_back_exactly():
    image = readonly(coordinate_image())
    before = image.copy()
    for cell_id in (1, 8, 36, 57, 64):
        url, origin = fine_overlay(image, cell_id, resample="nearest")
        assert origin == fine_crop_origin(cell_id, W) and all(type(v) is int for v in origin)
        out = decoded(url)
        assert out.shape == (480, 480, 3)
        x0, y0 = origin
        expected = np.repeat(np.repeat(image[y0:y0 + 96, x0:x0 + 96], 5, 0), 5, 1)
        changed = np.any(out != expected, axis=-1)
        assert 0.02 < changed.mean() < 0.25
        for point_id in range(1, 65):
            row, col = divmod(point_id - 1, 8)
            X, Y = int((col + 0.5) * 60), int((row + 0.5) * 60)
            assert changed[Y, X], "point mark drawn at its documented position"
            px, py = fine_point_pixel(origin, point_id, W)
            assert (x0 + X // 5, y0 + Y // 5) == (int(px), int(py))
            # Lower-left quadrant around each point is free of marks: real pixels at the mapped location.
            qx, qy = col * 60 + 15, row * 60 + 45
            assert not changed[qy - 8:qy + 8, qx - 8:qx + 8].any()
            assert np.array_equal(out[qy, qx], image[y0 + qy // 5, x0 + qx // 5])
    assert np.array_equal(image, before)


def test_fine_overlay_default_bilinear_uses_the_crop():
    y, x = np.mgrid[0:W, 0:W]
    image = np.stack([x // 2, y // 2, np.full_like(x, 90)], -1).astype(np.uint8)
    url, (x0, y0) = fine_overlay(image, 36)
    assert (x0, y0) == (176, 240)
    out = decoded(url).astype(float)
    for X, Y in ((15, 45), (195, 345), (435, 465)):
        u, v = x0 + (X + 0.5) / 5 - 0.5, y0 + (Y + 0.5) / 5 - 0.5
        assert abs(out[Y, X, 0] - u / 2) <= 1.5 and abs(out[Y, X, 1] - v / 2) <= 1.5


def test_fine_overlay_rejects_bad_resample():
    with pytest.raises(ValueError):
        fine_overlay(textured(), 1, resample="lanczos")


def test_overlays_do_not_mutate_writable_input():
    image = textured(seed=3)
    before = image.copy()
    coarse_overlay(image)
    fine_overlay(image, 12)
    fine_overlay(image, 64, resample="nearest")
    encode_png_data_url(image)
    assert np.array_equal(image, before)


def test_outputs_json_safe_and_fast():
    image = textured(seed=5)
    start = time.perf_counter()
    url = coarse_overlay(image)
    fine, origin = fine_overlay(image, 29)
    elapsed = time.perf_counter() - start
    record = {"origin": origin, "center": coarse_cell_center(29, W),
              "point": fine_point_pixel(origin, 64, W), "crop": fine_crop_size(W)}
    assert json.loads(json.dumps(record, allow_nan=False))["origin"] == list(origin)
    assert isinstance(url, str) and isinstance(fine, str)
    assert elapsed < 1.0


def test_small_square_images_supported():
    image = textured(64)
    out = decoded(coarse_overlay(image))
    assert out.shape == (64, 64, 3)
    url, origin = fine_overlay(image, 64)
    assert origin == (64 - 12, 64 - 12) and decoded(url).shape == (60, 60, 3)
    assert mp.fine_point_pixel(origin, 1, 64) == (52.75, 52.75)


def test_gravity_turns_and_pixel_rotation_round_trip():
    """Drawer live run 001: pointing images are turned so that world up is at the top (frontal hand)."""
    from gpt_field.mark_prompting import gravity_quarter_turns, rotate_pixel, rotate_quarter, unrotate_pixel
    assert [gravity_quarter_turns(*g) for g in ((0, 1), (1, 0), (0, -1), (-1, 0), (0.2, -0.3))] == [0, 1, 2, 3, 0]
    image = np.zeros((512, 512, 3), np.uint8)
    image[37, 100] = 255                                     # (u, v) = (100, 37)
    for turns in range(4):
        rotated = rotate_quarter(image, turns)
        v, u = np.argwhere(rotated[..., 0] == 255)[0]
        assert rotate_pixel((100, 37), turns, 512) == (u, v)
        assert unrotate_pixel((u, v), turns, 512) == (100.0, 37.0)
    marked = np.zeros((512, 512, 3), np.uint8)
    marked[:, 400:] = 200                                    # bright band on the right = world up for R1 = +z
    assert rotate_quarter(marked, gravity_quarter_turns(1.0, 0.0))[:100].mean() > 150   # now at the top


def test_wide_fine_crop_for_bars_maps_points_back():
    from gpt_field.mark_prompting import WIDE_MARGIN_CELLS, fine_crop_size, fine_point_pixel, fine_upsample
    assert fine_crop_size(512, WIDE_MARGIN_CELLS) == 128 and fine_upsample(512, WIDE_MARGIN_CELLS) == 4
    image = np.zeros((512, 512, 3), np.uint8)
    url, origin = mp.fine_overlay(image, 28, margin_cells=WIDE_MARGIN_CELLS)
    assert origin == (3 * 64 - 32, 3 * 64 - 32)
    assert np.asarray(decode_image(url)).shape[:2] == (512, 512)
    assert fine_point_pixel(origin, 1, 512, WIDE_MARGIN_CELLS) == (origin[0] + 8.0, origin[1] + 8.0)
    with pytest.raises(ValueError):
        fine_crop_size(512, 0.3)
