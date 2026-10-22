"""Set-of-Mark style coarse/fine pointing overlays for agent_SpeedUp_v1.

All pixels are UPRIGHT wrist-image coordinates (u right, v down). Coarse: an
8x8 grid of numbered cells (ids 1..64, row-major from the top left). Fine: a
crop of the chosen cell plus a margin (clamped inside the image), upsampled x5
with 8x8 numbered points. Marks are drawn on a copy; the input array is never
written, so tracking always runs on unmarked pixels.
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont


MARK_PROTOCOL_VERSION = "agent-speedup-v1-marks-8x8-crop96x5-8x8"
COARSE_GRID = 8
FINE_GRID = 8
FINE_UPSAMPLE = 5
PNG_PREFIX = "data:image/png;base64,"
_RESAMPLE = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR}
_LABEL_BOX = (0, 0, 0, 165)
_LABEL_TEXT = (255, 255, 255, 255)
_LINE_HALO = (0, 0, 0, 140)
_LINE_CORE = (255, 255, 255, 210)
_POINT_FILL = (255, 225, 20, 255)


@lru_cache(maxsize=8)
def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _mark_id(value, kind: str = "mark") -> int:
    count = COARSE_GRID * COARSE_GRID if kind == "cell" else FINE_GRID * FINE_GRID
    if not _integer(value) or not 1 <= int(value) <= count:
        raise ValueError(f"{kind} id must be an integer in 1..{count}")
    return int(value)


def _width(width) -> int:
    if not _integer(width) or int(width) < COARSE_GRID or int(width) % COARSE_GRID:
        raise ValueError(f"image width must be a positive integer divisible by {COARSE_GRID}")
    return int(width)


def _rgb(rgb) -> np.ndarray:
    pixels = np.asarray(rgb)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("images must be H x W x 3 uint8 RGB arrays")
    return pixels


def _square(rgb) -> np.ndarray:
    pixels = _rgb(rgb)
    if pixels.shape[0] != pixels.shape[1]:
        raise ValueError("mark overlays require a square image")
    _width(pixels.shape[1])
    return pixels


def _frame(pixels: np.ndarray) -> Image.Image:
    # np.array copies, so drawing can never reach the (possibly read-only) input.
    return Image.fromarray(np.array(pixels, dtype=np.uint8, order="C"), mode="RGB")


def _encode(frame: Image.Image, compress_level: int) -> str:
    buffer = io.BytesIO()
    frame.save(buffer, format="PNG", compress_level=compress_level)
    return PNG_PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_png_data_url(rgb: np.ndarray, *, compress_level: int = 1) -> str:
    """Lossless RGB PNG data URL; level 1 is ~5x faster than PIL's default 6."""
    if not _integer(compress_level) or not 0 <= int(compress_level) <= 9:
        raise ValueError("compress_level must be an integer in 0..9")
    return _encode(_frame(_rgb(rgb)), int(compress_level))


def coarse_cell_size(width: int) -> int:
    return _width(width) // COARSE_GRID


def _cell_rc(cell_id) -> tuple[int, int]:
    return divmod(_mark_id(cell_id, "cell") - 1, COARSE_GRID)


def coarse_cell_center(cell_id: int, width: int) -> tuple[float, float]:
    row, col = _cell_rc(cell_id)
    cell = coarse_cell_size(width)
    return (float((col + 0.5) * cell), float((row + 0.5) * cell))


def fine_crop_size(width: int) -> int:
    cell = coarse_cell_size(width)
    return cell + 2 * (cell // 4)


def fine_crop_origin(cell_id: int, width: int) -> tuple[int, int]:
    row, col = _cell_rc(cell_id)
    width = _width(width)
    cell, crop = width // COARSE_GRID, fine_crop_size(width)
    margin = cell // 4
    clamp = lambda start: int(min(max(start - margin, 0), width - crop))
    return (clamp(col * cell), clamp(row * cell))


def _origin(origin, width: int) -> tuple[int, int]:
    limit = width - fine_crop_size(width)
    if (not isinstance(origin, (tuple, list, np.ndarray)) or len(origin) != 2
            or not all(_integer(v) and 0 <= int(v) <= limit for v in origin)):
        raise ValueError(f"crop origin must be two integers in 0..{limit}")
    return (int(origin[0]), int(origin[1]))


def fine_point_pixel(origin: tuple[int, int], point_id: int, width: int) -> tuple[float, float]:
    width = _width(width)
    x0, y0 = _origin(origin, width)
    row, col = divmod(_mark_id(point_id, "point") - 1, FINE_GRID)
    spacing = fine_crop_size(width) / FINE_GRID
    return (float(x0 + (col + 0.5) * spacing), float(y0 + (row + 0.5) * spacing))


def _label(draw: ImageDraw.ImageDraw, xy, text: str, font, bounds, *, anchor="top_left", pad=2):
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left + 2 * pad, bottom - top + 2 * pad
    x, y = xy if anchor == "top_left" else (xy[0], xy[1] - h)
    x = min(max(int(x), 0), bounds[0] - w)
    y = min(max(int(y), 0), bounds[1] - h)
    draw.rectangle((x, y, x + w - 1, y + h - 1), fill=_LABEL_BOX)
    draw.text((x + pad - left, y + pad - top), text, font=font, fill=_LABEL_TEXT)


def _grid_lines(draw: ImageDraw.ImageDraw, size: int, count: int, *, halo=_LINE_HALO, core=_LINE_CORE):
    step = size / count
    for k in range(1, count):
        p = int(round(k * step))
        for fill, width in ((halo, 3), (core, 1)):
            if fill is not None:
                draw.line((p, 0, p, size - 1), fill=fill, width=width)
                draw.line((0, p, size - 1, p), fill=fill, width=width)


def coarse_overlay(rgb_upright: np.ndarray) -> str:
    """Upright image with an 8x8 grid; id label boxes in each cell's top-left corner."""
    pixels = _square(rgb_upright)
    width = pixels.shape[1]
    cell = width // COARSE_GRID
    frame = _frame(pixels)
    draw = ImageDraw.Draw(frame, "RGBA")
    _grid_lines(draw, width, COARSE_GRID)
    font = _font(max(8, round(cell * 13 / 64)))
    for index in range(COARSE_GRID * COARSE_GRID):
        row, col = divmod(index, COARSE_GRID)
        _label(draw, (col * cell + 2, row * cell + 2), str(index + 1), font, frame.size)
    return _encode(frame, 1)


def fine_overlay(rgb_upright: np.ndarray, cell_id: int, *,
                 resample: str = "bilinear") -> tuple[str, tuple[int, int]]:
    """Crop around a coarse cell, upsample x5, draw 8x8 numbered points.

    Point id ((row*8 + col) + 1) sits at ((col+.5), (row+.5)) * crop*5/8 in the
    returned image and maps back via fine_point_pixel(origin, id, width).
    """
    pixels = _square(rgb_upright)
    if resample not in _RESAMPLE:
        raise ValueError("resample must be 'nearest' or 'bilinear'")
    width = pixels.shape[1]
    x0, y0 = origin = fine_crop_origin(cell_id, width)
    crop = fine_crop_size(width)
    size = crop * FINE_UPSAMPLE
    frame = _frame(pixels[y0:y0 + crop, x0:x0 + crop]).resize((size, size), _RESAMPLE[resample])
    draw = ImageDraw.Draw(frame, "RGBA")
    _grid_lines(draw, size, FINE_GRID, halo=None, core=(255, 255, 255, 70))
    spacing = size / FINE_GRID
    font = _font(max(8, round(spacing * 13 / 60)))
    radius = max(2, round(spacing / 15))
    for index in range(FINE_GRID * FINE_GRID):
        row, col = divmod(index, FINE_GRID)
        x, y = (col + 0.5) * spacing, (row + 0.5) * spacing
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=_POINT_FILL, outline=(0, 0, 0, 255), width=1)
        _label(draw, (x + radius + 2, y - radius - 1), str(index + 1), font, frame.size,
               anchor="bottom_left")
    return _encode(frame, 1), origin
