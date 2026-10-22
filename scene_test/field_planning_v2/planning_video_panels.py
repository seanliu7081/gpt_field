"""Pillow-only planning-field panel for drawer003 video replay.

The field is the current Astra environment adapter, evaluated in camera
coordinates. This horizontal slice and projected EEF line are visualization
only: they do not replace full-robot collision checks or certify a path.
No ground-truth geometry, segmentation, or evaluation is imported here.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size, bold=False):
    path = Path('/usr/share/fonts/truetype/freefont') / ('FreeSansBold.ttf' if bold else 'FreeSans.ttf')
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


class FieldPanel:
    """Render a 600-pixel Astra field panel, optionally scaled at the end.

    ``robot`` provides ``eef_pose(q, frame='world')`` and
    ``camera_from_world``; ``field`` provides the current ``query`` API.
    ``progress_index`` selects the last reached point in the *supplied* EEF
    path. If omitted, the entire saved path is highlighted.
    """

    lower_xy = np.array([-.60, -.60])
    upper_xy = np.array([.60, .60])
    plot_box = (90, 78, 510, 498)
    grid_resolution = 192
    value_cap_m = .100

    def __init__(self, field, robot, size=600):
        if int(size) < 1:
            raise ValueError('size must be positive')
        self.field, self.robot, self.size = field, robot, int(size)
        self.font = _font(15)
        self.small_font = _font(13)
        self.title_font = _font(22, True)
        self._slice_cache = OrderedDict()
        # Image rows run downward, so the top row samples the largest world Y.
        x = np.linspace(self.lower_xy[0], self.upper_xy[0], self.grid_resolution)
        y = np.linspace(self.upper_xy[1], self.lower_xy[1], self.grid_resolution)
        xx, yy = np.meshgrid(x, y)
        self._world_xy = np.column_stack((xx.ravel(), yy.ravel()))

    @staticmethod
    def _colors(state, phi):
        state, phi = np.asarray(state), np.asarray(phi)
        colors = np.full((len(state), 3), (150, 157, 165), dtype=np.uint8)
        free = (state == 1) & np.isfinite(phi)
        t = np.clip(phi[free] / FieldPanel.value_cap_m, 0., 1.)[:, None]
        colors[free] = np.rint(np.array([224, 241, 253]) * (1.-t)
                              + np.array([32, 105, 180]) * t).astype(np.uint8)
        colors[state == -1] = (212, 73, 70)
        return colors

    def _xy_pixels(self, points):
        """Map world XY into the plotted rectangle, before final scaling."""
        xy = np.asarray(points, float)
        left, top, right, bottom = self.plot_box
        x = left + (xy[..., 0] - self.lower_xy[0]) / (self.upper_xy[0] - self.lower_xy[0]) * (right-left-1)
        y = top + (self.upper_xy[1] - xy[..., 1]) / (self.upper_xy[1] - self.lower_xy[1]) * (bottom-top-1)
        return np.stack((x, y), axis=-1)

    def _slice(self, z):
        # Exact-Z cache, no unreported height quantization. Repeated held poses
        # are cheap; an LRU cap avoids accumulating all video frames in memory.
        key = float(z)
        if key in self._slice_cache:
            self._slice_cache.move_to_end(key)
            return self._slice_cache[key].copy()
        world = np.column_stack((self._world_xy, np.full(len(self._world_xy), z)))
        transform = np.asarray(self.robot.camera_from_world, float)
        camera = world @ transform[:3, :3].T + transform[:3, 3]
        query = self.field.query(camera)
        rgb = self._colors(query.state, query.raw_phi_m)
        image = Image.fromarray(rgb.reshape(self.grid_resolution, self.grid_resolution, 3), 'RGB')
        left, top, right, bottom = self.plot_box
        image = image.resize((right-left, bottom-top), Image.Resampling.NEAREST)
        self._slice_cache[key] = image
        while len(self._slice_cache) > 32:
            self._slice_cache.popitem(last=False)
        return image.copy()

    def render(self, q, eef_path_world, progress_index=None):
        pose = np.asarray(self.robot.eef_pose(q, frame='world'), float)
        current = pose[:3, 3]
        panel = Image.new('RGB', (600, 600), (245, 247, 250))
        draw = ImageDraw.Draw(panel)
        draw.text((20, 13), 'Astra field | horizontal slice', fill=(23, 35, 50), font=self.title_font)
        draw.text((20, 43), f'World Z = {current[2]:.3f} m  |  robot components removed',
                  fill=(66, 76, 87), font=self.font)
        left, top, right, bottom = self.plot_box
        tile = self._slice(current[2])
        tile_draw = ImageDraw.Draw(tile)
        local_offset = np.array([left, top])
        path = np.asarray(eef_path_world, float)
        if path.size:
            if path.ndim != 2 or path.shape[1] != 3 or not np.isfinite(path).all():
                raise ValueError('eef_path_world must be finite [N,3] points')
            pixels = self._xy_pixels(path[:, :2]) - local_offset
            coords = [tuple(row) for row in pixels]
            if len(coords) > 1:
                tile_draw.line(coords, fill=(55, 55, 55), width=4)
                tile_draw.line(coords, fill=(230, 230, 230), width=2)
                stop = len(coords)-1 if progress_index is None else int(np.clip(progress_index, 0, len(coords)-1))
                if stop > 0:
                    tile_draw.line(coords[:stop+1], fill=(25, 25, 25), width=5)
                    tile_draw.line(coords[:stop+1], fill=(255, 214, 55), width=3)
        point = self._xy_pixels(current[:2]) - local_offset
        px, py = point
        tile_draw.ellipse((px-6, py-6, px+6, py+6), fill=(255, 255, 255), outline=(20, 25, 30), width=2)
        panel.paste(tile, (left, top))
        draw.rectangle((left-1, top-1, right, bottom), outline=(55, 66, 77), width=1)
        for value in (-.6, -.3, 0., .3, .6):
            px = float(self._xy_pixels(np.array([value, 0.]))[0])
            py = float(self._xy_pixels(np.array([0., value]))[1])
            draw.line((px, bottom, px, bottom+4), fill=(55, 66, 77))
            draw.text((px, bottom+7), f'{value:.1f}', anchor='mt', fill=(55, 66, 77), font=self.small_font)
            draw.line((left-5, py, left, py), fill=(55, 66, 77))
            draw.text((left-10, py), f'{value:.1f}', anchor='rm', fill=(55, 66, 77), font=self.small_font)
        draw.text((20, 263), 'Y (m)', fill=(55, 66, 77), font=self.small_font)
        draw.text((522, 503), 'X (m)', fill=(55, 66, 77), font=self.small_font)
        draw.text((20, 531), 'EEF path XY projection (heights omitted); white dot = current EEF',
                  fill=(55, 66, 77), font=self.small_font)
        draw.rectangle((20, 555, 32, 567), fill=(212, 73, 70))
        draw.text((38, 553), 'Occupied', fill=(55, 66, 77), font=self.small_font)
        draw.rectangle((118, 555, 130, 567), fill=(150, 157, 165))
        draw.text((136, 553), 'Unknown', fill=(55, 66, 77), font=self.small_font)
        gradient = self._colors(np.ones(120), np.linspace(0., self.value_cap_m, 120))
        gradient_image = Image.fromarray(np.tile(gradient[None, :, :], (12, 1, 1)), 'RGB')
        panel.paste(gradient_image, (222, 555))
        draw.text((350, 553), 'Free: model value 0 to 100+ mm', fill=(55, 66, 77), font=self.small_font)
        draw.text((20, 579), 'Slice is not a full-robot clearance check; model values are not ground truth.',
                  fill=(85, 94, 105), font=self.small_font)
        if self.size != 600:
            panel = panel.resize((self.size, self.size), Image.Resampling.LANCZOS)
        return panel
