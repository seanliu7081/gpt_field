"""Asynchronous V3 video with explicitly labelled Astra obstacle-box geometry."""
from itertools import product

import numpy as np
from PIL import ImageDraw

from ..foundation_vision_harness_V1.recording import (
    AsyncVideoRecorder as _AsyncRecorder, _V1VideoRecorder,
)
from ..foundation_harness.video import PANELS, _points
from ..agent_speedup_v1_video import BG, PANEL, TEXT, CYAN, MUTED, _font

BOX_COLOUR = (225, 150, 60)
POSE_COLOUR = (245, 210, 70)
PATH_COLOUR = (225, 95, 175)
BOX_EDGES = [(i, i ^ bit) for i in range(8) for bit in (1, 2, 4) if i < (i ^ bit)]


def _boxes(snapshot):
    """Display explicit AABBs as wireframes, never as synthetic depth points."""
    result = []
    for index, item in enumerate(snapshot.get("obstacle_boxes", [])):
        if not isinstance(item, dict):
            continue
        try:
            lo, hi = np.asarray(item["lower_xyz_m"], float), np.asarray(item["upper_xyz_m"], float)
            if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(hi < lo):
                continue
            corners = np.array([[hi[axis] if side[axis] else lo[axis] for axis in range(3)]
                                for side in product((0, 1), repeat=3)])
            result.append((str(item.get("id", item.get("name", f"box {index+1}"))), corners))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def draw_box_projection(draw, box, axes, snapshot, current):
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill=PANEL)
    draw.text((left+12, top+9), "ASTRA BOXES " + "XYZ"[axes[0]]+"XYZ"[axes[1]],
              font=_font(14, True), fill=BOX_COLOUR)
    boxes = _boxes(snapshot)
    selected = _points(snapshot.get("selected_path", []))
    measured = _points(snapshot.get("measured_path", []))
    present = _points([current])
    poses = [(item, _points([item.get("position")])) for item in snapshot.get("keyposes", [])
             if isinstance(item, dict)]
    poses = [(item, points[0]) for item, points in poses if len(points)]
    groups = [selected, measured, present] + [vertices for _, vertices in boxes]
    if poses:
        groups.append(np.asarray([point for _, point in poses]))
    all_points = np.concatenate(groups)
    if not len(all_points):
        draw.text((left+14, top+60), "No current box or pose proposal", font=_font(13), fill=MUTED)
        return
    values = all_points[:, list(axes)]
    lo, hi = values.min(0), values.max(0)
    centre, span = (lo+hi)/2., np.maximum(hi-lo, .12)*1.12
    lo, hi = centre-span/2., centre+span/2.
    x0, x1, y0, y1 = left+32, right-20, top+43, bottom-34

    def project(point):
        uv = (np.asarray(point)[list(axes)]-lo)/(hi-lo)
        return float(x0+uv[0]*(x1-x0)), float(y1-uv[1]*(y1-y0))

    for label, vertices in boxes:
        for first, second in BOX_EDGES:
            draw.line([project(vertices[first]), project(vertices[second])], fill=BOX_COLOUR, width=1)
        x, y = project(vertices.mean(0))
        draw.text((x+2, y+2), label[:20], font=_font(9), fill=BOX_COLOUR)
    for path, colour, width in ((selected, POSE_COLOUR, 2), (measured, PATH_COLOUR, 3)):
        if len(path) > 1:
            draw.line([project(point) for point in path], fill=colour, width=width)
    for index, (item, point) in enumerate(poses):
        x, y = project(point)
        draw.rectangle((x-3, y-3, x+3, y+3), outline=POSE_COLOUR)
        draw.text((x+4, y-9), str(index+1), font=_font(10), fill=POSE_COLOUR)
        try:
            rotation = np.asarray(item.get("rotation"), float).reshape(3, 3)
            if np.isfinite(rotation).all():
                for axis, colour in enumerate(((220, 90, 90), (100, 210, 120), (100, 140, 235))):
                    draw.line([(x, y), project(point+rotation[:, axis]*.025)], fill=colour, width=1)
        except (TypeError, ValueError):
            pass
    if len(present):
        x, y = project(present[0])
        draw.ellipse((x-4, y-4, x+4, y+4), fill=TEXT)
    draw.text((left+12, bottom-25),
              f"{lo[0]:+.2f}..{hi[0]:+.2f} / {lo[1]:+.2f}..{hi[1]:+.2f} m", font=_font(10), fill=MUTED)


class _V3VideoRecorder(_V1VideoRecorder):
    def _chunk_panel(self, draw):
        super()._chunk_panel(draw)
        draw.rectangle((440, 150, 1555, 183), fill=PANEL)
        draw.text((444, 155), "ASTRA KEYPOSES / MEASURED V2 FRAGMENTS / OPTIONAL VISION",
                  font=_font(18, True), fill=CYAN)

    def render_frame(self):
        canvas = super().render_frame()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((20, 10, 1580, 49), fill=BG)
        draw.text((24, 17), "FOUNDATION VISION HARNESS V3 | ASTRA CONTROL + ON-DEMAND TOOLS",
                  font=_font(24, True), fill=TEXT)
        if self.observation is not None:
            for box, axes in PANELS:
                self._guarded("Astra obstacle boxes", box, draw,
                    lambda box=box, axes=axes: draw_box_projection(
                        draw, box, axes, self.planning_debug, self.observation.state[:3]))
        draw.rectangle((20, 919, 1580, 961), fill=BG)
        draw.text((24, 923), "Orange: Astra approximate boxes | yellow: nominal keyposes/path | pink: measured path | white: current EEF",
                  font=_font(13), fill=MUTED)
        draw.text((24, 943), "Boxes are model proposals, not DA3 measurements. Sampled asynchronous playback; actual capture times remain in the event index.",
                  font=_font(12), fill=MUTED)
        return canvas

    def _emit(self, event, repeats, **extra):
        return super()._emit(event, repeats, **{**extra,
            "harness_version": "foundation_vision_harness_V3", "policy": "foundation_vision_harness_V3",
            "video_version": "foundation_vision_harness_V3", "obstacle_geometry_source": "astra_approximate_boxes"})


class AsyncVideoRecorder(_AsyncRecorder):
    def __init__(self, path, *, episode_id="foundation_vision_harness_V3", **kwargs):
        kwargs.setdefault("renderer_factory", _V3VideoRecorder)
        super().__init__(path, episode_id=episode_id, **kwargs)
        self._worker.name = "fvh-v3-video"
