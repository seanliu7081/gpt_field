"""PIL geometry for the reference obstacle-video style; saved records only.

No robot/model imports, no scene inference, no historical task constants.
``bounds`` is a fixed run-wide (lower_xyz, upper_xyz) display envelope. Its
bottom grid is a display reference plane, never an inferred table surface.
"""
from functools import lru_cache
import math

import numpy as np
from PIL import ImageFont


PROJECTION = np.array([[.78, -.63, 0.], [.30, .38, -.875]])
EDGES = [(i, j) for i in range(8) for j in range(i+1, 8) if i ^ j in (1, 2, 4)]
DEFAULT_COLORS = {
    "panel": "#15202d", "text": "#e9f0f7", "muted": "#9badc1",
    "grid": "#2f4052", "box": "#ffa74f", "planned": "#4bd1ef",
    "measured": "#ee73ba", "goal": "#ffdb6d", "current": "#e9f0f7",
    "contact": "#ff6968", "axis_x": "#f27c78", "axis_y": "#75ce90",
    "axis_z": "#7eaaf5",
}


@lru_cache(maxsize=16)
def _font(size, bold=False):
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{suffix}.ttf", size)


def _cube(lower, upper):
    return np.array([[x, y, z] for z in (lower[2], upper[2])
                     for y in (lower[1], upper[1]) for x in (lower[0], upper[0])])


def _xyz(values):
    points = np.asarray(values, dtype=float)
    return points.reshape(-1, 3) if points.size else np.empty((0, 3))


def draw_isometric(draw, rect, bounds, snapshot, state, contact_points=None, *, colors=None):
    """Draw exact recorded 3D paths, poses, boxes and optional contact XYZs.

    ``draw`` is a PIL ImageDraw. ``snapshot`` is the archived
    foundation_harness_planning object. ``contact_points`` is an iterable of
    actual evaluator positions; the caller labels them as current or historical.
    Historical hits must never be labelled as present contact.
    Optional ``colors`` overrides DEFAULT_COLORS.
    The returned function projects a world XYZ point into this plot's pixels.
    """
    color = {**DEFAULT_COLORS, **(colors or {})}
    left, top, right, bottom = rect
    lower, upper = np.asarray(bounds, dtype=float)
    projected = _cube(lower, upper) @ PROJECTION.T
    lo, hi = projected.min(0), projected.max(0)
    # Reserve header/footnote and enough room for coordinate-axis labels.
    plot_rect = (left+46, top+94, right-55, bottom-52)
    scale = min((plot_rect[2]-plot_rect[0])/max(hi[0]-lo[0], .001),
                (plot_rect[3]-plot_rect[1])/max(hi[1]-lo[1], .001))
    offset = np.array([(plot_rect[0]+plot_rect[2])/2,
                       (plot_rect[1]+plot_rect[3])/2])-scale*(lo+hi)/2

    def project(point):
        return tuple(np.asarray(point, dtype=float) @ PROJECTION.T*scale+offset)

    boxes = snapshot.get("obstacle_boxes", [])
    draw.rounded_rectangle(rect, radius=16, fill=color["panel"])
    draw.text((left+20, top+15), "3D OBSTACLE MAP / WORLD METRES",
              font=_font(24, True), fill=color["text"])
    subtitle = (f"Recorded Astra obstacle boxes: {len(boxes)}" if boxes else
                "Recorded obstacle boxes: 0 / no geometry supplied")
    draw.text((left+20, top+51), subtitle, font=_font(18), fill=color["box"])

    # Fixed scale reference grid in the same oblique projection as the user's
    # reference video. No geometry outside the archived map is inferred.
    for axis in (0, 1):
        for value in np.arange(math.ceil(lower[axis]*10)/10, upper[axis]+1e-8, .1):
            start, end = lower.copy(), lower.copy()
            start[axis] = end[axis] = value
            end[1-axis] = upper[1-axis]
            draw.line([project(start), project(end)], fill=color["grid"], width=1)
    for axis, label in enumerate("XYZ"):
        end = lower.copy()
        end[axis] = upper[axis]
        draw.line([project(lower), project(end)], fill=color["muted"], width=2)
        x, y = project(end)
        draw.text((x+6, y-8), label+" m", font=_font(17), fill=color["muted"])

    for index, item in enumerate(boxes):
        vertices = _cube(np.asarray(item["lower_xyz_m"], float),
                         np.asarray(item["upper_xyz_m"], float))
        screen = [project(point) for point in vertices]
        for first, second in EDGES:
            draw.line([screen[first], screen[second]], fill=color["box"], width=2)
        x, y = project(vertices[4:].mean(0))
        draw.text((x+3, y-22), str(item.get("id", f"B{index+1}")),
                  font=_font(16), fill=color["box"])

    for field, key, width in (("measured_path", "measured", 3),
                              ("selected_path", "planned", 4)):
        points = _xyz(snapshot.get(field, []))
        if len(points) > 1:
            draw.line([project(point) for point in points], fill=color[key], width=width)
        if field == "selected_path":
            for point in points:
                x, y = project(point)
                draw.ellipse((x-2, y-2, x+2, y+2), fill=color[key])

    for pose in snapshot.get("keyposes", []):
        point = np.asarray(pose["position"], dtype=float)
        x, y = project(point)
        draw.line((x-8, y, x+8, y), fill=color["goal"], width=3)
        draw.line((x, y-8, x, y+8), fill=color["goal"], width=3)
        # Only recorded rotation is drawn. Missing rotation stays missing.
        if pose.get("rotation") is not None:
            rotation = np.asarray(pose["rotation"], dtype=float).reshape(3, 3)
            for axis, key in enumerate(("axis_x", "axis_y", "axis_z")):
                draw.line([(x, y), project(point+rotation[:, axis]*.035)],
                          fill=color[key], width=3)
    x, y = project(state[:3])
    draw.ellipse((x-7, y-7, x+7, y+7), fill=color["current"],
                 outline=color["panel"], width=2)

    for point in (() if contact_points is None else contact_points):
        x, y = project(point)
        draw.ellipse((x-11, y-11, x+11, y+11), outline=color["contact"], width=3)
        draw.ellipse((x-3, y-3, x+3, y+3), fill=color["contact"])
    draw.text((left+20, bottom-28), "Reference grid only; not a measured surface",
              font=_font(15), fill=color["muted"])
    return project
