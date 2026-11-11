"""Color distance slices for the native-RGB lazy-grid profile.

Uniform, box-free fields admit an exact Euclidean distance to the recorded
spherical goal region. Sample only three planes for the display; never allocate
or claim a solved 161-cubed map. Weighted/obstructed fields remain path-only when
no solved distance map is supplied. Display reconstruction never feeds control.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw

from .hybrid_video import HybridVideoRecorder
from .planner import PlannerConfig
from .video import BG, CYAN, MAGENTA, MUTED, PANEL, TEXT, WIDTH, YELLOW, _font, _short, project_world


_LOW_COLOR = np.asarray((28, 135, 137), dtype=float)
_HIGH_COLOR = np.asarray((227, 124, 63), dtype=float)
_MASK_COLOR = (78, 87, 104)
_PLANES = ((0, 1, 2), (0, 2, 1), (1, 2, 0))


def _heat_colors(values, maximum):
    fractions = np.clip(np.nan_to_num(values, nan=0.) / maximum, 0., 1.)
    colors = (_LOW_COLOR + fractions[..., None] * (_HIGH_COLOR - _LOW_COLOR)).astype(np.uint8)
    colors[~np.isfinite(values)] = _MASK_COLOR
    return colors


class SpeedFirstV3VideoRecorder(HybridVideoRecorder):
    def __init__(self, *args, planner_config=None, **kwargs):
        # V3 uses these same fixed PlannerConfig workspace bounds. An offline
        # reader for a custom profile may supply its recorded config explicitly.
        self._display_config = planner_config if planner_config is not None else PlannerConfig()
        self._workspace_source = "supplied_planner_config" if planner_config is not None else "v3_default_planner_config"
        self._slice_cache_key = None
        self._slice_cache = None
        super().__init__(*args, **kwargs)

    def _set_fresh_field(self):
        self.field = None
        self.field_step = self.step

    def _distance_slices(self):
        """Return bounded display-only samples and truthful source metadata.

        Plan-free routing events do not reuse an earlier search or its tolerance.
        An analytic path can coexist with off-path boxes, so the recorded search
        method alone is insufficient to call a whole slice an exact distance.
        """
        unavailable = {"source": "unavailable", "reason": "no_current_field_plan"}
        if self.controller != "coarse_field" or self.details.get("field_center") is None:
            return unavailable
        compact = self.details.get("effective_compact_field")
        search = (self.details.get("plan") or {}).get("search") or {}
        if not isinstance(compact, dict) or not search:
            return unavailable
        if any(compact.get(key) for key in ("repulsors", "blocked_boxes", "unknown_boxes")):
            return {"source": "unavailable", "reason": "weighted_or_masked_distance_not_solved_for_display"}
        if not all(key in compact for key in ("repulsors", "blocked_boxes", "unknown_boxes", "goal_offset_m")):
            return {"source": "unavailable", "reason": "incomplete_compact_field"}
        if search.get("method") != "analytic_uniform_goal_distance":
            return {"source": "unavailable", "reason": "no_current_analytic_distance_plan"}
        try:
            center = np.asarray(self.details["field_center"], dtype=float)
            goal = np.asarray(search["goal_xyz"], dtype=float)
            tolerance = float(search["goal_tolerance_m"])
            if center.shape != (3,) or goal.shape != (3,) or not np.all(np.isfinite(np.r_[center, goal, tolerance])):
                raise ValueError("invalid geometry")
            if tolerance <= 0 or not np.allclose(center + compact["goal_offset_m"], goal, rtol=0., atol=1e-10):
                raise ValueError("goal does not match current compact field")
        except (KeyError, TypeError, ValueError):
            return {"source": "unavailable", "reason": "missing_or_inconsistent_recorded_goal"}
        lower, upper = center - self.grid.half_width_m, center + self.grid.half_width_m
        workspace_min = np.asarray(self._display_config.workspace_min)
        workspace_max = np.asarray(self._display_config.workspace_max)
        if np.any(goal < np.maximum(lower, workspace_min)) or np.any(goal > np.minimum(upper, workspace_max)):
            return {"source": "unavailable", "reason": "recorded_goal_outside_display_domain"}
        indices = tuple(np.clip(np.rint((np.asarray(self.observation.state[:3]) - lower) / self.grid.spacing_m),
                                0, self.grid.size - 1).astype(int))
        key = (self.grid.size, self.grid.spacing_m, tuple(center), tuple(goal), tolerance,
               tuple(workspace_min), tuple(workspace_max), indices)
        if key == self._slice_cache_key:
            return self._slice_cache
        # Stable shared color scale across the three moving slices, with no 3D
        # cost array. Max distance on an axis-aligned cube occurs at a corner.
        farthest_delta = np.maximum(np.abs(lower - goal), np.abs(upper - goal))
        maximum = max(float(np.linalg.norm(farthest_delta) - tolerance), self.grid.spacing_m)
        planes = []
        for horizontal, vertical, fixed in _PLANES:
            x = np.linspace(lower[horizontal], upper[horizontal], self.grid.size)
            y = np.linspace(upper[vertical], lower[vertical], self.grid.size)
            xx, yy = np.meshgrid(x, y)
            fixed_world = float(lower[fixed] + indices[fixed] * self.grid.spacing_m)
            distance = np.maximum(np.sqrt((xx - goal[horizontal]) ** 2 + (yy - goal[vertical]) ** 2
                                  + (fixed_world - goal[fixed]) ** 2) - tolerance, 0.)
            valid = ((xx >= workspace_min[horizontal] - 1e-12) & (xx <= workspace_max[horizontal] + 1e-12)
                     & (yy >= workspace_min[vertical] - 1e-12) & (yy <= workspace_max[vertical] + 1e-12)
                     & (workspace_min[fixed] - 1e-12 <= fixed_world <= workspace_max[fixed] + 1e-12))
            distance[~valid] = np.nan
            planes.append({"axes": (horizontal, vertical), "fixed_axis": fixed, "fixed_world_m": fixed_world,
                           "values": distance, "image": Image.fromarray(_heat_colors(distance, maximum))})
        result = {"source": "reconstructed_analytic_distance_slices", "reason": None,
                  "goal_xyz": goal.tolist(), "goal_tolerance_m": tolerance,
                  "color_range_m": [0., maximum], "planes": planes}
        self._slice_cache_key, self._slice_cache = key, result
        return result

    def _emit(self, event, repeats, **extra):
        field_active = self.controller == "coarse_field"
        display = self._distance_slices()
        has_color = display["source"] == "reconstructed_analytic_distance_slices"
        metadata = {
            "field_display_source": display["source"],
            "field_display_unavailable_reason": display["reason"],
            "field_display_cost_unit": "m" if has_color else None,
            "distance_slice_heatmaps_rendered": has_color,
            "distance_display_scope": "three_sampled_2d_planes_not_dense_3d_solve" if has_color else None,
            "distance_slice_shape": [self.grid.size, self.grid.size] if has_color else None,
            "distance_slice_fixed_coordinates_m": [p["fixed_world_m"] for p in display.get("planes", [])],
            "distance_display_goal_xyz": display.get("goal_xyz"),
            "distance_display_goal_tolerance_m": display.get("goal_tolerance_m"),
            "distance_display_color_range_m": display.get("color_range_m"),
            "distance_display_workspace_min": list(self._display_config.workspace_min) if has_color else None,
            "distance_display_workspace_max": list(self._display_config.workspace_max) if has_color else None,
            "distance_display_workspace_source": self._workspace_source if has_color else None,
            "dense_distance_map_rendered": False,
            "field_center": deepcopy(self.details.get("field_center")) if field_active else None,
            "effective_compact_field": deepcopy(self.details.get("effective_compact_field")) if field_active else None,
            "local_visual_feedback": deepcopy(self.details.get("local_visual_feedback")),
            "local_goal_updates": self.details.get("local_goal_updates", 0),
            "local_field_revision": self.details.get("local_field_revision", 0),
            "local_metadata_scope": "pre_action_decision",
            "task_success_certified": False,
        }
        super()._emit(event, repeats, **{**metadata, **extra})

    def render_frame(self):
        canvas = super().render_frame()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, WIDTH, 50), fill=BG)
        draw.text((24, 17), "ASTRA / LIBERO | NATIVE 512 RGB + LOCAL CORRECTION + 1 mm FIELD",
                  font=_font(25, True), fill=TEXT)
        return canvas

    def _field_panels(self, canvas, draw):
        display = self._distance_slices()
        has_color = display["source"] == "reconstructed_analytic_distance_slices"
        title = "RECONSTRUCTED ANALYTIC DISTANCE SLICES + RECORDED PATH" if has_color else "RECORDED PATH | DISTANCE COLORS UNAVAILABLE"
        draw.text((425, 142), title, font=_font(20, True), fill=TEXT)
        subtitle = (f"{self.grid.size} x {self.grid.size} samples per plane, {self.grid.spacing_m:g} m spacing. "
                    "Display reconstruction; no full 3D distance solve." if has_color else
                    "No current uniform, box-free distance solve to reconstruct. No preceding field colors are reused.")
        draw.text((425, 174), subtitle, font=_font(15), fill=MUTED)
        center = self.details.get("field_center")
        if self.controller != "coarse_field" or center is None:
            draw.rounded_rectangle((425, 206, 1576, 600), radius=12, fill=PANEL)
            draw.text((450, 245), "Awaiting a current local Field plan", font=_font(25), fill=MUTED)
            return
        center = np.asarray(center, dtype=float)
        half_extent = self.grid.half_width_m
        lower, upper = center - half_extent, center + half_extent
        plan = self.details.get("plan") or {}
        candidates = plan.get("candidates", [])
        selected = plan.get("selected_index")
        for number, (panel_x, axes) in enumerate(((425, (0, 1)), (816, (0, 2)), (1207, (1, 2)))):
            side = 288
            plot_x, plot_y = panel_x + 54, 245
            draw.rounded_rectangle((panel_x, 207, panel_x + 369, 553), radius=10, fill=PANEL)
            names = "XYZ"
            if has_color:
                plane = display["planes"][number]
                label = f"{names[axes[0]]}{names[axes[1]]} at {names[plane['fixed_axis']]}={plane['fixed_world_m']:+.3f} m"
                plot = plane["image"].resize((side, side), Image.Resampling.NEAREST)
            else:
                label = f"{names[axes[0]]}{names[axes[1]]} path projection (meters)"
                plot = Image.new("RGB", (side, side), (22, 33, 44))
            draw.text((panel_x + 14, 216), label, font=_font(16, True), fill=TEXT)
            pd = ImageDraw.Draw(plot)
            if not has_color:
                for fraction in (0., .25, .5, .75, 1.):
                    p = round(fraction * (side - 1))
                    pd.line((p, 0, p, side - 1), fill=(41, 53, 65))
                    pd.line((0, p, side - 1, p), fill=(41, 53, 65))
            def line(points, color, width):
                if len(points) < 2:
                    return
                pixels = project_world(points, lower, upper, axes, side)
                pd.line([tuple(p) for p in pixels], fill=color, width=width)
            for candidate in candidates:
                if candidate.get("valid"):
                    line(candidate.get("positions", []), CYAN, 1)
            line(self.trace, MAGENTA, 2)
            if selected is not None and 0 <= selected < len(candidates):
                line(candidates[selected].get("positions", []), YELLOW, 4)
            px, py = project_world([self.observation.state[:3]], lower, upper, axes, side)[0]
            pd.ellipse((px - 5, py - 5, px + 5, py + 5), fill=TEXT)
            reference = (self.details.get("tracking") or {}).get("reference_xyz")
            if reference is not None:
                px, py = project_world([reference], lower, upper, axes, side)[0]
                pd.rectangle((px - 5, py - 5, px + 5, py + 5), outline=YELLOW, width=2)
            canvas.paste(plot, (plot_x, plot_y))
            for fraction in (0., .5, 1.):
                value = lower[axes[0]] + fraction * (upper[axes[0]] - lower[axes[0]])
                draw.text((plot_x + fraction * (side - 1) - 22, plot_y + side + 3),
                          f"{value:+.2f}", font=_font(11), fill=MUTED)
                value = lower[axes[1]] + fraction * (upper[axes[1]] - lower[axes[1]])
                draw.text((panel_x + 3, plot_y + (1 - fraction) * (side - 1) - 6),
                          f"{value:+.2f}", font=_font(11), fill=MUTED)
        search = plan.get("search") or {}
        if has_color:
            maximum = display["color_range_m"][1]
            gradient = _heat_colors(np.linspace(0., maximum, 270)[None, :], maximum)
            canvas.paste(Image.fromarray(gradient).resize((270, 12)), (425, 567))
            draw.text((425, 581), "0 m (lower)", font=_font(12), fill=MUTED)
            draw.text((602, 581), f"{maximum:.3f} m (higher)", font=_font(12), fill=MUTED)
            draw.text((720, 565), "Distance to recorded goal region | gray: outside workspace", font=_font(16), fill=TEXT)
        else:
            draw.text((425, 565), _short(f"Distance colors: {display['reason']}", 130), font=_font(16), fill=MUTED)
        draw.text((425, 600), _short(f"Planner: {search.get('method', 'unavailable')} | "
                  f"{self.grid.size} cubed lattice, {self.grid.spacing_m:g} m spacing, {2 * half_extent:g} m extent", 145),
                  font=_font(16, True), fill=TEXT)
        feedback = self.details.get("local_visual_feedback") or {}
        label = feedback.get("status", feedback.get("reason", "not active"))
        draw.text((425, 625), _short(f"Local object feedback: {label} | "
                  f"applied goal updates: {self.details.get('local_goal_updates', 0)}", 145),
                  font=_font(15), fill=MUTED)
