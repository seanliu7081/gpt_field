"""Recorder-only visualization of predicted surfaces and local geometric plans."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from PIL import ImageDraw

from ..agent_speedup_v2_video import AgentSpeedUpV2VideoRecorder
from ..video import BG, CYAN, MAGENTA, MUTED, PANEL, TEXT, YELLOW, _font, _short
from .types import jsonable

VERSION = "foundation-harness-video-2"
PANELS = (((425, 443, 795, 651), (0, 1)), ((815, 443, 1185, 651), (0, 2)),
          ((1205, 443, 1576, 651), (2, 1)))


def _points(value):
    try:
        points = np.asarray(value, dtype=float).reshape(-1, 3)
        return points[np.isfinite(points).all(axis=1)]
    except (TypeError, ValueError):
        return np.empty((0, 3))


def draw_projection(draw, box, axes, snapshot, current):
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill=PANEL)
    draw.text((left+12, top+9), "PREDICTED GEOMETRY " + "XYZ"[axes[0]]+"XYZ"[axes[1]],
              font=_font(14, True), fill=CYAN)
    cloud = _points(snapshot.get("scene_points", []))
    selected = _points(snapshot.get("selected_path", []))
    measured = _points(snapshot.get("measured_path", []))
    poses = [(item, _points([item.get("position")])) for item in snapshot.get("keyposes", [])
             if isinstance(item, dict)]
    poses = [(item, point[0]) for item, point in poses if len(point)]
    present = _points([current])
    groups = [cloud, selected, measured, present]
    if poses:
        groups.append(np.asarray([point for _, point in poses]))
    all_points = np.concatenate(groups)
    if not len(all_points):
        draw.text((left+14, top+60), "No geometric hypothesis yet", font=_font(13), fill=MUTED)
        return
    values = all_points[:, list(axes)]
    low, high = values.min(0), values.max(0)
    centre = (low+high)/2
    span = np.maximum(high-low, .12)*1.12
    low, high = centre-span/2, centre+span/2
    x0, x1, y0, y1 = left+32, right-20, top+43, bottom-34

    def project(point):
        uv = (np.asarray(point)[list(axes)]-low)/(high-low)
        return float(x0+uv[0]*(x1-x0)), float(y1-uv[1]*(y1-y0))

    # Bounded display sampling only; the planner retains its own geometric data.
    strength = np.asarray(snapshot.get("scene_point_strong", []), dtype=bool)
    if strength.shape != (len(cloud),):
        strength = np.zeros(len(cloud), bool)
    for index in np.linspace(0, len(cloud)-1, min(len(cloud), 1200), dtype=int) if len(cloud) else []:
        point = cloud[index]
        x, y = project(point)
        draw.point((x, y), fill=(70, 155, 195) if strength[index] else (80, 85, 90))
    for path, colour, width in ((selected, YELLOW, 2), (measured, MAGENTA, 3)):
        if len(path) > 1:
            draw.line([project(point) for point in path], fill=colour, width=width)
    for index, (item, point) in enumerate(poses):
        x, y = project(point)
        draw.rectangle((x-3, y-3, x+3, y+3), outline=YELLOW)
        draw.text((x+4, y-9), str(index+1), font=_font(10), fill=YELLOW)
        try:
            rotation = np.asarray(item.get("rotation"), float).reshape(3, 3)
            if np.isfinite(rotation).all():
                for axis, colour in enumerate(((220, 90, 90), (100, 210, 120), (100, 140, 235))):
                    draw.line([(x, y), project(point+rotation[:, axis]*.025)], fill=colour, width=1)
        except (ValueError, TypeError):
            pass
    if len(present):
        x, y = project(present[0])
        draw.ellipse((x-4, y-4, x+4, y+4), fill=TEXT)
    draw.text((left+12, bottom-25),
              f"{low[0]:+.2f}..{high[0]:+.2f} / {low[1]:+.2f}..{high[1]:+.2f} m",
              font=_font(10), fill=MUTED)


class FoundationHarnessVideoRecorder(AgentSpeedUpV2VideoRecorder):
    def __init__(self, *args, planning_provider=None, **kwargs):
        self.planning_provider = planning_provider
        self.planning_debug = {"status": "no_observation", "keyposes": [], "selected_path": []}
        super().__init__(*args, **kwargs)

    def _refresh(self):
        # Replace, never merge: a new decision cannot inherit an old active path.
        self.planning_debug = {"status": "unavailable", "keyposes": [], "selected_path": []}
        if self.planning_provider is not None:
            try:
                value = self.planning_provider()
                if isinstance(value, dict):
                    self.planning_debug = jsonable(deepcopy(value))
            except Exception as error:
                self.planning_debug["status"] = "provider_error:"+type(error).__name__

    def decision(self, observation, decision, *, decision_latency_s=None):
        self._refresh()
        super().decision(observation, decision, decision_latency_s=decision_latency_s)

    def executed(self, observation, *, step, success, sim_status=None, feedback=None, history_stats=None):
        self._refresh()
        if success or (sim_status or {}).get("terminated"):
            self.planning_debug["status"] = "terminal"
            self.planning_debug["selected_path"] = []
        super().executed(observation, step=step, success=success, sim_status=sim_status,
                         feedback=feedback, history_stats=history_stats)

    def _emit(self, event, repeats, **extra):
        if event == "terminal":
            self.planning_debug["status"] = "terminal"
            self.planning_debug["selected_path"] = []
        super()._emit(event, repeats, **{**extra,
            "foundation_harness_planning": deepcopy(self.planning_debug),
            "foundation_harness_video_version": VERSION})

    def _trace_panel(self, draw, box, axes):
        pass  # One three-projection strip replaces the base recorder's two traces.

    def _chunk_panel(self, draw):
        draw.rounded_rectangle((425, 139, 1576, 427), radius=12, fill=PANEL)
        view = self.planning_debug
        draw.text((444, 155), "ASTRA KEYPOSES / LOCAL TRAJECTORY EXECUTION", font=_font(19, True), fill=CYAN)
        line = f"step {view.get('step', '-')} | stage {view.get('stage', '-')} | object {view.get('object_id', '-')}"
        draw.text((444, 190), _short(line, 120), font=_font(17), fill=TEXT)
        draw.text((444, 220), _short("Status: "+str(view.get("status", "unknown")), 120), font=_font(15), fill=MUTED)
        poses = view.get("keyposes", [])
        for index, pose in enumerate(poses[:7]):
            point = _points([pose.get("position")]) if isinstance(pose, dict) else []
            xyz = "unknown" if not len(point) else ", ".join(f"{v:+.3f}" for v in point[0])
            label = f"{index+1}. {pose.get('name', 'pose')}  [{xyz}] m" if isinstance(pose, dict) else "invalid pose"
            draw.text((444+(index//4)*550, 257+(index%4)*31), _short(label, 60), font=_font(15), fill=YELLOW)
        if not poses:
            draw.text((444, 264), "No current local keypose plan", font=_font(16), fill=MUTED)

    def render_frame(self):
        canvas = super().render_frame()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((20, 10, 1580, 49), fill=BG)
        draw.text((24, 17), "ASTRA / LIBERO | FOUNDATION HARNESS | SEMANTIC + LOCAL GEOMETRY",
                  font=_font(25, True), fill=TEXT)
        if self.observation is not None:
            for box, axes in PANELS:
                self._guarded("predicted geometry", box, draw,
                    lambda box=box, axes=axes: draw_projection(draw, box, axes,
                        self.planning_debug, self.observation.state[:3]))
        draw.rectangle((20, 919, 1580, 959), fill=BG)
        draw.text((24, 923), "Blue: supported surfaces | grey: uncertain geometry | yellow: keyposes / path | pink: measured path | white: current EEF",
                  font=_font(13), fill=MUTED)
        draw.text((24, 943), "Unknown geometry remains unknown. No collision-free or grasp guarantee. Annotated 20 Hz playback; API waits omitted.",
                  font=_font(12), fill=MUTED)
        return canvas
