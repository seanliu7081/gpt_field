"""Revision 7 playback with measured paths and a derived translation control field.

This runner-only view does not change the controller or call a simulator/model.
Arrows visualize delta = clip_norm(target - position, profile step limit), the
existing fragment translation law. They are neither a learned scalar planning
field nor an obstacle/collision model. All targets retain their recorded source.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from .agent_speedup_v2_user_check_fragments import PROFILES
from .agent_speedup_v2_video import AgentSpeedUpV2VideoRecorder
from .video import BG, CYAN, MAGENTA, MUTED, PANEL, TEXT, YELLOW, _font, _short
from PIL import ImageDraw

PLANNING_VIDEO_VERSION = "user-check-planning-video-3"
PLANNING_PANELS = (
    ((425, 443, 795, 651), (0, 1)),
    ((815, 443, 1185, 651), (0, 2)),
    ((1205, 443, 1576, 651), (2, 1)),
)
PLANNING_STRIP = (425, 443, 1576, 651)
FIELD_COLOUR = (49, 91, 105)
PLANNING_CAPTION_BOX = (24, 919, 1576, 959)
PLANNING_CAPTION_LINES = (
    "Control field (XY / XZ / ZY): blue arrows show capped motion toward the yellow + target; white dot = gripper, pink line = measured path.",
    "Dashed yellow line = entry-to-target guide; boxes = pending targets. Target error only, not obstacle distance or collision safety. 20 Hz; API waits omitted.",
)


def _xyz(value):
    if (not isinstance(value, (list, tuple, np.ndarray)) or (isinstance(value, np.ndarray) and value.ndim == 0) or len(value) < 3
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                   or not math.isfinite(float(v)) for v in value[:3])):
        return None
    return [float(v) for v in value[:3]]


def fragment_from_feedback(feedback):
    """Return a detached authoritative transition fragment, never a model belief."""
    if not isinstance(feedback, Mapping):
        return None
    primitive = feedback.get("primitive_feedback")
    fragment = primitive.get("fragment") if isinstance(primitive, Mapping) else None
    return deepcopy(dict(fragment)) if isinstance(fragment, Mapping) else None


def make_planning_view(fragment=None, *, source="unavailable",
                       execution_kind="framework_fragment", event=None):
    """Build JSON-safe diagnostics; absent/stopped targets are never carried forward."""
    result = {
        "version": PLANNING_VIDEO_VERSION,
        "planes": ["XY", "XZ", "ZY"],
        "caption_lines": list(PLANNING_CAPTION_LINES),
        "method": "derived_clipped_translation_control",
        "source": str(source), "active": False, "target_status": "unavailable",
        "target_xyz_m": None, "entry_xyz_m": None, "profile": None,
        "translation_limit_m": None, "current_skill": None,
        "current_segment_index": None, "pending_absolute_targets": [],
        "unresolved_relative_segments": 0, "scalar_planning_field": False,
        "obstacle_model": False, "collision_free_claim": False,
        "formula": "delta = target - position; clip Euclidean norm to profile action_translation_m",
    }
    if event in ("observation", "terminal"):
        result["target_status"] = "awaiting_command" if event == "observation" else "terminal"
        return result
    if not isinstance(execution_kind, str) or not execution_kind.startswith("framework_fragment"):
        result["target_status"] = "not_fragment_control"
        return result
    if not isinstance(fragment, Mapping):
        return result
    result["current_skill"] = str(fragment.get("current_skill") or "")
    index = fragment.get("current_segment_index")
    if type(index) is int:
        result["current_segment_index"] = index
    if (fragment.get("status") != "running"
            or fragment.get("current_segment_status") != "running"
            or fragment.get("handoff_kind") != "executing"
            or fragment.get("observation_requested") is True):
        result["target_status"] = str(fragment.get("reason") or fragment.get("handoff_kind") or "inactive")
        return result
    records = fragment.get("segments")
    current = next((r for r in records if isinstance(r, Mapping) and r.get("index") == index), None) \
        if isinstance(records, (list, tuple)) else None
    request = current.get("request") if isinstance(current, Mapping) else None
    profile = request.get("profile") if isinstance(request, Mapping) else None
    target = _xyz(fragment.get("target_xyz_m"))
    entry = _xyz(current.get("entry_pose_wxyz")) if isinstance(current, Mapping) else None
    if not isinstance(profile, str) or profile not in PROFILES or target is None or entry is None:
        result["target_status"] = "incomplete_recorded_target"
        return result
    result.update(active=True, target_status="active_resolved", target_xyz_m=target,
                  entry_xyz_m=entry, profile=profile,
                  translation_limit_m=float(PROFILES[profile]["action_translation_m"]))
    pending = fragment.get("pending_segments")
    if isinstance(pending, (list, tuple)):
        for offset, segment in enumerate(pending):
            if not isinstance(segment, Mapping):
                continue
            skill = segment.get("skill")
            if skill == "move_to_pose":
                xyz = _xyz(segment.get("xyz_m"))
                if xyz is not None:
                    result["pending_absolute_targets"].append({
                        "segment_index": index + offset + 1 if type(index) is int else None,
                        "xyz_m": xyz, "status": "requested_unexecuted"})
            elif skill in ("move_relative", "approach"):
                result["unresolved_relative_segments"] += 1
    return result


def control_delta(target, position, limit):
    """The existing fragment's Euclidean-clipped translation command in meters."""
    target, position = _xyz(target), _xyz(position)
    if target is None or position is None or not math.isfinite(float(limit)) or limit <= 0:
        raise ValueError("Control field requires finite target, position and positive limit")
    delta = np.asarray(target) - np.asarray(position)
    length = float(np.linalg.norm(delta))
    if length > limit:
        delta *= float(limit) / length
    return delta


def _dashed(draw, a, b, *, fill, width=1, dash=5):
    a, b = np.asarray(a), np.asarray(b)
    length = float(np.linalg.norm(b - a))
    if length <= 0:
        return
    for distance in np.arange(0, length, 2 * dash):
        p, q = a + (b-a) * (distance/length), a + (b-a) * (min(distance+dash, length)/length)
        draw.line([tuple(p), tuple(q)], fill=fill, width=width)


def _arrow(draw, a, b, colour):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    delta = b - a
    length = float(np.linalg.norm(delta))
    if length < 1.5:
        return
    draw.line([tuple(a), tuple(b)], fill=colour, width=1)
    direction = delta / length
    side = np.array([-direction[1], direction[0]])
    tip = b - direction * min(3.5, length*.4)
    draw.line([tuple(tip+side*2), tuple(b), tuple(tip-side*2)], fill=colour, width=1)


def draw_planning_panel(draw, box, axes, *, trace, state, view):
    """Pure existing-box renderer for live playback and historical frame previews."""
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill=PANEL)
    def label_text(x, y, text, size, *, colour=MUTED, bold=False):
        font = _font(size, bold)
        width = right - x - 14
        clipped = str(text)
        if draw.textlength(clipped, font=font) > width:
            while clipped and draw.textlength(clipped + "...", font=font) > width:
                clipped = clipped[:-1]
            clipped += "..."
        draw.text((x, y), clipped, font=font, fill=colour)
    names = "XYZ"
    active = bool(view.get("active"))
    label = "CONTROL FIELD" if active else "PLANNING"
    suffix = " (derived)" if active else " / measured path"
    label_text(left+16, top+10, f"{label} {names[axes[0]]}{names[axes[1]]}{suffix}",
               15, bold=True, colour=CYAN if active else MUTED)
    note = (f"{view['current_skill']} | {view['profile']} | cap {view['translation_limit_m']*1000:g} mm"
            if active else "No active fragment target: " + str(view.get("target_status", "unavailable")))
    if active and view["unresolved_relative_segments"]:
        note += f" | {view['unresolved_relative_segments']} relative pending"
    label_text(left+16, top+32, note, 11)
    current = _xyz(state)
    points = [xyz for item in trace if (xyz := _xyz(item)) is not None]
    if current is not None and (not points or points[-1] != current):
        points.append(current)
    if not points:
        return
    bounds_points = list(points)
    if active:
        bounds_points.extend((view["target_xyz_m"], view["entry_xyz_m"]))
        bounds_points.extend(item["xyz_m"] for item in view["pending_absolute_targets"])
    values = np.asarray(bounds_points)[:, axes]
    low, high = values.min(axis=0), values.max(axis=0)
    center = (low + high) / 2
    span = np.maximum(high-low, .06) * 1.20
    low, high = center-span/2, center+span/2
    x0, x1, y0, y1 = left+40, right-25, top+57, bottom-44

    def project(xyz):
        uv = (np.asarray(xyz)[list(axes)]-low)/(high-low)
        return (x0+float(uv[0])*(x1-x0), y1-float(uv[1])*(y1-y0))

    if active and current is not None:
        for x in np.linspace(low[0], high[0], 11)[1:-1]:
            for y in np.linspace(low[1], high[1], 6)[1:-1]:
                probe = np.asarray(current, dtype=float)
                probe[axes[0]], probe[axes[1]] = x, y
                delta = control_delta(view["target_xyz_m"], probe, view["translation_limit_m"])
                _arrow(draw, project(probe), project(probe+delta), FIELD_COLOUR)
        entry, target = project(view["entry_xyz_m"]), project(view["target_xyz_m"])
        _dashed(draw, entry, target, fill=YELLOW)
        draw.rectangle((entry[0]-3,entry[1]-3,entry[0]+3,entry[1]+3), outline=MUTED, width=1)
        for item in view["pending_absolute_targets"]:
            p=project(item["xyz_m"])
            _dashed(draw, (p[0]-5,p[1]-5),(p[0]+5,p[1]-5),fill=YELLOW,dash=2)
            _dashed(draw, (p[0]+5,p[1]-5),(p[0]+5,p[1]+5),fill=YELLOW,dash=2)
            _dashed(draw, (p[0]+5,p[1]+5),(p[0]-5,p[1]+5),fill=YELLOW,dash=2)
            _dashed(draw, (p[0]-5,p[1]+5),(p[0]-5,p[1]-5),fill=YELLOW,dash=2)
        draw.line((target[0]-6,target[1],target[0]+6,target[1]), fill=YELLOW, width=2)
        draw.line((target[0],target[1]-6,target[0],target[1]+6), fill=YELLOW, width=2)
    pixels = [project(p) for p in points]
    if len(pixels)>1:
        draw.line(pixels, fill=MAGENTA, width=3)
    px,py=pixels[-1]
    draw.ellipse((px-4,py-4,px+4,py+4), fill=TEXT)
    legend = "White: EEF | pink: path | +: target"
    if active and view["pending_absolute_targets"]:
        legend += " | box: pending"
    label_text(left+16, bottom-36, legend, 10)
    label_text(left+16, bottom-20,
               f"{names[axes[0]]} [{low[0]:+.2f}, {high[0]:+.2f}]  {names[axes[1]]} [{low[1]:+.2f}, {high[1]:+.2f}] m", 11)


def draw_planning_panels(draw, *, trace, state, view):
    """Shared XY, XZ, ZY strip for the live video and offline preview."""
    draw.rectangle(PLANNING_STRIP, fill=BG)
    for box, axes in PLANNING_PANELS:
        draw_planning_panel(draw, box, axes, trace=trace, state=state, view=view)


def draw_planning_caption(draw):
    """Keep the short field explanation visible inside every video frame."""
    draw.rectangle(PLANNING_CAPTION_BOX, fill=BG)
    for index, line in enumerate(PLANNING_CAPTION_LINES):
        draw.text((24, 921 + index * 20), line, font=_font(14), fill=TEXT)


class AgentSpeedUpV2UserCheckVideoRecorder(AgentSpeedUpV2VideoRecorder):
    """The reference v2 layout, with three read-only planning projections."""

    def __init__(self, path, *, planning_provider=None, **kwargs):
        self.planning_provider = planning_provider
        self.planning_debug = make_planning_view()
        super().__init__(path, **kwargs)

    def _set_planning(self, fragment=None, *, source="unavailable", execution_kind=None, event=None):
        try:
            self.planning_debug = make_planning_view(
                fragment, source=source, execution_kind=execution_kind, event=event)
        except Exception as error:
            self.planning_debug = make_planning_view(source="malformed_planning_data:" + type(error).__name__)

    def decision(self, observation, decision, *, decision_latency_s=None):
        # Every decision starts empty. A new response or mode switch cannot reuse
        # a previous fragment's target; only the current primitive can supply it.
        details = decision.diagnostics
        kind = details.get("execution_kind")
        fragment, source = None, "unavailable"
        if isinstance(kind, str) and kind.startswith("framework_fragment") and self.planning_provider is not None:
            try:
                fragment = self.planning_provider()
                source = "current_fragment_progress"
            except Exception as error:
                source = "provider_unavailable:" + type(error).__name__
        self._set_planning(fragment, source=source, execution_kind=kind)
        super().decision(observation, decision, decision_latency_s=decision_latency_s)

    def executed(self, observation, *, step, success, sim_status=None, feedback=None, history_stats=None):
        try:
            fragment = fragment_from_feedback(feedback)
        except Exception:
            fragment = None
        self._set_planning(
            fragment, source="recorded_transition_feedback", execution_kind=self.execution_kind,
            event="terminal" if success or (sim_status or {}).get("terminated") else "executed")
        super().executed(observation, step=step, success=success, sim_status=sim_status,
                         feedback=feedback, history_stats=history_stats)

    def _emit(self, event, repeats, **extra):
        if event in ("observation", "terminal"):
            self.planning_debug = make_planning_view(event=event)
        super()._emit(event, repeats, **{**extra, "planning_debug": deepcopy(self.planning_debug)})

    def _trace_panel(self, draw, box, axes):
        # The base renderer calls this twice. Render the shared three-plane strip
        # once after it finishes, so live output matches the offline preview.
        pass

    def render_frame(self):
        canvas = super().render_frame()
        if self.observation is not None:
            draw = ImageDraw.Draw(canvas)
            self._guarded("planning", PLANNING_STRIP, draw, lambda: draw_planning_panels(
                draw, trace=self.trace, state=self.observation.state, view=self.planning_debug))
            self._guarded("planning caption", PLANNING_CAPTION_BOX, draw,
                          lambda: draw_planning_caption(draw))
        return canvas
